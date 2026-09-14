"""Blockwise Sparse MatMul with 16D Foveal Indexer.

Implements blockwise sparse linear projections and matrix multiplications (e.g. 64x64 or 128x128):
1. Weight matrix is partitioned into blocks (e.g. column blocks of size K x 64 or 2D tiles 64x64).
2. Each block has a 16D key and value representation in the indexer.
3. Base blocks provide a permanent dense foundation.
4. Input tokens/chunks dynamically select remote blocks via top-p attention mass.
5. Dual-gradient training:
   - 16D additive stream provides smooth end-to-end task loss gradients.
   - Distillation KL loss matches indexer block probabilities to actual block activation norms.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .core import FovealIndexer, Route


class BlockSparseLinear(nn.Module):
    """Linear projection (in_features -> out_features) with blockwise dynamic sparsity.

    Partitions out_features into blocks of size `block_size` (e.g. 64 or 128).
    For each input token or token chunk, FovealIndexer predicts which blocks
    are active. Only active blocks are computed in the GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 64,
        base_blocks: int = 2,
        index_dim: int = 16,
        top_p: float = 0.95,
        min_remote_blocks: int = 0,
        max_remote_blocks: int = 8,
        remote_capacity: int = 8,
        use_additive_stream: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if out_features % block_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by block_size ({block_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_blocks = out_features // block_size
        self.base_blocks = min(base_blocks, self.num_blocks)
        self.use_additive_stream = use_additive_stream

        # Full weight tensor: (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Embedded Foveal Indexer
        self.indexer = FovealIndexer(
            hidden_size=in_features,
            index_dim=index_dim,
            block_size=block_size,
            local_window_blocks=self.base_blocks,
            top_p=top_p,
            min_remote_blocks=min_remote_blocks,
            max_remote_blocks=max_remote_blocks,
            remote_capacity=remote_capacity,
            causal=False,
            use_additive_stream=use_additive_stream,
        )

        # Indexer block keys & values projection from weight blocks
        # Weight block is (block_size, in_features)
        self.weight_to_k = nn.Linear(in_features, index_dim, bias=False)
        self.weight_to_v = nn.Linear(in_features, index_dim, bias=False)
        self.indexer.out_proj = nn.Linear(index_dim, out_features, bias=False)

        nn.init.normal_(self.weight_to_k.weight, std=in_features ** -0.5)
        nn.init.normal_(self.weight_to_v.weight, std=in_features ** -0.5)
        nn.init.normal_(self.indexer.out_proj.weight, std=1e-3)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_block_keys_and_values(self) -> Tuple[Tensor, Tensor]:
        """Compute 16D key and value vectors for all weight blocks.

        Returns:
            k_blocks: (1, num_blocks, index_dim)
            v_blocks: (1, num_blocks, index_dim)
        """
        # Weight shape: (num_blocks, block_size, in_features)
        w_blocked = self.weight.view(self.num_blocks, self.block_size, self.in_features)

        # Pool over block_size
        k_elem = F.rms_norm(self.weight_to_k(w_blocked), (self.indexer.index_dim,))
        v_elem = self.weight_to_v(w_blocked)

        k_blocks = k_elem.mean(dim=1, keepdim=True)  # (num_blocks, 1, index_dim)
        v_blocks = v_elem.mean(dim=1, keepdim=True)  # (num_blocks, 1, index_dim)

        k_blocks = F.normalize(k_blocks.squeeze(1).float(), dim=-1).unsqueeze(0)  # (1, num_blocks, index_dim)
        v_blocks = v_blocks.squeeze(1).unsqueeze(0)  # (1, num_blocks, index_dim)

        return k_blocks, v_blocks

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass computing only active weight blocks.

        Args:
            x: (B, T, in_features) or (B, in_features)
            compute_teacher_loss: If True, computes dense output and distillation KL loss.
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, in_features)

        batch, seq_len, in_feat = x.shape
        device = x.device

        # 1. 16D queries from input
        x_detach = x.detach()
        q_16d = self.indexer.compute_query_16d(x_detach)  # (B, T, index_dim)

        # 2. 16D keys and values for weight blocks
        kp_16d, vp_16d = self.get_block_keys_and_values()  # (1, num_blocks, index_dim)
        kp_16d = kp_16d.expand(batch, -1, -1)
        vp_16d = vp_16d.expand(batch, -1, -1)

        # 3. Block scores
        block_scores = self.indexer.compute_block_scores(q_16d, kp_16d)  # (B, T, num_blocks)

        # 4. Route blocks
        route = self.indexer.route(block_scores)

        # 5. Sparse MatMul Execution
        # Base blocks: [0, base_blocks)
        active_block_mask = torch.zeros(self.num_blocks, dtype=torch.bool, device=device)
        active_block_mask[: self.base_blocks] = True

        # Mark remote blocks selected across the batch (respecting remote_counts)
        capacity = route.remote_indices.shape[-1]
        slot_idx = torch.arange(capacity, device=device).view(1, 1, capacity)
        valid_slots = slot_idx < route.remote_counts[..., None]
        selected_remote = route.remote_indices[valid_slots]
        selected_remote = selected_remote[selected_remote >= self.base_blocks]
        if selected_remote.numel() > 0:
            active_block_mask[selected_remote.long()] = True

        active_block_indices = torch.where(active_block_mask)[0]
        num_active = len(active_block_indices)

        # Gather active slices of weight
        # Construct index tensor of active output channels
        active_channels = []
        for b_idx in active_block_indices.tolist():
            start = b_idx * self.block_size
            active_channels.extend(range(start, start + self.block_size))
        active_chan_idx = torch.tensor(active_channels, device=device, dtype=torch.long)

        active_w = self.weight[active_chan_idx]  # (N_active, in_features)

        # Compute active GEMM: (B, T, in_features) @ (in_features, N_active) -> (B, T, N_active)
        active_out = F.linear(x, active_w, None)

        # Scatter active output into full output tensor
        out = torch.zeros(batch, seq_len, self.out_features, device=device, dtype=x.dtype)
        out[:, :, active_chan_idx] = active_out

        if self.bias is not None:
            out[:, :, active_chan_idx] = out[:, :, active_chan_idx] + self.bias[active_chan_idx]

        # 6. Additive 16D stream
        additive_out = None
        if self.use_additive_stream:
            probs = F.softmax(block_scores, dim=-1).to(vp_16d.dtype)
            context = torch.einsum("btp,bpr->btr", probs, vp_16d)
            additive_out = self.indexer.out_proj(context)
            out = out + additive_out

        if orig_ndim == 2:
            out = out.squeeze(1)

        aux: Dict[str, Any] = {
            "route": route,
            "additive_out": additive_out,
            "active_blocks_union": num_active,
            "mean_active_blocks": self.base_blocks + route.mean_remote_blocks.item(),
            "total_blocks": self.num_blocks,
            "sparsity": 1.0 - ((self.base_blocks + route.mean_remote_blocks.item()) / self.num_blocks),
            "distill_loss": torch.tensor(0.0, device=device),
        }

        # 7. Teacher Distillation Loss
        if compute_teacher_loss:
            with torch.no_grad():
                # Full dense forward
                dense_out = F.linear(x, self.weight, self.bias)
                # Compute L2 activation norm per block: (B, T, num_blocks)
                dense_blocks = dense_out.view(batch, seq_len, self.num_blocks, self.block_size)
                block_energy = torch.norm(dense_blocks, p=2, dim=-1)
                # Softmax over block energies to form target distribution
                teacher_mass = F.softmax(block_energy, dim=-1)

            aux["distill_loss"] = self.indexer.distillation_loss(block_scores, teacher_mass)

        return out, aux


class BlockwiseMatMul2D(nn.Module):
    """2D Block-sparse matrix multiplication (e.g. 64x64 or 128x128 tiles).

    Computes Y = X @ W where W is tiled into (num_k_blocks, num_n_blocks) tiles.
    16D Foveal Indexer scores tile relevance and computes only active tiles.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_k: int = 64,
        block_n: int = 64,
        base_tiles_ratio: float = 0.25,
        index_dim: int = 16,
        top_p: float = 0.95,
    ):
        super().__init__()
        if in_features % block_k != 0 or out_features % block_n != 0:
            raise ValueError(
                f"Shapes ({in_features}, {out_features}) must be divisible by "
                f"tile sizes ({block_k}, {block_n})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_k = block_k
        self.block_n = block_n

        self.num_k_blocks = in_features // block_k
        self.num_n_blocks = out_features // block_n
        self.total_tiles = self.num_k_blocks * self.num_n_blocks
        self.base_tiles = max(1, int(self.total_tiles * base_tiles_ratio))

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 16D tile representations
        self.tile_q = nn.Linear(block_k, index_dim, bias=False)
        self.tile_k = nn.Linear(block_k * block_n, index_dim, bias=False)
        self.tile_v = nn.Linear(block_k * block_n, index_dim, bias=False)
        self.out_proj = nn.Linear(index_dim, out_features, bias=False)

        nn.init.normal_(self.tile_q.weight, std=block_k ** -0.5)
        nn.init.normal_(self.tile_k.weight, std=(block_k * block_n) ** -0.5)
        nn.init.normal_(self.tile_v.weight, std=(block_k * block_n) ** -0.5)
        nn.init.normal_(self.out_proj.weight, std=1e-3)

        self.top_p = top_p
        self.index_dim = index_dim

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Perform 2D block-sparse multiplication."""
        batch, seq_len, _ = x.shape
        device = x.device

        # Tile weights into (num_k_blocks, num_n_blocks, block_n, block_k)
        # Reshape weight: (num_n_blocks, block_n, num_k_blocks, block_k)
        w_tiles = (
            self.weight.view(self.num_n_blocks, self.block_n, self.num_k_blocks, self.block_k)
            .permute(2, 0, 1, 3)
            .contiguous()
            .view(self.total_tiles, self.block_n * self.block_k)
        )

        # 16D keys and values for tiles
        tile_k = F.normalize(F.rms_norm(self.tile_k(w_tiles), (self.index_dim,)), dim=-1)  # (total_tiles, 16)
        tile_v = self.tile_v(w_tiles)  # (total_tiles, 16)

        # Average input along K per block: (B, T, num_k_blocks, block_k)
        x_k = x.view(batch, seq_len, self.num_k_blocks, self.block_k).mean(dim=2)  # (B, T, block_k)
        q_16d = F.normalize(F.rms_norm(self.tile_q(x_k), (self.index_dim,)), dim=-1)  # (B, T, 16)

        # Dot-product scores between input and all 2D tiles
        scores = torch.einsum("btr,pr->btp", q_16d, tile_k) / math.sqrt(self.index_dim)  # (B, T, total_tiles)
        probs = F.softmax(scores, dim=-1)

        # Select top-p tiles
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        needed = (cumulative < self.top_p).sum(dim=-1) + 1
        max_needed = needed.max().item()

        # Always include base tiles [0, base_tiles)
        active_tile_mask = torch.zeros(self.total_tiles, dtype=torch.bool, device=device)
        active_tile_mask[: self.base_tiles] = True
        for b in range(batch):
            for t in range(seq_len):
                count = min(int(needed[b, t].item()), self.total_tiles)
                active_tile_mask[sorted_indices[b, t, :count]] = True

        active_indices = torch.where(active_tile_mask)[0]

        # Compute block multiplication for active tiles
        out = torch.zeros(batch, seq_len, self.out_features, device=device, dtype=x.dtype)
        for tile_idx in active_indices.tolist():
            k_b = tile_idx // self.num_n_blocks
            n_b = tile_idx % self.num_n_blocks

            x_slice = x[:, :, k_b * self.block_k : (k_b + 1) * self.block_k]
            w_slice = self.weight[
                n_b * self.block_n : (n_b + 1) * self.block_n,
                k_b * self.block_k : (k_b + 1) * self.block_k,
            ]
            out[:, :, n_b * self.block_n : (n_b + 1) * self.block_n] += F.linear(x_slice, w_slice)

        # Additive stream
        context = torch.einsum("btp,pr->btr", probs.to(tile_v.dtype), tile_v)
        out = out + self.out_proj(context)

        aux: Dict[str, Any] = {
            "active_tiles": len(active_indices),
            "total_tiles": self.total_tiles,
            "tile_sparsity": 1.0 - (len(active_indices) / self.total_tiles),
            "distill_loss": torch.tensor(0.0, device=device),
        }

        if compute_teacher_loss:
            with torch.no_grad():
                dense_out = F.linear(x, self.weight)
                tile_energy = torch.zeros(batch, seq_len, self.total_tiles, device=device)
                for idx in range(self.total_tiles):
                    k_b = idx // self.num_n_blocks
                    n_b = idx % self.num_n_blocks
                    slice_out = dense_out[:, :, n_b * self.block_n : (n_b + 1) * self.block_n]
                    tile_energy[:, :, idx] = torch.norm(slice_out, p=2, dim=-1)
                teacher_mass = F.softmax(tile_energy, dim=-1)

            log_student = F.log_softmax(scores, dim=-1)
            aux["distill_loss"] = -(teacher_mass * log_student).sum(dim=-1).mean()

        return out, aux
