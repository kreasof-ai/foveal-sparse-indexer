"""Foveal Tokenformer: Token-Parameter Attention with Foveal Sparse Indexing.

Implements attention-reparameterized weights as proposed in Tokenformer (arXiv:2410.23168),
augmented with Foveal Sparse Indexing:
1. Model parameters are represented as key-value parameter tokens (K_P, V_P).
2. Parameter tokens are grouped into blocks (e.g. 64 parameter tokens per block).
3. Base parameter blocks form an always-active dense bottleneck (parafoveal band).
4. Remote parameter blocks are conditionally retrieved via 16D top-p mass routing.
5. Dual-gradient training:
   - Differentiable 16D additive stream (soft read projected to residual).
   - Distillation KL loss against full dense teacher parameter attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .core import FovealIndexer, Route, masked_softmax


class FovealTokenformerLinear(nn.Module):
    """Token-Parameter Attention (Pattention) Layer with Foveal Indexing.

    Replaces a linear projection (in_features -> out_features) with cross-attention
    between input tokens (queries) and learnable parameter tokens (K_P, V_P).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_param_tokens: int = 512,
        param_block_size: int = 64,
        base_blocks: int = 2,
        index_dim: int = 16,
        top_p: float = 0.95,
        min_remote_blocks: int = 0,
        max_remote_blocks: int = 4,
        remote_capacity: int = 4,
        activation: Literal["softmax", "gelu_l2"] = "softmax",
        use_additive_stream: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_param_tokens = num_param_tokens
        self.param_block_size = param_block_size
        self.base_blocks = base_blocks
        self.activation = activation
        self.use_additive_stream = use_additive_stream

        if num_param_tokens % param_block_size != 0:
            raise ValueError(
                f"num_param_tokens ({num_param_tokens}) must be divisible by "
                f"param_block_size ({param_block_size})"
            )
        self.num_blocks = num_param_tokens // param_block_size
        assert base_blocks <= self.num_blocks, "base_blocks cannot exceed total parameter blocks"

        # Learnable parameter tokens
        self.k_param = nn.Parameter(torch.empty(num_param_tokens, in_features))
        self.v_param = nn.Parameter(torch.empty(num_param_tokens, out_features))

        # Weight initialization
        nn.init.normal_(self.k_param, std=in_features ** -0.5)
        nn.init.normal_(self.v_param, std=out_features ** -0.5)

        # Foveal Indexer for parameter blocks
        self.indexer = FovealIndexer(
            hidden_size=in_features,
            index_dim=index_dim,
            block_size=param_block_size,
            local_window_blocks=base_blocks,
            top_p=top_p,
            min_remote_blocks=min_remote_blocks,
            max_remote_blocks=max_remote_blocks,
            remote_capacity=remote_capacity,
            causal=False,  # parameter blocks are non-causal
            use_additive_stream=use_additive_stream,
        )

        # Map 16D additive context to out_features
        self.indexer.out_proj = nn.Linear(index_dim, out_features, bias=False)
        nn.init.normal_(self.indexer.out_proj.weight, std=1e-3)

        self._cached_kp: Optional[Tensor] = None
        self._cached_vp: Optional[Tensor] = None

    def refresh_param_cache(self) -> None:
        """Update cached 16D representations for parameter blocks."""
        with torch.no_grad():
            self._cached_kp, self._cached_vp = self.indexer.compute_kv_blocks_16d(self.k_param.detach())

    def _activate_scores(self, scores: Tensor) -> Tensor:
        """Apply attention activation function."""
        if self.activation == "softmax":
            return F.softmax(scores / math.sqrt(self.in_features), dim=-1)
        elif self.activation == "gelu_l2":
            # Tokenformer modified activation: L2 normalization along token dim + GeLU
            norm_scores = F.normalize(scores, p=2, dim=-1)
            return F.gelu(norm_scores)
        raise ValueError(f"Unknown activation: {self.activation}")

    def compute_dense_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute full dense Tokenformer attention and block mass for distillation.

        Args:
            x: (B, T, D_in)
        Returns:
            dense_output: (B, T, D_out)
            teacher_block_mass: (B, T, num_blocks)
        """
        # (B, T, num_param_tokens)
        raw_scores = torch.matmul(x, self.k_param.t())
        attn_weights = self._activate_scores(raw_scores)
        dense_out = torch.matmul(attn_weights, self.v_param)

        # Pool attention weights into parameter blocks (ensuring non-negative mass)
        # (B, T, num_blocks, block_size) -> sum over block_size
        batch, seq_len, _ = x.shape
        block_mass = attn_weights.clamp_min(0.0).view(
            batch, seq_len, self.num_blocks, self.param_block_size
        ).sum(dim=-1)

        return dense_out, block_mass

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass with dynamic parameter block selection.

        Args:
            x: (B, T, D_in)
            compute_teacher_loss: If True, computes dense teacher and distillation KL loss.
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # 1. 16D Indexer representation of input query tokens
        x_detach = x.detach()
        q_16d = self.indexer.compute_query_16d(x_detach)  # (B, T, 16)

        # 2. 16D Indexer representation of parameter blocks
        if not self.training and self._cached_kp is not None:
            kp_16d, vp_16d = self._cached_kp, self._cached_vp
        else:
            kp_16d, vp_16d = self.indexer.compute_kv_blocks_16d(self.k_param.detach())
            if not self.training:
                self._cached_kp, self._cached_vp = kp_16d, vp_16d
        kp_16d = kp_16d.expand(batch, -1, -1)
        vp_16d = vp_16d.expand(batch, -1, -1)

        # 3. Score parameter blocks: (B, T, num_blocks)
        block_scores = self.indexer.compute_block_scores(q_16d, kp_16d)

        # 4. Route parameter blocks per query token (or token chunk)
        route = self.indexer.route(block_scores)

        # 5. Sparse Token-Parameter Attention
        # Base parameter tokens: always active [0, base_blocks * block_size)
        base_param_count = self.base_blocks * self.param_block_size
        base_k = self.k_param[:base_param_count]  # (N_base, D_in)
        base_v = self.v_param[:base_param_count]  # (N_base, D_out)

        # For efficient batched execution on CPU, we evaluate tokens with their gathered parameter blocks
        # Or gather parameters for each token.
        # Vectorized gather:
        # Determine union of selected remote blocks across batch or per-sample
        # Since base blocks are identical for all tokens, compute base attention first:
        base_raw = torch.matmul(x, base_k.t())  # (B, T, N_base)

        # Active remote parameter blocks
        # Only consider slots up to route.remote_counts
        capacity = route.remote_indices.shape[-1]
        slot_idx = torch.arange(capacity, device=device).view(1, 1, capacity)
        valid_slots = slot_idx < route.remote_counts[..., None]
        selected_remote = route.remote_indices[valid_slots]
        valid_remote = torch.unique(selected_remote[selected_remote >= self.base_blocks])

        if valid_remote.numel() > 0:
            offsets = torch.arange(self.param_block_size, device=device)
            remote_token_idx = (valid_remote.view(-1, 1) * self.param_block_size + offsets).flatten()
            remote_k = self.k_param[remote_token_idx]  # (N_remote_active, D_in)
            remote_v = self.v_param[remote_token_idx]  # (N_remote_active, D_out)

            # Combined active parameters for this forward call
            active_k = torch.cat([base_k, remote_k], dim=0)  # (N_active, D_in)
            active_v = torch.cat([base_v, remote_v], dim=0)  # (N_active, D_out)

            active_raw = torch.matmul(x, active_k.t())  # (B, T, N_active)
            active_weights = self._activate_scores(active_raw)
            out = torch.matmul(active_weights, active_v)
        else:
            active_weights = self._activate_scores(base_raw)
            out = torch.matmul(active_weights, base_v)

        # 6. Additive 16D stream
        additive_out = None
        if self.use_additive_stream:
            # Soft read over all parameter blocks
            probs = F.softmax(block_scores, dim=-1).to(vp_16d.dtype)
            context = torch.einsum("btp,bpr->btr", probs, vp_16d)
            additive_out = self.indexer.out_proj(context)
            out = out + additive_out

        aux: Dict[str, Any] = {
            "route": route,
            "additive_out": additive_out,
            "active_param_tokens_union": base_param_count + (valid_remote.numel() * self.param_block_size),
            "mean_active_param_tokens": base_param_count + (route.mean_remote_blocks.item() * self.param_block_size),
            "total_param_tokens": self.num_param_tokens,
            "distill_loss": torch.tensor(0.0, device=device),
        }

        # 7. Distillation loss
        if compute_teacher_loss:
            with torch.no_grad():
                _, teacher_mass = self.compute_dense_teacher(x)
            aux["distill_loss"] = self.indexer.distillation_loss(block_scores, teacher_mass)

        return out, aux


class FovealTokenformerFFN(nn.Module):
    """Feed-Forward Network (FFN) implemented using Foveal Tokenformer Layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_param_tokens: int = 512,
        param_block_size: int = 64,
        base_blocks: int = 2,
        index_dim: int = 16,
        top_p: float = 0.95,
        max_remote_blocks: int = 4,
    ):
        super().__init__()
        # In Tokenformer, FFN is formulated with token-parameter attention
        self.fc1 = FovealTokenformerLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_param_tokens=num_param_tokens,
            param_block_size=param_block_size,
            base_blocks=base_blocks,
            index_dim=index_dim,
            top_p=top_p,
            max_remote_blocks=max_remote_blocks,
            activation="gelu_l2",
        )
        self.fc2 = FovealTokenformerLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_param_tokens=num_param_tokens,
            param_block_size=param_block_size,
            base_blocks=base_blocks,
            index_dim=index_dim,
            top_p=top_p,
            max_remote_blocks=max_remote_blocks,
            activation="softmax",
        )

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        h, aux1 = self.fc1(x, compute_teacher_loss=compute_teacher_loss)
        out, aux2 = self.fc2(h, compute_teacher_loss=compute_teacher_loss)

        total_distill = aux1["distill_loss"] + aux2["distill_loss"]
        return out, {
            "fc1_aux": aux1,
            "fc2_aux": aux2,
            "distill_loss": total_distill,
        }
