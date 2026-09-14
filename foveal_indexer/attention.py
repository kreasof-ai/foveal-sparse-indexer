"""Foveal Sparse Attention.

Combines:
1. Parafoveal local sliding window (exact dense local context).
2. Remote foveal pages selected by the 16D FovealIndexer via top-p mass.
3. Dual-gradient training (additive stream + KL distillation against dense teacher).
4. O(1) decode step with KV cache for flat latency across sequence length.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .core import FovealIndexer, Route, masked_softmax


class FovealKVCache:
    """Paged KV cache for autoregressive decoding with flat latency."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        index_dim: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.index_dim = index_dim
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        self.max_pages = math.ceil(max_seq_len / page_size)

        # Full-rank KV cache
        self.k_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)

        # 16D Index page representations
        self.kp_cache = torch.zeros(batch_size, self.max_pages, index_dim, device=device, dtype=torch.float32)
        self.vp_cache = torch.zeros(batch_size, self.max_pages, index_dim, device=device, dtype=dtype)

        # Buffers for accumulating current partial page
        self.partial_kp = torch.zeros(batch_size, page_size, index_dim, device=device, dtype=dtype)
        self.partial_vp = torch.zeros(batch_size, page_size, index_dim, device=device, dtype=dtype)

        self.seq_len = 0
        self.completed_pages = 0

    def update(
        self,
        k_step: Tensor,
        v_step: Tensor,
        ki_step: Tensor,
        vi_step: Tensor,
    ) -> int:
        """Append one token's KV and 16D index representation to the cache."""
        pos = self.seq_len
        if pos >= self.max_seq_len:
            raise RuntimeError(f"KV cache capacity {self.max_seq_len} exceeded.")

        self.k_cache[:, :, pos : pos + 1] = k_step
        self.v_cache[:, :, pos : pos + 1] = v_step

        offset = pos % self.page_size
        page = pos // self.page_size

        self.partial_kp[:, offset] = ki_step.squeeze(1)
        self.partial_vp[:, offset] = vi_step.squeeze(1)

        # If we just finished a page, pool it into kp_cache / vp_cache
        if offset == self.page_size - 1:
            # Mean over block, then L2-normalize key
            km = self.partial_kp.mean(dim=1)
            vm = self.partial_vp.mean(dim=1)
            km_norm = F.normalize(km.float(), dim=-1)
            self.kp_cache[:, page] = km_norm
            self.vp_cache[:, page] = vm
            self.completed_pages += 1

        self.seq_len += 1
        return pos


class FovealSparseAttention(nn.Module):
    """Causal Sparse Attention using Foveal Indexing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        page_size: int = 64,
        local_window_blocks: int = 4,
        index_dim: int = 16,
        top_p: float = 0.95,
        min_remote_pages: int = 0,
        max_remote_pages: int = 8,
        remote_capacity: int = 8,
        use_additive_stream: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.page_size = page_size
        self.local_window_blocks = local_window_blocks
        self.local_window = local_window_blocks * page_size
        self.use_additive_stream = use_additive_stream

        self.groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Base Full-Rank GQA Projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # Embedded 16D Foveal Indexer
        self.indexer = FovealIndexer(
            hidden_size=hidden_size,
            index_dim=index_dim,
            block_size=page_size,
            local_window_blocks=local_window_blocks,
            top_p=top_p,
            min_remote_blocks=min_remote_pages,
            max_remote_blocks=max_remote_pages,
            remote_capacity=remote_capacity,
            causal=True,
            use_additive_stream=use_additive_stream,
        )

    def _build_sparse_mask(
        self,
        route: Route,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Construct portable causal sparse attention boolean mask."""
        q = torch.arange(seq_len, device=device)
        k = torch.arange(seq_len, device=device)

        # 1. Local sliding window
        allowed = (k[None, :] <= q[:, None]) & (k[None, :] > q[:, None] - self.local_window)
        allowed = allowed[None].expand(batch_size, -1, -1).clone()

        # 2. Selected remote pages
        query_blocks = route.remote_counts.shape[1]
        for b in range(batch_size):
            for qb in range(query_blocks):
                q0 = qb * self.page_size
                q1 = min((qb + 1) * self.page_size, seq_len)
                count = int(route.remote_counts[b, qb].item())
                for slot in range(count):
                    page = int(route.remote_indices[b, qb, slot].item())
                    k0 = page * self.page_size
                    k1 = min((page + 1) * self.page_size, seq_len)
                    allowed[b, q0:q1, k0:k1] = True

        return allowed

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Prefill forward pass.

        Args:
            x: (B, T, D) input representations
            compute_teacher_loss: If True, computes dense teacher attention and distillation KL
        Returns:
            output: (B, T, D)
            aux: dict with 'route', 'distill_loss', 'additive_stream'
        """
        batch, seq_len, _ = x.shape
        device = x.device

        if seq_len % self.page_size != 0:
            raise ValueError(f"Sequence length {seq_len} must be multiple of page_size {self.page_size}")

        num_pages = seq_len // self.page_size

        # 1. Compute 16D Indexer representation
        # detach x for index projections to isolate auxiliary branch
        x_detach = x.detach()
        q_16d = self.indexer.compute_query_16d(x_detach)
        kp_16d, vp_16d = self.indexer.compute_kv_blocks_16d(x_detach)

        # Causal routing representative: first query in each block
        q_block_16d = q_16d[:, :: self.page_size]
        block_scores = self.indexer.compute_block_scores(q_block_16d, kp_16d)

        # 2. Select remote pages
        route = self.indexer.route(block_scores)

        # 3. Base full-rank projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand GQA KV heads if needed
        if self.groups > 1:
            k_exp = k.repeat_interleave(self.groups, dim=1)
            v_exp = v.repeat_interleave(self.groups, dim=1)
        else:
            k_exp = k
            v_exp = v

        # 4. Sparse attention execution
        sparse_mask = self._build_sparse_mask(route, batch, seq_len, device)
        attn_out = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=sparse_mask[:, None, :, :],
            scale=self.scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # 5. Additive 16D stream (LM-output)
        additive_out = None
        if self.use_additive_stream:
            additive_out = self.indexer.additive_stream(block_scores, vp_16d)
            out = out + additive_out

        aux: Dict[str, Any] = {
            "route": route,
            "additive_out": additive_out,
            "distill_loss": torch.tensor(0.0, device=device),
        }

        # 6. Distillation loss (KL divergence) against dense teacher if requested
        if compute_teacher_loss:
            with torch.no_grad():
                # Full causal dense attention teacher
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
                raw_dense_scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
                raw_dense_scores = raw_dense_scores.masked_fill(~causal_mask[None, None], -float("inf"))
                dense_probs = torch.softmax(raw_dense_scores, dim=-1)  # (B, H, T, T)

                # Teacher page mass for query block anchors (first token of each query block)
                anchor_indices = torch.arange(0, seq_len, self.page_size, device=device)
                anchor_probs = dense_probs[:, :, anchor_indices, :]  # (B, H, Q_blocks, T)

                # Sum attention probability within each key page
                # Reshape T -> (num_pages, page_size)
                page_mass = anchor_probs.view(batch, self.num_heads, num_pages, num_pages, self.page_size).sum(dim=-1)
                # Average over heads
                teacher_mass = page_mass.mean(dim=1)  # (B, Q_blocks, num_pages)

            aux["distill_loss"] = self.indexer.distillation_loss(block_scores, teacher_mass)

        return out, aux

    def init_kv_cache(self, batch_size: int, max_seq_len: int, device: torch.device) -> FovealKVCache:
        """Create a new KV cache instance for decoding."""
        return FovealKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            index_dim=self.indexer.index_dim,
            page_size=self.page_size,
            device=device,
            dtype=self.q_proj.weight.dtype,
        )

    def decode_step(
        self,
        x_step: Tensor,
        kv_cache: FovealKVCache,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Fast autoregressive decode step for a single token.

        Demonstrates flat O(1) decoding latency across sequence length.
        Only queries the local window + selected remote pages!
        """
        batch, seq_one, _ = x_step.shape
        assert seq_one == 1, "decode_step processes one token at a time"
        device = x_step.device
        pos = kv_cache.seq_len

        # 1. Compute full-rank Q, K, V for this token
        q = self.q_proj(x_step).view(batch, self.num_heads, 1, self.head_dim)
        k = self.k_proj(x_step).view(batch, self.num_kv_heads, 1, self.head_dim)
        v = self.v_proj(x_step).view(batch, self.num_kv_heads, 1, self.head_dim)

        # 2. Compute 16D index representation for this token
        x_detach = x_step.detach()
        qi = self.indexer.compute_query_16d(x_detach)
        ki = F.rms_norm(self.indexer.k_proj(x_detach), (self.indexer.index_dim,))
        vi = self.indexer.v_proj(x_detach)

        # 3. Update KV cache
        kv_cache.update(k, v, ki, vi)

        # 4. Route remote pages from completed pages outside local window
        completed_pages = kv_cache.completed_pages
        local_start = max(0, pos + 1 - self.local_window)
        first_local_page = local_start // self.page_size

        remote_page_indices = []
        if completed_pages > first_local_page and first_local_page > 0:
            # Candidate remote pages are [0, first_local_page)
            # Query vector: qi normalized
            q_norm = F.normalize(qi.float(), dim=-1)
            kp_remote = kv_cache.kp_cache[:, :first_local_page]  # (B, remote_pages, index_dim)
            remote_scores = torch.einsum("btr,bpr->btp", q_norm, kp_remote).squeeze(1) / math.sqrt(self.indexer.index_dim)

            probs = F.softmax(remote_scores, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)

            # Top-p selection
            k_max = min(self.indexer.max_remote_blocks, first_local_page)
            for b in range(batch):
                needed = (cumulative[b] < self.indexer.top_p).sum().item() + 1
                needed = max(self.indexer.min_remote_blocks, min(needed, k_max))
                remote_page_indices.append(sorted_idx[b, :needed].tolist())
        else:
            remote_page_indices = [[] for _ in range(batch)]

        # 5. Gather active support: Local tokens + Selected remote page tokens
        # Local tokens: [local_start, pos]
        local_tokens_k = kv_cache.k_cache[:, :, local_start : pos + 1]
        local_tokens_v = kv_cache.v_cache[:, :, local_start : pos + 1]

        # Gather remote tokens for batch (assume batch=1 for simplicity or pad to max)
        gathered_k_list = [local_tokens_k]
        gathered_v_list = [local_tokens_v]

        for p_idx in remote_page_indices[0]:
            p_start = p_idx * self.page_size
            p_end = p_start + self.page_size
            gathered_k_list.append(kv_cache.k_cache[:, :, p_start:p_end])
            gathered_v_list.append(kv_cache.v_cache[:, :, p_start:p_end])

        active_k = torch.cat(gathered_k_list, dim=2)
        active_v = torch.cat(gathered_v_list, dim=2)

        if self.groups > 1:
            active_k = active_k.repeat_interleave(self.groups, dim=1)
            active_v = active_v.repeat_interleave(self.groups, dim=1)

        # 6. Attention over active support only! (Size <= W + K_max * page_size)
        attn_out = F.scaled_dot_product_attention(
            q, active_k, active_v,
            scale=self.scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, 1, self.num_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # Additive stream contribution if enabled
        if self.use_additive_stream and completed_pages > 0:
            kp_all = kv_cache.kp_cache[:, :completed_pages]
            vp_all = kv_cache.vp_cache[:, :completed_pages]
            q_norm = F.normalize(qi.float(), dim=-1)
            scores_all = torch.einsum("btr,bpr->btp", q_norm, kp_all) / math.sqrt(self.indexer.index_dim)
            probs_all = F.softmax(scores_all, dim=-1).to(vp_all.dtype)
            context = torch.einsum("btp,bpr->btr", probs_all, vp_all)
            out = out + self.indexer.out_proj(context)

        return out, {"selected_remote_pages": remote_page_indices}
