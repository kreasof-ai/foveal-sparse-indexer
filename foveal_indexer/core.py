"""Core 16D Foveal Indexer implementation.

Key ideas from Foveal Attention / Foveal-MoE:
1. Multi-element blocks (pages) of tokens or parameters.
2. 16-dimensional MQA representations for keys and values per block.
3. Adaptive top-p attention mass routing with [K_min, K_max] guardrails.
4. Dual-gradient recipe:
   - Differentiable 16D additive stream (soft read projected to residual).
   - Auxiliary teacher distillation (block-level KL loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class Route:
    """Stores routing decisions for foveal block selection."""
    remote_indices: Tensor  # (B, Q_blocks, remote_capacity) - selected remote block indices
    remote_counts: Tensor   # (B, Q_blocks) - valid count of remote blocks per query block
    base_indices: Optional[Tensor]  # (B, Q_blocks, max_base_blocks) - base/local block indices
    base_counts: Optional[Tensor]   # (B, Q_blocks) - valid count of base blocks
    block_scores: Tensor    # (B, Q_blocks, total_blocks) - differentiable 16D score logits
    cap_rate: Tensor        # fraction of query blocks hitting max_remote limit
    mean_remote_blocks: Tensor # average number of remote blocks selected
    base_only_rate: Tensor  # fraction of query blocks requiring no remote blocks


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Softmax that strictly zeroes out rows/elements without valid entries."""
    safe = scores.float().masked_fill(~mask, -float("inf"))
    row_max = safe.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    numer = torch.exp(safe - row_max) * mask
    denom = numer.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return (numer / denom).to(scores.dtype)


def select_blocks(
    block_scores: Tensor,
    *,
    causal: bool = True,
    block_size: int = 64,
    local_window_blocks: int = 0,
    top_p: float | Tensor = 0.95,
    min_remote_blocks: int | Tensor = 0,
    max_remote_blocks: int | Tensor = 16,
    remote_capacity: int = 16,
) -> Route:
    """Adaptive top-p mass routing over block scores.

    Args:
        block_scores: (B, Q_blocks, total_blocks) - pre-softmax dot-product scores.
        causal: If True, query block q can only attend to blocks p < q (or p <= q).
        block_size: Elements per block.
        local_window_blocks: Number of preceding blocks unconditionally kept in base/local.
        top_p: Desired cumulative attention mass threshold in (0, 1].
        min_remote_blocks: Minimum remote blocks to select (K_min).
        max_remote_blocks: Maximum remote blocks to select (K_max guardrail).
        remote_capacity: Output tensor capacity for remote indices.
    """
    batch, query_blocks, total_blocks = block_scores.shape
    device = block_scores.device

    top_p_t = torch.as_tensor(top_p, device=device, dtype=torch.float32)
    min_remote_t = torch.as_tensor(min_remote_blocks, device=device, dtype=torch.int64)
    max_remote_t = torch.as_tensor(max_remote_blocks, device=device, dtype=torch.int64)

    qb = torch.arange(query_blocks, device=device)
    pb = torch.arange(total_blocks, device=device)

    if causal:
        # Query block q completes blocks [0, q-1].
        # In causal routing, block q itself is current/local, candidate complete blocks are pb < qb.
        complete = pb.view(1, total_blocks) < qb.view(query_blocks, 1)
        if local_window_blocks > 0:
            first_local = (qb - local_window_blocks).clamp_min(0)
            local_complete = complete & (pb.view(1, total_blocks) >= first_local[:, None])
            remote = complete & (pb.view(1, total_blocks) < first_local[:, None])
        else:
            local_complete = torch.zeros_like(complete)
            remote = complete
    else:
        # Non-causal (e.g. parameter tokens or bidirectional matmul blocks)
        complete = torch.ones((query_blocks, total_blocks), dtype=torch.bool, device=device)
        if local_window_blocks > 0:
            # Base blocks are fixed initial blocks [0, local_window_blocks)
            local_complete = (pb.view(1, total_blocks) < local_window_blocks).expand(query_blocks, -1)
            remote = (pb.view(1, total_blocks) >= local_window_blocks).expand(query_blocks, -1)
        else:
            local_complete = torch.zeros_like(complete)
            remote = complete

    complete_expanded = complete[None].expand(batch, -1, -1)
    probs = masked_softmax(block_scores, complete_expanded)

    local_mass = (probs * local_complete[None]).sum(dim=-1)
    remote_probs = probs.masked_fill(~remote[None], -1.0)

    sorted_prob, sorted_idx = remote_probs.sort(dim=-1, descending=True)
    sorted_prob = sorted_prob.clamp_min(0.0)
    cumulative = sorted_prob.cumsum(dim=-1)

    target = (top_p_t - local_mass).clamp_min(0.0)
    available = remote.sum(dim=-1).view(1, query_blocks).expand(batch, -1)

    # Smallest n for which cumulative[n-1] >= target.
    needed = (cumulative < target[..., None]).sum(dim=-1) + (target > 0).long()
    needed = torch.where(available > 0, needed, torch.zeros_like(needed))
    needed = torch.where(available > 0, torch.maximum(needed, min_remote_t), needed)
    counts = torch.minimum(needed, available)
    counts = torch.minimum(counts, max_remote_t)

    indices = sorted_idx[..., :remote_capacity].to(torch.int32)
    if indices.shape[-1] < remote_capacity:
        indices = F.pad(indices, (0, remote_capacity - indices.shape[-1]))

    # Setup base/local indices if applicable
    if local_window_blocks > 0:
        if causal:
            max_local = local_window_blocks + 1  # includes current block
            offsets = torch.arange(max_local, device=device)
            local_start = (qb - local_window_blocks).clamp_min(0)
            local_indices = local_start[:, None] + offsets[None]
            local_counts = qb - local_start + 1
            local_valid = offsets[None] < local_counts[:, None]
            local_indices = torch.where(local_valid, local_indices, torch.zeros_like(local_indices))
            base_indices = local_indices[None].expand(batch, -1, -1).to(torch.int32)
            base_counts = local_counts[None].expand(batch, -1).to(torch.int32)
        else:
            base_indices = torch.arange(local_window_blocks, device=device, dtype=torch.int32)
            base_indices = base_indices[None, None, :].expand(batch, query_blocks, -1)
            base_counts = torch.full((batch, query_blocks), local_window_blocks, device=device, dtype=torch.int32)
    else:
        base_indices = None
        base_counts = None

    cap_rate = torch.where(
        max_remote_t > 0,
        (counts >= max_remote_t).float().mean(),
        counts.new_zeros((), dtype=torch.float32),
    )

    return Route(
        remote_indices=indices,
        remote_counts=counts.to(torch.int32),
        base_indices=base_indices,
        base_counts=base_counts,
        block_scores=block_scores,
        cap_rate=cap_rate,
        mean_remote_blocks=counts.float().mean(),
        base_only_rate=(counts == 0).float().mean(),
    )


class FovealIndexer(nn.Module):
    """Modular 16-dimensional Foveal Indexer.

    Computes:
    - 16D Query, Key, and Value projections.
    - Blockwise pooling of Key/Value representations.
    - Dot-product scoring and adaptive top-p mass routing.
    - Differentiable 16D additive output stream (LM-output).
    - Distillation loss (KL divergence) matching predicted block logits to teacher mass.
    """

    def __init__(
        self,
        hidden_size: int,
        index_dim: int = 16,
        block_size: int = 64,
        local_window_blocks: int = 8,
        top_p: float = 0.95,
        min_remote_blocks: int = 0,
        max_remote_blocks: int = 16,
        remote_capacity: int = 16,
        causal: bool = True,
        use_additive_stream: bool = True,
        distill_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.index_dim = index_dim
        self.block_size = block_size
        self.local_window_blocks = local_window_blocks
        self.top_p = top_p
        self.min_remote_blocks = min_remote_blocks
        self.max_remote_blocks = max_remote_blocks
        self.remote_capacity = remote_capacity
        self.causal = causal
        self.use_additive_stream = use_additive_stream
        self.distill_temperature = distill_temperature

        # 16D Projections
        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.out_proj = nn.Linear(index_dim, hidden_size, bias=False)

        # Initialization
        nn.init.normal_(self.q_proj.weight, std=hidden_size ** -0.5)
        nn.init.normal_(self.k_proj.weight, std=hidden_size ** -0.5)
        nn.init.normal_(self.v_proj.weight, std=hidden_size ** -0.5)
        # Small initialization for the additive residual stream so it starts gentle
        nn.init.normal_(self.out_proj.weight, std=1e-3)

    def compute_query_16d(self, x: Tensor) -> Tensor:
        """Project query representations to normalized 16D space.

        Args:
            x: (B, Q, D)
        Returns:
            (B, Q, index_dim)
        """
        q = self.q_proj(x)
        return F.rms_norm(q, (self.index_dim,))

    def compute_kv_blocks_16d(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute pooled 16D key and value vectors per block.

        Args:
            x: (B, N, D) or (N, D) - full element sequence to be blocked
        Returns:
            k_blocks: (B, num_blocks, index_dim)
            v_blocks: (B, num_blocks, index_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        batch, n_elements, _ = x.shape
        if n_elements % self.block_size != 0:
            raise ValueError(
                f"Element length {n_elements} must be divisible by block_size={self.block_size}"
            )
        num_blocks = n_elements // self.block_size

        k_elem = F.rms_norm(self.k_proj(x), (self.index_dim,))
        v_elem = self.v_proj(x)

        # Reshape to (B, num_blocks, block_size, index_dim)
        k_blocks = k_elem.view(batch, num_blocks, self.block_size, self.index_dim).mean(dim=2)
        v_blocks = v_elem.view(batch, num_blocks, self.block_size, self.index_dim).mean(dim=2)

        # L2-normalize block keys for cosine scoring
        k_blocks = F.normalize(k_blocks.float(), dim=-1).to(x.dtype)
        return k_blocks, v_blocks

    def compute_block_scores(self, q_blocks: Tensor, k_blocks: Tensor) -> Tensor:
        """Calculate scaled dot-product block scores.

        Args:
            q_blocks: (B, Q_blocks, index_dim)
            k_blocks: (B, total_blocks, index_dim)
        Returns:
            (B, Q_blocks, total_blocks)
        """
        q_norm = F.normalize(q_blocks.float(), dim=-1).to(q_blocks.dtype)
        k_norm = F.normalize(k_blocks.float(), dim=-1).to(k_blocks.dtype)
        scores = torch.bmm(q_norm, k_norm.transpose(-2, -1)) / math.sqrt(self.index_dim)
        return scores

    def route(
        self,
        block_scores: Tensor,
        top_p: Optional[float] = None,
        max_remote_blocks: Optional[int] = None,
        min_remote_blocks: Optional[int] = None,
    ) -> Route:
        """Perform adaptive top-p mass routing."""
        detached_scores = block_scores.detach()
        return select_blocks(
            detached_scores,
            causal=self.causal,
            block_size=self.block_size,
            local_window_blocks=self.local_window_blocks,
            top_p=self.top_p if top_p is None else top_p,
            min_remote_blocks=self.min_remote_blocks if min_remote_blocks is None else min_remote_blocks,
            max_remote_blocks=self.max_remote_blocks if max_remote_blocks is None else max_remote_blocks,
            remote_capacity=self.remote_capacity,
        )

    def additive_stream(self, block_scores: Tensor, v_blocks: Tensor) -> Tensor:
        """Differentiable 16D soft read over all valid blocks projected to hidden_size.

        This provides continuous gradients directly from the task/LM loss into q_proj,
        k_proj, v_proj, and out_proj.
        """
        batch, query_blocks, total_blocks = block_scores.shape
        device = block_scores.device

        if self.causal:
            qb = torch.arange(query_blocks, device=device)
            pb = torch.arange(total_blocks, device=device)
            valid_mask = pb.view(1, total_blocks) < qb.view(query_blocks, 1)
        else:
            valid_mask = torch.ones((query_blocks, total_blocks), dtype=torch.bool, device=device)

        probs = masked_softmax(
            block_scores,
            valid_mask[None].expand(batch, -1, -1),
        ).to(v_blocks.dtype)

        # context: (B, query_blocks, index_dim)
        context = torch.bmm(probs, v_blocks)
        # Efficient projection before expanding across constituent elements
        out_blocks = self.out_proj(context)
        return out_blocks.repeat_interleave(self.block_size, dim=1)

    def distillation_loss(
        self,
        student_scores: Tensor,
        teacher_block_mass: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Auxiliary KL divergence loss between teacher block mass and student block scores.

        Args:
            student_scores: (B, Q_blocks, total_blocks)
            teacher_block_mass: (B, Q_blocks, total_blocks) - target probabilities
            valid_mask: Optional boolean mask (B, Q_blocks, total_blocks)
        """
        if valid_mask is None:
            if self.causal:
                _, qb_len, pb_len = student_scores.shape
                qb = torch.arange(qb_len, device=student_scores.device)
                pb = torch.arange(pb_len, device=student_scores.device)
                valid_mask = (pb.view(1, pb_len) < qb.view(qb_len, 1))[None].expand_as(student_scores)
            else:
                valid_mask = torch.ones_like(student_scores, dtype=torch.bool)

        masked_student = student_scores.masked_fill(~valid_mask, -float("inf"))
        log_student = F.log_softmax(masked_student.float() / self.distill_temperature, dim=-1)
        log_student = torch.where(valid_mask, log_student, torch.zeros_like(log_student))

        # Teacher target normalized over valid candidate blocks (strictly non-negative)
        teacher_target = teacher_block_mass.float().clamp_min(0.0).masked_fill(~valid_mask, 0.0)
        denom = teacher_target.sum(dim=-1, keepdim=True)
        uniform_fallback = valid_mask.float() / valid_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        target = torch.where(denom > 1e-12, teacher_target / denom.clamp_min(1e-12), uniform_fallback)

        # Cross-entropy / KL: - sum(P_teacher * log P_student)
        kl = -(target.detach() * log_student).sum(dim=-1)
        # Average over valid query blocks
        has_valid_queries = valid_mask.any(dim=-1)
        if has_valid_queries.any():
            return kl[has_valid_queries].mean()
        return student_scores.sum() * 0.0
