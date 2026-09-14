"""Sparsification of pretrained Vision Transformers using 16D Foveal Indexer with Offline SVD Router Initialization."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SVDRouter(nn.Module):
    """16-dimensional Foveal Router with Offline SVD Initialization.

    Projects input token features into a normalized 16D indexer subspace,
    predicts token saliency scores, and optionally provides a low-rank
    differentiable additive stream.
    """

    def __init__(
        self,
        hidden_size: int,
        index_dim: int = 16,
        use_additive_stream: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.index_dim = index_dim
        self.use_additive_stream = use_additive_stream

        # 16D Projections
        self.proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.score = nn.Linear(index_dim, 1, bias=False)

        if use_additive_stream:
            self.additive_proj = nn.Linear(index_dim, hidden_size, bias=False)
            nn.init.zeros_(self.additive_proj.weight)
        else:
            self.register_parameter("additive_proj", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proj.weight, std=self.hidden_size ** -0.5)
        nn.init.normal_(self.score.weight, std=self.index_dim ** -0.5)
        if self.additive_proj is not None:
            nn.init.zeros_(self.additive_proj.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute saliency scores and optional additive stream.

        Args:
            x: (B, N, D) token representations.

        Returns:
            scores: (B, N) predicted saliency values.
            additive: (B, N, D) or None if additive stream disabled.
        """
        # Project to 16D and apply RMSNorm
        z = F.rms_norm(self.proj(x), (self.index_dim,))
        scores = self.score(z).squeeze(-1)

        additive = self.additive_proj(z) if self.additive_proj is not None else None
        return scores, additive

    @torch.no_grad()
    def calibrate_offline_svd(
        self,
        features: Tensor,
        teacher_scores: Tensor,
        ridge_lambda: float = 1e-2,
    ) -> Dict[str, float]:
        """Compute closed-form SVD router weights from calibration features.

        Args:
            features: (M, D) or (B, N, D) patch representations.
            teacher_scores: (M, 1) or (B, N) target saliency values from teacher attention.
            ridge_lambda: L2 regularization strength for closed-form ridge regression.

        Returns:
            metrics: dictionary with singular value metrics and alignment statistics.
        """
        device = features.device
        dtype = features.dtype

        # Flatten to 2D
        X = features.reshape(-1, self.hidden_size).float()
        Y = teacher_scores.reshape(-1, 1).float()

        # Compute SVD of token covariance
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # Top-16 right singular vectors span the dominant feature subspace
        V16 = Vh[: self.index_dim, :].T  # (D, 16)

        # 16D normalized projections on calibration data
        Z = F.rms_norm(X @ V16, (self.index_dim,))

        # Closed-form ridge regression for the 16D score head:
        # w = (Z^T Z + lambda * I)^(-1) Z^T Y
        reg = ridge_lambda * torch.eye(self.index_dim, device=device)
        w_head = torch.linalg.solve(Z.T @ Z + reg, Z.T @ Y)  # (16, 1)

        # Update module parameters
        self.proj.weight.copy_(V16.T.to(dtype=dtype))
        self.score.weight.copy_(w_head.T.to(dtype=dtype))

        if self.additive_proj is not None:
            # Scale additive projection gently
            self.additive_proj.weight.copy_((V16 * 1e-3).to(dtype=dtype))

        # Variance explained by top 16 singular components
        total_variance = (S ** 2).sum().item()
        top16_variance = (S[: self.index_dim] ** 2).sum().item()
        explained_ratio = top16_variance / max(total_variance, 1e-12)

        return {
            "top16_explained_variance": explained_ratio,
            "top_singular_value": S[0].item(),
            "score_head_norm": w_head.norm().item(),
        }


class FovealSparsifiedModel(nn.Module):
    """Vision Transformer wrapper with Foveal Sparsification.

    Sparsifies token representations after `stage_split_block` transformer blocks.
    Keeps only `k_keep` most salient patch tokens selected by the 16D SVDRouter.
    Supports rotary positional embeddings (RoPE) as used in EVA-02.
    """

    def __init__(
        self,
        vit: nn.Module,
        stage_split_block: int = 2,
        k_keep: int = 384,
        index_dim: int = 16,
        use_additive_stream: bool = True,
        use_soft_drop_summary: bool = True,
        soft_drop_scale: float = 0.05,
    ):
        super().__init__()
        self.vit = vit
        self.stage_split_block = stage_split_block
        self.k_keep = k_keep
        self.use_soft_drop_summary = use_soft_drop_summary
        self.soft_drop_scale = soft_drop_scale

        # Hidden size from backbone
        if hasattr(vit, "embed_dim"):
            self.embed_dim = vit.embed_dim
        elif hasattr(vit, "head"):
            self.embed_dim = vit.head.in_features
        else:
            self.embed_dim = vit.blocks[0].norm1.weight.shape[0]

        # 16D Foveal Router
        self.router = SVDRouter(
            hidden_size=self.embed_dim,
            index_dim=index_dim,
            use_additive_stream=use_additive_stream,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Patch embedding and initial positional embeddings
        if hasattr(self.vit, "_pos_embed"):
            pos_out = self.vit._pos_embed(self.vit.patch_embed(x))
            if isinstance(pos_out, tuple):
                feats, rope = pos_out
            else:
                feats, rope = pos_out, None
        else:
            feats = self.vit.patch_embed(x)
            if hasattr(self.vit, "pos_embed"):
                feats = feats + self.vit.pos_embed
            rope = None

        if hasattr(self.vit, "norm_pre"):
            feats = self.vit.norm_pre(feats)

        # Stage 1: Dense execution for the first `stage_split_block` layers
        for i in range(self.stage_split_block):
            block = self.vit.blocks[i]
            if rope is not None:
                res = block(feats, rope=rope)
            else:
                res = block(feats)
            feats = res[0] if isinstance(res, tuple) else res

        # Split prefix tokens (CLS token) and patch tokens
        num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        prefix_tokens = feats[:, :num_prefix]
        patches = feats[:, num_prefix:]
        B, P, C = patches.shape

        # Router prediction
        scores, additive = self.router(patches)

        k = min(self.k_keep, P)
        top_idx = scores.topk(k, dim=1).indices  # (B, k)
        b_idx = torch.arange(B, device=x.device)[:, None]
        top_patches = patches[b_idx, top_idx]

        if additive is not None:
            top_patches = top_patches + additive[b_idx, top_idx]

        # Slice RoPE for retained patch coordinates if model uses RoPE
        rope_active = rope[top_idx] if rope is not None else None

        # Discarded token soft-pooling to preserve residual energy
        if self.use_soft_drop_summary and num_prefix > 0:
            mask = torch.zeros(B, P, dtype=torch.bool, device=x.device).scatter_(1, top_idx, True)
            drop_mask = ~mask
            drop_sum = (patches * drop_mask.unsqueeze(-1)).sum(dim=1)
            drop_count = drop_mask.sum(dim=1, keepdim=True).clamp_min(1)
            drop_mean = drop_sum / drop_count
            prefix_tokens = prefix_tokens + self.soft_drop_scale * drop_mean.unsqueeze(1)

        feats = torch.cat([prefix_tokens, top_patches], dim=1)

        # Stage 2: Fast sparse execution on only retained tokens
        num_blocks = len(self.vit.blocks)
        for i in range(self.stage_split_block, num_blocks):
            block = self.vit.blocks[i]
            if rope_active is not None:
                res = block(feats, rope=rope_active)
            else:
                res = block(feats)
            feats = res[0] if isinstance(res, tuple) else res

        # Head / Classification
        out = self.vit.norm(feats)
        if hasattr(self.vit, "forward_head"):
            return self.vit.forward_head(out)
        elif hasattr(self.vit, "head"):
            pooled = out.mean(dim=1)
            return self.vit.head(pooled)
        return out


def sparsify_vision_transformer(
    model: nn.Module,
    stage_split_block: int = 2,
    k_keep: int = 384,
    index_dim: int = 16,
    use_additive_stream: bool = True,
    use_soft_drop_summary: bool = True,
    soft_drop_scale: float = 0.05,
) -> FovealSparsifiedModel:
    """Convenience factory function to wrap a vision model into Foveal Sparse."""
    return FovealSparsifiedModel(
        vit=model,
        stage_split_block=stage_split_block,
        k_keep=k_keep,
        index_dim=index_dim,
        use_additive_stream=use_additive_stream,
        use_soft_drop_summary=use_soft_drop_summary,
        soft_drop_scale=soft_drop_scale,
    )
