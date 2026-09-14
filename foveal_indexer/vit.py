"""Vision Transformer (ViT) with Dense and Foveal Sparse variants for MNIST / Vision benchmarks."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .core import FovealIndexer, masked_softmax
from .block_matmul import BlockSparseLinear
from .triton_ops import (
    is_triton_available,
    triton_foveal_sparse_attention,
    triton_fused_sram_indexer_attention,
)


class PatchEmbedding(nn.Module):
    """Converts 2D images into flattened sequences of patch embeddings."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        h = self.proj(x)  # (B, D, H/P, W/P)
        h = h.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return h + self.pos_embed


class FovealVisionAttention(nn.Module):
    """Bidirectional Vision Attention with 16D Foveal Indexing over patch blocks."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        block_size: int = 16,
        index_dim: int = 16,
        top_p: float = 0.8,
        max_remote_blocks: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.indexer = FovealIndexer(
            hidden_size=embed_dim,
            index_dim=index_dim,
            block_size=block_size,
            local_window_blocks=0,
            top_p=top_p,
            min_remote_blocks=0,
            max_remote_blocks=max_remote_blocks,
            remote_capacity=max_remote_blocks,
            causal=False,
            use_additive_stream=True,
        )

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        batch, seq_len, _ = x.shape
        num_blocks = seq_len // self.block_size
        device = x.device

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 16D Indexer projections
        x_detach = x.detach()
        qi = self.indexer.compute_query_16d(x_detach)
        kp, vp = self.indexer.compute_kv_blocks_16d(x_detach)
        qb = qi[:, :: self.block_size]
        scores = self.indexer.compute_block_scores(qb, kp)  # (B, num_blocks, num_blocks)

        if device.type == "cuda" and not compute_teacher_loss and num_blocks > 2:
            # Active-block gathering on GPU: enables unmasked FlashAttention forward & backward with zero mask overhead!
            eye_mask = torch.eye(num_blocks, dtype=torch.bool, device=device)[None]
            remote_scores = scores.masked_fill(eye_mask, -1e4)
            k_remote = min(self.indexer.max_remote_blocks, max(1, num_blocks - 1))
            best_remote = remote_scores.mean(dim=1).topk(k=k_remote, dim=-1).indices

            offsets = torch.arange(self.block_size, device=device)
            act_list = [offsets[None, :].expand(batch, -1)]
            for r in range(k_remote):
                act_list.append(best_remote[:, r:r+1] * self.block_size + offsets[None, :])
            act_tokens = torch.cat(act_list, dim=1)

            batch_idx = torch.arange(batch, device=device)[:, None]
            k_act = k.transpose(1, 2)[batch_idx, act_tokens].transpose(1, 2)
            v_act = v.transpose(1, 2)[batch_idx, act_tokens].transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(q, k_act, v_act, scale=self.scale)
            sparsity = 1.0 - (act_tokens.shape[-1] / seq_len)
        else:
            # CPU fallback or teacher distillation mode
            eye_mask = torch.eye(num_blocks, dtype=torch.bool, device=device)[None].expand(batch, -1, -1)
            remote_scores = scores.masked_fill(eye_mask, -1e4)
            probs = F.softmax(remote_scores, dim=-1)

            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            needed = (cumulative < self.indexer.top_p).sum(dim=-1) + 1
            needed = torch.clamp(needed, 0, self.indexer.max_remote_blocks)

            # Vectorized block-sparse attention mask (zero Python loops)
            q_blk_idx = torch.arange(seq_len, device=device) // self.block_size
            self_mask = (q_blk_idx[:, None] == q_blk_idx[None, :])[None].expand(batch, -1, -1)

            capacity = sorted_idx.shape[-1]
            slot_idx = torch.arange(capacity, device=device).view(1, 1, capacity)
            valid = slot_idx < needed[..., None]
            safe_remote = sorted_idx.masked_fill(~valid, 0)

            block_active = torch.zeros(batch, num_blocks, num_blocks, dtype=torch.bool, device=device)
            block_active.scatter_(2, safe_remote, valid)

            expanded_remote = block_active.repeat_interleave(self.block_size, dim=1).repeat_interleave(
                self.block_size, dim=2
            )
            allowed = self_mask | expanded_remote

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed[:, None, :, :], scale=self.scale
            )
            sparsity = 1.0 - allowed.float().mean().item()

        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        out = self.out_proj(attn_out)

        # 16D Additive stream
        add_out = self.indexer.additive_stream(scores, vp)
        out = out + add_out

        # Auxiliary teacher distillation
        distill_loss = torch.tensor(0.0, device=device)
        if compute_teacher_loss:
            with torch.no_grad():
                dense_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                dense_probs = torch.softmax(dense_scores, dim=-1).mean(dim=1)
                anchor_probs = dense_probs[:, :: self.block_size, :]
                teacher_mass = anchor_probs.view(
                    batch, num_blocks, num_blocks, self.block_size
                ).sum(dim=-1)
            distill_loss = self.indexer.distillation_loss(scores, teacher_mass)

        return out, {"distill_loss": distill_loss, "sparsity": sparsity}


class DenseViTBlock(nn.Module):
    """Standard Dense Vision Transformer Block."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_dim: int = 128):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: Tensor, compute_teacher_loss: bool = False) -> Tuple[Tensor, Dict[str, Any]]:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, {"distill_loss": torch.tensor(0.0, device=x.device), "sparsity": 0.0}


class FovealViTBlock(nn.Module):
    """Foveal Sparse Vision Transformer Block (Sparse Attention + Sparse MLP)."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_dim: int = 128,
        block_size: int = 16,
        top_p: float = 0.8,
        max_remote_blocks: int = 2,
        mlp_block_size: int = 32,
        mlp_base_blocks: int = 1,
        mlp_max_remote_blocks: int = 2,
        mlp_top_p: float = 0.5,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = FovealVisionAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            top_p=top_p,
            max_remote_blocks=max_remote_blocks,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        # Block-sparse MLP expansion
        self.fc1 = BlockSparseLinear(
            in_features=embed_dim,
            out_features=mlp_dim,
            block_size=mlp_block_size,
            base_blocks=mlp_base_blocks,
            max_remote_blocks=mlp_max_remote_blocks,
            top_p=mlp_top_p,
        )
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: Tensor, compute_teacher_loss: bool = False) -> Tuple[Tensor, Dict[str, Any]]:
        h = self.ln1(x)
        attn_out, aux = self.attn(h, compute_teacher_loss=compute_teacher_loss)
        x = x + attn_out

        # Foveal Block-Sparse MLP
        mlp_in = self.ln2(x)
        h_mlp, aux_mlp = self.fc1(mlp_in, compute_teacher_loss=compute_teacher_loss)
        h_mlp = F.gelu(h_mlp)
        mlp_out = self.fc2(h_mlp)
        x = x + mlp_out

        total_distill = aux["distill_loss"] + aux_mlp["distill_loss"]
        return x, {
            "distill_loss": total_distill,
            "attn_sparsity": aux["sparsity"],
            "mlp_sparsity": aux_mlp["sparsity"],
        }

    def refresh_cache(self) -> None:
        """Update cached block keys and values for evaluation."""
        self.fc1.refresh_block_cache()


class SmallViT(nn.Module):
    """Small Vision Transformer supporting Dense or Foveal Sparse blocks."""

    def __init__(
        self,
        variant: str = "dense",  # "dense" or "foveal"
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 128,
        patch_block_size: int = 16,
        attn_top_p: float = 0.8,
        attn_max_remote_blocks: int = 2,
        mlp_block_size: int = 32,
        mlp_base_blocks: int = 1,
        mlp_max_remote_blocks: int = 2,
        mlp_top_p: float = 0.5,
    ):
        super().__init__()
        self.variant = variant
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        blocks = []
        for _ in range(depth):
            if variant == "dense":
                blocks.append(DenseViTBlock(embed_dim, num_heads, mlp_dim))
            elif variant == "foveal":
                blocks.append(
                    FovealViTBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        block_size=patch_block_size,
                        top_p=attn_top_p,
                        max_remote_blocks=attn_max_remote_blocks,
                        mlp_block_size=mlp_block_size,
                        mlp_base_blocks=mlp_base_blocks,
                        mlp_max_remote_blocks=mlp_max_remote_blocks,
                        mlp_top_p=mlp_top_p,
                    )
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def refresh_cache(self) -> None:
        """Update cached block keys and values across all sparse layers."""
        for block in self.blocks:
            if hasattr(block, "refresh_cache"):
                block.refresh_cache()

    def forward(
        self,
        x: Tensor,
        compute_teacher_loss: bool = False,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        h = self.patch_embed(x)
        total_distill = torch.tensor(0.0, device=x.device)
        attn_sparsities = []
        mlp_sparsities = []

        for block in self.blocks:
            h, aux = block(h, compute_teacher_loss=compute_teacher_loss)
            total_distill = total_distill + aux["distill_loss"]
            if "attn_sparsity" in aux:
                attn_sparsities.append(aux["attn_sparsity"])
            if "mlp_sparsity" in aux:
                mlp_sparsities.append(aux["mlp_sparsity"])

        h = self.norm(h)
        # Global Average Pooling over patches
        pooled = h.mean(dim=1)
        logits = self.head(pooled)

        mean_attn_sp = sum(attn_sparsities) / len(attn_sparsities) if attn_sparsities else 0.0
        mean_mlp_sp = sum(mlp_sparsities) / len(mlp_sparsities) if mlp_sparsities else 0.0

        return logits, {
            "distill_loss": total_distill,
            "mean_attn_sparsity": mean_attn_sp,
            "mean_mlp_sparsity": mean_mlp_sp,
        }
