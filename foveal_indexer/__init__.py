"""Foveal Sparse Indexer: Unified multi-resolution sparsity for Attention and MatMul."""

from .core import FovealIndexer, Route, select_blocks, masked_softmax
from .attention import FovealSparseAttention, FovealKVCache
from .tokenformer import FovealTokenformerLinear, FovealTokenformerFFN
from .block_matmul import BlockSparseLinear, BlockwiseMatMul2D
from .vit import SmallViT, PatchEmbedding, FovealVisionAttention, DenseViTBlock, FovealViTBlock
from .sparsify import SVDRouter, FovealSparsifiedModel, sparsify_vision_transformer
from .triton_ops import (
    is_triton_available,
    triton_foveal_sparse_attention,
    triton_fused_sram_indexer_attention,
    triton_block_sparse_linear,
)

__all__ = [
    "FovealIndexer",
    "Route",
    "select_blocks",
    "masked_softmax",
    "FovealSparseAttention",
    "FovealKVCache",
    "FovealTokenformerLinear",
    "FovealTokenformerFFN",
    "BlockSparseLinear",
    "BlockwiseMatMul2D",
    "SmallViT",
    "PatchEmbedding",
    "FovealVisionAttention",
    "DenseViTBlock",
    "FovealViTBlock",
    "SVDRouter",
    "FovealSparsifiedModel",
    "sparsify_vision_transformer",
    "is_triton_available",
    "triton_foveal_sparse_attention",
    "triton_fused_sram_indexer_attention",
    "triton_block_sparse_linear",
]
