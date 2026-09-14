"""Foveal Sparse Indexer: Unified multi-resolution sparsity for Attention and MatMul."""

from .core import FovealIndexer, Route, select_blocks, masked_softmax
from .attention import FovealSparseAttention, FovealKVCache
from .tokenformer import FovealTokenformerLinear, FovealTokenformerFFN
from .block_matmul import BlockSparseLinear, BlockwiseMatMul2D
from .vit import SmallViT, PatchEmbedding, FovealVisionAttention, DenseViTBlock, FovealViTBlock

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
]
