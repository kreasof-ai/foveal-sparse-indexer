"""Foveal Sparse Indexer: Unified multi-resolution sparsity for Attention and MatMul."""

from .core import FovealIndexer, Route, select_blocks, masked_softmax
from .attention import FovealSparseAttention, FovealKVCache
from .tokenformer import FovealTokenformerLinear, FovealTokenformerFFN
from .block_matmul import BlockSparseLinear, BlockwiseMatMul2D

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
]
