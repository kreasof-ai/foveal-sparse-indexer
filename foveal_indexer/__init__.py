"""Foveal Sparse Indexer: Unified multi-resolution sparsity for Attention and MatMul."""

from .core import FovealIndexer, Route, select_blocks, masked_softmax
from .attention import FovealSparseAttention, FovealKVCache
from .tokenformer import FovealTokenformerLinear, FovealTokenformerFFN
from .block_matmul import BlockSparseLinear, BlockwiseMatMul2D
from .vit import SmallViT, PatchEmbedding, FovealVisionAttention, DenseViTBlock, FovealViTBlock
from .sparsify import SVDRouter, FovealSparsifiedModel, sparsify_vision_transformer
from .yolo_sparsify import (
    FovealAttention2D,
    FovealPSABlock,
    OfflineSVDChannelCompressor,
    FovealYOLOCascade,
    YOLOKnowledgeDistillationLoss,
)
from .llm_sparsify import (
    SVDBridge,
    compute_layer_redundancy,
    calibrate_svd_bridge,
    LLMDistillationLoss,
    FovealBlockSparseMLP,
    sparsify_llm_mlps,
    FovealDynamicSkipMLP,
    sparsify_llm_dynamic_skips,
    DynamicSparsityDistillationLoss,
)
from .pattention_triton import (
    triton_pattention,
    TritonSwiGLUPattentionMLP,
    convert_llm_mlp_to_pattention,
)
from .foveal_pattention import (
    FovealPattentionMLP,
    sparsify_model_with_foveal_pattention,
    FovealPattentionDistillationLoss,
)
from .triton_ops import (
    is_triton_available,
    triton_foveal_sparse_attention,
    triton_fused_sram_indexer_attention,
    triton_block_sparse_linear,
)
from .optimizers import (
    Muon,
    HybridMuonAdamW,
    zeropower_via_newtonschulz5,
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
    "FovealAttention2D",
    "FovealPSABlock",
    "OfflineSVDChannelCompressor",
    "FovealYOLOCascade",
    "YOLOKnowledgeDistillationLoss",
    "SVDBridge",
    "compute_layer_redundancy",
    "calibrate_svd_bridge",
    "LLMDistillationLoss",
    "FovealBlockSparseMLP",
    "sparsify_llm_mlps",
    "FovealDynamicSkipMLP",
    "sparsify_llm_dynamic_skips",
    "DynamicSparsityDistillationLoss",
    "triton_pattention",
    "TritonSwiGLUPattentionMLP",
    "convert_llm_mlp_to_pattention",
    "FovealPattentionMLP",
    "sparsify_model_with_foveal_pattention",
    "FovealPattentionDistillationLoss",
    "Muon",
    "HybridMuonAdamW",
    "zeropower_via_newtonschulz5",
]
