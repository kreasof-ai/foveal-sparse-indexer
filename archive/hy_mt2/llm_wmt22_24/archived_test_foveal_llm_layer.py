import pytest
import torch
import torch.nn as nn
from foveal_indexer.foveal_llm_layer import FovealSparseSwiGLU, FovealSparseAttentionLayer


def test_foveal_sparse_swiglu():
    D = 256
    D_ffn = 768
    B = 32
    base = 4
    remote = 4
    
    layer = FovealSparseSwiGLU(
        hidden_dim=D,
        intermediate_dim=D_ffn,
        block_size=B,
        base_blocks=base,
        remote_blocks=remote,
        index_dim=16,
        dtype=torch.float32,
    )
    
    x = torch.randn(2, 8, D)
    out = layer(x)
    assert out.shape == (2, 8, D)
    
    # Test with distillation loss
    out, info = layer(x, compute_distill_loss=True)
    assert "kl_loss" in info
    assert info["kl_loss"].item() >= 0.0
    
    # Test gradient flow
    out.sum().backward()
    assert layer.w_gate_blocks.grad is not None
    assert layer.stream_in.weight.grad is not None


def test_foveal_sparse_attention():
    hidden_size = 256
    num_heads = 4
    num_kv = 2
    head_dim = 64
    chunk_size = 8
    active_blocks = 2 # 2 * 8 = 16 active dims
    
    layer = FovealSparseAttentionLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_key_value_heads=num_kv,
        head_dim=head_dim,
        chunk_size=chunk_size,
        active_blocks_per_head=active_blocks,
        dtype=torch.float32,
    )
    
    x = torch.randn(2, 10, hidden_size)
    out, _ = layer(x)
    assert out.shape == (2, 10, hidden_size)
    
    # Test with distillation loss
    (out, _), info = layer(x, compute_distill_loss=True)
    assert "kl_attn" in info
    assert info["active_dim"] == 16
    
    out.sum().backward()
    assert layer.q_proj.weight.grad is not None
