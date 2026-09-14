"""Unit tests for YOLO11 Foveal Sparsification modules."""

import pytest
import torch
import torch.nn as nn
from foveal_indexer.yolo_sparsify import (
    FovealAttention2D,
    FovealPSABlock,
    OfflineSVDChannelCompressor,
    FovealYOLOCascade,
    YOLOKnowledgeDistillationLoss,
)


def test_foveal_attention_2d_forward_and_backward():
    """Test 2D Foveal Attention with spatial token routing and gradients."""
    dim = 64
    attn = FovealAttention2D(dim=dim, num_heads=4, k_keep=16, index_dim=16)
    x = torch.randn(2, dim, 8, 8, requires_grad=True)

    out = attn(x)
    assert out.shape == (2, dim, 8, 8)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert attn.qkv.weight.grad is not None
    assert attn.router.proj.weight.grad is not None


def test_foveal_psa_block_forward():
    """Test drop-in FovealPSABlock module."""
    c = 128
    block = FovealPSABlock(c=c, num_heads=4, shortcut=True, k_keep=32)
    x = torch.randn(2, c, 10, 10)

    out = block(x)
    assert out.shape == (2, c, 10, 10)
    assert torch.isfinite(out).all()


def test_offline_svd_channel_compressor():
    """Test SVD-based weight transfer from teacher conv to student conv."""
    src_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    tgt_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    # Initialize src with known pattern
    nn.init.orthogonal_(src_conv.weight.data.flatten(1))

    OfflineSVDChannelCompressor.compress_conv2d(src_conv, tgt_conv)

    assert tgt_conv.weight.shape == (256, 256, 3, 3)
    assert torch.isfinite(tgt_conv.weight).all()
    assert tgt_conv.weight.abs().sum() > 0.0


def test_yolo_distillation_loss():
    """Test Knowledge Distillation feature loss computation."""
    loss_fn = YOLOKnowledgeDistillationLoss()
    s_feats = [torch.randn(2, 64, 20, 20, requires_grad=True)]
    t_feats = [torch.randn(2, 128, 20, 20)]

    loss_dict = loss_fn(s_feats, t_feats)
    assert "total_loss" in loss_dict
    assert loss_dict["total_loss"].item() >= 0.0

    loss_dict["total_loss"].backward()
    assert s_feats[0].grad is not None
