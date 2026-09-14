"""Tests for SVDRouter and FovealSparsifiedModel."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from foveal_indexer.sparsify import SVDRouter, FovealSparsifiedModel, sparsify_vision_transformer
from foveal_indexer.vit import SmallViT


def test_svd_router_forward():
    router = SVDRouter(hidden_size=64, index_dim=16, use_additive_stream=True)
    x = torch.randn(4, 32, 64)
    scores, additive = router(x)

    assert scores.shape == (4, 32)
    assert additive is not None
    assert additive.shape == (4, 32, 64)


def test_svd_router_calibration():
    router = SVDRouter(hidden_size=64, index_dim=16, use_additive_stream=True)
    features = torch.randn(128, 32, 64)
    # Synthetic target attention scores
    teacher_scores = torch.rand(128, 32)

    metrics = router.calibrate_offline_svd(features, teacher_scores, ridge_lambda=1e-2)

    assert "top16_explained_variance" in metrics
    assert metrics["top16_explained_variance"] > 0.0
    assert metrics["score_head_norm"] > 0.0

    # Ensure weights were populated and are finite
    assert torch.isfinite(router.proj.weight).all()
    assert torch.isfinite(router.score.weight).all()
    assert torch.isfinite(router.additive_proj.weight).all()


def test_foveal_sparsified_small_vit():
    vit = SmallViT(
        variant="dense",
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
    )

    sparse_model = sparsify_vision_transformer(
        model=vit,
        stage_split_block=1,
        k_keep=32,
        index_dim=16,
    )

    x = torch.randn(2, 3, 32, 32)
    out = sparse_model(x)

    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_svd_router_backward():
    router = SVDRouter(hidden_size=64, index_dim=16, use_additive_stream=True)
    x = torch.randn(2, 16, 64, requires_grad=True)
    scores, additive = router(x)

    loss = scores.sum() + additive.sum()
    loss.backward()

    assert router.proj.weight.grad is not None
    assert router.score.weight.grad is not None
    assert router.additive_proj.weight.grad is not None
    assert x.grad is not None
