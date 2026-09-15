"""
Unit tests for Foveal Token-Parameter Attention (Pattention), SVD Router Init,
Additive Stream, and Dense KL Distillation.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from foveal_indexer.foveal_pattention import (
    FovealPattentionMLP,
    sparsify_model_with_foveal_pattention,
    FovealPattentionDistillationLoss,
)
from foveal_indexer.pattention_triton import triton_fused_swiglu, HAS_TRITON


class MockDenseMLP(nn.Module):
    def __init__(self, hidden_dim=256, intermediate_dim=768):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


def test_foveal_pattention_conversion():
    hidden_dim = 256
    intermediate_dim = 512
    dense_mlp = MockDenseMLP(hidden_dim, intermediate_dim)
    
    patt = FovealPattentionMLP.from_dense_mlp(
        dense_mlp,
        block_size=64,
        base_blocks=2,
        remote_blocks=4,
        index_dim=16,
    )
    
    assert patt.k_gate.shape == (intermediate_dim, hidden_dim)
    assert patt.k_up.shape == (intermediate_dim, hidden_dim)
    assert patt.v.shape == (intermediate_dim, hidden_dim)
    assert patt.num_blocks == 8
    assert patt.total_active_blocks == 6


def test_foveal_pattention_svd_calibration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 256
    intermediate_dim = 512
    dense_mlp = MockDenseMLP(hidden_dim, intermediate_dim).to(device)
    
    patt = FovealPattentionMLP.from_dense_mlp(
        dense_mlp,
        block_size=64,
        base_blocks=2,
        remote_blocks=4,
        index_dim=16,
    ).to(device)
    
    calib_feats = torch.randn(64, hidden_dim, device=device)
    metrics = patt.calibrate_offline_svd(calib_feats)
    
    assert "top16_explained_variance" in metrics
    assert "initial_kl_divergence" in metrics
    assert metrics["sparsity_pct"] == 25.0


def test_foveal_pattention_forward_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 256
    intermediate_dim = 512
    dense_mlp = MockDenseMLP(hidden_dim, intermediate_dim).to(device)
    
    patt = FovealPattentionMLP.from_dense_mlp(
        dense_mlp,
        block_size=64,
        base_blocks=2,
        remote_blocks=4,
        index_dim=16,
    ).to(device)
    
    x = torch.randn(2, 16, hidden_dim, device=device, requires_grad=True)
    patt.train()
    out, info = patt(x, return_router_info=True)
    
    assert out.shape == (2, 16, hidden_dim)
    assert "scores" in info
    assert "active_blocks" in info
    
    loss = out.sum()
    loss.backward()
    
    assert patt.q_proj_16d.weight.grad is not None
    assert patt.block_keys_16d.grad is not None
    assert patt.out_proj_16d.weight.grad is not None


def test_foveal_pattention_without_base():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 256
    intermediate_dim = 512
    dense_mlp = MockDenseMLP(hidden_dim, intermediate_dim).to(device)
    
    patt = FovealPattentionMLP.from_dense_mlp(
        dense_mlp,
        block_size=64,
        base_blocks=0,
        remote_blocks=4,
        index_dim=16,
        use_fused_kernel=True,
    ).to(device)
    
    x = torch.randn(2, 16, hidden_dim, device=device)
    patt.eval()
    out, info = patt(x, return_router_info=True)
    
    assert out.shape == (2, 16, hidden_dim)
    assert len(info["active_blocks"]) == 4
    # Test decode caching
    patt.clear_cache()
    x_step = torch.randn(2, 1, hidden_dim, device=device)
    out_step = patt(x_step)
    assert out_step.shape == (2, 1, hidden_dim)


def test_distillation_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FovealPattentionDistillationLoss(alpha_ce=1.0, beta_cos=1.0, gamma_mid=0.5, lambda_kl=0.5)
    
    s_logits = torch.randn(2, 10, 100, device=device)
    labels = torch.randint(0, 100, (2, 10), device=device)
    s_hidden = torch.randn(2, 10, 64, device=device)
    t_hidden = torch.randn(2, 10, 64, device=device)
    
    s_router = [torch.randn(2, 10, 8, device=device)]
    t_targets = [F.softmax(torch.randn(2, 10, 8, device=device), dim=-1)]
    
    losses = criterion(
        student_logits=s_logits,
        student_hidden=s_hidden,
        teacher_hidden=t_hidden,
        student_router_scores=s_router,
        teacher_block_targets=t_targets,
        labels=labels,
    )
    
    assert "loss" in losses
    assert "loss_ce" in losses
    assert "loss_cos" in losses
    assert "loss_kl" in losses
    assert losses["loss"].item() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_triton_fused_swiglu_correctness():
    device = torch.device("cuda")
    g = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    u = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    g_ref = g.detach().clone().requires_grad_(True)
    u_ref = u.detach().clone().requires_grad_(True)

    out_triton = triton_fused_swiglu(g, u)
    out_ref = F.silu(g_ref) * u_ref

    cos_fwd = F.cosine_similarity(out_triton.float().flatten(), out_ref.float().flatten(), dim=0).item()
    assert cos_fwd > 0.9999

    loss_triton = (out_triton * 3.0).sum()
    loss_triton.backward()

    loss_ref = (out_ref * 3.0).sum()
    loss_ref.backward()

    cos_dg = F.cosine_similarity(g.grad.float().flatten(), g_ref.grad.float().flatten(), dim=0).item()
    cos_du = F.cosine_similarity(u.grad.float().flatten(), u_ref.grad.float().flatten(), dim=0).item()
    assert cos_dg > 0.9999
    assert cos_du > 0.9999
