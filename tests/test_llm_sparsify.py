import pytest
import torch
import torch.nn as nn
from foveal_indexer.llm_sparsify import (
    SVDBridge,
    compute_layer_redundancy,
    LLMDistillationLoss,
)


def test_svd_bridge_forward():
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    rank = 16

    bridge = SVDBridge(hidden_dim=hidden_dim, rank=rank, dtype=torch.float32)
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    out = bridge(x)

    assert out.shape == (batch_size, seq_len, hidden_dim)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert bridge.down_proj.grad is not None
    assert bridge.up_proj.grad is not None


def test_svd_bridge_from_weight():
    hidden_dim = 64
    rank = 8

    # Generate low-rank weight
    U = torch.randn(hidden_dim, rank)
    V = torch.randn(rank, hidden_dim)
    W = U @ V

    bridge = SVDBridge.from_weight(W, rank=rank)
    x = torch.randn(4, 10, hidden_dim)

    # Ground truth delta: x @ W^T
    expected_delta = x @ W.T
    actual_out = bridge(x)
    actual_delta = actual_out - x

    assert torch.allclose(actual_delta, expected_delta, atol=1e-4)


def test_llm_distillation_loss():
    batch_size = 2
    seq_len = 8
    vocab_size = 50
    hidden_dim = 32

    criterion = LLMDistillationLoss(alpha=1.0, beta=2.0, gamma=1.0)

    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    student_hidden = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss_dict = criterion(student_logits, student_hidden, teacher_hidden, labels)

    assert "loss" in loss_dict
    assert "ce" in loss_dict
    assert "rep_cos" in loss_dict
    assert "rep_mse" in loss_dict

    loss_dict["loss"].backward()
    assert student_logits.grad is not None
    assert student_hidden.grad is not None


def test_compute_layer_redundancy_mock():
    hidden_dim = 32

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return x + 0.01 * self.linear(x)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = MockModel()
    x = torch.randn(2, 5, hidden_dim)
    redundancies = compute_layer_redundancy(model, x)

    assert len(redundancies) == 3
    for idx, dist in redundancies:
        assert isinstance(idx, int)
        assert isinstance(dist, float)
        assert dist >= 0.0


def test_foveal_block_sparse_mlp():
    from foveal_indexer.llm_sparsify import FovealBlockSparseMLP

    hidden_dim = 64
    intermediate_dim = 256
    block_size = 32
    base_blocks = 1
    max_remote_blocks = 2  # Total active = 3 blocks out of 8 = 96 channels = 62.5% sparse

    mlp = FovealBlockSparseMLP(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        block_size=block_size,
        base_blocks=base_blocks,
        max_remote_blocks=max_remote_blocks,
        dtype=torch.float32,
    )

    x = torch.randn(2, 8, hidden_dim, requires_grad=True)
    out, sparsity = mlp(x, return_sparsity=True)

    assert out.shape == (2, 8, hidden_dim)
    assert sparsity == (1.0 - (3 * 32) / 256) * 100.0  # 62.5%

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert mlp.q_proj_16d.weight.grad is not None
    assert mlp.out_proj_16d.weight.grad is not None


def test_foveal_mlp_from_dense():
    from foveal_indexer.llm_sparsify import FovealBlockSparseMLP

    hidden_dim = 64
    intermediate_dim = 128

    class DummyDenseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    dense = DummyDenseMLP()
    foveal_mlp = FovealBlockSparseMLP.from_dense_mlp(
        dense, block_size=32, base_blocks=1, max_remote_blocks=1
    )

    assert foveal_mlp.gate_proj.weight.shape == (intermediate_dim, hidden_dim)
    assert foveal_mlp.block_keys_16d.shape == (4, 16)

    x = torch.randn(2, 4, hidden_dim)
    out = foveal_mlp(x)
    assert out.shape == (2, 4, hidden_dim)


def test_foveal_dynamic_skip_mlp():
    from foveal_indexer.llm_sparsify import FovealDynamicSkipMLP

    hidden_dim = 64
    intermediate_dim = 128

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
        def forward(self, x):
            return self.linear(x)

    base_mlp = DummyMLP()
    skip_mlp = FovealDynamicSkipMLP(base_mlp, hidden_dim=hidden_dim, index_dim=16, threshold=0.5, dtype=torch.float32)

    # 1. Test training mode (differentiable soft gate)
    skip_mlp.train()
    x = torch.randn(2, 4, hidden_dim, requires_grad=True)
    out, gate = skip_mlp(x, return_gate=True)
    assert out.shape == (2, 4, hidden_dim)
    assert gate.shape == (2, 4, 1)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert skip_mlp.q_proj_16d.weight.grad is not None
    assert skip_mlp.bridge_in.weight.grad is not None

    # 2. Test eval mode (hard skip vs exec)
    skip_mlp.eval()
    skip_mlp.threshold = 0.999 # force skip
    out_skip = skip_mlp(x)
    assert skip_mlp.skip_calls > 0

    skip_mlp.threshold = 0.001 # force execute
    out_exec = skip_mlp(x)
    assert skip_mlp.exec_calls > 0


def test_dynamic_sparsity_distillation_loss():
    from foveal_indexer.llm_sparsify import DynamicSparsityDistillationLoss

    batch_size = 2
    seq_len = 4
    hidden_dim = 32
    vocab_size = 100

    criterion = DynamicSparsityDistillationLoss(alpha=1.0, beta=1.0, gamma=0.5, lambda_sparse=0.1)

    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    student_hidden = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    gate = torch.rand(batch_size, seq_len, 1, requires_grad=True)
    gate_probs = [gate]

    loss_dict = criterion(student_logits, student_hidden, teacher_hidden, gate_probs, labels)
    assert "loss" in loss_dict
    assert "sparse_reg" in loss_dict
    assert "mean_gate" in loss_dict

    loss_dict["loss"].backward()
    assert student_logits.grad is not None
    assert gate.grad is not None


def test_triton_swiglu_pattention():
    from foveal_indexer.pattention_triton import triton_pattention, _pytorch_pattention_fallback

    M, K, N_p, D_out = 4, 32, 64, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    x = torch.randn(M, K, device=device, dtype=dtype)
    kg = torch.randn(N_p, K, device=device, dtype=dtype)
    ku = torch.randn(N_p, K, device=device, dtype=dtype)
    v = torch.randn(N_p, D_out, device=device, dtype=dtype)

    for act in ["swiglu", "silu", "relu", "relu2"]:
        out = triton_pattention(x, kg, v, k_up=ku if act == "swiglu" else None, activation=act)
        ref = _pytorch_pattention_fallback(x, kg, v, ku if act == "swiglu" else None, None, act, 32)
        assert out.shape == (M, D_out)
        cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1).mean().item()
        assert cos > 0.999


def test_pattention_conversion_from_dense():
    from foveal_indexer.pattention_triton import TritonSwiGLUPattentionMLP

    hidden_dim = 32
    intermediate_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DummySwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    dense = DummySwiGLU().to(device)
    pattn_mlp = TritonSwiGLUPattentionMLP.from_dense_mlp(dense, block_size=32)

    x = torch.randn(2, 4, hidden_dim, device=device)
    with torch.no_grad():
        y_dense = dense(x)
        y_pattn = pattn_mlp(x)

    cos = torch.nn.functional.cosine_similarity(y_pattn.float(), y_dense.float(), dim=-1).mean().item()
    assert cos > 0.999
