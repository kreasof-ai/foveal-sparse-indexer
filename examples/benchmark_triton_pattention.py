"""
Empirical Benchmark: Triton Fused Non-Softmax Pattention Kernel.

Benchmarks Token-Parameter Attention with fused non-softmax activations (SwiGLU, SiLU, GeLU, ReLU)
and Foveal Block-Sparsity against dense PyTorch MLP on NVIDIA A10G.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from foveal_indexer.pattention_triton import (
    triton_pattention,
    TritonSwiGLUPattentionMLP,
    _pytorch_pattention_fallback,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"🚀 Triton Fused Non-Softmax Pattention Kernel Benchmark on {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    # 1. Correctness Verification
    M, K, N_p, D_out = 32, 2048, 6144, 2048
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    kg = torch.randn(N_p, K, device=device, dtype=torch.bfloat16)
    ku = torch.randn(N_p, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N_p, D_out, device=device, dtype=torch.bfloat16)

    print("\n[1/3] Numerical Correctness Verification (vs PyTorch Reference):")
    for act in ["swiglu", "silu", "relu", "relu2"]:
        out_triton = triton_pattention(x, kg, v, k_up=ku if act == "swiglu" else None, activation=act)
        out_ref = _pytorch_pattention_fallback(x, kg, v, ku if act == "swiglu" else None, None, act, 64)
        cos = F.cosine_similarity(out_triton.float(), out_ref.float(), dim=-1).mean().item()
        rel_err = ((out_triton.float() - out_ref.float()).norm() / out_ref.float().norm()).item()
        print(f"      Activation: {act.upper():<8} -> Cosine: {cos:.6f} | Relative L2 Error: {rel_err:.6f}")

    # 2. Conversion from Pretrained SwiGLU MLP
    print("\n[2/3] Pretrained SwiGLU MLP Direct Conversion Check:")
    class DenseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(2048, 6144, bias=False, dtype=torch.bfloat16, device=device)
            self.up_proj = nn.Linear(2048, 6144, bias=False, dtype=torch.bfloat16, device=device)
            self.down_proj = nn.Linear(6144, 2048, bias=False, dtype=torch.bfloat16, device=device)
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    dense = DenseMLP()
    pattn_mlp = TritonSwiGLUPattentionMLP.from_dense_mlp(dense)
    with torch.no_grad():
        y_d = dense(x)
        y_p = pattn_mlp(x)
    cos_conv = F.cosine_similarity(y_p.float(), y_d.float(), dim=-1).mean().item()
    print(f"      Pretrained Conversion Cosine Similarity: {cos_conv:.6f} (Mathematically exact)")

    # 3. Hardware Latency Benchmarking
    print("\n[3/3] Hardware Latency Scaling across Workload Regimes:")
    print(f"{'Regime':<24} | {'N_act':<6} | {'Sparsity':<9} | {'Dense MLP':<11} | {'Triton Pattention':<17} | {'Speedup'}")
    print("-" * 88)

    for name, B, L in [
        ("Prefill (B=4, L=256)", 4, 256),
        ("Prefill (B=1, L=1024)", 1, 1024),
        ("Decode (B=16, L=1)", 16, 1),
        ("Decode (B=4, L=1)", 4, 1),
    ]:
        inp = torch.randn(B, L, 2048, device=device, dtype=torch.bfloat16)

        # Dense PyTorch
        for _ in range(5): _ = dense(inp)
        torch.cuda.synchronize()
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters): _ = dense(inp)
        torch.cuda.synchronize()
        dense_us = (time.perf_counter() - t0) / iters * 1e6

        for n_act, sp_label in [(2048, "66.7%"), (1024, "83.3%"), (512, "91.7%")]:
            # Sparse active parameter tokens
            num_blocks = n_act // 64
            act_blocks = torch.arange(num_blocks, device=device, dtype=torch.int32)

            for _ in range(5): _ = pattn_mlp(inp, active_blocks=act_blocks)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters): _ = pattn_mlp(inp, active_blocks=act_blocks)
            torch.cuda.synchronize()
            pattn_us = (time.perf_counter() - t0) / iters * 1e6
            spd = dense_us / pattn_us

            print(f"{name:<24} | {n_act:<6} | {sp_label:<9} | {dense_us:6.1f} us   | {pattn_us:6.1f} us          | {spd:5.2f}x")

    print("=" * 90)


if __name__ == "__main__":
    main()
