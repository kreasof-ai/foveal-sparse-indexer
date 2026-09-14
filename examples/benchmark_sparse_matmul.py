"""Comprehensive Benchmark: Foveal Sparse MatMul vs Dense MatMul on CPU.

Benchmarks:
1. Blockwise Sparse Linear / GEMM Scaling across Output Dimension (K=2048, N=2048 to 16384).
2. Tokenformer Parameter-Attention MatMul across Parameter Token Count (D=1024, N_param=2048 to 16384).
3. Arithmetic FLOP Breakdown & Theoretical Reduction.
"""

import time
import torch
import torch.nn.functional as F

from foveal_indexer.block_matmul import BlockSparseLinear
from foveal_indexer.tokenformer import FovealTokenformerLinear


def time_benchmark(fn, warmup=5, reps=40):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def benchmark_block_sparse_gemm():
    print("=" * 86)
    print(" 1. BLOCKWISE SPARSE LINEAR vs DENSE LINEAR (Decode Step M=1, K=2048, Block=64)")
    print("=" * 86)

    device = torch.device("cpu")
    torch.manual_seed(42)
    torch.set_num_threads(1)  # Single thread for stable, deterministic benchmarking
    K = 2048

    print(f"{'N_dim':<7} | {'Dense (ms)':<11} | {'Foveal 75% (ms)':<17} | {'Foveal 87.5% (ms)':<19} | {'Speedup (87.5%)':<15} | {'FLOPs Reduced':<13}")
    print("-" * 86)

    for N in [2048, 4096, 8192, 16384]:
        dense = torch.nn.Linear(K, N, bias=False).to(device)

        # 75% sparse (active = 1/4 of N)
        base_75 = max(1, N // 256)
        max_rem_75 = max(1, (N // 64) // 4 * 3)
        sp_75 = BlockSparseLinear(
            in_features=K, out_features=N, block_size=64,
            base_blocks=base_75, max_remote_blocks=max_rem_75,
            remote_capacity=max_rem_75, top_p=0.4
        ).to(device)
        sp_75.eval()
        sp_75.refresh_block_cache()

        # 87.5% sparse (active = 1/8 of N)
        base_87 = max(1, N // 512)
        max_rem_87 = max(1, N // 512)
        sp_87 = BlockSparseLinear(
            in_features=K, out_features=N, block_size=64,
            base_blocks=base_87, max_remote_blocks=max_rem_87,
            remote_capacity=max_rem_87, top_p=0.25
        ).to(device)
        sp_87.eval()
        sp_87.refresh_block_cache()

        x = torch.randn(1, 1, K, device=device)

        d_ms = time_benchmark(lambda: dense(x))
        s75_ms = time_benchmark(lambda: sp_75(x))
        s87_ms = time_benchmark(lambda: sp_87(x))

        speedup = d_ms / s87_ms
        print(f"{N:<7} | {d_ms:<11.3f} | {s75_ms:<17.3f} | {s87_ms:<19.3f} | {speedup:<15.2f}x | 87.5%")


def benchmark_tokenformer():
    print("\n" + "=" * 86)
    print(" 2. TOKENFORMER TOKEN-PARAMETER ATTENTION (Decode Step M=1, D=1024, Block=64)")
    print("=" * 86)

    device = torch.device("cpu")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    D = 1024

    print(f"{'N_param':<8} | {'Dense TF (ms)':<14} | {'Foveal TF (ms)':<15} | {'Active Tokens':<14} | {'Param Sparsity':<15} | {'Speedup':<8}")
    print("-" * 86)

    for N in [2048, 4096, 8192, 16384]:
        dense = FovealTokenformerLinear(
            in_features=D, out_features=D, num_param_tokens=N,
            param_block_size=64, base_blocks=N // 64, max_remote_blocks=0
        ).to(device)
        dense.eval()
        dense.refresh_param_cache()

        sparse = FovealTokenformerLinear(
            in_features=D, out_features=D, num_param_tokens=N,
            param_block_size=64, base_blocks=4, max_remote_blocks=12,
            remote_capacity=12, top_p=0.3
        ).to(device)
        sparse.eval()
        sparse.refresh_param_cache()

        x = torch.randn(1, 1, D, device=device)

        d_ms = time_benchmark(lambda: dense(x))
        s_ms = time_benchmark(lambda: sparse(x))

        _, aux = sparse(x)
        act = aux["active_param_tokens_union"]
        sp_pct = (1.0 - act / N) * 100.0
        speedup = d_ms / s_ms

        print(f"{N:<8} | {d_ms:<14.3f} | {s_ms:<15.3f} | {act:<14} | {f'{sp_pct:.1f}%':<15} | {speedup:<8.2f}x")


def benchmark_flop_table():
    print("\n" + "=" * 86)
    print(" 3. ARITHMETIC FLOPs BREAKDOWN (Single Token M=1, K=2048, N=16384)")
    print("=" * 86)

    K = 2048
    N = 16384

    configs = [
        ("Dense GEMM", N, 0.0),
        ("Foveal BlockSparse (50%)", N // 2, 50.0),
        ("Foveal BlockSparse (75%)", N // 4, 75.0),
        ("Foveal BlockSparse (87.5%)", N // 8, 87.5),
    ]

    print(f"{'Variant':<28} | {'Active Channels':<16} | {'Compute MFLOPs':<15} | {'FLOP Reduction':<15}")
    print("-" * 86)
    for name, cols, red in configs:
        mflops = (2 * 1 * K * cols) / 1e6
        print(f"{name:<28} | {cols:<16} | {mflops:<15.2f} | {f'{red:.1f}%':<15}")
    print("=" * 86)


if __name__ == "__main__":
    benchmark_block_sparse_gemm()
    benchmark_tokenformer()
    benchmark_flop_table()
