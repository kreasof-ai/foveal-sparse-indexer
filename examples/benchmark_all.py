"""Comprehensive Performance Measurement Benchmark on CPU.

Benchmarks:
1. Attention Decoding Scaling: Dense vs Foveal Sparse Attention (128 -> 8192 tokens).
2. Large Linear Projection: Dense vs Blockwise Sparse Linear (In=1024, Out=4096, 64x64 blocks).
3. Tokenformer Parameter Scaling: Dense Tokenformer vs Foveal Tokenformer (N_param=2048).
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from foveal_indexer.attention import FovealSparseAttention
from foveal_indexer.block_matmul import BlockSparseLinear
from foveal_indexer.tokenformer import FovealTokenformerLinear


def time_fn(fn, warmup=10, reps=50):
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms
    timings.sort()
    median = timings[len(timings) // 2]
    mean = sum(timings) / len(timings)
    min_t = timings[0]
    return median, min_t, mean


def benchmark_attention_decoding():
    print("=" * 78)
    print(" 1. ATTENTION DECODING LATENCY SCALING (CPU, Batch=1, hidden_size=256, heads=8)")
    print("=" * 78)

    device = torch.device("cpu")
    torch.manual_seed(42)

    hidden_size = 256
    num_heads = 8
    num_kv_heads = 4
    head_dim = 32
    page_size = 32
    local_window_blocks = 4  # 128 tokens local window
    max_remote_pages = 4     # at most 128 tokens remote
    # Active support cap = 128 + 128 = 256 tokens

    attn = FovealSparseAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        local_window_blocks=local_window_blocks,
        top_p=0.9,
        min_remote_pages=1,
        max_remote_pages=max_remote_pages,
    ).to(device)

    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    max_context = max(context_lengths)

    # Pre-populate cache
    cache = attn.init_kv_cache(batch_size=1, max_seq_len=max_context + 128, device=device)
    dummy_k = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
    dummy_v = torch.randn(1, num_kv_heads, 1, head_dim, device=device)
    dummy_ki = torch.randn(1, 1, 16, device=device)
    dummy_vi = torch.randn(1, 1, 16, device=device)

    for _ in range(max_context):
        cache.update(dummy_k, dummy_v, dummy_ki, dummy_vi)

    dense_k = torch.randn(1, num_heads, max_context, head_dim, device=device)
    dense_v = torch.randn(1, num_heads, max_context, head_dim, device=device)
    q_tok = torch.randn(1, num_heads, 1, head_dim, device=device)
    tok_in = torch.randn(1, 1, hidden_size, device=device)

    print(f"{'Context':<9} | {'Dense (ms)':<11} | {'Dense tok/s':<12} | {'Foveal (ms)':<11} | {'Foveal tok/s':<12} | {'Active Tokens':<13} | {'Speedup':<8}")
    print("-" * 78)

    for ctx in context_lengths:
        # Benchmark Dense Step
        def run_dense(c=ctx):
            _ = F.scaled_dot_product_attention(q_tok, dense_k[:, :, :c], dense_v[:, :, :c])

        dense_med, _, _ = time_fn(run_dense, warmup=10, reps=60)
        dense_tps = 1000.0 / dense_med

        # Benchmark Foveal Step
        cache.seq_len = ctx
        cache.completed_pages = ctx // page_size

        def run_foveal(c=ctx):
            cache.seq_len = c
            _ = attn.decode_step(tok_in, cache)

        foveal_med, _, _ = time_fn(run_foveal, warmup=10, reps=40)
        foveal_tps = 1000.0 / foveal_med

        # Reset cache position
        cache.seq_len = ctx
        _, aux = attn.decode_step(tok_in, cache)
        remote_cnt = len(aux["selected_remote_pages"][0])
        local_cnt = min(ctx, local_window_blocks * page_size)
        active_toks = local_cnt + remote_cnt * page_size

        speedup = dense_med / foveal_med
        print(f"{ctx:<9} | {dense_med:<11.4f} | {dense_tps:<12.1f} | {foveal_med:<11.4f} | {foveal_tps:<12.1f} | {active_toks:<13} | {speedup:<8.2f}x")


def benchmark_block_matmul():
    print("\n" + "=" * 78)
    print(" 2. BLOCKWISE SPARSE MATMUL: DENSE vs FOVEAL (In=1024, Out=4096, block=64)")
    print("=" * 78)

    device = torch.device("cpu")
    torch.manual_seed(42)

    in_features = 1024
    out_features = 4096
    block_size = 64
    total_blocks = out_features // block_size  # 64 blocks

    dense_linear = torch.nn.Linear(in_features, out_features, bias=False).to(device)

    # Config 1: 75% Sparsity (16 active blocks out of 64)
    sparse_linear_75 = BlockSparseLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        base_blocks=4,
        max_remote_blocks=12,
        top_p=0.5,
    ).to(device)

    # Config 2: 87.5% Sparsity (8 active blocks out of 64)
    sparse_linear_87 = BlockSparseLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        base_blocks=2,
        max_remote_blocks=6,
        top_p=0.3,
    ).to(device)

    # Input: single token (inference batch=1, seq=1)
    x = torch.randn(1, 1, in_features, device=device)

    # Dense Linear
    dense_med, _, _ = time_fn(lambda: dense_linear(x), warmup=20, reps=100)
    dense_flops = 2 * in_features * out_features / 1e6  # MFLOPs

    # Sparse Linear (75% sparse)
    sparse_75_med, _, _ = time_fn(lambda: sparse_linear_75(x), warmup=20, reps=100)
    _, aux_75 = sparse_linear_75(x)
    active_chan_75 = aux_75["active_blocks_union"] * block_size
    sparse_75_flops = 2 * in_features * active_chan_75 / 1e6

    # Sparse Linear (87.5% sparse)
    sparse_87_med, _, _ = time_fn(lambda: sparse_linear_87(x), warmup=20, reps=100)
    _, aux_87 = sparse_linear_87(x)
    active_chan_87 = aux_87["active_blocks_union"] * block_size
    sparse_87_flops = 2 * in_features * active_chan_87 / 1e6

    print(f"{'Variant':<26} | {'Active Channels':<16} | {'Compute MFLOPs':<15} | {'FLOP Savings':<13} | {'Latency (ms)':<12}")
    print("-" * 78)
    sp_str_75 = f"{aux_75['sparsity']*100:.1f}%"
    sp_str_87 = f"{aux_87['sparsity']*100:.1f}%"
    print(f"{'Dense nn.Linear':<26} | {out_features:<16} | {dense_flops:<15.2f} | {'0.0%':<13} | {dense_med:<12.4f}")
    print(f"{'Foveal BlockSparse (75%)':<26} | {active_chan_75:<16} | {sparse_75_flops:<15.2f} | {sp_str_75:<13} | {sparse_75_med:<12.4f}")
    print(f"{'Foveal BlockSparse (87.5%)':<26} | {active_chan_87:<16} | {sparse_87_flops:<15.2f} | {sp_str_87:<13} | {sparse_87_med:<12.4f}")


def benchmark_tokenformer():
    print("\n" + "=" * 78)
    print(" 3. FOVEAL TOKENFORMER (N_param=2048, block=64, in_features=256, out_features=256)")
    print("=" * 78)

    device = torch.device("cpu")
    torch.manual_seed(42)

    in_features = 256
    out_features = 256
    num_param_tokens = 2048
    param_block_size = 64
    total_blocks = num_param_tokens // param_block_size  # 32 blocks

    # Dense Tokenformer (all 32 blocks active)
    dense_tf = FovealTokenformerLinear(
        in_features=in_features,
        out_features=out_features,
        num_param_tokens=num_param_tokens,
        param_block_size=param_block_size,
        base_blocks=total_blocks,
        max_remote_blocks=0,
    ).to(device)

    # Sparse Tokenformer (4 base blocks + 4 remote blocks = 8 blocks = 512 tokens -> 75% parameter sparsity)
    sparse_tf = FovealTokenformerLinear(
        in_features=in_features,
        out_features=out_features,
        num_param_tokens=num_param_tokens,
        param_block_size=param_block_size,
        base_blocks=4,
        max_remote_blocks=4,
        top_p=0.5,
    ).to(device)

    x = torch.randn(1, 1, in_features, device=device)

    dense_med, _, _ = time_fn(lambda: dense_tf(x), warmup=20, reps=80)
    sparse_med, _, _ = time_fn(lambda: sparse_tf(x), warmup=20, reps=80)

    _, aux_sparse = sparse_tf(x)
    active_tokens = aux_sparse["active_param_tokens_union"]
    sparsity = (1.0 - active_tokens / num_param_tokens) * 100.0

    print(f"{'Variant':<24} | {'Active Tokens':<15} | {'Parameter Sparsity':<19} | {'Latency (ms)':<12}")
    print("-" * 78)
    print(f"{'Dense Tokenformer':<24} | {num_param_tokens:<15} | {'0.0%':<19} | {dense_med:<12.4f}")
    print(f"{'Foveal Tokenformer':<24} | {active_tokens:<15} | {f'{sparsity:.1f}%':<19} | {sparse_med:<12.4f}")
    print("=" * 78)


if __name__ == "__main__":
    benchmark_attention_decoding()
    benchmark_block_matmul()
    benchmark_tokenformer()
