"""Benchmark Peak Memory Scaling: Static Parameters vs Constant Active Parameter Size on NVIDIA A10G.

Compares Dense Tokenformer vs Foveal Sparse Tokenformer as static parameters scale
from 2,048 to 524,288 parameter tokens while active parameter size remains CONSTANT (512 tokens).
"""

import os
import sys
import gc
import math
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.cifar10.train_cifar10_faster_and_better import DenseTokenformerBlock, FovealTokenformerBlock


def profile_memory_scaling(
    device: torch.device,
    B: int = 16,
    N: int = 512,
    D: int = 768,
    active_params: int = 512,
    heads: Optional[int] = None,
    param_counts: List[int] = None,
) -> Dict[str, Any]:
    if heads is None:
        heads = max(1, D // 64)
        while heads > 1 and D % heads != 0:
            heads -= 1
    if param_counts is None:
        param_counts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    gpu_name = torch.cuda.get_device_name(0)
    total_tokens = B * N

    print("=" * 105)
    print(f"  PEAK MEMORY SCALING: STATIC PARAMETERS VS CONSTANT ACTIVE PARAMETERS ON {gpu_name}")
    print(f"  Configuration: Batch={B}, SeqLen={N} (Total Tokens={total_tokens}), Hidden D={D}, Active Budget={active_params}")
    print("=" * 105)

    headers = [
        ("Static Params (N_p)", 20),
        ("Sparsity", 10),
        ("Static (MB)", 12),
        ("Dense Inf", 14),
        ("Foveal Inf", 14),
        ("Dense Train", 16),
        ("Foveal Train", 16),
        ("Train Mem Savings", 18),
    ]
    header_str = " | ".join(f"{h[0]:<{h[1]}}" for h in headers)
    print(header_str)
    print("-" * len(header_str))

    results = []

    for N_param in param_counts:
        sparsity = 100.0 * (1.0 - active_params / N_param)

        # Baseline: Static weight memory
        # 2 * N_param * D * 2 bytes (FP16)
        static_weight_mb = (2 * N_param * D * 2) / (1024 ** 2)

        # ----------------- 1. Dense Tokenformer -----------------
        dense_inf_mb = None
        dense_tr_mb = None
        dense_act_mb = None
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            dense = DenseTokenformerBlock(D=D, heads=heads, num_params=N_param).to(device).half()
            static_dense = torch.cuda.memory_allocated() / (1024 ** 2)

            # Inference
            x_inf = torch.randn(B, N, D, device=device, dtype=torch.float16)
            with torch.no_grad():
                _ = dense(x_inf)
            dense_inf_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            dense_act_mb = dense_inf_mb - static_dense
            del x_inf

            # Training (Forward + Backward)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            x_tr = torch.randn(B, N, D, device=device, dtype=torch.float16, requires_grad=True)
            out = dense(x_tr)
            loss = out.sum()
            loss.backward()
            dense_tr_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            del dense, x_tr, out, loss
        except torch.cuda.OutOfMemoryError:
            dense_inf_mb = "OOM"
            dense_tr_mb = "OOM"
            dense_act_mb = "OOM"
            torch.cuda.empty_cache()
            gc.collect()

        # ----------------- 2. Foveal Sparse Tokenformer -----------------
        foveal_inf_mb = None
        foveal_tr_mb = None
        foveal_act_mb = None
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            foveal = FovealTokenformerBlock(
                D=D, heads=heads, num_params=N_param, active_params=active_params
            ).to(device).half()
            static_foveal = torch.cuda.memory_allocated() / (1024 ** 2)

            # Inference
            x_inf = torch.randn(B, N, D, device=device, dtype=torch.float16)
            with torch.no_grad():
                _ = foveal(x_inf)
            foveal_inf_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            foveal_act_mb = foveal_inf_mb - static_foveal
            del x_inf

            # Training (Forward + Backward)
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            x_tr = torch.randn(B, N, D, device=device, dtype=torch.float16, requires_grad=True)
            out = foveal(x_tr)
            loss = out.sum()
            loss.backward()
            foveal_tr_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            del foveal, x_tr, out, loss
        except torch.cuda.OutOfMemoryError:
            foveal_inf_mb = "OOM"
            foveal_tr_mb = "OOM"
            foveal_act_mb = "OOM"
            torch.cuda.empty_cache()
            gc.collect()

        # Formatting
        d_inf_s = f"{dense_inf_mb:.1f} MB" if isinstance(dense_inf_mb, float) else dense_inf_mb
        f_inf_s = f"{foveal_inf_mb:.1f} MB" if isinstance(foveal_inf_mb, float) else foveal_inf_mb
        d_tr_s = f"{dense_tr_mb:.1f} MB" if isinstance(dense_tr_mb, float) else dense_tr_mb
        f_tr_s = f"{foveal_tr_mb:.1f} MB" if isinstance(foveal_tr_mb, float) else foveal_tr_mb

        if isinstance(dense_tr_mb, float) and isinstance(foveal_tr_mb, float):
            savings_ratio = f"{dense_tr_mb / foveal_tr_mb:.2f}x less"
        elif dense_tr_mb == "OOM" and isinstance(foveal_tr_mb, float):
            savings_ratio = "> 1.6x (Dense OOM)"
        else:
            savings_ratio = "N/A"

        row = [
            f"{N_param:<20}",
            f"{sparsity:<8.1f} %",
            f"{static_weight_mb:<10.1f} MB",
            f"{d_inf_s:<14}",
            f"{f_inf_s:<14}",
            f"{d_tr_s:<16}",
            f"{f_tr_s:<16}",
            f"{savings_ratio:<18}",
        ]
        print(" | ".join(row))

        results.append({
            "N_param": N_param,
            "sparsity": sparsity,
            "static_weight_mb": static_weight_mb,
            "dense_inf_mb": dense_inf_mb,
            "foveal_inf_mb": foveal_inf_mb,
            "dense_tr_mb": dense_tr_mb,
            "foveal_tr_mb": foveal_tr_mb,
            "dense_act_mb": dense_act_mb,
            "foveal_act_mb": foveal_act_mb,
        })

    print("\n" + "=" * 105)
    print("  NET ACTIVATION MEMORY BREAKDOWN (EXCLUDING STATIC MODEL WEIGHTS)")
    print("=" * 105)
    act_headers = [
        ("Static Params (N_p)", 20),
        ("Sparsity", 10),
        ("Dense Inf Act", 16),
        ("Foveal Inf Act", 16),
        ("Act Memory Ratio", 18),
    ]
    print(" | ".join(f"{h[0]:<{h[1]}}" for h in act_headers))
    print("-" * 88)
    for r in results:
        d_act = f"{r['dense_act_mb']:.1f} MB" if isinstance(r['dense_act_mb'], float) else str(r['dense_act_mb'])
        f_act = f"{r['foveal_act_mb']:.1f} MB" if isinstance(r['foveal_act_mb'], float) else str(r['foveal_act_mb'])
        if isinstance(r['dense_act_mb'], float) and isinstance(r['foveal_act_mb'], float):
            ratio = f"{r['dense_act_mb'] / r['foveal_act_mb']:.2f}x less"
        elif r['dense_act_mb'] == "OOM":
            ratio = "Dense OOM"
        else:
            ratio = "N/A"
        row = [
            f"{r['N_param']:<20}",
            f"{r['sparsity']:<8.1f} %",
            f"{d_act:<16}",
            f"{f_act:<16}",
            f"{ratio:<18}",
        ]
        print(" | ".join(row))

    return {
        "config": {"B": B, "N": N, "D": D, "active_params": active_params},
        "results": results,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_memory_scaling(device)
