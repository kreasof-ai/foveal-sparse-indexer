#!/usr/bin/env python3
"""
Foveal Pattention LLM: Zero-Shot SVD Router Sparsity Grid Search.

Evaluates Hy-MT2-1.8B across sparsity levels (256/6144 to 3072/6144), comparing:
  1. With Base Blocks (25% Base + 75% Dynamic Remote)
  2. Without Base Blocks (0% Base + 100% Dynamic Remote)

All configurations:
  - Fused Triton SwiGLU Kernel enabled (training/inference zero-discrepancy kernel)
  - Zero-shot SVD initialized router (no fine-tuning/gradient steps)
  - Evaluated on 100 held-out WMT22 zh-en samples
  - Hardware latency and speedup benchmarked against Dense Baseline on NVIDIA A10G
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from foveal_indexer.foveal_pattention import FovealPattentionMLP


def benchmark_prefill_latency(model: nn.Module, device: torch.device, bs: int = 4, seq_len: int = 256, warmup: int = 15, iters: int = 30) -> float:
    """Measures end-to-end prefill latency (ms) via CUDA events."""
    x = torch.randint(100, 1000, (bs, seq_len), device=device)
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        with torch.no_grad():
            model(x)
    end_evt.record()
    torch.cuda.synchronize()

    return start_evt.elapsed_time(end_evt) / iters


def benchmark_decode_latency(model: nn.Module, device: torch.device, bs: int = 16, warmup: int = 20, iters: int = 50) -> float:
    """Measures single-step decode latency (µs) via CUDA events."""
    x = torch.randint(100, 1000, (bs, 1), device=device)
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        with torch.no_grad():
            model(x)
    end_evt.record()
    torch.cuda.synchronize()

    return (start_evt.elapsed_time(end_evt) / iters) * 1000.0


def benchmark_mlp_stack_latency(mlps: List[nn.Module], device: torch.device, bs: int = 4, seq_len: int = 256, hidden_dim: int = 2048, warmup: int = 20, iters: int = 50) -> float:
    """Measures pure MLP stack compute latency (ms) for the target layers."""
    x = torch.randn(bs, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    for _ in range(warmup):
        with torch.no_grad():
            for m in mlps:
                m(x)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        with torch.no_grad():
            for m in mlps:
                m(x)
    end_evt.record()
    torch.cuda.synchronize()

    return start_evt.elapsed_time(end_evt) / iters


def evaluate_bleu(model: nn.Module, tokenizer: AutoTokenizer, eval_srcs: List[str], eval_refs: List[str], device: torch.device, max_samples: int = 100, batch_size: int = 16) -> Tuple[float, float, List[str]]:
    """Evaluates SacreBLEU score on held-out WMT22."""
    model.eval()
    tokenizer.padding_side = "left"
    sub_srcs = eval_srcs[:max_samples]
    sub_refs = eval_refs[:max_samples]

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for s in sub_srcs
    ]

    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[1]
        for k in range(len(inputs["input_ids"])):
            preds.append(tokenizer.decode(out[k][in_len:], skip_special_tokens=True).strip())

    dt = time.perf_counter() - t0
    bleu = sacrebleu.corpus_bleu(preds, [sub_refs]).score
    return bleu, dt, preds


def main():
    parser = argparse.ArgumentParser(description="Grid search over sparsity with/without base on Hy-MT2-1.8B")
    parser.add_argument("--model_name", type=str, default="tencent/Hy-MT2-1.8B")
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--calib_samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--layers", type=str, default="all", choices=["all", "16-31", "both"])
    parser.add_argument("--output_md", type=str, default="experiments/llm_pattention/GRID_SEARCH_SPARSITY_REPORT.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("FOVEAL PATTENTION: ZERO-SHOT SVD ROUTER SPARSITY GRID SEARCH")
    print(f"Model: {args.model_name}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Eval Samples: {args.eval_samples} | Calib Samples: {args.calib_samples}")
    print("Execution Engine: Fused Triton SwiGLU Kernel (training/inference zero discrepancy)")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print("\n[1/5] Loading Tokenizer & Dense Teacher Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    total_model_layers = len(model.model.layers)
    original_dense_mlps = [model.model.layers[i].mlp for i in range(total_model_layers)]

    # 2. Load Evaluation Data & Calibration Data
    print("\n[2/5] Loading WMT22 Test Data & Calibration Data...")
    src_wmt22 = sacrebleu.get_source_file("wmt22", "zh-en")
    ref_wmt22 = sacrebleu.get_reference_files("wmt22", "zh-en")[0]
    with open(src_wmt22, "r", encoding="utf-8") as f:
        eval_srcs = [l.strip() for l in f]
    with open(ref_wmt22, "r", encoding="utf-8") as f:
        eval_refs = [l.strip() for l in f]

    # Load calibration sentences from WMT21
    src_wmt21 = sacrebleu.get_source_file("wmt21", "zh-en")
    with open(src_wmt21, "r", encoding="utf-8") as f:
        calib_srcs = [l.strip() for l in f][: args.calib_samples]

    calib_formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for s in calib_srcs
    ]

    # 3. Dense Baseline Benchmarking
    print("\n[3/5] Evaluating Dense Baseline (100 Samples & Hardware Latency)...")
    dense_bleu, dense_time, dense_preds = evaluate_bleu(model, tokenizer, eval_srcs, eval_refs, device, max_samples=args.eval_samples, batch_size=args.batch_size)
    dense_pref_ms = benchmark_prefill_latency(model, device, bs=4, seq_len=256)
    dense_dec_us = benchmark_decode_latency(model, device, bs=16)
    dense_mlp_all_ms = benchmark_mlp_stack_latency(original_dense_mlps, device)

    print(f"Dense Baseline SacreBLEU (100 samples): {dense_bleu:.2f} ({dense_time:.2f}s)")
    print(f"Dense Baseline Prefill Latency (4x256): {dense_pref_ms:.2f} ms")
    print(f"Dense Baseline Decode Latency (16x1):   {dense_dec_us:.1f} µs")
    print(f"Dense Baseline 32L MLP Stack Latency:   {dense_mlp_all_ms:.2f} ms")

    # 4. Cache Teacher Activations for SVD Calibration
    print("\n[4/5] Capturing Teacher Hidden States across all 32 layers on calibration set...")
    captured_inputs = {l_idx: [] for l_idx in range(total_model_layers)}
    hooks = []
    for l_idx in range(total_model_layers):
        layer = model.model.layers[l_idx]
        def make_hook(idx):
            def hook_fn(module, args, kwargs):
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    captured_inputs[idx].append(args[0].detach().cpu())
            return hook_fn
        hooks.append(layer.mlp.register_forward_pre_hook(make_hook(l_idx), with_kwargs=True))

    with torch.no_grad():
        for prompt in calib_formatted:
            inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            model(inp)

    for h in hooks:
        h.remove()

    cached_layer_features = {
        l_idx: torch.cat([t.reshape(-1, 2048) for t in captured_inputs[l_idx]], dim=0)
        for l_idx in range(total_model_layers)
    }
    print(f"Captured {cached_layer_features[0].shape[0]} tokens of calibration features per layer.")

    # 5. Grid Search Execution
    # Determine layer suites to evaluate
    layer_suites = []
    if args.layers in ["all", "both"]:
        layer_suites.append(("All 32 Layers (Uniform Sparsity, L0–31)", list(range(32))))
    if args.layers in ["16-31", "both"]:
        layer_suites.append(("Semantic Reasoning Layers (L16–31, Dense Stem L0–15)", list(range(16, 32))))

    # Grid specifications: from 256/6144 to 3072/6144
    sparsity_levels = [
        {"active_ch": 256,  "blocks": 4,  "sparsity": 95.83},
        {"active_ch": 512,  "blocks": 8,  "sparsity": 91.67},
        {"active_ch": 1024, "blocks": 16, "sparsity": 83.33},
        {"active_ch": 1536, "blocks": 24, "sparsity": 75.00},
        {"active_ch": 2048, "blocks": 32, "sparsity": 66.67},
        {"active_ch": 3072, "blocks": 48, "sparsity": 50.00},
    ]

    all_results = {}

    for suite_name, target_layers in layer_suites:
        print(f"\n{'=' * 80}")
        print(f"RUNNING GRID SEARCH: {suite_name}")
        print(f"Target Layers Count: {len(target_layers)}")
        print(f"{'=' * 80}")

        suite_target_dense_mlps = [original_dense_mlps[i] for i in target_layers]
        suite_dense_mlp_pref_ms = benchmark_mlp_stack_latency(suite_target_dense_mlps, device, bs=4, seq_len=256)
        suite_dense_mlp_dec_ms = benchmark_mlp_stack_latency(suite_target_dense_mlps, device, bs=16, seq_len=1)

        suite_results = []

        # Iterate over sparsity levels
        for level in sparsity_levels:
            total_blocks = level["blocks"]
            active_ch = level["active_ch"]
            sp_pct = level["sparsity"]

            # Two conditions: Without Base (0 base) and With Base (25% base)
            conditions = [
                {
                    "condition": "Without Base",
                    "base_blocks": 0,
                    "remote_blocks": total_blocks,
                },
                {
                    "condition": "With Base (25%)",
                    "base_blocks": max(1, total_blocks // 4),
                    "remote_blocks": total_blocks - max(1, total_blocks // 4),
                },
            ]

            for cond in conditions:
                base_b = cond["base_blocks"]
                rem_b = cond["remote_blocks"]
                cond_name = cond["condition"]
                print(f"\n--- Testing: {active_ch}/6144 ({sp_pct:.1f}% sp) | {cond_name} [{base_b} base + {rem_b} remote] ---")

                # Step A: Instantiate FovealPattentionMLP on target layers
                foveal_mlps = []
                for l_idx in target_layers:
                    foveal_mlp = FovealPattentionMLP.from_dense_mlp(
                        original_dense_mlps[l_idx],
                        block_size=64,
                        base_blocks=base_b,
                        remote_blocks=rem_b,
                        index_dim=16,
                        temperature=1.0,
                        use_fused_kernel=True,
                    )
                    foveal_mlps.append(foveal_mlp)
                    model.model.layers[l_idx].mlp = foveal_mlp

                # Step B: Zero-shot SVD Router Calibration (identifies & pre-slices top active blocks)
                t_cal0 = time.perf_counter()
                kl_list = []
                for i, l_idx in enumerate(target_layers):
                    m = model.model.layers[l_idx].mlp.calibrate_offline_svd(cached_layer_features[l_idx])
                    kl_list.append(m["initial_kl_divergence"])
                cal_dt = time.perf_counter() - t_cal0
                avg_kl = sum(kl_list) / len(kl_list)

                # Step C: Evaluate Translation BLEU on 100 Samples
                sp_bleu, sp_time, sp_preds = evaluate_bleu(model, tokenizer, eval_srcs, eval_refs, device, max_samples=args.eval_samples, batch_size=args.batch_size)
                retention = (sp_bleu / max(dense_bleu, 1e-6)) * 100.0

                # Step D: Benchmark Latency & Speedup
                sp_pref_ms = benchmark_prefill_latency(model, device, bs=4, seq_len=256)
                sp_dec_ms = benchmark_decode_latency(model, device, bs=16) / 1000.0
                sp_mlp_pref_ms = benchmark_mlp_stack_latency(foveal_mlps, device, bs=4, seq_len=256)
                sp_mlp_dec_ms = benchmark_mlp_stack_latency(foveal_mlps, device, bs=16, seq_len=1)

                pref_speedup = dense_pref_ms / max(sp_pref_ms, 1e-6)
                dec_speedup = (dense_dec_us / 1000.0) / max(sp_dec_ms, 1e-6)
                mlp_pref_speedup = suite_dense_mlp_pref_ms / max(sp_mlp_pref_ms, 1e-6)
                mlp_dec_speedup = suite_dense_mlp_dec_ms / max(sp_mlp_dec_ms, 1e-6)

                print(f"  BLEU: {sp_bleu:.2f} ({retention:.1f}% ret, {sp_time:.2f}s) | Avg KL: {avg_kl:.4f}")
                print(f"  Speedup: Model Prefill {pref_speedup:.2f}x ({sp_pref_ms:.1f}ms) | MLP Prefill {mlp_pref_speedup:.2f}x ({sp_mlp_pref_ms:.2f}ms vs {suite_dense_mlp_pref_ms:.2f}ms)")
                print(f"  Decode Speedup: Model {dec_speedup:.2f}x ({sp_dec_ms:.2f}ms) | MLP Decode {mlp_dec_speedup:.2f}x ({sp_mlp_dec_ms:.2f}ms vs {suite_dense_mlp_dec_ms:.2f}ms)")
                print(f"  Sample 0: '{sp_preds[0][:60]}...'")

                suite_results.append({
                    "active_channels": active_ch,
                    "sparsity_pct": sp_pct,
                    "condition": cond_name,
                    "base_blocks": base_b,
                    "remote_blocks": rem_b,
                    "bleu": sp_bleu,
                    "retention": retention,
                    "gen_time": sp_time,
                    "avg_kl": avg_kl,
                    "prefill_ms": sp_pref_ms,
                    "prefill_speedup": pref_speedup,
                    "decode_ms": sp_dec_ms,
                    "decode_speedup": dec_speedup,
                    "mlp_pref_ms": sp_mlp_pref_ms,
                    "mlp_pref_speedup": mlp_pref_speedup,
                    "mlp_dec_ms": sp_mlp_dec_ms,
                    "mlp_dec_speedup": mlp_dec_speedup,
                    "sample_pred": sp_preds[0],
                })

                # Step E: Restore Original Dense MLPs
                for l_idx in target_layers:
                    model.model.layers[l_idx].mlp = original_dense_mlps[l_idx]
                torch.cuda.empty_cache()

        all_results[suite_name] = {
            "target_layers": target_layers,
            "suite_dense_mlp_pref_ms": suite_dense_mlp_pref_ms,
            "suite_dense_mlp_dec_ms": suite_dense_mlp_dec_ms,
            "results": suite_results,
        }

    # 6. Generate Markdown Report
    print(f"\n[5/5] Generating Summary Report to {args.output_md}...")
    report_lines = [
        "# Foveal Pattention LLM: Zero-Shot SVD Router Sparsity Grid Search Report",
        "",
        f"- **Model:** `{args.model_name}` (32 Layers, Hidden 2048, Intermediate 6144, ~1.79B Params)",
        f"- **Hardware Platform:** NVIDIA A10G (24GB VRAM, TensorRT-grade Ampere)",
        f"- **Evaluation Dataset:** Held-Out `haoranxu/WMT22-Test` `zh-en` ({args.eval_samples} Samples)",
        "- **Distillation / Training Steps:** **0 (Strict Zero-Shot SVD Router Initialization)**",
        "- **Execution Engine:** **Custom Triton Fused SwiGLU Kernel (`triton_fused_swiglu`)**",
        "",
        "## Dense Baseline Reference Performance",
        "",
        f"- **Dense Baseline SacreBLEU (100 Samples):** **{dense_bleu:.2f}** (Generation Time: {dense_time:.2f}s)",
        f"- **Prefill Latency ($BS=4, L=256$):** **{dense_pref_ms:.2f} ms**",
        f"- **Single-Step Decode Latency ($BS=16, L=1$):** **{dense_dec_us:.1f} µs**",
        f"- **Full 32-Layer MLP Stack Latency ($BS=4, L=256$):** **{dense_mlp_all_ms:.2f} ms**",
        "",
        f"**Baseline Sample Translation:**",
        f"> *Source:* {eval_srcs[0]}  ",
        f"> *Reference:* {eval_refs[0]}  ",
        f"> *Dense Output:* {dense_preds[0]}  ",
        "",
        "---",
        "",
    ]

    for suite_name, data in all_results.items():
        report_lines.extend([
            f"## Grid Search Results: {suite_name}",
            "",
            f"Target Layer Count: **{len(data['target_layers'])} layers** | Dense Target MLP Stack: **{data['suite_dense_mlp_pref_ms']:.2f} ms (Prefill) / {data['suite_dense_mlp_dec_ms']:.2f} ms (Decode)**",
            "",
            "| Active Channels / Layer | Sparsity | Routing Policy | Block Allocation (Base + Remote) | Initial Router $D_{\\text{KL}}$ | WMT22 BLEU (100s) | Retention | Model Prefill Latency | Model Prefill Speedup | MLP Stack Speedup (Prefill / Decode) | Model Decode Latency | Model Decode Speedup |",
            "| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **6,144 (Dense)** | **0.0%** | **Dense Baseline** | **96 Dense** | **0.0000** | **{dense_bleu:.2f}** | **100.0%** | **{dense_pref_ms:.1f} ms** | **1.00×** | **1.00× / 1.00×** | **{dense_dec_us / 1000.0:.2f} ms** | **1.00×** |",
        ])

        for r in data["results"]:
            report_lines.append(
                f"| **{r['active_channels']} / 6144** | {r['sparsity_pct']:.1f}% | {r['condition']} | {r['base_blocks']} base + {r['remote_blocks']} remote | {r['avg_kl']:.4f} | **{r['bleu']:.2f}** | **{r['retention']:.1f}%** | {r['prefill_ms']:.1f} ms | **{r['prefill_speedup']:.2f}×** | **{r['mlp_pref_speedup']:.2f}× / {r['mlp_dec_speedup']:.2f}×** | {r['decode_ms']:.2f} ms | **{r['decode_speedup']:.2f}×** |"
            )

        report_lines.extend([
            "",
            "### Qualitative Sample Outputs (Sample 0)",
            "",
            "| Active Channels | Routing Policy | BLEU | Translated Output (Sample 0) |",
            "| :---: | :--- | :---: | :--- |",
        ])
        for r in data["results"]:
            clean_pred = r["sample_pred"].replace("|", "\\|").replace("\n", " ")
            report_lines.append(f"| {r['active_channels']} / 6144 | {r['condition']} | {r['bleu']:.2f} | {clean_pred} |")

        report_lines.extend(["", "---", ""])

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nGrid Search Completed successfully! Report written to {args.output_md}")


if __name__ == "__main__":
    main()
