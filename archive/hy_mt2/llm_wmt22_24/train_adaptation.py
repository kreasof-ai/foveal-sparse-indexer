"""
End-to-End Adaptation of Foveal Sparse Attention & MLP on WMT22-24.

Features:
- Sparsifies Attention (Chunk 8) & SwiGLU MLP (Chunk 32) while keeping LM Head 100% dense.
- Uses HybridMuonAdamW (Muon for broad 2D linear projections, AdamW for 16D routers & 1D params).
- Zero cold-start SVD initialization.
- Distillation loss (Next-token cross-entropy + Attention KL + MLP KL).
- Comprehensive SacreBLEU evaluation across zh-en, en-zh, en-de, en-ja.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import math
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from tqdm import tqdm

from foveal_indexer.optimizers import HybridMuonAdamW
from foveal_indexer.foveal_llm_layer import FovealSparseSwiGLU, FovealSparseAttentionLayer
from experiments.llm_wmt22_24.data_loader import load_wmt22_24_splits
from experiments.llm_wmt22_24.calibrate_svd import calibrate_model_svd


def evaluate_sacrebleu(
    model: nn.Module,
    tokenizer: Any,
    test_data: List[Dict[str, Any]],
    device: torch.device,
    max_samples: int = 100,
) -> Dict[str, float]:
    """Generates translations and computes SacreBLEU by language pair."""
    model.eval()
    by_lp_hyp: Dict[str, List[str]] = {}
    by_lp_ref: Dict[str, List[str]] = {}

    samples = test_data[:max_samples]
    print(f"\nEvaluating SacreBLEU on {len(samples)} held-out translation test samples...")

    for ex in tqdm(samples):
        lp = ex["lp"]
        if lp not in by_lp_hyp:
            by_lp_hyp[lp] = []
            by_lp_ref[lp] = []

        prompt = ex["prompt_text"]
        ref = ex["ref_text"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        by_lp_hyp[lp].append(pred_text)
        by_lp_ref[lp].append(ref)

    bleu_results: Dict[str, float] = {}
    all_hyp = []
    all_ref = []

    for lp in by_lp_hyp:
        hyp = by_lp_hyp[lp]
        ref = by_lp_ref[lp]
        all_hyp.extend(hyp)
        all_ref.extend(ref)

        # Chinese and Japanese use character tokenization, Western uses 13a
        tokenizer_type = "zh" if "zh" in lp or "ja" in lp else "13a"
        score = sacrebleu.corpus_bleu(hyp, [ref], tokenize=tokenizer_type).score
        bleu_results[lp] = score
        print(f"  BLEU ({lp}): {score:.2f}")

    # Overall corpus BLEU
    overall = sacrebleu.corpus_bleu(all_hyp, [all_ref], tokenize="13a").score
    bleu_results["overall"] = overall
    print(f"  Overall Corpus BLEU: {overall:.2f}\n")
    return bleu_results


def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, max_len: int = 256) -> Dict[str, torch.Tensor]:
    texts = [b["full_text"] for b in batch]
    toks = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # Mask out prompt tokens in labels so loss only calculates on target translation
    labels = toks["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Also mask prompt part
    for i, b in enumerate(batch):
        p_len = len(tokenizer(b["prompt_text"])["input_ids"])
        labels[i, :p_len] = -100

    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "labels": labels,
    }


def run_adaptation(
    model_id: str = "tencent/Hy-MT2-1.8B",
    num_layers: int = 32,        # ALL 32 layers uniformly sparsified
    chunk_size_attn: int = 8,
    active_blocks_attn: int = 4, # 4 blocks of 8 = 32 dims (25.0% compute)
    block_size_mlp: int = 32,
    active_blocks_mlp: int = 48, # 48 blocks of 32 = 1536 dims (25.0% compute)
    base_blocks_mlp: int = 0,    # 0 = pure uniform Top-K
    max_steps: int = 3000,
    batch_size: int = 4,         # Micro-batch size
    grad_accum_steps: int = 16,  # Global Batch Size = 4 * 16 = 64 (or 32 for 128)
    max_train: int = 100000,     # Near 100K training rows
    lr_muon: float = 0.01,
    lr_adamw: float = 2e-4,
    eval_interval: int = 250,
    eval_dense_samples: int = 60,
    max_eval_samples: int = 500,
    output_report: str = "experiments/llm_wmt22_24/ADAPTATION_REPORT.md",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_batch_size = batch_size * grad_accum_steps
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Training Config: Micro-batch {batch_size}, Grad Accum {grad_accum_steps} => GLOBAL BATCH SIZE {global_batch_size}")
    print(f"Total Steps: {max_steps} (~{max_steps * global_batch_size:,} translation sequences processed)")

    # 1. Load Tokenizer & Dataset (near 100K rows)
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data, val_data, test_data = load_wmt22_24_splits(
        tokenizer,
        val_per_lp=15,         # Lean validation set (non-blocking)
        test_per_lp=250,       # Large test set (~1,500-2,000 pairs)
        max_train=max_train,   # Near 100K training pairs
    )

    # 2. Load Dense Base Model
    print(f"Loading base model {model_id} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.config.use_cache = False

    # 3. Evaluate Dense Baseline SacreBLEU on representative sample
    print(f"\n--- Evaluating Dense Teacher Baseline BLEU ({eval_dense_samples} samples) ---")
    dense_bleu = evaluate_sacrebleu(model, tokenizer, test_data, device, max_samples=eval_dense_samples)

    # 4. Prepare Calibration Data
    calib_batches = []
    for i in range(0, 128, batch_size):
        b = train_data[i : i + batch_size]
        calib_batches.append(collate_fn(b, tokenizer))

    # 5. Apply Uniform Foveal Sparsification and Offline SVD Calibration
    total_layers = len(model.model.layers)
    if num_layers >= total_layers:
        layers_to_sparsify = list(range(total_layers))
    else:
        start_layer = (total_layers - num_layers) // 2
        layers_to_sparsify = list(range(start_layer, start_layer + num_layers))

    model = calibrate_model_svd(
        model=model,
        calib_tokens=calib_batches,
        device=device,
        block_size_mlp=block_size_mlp,
        active_blocks_mlp=active_blocks_mlp,
        base_blocks_mlp=base_blocks_mlp,
        chunk_size_attn=chunk_size_attn,
        active_blocks_attn=active_blocks_attn,
        layers_to_sparsify=layers_to_sparsify,
    )

    # 6. Setup Hybrid Optimizer (Muon + AdamW)
    print("\nSetting up HybridMuonAdamW Optimizer...")
    # Train only parameters in sparsified layers (routers, additive stream, projections)
    trainable_params = []
    for l_idx in layers_to_sparsify:
        for name, p in model.model.layers[l_idx].named_parameters():
            p.requires_grad = True
            trainable_params.append((f"layer_{l_idx}.{name}", p))

    # Freeze rest of the model (embedding, LM Head, unsparsified layers)
    for name, p in model.named_parameters():
        if not any(f"layers.{l_idx}." in name for l_idx in layers_to_sparsify):
            p.requires_grad = False

    trainable_count = sum(p.numel() for _, p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count / 1e6:.2f} M ({trainable_count/total_count*100:.2f}% of model)")

    optimizer = HybridMuonAdamW(
        named_params=trainable_params,
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
    )

    # 7. Training DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    data_iter = iter(train_loader)

    # 8. Adaptation Loop
    print(f"\n--- Starting Adaptation ({max_steps} steps, Grad Accum: {grad_accum_steps}) ---")
    model.train()
    start_time = time.perf_counter()
    running_loss = 0.0
    running_kl_mlp = 0.0
    running_kl_attn = 0.0

    step_history = []

    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        task_loss = outputs.loss

        # Collect KL distillation losses directly from sparsified layers
        kl_mlp_loss = torch.tensor(0.0, device=device)
        kl_attn_loss = torch.tensor(0.0, device=device)
        num_sp = len(layers_to_sparsify)

        for l_idx in layers_to_sparsify:
            layer = model.model.layers[l_idx]
            if hasattr(layer.mlp, "last_kl_loss") and layer.mlp.last_kl_loss is not None:
                kl_mlp_loss = kl_mlp_loss + layer.mlp.last_kl_loss
            if hasattr(layer.self_attn, "last_kl_loss") and layer.self_attn.last_kl_loss is not None:
                kl_attn_loss = kl_attn_loss + layer.self_attn.last_kl_loss

        total_loss = (task_loss + 0.1 * (kl_mlp_loss + kl_attn_loss) / num_sp) / grad_accum_steps
        total_loss.backward()

        running_loss += task_loss.item()
        running_kl_mlp += (kl_mlp_loss.item() / num_sp)

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 25 == 0:
            avg_loss = running_loss / 25
            avg_kl = running_kl_mlp / 25
            elapsed = time.perf_counter() - start_time
            tok_per_sec = (step * batch_size * 128) / elapsed
            print(f"Step {step:4d}/{max_steps} | Task Loss: {avg_loss:.4f} | Router Reg: {avg_kl:.4f} | Tok/s: {tok_per_sec:.1f}")
            running_loss = 0.0
            running_kl_mlp = 0.0

        if step % eval_interval == 0 or step == max_steps:
            # Evaluate validation perplexity
            model.eval()
            val_losses = []
            with torch.no_grad():
                for v_batch in [collate_fn(val_data[i:i+batch_size], tokenizer) for i in range(0, min(100, len(val_data)), batch_size)]:
                    v_in = v_batch["input_ids"].to(device)
                    v_mask = v_batch["attention_mask"].to(device)
                    v_lbl = v_batch["labels"].to(device)
                    v_loss = model(input_ids=v_in, attention_mask=v_mask, labels=v_lbl).loss
                    val_losses.append(v_loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"\n[Validation Step {step}] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}\n")
            step_history.append((step, val_loss, val_ppl))
            model.train()

    # 9. Final SacreBLEU Evaluation on Held-Out Test Set
    print("\n--- Evaluating Adapted Foveal Model SacreBLEU ---")
    sparse_bleu = evaluate_sacrebleu(model, tokenizer, test_data, device, max_samples=max_eval_samples)

    # 10. Generate Markdown Report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w") as f:
        f.write("# Foveal Sparsification Adaptation Report on WMT22-24\n\n")
        f.write(f"- **Base Model:** `{model_id}`\n")
        f.write(f"- **Sparsified Layers:** ALL {num_layers} of {total_layers} layers (100% Uniform)\n")
        f.write(f"- **Attention Configuration:** Chunk Size 8, {active_blocks_attn}/16 active blocks per head ({(active_blocks_attn/16)*100:.1f}% compute)\n")
        f.write(f"- **SwiGLU MLP Configuration:** Chunk Size 32, {active_blocks_mlp}/192 active blocks ({(active_blocks_mlp/192)*100:.1f}% compute)\n")
        f.write(f"- **LM Head:** 100% Dense and Untouched\n")
        f.write(f"- **Optimizer:** `HybridMuonAdamW` (Muon lr={lr_muon}, AdamW lr={lr_adamw})\n")
        f.write(f"- **Adaptation Steps:** {max_steps} steps\n\n")
        f.write("## SacreBLEU Score Comparison\n\n")
        f.write("| Language Pair | Dense Teacher BLEU | Foveal Sparse BLEU | BLEU Retention |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for lp in dense_bleu:
            if lp == "overall":
                continue
            d_b = dense_bleu.get(lp, 0.0)
            s_b = sparse_bleu.get(lp, 0.0)
            ret = (s_b / d_b * 100) if d_b > 0 else 0.0
            f.write(f"| **`{lp}`** | {d_b:.2f} | {s_b:.2f} | **{ret:.1f}%** |\n")
        
        d_tot = dense_bleu.get("overall", 0.0)
        s_tot = sparse_bleu.get("overall", 0.0)
        ret_tot = (s_tot / d_tot * 100) if d_tot > 0 else 0.0
        f.write(f"| **Overall** | **{d_tot:.2f}** | **{s_tot:.2f}** | **{ret_tot:.1f}%** |\n\n")

    print(f"\nReport written to {output_report}!")
    return dense_bleu, sparse_bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=32, help="Uniform: all 32 layers")
    parser.add_argument("--active_blocks_attn", type=int, default=4, help="Active blocks/head out of 16 (4 = 25% compute)")
    parser.add_argument("--active_blocks_mlp", type=int, default=48, help="Active blocks out of 192 (48 = 25% compute)")
    parser.add_argument("--base_blocks_mlp", type=int, default=0, help="Static base blocks (0 = pure uniform Top-K)")
    parser.add_argument("--max_steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per forward pass")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps (4 * 16 = 64 Global Batch Size, or 32 for 128)")
    parser.add_argument("--max_train", type=int, default=100000, help="Max training rows to load (~100K target)")
    parser.add_argument("--eval_interval", type=int, default=250, help="Validation interval in steps")
    parser.add_argument("--eval_dense_samples", type=int, default=60, help="Samples for initial baseline SacreBLEU")
    parser.add_argument("--max_eval_samples", type=int, default=500, help="Samples for final SacreBLEU on test set")
    parser.add_argument("--lr_muon", type=float, default=0.01)
    parser.add_argument("--lr_adamw", type=float, default=2e-4)
    parser.add_argument("--output_report", type=str, default="experiments/llm_wmt22_24/ADAPTATION_REPORT.md")
    args = parser.parse_args()

    run_adaptation(
        num_layers=args.num_layers,
        active_blocks_attn=args.active_blocks_attn,
        active_blocks_mlp=args.active_blocks_mlp,
        base_blocks_mlp=args.base_blocks_mlp,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_train=args.max_train,
        eval_interval=args.eval_interval,
        eval_dense_samples=args.eval_dense_samples,
        max_eval_samples=args.max_eval_samples,
        lr_muon=args.lr_muon,
        lr_adamw=args.lr_adamw,
        output_report=args.output_report,
    )
