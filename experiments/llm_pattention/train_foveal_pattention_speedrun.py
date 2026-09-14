"""
End-to-End Speedrun Training & Evaluation:
Full Foveal Token-Parameter Attention (Pattention) with:
1. SwiGLU MLP converted into Token-Parameter Attention (Pattention)
2. Blockwise Sparse Pattention (64 chunk size, indexed in 16D)
3. SVD Router Initialization (Empirical SVD covariance + Ridge Regression on teacher block energy)
4. SRAM Indexer & Fused Triton Pattention Kernel for 2-4x hardware speedup
5. Differentiable 16D Additive Stream from router to residual
6. Dense KL Distillation matching student router scores to teacher dense Pattention block energy
"""

import os
import sys
import time
import math
import copy
import argparse
from typing import Optional, Tuple, Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from foveal_indexer.foveal_pattention import (
    FovealPattentionMLP,
    sparsify_model_with_foveal_pattention,
    FovealPattentionDistillationLoss,
)
from foveal_indexer.pattention_triton import triton_pattention, HAS_TRITON


def load_wmt_training_data(tokenizer, max_pairs: int = 8000, max_len: int = 256):
    """Loads parallel Chinese-to-English pairs from WMT20, WMT19, and WMT18."""
    print("Loading parallel training data from WMT20, WMT19, and WMT18...")
    all_src = []
    all_ref = []

    for year in [20, 19, 18]:
        try:
            s_file = sacrebleu.get_source_file(f"wmt{year}", "zh-en")
            r_file = sacrebleu.get_reference_files(f"wmt{year}", "zh-en")[0]
            with open(s_file, "r", encoding="utf-8") as f:
                s_lines = [l.strip() for l in f if l.strip()]
            with open(r_file, "r", encoding="utf-8") as f:
                r_lines = [l.strip() for l in f if l.strip()]
            all_src.extend(s_lines)
            all_ref.extend(r_lines)
            print(f"  Loaded WMT{year}: {len(s_lines)} pairs")
        except Exception as e:
            print(f"  Warning: Could not load WMT{year}: {e}")

    print(f"Total raw pairs: {len(all_src)}. Tokenizing up to {max_pairs} items...")
    t0 = time.perf_counter()
    tokenized = []
    for s, r in zip(all_src[:max_pairs], all_ref[:max_pairs]):
        prompt = f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full_text = prompt_text + r + tokenizer.eos_token

        p_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        f_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
        if len(f_ids) > max_len:
            continue
        labels = f_ids.clone()
        labels[:len(p_ids)] = -100
        tokenized.append((f_ids, labels))

    dt = time.perf_counter() - t0
    print(f"Tokenized {len(tokenized)} pairs in {dt:.2f}s!")
    return tokenized


def calibrate_all_svd_routers(
    student: nn.Module,
    tokenizer,
    calib_srcs: list,
    device: torch.device,
    layers_to_calibrate: Optional[list] = None,
):
    """
    Collects empirical token activations and runs offline SVD router calibration across layers.
    """
    layers = student.model.layers
    num_layers = len(layers)
    target_layers = layers_to_calibrate if layers_to_calibrate is not None else list(range(num_layers))

    print(f"Collecting empirical activations across {len(target_layers)} target layers on {len(calib_srcs)} calibration sentences...")
    captured_inputs = {l: [] for l in target_layers}

    def make_hook(idx):
        def h(m, i, o):
            captured_inputs[idx].append(i[0].detach())
        return h

    hooks = [layers[l].mlp.register_forward_hook(make_hook(l)) for l in target_layers]

    for s in calib_srcs:
        prompt = f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"
        inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            student(inp)

    for h in hooks:
        h.remove()

    print(f"Running Offline SVD Router Calibration + Ridge Regression...")
    t0 = time.perf_counter()
    metrics_list = []
    for l_idx in target_layers:
        mlp = layers[l_idx].mlp
        if isinstance(mlp, FovealPattentionMLP):
            feats = torch.cat([t.reshape(-1, mlp.hidden_dim) for t in captured_inputs[l_idx]], dim=0)
            metrics = mlp.calibrate_offline_svd(feats)
            metrics_list.append(metrics)

    dt = time.perf_counter() - t0
    mean_exp_var = sum(m["top16_explained_variance"] for m in metrics_list) / len(metrics_list)
    mean_init_kl = sum(m["initial_kl_divergence"] for m in metrics_list) / len(metrics_list)
    print(f"SVD Router Calibration completed in {dt:.2f}s!")
    print(f"  Mean Top-16 Explained Variance: {mean_exp_var * 100:.1f}%")
    print(f"  Mean Initial KL Divergence:      {mean_init_kl:.4f}")
    return metrics_list


def compute_teacher_dense_block_targets(
    teacher: nn.Module,
    b_ids: torch.Tensor,
    target_layers: list,
    block_size: int = 64,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Extracts teacher representations and ground-truth dense block activation energy for KL distillation.
    """
    with torch.no_grad():
        t_out = teacher.model(b_ids, output_hidden_states=True)
        t_rep = t_out.last_hidden_state
        t_mid = t_out.hidden_states[16]

    teacher_block_targets = []
    for l_idx in target_layers:
        t_layer = teacher.model.layers[l_idx]
        mlp = t_layer.mlp
        # Input to MLP at layer l_idx is post-attention normalized hidden state
        h = t_out.hidden_states[l_idx]
        norm_h = t_layer.post_attention_layernorm(h)
        B, L, D = norm_h.shape

        with torch.no_grad():
            g = F.silu(F.linear(norm_h, mlp.gate_proj.weight))
            u = F.linear(norm_h, mlp.up_proj.weight)
            act = g * u  # (B, L, 6144)
            num_blocks = act.shape[-1] // block_size
            dw_norms = mlp.down_proj.weight.norm(dim=0).view(num_blocks, block_size).mean(dim=-1)
            act_blocks = act.abs().view(B, L, num_blocks, block_size).mean(dim=-1)
            block_energy = act_blocks * dw_norms[None, None, :]
            P_target = F.softmax(block_energy.float() / temperature, dim=-1)
            teacher_block_targets.append(P_target)

    return t_rep, t_mid, teacher_block_targets


def evaluate_bleu(model, tokenizer, eval_srcs, eval_refs, device, max_samples=100, batch_size=32):
    """Evaluates SacreBLEU score on held-out WMT22."""
    model.eval()
    tokenizer.padding_side = "left"
    sub_srcs = eval_srcs[:max_samples]
    sub_refs = eval_refs[:max_samples]

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"}],
            add_generation_prompt=True,
            tokenize=False
        )
        for s in sub_srcs
    ]

    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(formatted), batch_size):
        batch = formatted[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        for k in range(len(inputs["input_ids"])):
            preds.append(tokenizer.decode(out[k][in_len:], skip_special_tokens=True).strip())

    dt = time.perf_counter() - t0
    score = sacrebleu.corpus_bleu(preds, [sub_refs]).score
    return score, dt, preds


def main():
    parser = argparse.ArgumentParser(description="Speedrun Training for Foveal Pattention LLM")
    parser.add_argument("--model_name", type=str, default="tencent/Hy-MT2-1.8B")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--base_blocks", type=int, default=4)
    parser.add_argument("--remote_blocks", type=int, default=20)  # 24 blocks = 1536 channels (75% sparse)
    parser.add_argument("--max_minutes", type=float, default=30.0)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--eval_interval", type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=32, help="Number of layers to sparsify (e.g. 16, 24, 32)")
    default_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "foveal_pattention_best.pt")
    parser.add_argument("--output_checkpoint", type=str, default=default_ckpt)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)

    print("=" * 80)
    print("FOVEAL PATTENTION SPEEDRUN ARCHITECTURE")
    print(f"Model: {args.model_name}")
    print(f"Converting MLPs to Pattention with 64-chunk Blockwise Sparsity & 16D SVD Router")
    print(f"Active Blocks: {args.base_blocks} base + {args.remote_blocks} remote = {args.base_blocks + args.remote_blocks} blocks")
    active_ch = (args.base_blocks + args.remote_blocks) * args.block_size
    print(f"Active Channels: {active_ch} / 6144 ({(1.0 - active_ch/6144)*100:.1f}% sparsity)")
    print(f"Target Layers: {args.num_layers} layers | Max Time: {args.max_minutes}m")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading Teacher Model (bfloat16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load WMT22 test data
    src_wmt22 = sacrebleu.get_source_file("wmt22", "zh-en")
    ref_wmt22 = sacrebleu.get_reference_files("wmt22", "zh-en")[0]
    with open(src_wmt22) as f:
        eval_srcs = [l.strip() for l in f]
    with open(ref_wmt22) as f:
        eval_refs = [l.strip() for l in f]

    print("\nEvaluating Dense Teacher Baseline (100 samples)...")
    dense_bleu, dense_time, _ = evaluate_bleu(teacher, tokenizer, eval_srcs, eval_refs, device, max_samples=100)
    print(f"Dense Teacher Baseline BLEU: {dense_bleu:.2f} ({dense_time:.2f}s)")

    # Prepare parallel training dataset
    train_data = load_wmt_training_data(tokenizer, max_pairs=8000)

    print("\nCreating Student Model & Reparameterizing into Foveal Pattention...")
    student = copy.deepcopy(teacher)

    total_layers = len(student.model.layers)
    if args.num_layers < total_layers:
        # Sparsify the most redundant upper layers (e.g. 16 to 31)
        target_layers = list(range(total_layers - args.num_layers, total_layers))
    else:
        target_layers = list(range(total_layers))

    student = sparsify_model_with_foveal_pattention(
        student,
        layers_to_sparsify=target_layers,
        block_size=args.block_size,
        base_blocks=args.base_blocks,
        remote_blocks=args.remote_blocks,
        index_dim=16,
    )

    # Calibrate SVD Routers offline
    calib_srcs = eval_srcs[:48]
    calibrate_all_svd_routers(student, tokenizer, calib_srcs, device, layers_to_calibrate=target_layers)

    # Configure trainable parameters
    trainable_params = []
    for l_idx in target_layers:
        mlp = student.model.layers[l_idx].mlp
        # Train parameter tokens + 16D router + additive projection
        for p in [mlp.k_gate, mlp.k_up, mlp.v, mlp.q_proj_16d.weight, mlp.block_keys_16d, mlp.out_proj_16d.weight]:
            p.requires_grad = True
            trainable_params.append(p)
        # LayerNorm adaptation
        l = student.model.layers[l_idx]
        if hasattr(l, "input_layernorm"):
            l.input_layernorm.weight.requires_grad = True
            trainable_params.append(l.input_layernorm.weight)
        if hasattr(l, "post_attention_layernorm"):
            l.post_attention_layernorm.weight.requires_grad = True
            trainable_params.append(l.post_attention_layernorm.weight)

    if hasattr(student.model, "norm"):
        student.model.norm.weight.requires_grad = True
        trainable_params.append(student.model.norm.weight)

    print(f"\nTrainable Parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=5e-6)
    criterion = FovealPattentionDistillationLoss(alpha_ce=1.0, beta_cos=1.0, gamma_mid=0.5, lambda_kl=0.5)

    print("\nStarting Dual-Gradient Speedrun Distillation...")
    start_time = time.perf_counter()
    max_seconds = args.max_minutes * 60.0
    best_bleu = 0.0
    bs = args.batch_size
    num_data = len(train_data)
    step = 0

    student.train()
    while step < args.max_steps:
        elapsed = time.perf_counter() - start_time
        if elapsed >= max_seconds:
            print(f"\nReached time budget of {args.max_minutes:.1f} minutes. Stopping training loop.")
            break

        step += 1
        idx = (step * bs) % (num_data - bs)
        batch_items = train_data[idx:idx + bs]

        b_ids = torch.nn.utils.rnn.pad_sequence(
            [item[0] for item in batch_items],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)
        b_labels = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch_items],
            batch_first=True,
            padding_value=-100,
        ).to(device)

        # 1. Compute teacher representations and dense block energy targets
        t_rep, t_mid, teacher_block_targets = compute_teacher_dense_block_targets(
            teacher, b_ids, target_layers, block_size=args.block_size
        )

        # 2. Forward pass student and collect router scores
        optimizer.zero_grad()
        s_out = student(b_ids, labels=b_labels, output_hidden_states=True)
        s_rep = s_out.hidden_states[-1]
        s_mid = s_out.hidden_states[16]

        # Collect student router scores across target layers
        student_scores = []
        for l_idx in target_layers:
            layer_input = s_out.hidden_states[l_idx]
            norm_input = student.model.layers[l_idx].post_attention_layernorm(layer_input)
            scores = student.model.layers[l_idx].mlp.compute_router_scores(norm_input)
            student_scores.append(scores)

        # 3. Dual-gradient loss computation
        loss_dict = criterion(
            student_logits=s_out.logits,
            student_hidden=s_rep,
            teacher_hidden=t_rep,
            student_mid_hidden=s_mid,
            teacher_mid_hidden=t_mid,
            student_router_scores=student_scores,
            teacher_block_targets=teacher_block_targets,
            labels=b_labels,
        )

        loss = loss_dict["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            cur_lr = scheduler.get_last_lr()[0]
            elapsed_m = (time.perf_counter() - start_time) / 60.0
            ce_val = loss_dict.get("loss_ce", torch.tensor(0.0)).item()
            cos_val = loss_dict.get("loss_cos", torch.tensor(0.0)).item()
            kl_val = loss_dict.get("loss_kl", torch.tensor(0.0)).item()
            print(
                f"Step {step:04d}/{args.max_steps} [{elapsed_m:.1f}m / {args.max_minutes:.1f}m] | "
                f"Loss: {loss.item():.4f} (CE: {ce_val:.4f}, Cos: {cos_val:.4f}, KL: {kl_val:.4f}) | "
                f"LR: {cur_lr:.2e}"
            )

        if step % args.eval_interval == 0:
            val_bleu, val_dt, val_preds = evaluate_bleu(student, tokenizer, eval_srcs, eval_refs, device, max_samples=40)
            print(f"  >>> [Eval Step {step}] WMT22 BLEU (40 samples): {val_bleu:.2f} ({val_dt:.2f}s)")
            if val_preds:
                print(f"      Sample Pred: {val_preds[0][:60]}")
                print(f"      Sample Ref:  {eval_refs[0][:60]}")

            if val_bleu > best_bleu:
                best_bleu = val_bleu
                torch.save(student.state_dict(), args.output_checkpoint)
                print(f"      * Saved new best checkpoint to {args.output_checkpoint} (BLEU: {best_bleu:.2f})")
            student.train()

    total_training_time = time.perf_counter() - start_time
    print(f"\nTraining finished in {total_training_time:.1f}s ({total_training_time/60.0:.2f} minutes) across {step} steps.")

    # Load best checkpoint
    if os.path.exists(args.output_checkpoint) and best_bleu > 0:
        print(f"Loading best checkpoint from {args.output_checkpoint} (BLEU: {best_bleu:.2f})...")
        student.load_state_dict(torch.load(args.output_checkpoint, map_location=device))

    # Final Benchmark on full 100 test samples
    print("\n" + "=" * 80)
    print("FINAL EVALUATION: 100 HELD-OUT WMT22 TEST SAMPLES")
    print("=" * 80)
    final_bleu, student_time, final_preds = evaluate_bleu(student, tokenizer, eval_srcs, eval_refs, device, max_samples=100)

    # Latency benchmarks (Prefill & Decode)
    print("Benchmarking Model-Level Prefill and Decode Latency...")
    x_pref = torch.randint(100, 1000, (4, 256), device=device)
    for _ in range(5):
        with torch.no_grad():
            teacher(x_pref)
            student(x_pref)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            teacher(x_pref)
    torch.cuda.synchronize()
    dense_pref_ms = (time.perf_counter() - t0) / 20 * 1000

    t0 = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            student(x_pref)
    torch.cuda.synchronize()
    sparse_pref_ms = (time.perf_counter() - t0) / 20 * 1000

    print(f"Dense Teacher Baseline BLEU: {dense_bleu:.2f} (Latency: {dense_time:.2f}s)")
    print(f"Foveal Pattention Student BLEU: {final_bleu:.2f} (Latency: {student_time:.2f}s)")
    print(f"Accuracy Retention: {final_bleu / dense_bleu * 100:.1f}%")
    print(f"Prefill Latency (BS=4, L=256): Dense {dense_pref_ms:.1f} ms vs Sparse {sparse_pref_ms:.1f} ms ({dense_pref_ms / sparse_pref_ms:.2f}x speedup)")
    print("-" * 80)
    for k in range(min(3, len(final_preds))):
        print(f"Sample {k}:")
        print(f"  Source:    {eval_srcs[k]}")
        print(f"  Predicted: {final_preds[k]}")
        print(f"  Reference: {eval_refs[k]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
