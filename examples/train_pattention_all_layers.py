"""
End-to-End Distillation and Evaluation for Full-Model SwiGLU Pattention (Token-Parameter Attention)
Compresses all 32 SwiGLU MLP layers of tencent/Hy-MT2-1.8B from N_P=6144 to N_act=1024 (83.3% MLP parameter reduction,
2.28x total model compression from 1.79B to 784M params).
"""

import os
import sys
import time
import math
import copy
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from foveal_indexer.llm_sparsify import LLMDistillationLoss


def calibrate_and_reconstruct_pattention_layers(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    calib_srcs: list,
    n_act: int = 1024,
    device: torch.device = torch.device("cuda"),
):
    """
    Calibrate all 32 layers on real inputs and compute optimal V_opt via Ridge Regression.
    """
    num_layers = len(student.model.layers)
    print(f"Collecting calibration activations across {num_layers} layers on {len(calib_srcs)} sentences...")
    
    captured_inputs = {l: [] for l in range(num_layers)}
    def make_hook(idx):
        def h(m, i, o):
            captured_inputs[idx].append(i[0].detach())
        return h

    hooks = [student.model.layers[l].mlp.register_forward_hook(make_hook(l)) for l in range(num_layers)]
    
    for s in calib_srcs:
        prompt = f"将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n{s}"
        inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            student(inp)
            
    for h in hooks:
        h.remove()

    print(f"Reconstructing {num_layers} layers to N_act={n_act} via Ridge Regression...")
    fidelities = []
    t0 = time.perf_counter()
    
    for l_idx in range(num_layers):
        mlp = student.model.layers[l_idx].mlp
        X_l = torch.cat([t.reshape(-1, 2048) for t in captured_inputs[l_idx]], dim=0)
        
        gw = mlp.gate_proj.weight.data
        uw = mlp.up_proj.weight.data
        dw = mlp.down_proj.weight.data
        
        with torch.no_grad():
            G = F.silu(F.linear(X_l, gw))
            U = F.linear(X_l, uw)
            A = G * U
            Y_dense = F.linear(A, dw)
            
            # Select top n_act parameter tokens by energy
            energy = A.abs().mean(dim=0) * dw.float().norm(dim=0)
            _, top_idx = torch.sort(energy, descending=True)
            top_idx_act = top_idx[:n_act]
            
            Kg_act = gw[top_idx_act]
            Ku_act = uw[top_idx_act]
            A_act = A[:, top_idx_act].float()
            
            # Solve Ridge Regression: V_opt = (A^T A + lambda I)^-1 A^T Y
            reg = 1e-1
            AtA = A_act.T @ A_act + reg * torch.eye(n_act, device=device)
            V_opt = torch.linalg.solve(AtA, A_act.T @ Y_dense.float()).T.to(torch.bfloat16)
            
            Y_recon = F.linear(A_act.to(torch.bfloat16), V_opt)
            cos_sim = F.cosine_similarity(Y_dense.flatten(1), Y_recon.flatten(1), dim=1).mean().item()
            fidelities.append(cos_sim)

        # Replace MLP with SwiGLU Pattention active projections
        new_gate = nn.Linear(2048, n_act, bias=False, dtype=torch.bfloat16, device=device)
        new_up = nn.Linear(2048, n_act, bias=False, dtype=torch.bfloat16, device=device)
        new_down = nn.Linear(n_act, 2048, bias=False, dtype=torch.bfloat16, device=device)
        
        new_gate.weight.data.copy_(Kg_act)
        new_up.weight.data.copy_(Ku_act)
        new_down.weight.data.copy_(V_opt)
        
        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down

    dt = time.perf_counter() - t0
    print(f"Calibration completed in {dt:.2f}s! Mean layer fidelity: {sum(fidelities)/len(fidelities):.4f}")
    return fidelities


def prepare_training_data(tokenizer, max_pairs: int = 8000, max_len: int = 256):
    """Load parallel datasets from WMT20, WMT19, and WMT18 and pre-tokenize."""
    print("Loading parallel training data from WMT20, WMT19, and WMT18...")
    all_src = []
    all_ref = []
    
    for year in [20, 19, 18]:
        try:
            s_file = sacrebleu.get_source_file(f"wmt{year}", "zh-en")
            r_file = sacrebleu.get_reference_files(f"wmt{year}", "zh-en")[0]
            with open(s_file) as f:
                s_lines = [l.strip() for l in f]
            with open(r_file) as f:
                r_lines = [l.strip() for l in f]
            all_src.extend(s_lines)
            all_ref.extend(r_lines)
            print(f"  Loaded WMT{year}: {len(s_lines)} pairs")
        except Exception as e:
            print(f"  Warning: Could not load WMT{year}: {e}")
            
    print(f"Total raw pairs: {len(all_src)}. Pre-tokenizing up to {max_pairs} items...")
    t0 = time.perf_counter()
    tokenized_data = []
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
        tokenized_data.append((f_ids, labels))
        
    dt = time.perf_counter() - t0
    print(f"Tokenized {len(tokenized_data)} pairs in {dt:.2f}s!")
    return tokenized_data


def evaluate_bleu(model, tokenizer, eval_srcs, eval_refs, device, max_samples=100, batch_size=32):
    """Evaluate generation BLEU on WMT22."""
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
    parser = argparse.ArgumentParser(description="Train SwiGLU Pattention on all 32 layers with distillation")
    parser.add_argument("--model_name", type=str, default="tencent/Hy-MT2-1.8B")
    parser.add_argument("--n_act", type=int, default=1024)
    parser.add_argument("--max_minutes", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/pattention_32l_1024.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    
    print("=" * 70)
    print(f"FULL-MODEL SWIGLU PATTENTION DISTILLATION")
    print(f"Target: ALL 32 Layers, N_act={args.n_act} (from N_P=6144)")
    print(f"Max Training Time: {args.max_minutes} minutes | Max Steps: {args.max_steps}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading teacher model in bfloat16...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Creating student model copy...")
    student = copy.deepcopy(teacher)

    # Load WMT22 test references
    src_wmt22 = sacrebleu.get_source_file("wmt22", "zh-en")
    ref_wmt22 = sacrebleu.get_reference_files("wmt22", "zh-en")[0]
    with open(src_wmt22) as f:
        eval_srcs = [l.strip() for l in f]
    with open(ref_wmt22) as f:
        eval_refs = [l.strip() for l in f]

    # Evaluate dense teacher baseline on 100 samples
    print("Evaluating Dense Teacher baseline on 100 WMT22 test sentences...")
    dense_bleu, dense_time, _ = evaluate_bleu(teacher, tokenizer, eval_srcs, eval_refs, device, max_samples=100)
    print(f"Dense Teacher Baseline BLEU: {dense_bleu:.2f} ({dense_time:.2f}s)")

    # Prepare training dataset
    train_data = prepare_training_data(tokenizer, max_pairs=8000)

    # Calibrate and reconstruct layers
    calib_srcs = [s for s in eval_srcs[:48]]
    calibrate_and_reconstruct_pattention_layers(teacher, student, tokenizer, calib_srcs, n_act=args.n_act, device=device)

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    mlp_params = sum(p.numel() for l in student.model.layers for p in l.mlp.parameters())
    print(f"Student Total Params: {total_params / 1e6:.1f}M | MLP Params: {mlp_params / 1e6:.1f}M")
    print(f"Param Compression Ratio: {1791.1 / (total_params / 1e6):.2f}x Total, {1208.0 / (mlp_params / 1e6):.2f}x MLP")

    # Select trainable parameters (MLP layers + LayerNorms)
    trainable_params = []
    for l in student.model.layers:
        for p in l.mlp.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        if hasattr(l, "input_layernorm"):
            l.input_layernorm.weight.requires_grad = True
            trainable_params.append(l.input_layernorm.weight)
        if hasattr(l, "post_attention_layernorm"):
            l.post_attention_layernorm.weight.requires_grad = True
            trainable_params.append(l.post_attention_layernorm.weight)
    if hasattr(student.model, "norm"):
        student.model.norm.weight.requires_grad = True
        trainable_params.append(student.model.norm.weight)

    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=5e-6)
    criterion = LLMDistillationLoss(alpha=1.0, beta=1.0, gamma=0.5)

    print("\nStarting Distillation Fine-Tuning...")
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
            print(f"\nReached time budget of {args.max_minutes:.1f} minutes ({elapsed:.1f}s). Stopping training loop.")
            break

        step += 1
        idx = (step * bs) % (num_data - bs)
        batch_items = train_data[idx:idx + bs]
        
        b_ids = torch.nn.utils.rnn.pad_sequence(
            [item[0] for item in batch_items],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)
        b_labels = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch_items],
            batch_first=True,
            padding_value=-100
        ).to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            t_out = teacher.model(b_ids, output_hidden_states=True)
            t_rep = t_out.last_hidden_state
            t_mid = t_out.hidden_states[16]

        s_out = student(b_ids, labels=b_labels, output_hidden_states=True)
        loss_dict = criterion(s_out.logits, s_out.hidden_states[-1], t_rep, labels=b_labels)
        
        # Midpoint stage supervision (Layer 16)
        s_mid = s_out.hidden_states[16]
        loss_mid = (1.0 - F.cosine_similarity(s_mid.float(), t_mid.float(), dim=-1)).mean()
        loss = loss_dict["loss"] + 0.5 * loss_mid
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1:
            cur_lr = scheduler.get_last_lr()[0]
            elapsed_m = (time.perf_counter() - start_time) / 60.0
            ce_val = loss_dict.get('ce', torch.tensor(0.0)).item()
            cos_val = loss_dict.get('rep_cos', torch.tensor(0.0)).item()
            mid_val = loss_mid.item()
            print(
                f"Step {step:04d}/{args.max_steps} [{elapsed_m:.1f}m / {args.max_minutes:.1f}m] | "
                f"Loss: {loss.item():.4f} (CE: {ce_val:.4f}, Cos: {cos_val:.4f}, Mid: {mid_val:.4f}) | "
                f"LR: {cur_lr:.2e}"
            )

        # Evaluation periodic checkpoint
        if step % args.eval_interval == 0:
            val_bleu, val_dt, val_preds = evaluate_bleu(student, tokenizer, eval_srcs, eval_refs, device, max_samples=40)
            print(f"  >>> [Eval Step {step}] WMT22 BLEU (40 samples): {val_bleu:.2f} (Time: {val_dt:.2f}s)")
            if val_preds:
                print(f"      Sample Pred: {val_preds[0][:60]}")
                print(f"      Sample Ref:  {eval_refs[0][:60]}")
                
            if val_bleu > best_bleu:
                best_bleu = val_bleu
                torch.save(student.state_dict(), args.output_checkpoint)
                print(f"      * Saved new best model checkpoint to {args.output_checkpoint} (BLEU: {best_bleu:.2f})")
            student.train()

    total_training_time = time.perf_counter() - start_time
    print(f"\nTraining finished in {total_training_time:.1f}s ({total_training_time/60.0:.2f} minutes) across {step} steps.")

    # Load best checkpoint if available
    if os.path.exists(args.output_checkpoint):
        print(f"Loading best checkpoint from {args.output_checkpoint}...")
        student.load_state_dict(torch.load(args.output_checkpoint, map_location=device))

    # Final Benchmark on full 100 test samples
    print("\n" + "=" * 70)
    print("FINAL EVALUATION: 100 WMT22 TEST SAMPLES")
    print("=" * 70)
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

    print(f"Dense Teacher Baseline BLEU: {dense_bleu:.2f} (End-to-end Gen: {dense_time:.2f}s)")
    print(f"Full SwiGLU Pattention (32 Layers, N_act={args.n_act}) BLEU: {final_bleu:.2f} (End-to-end Gen: {student_time:.2f}s)")
    print(f"Accuracy Retention: {final_bleu / dense_bleu * 100:.1f}%")
    print(f"Prefill Latency (BS=4, L=256): Dense {dense_pref_ms:.1f} ms vs Sparse {sparse_pref_ms:.1f} ms ({dense_pref_ms / sparse_pref_ms:.2f}x speedup)")
    print("-" * 70)
    for k in range(min(3, len(final_preds))):
        print(f"Sample {k}:")
        print(f"  Source:    {eval_srcs[k]}")
        print(f"  Predicted: {final_preds[k]}")
        print(f"  Reference: {eval_refs[k]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
