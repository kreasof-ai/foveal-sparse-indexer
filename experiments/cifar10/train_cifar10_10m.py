"""Full CIFAR-10 Training Comparison: Dense ViT (~10M params) vs Foveal Sparse ViT (>90% Sparsity).

Architecture (~10-11M parameters):
- Input: 32x32 RGB (CIFAR-10, 10 classes)
- Patches: 1x1 (1024 patches total) or 2x2
- Embed Dim: D = 384
- Depth: 6 Transformer Blocks
- Attention Heads: 6 (head_dim = 64)
- MLP Expansion: 384 -> 1536 (4x expansion)
- Total Parameters: ~11.0M (Dense) / ~11.5M (Foveal)

Sparsity Configuration (>90% Sparse Ratio):
- Attention: 1024 tokens = 64 blocks of 16.
  Active blocks: 1 base self-block + up to 2 remote blocks = 3 / 64 blocks active.
  Attention Sparsity = 95.3% (>90% pruned)!
- MLP FFN: 1536 channels = 24 blocks of 64.
  Active blocks: 1 base block + up to 1 remote block = 2 / 24 blocks active.
  MLP Channel Sparsity = 91.7% (>90% pruned)!
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Tuple, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from foveal_indexer.vit import SmallViT


def get_cifar10_loaders(
    batch_size: int = 64,
    train_samples: int = 50000,
    test_samples: int = 10000,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """Loads CIFAR-10 with standard augmentation and normalization."""
    os.makedirs(data_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    cifar_folder = os.path.join(data_dir, "cifar10")
    if os.path.exists(os.path.join(cifar_folder, "train")):
        train_ds = datasets.ImageFolder(os.path.join(cifar_folder, "train"), transform=transform_train)
        test_ds = datasets.ImageFolder(os.path.join(cifar_folder, "test"), transform=transform_test)
    else:
        train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    n_train = min(train_samples, len(train_ds))
    n_test = min(test_samples, len(test_ds))

    g = torch.Generator().manual_seed(42)
    train_idx = torch.randperm(len(train_ds), generator=g)[:n_train].tolist()
    test_idx = torch.randperm(len(test_ds), generator=g)[:n_test].tolist()

    train_subset = Subset(train_ds, train_idx)
    test_subset = Subset(test_ds, test_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    correct = 0
    total = 0
    attn_sparsities = []
    mlp_sparsities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, aux = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            attn_sparsities.append(aux.get("mean_attn_sparsity", 0.0))
            mlp_sparsities.append(aux.get("mean_mlp_sparsity", 0.0))

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    mean_attn_sp = (sum(attn_sparsities) / len(attn_sparsities)) * 100.0 if attn_sparsities else 0.0
    mean_mlp_sp = (sum(mlp_sparsities) / len(mlp_sparsities)) * 100.0 if mlp_sparsities else 0.0
    return acc, mean_attn_sp, mean_mlp_sp


def create_10m_vit(variant: str, patch_size: int = 2, device: torch.device = torch.device("cuda")) -> nn.Module:
    """Instantiates a ~10-11M parameter ViT for CIFAR-10 with >90% target sparsity."""
    if variant == "dense":
        model = SmallViT(
            variant="dense",
            img_size=32,
            patch_size=patch_size,
            in_channels=3,
            num_classes=10,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_dim=1536,
        )
    elif variant == "foveal":
        if patch_size == 1:
            # 1024 patches = 64 blocks of 16
            attn_blocks = 16
            max_rem_attn = 2  # 3/64 blocks active -> 95.3% sparsity
            top_p_attn = 0.4
        else:
            # 256 patches = 16 blocks of 16
            attn_blocks = 16
            max_rem_attn = 1  # <= 2/16 blocks active -> 87.5% - 93.8% sparsity
            top_p_attn = 0.25

        model = SmallViT(
            variant="foveal",
            img_size=32,
            patch_size=patch_size,
            in_channels=3,
            num_classes=10,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_dim=1536,
            patch_block_size=attn_blocks,
            attn_top_p=top_p_attn,
            attn_max_remote_blocks=max_rem_attn,
            mlp_block_size=64,             # 24 blocks of 64
            mlp_base_blocks=1,
            mlp_max_remote_blocks=1,      # 2/24 blocks active -> 91.7% sparsity
            mlp_top_p=0.20,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model.to(device)


def train_model(
    variant: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    patch_size: int = 2,
    epochs: int = 5,
    lr: float = 5e-4,
    warmup_epochs: int = 1,
    teacher_interval: int = 10,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    print(f"\n{'='*30} Training {variant.upper()} ViT (~10M Params) {'='*30}")
    model = create_10m_vit(variant, patch_size=patch_size, device=device)
    param_count = sum(p.numel() for p in model.parameters())
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Total Parameters: {param_count / 1e6:.2f}M | Device: {torch.cuda.get_device_name(0)} | AMP FP16: {use_amp}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay to 0.05 of base lr
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    t_start = time.perf_counter()
    history = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        ep_t0 = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Warmup teacher schedule: slightly higher distillation early to rapidly shape indexer projections
            is_warmup = (epoch <= warmup_epochs)
            compute_teacher = (variant == "foveal" and (global_step % teacher_interval == 0))
            distill_coeff = 0.3 if is_warmup else 0.15

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                logits, aux = model(images, compute_teacher_loss=compute_teacher)
                loss = criterion(logits, labels)
                if compute_teacher and "distill_loss" in aux:
                    loss = loss + distill_coeff * aux["distill_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            batches += 1
            global_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_duration = time.perf_counter() - ep_t0
        avg_loss = total_loss / batches if batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        test_acc, attn_sp, mlp_sp = evaluate(model, test_loader, device)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "acc": test_acc,
            "attn_sparsity": attn_sp,
            "mlp_sparsity": mlp_sp,
            "duration": ep_duration,
            "lr": current_lr,
        })

        if variant == "foveal":
            print(
                f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:5.2f}s) | Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | Test Acc: {test_acc:5.2f}% | Attn Pruned: {attn_sp:.1f}% | MLP Pruned: {mlp_sp:.1f}%"
            )
        else:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:5.2f}s) | Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | Test Acc: {test_acc:5.2f}% (Dense 0.0% Sparsity)"
            )

    total_time = time.perf_counter() - t_start
    final_acc, final_attn_sp, final_mlp_sp = evaluate(model, test_loader, device)

    return {
        "model": model,
        "variant": variant,
        "param_count": param_count,
        "final_acc": final_acc,
        "attn_sparsity": final_attn_sp,
        "mlp_sparsity": final_mlp_sp,
        "total_time": total_time,
        "history": history,
    }


def profile_inference(dense_model: nn.Module, foveal_model: nn.Module, device: torch.device):
    print("\n" + "=" * 84)
    print("           INFERENCE FORWARD PASS BENCHMARK (DENSE vs FOVEAL SPARSE)")
    print("=" * 84)

    dense_model.eval()
    foveal_model.eval()
    if hasattr(foveal_model, "refresh_cache"):
        foveal_model.refresh_cache()

    print(f"{'Batch Size':<12} | {'Dense Latency':<16} | {'Foveal Latency':<16} | {'Speedup':<10} | {'Attention Pruned':<18} | {'MLP Pruned':<14}")
    print("-" * 92)

    with torch.no_grad():
        for b_size in [1, 8, 16, 32]:
            x = torch.randn(b_size, 3, 32, 32, device=device)

            # Warmup
            for _ in range(5):
                _ = dense_model(x)
                _ = foveal_model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            reps = 15
            t0 = time.perf_counter()
            for _ in range(reps): _ = dense_model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            d_ms = (time.perf_counter() - t0) * 1000.0 / reps

            t0 = time.perf_counter()
            for _ in range(reps): _ = foveal_model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            f_ms = (time.perf_counter() - t0) * 1000.0 / reps

            _, aux = foveal_model(x)
            attn_sp = aux.get("mean_attn_sparsity", 0.0) * 100.0
            mlp_sp = aux.get("mean_mlp_sparsity", 0.0) * 100.0
            speedup = d_ms / f_ms

            print(
                f"{b_size:<12} | {d_ms:<13.2f} ms | {f_ms:<13.2f} ms | {speedup:<8.2f}x | {f'{attn_sp:.1f}%':<18} | {f'{mlp_sp:.1f}%':<14}"
            )


def main():
    parser = argparse.ArgumentParser(description="Full CIFAR-10 Training: Dense ViT (~10M) vs Foveal Sparse ViT (>90% Sparsity)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--patch-size", type=int, default=2, choices=[1, 2], help="Patch size: 2 (256 tokens) or 1 (1024 tokens)")
    parser.add_argument("--train-samples", type=int, default=50000, help="Number of CIFAR-10 training images (up to 50000)")
    parser.add_argument("--test-samples", type=int, default=10000, help="Number of CIFAR-10 test images (up to 10000)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--teacher-interval", type=int, default=10, help="Distillation frequency in steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 88)
    print("  FULL CIFAR-10 TRAINING COMPARISON: DENSE ViT vs FOVEAL SPARSE ViT")
    print(f"  Hardware: {gpu_name} | Dataset: CIFAR-10 ({args.train_samples} train, {args.test_samples} test)")
    print(f"  Target Model Size: ~10-11M parameters | Patch Size: {args.patch_size}x{args.patch_size} | Target Sparsity: >90%")
    print("=" * 88)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )

    # 1. Train Dense ViT (~10.7M parameters)
    torch.manual_seed(42)
    dense_res = train_model(
        "dense", train_loader, test_loader, patch_size=args.patch_size,
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs, device=device
    )

    # 2. Train Foveal Sparse ViT (~11.2M parameters, >90% sparsity)
    torch.manual_seed(42)
    foveal_res = train_model(
        "foveal", train_loader, test_loader, patch_size=args.patch_size,
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs,
        teacher_interval=args.teacher_interval, device=device
    )

    # 3. Final Summary Comparison Table
    print("\n" + "=" * 88)
    print("                FINAL COMPARISON SUMMARY (FULL CIFAR-10)")
    print("=" * 88)
    print(f"{'Model Variant':<20} | {'Parameters':<12} | {'Test Accuracy':<15} | {'Attention Pruned':<18} | {'MLP Pruned':<14} | {'Train Time':<12}")
    print("-" * 102)
    print(
        f"{'Dense ViT':<20} | {dense_res['param_count']/1e6:<6.2f}M     | "
        f"{dense_res['final_acc']:<6.2f}%        | {'0.0%':<18} | {'0.0%':<14} | {dense_res['total_time']:<6.2f}s"
    )
    attn_sp_str = f"{foveal_res['attn_sparsity']:.1f}%"
    mlp_sp_str = f"{foveal_res['mlp_sparsity']:.1f}%"
    print(
        f"{'Foveal Sparse ViT':<20} | {foveal_res['param_count']/1e6:<6.2f}M     | "
        f"{foveal_res['final_acc']:<6.2f}%        | {attn_sp_str:<18} | "
        f"{mlp_sp_str:<14} | {foveal_res['total_time']:<6.2f}s"
    )
    print("=" * 88)

    # 4. Inference Profiling
    profile_inference(dense_res["model"], foveal_res["model"], device=device)


if __name__ == "__main__":
    main()
