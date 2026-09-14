"""Demonstration of Blockwise Sparse MatMul with 16D Foveal Indexer (CPU).

Demonstrates:
1. Blockwise partitioned weight matrix (e.g. column blocks of size K x 64).
2. Adaptive block selection based on 16D Foveal Indexer.
3. FLOP reduction and sparsity profiling.
4. Dual-gradient training with additive stream + teacher distillation.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from foveal_indexer.block_matmul import BlockSparseLinear


def main():
    print("=" * 72)
    print("  Foveal Blockwise Sparse MatMul Demo (CPU)")
    print("=" * 72)

    torch.manual_seed(42)
    device = torch.device("cpu")

    in_features = 256
    out_features = 1024
    block_size = 64
    base_blocks = 2  # 2 * 64 = 128 output channels base
    max_remote = 2   # at most 2 * 64 = 128 output channels remote
    # Total blocks = 16. Active blocks <= 4 (>= 75% block sparsity!)

    layer = BlockSparseLinear(
        in_features=in_features,
        out_features=out_features,
        block_size=block_size,
        base_blocks=base_blocks,
        max_remote_blocks=max_remote,
        top_p=0.5,
    ).to(device)

    print(f"Matrix shape: {in_features} -> {out_features}")
    print(f"Partitioned into {layer.num_blocks} blocks of size {in_features} x {block_size}.")
    print(f"Base blocks: {base_blocks} ({base_blocks * block_size} channels).")
    print(f"Max remote blocks: {max_remote} ({max_remote * block_size} channels).")

    # Single query test
    x = torch.randn(1, 1, in_features, device=device)
    out, aux = layer(x)

    print("\nInference Sparsity Profile:")
    print(f"  Active blocks: {aux['active_blocks_union']} / {aux['total_blocks']}")
    print(f"  Theoretical GEMM FLOP reduction: {aux['sparsity'] * 100:.1f}%")
    print(f"  Output shape: {out.shape}")

    # Training step demo
    print("\nTesting dual-gradient training step (Task Loss + KL Distill):")
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

    for step in range(1, 6):
        batch_x = torch.randn(4, 8, in_features, device=device)
        target = torch.randn(4, 8, out_features, device=device)

        optimizer.zero_grad()
        out, aux = layer(batch_x, compute_teacher_loss=True)

        task_loss = F.mse_loss(out, target)
        distill_loss = aux["distill_loss"]
        total_loss = task_loss + 0.1 * distill_loss

        total_loss.backward()
        optimizer.step()

        print(
            f"  Step {step}: Task Loss = {task_loss.item():.4f}, "
            f"Distill KL = {distill_loss.item():.4f}, "
            f"Mean Sparsity = {aux['sparsity']*100:.1f}%"
        )

    print("=" * 72)
    print("Success: Blockwise Sparse MatMul executed and trained seamlessly!")
    print("=" * 72)


if __name__ == "__main__":
    main()
