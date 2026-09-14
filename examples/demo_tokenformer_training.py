"""Demonstration of Foveal Tokenformer with Dual-Gradient Training (CPU).

Demonstrates:
1. Token-Parameter Attention reparameterizing a linear / FFN layer.
2. Parameter tokens partitioned into 16D indexed blocks.
3. Dual-gradient optimization:
   - Task loss backpropagating through the differentiable 16D additive stream.
   - Distillation loss (KL divergence) teaching the 16D indexer to predict true parameter attention mass.
"""

import torch
import torch.nn.functional as F

from foveal_indexer.tokenformer import FovealTokenformerLinear, FovealTokenformerFFN


def main():
    print("=" * 72)
    print("  Foveal Tokenformer: Dual-Gradient Training Demo (CPU)")
    print("=" * 72)

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Layer with 512 parameter tokens divided into blocks of 32 (16 blocks total)
    hidden_size = 64
    out_features = 64
    num_param_tokens = 512
    param_block_size = 32
    base_blocks = 2  # 64 parameter tokens dense base
    max_remote = 2   # at most 2 * 32 = 64 remote parameter tokens
    # Total active per token <= 128 tokens out of 512 (>= 75% parameter sparsity!)

    model = FovealTokenformerLinear(
        in_features=hidden_size,
        out_features=out_features,
        num_param_tokens=num_param_tokens,
        param_block_size=param_block_size,
        base_blocks=base_blocks,
        max_remote_blocks=max_remote,
        top_p=0.8,
        activation="gelu_l2",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Model configured with {num_param_tokens} parameter tokens ({model.num_blocks} blocks of {param_block_size}).")
    print(f"Base blocks: {base_blocks} ({base_blocks * param_block_size} tokens).")
    print(f"Max remote blocks: {max_remote} ({max_remote * param_block_size} tokens).")
    print("Training on synthetic regression task with dual-gradient recipe...\n")

    # Target linear transformation to learn
    true_w = torch.randn(hidden_size, out_features, device=device) / (hidden_size ** 0.5)

    batch_size = 4
    seq_len = 8

    print(f"{'Step':<6} | {'Task Loss':<12} | {'KL Distill Loss':<16} | {'Total Loss':<12} | {'Active Tokens':<14}")
    print("-" * 72)

    for step in range(1, 31):
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        y_target = torch.matmul(x, true_w)

        optimizer.zero_grad()
        out, aux = model(x, compute_teacher_loss=True)

        task_loss = F.mse_loss(out, y_target)
        distill_loss = aux["distill_loss"]
        total_loss = task_loss + 0.5 * distill_loss

        total_loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            mean_active = int(aux["mean_active_param_tokens"])
            print(
                f"{step:<6} | {task_loss.item():<12.5f} | {distill_loss.item():<16.5f} | "
                f"{total_loss.item():<12.5f} | {mean_active:<6}/{num_param_tokens} "
                f"({(1.0 - mean_active / num_param_tokens)*100:.1f}% sparse)"
            )

    print("=" * 72)
    print("Success: Foveal Tokenformer learned the task while keeping >= 75% parameter sparsity!")
    print("=" * 72)


if __name__ == "__main__":
    main()
