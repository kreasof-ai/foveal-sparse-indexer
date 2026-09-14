"""Unit tests for Foveal Blockwise Sparse MatMul."""

import unittest
import torch

from foveal_indexer.block_matmul import BlockSparseLinear, BlockwiseMatMul2D


class TestFovealBlockMatMul(unittest.TestCase):
    def test_block_sparse_linear(self):
        linear = BlockSparseLinear(
            in_features=64,
            out_features=256,
            block_size=32,
            base_blocks=2,
            index_dim=16,
            top_p=0.5,
            max_remote_blocks=2,
            bias=True,
        )

        x = torch.randn(2, 4, 64, requires_grad=True)
        out, aux = linear(x, compute_teacher_loss=True)

        self.assertEqual(out.shape, (2, 4, 256))
        # Total blocks = 256 // 32 = 8
        # Base blocks = 2, max_remote = 2 -> mean active <= 4 blocks
        self.assertLessEqual(aux["mean_active_blocks"], 4)
        self.assertLessEqual(aux["route"].remote_counts.max().item(), 2)
        self.assertGreaterEqual(aux["sparsity"], 0.5)
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        # Single token test: union of active blocks <= base + max_remote
        x_single = torch.randn(1, 1, 64)
        _, aux_single = linear(x_single)
        self.assertLessEqual(aux_single["active_blocks_union"], 4)

        # Test backward pass
        loss = out.sum() + aux["distill_loss"]
        loss.backward()

        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)
        self.assertIsNotNone(linear.weight_to_k.weight.grad)

    def test_blockwise_matmul_2d(self):
        matmul = BlockwiseMatMul2D(
            in_features=64,
            out_features=64,
            block_k=32,
            block_n=32,
            base_tiles_ratio=0.25,
            top_p=0.8,
        )

        x = torch.randn(2, 4, 64, requires_grad=True)
        out, aux = matmul(x, compute_teacher_loss=True)

        self.assertEqual(out.shape, (2, 4, 64))
        self.assertEqual(aux["total_tiles"], 4)
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        loss = out.sum() + aux["distill_loss"]
        loss.backward()
        self.assertIsNotNone(matmul.weight.grad)


if __name__ == "__main__":
    unittest.main()
