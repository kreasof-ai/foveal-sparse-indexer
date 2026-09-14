"""Unit tests for Foveal Tokenformer."""

import unittest
import torch

from foveal_indexer.tokenformer import FovealTokenformerLinear, FovealTokenformerFFN


class TestFovealTokenformer(unittest.TestCase):
    def test_tokenformer_linear(self):
        layer = FovealTokenformerLinear(
            in_features=64,
            out_features=64,
            num_param_tokens=256,
            param_block_size=32,
            base_blocks=2,
            index_dim=16,
            top_p=0.8,
            max_remote_blocks=2,
        )

        x = torch.randn(2, 8, 64, requires_grad=True)
        out, aux = layer(x, compute_teacher_loss=True)

        self.assertEqual(out.shape, (2, 8, 64))
        # Mean active parameter tokens should be bounded by base + max_remote
        max_active = (2 + 2) * 32
        self.assertLessEqual(aux["mean_active_param_tokens"], max_active)
        self.assertLessEqual(aux["route"].remote_counts.max().item(), 2)
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        # Single-token test: union must also be bounded by max_active
        x_single = torch.randn(1, 1, 64)
        _, aux_single = layer(x_single)
        self.assertLessEqual(aux_single["active_param_tokens_union"], max_active)

        # Backward pass
        loss = out.sum() + aux["distill_loss"]
        loss.backward()

        self.assertIsNotNone(layer.k_param.grad)
        self.assertIsNotNone(layer.v_param.grad)
        self.assertIsNotNone(layer.indexer.q_proj.weight.grad)

    def test_tokenformer_ffn(self):
        ffn = FovealTokenformerFFN(
            hidden_size=32,
            intermediate_size=64,
            num_param_tokens=128,
            param_block_size=16,
            base_blocks=2,
        )

        x = torch.randn(2, 4, 32, requires_grad=True)
        out, aux = ffn(x, compute_teacher_loss=True)

        self.assertEqual(out.shape, (2, 4, 32))
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        loss = out.sum() + aux["distill_loss"]
        loss.backward()

        self.assertIsNotNone(ffn.fc1.k_param.grad)
        self.assertIsNotNone(ffn.fc2.v_param.grad)


if __name__ == "__main__":
    unittest.main()
