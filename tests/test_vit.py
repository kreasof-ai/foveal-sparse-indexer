"""Unit tests for Small Vision Transformer (ViT)."""

import unittest
import torch

from foveal_indexer.vit import SmallViT


class TestSmallViT(unittest.TestCase):
    def test_dense_vit_forward(self):
        model = SmallViT(variant="dense", img_size=32, patch_size=4, depth=1)
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out, aux = model(x)
        self.assertEqual(out.shape, (2, 10))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_foveal_vit_forward_and_loss(self):
        model = SmallViT(variant="foveal", img_size=32, patch_size=4, depth=1)
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out, aux = model(x, compute_teacher_loss=True)
        self.assertEqual(out.shape, (2, 10))
        self.assertGreater(aux["mean_attn_sparsity"], 0.0)
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        loss = out.sum() + aux["distill_loss"]
        loss.backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
