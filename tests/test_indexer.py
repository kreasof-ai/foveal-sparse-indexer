"""Unit tests for Core 16D Foveal Indexer."""

import unittest
import torch

from foveal_indexer.core import FovealIndexer, Route, select_blocks, masked_softmax


class TestFovealIndexer(unittest.TestCase):
    def test_masked_softmax(self):
        scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[True, True, False], [False, True, True]])
        probs = masked_softmax(scores, mask)

        self.assertEqual(probs[0, 2].item(), 0.0)
        self.assertEqual(probs[1, 0].item(), 0.0)
        self.assertTrue(torch.isclose(probs[0, :2].sum(), torch.tensor(1.0), atol=1e-5))
        self.assertTrue(torch.isclose(probs[1, 1:].sum(), torch.tensor(1.0), atol=1e-5))

    def test_select_blocks_causal(self):
        batch = 1
        query_blocks = 4
        total_blocks = 4
        scores = torch.zeros(batch, query_blocks, total_blocks)
        scores[0, 3, 0] = 10.0

        route = select_blocks(
            scores,
            causal=True,
            block_size=16,
            local_window_blocks=1,
            top_p=0.9,
            min_remote_blocks=0,
            max_remote_blocks=2,
            remote_capacity=2,
        )

        self.assertEqual(route.remote_counts[0, 0].item(), 0)
        self.assertEqual(route.remote_counts[0, 3].item(), 1)
        self.assertEqual(route.remote_indices[0, 3, 0].item(), 0)

    def test_select_blocks_guardrails(self):
        batch = 2
        query_blocks = 8
        total_blocks = 8
        scores = torch.randn(batch, query_blocks, total_blocks)

        route = select_blocks(
            scores,
            causal=False,
            local_window_blocks=2,
            top_p=0.99,
            min_remote_blocks=1,
            max_remote_blocks=3,
            remote_capacity=4,
        )

        for b in range(batch):
            for q in range(query_blocks):
                cnt = route.remote_counts[b, q].item()
                self.assertTrue(1 <= cnt <= 3, f"Count {cnt} violated [K_min=1, K_max=3]")

    def test_foveal_indexer_projections_and_gradients(self):
        indexer = FovealIndexer(
            hidden_size=32,
            index_dim=16,
            block_size=16,
            local_window_blocks=1,
            top_p=0.9,
            use_additive_stream=True,
        )

        x = torch.randn(2, 32, 32, requires_grad=True)
        q_16d = indexer.compute_query_16d(x)
        self.assertEqual(q_16d.shape, (2, 32, 16))

        k_b, v_b = indexer.compute_kv_blocks_16d(x)
        self.assertEqual(k_b.shape, (2, 2, 16))
        self.assertEqual(v_b.shape, (2, 2, 16))

        q_b = q_16d[:, ::16]
        scores = indexer.compute_block_scores(q_b, k_b)
        self.assertEqual(scores.shape, (2, 2, 2))

        # Additive stream
        add_out = indexer.additive_stream(scores, v_b)
        self.assertEqual(add_out.shape, (2, 32, 32))

        # Distillation loss
        teacher_mass = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]).expand(2, 2, 2)
        distill = indexer.distillation_loss(scores, teacher_mass)
        self.assertGreaterEqual(distill.item(), 0.0)

        # Test backward pass
        loss = add_out.sum() + distill
        loss.backward()
        self.assertIsNotNone(indexer.q_proj.weight.grad)
        self.assertIsNotNone(indexer.k_proj.weight.grad)
        self.assertIsNotNone(indexer.v_proj.weight.grad)
        self.assertIsNotNone(indexer.out_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
