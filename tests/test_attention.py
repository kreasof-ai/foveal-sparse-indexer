"""Unit tests for Foveal Sparse Attention."""

import unittest
import torch

from foveal_indexer.attention import FovealSparseAttention, FovealKVCache


class TestFovealAttention(unittest.TestCase):
    def test_attention_prefill_and_loss(self):
        attn = FovealSparseAttention(
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            page_size=16,
            local_window_blocks=2,
            index_dim=16,
            top_p=0.9,
            max_remote_pages=2,
        )

        x = torch.randn(2, 64, 64, requires_grad=True)
        out, aux = attn(x, compute_teacher_loss=True)

        self.assertEqual(out.shape, (2, 64, 64))
        self.assertEqual(aux["route"].remote_counts.shape, (2, 4))
        self.assertGreaterEqual(aux["distill_loss"].item(), 0.0)

        # Backward pass
        total_loss = out.sum() + aux["distill_loss"]
        total_loss.backward()

        self.assertIsNotNone(attn.q_proj.weight.grad)
        self.assertIsNotNone(attn.indexer.q_proj.weight.grad)

    def test_attention_kv_cache_decode(self):
        attn = FovealSparseAttention(
            hidden_size=32,
            num_heads=2,
            num_kv_heads=2,
            head_dim=16,
            page_size=8,
            local_window_blocks=2,  # 16 tokens local
            index_dim=16,
            top_p=0.9,
            min_remote_pages=1,
            max_remote_pages=2,
        )

        max_seq_len = 48
        kv_cache = attn.init_kv_cache(batch_size=1, max_seq_len=max_seq_len, device=torch.device("cpu"))

        # Decode 40 tokens one-by-one
        for step in range(40):
            tok = torch.randn(1, 1, 32)
            out, dec_aux = attn.decode_step(tok, kv_cache)
            self.assertEqual(out.shape, (1, 1, 32))

        # After 40 steps, local window covers tokens [25, 40].
        # Pages completed outside local window: page 0 and page 1.
        self.assertGreaterEqual(len(dec_aux["selected_remote_pages"][0]), 1)


if __name__ == "__main__":
    unittest.main()
