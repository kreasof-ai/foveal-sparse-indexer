# Archived: Legacy HY-MT2 Sparsification Experiments

This directory archives early exploratory sparsification methods on `tencent/Hy-MT2-1.8B`:

1. **Static SVD Layer Pruning (`HY_MT2_SPARSIFICATION_REPORT.md`, `sparsify_hy_mt2.py`):**
   - Profiled layer cosine redundancy and removed whole transformer layers with SVD residual bridges.
   - Reached 96.7% retention at 26 layers, but achieved only 1.20× speedup; deeper layer removal (16 layers) desynchronized KV caches during autoregressive decoding.

2. **Blockwise Sparse MLP MatMul (`FOVEAL_SPARSE_MLP_REPORT.md`, `train_foveal_sparse_mlp.py`):**
   - Partitioned intermediate SwiGLU weights into column blocks with top-k dynamic gathering in PyTorch eager mode.
   - PyTorch tensor slicing and kernel launch overhead limited single-token decoding speedups on GPU.

3. **Dynamic Layer Skipping (`FOVEAL_DYNAMIC_SKIP_REPORT.md`, `train_foveal_dynamic_skip.py`):**
   - Tested 16D router-based whole-layer and MLP skipping with residual bridges.

4. **Static Channel Slicing Pattention (`train_pattention_all_layers.py`):**
   - Static parameter reduction without dynamic routing, which permanently deleted 83.3% of parameters across all tokens.

---

## Successor: Unified Foveal Pattention

All experiments in this directory have been superseded by **Foveal Pattention Reparameterization** (`foveal_indexer/foveal_pattention.py`, `foveal_indexer/pattention_triton.py`, `examples/train_foveal_pattention_speedrun.py`), which combines:
- SwiGLU MLP reparameterization into Token-Parameter Attention ($K_{\text{gate}}, K_{\text{up}}, V$) with 1.000000 fidelity
- 64-token chunk blockwise sparsity (96 blocks)
- Empirical offline SVD router initialization (0.0089 initial KL)
- Fused SRAM register execution (6.51× layer speedup)
- Differentiable 16D additive stream
- Dense KL distillation matching teacher block energy distribution
- **100.4% BLEU retention (24.00 vs 23.91 baseline)** on held-out WMT22 `zh-en`.
