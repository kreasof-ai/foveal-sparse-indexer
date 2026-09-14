# Foveal Sparse Indexer

A unified multi-resolution architecture for **Sparse Attention** and **Sparse MatMul** powered by a learned 16-dimensional MQA indexer.

Originally conceived in [Foveal-MoE](https://github.com/kreasof-ai/Foveal-MoE) and first validated for sparse attention during continuous pre-training in [ATMA foveal_cpt](https://github.com/kreasof-ai/atma), this repository generalizes the Foveal Indexer beyond self-attention to matrix multiplications—both via **attention reparameterization** ([Tokenformer, arXiv:2410.23168](https://arxiv.org/abs/2410.23168)) and **blockwise tiled sparse linear projections** ($64 \times 64$ / $128 \times 128$).

---

## The Core Recipe

Across all modalities, the foundational indexing principles remain identical:

1. **Multi-Element Blocks (Pages):** Elements (tokens, parameter tokens, or weight columns) are partitioned into blocks of size $B$ (e.g., 32, 64, or 128).
2. **16-Dimensional MQA Representation:** Each block is pooled into a compact 16-dimensional key $k^I \in \mathbb{R}^{16}$ and value $v^I \in \mathbb{R}^{16}$. Queries are projected to $q^I \in \mathbb{R}^{16}$.
3. **Adaptive Attention-Mass Routing (Top-$p$):** 16D queries score blocks via cosine dot-product. Blocks are chosen dynamically by cumulative predicted attention mass ($p \in (0, 1]$) subject to $[K_{\min}, K_{\max}]$ service-level guardrails and parafoveal (dense base) reservation.
4. **Dual-Gradient Training Optimization:**
   - **Additive Stream (`lm_output`):** A continuous, differentiable soft read over 16D block values projected into the residual stream ($Y_{\text{add}} = W_{\text{out}}^I(\sum \pi_j v^I_j)$). Task/LM loss gradients flow directly and smoothly into the indexer projections without differentiating through discrete routing.
   - **Auxiliary Distillation (`distill_loss`):** Block-level KL divergence ($D_{\text{KL}}(P_{\text{teacher\_mass}} \parallel P_{\text{indexer\_scores}})$) matching indexer block probabilities to true dense attention mass or activation energy.

---

## Modalities

### 1. Foveal Sparse Attention (`FovealSparseAttention`)
- **Parafoveal Local Window:** Full-rank exact causal sliding window over the most recent $W$ tokens.
- **Foveal Remote Pages:** Top-$p$ remote KV pages retrieved dynamically by predicted attention mass.
- **Flat Decoding Latency:** In autoregressive decoding, active attention support is bounded by $W + K_{\max} \cdot \text{page\_size}$ regardless of total context length, yielding strictly $O(1)$ decoding steps.

### 2. Foveal Tokenformer (`FovealTokenformerLinear`, `FovealTokenformerFFN`)
- **Attention Reparameterization:** Parameters are reformulated as key-value parameter tokens ($K_P, V_P$) where input tokens act as queries:
  $$\text{Pattention}(X, K_P, V_P) = \Theta(X K_P^\top) V_P$$
- **Blocked Parameter Tokens:** Large parameter sets (e.g. 512–4096+ tokens) are grouped into blocks.
- **Dense Base + Sparse Dynamic Blocks:** Base parameter blocks act as an unconditional dense bottleneck, while specialized parameter blocks are retrieved on demand.

### 3. Blockwise Sparse MatMul (`BlockSparseLinear`, `BlockwiseMatMul2D`)
- **Tiled Linear Projections:** Weight matrix is partitioned into column blocks ($K \times 64$ / $K \times 128$) or 2D tiles ($64 \times 64$).
- **Dynamic Block Execution:** Only active weight blocks are computed during GEMM, delivering theoretical FLOP reductions proportional to block sparsity.

---

## Codebase Architecture

```
foveal-sparse-indexer/
├── foveal_indexer/
│   ├── __init__.py
│   ├── core.py             # 16D FovealIndexer, top-p block routing, additive stream & KL loss
│   ├── attention.py        # Foveal Sparse Attention & O(1) decode KV-cache
│   ├── tokenformer.py      # Foveal Tokenformer (Token-Parameter Attention with blocked parameter tokens)
│   └── block_matmul.py     # Blockwise Sparse MatMul (64x64 / 128x128 tiled linear projections)
├── tests/
│   ├── test_indexer.py     # Unit tests for routing, guardrails, and gradients
│   ├── test_attention.py   # Causal masking, KV cache decode, prefill distillation
│   ├── test_tokenformer.py # Parameter token block routing & gradient verification
│   └── test_block_matmul.py# Block matmul forward, backward & sparsity verification
├── examples/
│   ├── demo_attention_flat_decode.py  # Flat decoding latency benchmark (CPU)
│   ├── demo_tokenformer_training.py   # Dual-gradient Tokenformer training (CPU)
│   └── demo_block_matmul.py           # Blockwise sparse GEMM & FLOP reduction (CPU)
├── run_tests.py            # Test discovery runner
└── README.md
```

---

## Quickstart (CPU Reproducible)

The implementation is 100% portable and executes on CPU using pure PyTorch operations, with device-agnostic abstractions ready for GPU acceleration (FlexAttention / Triton).

### Running Unit Tests
```bash
python run_tests.py
```
Output:
```
test_attention_kv_cache_decode (test_attention.TestFovealAttention) ... ok
test_attention_prefill_and_loss (test_attention.TestFovealAttention) ... ok
test_block_sparse_linear (test_block_matmul.TestFovealBlockMatMul) ... ok
test_blockwise_matmul_2d (test_block_matmul.TestFovealBlockMatMul) ... ok
test_foveal_indexer_projections_and_gradients (test_indexer.TestFovealIndexer) ... ok
test_masked_softmax (test_indexer.TestFovealIndexer) ... ok
test_select_blocks_causal (test_indexer.TestFovealIndexer) ... ok
test_select_blocks_guardrails (test_indexer.TestFovealIndexer) ... ok
test_tokenformer_ffn (test_tokenformer.TestFovealTokenformer) ... ok
test_tokenformer_linear (test_tokenformer.TestFovealTokenformer) ... ok

Ran 10 tests in 0.110s - OK
```

### Running Demonstrations

1. **Flat Decoding Latency Demo:**
   ```bash
   python examples/demo_attention_flat_decode.py
   ```
   Measures per-step decoding latency from 128 to 4096 context tokens, showing bounded active memory and flat $O(1)$ scaling.

2. **Foveal Tokenformer Dual-Gradient Training:**
   ```bash
   python examples/demo_tokenformer_training.py
   ```
   Demonstrates training a 512-parameter-token layer with 75% parameter sparsity using task loss + KL distillation.

3. **Blockwise Sparse MatMul:**
   ```bash
   python examples/demo_block_matmul.py
   ```
   Demonstrates dynamic column block selection for a $256 \to 1024$ linear layer with 75% theoretical GEMM FLOP reduction.

---

## References

- [Foveal-MoE Research Proposal](https://github.com/kreasof-ai/Foveal-MoE)
- [ATMA foveal_cpt Sweep & Benchmarks](https://github.com/kreasof-ai/atma/tree/main/foveal_cpt)
- [Tokenformer: Rethinking Transformer Scaling with Tokenized Model Parameters (Wang et al., 2024)](https://arxiv.org/abs/2410.23168)
- [Native Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089)
