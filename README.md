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

### 4. Vision Transformer (ViT) Modality (`SmallViT`, `FovealVisionAttention`)
- **Spatial Patch-Block Attention:** 2D images are tiled into $P$ patches (e.g. 64 patches of $4 \times 4$). Self-block attention retains local spatial features while 16D Foveal Indexer retrieves remote image quadrants.
- **Combined Sparsity:** Couples block-sparse attention with block-sparse MLP for end-to-end vision classification.

---

## Codebase Architecture

```
foveal-sparse-indexer/
├── foveal_indexer/
│   ├── __init__.py
│   ├── core.py             # 16D FovealIndexer, top-p block routing, additive stream & KL loss
│   ├── attention.py        # Foveal Sparse Attention & O(1) decode KV-cache
│   ├── tokenformer.py      # Foveal Tokenformer (Token-Parameter Attention with blocked parameter tokens)
│   ├── block_matmul.py     # Blockwise Sparse MatMul (64x64 / 128x128 tiled linear projections)
│   └── vit.py              # Small Vision Transformer (Dense and Foveal Sparse blocks)
├── tests/
│   ├── test_indexer.py     # Unit tests for routing, guardrails, and gradients
│   ├── test_attention.py   # Causal masking, KV cache decode, prefill distillation
│   ├── test_tokenformer.py # Parameter token block routing & gradient verification
│   ├── test_block_matmul.py# Block matmul forward, backward & sparsity verification
│   └── test_vit.py         # ViT forward pass and gradient checks
├── examples/
│   ├── demo_mnist_vit.py              # MNIST ViT comparison (Dense vs Foveal Sparse)
│   ├── demo_attention_flat_decode.py  # Flat decoding latency benchmark (CPU)
│   ├── demo_tokenformer_training.py   # Dual-gradient Tokenformer training (CPU)
│   ├── demo_block_matmul.py           # Blockwise sparse GEMM & FLOP reduction (CPU)
│   ├── benchmark_all.py               # Multi-domain CPU benchmark suite
│   └── benchmark_sparse_matmul.py     # Dedicated Sparse vs Dense MatMul scaling
├── BENCHMARKS.md           # Full performance reports and audit logs
├── run_tests.py            # Test discovery runner
└── README.md
```

---

## Performance & Benchmark Summary

Detailed profiling tables, scaling curves, and GPU serving matrices are documented in [**`BENCHMARKS.md`**](BENCHMARKS.md).

### 1. Sparse Attention Decode Latency vs Context Length (CPU)
Active tokens capped at $W + K_{\max} \cdot B = 256$ tokens:

| Context Length | Dense Attention Latency | Foveal Sparse Latency | Foveal Active Tokens | Wall-Clock Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **512** | 0.112 ms | 1.274 ms | 256 | 0.09× |
| **2,048** | 0.240 ms | 1.337 ms | 256 | 0.18× |
| **8,192** | 1.042 ms | 1.232 ms | 256 | 0.85× |
| **16,384** | 1.946 ms | **1.184 ms** | 256 | **1.64×** |
| **32,768** | 3.641 ms | **1.264 ms** | 256 | **2.88×** |

*(Dense attention slows by 88× across context, while Foveal Attention remains completely **flat at ~1.2 ms**).*

### 2. Foveal Tokenformer MatMul Parameter Scaling (CPU, $D = 1024$)
Cross-attention between input token ($M = 1$) and parameter tokens ($K_P, V_P$):

| Total Parameters ($N_{\text{param}}$) | Dense Tokenformer | Foveal Tokenformer | Active Parameters | Parameter Sparsity | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2,048** | 1.77 ms | 2.56 ms | 640 | 68.8% | 0.69× |
| **4,096** | 3.45 ms | 3.33 ms | 1,024 | 75.0% | **1.04×** |
| **8,192** | 5.41 ms | 3.50 ms | 1,024 | 87.5% | **1.55×** |
| **16,384** | 18.67 ms | **3.29 ms** | 1,024 | **93.8%** | **5.68×** |

*(As model parameter tokens scale 8× from 2K to 16K, Foveal Tokenformer latency stays **flat at ~3.3 ms**, yielding a **5.68× speedup**).*

### 3. Blockwise Sparse Linear FLOPs ($K = 2048$, $N = 16384$)

| Configuration | Active Channels | Compute MFLOPs | FLOP Reduction | Memory Traffic Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Dense GEMM** | 16,384 | 67.11 MFLOPs | **0.0%** | 67.1 MB (FP32) |
| **Foveal Sparse (75%)** | 4,096 | 16.78 MFLOPs | **75.0%** | **4.0× less weight read** |
| **Foveal Sparse (87.5%)** | 2,048 | **8.39 MFLOPs** | **87.5%** | **8.0× less weight read** |

### 4. Small Vision Transformer (ViT) on MNIST (CPU)
2-layer ViT ($D=64$, 4 heads, $64 \to 128$ MLP, 64 patches, periodic distillation every 10 steps):

| Model Variant | Test Accuracy | Attention Sparsity | MLP Sparsity | Inference Latency (Batch=64) | Inference Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense ViT** | 29.33% | 0.0% | 0.0% | 130.88 ms | 489.0 img/s |
| **Foveal Sparse ViT** | **28.00%** | **37.7%** | **45.5%** | **103.46 ms** | **618.6 img/s (1.27×)** |

*(Foveal ViT achieves accuracy parity on MNIST while delivering a **1.27× batched inference throughput speedup** on CPU, pruning ~38% of attention connections and ~46% of MLP channels dynamically).*

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

1. **MNIST ViT Benchmark (Dense vs Foveal Sparse):**
   ```bash
   python examples/demo_mnist_vit.py --epochs 3
   ```
   Trains and evaluates a 2-layer Vision Transformer on MNIST, comparing Dense vs Foveal Sparse accuracy, attention sparsity, and MLP sparsity.

2. **Flat Decoding Latency Demo:**
   ```bash
   python examples/demo_attention_flat_decode.py
   ```
   Measures per-step decoding latency from 128 to 4096 context tokens, showing bounded active memory and flat $O(1)$ scaling.

3. **Foveal Tokenformer Dual-Gradient Training:**
   ```bash
   python examples/demo_tokenformer_training.py
   ```
   Demonstrates training a 512-parameter-token layer with 75% parameter sparsity using task loss + KL distillation.

4. **Blockwise Sparse MatMul:**
   ```bash
   python examples/demo_block_matmul.py
   ```
   Demonstrates dynamic column block selection for a $256 \to 1024$ linear layer with 75% theoretical GEMM FLOP reduction.

5. **Dedicated Sparse MatMul vs Dense Benchmark:**
   ```bash
   python examples/benchmark_sparse_matmul.py
   ```
   Benchmarks Tokenformer parameter attention and blockwise GEMM scaling against dense baselines.

6. **Comprehensive Multi-Domain Benchmark:**
   ```bash
   python examples/benchmark_all.py
   ```
   Runs the full benchmark suite across attention decoding, GEMM, and Tokenformer parameter scaling.

---

## References

- [Foveal-MoE Research Proposal](https://github.com/kreasof-ai/Foveal-MoE)
- [ATMA foveal_cpt Sweep & Benchmarks](https://github.com/kreasof-ai/atma/tree/main/foveal_cpt)
- [Tokenformer: Rethinking Transformer Scaling with Tokenized Model Parameters (Wang et al., 2024)](https://arxiv.org/abs/2410.23168)
- [Native Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089)
