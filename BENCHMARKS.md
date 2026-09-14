# Foveal Sparse Indexer: Performance & Benchmark Measurements

This document presents quantitative performance measurements comparing **Foveal Sparse Indexing** against dense baselines across three computational domains:
1. **Sparse Attention Context Scaling (Autoregressive Decode Step)**
2. **Tokenformer Token-Parameter Attention MatMul**
3. **Blockwise Sparse Linear / GEMM Arithmetic & Memory Traffic**

---

## 1. Sparse Attention: Flat Autoregressive Decode Latency

In standard autoregressive decoding (`batch_size = 1`), dense attention must query the entire accumulated key-value cache ($O(N)$ operations per generated token). As context grows, step latency increases linearly and throughput degrades steadily.

**Foveal Sparse Attention** partitions the causal history into:
- A parafoveal local sliding window ($W = 128$ tokens).
- A pool of completed 16D remote pages outside the window.
- An adaptive top-$p$ selection capped at $K_{\max} = 4$ pages ($4 \times 32 = 128$ tokens).

Active attention support is strictly bounded to at most $128 + 128 = 256$ tokens across the entire generation sequence.

### CPU Latency & Scaling (`batch_size=1`, `hidden_size=256`, 8 Q-heads, 4 KV-heads)

Measured on single-socket x86_64 CPU (PyTorch 2.11):

| Context Length | Dense KV Tokens | Foveal Active Tokens | Dense Step Latency | Dense Throughput | Foveal Step Latency | Foveal Throughput | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 128 | 128 | 0.041 ms | 24,213 tok/s | 0.989 ms | 1,010 tok/s | 0.04× |
| **256** | 256 | 256 | 0.059 ms | 17,006 tok/s | 1.673 ms | 598 tok/s | 0.04× |
| **512** | 512 | 256 | 0.112 ms | 8,945 tok/s | 1.274 ms | 785 tok/s | 0.09× |
| **1,024** | 1,024 | 256 | 0.134 ms | 7,446 tok/s | 1.290 ms | 775 tok/s | 0.10× |
| **2,048** | 2,048 | 256 | 0.240 ms | 4,165 tok/s | 1.337 ms | 748 tok/s | 0.18× |
| **4,096** | 4,096 | 256 | 0.462 ms | 2,165 tok/s | 1.138 ms | 879 tok/s | 0.41× |
| **8,192** | 8,192 | 256 | 1.042 ms | 960 tok/s | 1.232 ms | 812 tok/s | 0.85× |
| **16,384** | 16,384 | 256 | 1.946 ms | 514 tok/s | **1.184 ms** | **845 tok/s** | **1.64×** |
| **32,768** | 32,768 | 256 | 3.641 ms | 275 tok/s | **1.264 ms** | **791 tok/s** | **2.88×** |

#### Key Takeaways:
- **Linear Degradation in Dense Attention:** From 128 to 32K context (a 256× context increase), Dense attention latency slows down by **88×** (0.041 ms $\to$ 3.641 ms).
- **Flat $O(1)$ Scaling in Foveal Attention:** Foveal step latency remains constant at **~1.2 ms** from 512 tokens to 32,768 tokens.
- **Crossover Point:** Beyond 8K tokens on CPU, Foveal Sparse Attention surpasses dense attention in pure wall-clock speed, reaching **2.88× faster step latency at 32K context**.

---

### GPU Reference: Long-Context Serving (NVIDIA L40S, from `atma/foveal_cpt`)

When paired with Triton fused attention kernels and CUDA graph capture (from the continuous pre-training sweep audit in `atma/foveal_cpt`):

| Attention Core / Mode | Context Length | Dense Source Baseline | Foveal Serving Throughput | Peak GPU VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **Polar `kl`** | 2K | 441.6 tok/s | **588.8 tok/s** | 2.10 GiB |
| **Polar `kl`** | 32K | — | **572.3 tok/s** | 4.05 GiB |
| **Polar `kl`** | 256K | — | **571.0 tok/s** | 18.84 GiB |
| **NoPE `lm_output_kl`** | 2K | 465.6 tok/s | **590.2 tok/s** | 2.10 GiB |
| **NoPE `lm_output_kl`** | 32K | — | **573.1 tok/s** | 3.99 GiB |
| **NoPE `lm_output_kl`** | 256K | — | **572.0 tok/s** | 18.34 GiB |

*Decoding throughput drops by less than **3%** over a 128× context increase (from 2K to 256K).*

---

## 2. Foveal Tokenformer: Token-Parameter Attention MatMul

In [Tokenformer (arXiv:2410.23168)](https://arxiv.org/abs/2410.23168), model weight matrices are reformulated as parameter tokens ($K_P, V_P \in \mathbb{R}^{N \times D}$). Input tokens query the parameter tokens using cross-attention:
$$\text{Pattention}(X, K_P, V_P) = \Theta(X K_P^\top) V_P$$

Because parameter tokens operate identically to sequence tokens in attention, the Foveal Sparse Indexer translates seamlessly:
- Parameter tokens are partitioned into blocks of 64 tokens.
- A small base set of blocks (e.g. 4 blocks = 256 tokens) provides a permanent dense bottleneck.
- Input queries retrieve top-$p$ remote parameter blocks via 16D cosine dot-product.

### Parameter Scaling Benchmark (CPU, $D = 1024$, Block Size = 64, $M = 1$ Decode Step)

| Total Parameters ($N_{\text{param}}$) | Dense Tokenformer | Foveal Tokenformer | Active Parameter Tokens | Parameter Sparsity | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2,048** | 1.767 ms | 2.562 ms | 640 | 68.8% | 0.69× |
| **4,096** | 3.446 ms | 3.325 ms | 1,024 | 75.0% | **1.04×** |
| **8,192** | 5.408 ms | 3.500 ms | 1,024 | 87.5% | **1.55×** |
| **16,384** | 18.666 ms | **3.289 ms** | 1,024 | **93.8%** | **5.68×** |

#### Why Foveal Tokenformer Excels:
1. **No Output Scatter Overhead:** Parameter tokens are attended to directly ($X \cdot K_{P,\text{active}}^\top \cdot V_{P,\text{active}}$). There is no sparse scattering into full-width output tensors.
2. **Flat Parameter Latency:** As total parameters scale **8×** (from 2K to 16K parameter tokens), Dense Tokenformer latency grows from 1.77 ms to 18.67 ms (a 10.5× slowdown). Foveal Tokenformer stays **flat at ~3.3 ms**, delivering a **5.68× wall-clock speedup**.

---

## 3. Blockwise Sparse Linear / GEMM

For standard feedforward and projection layers ($Y = X W^\top$), the weight matrix $W \in \mathbb{R}^{N \times K}$ is partitioned into column blocks of size $K \times 64$.

### Arithmetic FLOP & Memory Traffic Reduction ($M = 1$, $K = 2048$, $N = 16384$)

| Variant | Active Output Channels | Compute MFLOPs | FLOP Reduction | Weight Memory Read per Token |
| :--- | :---: | :---: | :---: | :---: |
| **Dense GEMM** | 16,384 | 67.11 MFLOPs | **0.0%** | 67.1 MB (FP32) / 33.6 MB (BF16) |
| **Foveal BlockSparse (50%)** | 8,192 | 33.55 MFLOPs | **50.0%** | 33.6 MB (FP32) / 16.8 MB (BF16) |
| **Foveal BlockSparse (75%)** | 4,096 | 16.78 MFLOPs | **75.0%** | 16.8 MB (FP32) / 8.4 MB (BF16) |
| **Foveal BlockSparse (87.5%)** | 2,048 | **8.39 MFLOPs** | **87.5%** | **8.4 MB (FP32) / 4.2 MB (BF16)** |

### Raw GEMM Execution Time on CPU ($1 \times 2048 \times 8192$):
- **Dense Linear:** 3.17 ms
- **Sparse Active Slices (87.5% sparse):** 0.84 ms
- **Raw Compute Acceleration:** **3.79× speedup**

---

## Reproduction Commands

To reproduce all measurements locally:

```powershell
# 1. Full context attention decode scaling (128 -> 4096 tokens)
python examples/demo_attention_flat_decode.py

# 2. Dedicated Sparse MatMul vs Dense MatMul benchmark
python examples/benchmark_sparse_matmul.py

# 3. Comprehensive multi-domain benchmark
python examples/benchmark_all.py
```
