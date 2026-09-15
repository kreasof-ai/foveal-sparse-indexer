# Archived Experiments: HY-MT2-1.8B Sparsification & Pattention

This directory archives all historical experiments on `tencent/Hy-MT2-1.8B`:

1. **`llm_wmt22_24/`:** Multi-lingual WMT22-24 adaptation with Chunk 8 Attention dimension slimming, Chunk 32 SwiGLU MLP, HybridMuonAdamW, and SVD calibration.
2. **`llm_pattention/`:** Foveal Token-Parameter Attention speedrun experiments (uniform 32L and 16L semantic reasoning sparsification, SVD router calibration, dual-gradient distillation, and sparsity grid search).
3. **`legacy/`:** Early exploratory sparsification attempts on Hy-MT2 (static SVD layer pruning, PyTorch eager blockwise matmul, dynamic layer skipping).

---

### Reason for Archival

1. **Failure to Meet End-to-End Speedup & Convergence Targets:**
   - **WMT22-24 Slicing Failure:** Simultaneously slicing the attention head dimension $d_k = 128$ down to 32 or 64 across all 32 layers truncated the lower-frequency rotary positional embeddings ($\theta_{32} \dots \theta_{63}$), causing severe representation drift across deep sequential layers (cross-entropy loss plateaued around 7.7–11.4). Even with large dataset coverage and `HybridMuonAdamW`, 32-layer simultaneous attention slimming failed to converge to usable translation quality.
   - **Amdahl's Law Bottleneck:** In `tencent/Hy-MT2-1.8B`, dense Self-Attention (34.16 ms), the 120,818-token LM Head (7.88 ms), and Layernorms/Embeddings (5.40 ms) account for **47.44 ms** of the 95.72 ms prefill time. When only feedforward MLPs are sparsified, even with 100% MLP compute removal, the theoretical model speedup cannot exceed $95.72 / 47.44 = 2.02\times$.
   - **Decode Overhead Bottleneck:** In PyTorch eager mode without cross-layer kernel compilation, single-token decode is dominated by over 300 sequential CUDA kernel launches per step (taking ~30 ms in Self-Attention/RoPE), completely masking the MLP speedup.

2. **Eliminating Misleading Metrics:**
   - Earlier documentations reported layer-level micro-benchmarks ($456\ \mu\text{s}$ to $1,854\ \mu\text{s}$ decode) and analytical extrapolations (e.g. $2.15\times$ decode / $3.36\times$ prefill) alongside full model metrics, causing confusion regarding true end-to-end LLM throughput.
   - All Hy-MT2 artifacts are preserved here for historical reference and scientific reproducibility while the main codebase is reset for clean, verified architectures.
