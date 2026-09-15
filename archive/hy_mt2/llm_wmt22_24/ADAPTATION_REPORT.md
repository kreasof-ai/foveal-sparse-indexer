# Foveal Sparsification Adaptation Report on WMT22-24

- **Base Model:** `tencent/Hy-MT2-1.8B`
- **Sparsified Layers:** ALL 32 of 32 layers (100% Uniform)
- **Attention Configuration:** Chunk Size 8, 4/16 active blocks per head (25.0% compute)
- **SwiGLU MLP Configuration:** Chunk Size 32, 48/192 active blocks (25.0% compute)
- **LM Head:** 100% Dense and Untouched
- **Optimizer:** `HybridMuonAdamW` (Muon lr=0.01, AdamW lr=0.0002)
- **Adaptation Steps:** 5 steps

## SacreBLEU Score Comparison

| Language Pair | Dense Teacher BLEU | Foveal Sparse BLEU | BLEU Retention |
| :--- | :--- | :--- | :--- |
| **`en-ja`** | 13.09 | 0.00 | **0.0%** |
| **`de-en`** | 25.45 | 0.00 | **0.0%** |
| **`ja-en`** | 10.28 | 0.00 | **0.0%** |
| **`en-zh`** | 41.74 | 0.00 | **0.0%** |
| **`zh-en`** | 50.81 | 0.00 | **0.0%** |
| **Overall** | **39.02** | **0.00** | **0.0%** |

