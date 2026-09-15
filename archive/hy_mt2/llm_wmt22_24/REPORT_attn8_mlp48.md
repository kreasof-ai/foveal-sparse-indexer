# Foveal Sparsification Adaptation Report on WMT22-24

- **Base Model:** `tencent/Hy-MT2-1.8B`
- **Sparsified Layers:** ALL 32 of 32 layers (100% Uniform)
- **Attention Configuration:** Chunk Size 8, 8/16 active blocks per head (50.0% compute)
- **SwiGLU MLP Configuration:** Chunk Size 32, 48/192 active blocks (25.0% compute)
- **LM Head:** 100% Dense and Untouched
- **Optimizer:** `HybridMuonAdamW` (Muon lr=0.01, AdamW lr=0.0002)
- **Adaptation Steps:** 3000 steps

## SacreBLEU Score Comparison

| Language Pair | Dense Teacher BLEU | Foveal Sparse BLEU | BLEU Retention |
| :--- | :--- | :--- | :--- |
| **`de-en`** | 30.94 | 0.00 | **0.0%** |
| **`en-ru`** | 22.38 | 0.03 | **0.1%** |
| **`en-ja`** | 18.45 | 0.00 | **0.0%** |
| **`en-zh`** | 36.01 | 0.00 | **0.0%** |
| **`zh-en`** | 15.50 | 0.00 | **0.0%** |
| **`en-es`** | 10.14 | 0.00 | **0.0%** |
| **`ja-en`** | 11.64 | 0.00 | **0.0%** |
| **`en-de`** | 27.77 | 0.05 | **0.2%** |
| **Overall** | **20.86** | **0.00** | **0.0%** |

