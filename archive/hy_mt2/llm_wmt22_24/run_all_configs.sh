#!/usr/bin/env bash
# ==============================================================================
# Full Multi-Config Training Sweep for Foveal Sparse Hy-MT2-1.8B on WMT22-24
# Uniform 32 Layers, 100% Dense LM Head, HybridMuonAdamW Optimizer
# ==============================================================================

set -e
mkdir -p experiments/llm_wmt22_24/logs

LOG_FILE="experiments/llm_wmt22_24/logs/sweep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================================="
echo "Starting Foveal Sparsification Training Sweep across All Attention Configs"
echo "Target Model: tencent/Hy-MT2-1.8B (Uniform 32 Layers, 100% Dense LM Head)"
echo "Dataset:      Rexhaif/wmt22-24 (~100,000 train rows, 120 lean val, 2000 test)"
echo "Batch Size:   Micro-batch 4 * GradAccum 16 => GLOBAL BATCH SIZE 64"
echo "Optimizer:    HybridMuonAdamW"
echo "Log file:     $LOG_FILE"
echo "=========================================================================="

CONFIGS=(
    "8 48 REPORT_attn8_mlp48.md 50.0%_Attn_25%_MLP"
    "4 48 REPORT_attn4_mlp48.md 25.0%_Attn_25%_MLP"
    "2 48 REPORT_attn2_mlp48.md 12.5%_Attn_25%_MLP"
    "1 48 REPORT_attn1_mlp48.md  6.2%_Attn_25%_MLP"
)

for CFG in "${CONFIGS[@]}"; do
    read -r ATTN_BLKS MLP_BLKS REPORT_NAME DESC <<< "$CFG"
    echo ""
    echo "=========================================================================="
    echo ">>> Running Configuration: $DESC"
    echo "    Attention: $ATTN_BLKS/16 blocks per head (Chunk 8)"
    echo "    SwiGLU:    $MLP_BLKS/192 blocks (Chunk 32)"
    echo "    Report:    experiments/llm_wmt22_24/$REPORT_NAME"
    echo "=========================================================================="

    python3 experiments/llm_wmt22_24/train_adaptation.py \
        --num_layers 32 \
        --active_blocks_attn "$ATTN_BLKS" \
        --active_blocks_mlp "$MLP_BLKS" \
        --max_train 100000 \
        --max_steps 3000 \
        --batch_size 4 \
        --grad_accum_steps 16 \
        --eval_interval 250 \
        --eval_dense_samples 60 \
        --max_eval_samples 500 \
        --lr_muon 0.01 \
        --lr_adamw 2e-4 \
        --output_report "experiments/llm_wmt22_24/$REPORT_NAME"

    echo "Finished $DESC successfully!"
done

echo ""
echo "=========================================================================="
echo "All configurations completed successfully!"
echo "Generated Reports:"
ls -lh experiments/llm_wmt22_24/REPORT_*.md
echo "=========================================================================="
