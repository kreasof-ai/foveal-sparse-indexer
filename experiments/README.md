# Foveal Sparsification Experiments & Reports

This directory houses domain-specific empirical sparsification experiments, benchmark drivers, evaluation scripts, and official markdown validation reports.

---

## Experiment Catalog

| Domain Directory | Target Architecture / Task | Key Highlights & Validation | Official Report |
| :--- | :--- | :--- | :--- |
| [**`yolo11_coco/`**](yolo11_coco/) | `ultralytics/yolo11x` (COCO val2017) | **4.03× inference speedup** (32.80 ms vs 132.18 ms, 487.8 img/s), **95.1% mAP retention** (51.3% vs 54.5%) on full COCO val2017. | [`YOLO11X_SPARSIFICATION_REPORT.md`](yolo11_coco/YOLO11X_SPARSIFICATION_REPORT.md) |
| [**`vit_imagenet/`**](vit_imagenet/) | `timm/eva02_base_patch14_448` (ImageNet-1k) | **2.25× inference speedup** (105.8 vs 47.0 img/s), **99.5% accuracy retention** (88.23% vs 88.69% Top-1) on all 50,000 validation images. | [`SPARSIFICATION_REPORT.md`](vit_imagenet/SPARSIFICATION_REPORT.md) & [`IMAGENET_50K_REPORT.md`](vit_imagenet/IMAGENET_50K_REPORT.md) |
| [**`cifar10/`**](cifar10/) | Vision Transformer & Tokenformer (CIFAR-10) | **2.21× faster training to 90% accuracy** (57.02s vs 126.29s), **51.7% less peak VRAM**, **+2.03% higher accuracy** (93.8% sparse). | [`SPEEDRUN_REPORT.md`](cifar10/SPEEDRUN_REPORT.md) |

*(Note: Prior experiments on `tencent/Hy-MT2-1.8B` including WMT22-24 adaptation and Pattention failed to meet end-to-end LLM quality and speedup targets; all Hy-MT2 scripts and reports have been archived to [`archive/hy_mt2/`](../archive/hy_mt2/)).*

---

## Reproduction Commands

### 1. YOLO11x Object Detection on COCO (`experiments/yolo11_coco/`)
```bash
# Unit tests for YOLO sparsification modules
PYTHONPATH=. pytest tests/test_yolo_sparsify.py

# Full COCO val2017 sparsification and report generation
python3 experiments/yolo11_coco/sparsify_yolo11x_coco.py --adapt_steps 25
```

### 2. Vision Transformer on ImageNet-1k (`experiments/vit_imagenet/`)
```bash
# Practical ViT sparsification with offline SVD router
python3 experiments/vit_imagenet/sparsify_timm_imagenet.py --num_val 500 --adapt_steps 25

# Full 50,000 ImageNet validation evaluation
python3 experiments/vit_imagenet/evaluate_all_50k_imagenet.py
```

### 3. CIFAR-10 Speedrun (`experiments/cifar10/`)
```bash
# CIFAR-10 speedrun benchmark: Optimized Dense vs Foveal Sparse
python3 experiments/cifar10/speedrun_cifar10_dense_vs_sparse.py --target-acc 90.0

# Regimes where Foveal Sparse is strictly faster and more accurate
python3 experiments/cifar10/train_cifar10_faster_and_better.py --regime tokenformer --epochs 4

# Full CIFAR-10 ViT training (>90% sparsity)
python3 experiments/cifar10/train_cifar10_10m.py
```
python3 experiments/cifar10/train_cifar10_10m.py --epochs 5 --batch-size 128
```
