# Multi-Crop Leaf Disease Detection Using Explainable AI (XAI)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware Acceleration](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)%20%7C%20CUDA-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, publication-grade research framework for 15-class leaf disease diagnosis across **Papaya**, **Potato**, and **Rice**. Designed to address real-world research flaws identified in existing agricultural vision literature—specifically **data leakage from pre-augmented images**, **extreme class imbalance**, and **unvalidated "qualitative-only" Explainable AI (XAI)**.

---

## Table of Contents

- [Overview & Research Motivation](#overview--research-motivation)
- [Key Innovations](#key-innovations)
  - [1. Zero-Leakage Group Stratification](#1-zero-leakage-group-stratification)
  - [2. Imbalance-Aware Loss & Online Augmentation](#2-imbalance-aware-loss--online-augmentation)
  - [3. Two-Phase Transfer Learning](#3-two-phase-transfer-learning)
  - [4. Quantitative Explainable AI (XAI) Validation](#4-quantitative-explainable-ai-xai-validation)
- [Dataset Summary](#dataset-summary)
- [Repository Structure](#repository-structure)
- [Installation & Environment](#installation--environment)
- [Quick Start Guide](#quick-start-guide)
  - [1. Leakage-Free Dataset Splitting](#1-leakage-free-dataset-splitting)
  - [2. Model Training](#2-model-training)
  - [3. Comprehensive Evaluation](#3-comprehensive-evaluation)
  - [4. Explainable AI & Quantitative Localization](#4-explainable-ai--quantitative-localization)
  - [5. Automated Test Suite](#5-automated-test-suite)
- [Methodology & Formulations](#methodology--formulations)
  - [Class-Balanced Loss](#class-balanced-loss)
  - [Discriminative Fine-Tuning](#discriminative-fine-tuning)
  - [Quantitative XAI Localization](#quantitative-xai-localization)
- [Experimental Results](#experimental-results)
- [License & Citation](#license--citation)

---

## Overview & Research Motivation

Automated plant disease diagnosis via Deep Learning frequently suffers from methodological flaws that lead to inflated benchmark scores and poor real-world generalizability:

1. **The Pre-Augmentation Leakage Trap**: In many public datasets, images are pre-augmented offline. When split naively into train/test sets by row or filename, near-identical augmented variants of the same physical leaf photograph leak across splits. Models memorize the base photograph rather than learning pathology features, generating artificial ~99% test accuracies.
2. **Severe Class Imbalance**: Real-world agricultural collections exhibit orders-of-magnitude imbalance (e.g., 3,549 images in Potato Early Blight vs. 252 images in Rice Healthy Leaf). Standard Cross-Entropy produces models that sacrifice minority classes while maintaining high overall accuracy.
3. **Qualitative-Only Explainability**: Published plant pathology papers predominantly display illustrative Grad-CAM cherry-picked heatmaps without quantitative validation, making it impossible to verify whether the network genuinely attends to lesion pathology or spurious background artifacts.

**Mukti** resolves all three challenges systematically.

---

## Key Innovations

### 1. Zero-Leakage Group Stratification
- Automatically detects and strips augmentation suffixes (`_aug_\d+`) for Papaya images, mapping all 18,130 augmented files back to their **3,626 true physical leaf captures**.
- Executes group-aware stratified 70/15/15 partitioning such that **all variants of a given physical leaf remain strictly within one partition**.
- Guarantees **0.0% base image leakage** between train, validation, and test splits while preserving exact class distributions across all 15 categories.

### 2. Imbalance-Aware Loss & Online Augmentation
- Implements Class-Balanced Loss using the **Effective Number of Samples** weighting scheme ($E_c = \frac{1 - \beta}{1 - \beta^{n_c}}$) combined with **Focal Loss** ($\gamma = 2.0$) and **Label Smoothing** ($0.1$).
- Replaces static offline augmentation with **dynamic online augmentations** (random cropping, rotations, vertical/horizontal flips, color jitter) applied only to training mini-batches on the fly.

### 3. Two-Phase Transfer Learning
- Benchmarks ImageNet-pretrained **EfficientNet-B0** and **ResNet-50** backbones.
- **Phase 1 (Warmup)**: Backbone frozen; only the classification head is trained at a moderate learning rate ($1\times 10^{-3}$) to adapt to the 15-class target distribution.
- **Phase 2 (Discriminative Fine-Tuning)**: Top 30% of backbone layers are unfrozen and trained end-to-end with discriminative learning rates ($2\times 10^{-5}$ for backbone, $2\times 10^{-4}$ for head) under a Cosine Annealing scheduler.
- Checkpointing and early stopping are governed strictly by **Validation Macro-F1**, never biased accuracy.

### 4. Quantitative Explainable AI (XAI) Validation
- Implements **Grad-CAM** and **Grad-CAM++** with multi-panel visual overlays.
- Introduces **Energy Concentration Ratio** ($ECR$) to quantitatively evaluate how much saliency energy falls within segmented leaf foreground versus background:
  $$\text{Energy Concentration} = \frac{\sum_{(i,j) \in \text{Leaf}} H(i, j)}{\sum_{(i,j)} H(i, j)}$$
- Evaluates **Pointing Game Accuracy** to test whether the peak activation $\arg\max_{(i,j)} H(i, j)$ lands squarely on true leaf pathology.

---

## Dataset Summary

The dataset comprises **29,191 images** across **3 crops** and **15 classes**:

| Crop | Class Name | Total Images | Unique Base Captures | Offline Augmented? |
| :--- | :--- | :--- | :--- | :--- |
| **Papaya** | `Papaya_Anthracnose` | 1,150 | 230 | Yes (5x) |
| **Papaya** | `Papaya_BacterialSpot` | 1,070 | 214 | Yes (5x) |
| **Papaya** | `Papaya_Curl` | 3,890 | 778 | Yes (5x) |
| **Papaya** | `Papaya_Healthy` | 2,970 | 594 | Yes (5x) |
| **Papaya** | `Papaya_Mealybug` | 910 | 182 | Yes (5x) |
| **Papaya** | `Papaya_Mite disease` | 2,760 | 552 | Yes (5x) |
| **Papaya** | `Papaya_Mosaic` | 2,730 | 546 | Yes (5x) |
| **Papaya** | `Papaya_Ringspot` | 2,650 | 530 | Yes (5x) |
| **Potato** | `Potato_Potato Early blight` | 3,549 | 3,549 | No (1x) |
| **Potato** | `Potato_Potato Healthy` | 2,432 | 2,432 | No (1x) |
| **Potato** | `Potato_Potato Late blight` | 3,521 | 3,521 | No (1x) |
| **Rice** | `Rice_Bacterial Leaf Blight` | 421 | 421 | No (1x) |
| **Rice** | `Rice_Brown Spot` | 356 | 356 | No (1x) |
| **Rice** | `Rice_Healthy Leaf` | 252 | 252 | No (1x) |
| **Rice** | `Rice_Tungro Virus` | 530 | 530 | No (1x) |
| **Total** | **15 Classes** | **29,191** | **14,687** | — |

---

## Repository Structure

```
Mukti/
├── configs/
│   └── config.yaml               # Centralized hyperparameters, model configs & paths
├── pyproject.toml                # Dependencies managed with uv
├── data/
│   ├── dataset.csv               # Raw image catalog
│   ├── MyDataset/                # Image folder structure (Papaya, Potato, Rice)
│   └── splits/                   # Deterministic, leakage-free partitions
│       ├── train.csv             # Train split (20,415 images, 10,275 base captures)
│       ├── val.csv               # Validation split (4,356 images, 2,196 base captures)
│       └── test.csv              # Held-out test split (4,420 images, 2,216 base captures)
├── src/
│   ├── data/
│   │   ├── splitter.py           # Group-aware stratified splitting engine
│   │   └── dataset.py            # PyTorch Dataset, transforms & DataLoader factory
│   ├── models/
│   │   └── backbones.py          # Transfer learning architectures (EfficientNet, ResNet)
│   ├── losses/
│   │   └── focal_loss.py         # Effective number weighting & multi-class Focal Loss
│   ├── training/
│   │   ├── trainer.py            # Two-phase fine-tuning engine with early stopping
│   │   └── scheduler.py          # Cosine Annealing with warmup
│   ├── evaluation/
│   │   ├── metrics.py            # Macro-F1, per-class recall, precision, and accuracy
│   │   └── visualization.py      # Confusion matrix & learning curve visualizers
│   ├── xai/
│   │   ├── gradcam.py            # Grad-CAM and Grad-CAM++ hook implementations
│   │   └── quantitative.py       # Energy concentration & pointing game localization
│   └── utils/
│       ├── config.py             # YAML loader
│       └── device.py             # MPS / CUDA / CPU device resolver & seed manager
├── scripts/
│   ├── 01_prepare_splits.py      # Generate and verify leakage-free splits
│   ├── 02_train.py               # Two-phase training entry point
│   ├── 03_evaluate.py            # Test set evaluation & confusion matrix generator
│   └── 04_generate_xai.py        # Heatmap generation and quantitative XAI evaluation
├── tests/
│   ├── test_splits.py            # Unit tests verifying zero base ID leakage
│   └── test_pipeline.py          # Unit tests for models, loss functions, and XAI
└── main.py                       # Unified CLI runner
```

---

## Installation & Environment

This repository uses [`uv`](https://docs.astral.sh/uv/) for high-speed, reproducible Python package management.

### Prerequisites
- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)

### Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd Mukti

# Sync virtual environment and install all dependencies
uv sync
```

---

## Quick Start Guide

The framework provides a unified command-line entry point via `main.py`.

### 1. Leakage-Free Dataset Splitting
Generate the group-stratified splits and verify zero base-ID contamination:
```bash
uv run python main.py split
```
Outputs:
- `data/splits/train.csv` (20,415 samples)
- `data/splits/val.csv` (4,356 samples)
- `data/splits/test.csv` (4,420 samples)

### 2. Model Training
Train an ImageNet-pretrained model with two-phase transfer learning:
```bash
# Train EfficientNet-B0 (default) with class-weighted loss on Apple Silicon GPU
uv run python main.py train --model efficientnet_b0

# Train ResNet-50
uv run python main.py train --model resnet50

# Custom hyperparameter overrides
uv run python main.py train --model efficientnet_b0 --epochs-p1 3 --epochs-p2 15 --batch-size 32 --loss-type weighted_ce
```
Outputs:
- Best checkpoint saved to `models/best_model.pth`
- Last checkpoint saved to `models/last_model.pth`
- Loss & Macro-F1 curves saved to `results/efficientnet_b0_training_curves.png`

### 3. Comprehensive Evaluation
Evaluate the trained checkpoint on the held-out test split:
```bash
uv run python main.py evaluate --checkpoint models/best_model.pth --split test
```
Outputs:
- Full per-class classification report printed to terminal
- Confusion matrix image saved to `results/efficientnet_b0_test_confusion_matrix.png`
- Detailed CSV report saved to `results/efficientnet_b0_test_classification_report.csv`

### 4. Explainable AI & Quantitative Localization
Generate Grad-CAM or Grad-CAM++ saliency heatmaps with quantitative metrics:
```bash
# Grad-CAM (vanilla)
uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2

# Grad-CAM++ (higher-order gradient weighting)
uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2 --use-plusplus
```
Outputs:
- Four-panel explanatory visualizations for each class in `results/xai/{class_name}/`
- Aggregated metrics saved to `results/xai/quantitative_xai_metrics.json`
- Per-sample scores saved to `results/xai/per_sample_xai_metrics.csv`

### 5. Automated Test Suite
Verify model forward passes, loss formulations, metric calculations, and split integrity:
```bash
uv run python main.py test
```

---

## Methodology & Formulations

### Class-Balanced Loss
To prevent dominant categories from overwhelming gradients, we compute class weights based on the effective number of samples (Cui et al., CVPR 2019):

$$E_c = \frac{1 - \beta}{1 - \beta^{n_c}}, \quad w_c = \frac{C \cdot E_c}{\sum_{k=1}^C E_k}$$

where $\beta = 0.999$, $C = 15$, and $n_c$ is the number of training samples for class $c$.

The class weights are incorporated into Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$, and label smoothing ($0.1$) is applied to mitigate overconfidence on ambiguous foliar lesions.

### Discriminative Fine-Tuning
During Phase 2, learning rates are stratified across the model hierarchy:
- Pretrained backbone feature layers: $\eta_{\text{backbone}} = 2 \times 10^{-5}$
- Custom classification head: $\eta_{\text{head}} = 2 \times 10^{-4}$

Both parameter groups decay under Cosine Annealing:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

### Quantitative XAI Localization
Unlike existing studies that stop at subjective visual inspection, we evaluate localization quantitatively:

1. **Leaf Foreground Mask**: Extracted using the Excess Green Index ($2G - R - B$) combined with color contrast filtering to isolate foliar tissue and necrotic lesions from background.
2. **Energy Concentration Ratio**: Quantifies the percentage of CAM activation energy focused on actual leaf tissue:
   $$\text{Energy Concentration} = \frac{\sum_{(i,j) \in M_{\text{leaf}}} H(i, j)}{\sum_{(i,j)} H(i, j)}$$
3. **Pointing Game Hit**: Returns $1.0$ if the maximum saliency point $\arg\max H(i,j)$ resides inside the leaf foreground, otherwise $0.0$.

---

## Experimental Results

### XAI Localization Performance
Evaluated across all 15 crop disease categories on the held-out test partition:

| Metric | Grad-CAM | Grad-CAM++ |
| :--- | :--- | :--- |
| **Mean Leaf Energy Concentration** | **67.42%** ($\pm 33.2\%$) | **71.90%** ($\pm 27.4\%$) |
| **Pointing Game Peak Accuracy** | **86.67%** | **93.33%** |

*Grad-CAM++ demonstrates superior energy focus on multi-lesion and diffuse foliar symptoms.*

---

## License & Citation

This project is licensed under the MIT License - see the LICENSE file for details.

If you use this codebase or methodology in your research, please cite:
```bibtex
@misc{mukti_xai_2026,
  title={Multi crops leaf disease detection using Explainable AI (XAI)},
  author={Md. Nure Alam Siddiquee and Collaborators},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/md-nur/EdgeVision}}
}
```
