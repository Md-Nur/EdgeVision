# Walkthrough: Multi Crops Leaf Disease Detection Using Explainable AI (XAI)

An end-to-end, publication-grade research framework built in PyTorch following [`roadmap.md`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/roadmap.md). The project addresses dataset leakage, extreme class imbalance, transfer learning, and quantitative XAI validation across 29,191 images covering 15 disease classes in Papaya, Potato, and Rice.

---

## 1. Accomplishments & Methodology

### 1.1 Zero-Leakage Group Stratification
- **Problem Solved**: In the raw dataset, all 18,130 Papaya images were pre-augmented 5x (`image_X_aug_1` to `_aug_5` for 3,626 base images). A standard random split causes cross-split contamination and artificially inflated 99% accuracy.
- **Solution**: We implemented [`src/data/splitter.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/data/splitter.py) to extract base image IDs across all crops and execute group-stratified splitting (70% train, 15% val, 15% test).
- **Verification**: **Strictly 0 base ID overlap** confirmed between train, val, and test partitions:
  - **Train**: 20,415 images (10,275 unique base captures)
  - **Val**: 4,356 images (2,196 unique base captures)
  - **Test**: 4,420 images (2,216 unique base captures)
  - Fixed splits stored in [`data/splits/`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/data/splits/).

### 1.2 Imbalance-Aware Loss & Online Augmentation
- Implemented [`src/losses/focal_loss.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/losses/focal_loss.py) supporting:
  - **Effective Number of Samples Weighting** ($E_c = \frac{1 - \beta}{1 - \beta^{n_c}}$) to protect rare classes (e.g. Rice Healthy Leaf: 252 base images vs Potato Early Blight: 3,549 base images).
  - **Multi-class Focal Loss** ($\gamma = 2.0$) and **Label Smoothing** ($0.1$) to prevent overconfidence.
- PyTorch [`src/data/dataset.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/data/dataset.py) performs fresh online augmentations (random cropping, rotations, horizontal/vertical flips, color jitter) on training batches without duplicating near-identical images.

### 1.3 Two-Phase Transfer Learning Pipeline
- Model factory in [`src/models/backbones.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/models/backbones.py) supporting ImageNet-pretrained **EfficientNet-B0** and **ResNet-50** with dropout heads.
- Trainer in [`src/training/trainer.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/training/trainer.py) implements:
  - **Phase 1 (Warmup)**: Backbone frozen, classification head trained at LR $1\times 10^{-3}$.
  - **Phase 2 (Fine-tuning)**: Top 30% backbone layers unfrozen with discriminative learning rates ($2\times 10^{-5}$ for backbone, $2\times 10^{-4}$ for head) with Cosine Annealing.
  - **Checkpointing**: Tracks and saves the best model strictly on **Validation Macro-F1** (not biased accuracy).
  - Full Apple Silicon **Metal Performance Shaders (`mps`)** GPU acceleration.

### 1.4 Qualitative & Quantitative Explainable AI (XAI)
- Modules in [`src/xai/gradcam.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/xai/gradcam.py) and [`src/xai/quantitative.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/xai/quantitative.py) provide:
  - **Grad-CAM** and **Grad-CAM++** saliency extraction.
  - **Quantitative Localization Validation**: Evaluates the proportion of activation energy concentrated within the leaf foreground:
    \[
    \text{Energy Concentration} = \frac{\sum_{(i,j) \in \text{Leaf}} H(i, j)}{\sum_{(i,j)} H(i, j)}
    \]
  - **Pointing Game Accuracy**: Validates whether the peak saliency point $\arg\max H(i,j)$ lands on true pathology foreground.

---

## 2. Verification & Validation Results

### 2.1 Automated Test Suite
All 7 unit and integration tests passed (`pytest tests/ -v`):
- `tests/test_splits.py::test_extract_base_id_papaya_aug` -> **PASSED**
- `tests/test_splits.py::test_extract_base_id_rice_potato` -> **PASSED**
- `tests/test_splits.py::test_prepare_group_stratified_splits` -> **PASSED**
- `tests/test_pipeline.py::test_model_forward_and_freezing` -> **PASSED**
- `tests/test_pipeline.py::test_loss_functions` -> **PASSED**
- `tests/test_pipeline.py::test_metrics_computation` -> **PASSED**
- `tests/test_pipeline.py::test_gradcam_and_xai_metrics` -> **PASSED**

### 2.2 End-to-End Pipeline Smoke Test
Executed end-to-end two-phase training on a stratified subset, followed by full test evaluation on 4,420 images.

![Training Curves](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/training_curves.png)
Training and validation loss and macro-F1 curves recorded across epochs.

![Confusion Matrix](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/confusion_matrix.png)
15x15 normalized confusion matrix evaluated on the held-out test split.

### 2.3 Quantitative XAI Localization Results
Grad-CAM and Grad-CAM++ were evaluated across the 15 classes:

| Metric | Grad-CAM | Grad-CAM++ |
| :--- | :--- | :--- |
| **Mean Leaf Energy Concentration** | **67.42%** (±33.2%) | **71.90%** (±27.4%) |
| **Pointing Game Peak Accuracy** | **86.67%** | **93.33%** |

![XAI Visualization](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/xai_sample.png)
Sample Grad-CAM++ four-panel output: Input Leaf, Segmented Foreground Mask, Heatmap, and Overlay with quantitative concentration score.

---

## 3. Project Structure

```
Mukti/
├── configs/
│   └── config.yaml               # Centralized hyperparameters & paths
├── pyproject.toml                # Dependencies managed by uv
├── data/
│   ├── dataset.csv               # Original full CSV
│   └── splits/
│       ├── train.csv             # Leakage-free train split (20,415 rows)
│       ├── val.csv               # Leakage-free val split (4,356 rows)
│       └── test.csv              # Leakage-free test split (4,420 rows)
├── src/
│   ├── data/
│   │   ├── splitter.py           # Group-stratified split generator
│   │   └── dataset.py            # PyTorch Dataset, augmentation, and DataLoader
│   ├── models/
│   │   └── backbones.py          # EfficientNet-B0 / ResNet-50 with fine-tuning controls
│   ├── losses/
│   │   └── focal_loss.py         # Effective number weighting & Focal Loss
│   ├── training/
│   │   ├── trainer.py            # Two-phase fine-tuning engine
│   │   └── scheduler.py          # Cosine Annealing with warmup
│   ├── evaluation/
│   │   ├── metrics.py            # Top-1 Acc, Macro-F1, per-class recall
│   │   └── visualization.py      # Confusion matrix & curve plotters
│   ├── xai/
│   │   ├── gradcam.py            # Grad-CAM and Grad-CAM++ hooks & overlay
│   │   └── quantitative.py       # Energy concentration & pointing game metrics
│   └── utils/
│       ├── config.py             # YAML loader
│       └── device.py             # MPS/CUDA/CPU device selector & random seed
├── scripts/
│   ├── 01_prepare_splits.py      # Split preparation script
│   ├── 02_train.py               # Two-phase training script
│   ├── 03_evaluate.py            # Test evaluation script
│   └── 04_generate_xai.py        # Grad-CAM heatmap & quantitative metrics script
├── notebooks/
│   └── explainable_ai_leaf_disease.ipynb  # Interactive Explainable AI (XAI) notebook
├── tests/
│   ├── test_splits.py            # Group leakage unit tests
│   └── test_pipeline.py          # Model, loss, and XAI unit tests
└── main.py                       # Unified CLI runner
```

---

## 4. How to Run Full Experiments

All workflows can be executed via `main.py`:

1. **Re-generate or inspect data splits**:
   ```bash
   uv run python main.py split
   ```
2. **Train Full Model (Two-Phase Fine-Tuning)**:
   ```bash
   # Default: EfficientNet-B0 with class-weighted loss on MPS GPU
   uv run python main.py train --model efficientnet_b0

   # Or benchmark ResNet-50
   uv run python main.py train --model resnet50
   ```
3. **Evaluate on Held-Out Test Set**:
   ```bash
   uv run python main.py evaluate --checkpoint models/best_model.pth --split test
   ```
4. **Generate Explainable AI Visualizations & Metrics**:
   ```bash
   # Generate Grad-CAM heatmaps
   uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2

   # Generate Grad-CAM++ heatmaps
   uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2 --use-plusplus
   ```
5. **Run Test Suite**:
   ```bash
   uv run python main.py test
   ```




# Walkthrough: Multi Crops Leaf Disease Detection Using Explainable AI (XAI)

An end-to-end, publication-grade research framework built in PyTorch following [`roadmap.md`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/roadmap.md). The project addresses dataset leakage, extreme class imbalance, transfer learning, and quantitative XAI validation across 29,191 images covering 15 disease classes in Papaya, Potato, and Rice.

---

## 1. Accomplishments & Methodology

### 1.1 Zero-Leakage Group Stratification
- **Problem Solved**: In the raw dataset, all 18,130 Papaya images were pre-augmented 5x (`image_X_aug_1` to `_aug_5` for 3,626 base images). A standard random split causes cross-split contamination and artificially inflated 99% accuracy.
- **Solution**: We implemented [`src/data/splitter.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/data/splitter.py) to extract base image IDs across all crops and execute group-stratified splitting (70% train, 15% val, 15% test).
- **Verification**: **Strictly 0 base ID overlap** confirmed between train, val, and test partitions:
  - **Train**: 20,415 images (10,275 unique base captures)
  - **Val**: 4,356 images (2,196 unique base captures)
  - **Test**: 4,420 images (2,216 unique base captures)
  - Fixed splits stored in [`data/splits/`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/data/splits/).

### 1.2 Imbalance-Aware Loss & Online Augmentation
- Implemented [`src/losses/focal_loss.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/losses/focal_loss.py) supporting:
  - **Effective Number of Samples Weighting** ($E_c = \frac{1 - \beta}{1 - \beta^{n_c}}$) to protect rare classes (e.g. Rice Healthy Leaf: 252 base images vs Potato Early Blight: 3,549 base images).
  - **Multi-class Focal Loss** ($\gamma = 2.0$) and **Label Smoothing** ($0.1$) to prevent overconfidence.
- PyTorch [`src/data/dataset.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/data/dataset.py) performs fresh online augmentations (random cropping, rotations, horizontal/vertical flips, color jitter) on training batches without duplicating near-identical images.

### 1.3 Two-Phase Transfer Learning Pipeline
- Model factory in [`src/models/backbones.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/models/backbones.py) supporting ImageNet-pretrained **EfficientNet-B0** and **ResNet-50** with dropout heads.
- Trainer in [`src/training/trainer.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/training/trainer.py) implements:
  - **Phase 1 (Warmup)**: Backbone frozen, classification head trained at LR $1\times 10^{-3}$.
  - **Phase 2 (Fine-tuning)**: Top 30% backbone layers unfrozen with discriminative learning rates ($2\times 10^{-5}$ for backbone, $2\times 10^{-4}$ for head) with Cosine Annealing.
  - **Checkpointing**: Tracks and saves the best model strictly on **Validation Macro-F1** (not biased accuracy).
  - Full Apple Silicon **Metal Performance Shaders (`mps`)** GPU acceleration.

### 1.4 Qualitative & Quantitative Explainable AI (XAI)
- Modules in [`src/xai/gradcam.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/xai/gradcam.py) and [`src/xai/quantitative.py`](file:///Users/md.nurealamsiddiquee/Projects/Mukti/src/xai/quantitative.py) provide:
  - **Grad-CAM** and **Grad-CAM++** saliency extraction.
  - **Quantitative Localization Validation**: Evaluates the proportion of activation energy concentrated within the leaf foreground:
    \[
    \text{Energy Concentration} = \frac{\sum_{(i,j) \in \text{Leaf}} H(i, j)}{\sum_{(i,j)} H(i, j)}
    \]
  - **Pointing Game Accuracy**: Validates whether the peak saliency point $\arg\max H(i,j)$ lands on true pathology foreground.

---

## 2. Verification & Validation Results

### 2.1 Automated Test Suite
All 7 unit and integration tests passed (`pytest tests/ -v`):
- `tests/test_splits.py::test_extract_base_id_papaya_aug` -> **PASSED**
- `tests/test_splits.py::test_extract_base_id_rice_potato` -> **PASSED**
- `tests/test_splits.py::test_prepare_group_stratified_splits` -> **PASSED**
- `tests/test_pipeline.py::test_model_forward_and_freezing` -> **PASSED**
- `tests/test_pipeline.py::test_loss_functions` -> **PASSED**
- `tests/test_pipeline.py::test_metrics_computation` -> **PASSED**
- `tests/test_pipeline.py::test_gradcam_and_xai_metrics` -> **PASSED**

### 2.2 End-to-End Pipeline Smoke Test
Executed end-to-end two-phase training on a stratified subset, followed by full test evaluation on 4,420 images.

![Training Curves](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/training_curves.png)
Training and validation loss and macro-F1 curves recorded across epochs.

![Confusion Matrix](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/confusion_matrix.png)
15x15 normalized confusion matrix evaluated on the held-out test split.

### 2.3 Quantitative XAI Localization Results
Grad-CAM and Grad-CAM++ were evaluated across the 15 classes:

| Metric | Grad-CAM | Grad-CAM++ |
| :--- | :--- | :--- |
| **Mean Leaf Energy Concentration** | **67.42%** (±33.2%) | **71.90%** (±27.4%) |
| **Pointing Game Peak Accuracy** | **86.67%** | **93.33%** |

![XAI Visualization](/Users/md.nurealamsiddiquee/.gemini/antigravity/brain/00757977-c8a0-40fa-adb0-672422526c24/xai_sample.png)
Sample Grad-CAM++ four-panel output: Input Leaf, Segmented Foreground Mask, Heatmap, and Overlay with quantitative concentration score.

---

## 3. Project Structure

```
Mukti/
├── configs/
│   └── config.yaml               # Centralized hyperparameters & paths
├── pyproject.toml                # Dependencies managed by uv
├── data/
│   ├── dataset.csv               # Original full CSV
│   └── splits/
│       ├── train.csv             # Leakage-free train split (20,415 rows)
│       ├── val.csv               # Leakage-free val split (4,356 rows)
│       └── test.csv              # Leakage-free test split (4,420 rows)
├── src/
│   ├── data/
│   │   ├── splitter.py           # Group-stratified split generator
│   │   └── dataset.py            # PyTorch Dataset, augmentation, and DataLoader
│   ├── models/
│   │   └── backbones.py          # EfficientNet-B0 / ResNet-50 with fine-tuning controls
│   ├── losses/
│   │   └── focal_loss.py         # Effective number weighting & Focal Loss
│   ├── training/
│   │   ├── trainer.py            # Two-phase fine-tuning engine
│   │   └── scheduler.py          # Cosine Annealing with warmup
│   ├── evaluation/
│   │   ├── metrics.py            # Top-1 Acc, Macro-F1, per-class recall
│   │   └── visualization.py      # Confusion matrix & curve plotters
│   ├── xai/
│   │   ├── gradcam.py            # Grad-CAM and Grad-CAM++ hooks & overlay
│   │   └── quantitative.py       # Energy concentration & pointing game metrics
│   └── utils/
│       ├── config.py             # YAML loader
│       └── device.py             # MPS/CUDA/CPU device selector & random seed
├── scripts/
│   ├── 01_prepare_splits.py      # Split preparation script
│   ├── 02_train.py               # Two-phase training script
│   ├── 03_evaluate.py            # Test evaluation script
│   └── 04_generate_xai.py        # Grad-CAM heatmap & quantitative metrics script
├── notebooks/
│   └── explainable_ai_leaf_disease.ipynb  # Interactive Explainable AI (XAI) notebook
├── tests/
│   ├── test_splits.py            # Group leakage unit tests
│   └── test_pipeline.py          # Model, loss, and XAI unit tests
└── main.py                       # Unified CLI runner
```

---

## 4. How to Run Full Experiments

All workflows can be executed via `main.py`:

1. **Re-generate or inspect data splits**:
   ```bash
   uv run python main.py split
   ```
2. **Train Full Model (Two-Phase Fine-Tuning)**:
   ```bash
   # Default: EfficientNet-B0 with class-weighted loss on MPS GPU
   uv run python main.py train --model efficientnet_b0

   # Or benchmark ResNet-50
   uv run python main.py train --model resnet50
   ```
3. **Evaluate on Held-Out Test Set**:
   ```bash
   uv run python main.py evaluate --checkpoint models/best_model.pth --split test
   ```
4. **Generate Explainable AI Visualizations & Metrics**:
   ```bash
   # Generate Grad-CAM heatmaps
   uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2

   # Generate Grad-CAM++ heatmaps
   uv run python main.py xai --checkpoint models/best_model.pth --samples-per-class 2 --use-plusplus
   ```
5. **Run Test Suite**:
   ```bash
   uv run python main.py test
   ```
