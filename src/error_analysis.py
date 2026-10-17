"""
Error Analysis for Multi-Crop Leaf Disease Classifiers.

Models analysed (dedup test split, seed 42):
  - resnet18
  - mobilenetv3_small
  - custom_none
  - custom_se
  (and custom_cbam where custom models are analyzed)

Extended analyses:
  1. 3x3 crop confusion matrix (Papaya, Potato, Rice), number of cross-crop errors
     in flat 12-class prediction, and full list of those images in reports/cross_crop_errors.csv.
  2. Crop-masked inference macro-F1 vs flat macro-F1 for custom models with crop heads.
  3. Group-level error analysis: for each class, count of distinct group_ids with at least
     one error vs total group_ids in the test split.
  4. Montage PNG in reports/montages/ of every distinct image (deduplicated by group_id)
     in the 20 most confident errors of custom_se plus all cross-crop errors, labeled with
     true class, predicted class, confidence, and file name.
  5. Per-class precision / recall / F1 with group-bootstrap 95% CIs.
  6. Top-10 most confused class pairs.

Usage:
  PYTHONPATH=. .venv/bin/python src/error_analysis.py
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# ── project imports ─────────────────────────────────────────────────────────
from src.data.dataset import CropDiseaseDataset, get_transforms
from src.models.backbones import create_model
from src.models.custom_net import build_custom_net, CROP_TO_CLASSES, CLASS_TO_CROP

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
MONTAGES_DIR = REPORTS_DIR / "montages"
CHECKPOINTS_DIR = Path("checkpoints")
SPLITS_DIR = Path("splits_cache/dedup") if Path("splits_cache/dedup").exists() else Path("splits/dedup")
DATA_ROOT = Path("data_cache") if Path("data_cache").exists() else Path("data")

# Maps short names → result JSON paths
MODEL_JSON: Dict[str, str] = {
    "resnet18":          "results/resnet18_seed42.json",
    "mobilenetv3_small": "results/mobilenetv3_small_seed42.json",
    "custom_none":       "results/custom_none_seed42.json",
    "custom_se":         "results/custom_se_seed42.json",
}

# Crop name mapping
CROP_NAMES = ["Papaya", "Potato", "Rice"]
CLASS_TO_CROP_IDX = {
    0: 0, 1: 0, 2: 0, 3: 0,     # Papaya
    4: 1, 5: 1, 6: 1, 7: 1,     # Potato
    8: 2, 9: 2, 10: 2, 11: 2,   # Rice
}
CLASS_TO_CROP_NAME = {idx: CROP_NAMES[crop_idx] for idx, crop_idx in CLASS_TO_CROP_IDX.items()}

CROP_TO_CLASS_INDICES = {
    "Papaya": [0, 1, 2, 3],
    "Potato": [4, 5, 6, 7],
    "Rice":   [8, 9, 10, 11],
}

N_BOOTSTRAP = 2000
BOOTSTRAP_ALPHA = 0.05  # 95% CI
N_CONFIDENT_ERRORS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Reproducible bootstrap sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Determine fastest available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_image_path(raw_path: str) -> Path:
    """Resolve image path checking both data_cache and data root."""
    p = Path(raw_path)
    if p.exists():
        return p
    clean = str(raw_path).replace("\\", "/")
    if clean.startswith("data_cache/"):
        alt = Path("data") / clean[len("data_cache/"):]
        if alt.exists():
            return alt
    elif clean.startswith("data/"):
        alt = Path("data_cache") / clean[len("data/"):]
        if alt.exists():
            return alt
    return p


def load_predictions(model_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load y_true, y_pred, group_ids, and class_names from a saved result JSON.

    Returns:
        y_true      : (N,) int array of ground-truth labels
        y_pred      : (N,) int array of predicted labels
        group_ids   : (N,) array of group IDs for block-bootstrap
        class_names : ordered list of 12 class names
    """
    json_path = MODEL_JSON[model_key]
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    dt = result["dedup_test"]
    y_true = np.array(dt["y_true"], dtype=int)
    y_pred = np.array(dt["y_pred"], dtype=int)
    group_ids = np.array(dt["group_ids"]) if dt.get("group_ids") is not None else np.arange(len(y_true))
    class_names: List[str] = result["class_names"]
    return y_true, y_pred, group_ids, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Model Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_model_inference(
    model_name: str,
    test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Load saved checkpoint and run inference over test split.
    Returns dictionary with:
      - 'probs': (N, 12) softmax probabilities
      - 'preds': (N,) flat disease predictions
      - 'pred_crops': (N,) predicted crop indices (0..2)
      - 'masked_preds': (N,) crop-masked disease predictions (custom models only)
      - 'crop_logits': (N, 3) or None
      - 'disease_logits': (N, 12)
      - 'paths': list of image paths
    """
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_seed42_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model architecture
    is_custom = "custom" in model_name
    if is_custom:
        att = model_name.replace("custom_", "")
        model = build_custom_net(attention=att, dropout=0.3).to(device)
    else:
        model = create_model(backbone_name=model_name, num_classes=12, pretrained=False).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Pretrained models evaluated against raw data; custom models evaluated against data_cache
    if is_custom and Path("data_cache").exists():
        img_root = Path("data_cache")
        model_test_df = pd.read_csv("splits_cache/dedup/test.csv") if Path("splits_cache/dedup/test.csv").exists() else test_df
    else:
        img_root = Path("data")
        model_test_df = pd.read_csv("splits/dedup/test.csv") if Path("splits/dedup/test.csv").exists() else test_df

    tfms = get_transforms(image_size=224, resize_dim=256)["test"]
    dataset = CropDiseaseDataset(model_test_df, image_root=img_root, transform=tfms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: List[np.ndarray] = []
    all_preds: List[int] = []
    all_pred_crops: List[int] = []
    all_masked_preds: List[int] = []
    all_paths: List[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        paths = batch["image_path"]

        if is_custom:
            crop_logits, disease_logits = model(images)
            pred_crops, flat_preds, masked_preds = model.predict_with_masking(crop_logits, disease_logits)
            probs = F.softmax(disease_logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(flat_preds.cpu().numpy().tolist())
            all_pred_crops.extend(pred_crops.cpu().numpy().tolist())
            all_masked_preds.extend(masked_preds.cpu().numpy().tolist())
        else:
            disease_logits = model(images)
            probs = F.softmax(disease_logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            pred_crops = np.array([CLASS_TO_CROP_IDX[p] for p in preds])
            all_probs.extend(probs)
            all_preds.extend(preds.tolist())
            all_pred_crops.extend(pred_crops.tolist())

        all_paths.extend(paths)

    return {
        "probs": np.array(all_probs),
        "preds": np.array(all_preds),
        "pred_crops": np.array(all_pred_crops),
        "masked_preds": np.array(all_masked_preds) if is_custom else None,
        "paths": all_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. 3x3 Crop Confusion Matrix & Cross-Crop Errors
# ─────────────────────────────────────────────────────────────────────────────

def compute_crop_confusion_and_cross_crop_errors(
    models: List[str],
    inference_results: Dict[str, Dict[str, Any]],
    test_df: pd.DataFrame,
    class_names: List[str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """
    For resnet18, mobilenetv3_small, custom_none, custom_se:
    1. 3x3 crop confusion matrix.
    2. Count of cross-crop errors in flat 12-class prediction.
    3. List of those images saved to reports/cross_crop_errors.csv.
    """
    y_true_disease = test_df["label"].values
    y_true_crop = np.array([CLASS_TO_CROP_IDX[y] for y in y_true_disease])
    group_ids = test_df["group_id"].values
    image_paths = test_df["image_path"].values

    crop_cms: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, Any]] = []
    cross_crop_records: List[Dict[str, Any]] = []

    for m in models:
        preds = inference_results[m]["preds"]
        probs = inference_results[m]["probs"]
        # In flat 12-class prediction, the predicted crop is derived from the predicted disease class
        pred_crops_flat = np.array([CLASS_TO_CROP_IDX[p] for p in preds])

        # 3x3 crop confusion matrix
        cm = confusion_matrix(y_true_crop, pred_crops_flat, labels=[0, 1, 2])
        crop_cms[m] = cm

        # Cross-crop errors: off-diagonal where true_crop != pred_crop
        cross_mask = y_true_crop != pred_crops_flat
        num_cross = int(cross_mask.sum())
        total_errors = int((y_true_disease != preds).sum())
        cross_pct_of_errors = (num_cross / max(1, total_errors)) * 100
        cross_pct_of_test = (num_cross / len(test_df)) * 100

        summary_rows.append({
            "model":                m,
            "total_test_samples":   len(test_df),
            "total_errors":         total_errors,
            "cross_crop_errors":    num_cross,
            "cross_crop_pct_errs":  round(cross_pct_of_errors, 2),
            "cross_crop_pct_test":  round(cross_pct_of_test, 2),
        })

        cross_indices = np.where(cross_mask)[0]
        for idx in cross_indices:
            p_class = preds[idx]
            conf = float(probs[idx, p_class])
            cross_crop_records.append({
                "model":        m,
                "image_path":   image_paths[idx],
                "file_name":    Path(image_paths[idx]).name,
                "group_id":     int(group_ids[idx]),
                "true_crop":    CROP_NAMES[y_true_crop[idx]],
                "true_class":   class_names[y_true_disease[idx]],
                "pred_crop":    CROP_NAMES[pred_crops_flat[idx]],
                "pred_class":   class_names[p_class],
                "confidence":   round(conf, 6),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_records = pd.DataFrame(cross_crop_records)

    # Save to reports/cross_crop_errors.csv
    csv_path = REPORTS_DIR / "cross_crop_errors.csv"
    df_records.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path} ({len(df_records)} cross-crop error records)")

    return crop_cms, df_summary, df_records


# ─────────────────────────────────────────────────────────────────────────────
# 2. Crop-Masked Inference vs Flat Macro-F1 for Custom Models
# ─────────────────────────────────────────────────────────────────────────────

def compute_crop_masked_evaluation(
    custom_models: List[str],
    inference_results: Dict[str, Dict[str, Any]],
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For custom models with dual heads:
    Compare flat macro-F1 with crop-masked inference (logits restricted to predicted crop).
    """
    y_true = test_df["label"].values
    y_crop_true = np.array([CLASS_TO_CROP_IDX[y] for y in y_true])

    rows = []
    for m in custom_models:
        flat_preds = inference_results[m]["preds"]
        masked_preds = inference_results[m]["masked_preds"]
        pred_crops = inference_results[m]["pred_crops"]

        flat_f1 = f1_score(y_true, flat_preds, average="macro", zero_division=0)
        masked_f1 = f1_score(y_true, masked_preds, average="macro", zero_division=0)
        flat_acc = accuracy_score(y_true, flat_preds)
        masked_acc = accuracy_score(y_true, masked_preds)
        crop_acc = accuracy_score(y_crop_true, pred_crops)
        delta_f1 = masked_f1 - flat_f1

        rows.append({
            "model":            m,
            "flat_macro_f1":    round(float(flat_f1), 4),
            "masked_macro_f1":  round(float(masked_f1), 4),
            "delta_macro_f1":   f"{delta_f1:+.4f}",
            "flat_accuracy":    round(float(flat_acc), 4),
            "masked_accuracy":  round(float(masked_acc), 4),
            "crop_head_acc":    round(float(crop_acc), 4),
        })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "error_analysis_crop_masking.csv"
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Group-Level Error Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_level_errors(
    models: List[str],
    inference_results: Dict[str, Dict[str, Any]],
    test_df: pd.DataFrame,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Errors counted per group_id:
    For each class, the number of distinct groups with at least one error,
    and the total groups in the test split.
    """
    y_true = test_df["label"].values
    group_ids = test_df["group_id"].values
    n_classes = len(class_names)

    # Total distinct groups per class in test split
    total_groups_per_class = {}
    for c_idx in range(n_classes):
        mask = (y_true == c_idx)
        total_groups_per_class[c_idx] = len(np.unique(group_ids[mask]))

    # For each model, count distinct group_ids with >= 1 error
    model_err_groups: Dict[str, Dict[int, int]] = {m: {} for m in models}
    for m in models:
        preds = inference_results[m]["preds"]
        err_mask = (y_true != preds)
        for c_idx in range(n_classes):
            cls_err_mask = err_mask & (y_true == c_idx)
            model_err_groups[m][c_idx] = len(np.unique(group_ids[cls_err_mask]))

    rows = []
    for c_idx, cname in enumerate(class_names):
        tot_g = total_groups_per_class[c_idx]
        row: Dict[str, Any] = {
            "class_name":   cname,
            "crop":         CLASS_TO_CROP_NAME[c_idx],
            "total_groups": tot_g,
        }
        for m in models:
            err_g = model_err_groups[m][c_idx]
            pct = (err_g / tot_g * 100) if tot_g > 0 else 0.0
            row[f"{m}_err_groups"] = err_g
            row[f"{m}_err_pct"] = round(pct, 1)
        rows.append(row)

    # Add summary total row
    tot_row: Dict[str, Any] = {
        "class_name":   "TOTAL / OVERALL",
        "crop":         "All",
        "total_groups": int(test_df["group_id"].nunique()),
    }
    for m in models:
        preds = inference_results[m]["preds"]
        err_g_total = len(np.unique(group_ids[y_true != preds]))
        tot_all = test_df["group_id"].nunique()
        tot_row[f"{m}_err_groups"] = err_g_total
        tot_row[f"{m}_err_pct"] = round((err_g_total / tot_all * 100), 1)
    rows.append(tot_row)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "error_analysis_group_errors.csv"
    df.to_csv(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Error Montage Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_error_montage(
    custom_se_result: Dict[str, Any],
    test_df: pd.DataFrame,
    class_names: List[str],
    n_top_confident: int = N_CONFIDENT_ERRORS,
) -> Tuple[Path, pd.DataFrame]:
    """
    Generate a montage PNG in reports/montages/ of every DISTINCT image
    (deduplicated by group_id) in the 20 most confident errors of custom_se
    plus all cross-crop errors of custom_se.

    Each panel is labeled with true class, predicted class, confidence, and file name.
    """
    MONTAGES_DIR.mkdir(parents=True, exist_ok=True)

    probs = custom_se_result["probs"]
    preds = custom_se_result["preds"]
    y_true = test_df["label"].values
    group_ids = test_df["group_id"].values
    paths = test_df["image_path"].values

    wrong_mask = (preds != y_true)
    wrong_idxs = np.where(wrong_mask)[0]
    wrong_confs = probs[wrong_idxs, preds[wrong_idxs]]

    # 1. Top 20 most confident errors
    top_order = np.argsort(wrong_confs)[::-1][:n_top_confident]
    top_confident_idxs = list(wrong_idxs[top_order])

    # 2. All cross-crop errors
    y_crop_true = np.array([CLASS_TO_CROP_IDX[y] for y in y_true])
    pred_crops_flat = np.array([CLASS_TO_CROP_IDX[p] for p in preds])
    cross_crop_idxs = list(np.where(wrong_mask & (y_crop_true != pred_crops_flat))[0])

    # Combine candidates (top confident first, then cross-crop)
    candidate_idxs: List[int] = []
    seen_cand = set()
    for idx in top_confident_idxs + cross_crop_idxs:
        if idx not in seen_cand:
            seen_cand.add(idx)
            candidate_idxs.append(idx)

    # Deduplicate strictly by group_id: for duplicates, keep highest error confidence
    group_best: Dict[int, Dict[str, Any]] = {}
    for idx in candidate_idxs:
        gid = int(group_ids[idx])
        conf = float(probs[idx, preds[idx]])
        is_cross = (y_crop_true[idx] != pred_crops_flat[idx])
        in_top20 = (idx in top_confident_idxs)

        if (gid not in group_best) or (conf > group_best[gid]["confidence"]):
            group_best[gid] = {
                "idx":              idx,
                "group_id":         gid,
                "image_path":       paths[idx],
                "file_name":        Path(paths[idx]).name,
                "true_class":       class_names[y_true[idx]],
                "pred_class":       class_names[preds[idx]],
                "confidence":       round(conf, 4),
                "is_cross_crop":    is_cross,
                "in_top_confident": in_top20,
            }

    # Sort final distinct images: highest confidence first
    distinct_items = sorted(group_best.values(), key=lambda x: -x["confidence"])
    df_montage = pd.DataFrame(distinct_items)

    # Build Matplotlib montage grid (e.g. 4 rows x 5 cols for 20 images)
    n_images = len(distinct_items)
    cols = 5
    rows = int(np.ceil(n_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.5), dpi=150)
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_images:
            item = distinct_items[i]
            img_path = resolve_image_path(item["image_path"])
            try:
                with Image.open(img_path) as img:
                    ax.imshow(img.convert("RGB"))
            except Exception as e:
                ax.text(0.5, 0.5, f"Image not found:\n{img_path.name}", ha="center", va="center")

            ax.set_xticks([])
            ax.set_yticks([])

            # Title label formatting
            crop_flag = "  ★ CROSS-CROP" if item["is_cross_crop"] else ""
            title_color = "#b30000" if item["is_cross_crop"] else "#1a237e"

            title_str = (
                f"{item['file_name']}\n"
                f"True: {item['true_class']}\n"
                f"Pred: {item['pred_class']} ({item['confidence']:.1%}){crop_flag}\n"
                f"Group: {item['group_id']}"
            )
            ax.set_title(title_str, fontsize=8.5, fontweight="bold", color=title_color, pad=4)

            # Border color highlights error type
            for spine in ax.spines.values():
                spine.set_edgecolor(title_color)
                spine.set_linewidth(2.0 if item["is_cross_crop"] else 1.2)
        else:
            ax.axis("off")

    fig.suptitle(
        f"custom_se: Top-20 Confident Errors + Cross-Crop Errors (Deduplicated by group_id | N={n_images})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_png = MONTAGES_DIR / "custom_se_errors_montage.png"
    alias_png = MONTAGES_DIR / "error_montage.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(alias_png, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[Saved] {out_png}")
    print(f"[Saved] {alias_png}")
    return out_png, df_montage


# ─────────────────────────────────────────────────────────────────────────────
# Per-class P/R/F1 with group-bootstrap 95% CIs
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    class_names: List[str],
    model_key: str,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = BOOTSTRAP_ALPHA,
) -> pd.DataFrame:
    """Compute per-class P/R/F1 with group-bootstrap 95% CIs in a fast pre-indexed pass."""
    n_classes = len(class_names)
    rng = np.random.default_rng(SEED)

    labels = list(range(n_classes))
    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    rec  = recall_score(   y_true, y_pred, average=None, zero_division=0, labels=labels)
    f1   = f1_score(       y_true, y_pred, average=None, zero_division=0, labels=labels)
    support = np.array([(y_true == i).sum() for i in range(n_classes)])

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    # Pre-index samples per group for fast O(1) lookup
    group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}

    boot_prec = []
    boot_rec = []
    boot_f1 = []

    print(f"  [{model_key}] Running fast group-bootstrap CIs (n={n_boot}) ...")
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = np.concatenate([group_to_indices[g] for g in sampled_groups])

        bt = y_true[boot_idx]
        bp = y_pred[boot_idx]

        boot_prec.append(precision_score(bt, bp, average=None, zero_division=0, labels=labels))
        boot_rec.append(recall_score(bt, bp, average=None, zero_division=0, labels=labels))
        boot_f1.append(f1_score(bt, bp, average=None, zero_division=0, labels=labels))

    boot_prec = np.array(boot_prec)
    boot_rec = np.array(boot_rec)
    boot_f1 = np.array(boot_f1)

    prec_lo = np.percentile(boot_prec, 100 * alpha / 2, axis=0)
    prec_hi = np.percentile(boot_prec, 100 * (1 - alpha / 2), axis=0)
    rec_lo  = np.percentile(boot_rec,  100 * alpha / 2, axis=0)
    rec_hi  = np.percentile(boot_rec,  100 * (1 - alpha / 2), axis=0)
    f1_lo   = np.percentile(boot_f1,   100 * alpha / 2, axis=0)
    f1_hi   = np.percentile(boot_f1,   100 * (1 - alpha / 2), axis=0)

    rows = []
    for i, cname in enumerate(class_names):
        rows.append({
            "model":         model_key,
            "class_name":    cname,
            "support":       int(support[i]),
            "precision":     round(float(prec[i]), 4),
            "prec_ci_lo":    round(float(prec_lo[i]), 4),
            "prec_ci_hi":    round(float(prec_hi[i]), 4),
            "recall":        round(float(rec[i]),  4),
            "rec_ci_lo":     round(float(rec_lo[i]),  4),
            "rec_ci_hi":     round(float(rec_hi[i]),  4),
            "f1":            round(float(f1[i]),   4),
            "f1_ci_lo":      round(float(f1_lo[i]),   4),
            "f1_ci_hi":      round(float(f1_hi[i]),   4),
        })
    return pd.DataFrame(rows)


def compute_top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_key: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top-k off-diagonal confusion matrix entries by count."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))

    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "model":         model_key,
                    "true_class":    class_names[i],
                    "pred_class":    class_names[j],
                    "count":         int(cm[i, j]),
                    "true_support":  int((y_true == i).sum()),
                })

    df = pd.DataFrame(pairs).sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def compute_per_crop_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_key: str,
) -> pd.DataFrame:
    """Compute macro-F1 restricted to each crop's class subset."""
    rows = []
    for crop_name, class_indices in CROP_TO_CLASS_INDICES.items():
        mask = np.isin(y_true, class_indices)
        if mask.sum() == 0:
            continue
        crop_f1 = float(f1_score(
            y_true[mask], y_pred[mask],
            labels=class_indices,
            average="macro",
            zero_division=0,
        ))
        rows.append({
            "model":        model_key,
            "crop":         crop_name,
            "n_samples":    int(mask.sum()),
            "macro_f1":     round(crop_f1, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-Print Formatting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 95) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _print_crop_confusion_matrices(crop_cms: Dict[str, np.ndarray], df_summary: pd.DataFrame) -> None:
    _banner("1. 3x3 Crop Confusion Matrices & Cross-Crop Errors (Dedup Test Split)")
    for model_name, cm in crop_cms.items():
        print(f"\n--- Model: {model_name} ---")
        df_cm = pd.DataFrame(
            cm,
            index=[f"True_{c}" for c in CROP_NAMES],
            columns=[f"Pred_{c}" for c in CROP_NAMES],
        )
        print(df_cm.to_string())
        cross_errs = int(cm.sum() - np.trace(cm))
        print(f"  → Cross-Crop Errors: {cross_errs:3d} / {cm.sum():4d} ({cross_errs / cm.sum():.2%})")

    print("\n--- Summary of Cross-Crop Errors Across Models ---")
    print(df_summary.to_string(index=False))


def _print_crop_masked_results(df: pd.DataFrame) -> None:
    _banner("2. Custom Models: Flat vs Crop-Masked Inference Macro-F1")
    print(df.to_string(index=False))


def _print_group_level_errors(df: pd.DataFrame) -> None:
    _banner("3. Group-Level Error Analysis (Distinct group_ids with >= 1 Error vs Total Groups)")
    print(df.to_string(index=False))


def _print_montage_summary(df: pd.DataFrame, png_path: Path) -> None:
    _banner("4. custom_se Error Montage Summary (Deduplicated by group_id)")
    print(f"Montage PNG saved at: {png_path}\n")
    cols = ["group_id", "file_name", "true_class", "pred_class", "confidence", "is_cross_crop", "in_top_confident"]
    print(df[cols].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MONTAGES_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | Splits dir: {SPLITS_DIR} | Data root: {DATA_ROOT}")

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    models = list(MODEL_JSON.keys())   # resnet18, mobilenetv3_small, custom_none, custom_se

    # Load canonical class names
    with open(MODEL_JSON["resnet18"], "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)["class_names"]

    # ── Run fresh inference across all 4 models ──────────────────────────────
    print("\nRunning inference for models to extract probabilities and dual-head predictions...")
    inference_results: Dict[str, Dict[str, Any]] = {}
    for m in models:
        print(f"  Inferencing {m} ...")
        inference_results[m] = run_model_inference(m, test_df, device=device, batch_size=64)

    # ── 1. 3x3 Crop Confusion Matrix & Cross-Crop Errors ─────────────────────
    crop_cms, df_cross_summary, df_cross_records = compute_crop_confusion_and_cross_crop_errors(
        models=models,
        inference_results=inference_results,
        test_df=test_df,
        class_names=class_names,
    )

    # ── 2. Custom Models: Crop-Masked vs Flat Macro-F1 ────────────────────────
    custom_models = [m for m in models if "custom" in m]
    # Also evaluate custom_cbam if checkpoint and result json exist
    if (CHECKPOINTS_DIR / "custom_cbam_seed42_best.pth").exists():
        if "custom_cbam" not in inference_results:
            print("  Inferencing custom_cbam for crop-masked evaluation ...")
            inference_results["custom_cbam"] = run_model_inference("custom_cbam", test_df, device=device, batch_size=64)
            custom_models.append("custom_cbam")

    df_masked = compute_crop_masked_evaluation(
        custom_models=custom_models,
        inference_results=inference_results,
        test_df=test_df,
    )

    # ── 3. Group-Level Error Analysis ────────────────────────────────────────
    df_group_errors = compute_group_level_errors(
        models=models,
        inference_results=inference_results,
        test_df=test_df,
        class_names=class_names,
    )

    # ── 4. Error Montage for custom_se ───────────────────────────────────────
    montage_path, df_montage = generate_error_montage(
        custom_se_result=inference_results["custom_se"],
        test_df=test_df,
        class_names=class_names,
        n_top_confident=N_CONFIDENT_ERRORS,
    )

    # Also save confident_errors.csv for custom_se
    conf_path = REPORTS_DIR / "confident_errors.csv"
    df_montage[df_montage["in_top_confident"]].to_csv(conf_path, index=False)

    # ── Existing metrics: Bootstrap CIs, Confusion Pairs, Per-Crop F1 ─────────
    all_per_class: List[pd.DataFrame] = []
    all_conf_pairs: List[pd.DataFrame] = []
    all_per_crop: List[pd.DataFrame] = []

    for m in models:
        y_true, y_pred, group_ids, _ = load_predictions(m)
        all_per_class.append(compute_per_class_metrics_with_ci(y_true, y_pred, group_ids, class_names, m))
        all_conf_pairs.append(compute_top_confusion_pairs(y_true, y_pred, class_names, m))
        all_per_crop.append(compute_per_crop_f1(y_true, y_pred, m))

    df_per_class_all = pd.concat(all_per_class, ignore_index=True)
    df_conf_pairs_all = pd.concat(all_conf_pairs, ignore_index=True)
    df_per_crop_all = pd.concat(all_per_crop, ignore_index=True)

    df_per_class_all.to_csv(RESULTS_DIR / "error_analysis_per_class.csv", index=False)
    df_conf_pairs_all.to_csv(RESULTS_DIR / "error_analysis_confusion_pairs.csv", index=False)
    df_per_crop_all.to_csv(RESULTS_DIR / "error_analysis_per_crop_f1.csv", index=False)

    # ── PRINT ALL REQUIRED TABLES ─────────────────────────────────────────────
    _print_crop_confusion_matrices(crop_cms, df_cross_summary)
    _print_crop_masked_results(df_masked)
    _print_group_level_errors(df_group_errors)
    _print_montage_summary(df_montage, montage_path)

    # Per-crop macro-F1 summary
    _banner("Per-Crop Macro-F1 Summary")
    pivot_crop = df_per_crop_all.pivot_table(index="model", columns="crop", values="macro_f1", aggfunc="first")[CROP_NAMES]
    print(pivot_crop.round(4).to_string())

    # Overall Macro-F1 summary
    _banner("Overall Macro-F1 Summary (Dedup Test Split)")
    overall_rows = []
    for model_key, grp in df_per_class_all.groupby("model", sort=False):
        overall_rows.append({
            "model": model_key,
            "macro_f1": round(grp["f1"].mean(), 4),
        })
    df_overall = pd.DataFrame(overall_rows)
    print(df_overall.to_string(index=False))

    print("\n[Done] Error analysis complete.\n")


if __name__ == "__main__":
    main()
