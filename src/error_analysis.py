"""
Error Analysis for Multi-Crop Leaf Disease Classifiers.

Models analysed (dedup test split, seed 42):
  - resnet18
  - mobilenetv3_small  (mobilenetv3_small_seed42.json)
  - custom_none        (custom_none_seed42.json)
  - custom_se          (custom_se_seed42.json)

Outputs written to results/error_analysis_*.csv and printed to stdout:
  1. Per-class precision / recall / F1 with group-bootstrap 95 % CIs
     → results/error_analysis_per_class.csv
  2. Top-10 most confused class pairs (count of off-diagonal confusion matrix entries)
     → results/error_analysis_confusion_pairs.csv
  3. Per-crop macro-F1 (Papaya / Potato / Rice)
     → results/error_analysis_per_crop_f1.csv
  4. 20 most confident wrong predictions for custom_se
     → reports/confident_errors.csv   (image_path, true_class, pred_class, confidence)

Usage:
  PYTHONPATH=. .venv/bin/python src/error_analysis.py
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# ── project imports ─────────────────────────────────────────────────────────
from src.data.dataset import CropDiseaseDataset, get_strong_transforms, get_transforms
from src.models.backbones import create_model, normalize_model_name
from src.models.custom_net import build_custom_net, CROP_TO_CLASSES

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
CHECKPOINTS_DIR = Path("checkpoints")
SPLITS_DIR = Path("splits/dedup")
DATA_ROOT = Path("data")

# Maps short names → result JSON paths
MODEL_JSON: Dict[str, str] = {
    "resnet18":          "results/resnet18_seed42.json",
    "mobilenetv3_small": "results/mobilenetv3_small_seed42.json",
    "custom_none":       "results/custom_none_seed42.json",
    "custom_se":         "results/custom_se_seed42.json",
}

# Class → crop group (Papaya 0-3, Potato 4-7, Rice 8-11)
CLASS_TO_CROP_NAME = {
    0: "Papaya", 1: "Papaya", 2: "Papaya",  3: "Papaya",
    4: "Potato", 5: "Potato", 6: "Potato",  7: "Potato",
    8: "Rice",   9: "Rice",  10: "Rice",   11: "Rice",
}
CROP_NAMES = ["Papaya", "Potato", "Rice"]
CROP_TO_CLASS_INDICES = {
    "Papaya": list(range(0, 4)),
    "Potato": list(range(4, 8)),
    "Rice":   list(range(8, 12)),
}

N_BOOTSTRAP = 2000
BOOTSTRAP_ALPHA = 0.05  # 95 % CI
N_CONFIDENT_ERRORS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Reproducible bootstrap sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_predictions(model_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load y_true, y_pred, group_ids, and class_names from a saved result JSON.

    Returns:
        y_true    : (N,) int array of ground-truth labels
        y_pred    : (N,) int array of predicted labels
        group_ids : (N,) array of group IDs for block-bootstrap
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
# 1. Per-class P/R/F1 with group-bootstrap 95 % CIs
# ─────────────────────────────────────────────────────────────────────────────

def _group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    metric_fn,  # callable(y_true, y_pred) -> np.ndarray of shape (n_classes,)
    n_boot: int = N_BOOTSTRAP,
    alpha: float = BOOTSTRAP_ALPHA,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Block-bootstrap confidence intervals grouped by group_id (base image).

    Resamples whole groups (not individual samples) so that augmentation
    variants of the same base image stay together, preventing leakage.

    Returns:
        ci_lo, ci_hi  — lower and upper bounds, shape (n_classes,)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    bootstrap_stats = []

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_mask = np.isin(group_ids, sampled_groups)

        # For each resampled group, gather all sample indices belonging to it
        indices = []
        for g in sampled_groups:
            indices.append(np.where(group_ids == g)[0])
        if not indices:
            continue
        boot_idx = np.concatenate(indices)

        bt = y_true[boot_idx]
        bp = y_pred[boot_idx]
        stat = metric_fn(bt, bp)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)  # (n_boot, n_classes)
    lo = np.percentile(bootstrap_stats, 100 * alpha / 2, axis=0)
    hi = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def compute_per_class_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    class_names: List[str],
    model_key: str,
) -> pd.DataFrame:
    """Compute per-class P/R/F1 with group-bootstrap 95 % CIs."""
    n_classes = len(class_names)
    rng = np.random.default_rng(SEED)

    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    rec  = recall_score(   y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    f1   = f1_score(       y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    support = np.array([(y_true == i).sum() for i in range(n_classes)])

    def prec_fn(yt, yp): return precision_score(yt, yp, average=None, zero_division=0, labels=list(range(n_classes)))
    def rec_fn( yt, yp): return recall_score(   yt, yp, average=None, zero_division=0, labels=list(range(n_classes)))
    def f1_fn(  yt, yp): return f1_score(       yt, yp, average=None, zero_division=0, labels=list(range(n_classes)))

    print(f"  [{model_key}] Running group-bootstrap CIs (n={N_BOOTSTRAP}) ...")
    prec_lo, prec_hi = _group_bootstrap_ci(y_true, y_pred, group_ids, prec_fn, rng=rng)
    rec_lo,  rec_hi  = _group_bootstrap_ci(y_true, y_pred, group_ids, rec_fn,  rng=rng)
    f1_lo,   f1_hi   = _group_bootstrap_ci(y_true, y_pred, group_ids, f1_fn,   rng=rng)

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Top-10 confused class pairs
# ─────────────────────────────────────────────────────────────────────────────

def compute_top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_key: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the top_k off-diagonal confusion matrix entries by count."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Collect off-diagonal pairs
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-crop macro-F1
# ─────────────────────────────────────────────────────────────────────────────

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
# 4. Confident wrong predictions for custom_se (requires fresh inference)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_confident_errors_custom_se(
    test_df: pd.DataFrame,
    class_names: List[str],
    n_errors: int = N_CONFIDENT_ERRORS,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Load the custom_se checkpoint and run inference on the dedup test split.
    Return the n_errors most confident wrong predictions as a DataFrame with
    columns [rank, image_path, true_class, pred_class, confidence].
    """
    ckpt_path = CHECKPOINTS_DIR / "custom_se_seed42_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"  [custom_se] Loading checkpoint: {ckpt_path} → device={device}")
    model = build_custom_net(attention="se", dropout=0.3).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tfms = get_strong_transforms(image_size=224)["test"]
    dataset = CropDiseaseDataset(test_df, image_root=DATA_ROOT, transform=tfms)

    # Use num_workers=0 to keep it simple and avoid spawn issues on macOS
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs: List[np.ndarray] = []
    all_preds: List[int] = []
    all_true:  List[int] = []
    all_paths: List[str] = []

    print(f"  [custom_se] Running inference on {len(dataset)} samples ...")
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        paths  = batch["image_path"]

        crop_logits, disease_logits = model(images)
        probs = F.softmax(disease_logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds.tolist())
        all_true.extend(labels.cpu().numpy().tolist())
        all_paths.extend(paths)

    all_probs = np.array(all_probs)   # (N, 12)
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    # Wrong predictions only
    wrong_mask   = all_preds != all_true
    wrong_probs  = all_probs[wrong_mask]
    wrong_preds  = all_preds[wrong_mask]
    wrong_true   = all_true[wrong_mask]
    wrong_paths  = [p for p, w in zip(all_paths, wrong_mask) if w]

    # Confidence = max softmax probability for predicted class
    confidence   = wrong_probs[np.arange(len(wrong_preds)), wrong_preds]

    # Sort descending by confidence
    order = np.argsort(confidence)[::-1][:n_errors]

    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append({
            "rank":        rank,
            "image_path":  wrong_paths[idx],
            "true_class":  class_names[wrong_true[idx]],
            "pred_class":  class_names[wrong_preds[idx]],
            "confidence":  round(float(confidence[idx]), 6),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 90) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def _print_per_class(df: pd.DataFrame) -> None:
    """Print per-class table grouped by model."""
    for model_key, grp in df.groupby("model", sort=False):
        _banner(f"Per-class P/R/F1 with 95% bootstrap CI  ·  model={model_key}")
        cols = ["class_name", "support",
                "precision", "prec_ci_lo", "prec_ci_hi",
                "recall",    "rec_ci_lo",  "rec_ci_hi",
                "f1",        "f1_ci_lo",   "f1_ci_hi"]
        print(grp[cols].to_string(index=False))
        macro_f1 = grp["f1"].mean()
        print(f"\n  → Macro-F1 (unweighted mean over classes): {macro_f1:.4f}")


def _print_confusion_pairs(df: pd.DataFrame) -> None:
    """Print top-10 confusion pairs grouped by model."""
    for model_key, grp in df.groupby("model", sort=False):
        _banner(f"Top-10 most confused class pairs  ·  model={model_key}")
        print(grp[["rank", "true_class", "pred_class", "count", "true_support"]].to_string(index=False))


def _print_per_crop_f1(df: pd.DataFrame) -> None:
    """Print per-crop macro-F1 table."""
    _banner("Per-crop macro-F1")
    pivot = df.pivot_table(index="model", columns="crop", values="macro_f1", aggfunc="first")
    # Ensure consistent column order
    for c in CROP_NAMES:
        if c not in pivot.columns:
            pivot[c] = float("nan")
    pivot = pivot[CROP_NAMES]
    print(pivot.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODEL_JSON.keys())   # resnet18, mobilenetv3_small, custom_none, custom_se

    all_per_class: List[pd.DataFrame]      = []
    all_conf_pairs: List[pd.DataFrame]     = []
    all_per_crop: List[pd.DataFrame]       = []

    # ── Loop over models ─────────────────────────────────────────────────────
    for model_key in models:
        print(f"\n{'─' * 60}")
        print(f"Processing model: {model_key}")
        print(f"{'─' * 60}")

        y_true, y_pred, group_ids, class_names = load_predictions(model_key)
        print(f"  Loaded predictions: {len(y_true)} samples, {len(class_names)} classes")
        print(f"  Unique groups: {len(np.unique(group_ids))}")

        # ── 1. Per-class P/R/F1 with CI ──────────────────────────────────────
        df_pc = compute_per_class_metrics_with_ci(
            y_true, y_pred, group_ids, class_names, model_key
        )
        all_per_class.append(df_pc)

        # ── 2. Top-10 confusion pairs ─────────────────────────────────────────
        df_cp = compute_top_confusion_pairs(y_true, y_pred, class_names, model_key)
        all_conf_pairs.append(df_cp)

        # ── 3. Per-crop macro-F1 ──────────────────────────────────────────────
        df_crop = compute_per_crop_f1(y_true, y_pred, model_key)
        all_per_crop.append(df_crop)

    # ── Concatenate all model results ─────────────────────────────────────────
    df_per_class_all  = pd.concat(all_per_class,  ignore_index=True)
    df_conf_pairs_all = pd.concat(all_conf_pairs, ignore_index=True)
    df_per_crop_all   = pd.concat(all_per_crop,   ignore_index=True)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pc_path = RESULTS_DIR / "error_analysis_per_class.csv"
    cp_path = RESULTS_DIR / "error_analysis_confusion_pairs.csv"
    cr_path = RESULTS_DIR / "error_analysis_per_crop_f1.csv"

    df_per_class_all.to_csv(pc_path,  index=False)
    df_conf_pairs_all.to_csv(cp_path, index=False)
    df_per_crop_all.to_csv(cr_path,   index=False)

    print(f"\n[Saved] {pc_path}")
    print(f"[Saved] {cp_path}")
    print(f"[Saved] {cr_path}")

    # ── 4. Confident wrong predictions for custom_se ──────────────────────────
    print(f"\n{'─' * 60}")
    print("Computing most confident wrong predictions for custom_se ...")
    print(f"{'─' * 60}")

    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    # class_names from custom_se JSON (already sorted)
    with open(MODEL_JSON["custom_se"], "r", encoding="utf-8") as f:
        se_result = json.load(f)
    class_names_se: List[str] = se_result["class_names"]

    df_confident = collect_confident_errors_custom_se(
        test_df, class_names_se, n_errors=N_CONFIDENT_ERRORS
    )

    conf_path = REPORTS_DIR / "confident_errors.csv"
    df_confident.to_csv(conf_path, index=False)
    print(f"\n[Saved] {conf_path}")

    # ── Print all summary tables ───────────────────────────────────────────────
    _print_per_class(df_per_class_all)
    _print_confusion_pairs(df_conf_pairs_all)
    _print_per_crop_f1(df_per_crop_all)

    _banner("Top-20 Most Confident Wrong Predictions  ·  custom_se")
    print(df_confident.to_string(index=False))

    # ── Overall macro-F1 summary ──────────────────────────────────────────────
    _banner("Macro-F1 Summary (mean over 12 classes, dedup test split)")
    summary_rows = []
    for model_key, grp in df_per_class_all.groupby("model", sort=False):
        summary_rows.append({
            "model":    model_key,
            "macro_f1": round(grp["f1"].mean(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n[Done] Error analysis complete.\n")


if __name__ == "__main__":
    main()
