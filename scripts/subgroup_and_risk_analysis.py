"""
Subgroup, Source-Overlap, Background-Brightness, and Risk-Coverage Analysis.

Models evaluated:
  - resnet18
  - mobilenetv3_small
  - custom_none
  - custom_se

Tasks:
  1. Add image-source bucket per image: native resolution (from original file, not cache)
     grouped as {256x256, other} for Potato. Report per crop x bucket:
     number of images, number of groups, macro-F1 and accuracy (group-bootstrap 95% CI).
  2. For Potato_Leafroll_Virus and every other class, report share of images in 256x256 bucket
     and other buckets across train, val, and test splits.
  3. For every class in Papaya, compute mean brightness of outer 10% border of each image
     and report mean and std per class; flag any class whose border brightness separates it.
  4. Risk-coverage table for each model: accuracy, macro-F1, and fraction retained at confidence
     thresholds 0.0, 0.5, 0.6, 0.7, 0.8, 0.9.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import CropDiseaseDataset, get_transforms
from src.models.backbones import create_model
from src.models.custom_net import build_custom_net

SEED = 42
N_BOOTSTRAP = 2000
ALPHA = 0.05
CHECKPOINTS_DIR = Path("checkpoints")
SPLITS_DIR = Path("splits/dedup")
DATA_ROOT = Path("data")
RESULTS_DIR = Path("results")

MODELS = ["resnet18", "mobilenetv3_small", "custom_none", "custom_se"]

CROP_NAMES = ["Papaya", "Potato", "Rice"]
CLASS_TO_CROP_IDX = {
    0: 0, 1: 0, 2: 0, 3: 0,     # Papaya
    4: 1, 5: 1, 6: 1, 7: 1,     # Potato
    8: 2, 9: 2, 10: 2, 11: 2,   # Rice
}
CROP_TO_CLASSES = {
    "Papaya": [0, 1, 2, 3],
    "Potato": [4, 5, 6, 7],
    "Rice":   [8, 9, 10, 11],
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_native_size(path_str: str) -> Tuple[int, int]:
    """Get (width, height) of original image from data/ directory."""
    raw_p = str(path_str).replace("\\", "/").replace("data_cache/", "data/")
    p = Path(raw_p)
    if not p.exists():
        alt = Path("data") / "/".join(p.parts[1:])
        if alt.exists():
            p = alt
    with Image.open(p) as img:
        return img.size


@torch.no_grad()
def run_inference(model_name: str, test_df: pd.DataFrame, device: torch.device) -> Dict[str, Any]:
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_seed42_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    is_custom = "custom" in model_name
    if is_custom:
        att = model_name.replace("custom_", "")
        model = build_custom_net(attention=att, dropout=0.3).to(device)
        img_root = Path("data_cache") if Path("data_cache").exists() else Path("data")
        eval_df = pd.read_csv("splits_cache/dedup/test.csv") if Path("splits_cache/dedup/test.csv").exists() else test_df
    else:
        model = create_model(backbone_name=model_name, num_classes=12, pretrained=False).to(device)
        img_root = Path("data")
        eval_df = pd.read_csv("splits/dedup/test.csv") if Path("splits/dedup/test.csv").exists() else test_df

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tfms = get_transforms(image_size=224, resize_dim=256)["test"]
    dataset = CropDiseaseDataset(eval_df, image_root=img_root, transform=tfms)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_probs = []
    all_preds = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if is_custom:
            crop_logits, disease_logits = model(images)
            probs = F.softmax(disease_logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        else:
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)

    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)

    return {
        "probs": probs_arr,
        "preds": preds_arr,
        "confs": probs_arr.max(axis=1),
    }


def compute_group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute group-bootstrap 95% CI for Accuracy and Macro-F1.
    Returns:
      (acc_point, acc_lo, acc_hi), (f1_point, f1_lo, f1_hi)
    """
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    acc_point = accuracy_score(y_true, y_pred)
    f1_point = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    if n_groups <= 1:
        return (acc_point, acc_point, acc_point), (f1_point, f1_point, f1_point)

    group_to_idx = {g: np.where(group_ids == g)[0] for g in unique_groups}

    boot_accs = []
    boot_f1s = []

    for _ in range(n_boot):
        sampled_g = rng.choice(unique_groups, size=n_groups, replace=True)
        b_idx = np.concatenate([group_to_idx[g] for g in sampled_g])
        bt = y_true[b_idx]
        bp = y_pred[b_idx]

        boot_accs.append(accuracy_score(bt, bp))
        boot_f1s.append(f1_score(bt, bp, labels=labels, average="macro", zero_division=0))

    boot_accs = np.array(boot_accs)
    boot_f1s = np.array(boot_f1s)

    acc_lo = float(np.percentile(boot_accs, 100 * alpha / 2))
    acc_hi = float(np.percentile(boot_accs, 100 * (1 - alpha / 2)))
    f1_lo = float(np.percentile(boot_f1s, 100 * alpha / 2))
    f1_hi = float(np.percentile(boot_f1s, 100 * (1 - alpha / 2)))

    return (acc_point, acc_lo, acc_hi), (f1_point, f1_lo, f1_hi)


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load splits
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")

    with open("results/resnet18_seed42.json", "r") as f:
        class_names = json.load(f)["class_names"]

    # Native resolutions for test set
    print("\nReading native image resolutions for test set...")
    test_sizes = [get_native_size(p) for p in test_df["image_path"]]
    test_df["native_size"] = test_sizes
    # Source bucket: 256x256 vs other for Potato; other for non-Potato
    test_df["bucket"] = [
        "256x256" if (s == (256, 256) and c == "Potato") else "other"
        for s, c in zip(test_sizes, test_df["crop"])
    ]

    # Run inference for all 4 models
    print("\nRunning inference for models...")
    model_outputs = {}
    for m in MODELS:
        print(f"  Inferencing {m} ...")
        model_outputs[m] = run_inference(m, test_df, device)

    # ─────────────────────────────────────────────────────────────────────────
    # Task 1: Subgroup performance per crop x bucket
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 1: Subgroup Performance per Crop x Bucket (Group-Bootstrap 95% CI)")
    print("=" * 90)

    # Define crop x bucket slices
    slices = [
        ("Potato", "256x256"),
        ("Potato", "other"),
        ("Papaya", "other"),
        ("Rice", "other"),
    ]

    task1_rows = []
    for model_name in MODELS:
        y_pred = model_outputs[model_name]["preds"]
        y_true = test_df["label"].values
        group_ids = test_df["group_id"].values

        for crop, bucket in slices:
            mask = (test_df["crop"] == crop) & (test_df["bucket"] == bucket)
            sub_idx = np.where(mask)[0]
            n_images = len(sub_idx)
            sub_groups = np.unique(group_ids[sub_idx])
            n_groups = len(sub_groups)

            sub_yt = y_true[sub_idx]
            sub_yp = y_pred[sub_idx]
            sub_grp = group_ids[sub_idx]

            active_labels = np.unique(sub_yt)
            crop_labels = np.array(CROP_TO_CLASSES[crop])

            # Group bootstrap for Accuracy
            (acc, acc_lo, acc_hi), (f1_act, f1_act_lo, f1_act_hi) = compute_group_bootstrap_ci(
                sub_yt, sub_yp, sub_grp, labels=active_labels
            )
            # Also compute for all crop labels
            _, (f1_crp, f1_crp_lo, f1_crp_hi) = compute_group_bootstrap_ci(
                sub_yt, sub_yp, sub_grp, labels=crop_labels
            )

            task1_rows.append({
                "model": model_name,
                "crop": crop,
                "bucket": bucket,
                "n_images": n_images,
                "n_groups": n_groups,
                "accuracy": round(acc, 4),
                "acc_95_ci": f"[{acc_lo:.4f}, {acc_hi:.4f}]",
                "macro_f1_active": round(f1_act, 4),
                "f1_act_95_ci": f"[{f1_act_lo:.4f}, {f1_act_hi:.4f}]",
                "macro_f1_crop": round(f1_crp, 4),
                "f1_crp_95_ci": f"[{f1_crp_lo:.4f}, {f1_crp_hi:.4f}]",
            })

    df_task1 = pd.DataFrame(task1_rows)
    print(df_task1.to_string(index=False))
    df_task1.to_csv(RESULTS_DIR / "task1_crop_bucket_metrics.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Task 2: Class vs Bucket Overlap Across Splits
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 2: Class vs Capture-Source Bucket Overlap (Train, Val, Test)")
    print("=" * 90)

    print("Extracting native image resolutions for train and val splits...")
    train_sizes = [get_native_size(p) for p in train_df["image_path"]]
    train_df["bucket"] = [
        "256x256" if (s == (256, 256) and c == "Potato") else "other"
        for s, c in zip(train_sizes, train_df["crop"])
    ]

    val_sizes = [get_native_size(p) for p in val_df["image_path"]]
    val_df["bucket"] = [
        "256x256" if (s == (256, 256) and c == "Potato") else "other"
        for s, c in zip(val_sizes, val_df["crop"])
    ]

    task2_rows = []
    for cls in class_names:
        # Train
        tr_cls = train_df[train_df["class_name"] == cls]
        tr_tot = len(tr_cls)
        tr_256 = (tr_cls["bucket"] == "256x256").sum()
        tr_oth = (tr_cls["bucket"] == "other").sum()
        tr_256_pct = (tr_256 / tr_tot * 100) if tr_tot > 0 else 0.0
        tr_oth_pct = (tr_oth / tr_tot * 100) if tr_tot > 0 else 0.0

        # Val
        va_cls = val_df[val_df["class_name"] == cls]
        va_tot = len(va_cls)
        va_256 = (va_cls["bucket"] == "256x256").sum()
        va_oth = (va_cls["bucket"] == "other").sum()
        va_256_pct = (va_256 / va_tot * 100) if va_tot > 0 else 0.0
        va_oth_pct = (va_oth / va_tot * 100) if va_tot > 0 else 0.0

        # Test
        te_cls = test_df[test_df["class_name"] == cls]
        te_tot = len(te_cls)
        te_256 = (te_cls["bucket"] == "256x256").sum()
        te_oth = (te_cls["bucket"] == "other").sum()
        te_256_pct = (te_256 / te_tot * 100) if te_tot > 0 else 0.0
        te_oth_pct = (te_oth / te_tot * 100) if te_tot > 0 else 0.0

        # Combined
        all_tot = tr_tot + va_tot + te_tot
        all_256 = tr_256 + va_256 + te_256
        all_oth = tr_oth + va_oth + te_oth
        all_256_pct = (all_256 / all_tot * 100) if all_tot > 0 else 0.0
        all_oth_pct = (all_oth / all_tot * 100) if all_tot > 0 else 0.0

        task2_rows.append({
            "class_name": cls,
            "train_n": tr_tot,
            "train_256_%": round(tr_256_pct, 1),
            "train_oth_%": round(tr_oth_pct, 1),
            "val_n": va_tot,
            "val_256_%": round(va_256_pct, 1),
            "val_oth_%": round(va_oth_pct, 1),
            "test_n": te_tot,
            "test_256_%": round(te_256_pct, 1),
            "test_oth_%": round(te_oth_pct, 1),
            "total_n": all_tot,
            "total_256_%": round(all_256_pct, 1),
            "total_oth_%": round(all_oth_pct, 1),
        })

    df_task2 = pd.DataFrame(task2_rows)
    print(df_task2.to_string(index=False))
    df_task2.to_csv(RESULTS_DIR / "task2_source_bucket_overlap.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Task 3: Background Proxy (Outer 10% Border Brightness in Papaya)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 3: Outer 10% Border Brightness per Papaya Class")
    print("=" * 90)

    def compute_border_brightness(img_path_str: str) -> float:
        raw_p = str(img_path_str).replace("\\", "/").replace("data_cache/", "data/")
        p = Path(raw_p)
        if not p.exists():
            alt = Path("data") / "/".join(p.parts[1:])
            if alt.exists():
                p = alt
        with Image.open(p) as img:
            arr = np.array(img.convert("L"), dtype=np.float32)
            h, w = arr.shape
            top = int(round(h * 0.10))
            bottom = int(round(h * 0.90))
            left = int(round(w * 0.10))
            right = int(round(w * 0.90))
            mask = np.ones((h, w), dtype=bool)
            mask[top:bottom, left:right] = False
            return float(arr[mask].mean())

    papaya_test = test_df[test_df["crop"] == "Papaya"].copy()
    print("Computing border brightness for Papaya test images...")
    papaya_test["border_brightness"] = [compute_border_brightness(p) for p in papaya_test["image_path"]]

    task3_rows = []
    classes_brightness = {}
    for cls in sorted(papaya_test["class_name"].unique()):
        vals = papaya_test[papaya_test["class_name"] == cls]["border_brightness"].values
        classes_brightness[cls] = vals
        task3_rows.append({
            "class_name": cls,
            "n_images": len(vals),
            "mean_brightness": round(float(vals.mean()), 2),
            "std_brightness": round(float(vals.std()), 2),
            "median": round(float(np.median(vals)), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "pct_gt_230": round(float((vals >= 230).mean() * 100), 1),
        })

    # Statistical separation check
    overall_mean = papaya_test["border_brightness"].mean()
    overall_std = papaya_test["border_brightness"].std()

    for row in task3_rows:
        cls = row["class_name"]
        diff = row["mean_brightness"] - overall_mean
        z_score = diff / (overall_std + 1e-6)
        row["z_score_vs_overall"] = round(float(z_score), 2)
        # Flag condition: significantly separated from others
        # Leaf Curl is >238 with very narrow std (pure white lab background)
        # Anthracnose is <199 with wide std (darker/natural background)
        flags = []
        if row["mean_brightness"] > 235:
            flags.append("HIGH_BRIGHTNESS (White background shortcut)")
        elif row["mean_brightness"] < 205:
            flags.append("LOW_BRIGHTNESS (Darker/field background)")
        row["flag"] = "; ".join(flags) if flags else "Normal"

    df_task3 = pd.DataFrame(task3_rows)
    print(df_task3.to_string(index=False))
    df_task3.to_csv(RESULTS_DIR / "task3_papaya_border_brightness.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Task 4: Risk-Coverage Tables for Each Model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 4: Risk-Coverage Tables (Selective Abstention by Confidence)")
    print("=" * 90)

    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    task4_all_rows = []

    for model_name in MODELS:
        print(f"\nModel: {model_name}")
        probs = model_outputs[model_name]["probs"]
        preds = model_outputs[model_name]["preds"]
        confs = model_outputs[model_name]["confs"]
        y_true = test_df["label"].values
        total_samples = len(y_true)

        model_rows = []
        for th in thresholds:
            keep_mask = confs >= th
            n_kept = int(keep_mask.sum())
            frac_retained = n_kept / total_samples

            if n_kept > 0:
                sub_yt = y_true[keep_mask]
                sub_yp = preds[keep_mask]
                acc = accuracy_score(sub_yt, sub_yp)
                # Macro-F1 across all 12 classes
                macro_f1_12 = f1_score(sub_yt, sub_yp, labels=list(range(12)), average="macro", zero_division=0)
                # Macro-F1 across active classes retained
                active_lbls = np.unique(sub_yt)
                macro_f1_act = f1_score(sub_yt, sub_yp, labels=active_lbls, average="macro", zero_division=0)
                n_active_classes = len(active_lbls)
            else:
                acc = 0.0
                macro_f1_12 = 0.0
                macro_f1_act = 0.0
                n_active_classes = 0

            row_data = {
                "model": model_name,
                "confidence_threshold": th,
                "retained_count": n_kept,
                "total_count": total_samples,
                "fraction_retained": round(frac_retained, 4),
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1_12, 4),
                "macro_f1_active": round(macro_f1_act, 4),
                "active_classes_count": n_active_classes,
            }
            model_rows.append(row_data)
            task4_all_rows.append(row_data)

        df_m = pd.DataFrame(model_rows)
        print(df_m[["confidence_threshold", "retained_count", "fraction_retained", "accuracy", "macro_f1", "macro_f1_active", "active_classes_count"]].to_string(index=False))

    df_task4 = pd.DataFrame(task4_all_rows)
    df_task4.to_csv(RESULTS_DIR / "task4_risk_coverage.csv", index=False)

    print("\n" + "=" * 90)
    print("ALL ANALYSES COMPLETE. CSV FILES SAVED IN results/.")
    print("=" * 90)


if __name__ == "__main__":
    main()
