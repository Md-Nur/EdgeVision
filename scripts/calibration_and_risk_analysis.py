"""
Calibration, Temperature Scaling, Equal Risk-Coverage, and Background Split Analysis.

Models evaluated:
  - resnet18
  - mobilenetv3_small
  - custom_none
  - custom_se

Tasks:
  1. Print exact loss implementation used in training (class weights, label smoothing,
     focal flag, logit scaling) and the mean/median max-softmax for correct and incorrect
     predictions (VAL and dedup TEST), saving histograms to reports/confidence_hist_<model>.png.
  2. Compute Expected Calibration Error (15 bins) and reliability diagrams; fit a single
     temperature T on VAL and report ECE and NLL on dedup TEST before and after.
  3. Risk-coverage curves at EQUAL coverage (100%, 90%, 80%, 70%, 60%) with accuracy,
     macro-F1, and per-class retained fraction using temperature-scaled confidences.
  4. For Papaya: performance split by border brightness (>=230 vs <230) per class
     with group-bootstrap 95% CIs.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CropDiseaseDataset, get_transforms
from src.losses.focal_loss import compute_class_weights
from src.models.backbones import create_model
from src.models.custom_net import build_custom_net
from src.utils.paths import (
    resolve_checkpoint_path,
    resolve_result_path,
    print_and_verify_paths,
    verify_path_exists,
)

SEED = 42

N_BOOTSTRAP = 2000
ALPHA = 0.05
CHECKPOINTS_DIR = Path("checkpoints")
SPLITS_DIR = Path("splits/dedup")
SPLITS_CACHE_DIR = Path("splits_cache/dedup")
DATA_ROOT = Path("data")
DATA_CACHE_ROOT = Path("data_cache")
REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")

MODELS = ["resnet18", "mobilenetv3_small", "custom_none", "custom_se"]


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


@torch.no_grad()
def extract_logits(
    model_name: str,
    split_df: pd.DataFrame,
    split_name: str,
    device: torch.device,
    checkpoint_path: Optional[Union[str, Path]] = None,
    loss_version: str = "v2",
    seed: int = 42,
    checkpoints_dir: Union[str, Path] = "checkpoints",
) -> Dict[str, Any]:
    """Run inference to extract unnormalized logits, labels, and group_ids."""
    if checkpoint_path is None:
        checkpoint_path = resolve_checkpoint_path(model_name, seed=seed, loss_version=loss_version, checkpoints_dir=checkpoints_dir)
    ckpt_path = verify_path_exists(checkpoint_path, description=f"Checkpoint for '{model_name}'", loss_version=loss_version)

    is_custom = "custom" in model_name
    if is_custom:
        att = model_name.replace("custom_", "")
        model = build_custom_net(attention=att, dropout=0.3).to(device)
        img_root = DATA_CACHE_ROOT if DATA_CACHE_ROOT.exists() else DATA_ROOT
        cache_split = SPLITS_CACHE_DIR / f"{split_name}.csv"
        eval_df = pd.read_csv(cache_split) if cache_split.exists() else split_df
    else:
        model = create_model(backbone_name=model_name, num_classes=12, pretrained=False).to(device)
        img_root = DATA_ROOT
        eval_df = split_df

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tfms = get_transforms(image_size=224, resize_dim=256)["test"]
    dataset = CropDiseaseDataset(eval_df, image_root=img_root, transform=tfms)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    all_logits = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if is_custom:
            _, disease_logits = model(images)
            all_logits.append(disease_logits.cpu())
        else:
            logits = model(images)
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = F.softmax(logits_tensor, dim=1)
    confs, preds = torch.max(probs_tensor, dim=1)

    return {
        "logits": logits_tensor.numpy(),
        "probs": probs_tensor.numpy(),
        "confs": confs.numpy(),
        "preds": preds.numpy(),
        "labels": split_df["label"].values,
        "group_ids": split_df["group_id"].values,
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE) with n_bins uniform bins in [0, 1].
    Returns ECE and detailed bin statistics.
    """
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    corrects = (preds == labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    bin_accs = []
    bin_confs = []
    bin_counts = []
    bin_edges = []

    for i in range(n_bins):
        bin_lo = bin_boundaries[i]
        bin_hi = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confs >= bin_lo) & (confs <= bin_hi)
        else:
            in_bin = (confs > bin_lo) & (confs <= bin_hi)

        count = in_bin.sum()
        bin_counts.append(int(count))
        bin_edges.append((float(bin_lo), float(bin_hi)))

        if count > 0:
            acc = float(corrects[in_bin].mean())
            conf = float(confs[in_bin].mean())
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / total_samples) * abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lo + bin_hi) / 2))

    return float(ece), {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
    }


def compute_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute Negative Log-Likelihood (Cross Entropy loss)."""
    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_labels = torch.tensor(labels, dtype=torch.long)
    return float(F.cross_entropy(t_logits, t_labels).item())


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find optimal temperature T on validation set by minimizing NLL."""
    t_logits = torch.tensor(val_logits, dtype=torch.float32)
    t_labels = torch.tensor(val_labels, dtype=torch.long)

    def nll_eval(t_val: float) -> float:
        scaled = t_logits / t_val
        return float(F.cross_entropy(scaled, t_labels).item())

    res = minimize_scalar(nll_eval, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


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


def compute_group_bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    metric_type: str = "accuracy",
    target_class: int = None,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
) -> Tuple[float, float, float]:
    """Group bootstrap 95% CI for accuracy or class-specific F1/recall."""
    rng = np.random.default_rng(SEED)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    if metric_type == "accuracy":
        point_val = accuracy_score(y_true, y_pred)
    elif metric_type == "f1":
        point_val = f1_score(y_true, y_pred, labels=[target_class], average="macro", zero_division=0)
    elif metric_type == "recall":
        point_val = recall_score(y_true, y_pred, labels=[target_class], average="macro", zero_division=0)

    if n_groups <= 1 or len(y_true) == 0:
        return point_val, point_val, point_val

    group_to_idx = {g: np.where(group_ids == g)[0] for g in unique_groups}
    boot_vals = []

    for _ in range(n_boot):
        sampled_g = rng.choice(unique_groups, size=n_groups, replace=True)
        b_idx = np.concatenate([group_to_idx[g] for g in sampled_g])
        bt = y_true[b_idx]
        bp = y_pred[b_idx]

        if metric_type == "accuracy":
            boot_vals.append(accuracy_score(bt, bp))
        elif metric_type == "f1":
            boot_vals.append(f1_score(bt, bp, labels=[target_class], average="macro", zero_division=0))
        elif metric_type == "recall":
            boot_vals.append(recall_score(bt, bp, labels=[target_class], average="macro", zero_division=0))

    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return point_val, lo, hi


def main():
    parser = argparse.ArgumentParser(description="Calibration and Risk Analysis")
    parser.add_argument("--loss-version", type=str, choices=["v1", "v2"], default="v2", help="Loss version ({v1, v2}, default: v2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to analyze")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--splits-dir", type=str, default="splits/dedup", help="Splits directory")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'mps', 'cuda')")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device() if args.device is None else torch.device(args.device)
    reports_dir = Path(args.reports_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    splits_dir = Path(args.splits_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = args.models
    checkpoints_to_load = {
        m: resolve_checkpoint_path(m, seed=args.seed, loss_version=args.loss_version, checkpoints_dir=checkpoints_dir)
        for m in models
    }
    result_json_to_load = resolve_result_path(models[0], seed=args.seed, loss_version=args.loss_version, results_dir=results_dir)

    print_and_verify_paths(
        checkpoints=checkpoints_to_load,
        results={f"{models[0]}_result": result_json_to_load},
        loss_version=args.loss_version,
        header="CALIBRATION & RISK ANALYSIS TARGET PATHS",
    )

    print(f"Device: {device}")

    # Load splits
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    train_df = pd.read_csv(splits_dir / "train.csv")

    with open(result_json_to_load, "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    # ─────────────────────────────────────────────────────────────────────────
    # Requirement 1: Loss implementation & max-softmax distributions
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("REQUIREMENT 1: EXACT LOSS IMPLEMENTATION & MAX-SOFTMAX DISTRIBUTIONS")
    print("=" * 90)

    class_weights = compute_class_weights(train_df, num_classes=12, method="inverse")
    print(f"\n[Exact Loss Specification (loss_version={args.loss_version})]:")
    print(f"  - Loss Version       : {args.loss_version}")
    print("  - Loss Family        : Cross-Entropy (Weighted)")
    print("  - Focal Loss Flag    : False (use_focal=False in get_loss_function)")
    print("  - Label Smoothing    : 0.10 (applied across all models)")
    print("  - Logit Scaling      : None (raw unscaled logits during training, T=1.0)")
    print("  - Pretrained Models  : Single head (12 classes), weighted CE with label smoothing 0.1")
    print("  - Custom Models      : Multi-task dual loss: L_disease + 0.3 * L_crop")
    print("                         L_disease: 12-class weighted CE with label smoothing 0.1")
    print("                         L_crop   : 3-class unweighted CE with label smoothing 0.1")
    print("\n  - Class Weights (Method='inverse', normalized so mean is 1.0):")
    for idx, (cname, w) in enumerate(zip(class_names, class_weights)):
        print(f"      [{idx:2d}] {cname:<30}: {w.item():.4f}")

    print("\nExtracting logits for VAL and TEST splits across models...")
    val_outputs = {}
    test_outputs = {}

    for m in models:
        print(f"  Evaluating {m} on VAL ...")
        val_outputs[m] = extract_logits(
            m,
            val_df,
            "val",
            device,
            checkpoint_path=checkpoints_to_load[m],
            loss_version=args.loss_version,
            seed=args.seed,
            checkpoints_dir=checkpoints_dir,
        )
        print(f"  Evaluating {m} on TEST ...")
        test_outputs[m] = extract_logits(
            m,
            test_df,
            "test",
            device,
            checkpoint_path=checkpoints_to_load[m],
            loss_version=args.loss_version,
            seed=args.seed,
            checkpoints_dir=checkpoints_dir,
        )

    # Max-softmax summary table
    conf_summary_rows = []
    for m in MODELS:
        # VAL split
        val_corr = val_outputs[m]["preds"] == val_outputs[m]["labels"]
        val_c_confs = val_outputs[m]["confs"][val_corr]
        val_i_confs = val_outputs[m]["confs"][~val_corr]

        # TEST split
        test_corr = test_outputs[m]["preds"] == test_outputs[m]["labels"]
        test_c_confs = test_outputs[m]["confs"][test_corr]
        test_i_confs = test_outputs[m]["confs"][~test_corr]

        conf_summary_rows.append({
            "model": m,
            "split": "VAL",
            "correct_n": len(val_c_confs),
            "correct_mean": round(float(val_c_confs.mean()), 4),
            "correct_median": round(float(np.median(val_c_confs)), 4),
            "incorrect_n": len(val_i_confs),
            "incorrect_mean": round(float(val_i_confs.mean()), 4) if len(val_i_confs) > 0 else 0.0,
            "incorrect_median": round(float(np.median(val_i_confs)), 4) if len(val_i_confs) > 0 else 0.0,
            "gap_mean": round(float(val_c_confs.mean() - (val_i_confs.mean() if len(val_i_confs) > 0 else 0)), 4),
        })

        conf_summary_rows.append({
            "model": m,
            "split": "TEST",
            "correct_n": len(test_c_confs),
            "correct_mean": round(float(test_c_confs.mean()), 4),
            "correct_median": round(float(np.median(test_c_confs)), 4),
            "incorrect_n": len(test_i_confs),
            "incorrect_mean": round(float(test_i_confs.mean()), 4) if len(test_i_confs) > 0 else 0.0,
            "incorrect_median": round(float(np.median(test_i_confs)), 4) if len(test_i_confs) > 0 else 0.0,
            "gap_mean": round(float(test_c_confs.mean() - (test_i_confs.mean() if len(test_i_confs) > 0 else 0)), 4),
        })

        # Generate and save histogram
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        bins = np.linspace(0.0, 1.0, 35)
        ax.hist(test_c_confs, bins=bins, alpha=0.65, label=f"Correct (N={len(test_c_confs)})", color="#2e7d32", density=True)
        if len(test_i_confs) > 0:
            ax.hist(test_i_confs, bins=bins, alpha=0.65, label=f"Incorrect (N={len(test_i_confs)})", color="#c62828", density=True)
        ax.axvline(float(test_c_confs.mean()), color="#1b5e20", linestyle="--", linewidth=1.8, label=f"Mean Correct: {test_c_confs.mean():.3f}")
        if len(test_i_confs) > 0:
            ax.axvline(float(test_i_confs.mean()), color="#b71c1c", linestyle="--", linewidth=1.8, label=f"Mean Incorrect: {test_i_confs.mean():.3f}")

        ax.set_title(f"Max-Softmax Confidence Distribution: {m} (Dedup Test Split)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Max-Softmax Probability", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0.0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9)
        fig.tight_layout()
        hist_path = reports_dir / f"confidence_hist_{m}.png"
        fig.savefig(hist_path)
        plt.close(fig)

    df_conf_summary = pd.DataFrame(conf_summary_rows)
    print("\n[Confidence Summary Table]:")
    print(df_conf_summary.to_string(index=False))
    df_conf_summary.to_csv(results_dir / "confidence_summary_val_test.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Requirement 2: ECE (15 bins), Reliability Diagrams & Temperature Scaling
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("REQUIREMENT 2: CALIBRATION ERROR (ECE 15-BINS) & TEMPERATURE SCALING")
    print("=" * 90)

    calib_rows = []
    temperatures = {}
    scaled_test_probs = {}

    for m in models:
        val_logits = val_outputs[m]["logits"]
        val_labels = val_outputs[m]["labels"]
        test_logits = test_outputs[m]["logits"]
        test_labels = test_outputs[m]["labels"]

        # Fit temperature on VAL
        T_opt = fit_temperature(val_logits, val_labels)
        temperatures[m] = T_opt

        # Uncalibrated test metrics
        uncal_probs = test_outputs[m]["probs"]
        ece_before, bin_info_before = compute_ece(uncal_probs, test_labels, n_bins=15)
        nll_before = compute_nll(test_logits, test_labels)

        # Calibrated test metrics
        cal_logits = test_logits / T_opt
        cal_probs = F.softmax(torch.tensor(cal_logits), dim=1).numpy()
        scaled_test_probs[m] = cal_probs
        ece_after, bin_info_after = compute_ece(cal_probs, test_labels, n_bins=15)
        nll_after = compute_nll(cal_logits, test_labels)

        calib_rows.append({
            "model": m,
            "fitted_temperature_T": round(T_opt, 4),
            "test_ece_before": round(ece_before, 4),
            "test_ece_after": round(ece_after, 4),
            "ece_reduction_%": round((ece_before - ece_after) / ece_before * 100, 2),
            "test_nll_before": round(nll_before, 4),
            "test_nll_after": round(nll_after, 4),
            "nll_change": round(nll_after - nll_before, 4),
        })

        # Save reliability diagram
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
        for ax, bin_info, title_suffix, ece_val in [
            (axes[0], bin_info_before, "Uncalibrated (T=1.0)", ece_before),
            (axes[1], bin_info_after, f"Temperature Scaled (T={T_opt:.2f})", ece_after),
        ]:
            accs = bin_info["bin_accs"]
            confs = bin_info["bin_confs"]
            counts = bin_info["bin_counts"]
            midpoints = [b[0] + 1/30 for b in bin_info["bin_edges"]]

            # Plot bars
            ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            ax.bar(midpoints, accs, width=1/16, alpha=0.7, color="#1976d2", edgecolor="#0d47a1", label="Outputs")

            # Calibration gaps
            for x, acc, conf in zip(midpoints, accs, confs):
                if acc > 0:
                    ax.plot([x, x], [acc, conf], color="#d32f2f", linewidth=2.0)

            ax.set_title(f"{m}: {title_suffix}\nECE = {ece_val:.4f}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Confidence", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left")

        fig.tight_layout()
        rel_path = reports_dir / f"reliability_{m}.png"
        fig.savefig(rel_path)
        plt.close(fig)

    df_calib = pd.DataFrame(calib_rows)
    print("\n[Calibration & Temperature Scaling Results (Dedup TEST Split)]:")
    print(df_calib.to_string(index=False))
    df_calib.to_csv(results_dir / "calibration_temperature_scaling.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Requirement 3: Risk-Coverage Curves at EQUAL Coverage (100%, 90%, 80%, 70%, 60%)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("REQUIREMENT 3: RISK-COVERAGE TABLES AT EQUAL COVERAGE (TEMPERATURE-SCALED)")
    print("=" * 90)

    target_coverages = [1.0, 0.9, 0.8, 0.7, 0.6]
    test_labels = test_df["label"].values
    total_test = len(test_labels)

    equal_cov_overall_rows = []
    equal_cov_per_class_rows = []

    for m in models:
        probs = scaled_test_probs[m]
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)

        # Sort test indices by confidence descending
        sorted_indices = np.argsort(-confs)

        for cov in target_coverages:
            n_keep = int(round(cov * total_test))
            kept_idx = sorted_indices[:n_keep]

            kt = test_labels[kept_idx]
            kp = preds[kept_idx]
            min_conf_retained = float(confs[kept_idx[-1]]) if len(kept_idx) > 0 else 0.0

            acc = accuracy_score(kt, kp)
            macro_f1 = f1_score(kt, kp, labels=list(range(12)), average="macro", zero_division=0)
            macro_f1_active = f1_score(kt, kp, labels=np.unique(kt), average="macro", zero_division=0)

            cov_pct_str = f"{int(cov * 100)}%"
            equal_cov_overall_rows.append({
                "model": m,
                "target_coverage": cov_pct_str,
                "retained_count": n_keep,
                "min_retained_conf": round(min_conf_retained, 4),
                "accuracy": round(acc, 4),
                "macro_f1_12class": round(macro_f1, 4),
                "macro_f1_active": round(macro_f1_active, 4),
                "active_classes": len(np.unique(kt)),
            })

            # Per-class retention fraction
            for c_idx, cname in enumerate(class_names):
                total_c = int((test_labels == c_idx).sum())
                kept_c = int((kt == c_idx).sum())
                frac_c = (kept_c / total_c) if total_c > 0 else 0.0

                equal_cov_per_class_rows.append({
                    "model": m,
                    "target_coverage": cov_pct_str,
                    "class_idx": c_idx,
                    "class_name": cname,
                    "total_samples": total_c,
                    "retained_samples": kept_c,
                    "retained_fraction_%": round(frac_c * 100, 1),
                })

    df_equal_overall = pd.DataFrame(equal_cov_overall_rows)
    print("\n[Equal Coverage Overall Performance (Temperature-Scaled)]:")
    print(df_equal_overall.to_string(index=False))
    df_equal_overall.to_csv(results_dir / "risk_coverage_equal_coverage_overall.csv", index=False)

    df_equal_per_class = pd.DataFrame(equal_cov_per_class_rows)
    df_equal_per_class.to_csv(results_dir / "risk_coverage_equal_coverage_per_class.csv", index=False)

    # Pivot table of per-class retention fraction per model
    print("\n[Per-Class Retained Fraction (%) Across Coverage Levels]:")
    for m in models:
        print(f"\n--- Model: {m} ---")
        m_df = df_equal_per_class[df_equal_per_class["model"] == m]
        piv = m_df.pivot(index="class_name", columns="target_coverage", values="retained_fraction_%")[
            ["100%", "90%", "80%", "70%", "60%"]
        ]
        print(piv.to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # Requirement 4: Papaya Split by Border Brightness (>=230 vs <230)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("REQUIREMENT 4: PAPAYA PERFORMANCE SPLIT BY BORDER BRIGHTNESS (>=230 vs <230)")
    print("=" * 90)

    papaya_test = test_df[test_df["crop"] == "Papaya"].copy()
    print("Computing border brightness for Papaya test images...")
    papaya_test["border_brightness"] = [compute_border_brightness(p) for p in papaya_test["image_path"]]
    papaya_test["is_white_border"] = papaya_test["border_brightness"] >= 230.0

    papaya_indices = papaya_test.index.values
    papaya_classes = [c for c in class_names if c.startswith("Papaya___")]

    papaya_split_rows = []

    for m in models:
        preds_all = test_outputs[m]["preds"]

        for cname in papaya_classes:
            c_idx = class_names.index(cname)
            cls_df = papaya_test[papaya_test["class_name"] == cname]

            # High brightness (>= 230)
            hi_df = cls_df[cls_df["is_white_border"]]
            hi_idx = hi_df.index.values

            # Low brightness (< 230)
            lo_df = cls_df[~cls_df["is_white_border"]]
            lo_idx = lo_df.index.values

            for b_label, sub_idx, sub_df in [(">=230 (White)", hi_idx, hi_df), ("<230 (Dark)", lo_idx, lo_df)]:
                n_img = len(sub_df)
                n_grp = len(np.unique(sub_df["group_id"])) if n_img > 0 else 0

                if n_img > 0:
                    yt = sub_df["label"].values
                    yp = preds_all[sub_idx]
                    grp = sub_df["group_id"].values

                    acc, acc_lo, acc_hi = compute_group_bootstrap_metric(yt, yp, grp, metric_type="accuracy")
                    rec, rec_lo, rec_hi = compute_group_bootstrap_metric(yt, yp, grp, metric_type="recall", target_class=c_idx)
                else:
                    acc = acc_lo = acc_hi = 0.0
                    rec = rec_lo = rec_hi = 0.0

                papaya_split_rows.append({
                    "model": m,
                    "class_name": cname,
                    "brightness_bucket": b_label,
                    "n_images": n_img,
                    "n_groups": n_grp,
                    "accuracy": round(acc, 4),
                    "acc_95_ci": f"[{acc_lo:.4f}, {acc_hi:.4f}]" if n_img > 0 else "N/A",
                    "recall": round(rec, 4),
                    "rec_95_ci": f"[{rec_lo:.4f}, {rec_hi:.4f}]" if n_img > 0 else "N/A",
                })

    df_papaya_split = pd.DataFrame(papaya_split_rows)
    print("\n[Papaya Performance Split by Border Brightness (Group-Bootstrap 95% CI)]:")
    print(df_papaya_split.to_string(index=False))
    df_papaya_split.to_csv(results_dir / "papaya_border_brightness_split_metrics.csv", index=False)

    print("\n" + "=" * 90)
    print("ALL 4 REQUIREMENTS EVALUATED AND OUTPUTTED SUCCESSFULLY.")
    print("=" * 90)


if __name__ == "__main__":
    main()
