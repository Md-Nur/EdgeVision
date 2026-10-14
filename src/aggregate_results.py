"""
Results Aggregator for Stage A Multi-Crop Disease Diagnosis.

Responsibilities:
1. Load evaluation JSON files from results/<model>_seed42.json (or results/*.json).
2. Compute 95% group-bootstrap confidence interval (CI) for macro-F1 (1,000 resamples of group_ids with replacement)
   for both the standard test split and the exact-MD5-dedup test split.
3. Build results/stage_a_table.csv with columns:
   - model
   - macro-F1 (dedup test)
   - macro-F1 95% CI (dedup)
   - macro-F1 (standard test)
   - macro-F1 95% CI (standard)
   - accuracy
   - accuracy (dedup)
   - params
   - size_MB
   - GFLOPs
   - latency (ms)
4. Save confusion matrix PNG for every model into reports/<model>_confusion_matrix.png.
5. Print the formatted Markdown summary table.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score


STAGE_A_CANONICAL_MODELS = [
    "resnet18",
    "mobilenetv3_small",
    "mobilenetv3_large",
    "efficientnet_b0",
    "custom_none",
    "custom_se",
    "custom_cbam",
]


def compute_group_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute 95% group-bootstrap confidence interval for macro-F1.
    Resamples group_ids with replacement, collecting all samples from each sampled group.
    
    Returns:
        (observed_f1, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_ids = np.asarray(group_ids)

    # Observed macro-F1
    observed_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    if n_groups == 0:
        return observed_f1, observed_f1, observed_f1

    # Pre-index group mappings for fast lookup
    group_to_indices: Dict[Any, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        group_to_indices.setdefault(gid, []).append(idx)

    rng = np.random.default_rng(seed)
    bootstrap_f1s = []

    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sample_indices: List[int] = []
        for gid in sampled_groups:
            sample_indices.extend(group_to_indices[gid])

        b_true = y_true[sample_indices]
        b_pred = y_pred[sample_indices]
        b_f1 = f1_score(b_true, b_pred, average="macro", zero_division=0)
        bootstrap_f1s.append(b_f1)

    lower_p = 100.0 * (alpha / 2.0)
    upper_p = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_f1s, lower_p))
    ci_upper = float(np.percentile(bootstrap_f1s, upper_p))

    return round(observed_f1, 4), round(ci_lower, 4), round(ci_upper, 4)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    model_name: str,
    output_path: Path,
    title_suffix: str = "Standard Test Split",
) -> None:
    """Plot and save a publication-quality confusion matrix heatmap."""
    cm = np.asarray(cm)
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Calculate row-normalized percentages for annotating
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    # Shorten class names for readability if needed
    labels = [c.replace("___", " ") for c in class_names]

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="lightgray",
    )
    plt.title(f"{model_name} Confusion Matrix ({title_suffix})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def find_result_file(model_name: str, results_dir: Path) -> Optional[Path]:
    """Find the result JSON for a given model (e.g. resnet18_seed42.json or resnet18.json)."""
    candidates = [
        results_dir / f"{model_name}_seed42.json",
        results_dir / f"{model_name}.json",
    ]
    # Check normalized aliases
    clean = model_name.replace("-", "_")
    candidates.append(results_dir / f"{clean}_seed42.json")
    candidates.append(results_dir / f"{clean}.json")

    for c in candidates:
        if c.exists():
            return c

    # Fuzzy match by prefix
    for p in results_dir.glob(f"{model_name}*.json"):
        return p

    return None


def aggregate_stage_a(
    results_dir: str | Path = "results",
    output_table_path: str | Path = "results/stage_a_table.csv",
    reports_dir: str | Path = "reports",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate Stage A model evaluations into a summary table and generate confusion matrices."""
    results_dir = Path(results_dir)
    output_table_path = Path(output_table_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_table_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\n" + "=" * 80)
    print("STAGE A: AGGREGATING RESULTS & COMPUTING GROUP-BOOTSTRAP CIs")
    print(f"Results Dir: {results_dir} | Output: {output_table_path} | Reports: {reports_dir}")
    print("=" * 80)

    # Process models in canonical order
    for model_name in STAGE_A_CANONICAL_MODELS:
        json_file = find_result_file(model_name, results_dir)
        if not json_file:
            print(f"[WARNING] No result JSON found for model: {model_name}")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Standard test split metrics
        std_test = data.get("standard_test", {})
        if not std_test:
            # Fallback if old flat schema
            std_test = data

        # Dedup test split metrics
        dedup_test = data.get("dedup_test", {})

        std_macro_f1 = std_test.get("macro_f1", 0.0)
        std_acc = std_test.get("accuracy", 0.0)
        dedup_macro_f1 = dedup_test.get("macro_f1", 0.0) if dedup_test else std_macro_f1
        dedup_acc = dedup_test.get("accuracy", 0.0) if dedup_test else std_acc

        # Group-bootstrap CIs
        # Dedup split CI
        if dedup_test and "y_true" in dedup_test and "y_pred" in dedup_test and "group_ids" in dedup_test:
            _, dedup_ci_l, dedup_ci_u = compute_group_bootstrap_ci(
                dedup_test["y_true"],
                dedup_test["y_pred"],
                dedup_test["group_ids"],
                n_bootstraps=n_bootstraps,
                seed=seed,
            )
            dedup_ci_str = f"[{dedup_ci_l:.4f}, {dedup_ci_u:.4f}]"
        else:
            dedup_ci_str = "N/A"

        # Standard split CI
        if "y_true" in std_test and "y_pred" in std_test and "group_ids" in std_test:
            _, std_ci_l, std_ci_u = compute_group_bootstrap_ci(
                std_test["y_true"],
                std_test["y_pred"],
                std_test["group_ids"],
                n_bootstraps=n_bootstraps,
                seed=seed,
            )
            std_ci_str = f"[{std_ci_l:.4f}, {std_ci_u:.4f}]"
        else:
            std_ci_str = "N/A"

        # Model efficiency
        params = data.get("parameter_count", data.get("params", 0))
        size_mb = data.get("model_size_mb", data.get("size_mb", 0.0))
        gflops = data.get("gflops", 0.0)
        latency = data.get("cpu_latency_ms", data.get("latency", 0.0))

        # Save confusion matrix PNG into reports/
        class_names = data.get("class_names")
        if not class_names and "per_class" in std_test:
            class_names = list(std_test["per_class"].keys())

        cm = std_test.get("confusion_matrix")
        if cm and class_names:
            cm_png = reports_dir / f"{model_name}_confusion_matrix.png"
            plot_and_save_confusion_matrix(
                cm=np.array(cm),
                class_names=class_names,
                model_name=model_name,
                output_path=cm_png,
                title_suffix="Standard Test Split",
            )

        # Also dedup CM if present
        if dedup_test and "confusion_matrix" in dedup_test and class_names:
            dedup_cm_png = reports_dir / f"{model_name}_dedup_confusion_matrix.png"
            plot_and_save_confusion_matrix(
                cm=np.array(dedup_test["confusion_matrix"]),
                class_names=class_names,
                model_name=model_name,
                output_path=dedup_cm_png,
                title_suffix="Dedup Test Split",
            )

        row = {
            "model": model_name,
            "macro-F1 (dedup test)": round(float(dedup_macro_f1), 4),
            "macro-F1 (dedup test) 95% CI": dedup_ci_str,
            "macro-F1 (standard test)": round(float(std_macro_f1), 4),
            "macro-F1 (standard test) 95% CI": std_ci_str,
            "accuracy": round(float(std_acc), 4),
            "accuracy (dedup)": round(float(dedup_acc), 4),
            "params": int(params),
            "size_MB": round(float(size_mb), 2),
            "GFLOPs": round(float(gflops), 3),
            "latency": round(float(latency), 2),
        }
        rows.append(row)

    if not rows:
        print("[ERROR] No model rows aggregated. Check results/ directory.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(output_table_path, index=False)
    print(f"\n[SAVED] Stage A summary table written to: {output_table_path}")

    # Display clean table
    print("\n" + "=" * 100)
    print("STAGE A EVALUATION SUMMARY TABLE:")
    print("=" * 100)
    display_cols = [
        "model",
        "macro-F1 (dedup test)",
        "macro-F1 (dedup test) 95% CI",
        "macro-F1 (standard test)",
        "macro-F1 (standard test) 95% CI",
        "accuracy",
        "params",
        "size_MB",
        "GFLOPs",
        "latency",
    ]
    print(df[display_cols].to_string(index=False))
    print("=" * 100 + "\n")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage A results and compute CIs.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory with results/*.json")
    parser.add_argument("--output-table", type=str, default="results/stage_a_table.csv", help="Output CSV table path")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for confusion matrix PNGs")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed (default: 42)")
    args = parser.parse_args()

    aggregate_stage_a(
        results_dir=args.results_dir,
        output_table_path=args.output_table,
        reports_dir=args.reports_dir,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
