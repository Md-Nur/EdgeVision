"""
Robustness Stress Testing Harness for Multi-Crop Leaf Disease Diagnosis.

Evaluates trained model checkpoints on the exact-MD5-dedup test split under
7 controlled environmental corruptions across 3 severity levels:
1. Gaussian noise
2. Gaussian blur
3. Brightness shift
4. Contrast shift
5. Random shadow overlay
6. JPEG compression
7. Background replacement (HSV/ExG leaf mask, replacing non-leaf background)

Outputs:
- results/stress_<model>.csv (or results_dry/stress_<model>.csv in dry-run mode)
- reports/stress_<model>.png (or reports_dry/stress_<model>.png in dry-run mode)
"""

import argparse
import io
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms

from src.models.backbones import CropDiseaseClassifier
from src.models.custom_net import build_custom_net
from src.xai.quantitative import segment_leaf_foreground
from src.utils.paths import resolve_checkpoint_path, print_and_verify_paths


CLASS_NAMES = [
    "Papaya___Anthracnose",
    "Papaya___Black_Spot",
    "Papaya___Curl_Virus",
    "Papaya___Healthy",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Potato___PGR",
    "Rice___Bacterial_Blight",
    "Rice___Blast",
    "Rice___Brown_Spot",
    "Rice___Healthy",
]

MODEL_SPECS = {
    "resnet18": {"type": "pretrained", "arch": "resnet18"},
    "mobilenetv3_small": {"type": "pretrained", "arch": "mobilenetv3_small_100"},
    "mobilenetv3_large": {"type": "pretrained", "arch": "mobilenetv3_large_100"},
    "efficientnet_b0": {"type": "pretrained", "arch": "efficientnet_b0"},
    "custom_none": {"type": "custom", "attention": "none"},
    "custom_se": {"type": "custom", "attention": "se"},
    "custom_cbam": {"type": "custom", "attention": "cbam"},
}


# ==============================================================================
# Controlled Corruptions at 3 Severity Levels
# ==============================================================================

def apply_gaussian_noise(image: Image.Image, severity: int, seed: Optional[int] = None) -> Image.Image:
    """Add zero-mean Gaussian noise at 3 calibrated severity levels."""
    scales = {1: 18.0, 2: 40.0, 3: 75.0}
    scale = scales.get(severity, 25.0)
    rng = np.random.default_rng(seed)
    arr = np.array(image).astype(np.float32)
    noise = rng.normal(0.0, scale, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def apply_gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian blur simulating camera defocus / motion blur."""
    radii = {1: 1.5, 2: 3.5, 3: 6.0}
    radius = radii.get(severity, 2.0)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_shift(image: Image.Image, severity: int) -> Image.Image:
    """Simulate low-light / underexposure field conditions."""
    factors = {1: 0.70, 2: 0.45, 3: 0.25}
    factor = factors.get(severity, 0.5)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_contrast_shift(image: Image.Image, severity: int) -> Image.Image:
    """Simulate hazy, overcast, or washed-out lighting."""
    factors = {1: 0.65, 2: 0.40, 3: 0.20}
    factor = factors.get(severity, 0.5)
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_random_shadow(image: Image.Image, severity: int, seed: Optional[int] = None) -> Image.Image:
    """Simulate outdoor cast shadows across the leaf."""
    attenuations = {1: 0.70, 2: 0.45, 3: 0.25}
    factor = attenuations.get(severity, 0.5)
    arr = np.array(image).astype(np.float32)
    h, w, _ = arr.shape

    # Reproducible diagonal gradient shadow band
    rng = np.random.default_rng(seed)
    flip_x = rng.choice([False, True])
    x_lin = np.linspace(0.0, 1.0, w)
    if flip_x:
        x_lin = np.flip(x_lin)
    y_lin = np.linspace(0.0, 1.0, h)
    xx, yy = np.meshgrid(x_lin, y_lin)
    diag = np.clip((xx + yy) / 1.5, 0.0, 1.0)
    shadow_mask = 1.0 - (1.0 - factor) * diag[:, :, np.newaxis]

    shadowed_arr = np.clip(arr * shadow_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(shadowed_arr)


def apply_jpeg_compression(image: Image.Image, severity: int) -> Image.Image:
    """Simulate mobile upload compression artifacts."""
    qualities = {1: 50, 2: 25, 3: 10}
    quality = qualities.get(severity, 30)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_background_replacement(
    image: Image.Image,
    severity: int,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Replace non-leaf background pixels using HSV & ExG segmentation mask.
    Severity controls the replacement strength/type:
      1: 40% blend with high-variance outdoor texture
      2: 75% blend with structured multi-frequency texture
      3: 100% full background replacement with high-contrast synthetic texture
    """
    arr = np.array(image)
    leaf_mask = segment_leaf_foreground(arr)  # True = leaf, False = background

    rng = np.random.default_rng(seed)
    h, w, c = arr.shape

    # Generate synthetic outdoor texture (perlin-like low frequency noise + color wash)
    small_h, small_w = max(4, h // 16), max(4, w // 16)
    noise_coarse = rng.uniform(0.1, 0.9, size=(small_h, small_w, 3)).astype(np.float32)
    noise_img = Image.fromarray((noise_coarse * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    texture = np.array(noise_img).astype(np.float32)

    # Add high-frequency perturbation
    fine_noise = rng.normal(0.0, 20.0, size=(h, w, 3)).astype(np.float32)
    texture = np.clip(texture + fine_noise, 0.0, 255.0)

    # Severity blend factors
    blend_factors = {1: 0.40, 2: 0.75, 3: 1.00}
    alpha = blend_factors.get(severity, 1.0)

    leaf_3d = leaf_mask[:, :, np.newaxis]
    bg_orig = arr.astype(np.float32)
    bg_perturbed = (1.0 - alpha) * bg_orig + alpha * texture

    composite = np.where(leaf_3d, bg_orig, bg_perturbed)
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))


CORRUPTION_FUNCTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "gaussian_blur": apply_gaussian_blur,
    "brightness_shift": apply_brightness_shift,
    "contrast_shift": apply_contrast_shift,
    "random_shadow": apply_random_shadow,
    "jpeg_compression": apply_jpeg_compression,
    "background_replacement": apply_background_replacement,
}


# ==============================================================================
# Model Loading Helper
# ==============================================================================

def load_trained_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = 12,
) -> Tuple[nn.Module, bool]:
    """Load model architecture and checkpoint weights."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model_name: {model_name}. Allowed: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_name]
    is_custom = (spec["type"] == "custom")

    if is_custom:
        model = build_custom_net(attention=spec["attention"], dropout=0.3)
    else:
        model = CropDiseaseClassifier(
            backbone_name=spec["arch"],
            num_classes=num_classes,
            pretrained=False,
            dropout=0.3,
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, is_custom


# ==============================================================================
# Evaluation under Corruptions
# ==============================================================================

def evaluate_on_corrupted_split(
    model: nn.Module,
    is_custom: bool,
    df: pd.DataFrame,
    image_root: Path,
    eval_transform: Callable,
    device: torch.device,
    corruption_fn: Optional[Callable] = None,
    severity: int = 0,
    seed: int = 42,
    batch_size: int = 32,
) -> Tuple[float, float, np.ndarray]:
    """Evaluate model on dataset applying a specific corruption function and severity."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    # Process in batches
    n_samples = len(df)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_rows = df.iloc[start_idx:end_idx]

        batch_tensors = []
        batch_targets = []

        for row_idx, (_, row) in enumerate(batch_rows.iterrows()):
            rel_path = row["image_path"]
            full_path = image_root / rel_path
            if not full_path.exists():
                full_path = Path(rel_path)
            if not full_path.exists():
                raise FileNotFoundError(f"Image not found: {full_path}")

            img = Image.open(full_path).convert("RGB")

            # Apply corruption if severity > 0
            if corruption_fn is not None and severity > 0:
                sample_seed = seed + start_idx + row_idx
                # Check if corruption_fn takes seed
                if corruption_fn in (apply_gaussian_noise, apply_random_shadow, apply_background_replacement):
                    img = corruption_fn(img, severity, seed=sample_seed)
                else:
                    img = corruption_fn(img, severity)

            t = eval_transform(img)
            batch_tensors.append(t)
            batch_targets.append(int(row["label"]))

        images = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            if is_custom:
                crop_logits, disease_logits = model(images)
                _, flat_preds, masked_preds = model.predict_with_masking(crop_logits, disease_logits)
                preds = flat_preds.cpu().numpy()
            else:
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(batch_targets)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    labels_order = list(range(len(CLASS_NAMES)))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_order, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels_order, average=None, zero_division=0)

    return macro_f1, acc, per_class_f1


def plot_stress_results(
    results_df: pd.DataFrame,
    clean_f1: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot Macro-F1 degradation curves versus corruption severity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot clean baseline
    ax.axhline(
        clean_f1,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Clean Baseline (F1: {clean_f1:.3f})",
        alpha=0.85,
    )

    markers = ["o", "s", "^", "D", "v", "<", ">"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(CORRUPTION_FUNCTIONS)))

    corruptions = results_df["corruption"].unique()
    for idx, corr in enumerate(corruptions):
        if corr == "clean":
            continue
        corr_data = results_df[results_df["corruption"] == corr].sort_values("severity")
        # Prepend severity 0 (clean)
        x_vals = [0] + list(corr_data["severity"].values)
        y_vals = [clean_f1] + list(corr_data["macro_f1"].values)

        label_name = corr.replace("_", " ").title()
        ax.plot(
            x_vals,
            y_vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=2.2,
            markersize=7,
            label=label_name,
        )

    ax.set_title(f"Robustness Stress Test Degradation: {model_name} (MD5-Dedup Test Split)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Corruption Severity Level (0 = Clean)", fontsize=11)
    ax.set_ylabel("Validation / Test Macro-F1", fontsize=11)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0 (Clean)", "1 (Mild)", "2 (Medium)", "3 (Severe)"])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT SAVED]: Stress test degradation plot saved to: {output_path}")


# ==============================================================================
# Main Routine
# ==============================================================================

def run_stress_test(
    model_name: str,
    checkpoint_path: Path,
    splits_dir: Path = Path("splits"),
    results_dir: Path = Path("results"),
    reports_dir: Path = Path("reports"),
    image_root: Path = Path("."),
    sample_size: Optional[int] = None,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run full stress test suite for a given model."""
    print("=" * 80)
    print(f"STRESS TESTING MODEL: {model_name}")
    print(f"Checkpoint: {checkpoint_path} | Device: {device} | Dry Run: {dry_run}")
    print("=" * 80)

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact-MD5-dedup test split
    dedup_test_csv = splits_dir / "dedup" / "test.csv"
    if not dedup_test_csv.exists():
        raise FileNotFoundError(f"Dedup test split not found: {dedup_test_csv}")

    test_df = pd.read_csv(dedup_test_csv)
    total_available = len(test_df)
    if sample_size and sample_size > 0:
        test_df = test_df.head(min(sample_size, total_available)).reset_index(drop=True)
        print(f"[SUBSET MODE]: Evaluating on {len(test_df)} images (out of {total_available:,})...")
    else:
        print(f"[FULL EVAL MODE]: Evaluating on all {len(test_df):,} dedup test images...")

    # 2. Setup model
    model, is_custom = load_trained_model(model_name, checkpoint_path, device, num_classes=len(CLASS_NAMES))

    # Standard evaluation transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # 3. Clean baseline evaluation
    print("\n--- Evaluating Clean Baseline (Severity 0) ---")
    clean_f1, clean_acc, _ = evaluate_on_corrupted_split(
        model=model,
        is_custom=is_custom,
        df=test_df,
        image_root=image_root,
        eval_transform=eval_transform,
        device=device,
        corruption_fn=None,
        severity=0,
        seed=seed,
        batch_size=batch_size,
    )
    print(f"Clean Baseline -> Macro-F1: {clean_f1:.4f} | Accuracy: {clean_acc:.4f}")

    records = [{
        "model": model_name,
        "corruption": "clean",
        "severity": 0,
        "macro_f1": clean_f1,
        "accuracy": clean_acc,
        "delta_macro_f1": 0.0,
        "num_samples": len(test_df),
    }]

    # 4. Controlled corruptions evaluation
    print("\n--- Evaluating Controlled Corruptions (7 types x 3 severities) ---")
    for corr_name, corr_fn in CORRUPTION_FUNCTIONS.items():
        for severity in [1, 2, 3]:
            f1, acc, _ = evaluate_on_corrupted_split(
                model=model,
                is_custom=is_custom,
                df=test_df,
                image_root=image_root,
                eval_transform=eval_transform,
                device=device,
                corruption_fn=corr_fn,
                severity=severity,
                seed=seed,
                batch_size=batch_size,
            )
            delta = f1 - clean_f1
            print(f"  {corr_name:<25} Sev {severity} -> Macro-F1: {f1:.4f} (Delta: {delta:+.4f}) | Acc: {acc:.4f}")

            records.append({
                "model": model_name,
                "corruption": corr_name,
                "severity": severity,
                "macro_f1": f1,
                "accuracy": acc,
                "delta_macro_f1": delta,
                "num_samples": len(test_df),
            })

    results_df = pd.DataFrame(records)

    # 5. Save CSV & Plot
    out_csv = results_dir / f"stress_{model_name}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[RESULTS SAVED]: Stress test results written to: {out_csv}")

    out_plot = reports_dir / f"stress_{model_name}.png"
    plot_stress_results(results_df, clean_f1, model_name, out_plot)

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Crop Disease Robustness Stress Testing.")
    parser.add_argument("--model", type=str, default="resnet18", help=f"Model to test. Choices: {list(MODEL_SPECS.keys())}")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If omitted, resolved via --model/--seed/--loss-version.")
    parser.add_argument("--loss-version", choices=["v1", "v2"], default="v2", help="Loss version used for training (default: v2). Determines checkpoint name suffix.")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Directory containing splits")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output CSV")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for output plots")
    parser.add_argument("--image-root", type=str, default=".", help="Root directory for dataset image paths")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset sample size (e.g. 64 for dry run/smoke test)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation (default: 32)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', 'mps', 'cuda'). Defaults to 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run dry run with 64-image subset writing to *_dry directories")
    args = parser.parse_args()

    # Dry-run directory isolation rule
    if args.dry_run:
        results_dir = Path("results_dry")
        reports_dir = Path("reports_dry")
        sample_size = 64 if args.sample_size is None else args.sample_size
        checkpoints_dir = Path("checkpoints_dry")
    else:
        results_dir = Path(args.results_dir)
        reports_dir = Path(args.reports_dir)
        sample_size = args.sample_size
        checkpoints_dir = Path("checkpoints")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = resolve_checkpoint_path(
            args.model,
            seed=args.seed,
            loss_version=args.loss_version,
            checkpoints_dir=checkpoints_dir,
        )

    # Print resolved paths and exit with error if missing (no silent fallback)
    print_and_verify_paths(
        checkpoints={args.model: ckpt_path},
        results={},
        loss_version=args.loss_version,
        header="STRESS TEST",
    )

    run_stress_test(
        model_name=args.model,
        checkpoint_path=ckpt_path,
        splits_dir=Path(args.splits_dir),
        results_dir=results_dir,
        reports_dir=reports_dir,
        image_root=Path(args.image_root),
        sample_size=sample_size,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

