"""
Explainable AI (XAI) Background Attention Analysis via Grad-CAM++.

Quantifies shortcut learning by measuring the fraction of saliency mass that falls
outside the segmented leaf foreground (spurious-background attention ratio).

Generates:
1. Per-class spurious background attention table (results/xai_background_<model>.csv
   or results_dry/xai_background_<model>.csv in dry-run mode).
2. Diagnostic visual overlays for 5 correct and 5 incorrect test images per class
   saved to reports/xai/<model>/<class_name>/ (or reports_dry/xai/<model>/...).
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from src.models.backbones import CropDiseaseClassifier
from src.models.custom_net import build_custom_net
from src.xai.gradcam import GradCAM
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


def load_model_for_xai(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = 12,
) -> Tuple[nn.Module, nn.Module, bool]:
    """
    Load model and determine target layer for Grad-CAM++.
    Returns: (model, target_layer, is_custom)
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model_name: {model_name}. Allowed: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_name]
    is_custom = (spec["type"] == "custom")

    if is_custom:
        model = build_custom_net(attention=spec["attention"], dropout=0.3)
        target_layer = model.stages[-1]
    else:
        model = CropDiseaseClassifier(
            backbone_name=spec["arch"],
            num_classes=num_classes,
            pretrained=False,
            dropout=0.3,
        )
        target_layer = model.get_target_layer_for_cam()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, target_layer, is_custom


def save_diagnostic_overlay(
    image_pil: Image.Image,
    leaf_mask: np.ndarray,
    cam_heatmap: np.ndarray,
    true_label: int,
    pred_label: int,
    is_correct: bool,
    spurious_ratio: float,
    output_path: Path,
) -> None:
    """Save high-resolution 3-panel diagnostic visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = image_pil.size

    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray(cam_heatmap).resize((w, h), Image.BILINEAR))

    # Leaf mask visual (Green for leaf, dark grey for background)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[leaf_mask] = [46, 204, 113]      # Bright leaf green
    mask_rgb[~leaf_mask] = [52, 73, 94]       # Dark slate background

    # Colormap for Grad-CAM
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps["jet"]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap("jet")
    cam_colored = np.uint8(255 * cmap(cam_resized)[:, :, :3])
    img_np = np.array(image_pil)
    blended = np.uint8(0.55 * cam_colored + 0.45 * img_np)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    axes[0].imshow(img_np)
    axes[0].set_title(f"Input: {CLASS_NAMES[true_label]}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_rgb)
    fg_pct = leaf_mask.mean() * 100
    axes[1].set_title(f"Leaf Mask (Leaf: {fg_pct:.1f}% | BG: {100-fg_pct:.1f}%)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    status_str = "CORRECT" if is_correct else "INCORRECT"
    status_color = "green" if is_correct else "red"
    axes[2].imshow(blended)
    axes[2].set_title(
        f"Grad-CAM++ | Pred: {CLASS_NAMES[pred_label]} [{status_str}]\nSpurious Attn Outside Leaf: {spurious_ratio:.1%}",
        fontsize=10,
        fontweight="bold",
        color=status_color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_xai_background_analysis(
    model_name: str,
    checkpoint_path: Path,
    splits_dir: Path = Path("splits"),
    results_dir: Path = Path("results"),
    reports_dir: Path = Path("reports"),
    image_root: Path = Path("."),
    sample_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dry_run: bool = False,
    max_overlays_per_class: int = 5,
) -> pd.DataFrame:
    """Analyze spurious background attention and generate diagnostic overlays."""
    print("=" * 80)
    print(f"XAI BACKGROUND ATTENTION ANALYSIS: {model_name}")
    print(f"Checkpoint: {checkpoint_path} | Device: {device} | Dry Run: {dry_run}")
    print("=" * 80)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_reports_dir = reports_dir / "xai" / model_name
    model_reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact-MD5-dedup test split
    dedup_test_csv = splits_dir / "dedup" / "test.csv"
    if not dedup_test_csv.exists():
        raise FileNotFoundError(f"Dedup test split not found: {dedup_test_csv}")

    test_df = pd.read_csv(dedup_test_csv)
    total_available = len(test_df)
    if sample_size and sample_size > 0:
        test_df = test_df.head(min(sample_size, total_available)).reset_index(drop=True)
        print(f"[SUBSET MODE]: Evaluating XAI on {len(test_df)} images (out of {total_available:,})...")
    else:
        print(f"[FULL EVAL MODE]: Evaluating XAI on all {len(test_df):,} dedup test images...")

    # 2. Setup model and Grad-CAM++
    model, target_layer, is_custom = load_model_for_xai(model_name, checkpoint_path, device, num_classes=len(CLASS_NAMES))
    gradcam = GradCAM(model=model, target_layer=target_layer)

    # Evaluation transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Tracking storage
    per_image_records: List[Dict[str, Any]] = []
    # Counters for saving overlays: class_idx -> {"correct": count, "incorrect": count}
    saved_overlays = {i: {"correct": 0, "incorrect": 0} for i in range(len(CLASS_NAMES))}

    total_images = len(test_df)
    for idx, row in test_df.iterrows():
        rel_path = row["image_path"]
        full_path = image_root / rel_path
        if not full_path.exists():
            full_path = Path(rel_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        img_pil = Image.open(full_path).convert("RGB")
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # 1. Segment leaf foreground
        leaf_mask = segment_leaf_foreground(img_np)
        bg_mask = ~leaf_mask

        # 2. Forward pass & prediction
        input_tensor = eval_transform(img_pil).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        with torch.no_grad():
            if is_custom:
                crop_logits, disease_logits = model(input_tensor)
                _, flat_preds, masked_preds = model.predict_with_masking(crop_logits, disease_logits)
                pred_label = masked_preds.item()
            else:
                outputs = model(input_tensor)
                pred_label = torch.argmax(outputs, dim=1).item()

        true_label = int(row["label"])
        is_correct = (pred_label == true_label)

        # 3. Generate Grad-CAM++ for the predicted class
        cam_heatmap = gradcam.generate_cam(input_tensor, target_class=pred_label, use_gradcam_plusplus=True)

        # Resize CAM to full image resolution
        cam_resized = np.array(Image.fromarray(cam_heatmap).resize((w, h), Image.BILINEAR))

        # 4. Compute spurious background attention mass
        total_mass = float(cam_resized.sum())
        if total_mass > 0.0:
            bg_mass = float(cam_resized[bg_mask].sum())
            spurious_ratio = bg_mass / total_mass
        else:
            spurious_ratio = 0.0

        per_image_records.append({
            "image_path": str(rel_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "class_name": CLASS_NAMES[true_label],
            "is_correct": is_correct,
            "spurious_attention_ratio": spurious_ratio,
        })

        # 5. Save overlay if quota not met
        cat = "correct" if is_correct else "incorrect"
        if saved_overlays[true_label][cat] < max_overlays_per_class:
            saved_overlays[true_label][cat] += 1
            overlay_idx = saved_overlays[true_label][cat]
            spurious_pct = int(round(spurious_ratio * 100))
            class_folder = model_reports_dir / CLASS_NAMES[true_label]
            out_img = class_folder / f"{cat}_{overlay_idx:02d}_spurious_{spurious_pct}pct.png"
            save_diagnostic_overlay(
                image_pil=img_pil,
                leaf_mask=leaf_mask,
                cam_heatmap=cam_heatmap,
                true_label=true_label,
                pred_label=pred_label,
                is_correct=is_correct,
                spurious_ratio=spurious_ratio,
                output_path=out_img,
            )

        if (idx + 1) % 16 == 0 or (idx + 1) == total_images:
            print(f"  Processed [{idx + 1:3d}/{total_images:3d}] images...")

    gradcam.remove_hooks()

    # Aggregate statistics per class
    eval_df = pd.DataFrame(per_image_records)
    summary_records = []

    print("\n" + "=" * 80)
    print(f"SPURIOUS BACKGROUND ATTENTION SUMMARY: {model_name}")
    print(f"{'Class Name':<28} | {'Total':<5} | {'Correct':<7} | {'Mean Spurious':<13} | {'Corr Spurious':<13} | {'Incorr Spurious':<15}")
    print("-" * 88)

    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_sub = eval_df[eval_df["true_label"] == c_idx]
        tot = len(c_sub)
        if tot == 0:
            summary_records.append({
                "class_index": c_idx,
                "class_name": c_name,
                "total_samples": 0,
                "correct_samples": 0,
                "incorrect_samples": 0,
                "mean_spurious_attention": 0.0,
                "spurious_attention_correct": 0.0,
                "spurious_attention_incorrect": 0.0,
            })
            continue

        c_corr = c_sub[c_sub["is_correct"]]
        c_incorr = c_sub[~c_sub["is_correct"]]

        mean_spurious = float(c_sub["spurious_attention_ratio"].mean())
        corr_spurious = float(c_corr["spurious_attention_ratio"].mean()) if len(c_corr) > 0 else 0.0
        incorr_spurious = float(c_incorr["spurious_attention_ratio"].mean()) if len(c_incorr) > 0 else 0.0

        print(
            f"{c_name:<28} | {tot:<5d} | {len(c_corr):<7d} | "
            f"{mean_spurious:11.2%}   | {corr_spurious:11.2%}   | {incorr_spurious:13.2%}"
        )

        summary_records.append({
            "class_index": c_idx,
            "class_name": c_name,
            "total_samples": tot,
            "correct_samples": len(c_corr),
            "incorrect_samples": len(c_incorr),
            "mean_spurious_attention": round(mean_spurious, 4),
            "spurious_attention_correct": round(corr_spurious, 4),
            "spurious_attention_incorrect": round(incorr_spurious, 4),
        })

    summary_df = pd.DataFrame(summary_records)

    # Overall dataset stats
    dataset_mean_spurious = float(eval_df["spurious_attention_ratio"].mean())
    corr_all = eval_df[eval_df["is_correct"]]
    incorr_all = eval_df[~eval_df["is_correct"]]
    dataset_corr_spurious = float(corr_all["spurious_attention_ratio"].mean()) if len(corr_all) > 0 else 0.0
    dataset_incorr_spurious = float(incorr_all["spurious_attention_ratio"].mean()) if len(incorr_all) > 0 else 0.0

    print("-" * 88)
    print(
        f"{'OVERALL AVERAGE':<28} | {len(eval_df):<5d} | {len(corr_all):<7d} | "
        f"{dataset_mean_spurious:11.2%}   | {dataset_corr_spurious:11.2%}   | {dataset_incorr_spurious:13.2%}"
    )
    print("=" * 80)

    # Save summary CSV
    out_csv = results_dir / f"xai_background_{model_name}.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[RESULTS SAVED]: Per-class spurious attention table written to: {out_csv}")
    print(f"[OVERLAYS SAVED]: Diagnostic visual overlays saved to: {model_reports_dir}")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Crop Disease XAI Background Attention Analysis.")
    parser.add_argument("--model", type=str, default="resnet18", help=f"Model to test. Choices: {list(MODEL_SPECS.keys())}")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint. If omitted, resolved via --model/--seed/--loss-version.")
    parser.add_argument("--loss-version", choices=["v1", "v2"], default="v2", help="Loss version used for training (default: v2). Determines checkpoint name suffix.")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Directory containing splits")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output CSV")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for output overlays")
    parser.add_argument("--image-root", type=str, default=".", help="Root directory for dataset image paths")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset sample size (e.g. 64 for dry run / smoke test)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', 'mps', 'cuda'). Defaults to 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run dry run with 64-image subset writing to *_dry directories")
    parser.add_argument("--max-overlays", type=int, default=5, help="Max correct/incorrect overlays to save per class (default: 5)")
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
        header="XAI BACKGROUND ATTENTION",
    )

    run_xai_background_analysis(
        model_name=args.model,
        checkpoint_path=ckpt_path,
        splits_dir=Path(args.splits_dir),
        results_dir=results_dir,
        reports_dir=reports_dir,
        image_root=Path(args.image_root),
        sample_size=sample_size,
        device=torch.device(args.device),
        dry_run=args.dry_run,
        max_overlays_per_class=args.max_overlays,
    )


if __name__ == "__main__":
    main()

