import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import List
import pandas as pd
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.device import get_device
from src.data.dataset import get_transforms
from src.models.backbones import create_model
from src.xai.gradcam import GradCAM
from src.xai.quantitative import compute_xai_metrics, aggregate_xai_metrics, segment_leaf_foreground


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps and quantitative XAI metrics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--samples-per-class", type=int, default=2, help="Number of test images to explain per class")
    parser.add_argument("--use-plusplus", action="store_true", help="Use Grad-CAM++ instead of vanilla Grad-CAM")
    parser.add_argument("--output-dir", type=str, default="results/xai", help="Directory to save XAI figures")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device(args.device)
    print(f"[Device] Using compute device: {device}")

    # Load model
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    backbone_name = checkpoint.get("backbone_name", cfg["model"]["backbone"])

    model = create_model(
        backbone_name=backbone_name,
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Target layer for CAM
    target_layer = model.get_target_layer_for_cam()
    cam_generator = GradCAM(model=model, target_layer=target_layer)

    # Load test split
    test_csv = Path(data_cfg["splits_dir"]) / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found: {test_csv}")
    test_df = pd.read_csv(test_csv)

    eval_transform = get_transforms(image_size=data_cfg.get("image_size", 224))["test"]
    image_root = Path(data_cfg.get("image_root", "data"))

    method_name = "Grad-CAM++" if args.use_plusplus else "Grad-CAM"
    print(f"\nGenerating {method_name} visualizations for {args.samples_per_class} samples per class...")

    per_sample_results = []

    for c_idx, class_name in enumerate(class_names):
        class_samples = test_df[test_df["class_name"] == class_name].head(args.samples_per_class)
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for s_idx, (_, row) in enumerate(class_samples.iterrows()):
            img_rel_path = row["image_path"]
            img_full_path = image_root / img_rel_path
            if not img_full_path.exists():
                img_full_path = Path(img_rel_path)

            raw_pil = Image.open(img_full_path).convert("RGB")
            input_tensor = eval_transform(raw_pil).unsqueeze(0).to(device)

            # Generate CAM
            cam_map = cam_generator.generate_cam(
                input_tensor,
                target_class=c_idx,
                use_gradcam_plusplus=args.use_plusplus,
            )

            # Quantitative XAI metrics
            q_metrics = compute_xai_metrics(cam_map, raw_pil)
            q_metrics["class_name"] = class_name
            q_metrics["image_path"] = str(img_full_path)
            q_metrics["sample_idx"] = s_idx
            per_sample_results.append(q_metrics)

            # Overlay visualization
            blended = cam_generator.overlay_heatmap(raw_pil, cam_map, alpha=0.5)
            leaf_mask = segment_leaf_foreground(np.array(raw_pil))

            # Multi-panel figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(raw_pil)
            axes[0].set_title(f"Input Leaf: {class_name}", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(leaf_mask, cmap="gray")
            axes[1].set_title("Leaf Foreground Mask", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(cam_map, cmap="jet")
            axes[2].set_title(f"{method_name} Heatmap", fontsize=10)
            axes[2].axis("off")

            axes[3].imshow(blended)
            axes[3].set_title(
                f"Overlay (Energy Conc: {q_metrics['energy_concentration'] * 100:.1f}%)",
                fontsize=10,
            )
            axes[3].axis("off")

            plt.tight_layout()
            save_path = class_dir / f"sample_{s_idx:02d}_{method_name.lower().replace('+', 'p')}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()

    cam_generator.remove_hooks()

    # Aggregate quantitative summary
    summary = aggregate_xai_metrics(per_sample_results)
    summary["method"] = method_name
    summary["total_samples_analyzed"] = len(per_sample_results)

    print("\n" + "=" * 60)
    print(f"QUANTITATIVE EXPLAINABLE AI (XAI) EVALUATION ({method_name})")
    print("=" * 60)
    print(f"Total Samples Analyzed:       {summary['total_samples_analyzed']}")
    print(f"Mean Energy Concentration:    {summary['mean_energy_concentration'] * 100:.2f}% (±{summary.get('std_energy_concentration', 0.0) * 100:.2f}%)")
    print(f"Pointing Game Peak Accuracy:  {summary['pointing_game_accuracy'] * 100:.2f}%")
    print("=" * 60)

    # Save quantitative files
    json_path = output_dir / "quantitative_xai_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = output_dir / "per_sample_xai_metrics.csv"
    pd.DataFrame(per_sample_results).to_csv(csv_path, index=False)

    print(f"\n[Artifact] Saved quantitative XAI metrics to: {json_path}")
    print(f"[Artifact] Saved per-sample XAI metrics to: {csv_path}")
    print(f"[Artifact] Saved visual overlays to: {output_dir}")


if __name__ == "__main__":
    main()
