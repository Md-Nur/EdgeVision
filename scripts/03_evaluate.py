import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.device import get_device
from src.data.dataset import CropDiseaseDataset, get_transforms
from src.models.backbones import create_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained crop disease model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or val")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_cfg = cfg["outputs"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please train a model first.")

    device = get_device(args.device)
    print(f"[Device] Using compute device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    backbone_name = checkpoint.get("backbone_name", cfg["model"]["backbone"])
    num_classes = len(class_names)

    # Initialize model
    model = create_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load dataset split
    split_csv = Path(data_cfg["splits_dir"]) / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split file not found: {split_csv}")

    df = pd.read_csv(split_csv)
    eval_transform = get_transforms(image_size=data_cfg.get("image_size", 224))[args.split]
    dataset = CropDiseaseDataset(df, image_root=data_cfg.get("image_root", "data"), transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Evaluating {len(dataset)} samples from '{args.split}' split...")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)

            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred, class_names=class_names)

    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS ({backbone_name.upper()} on {args.split.upper()} SET)")
    print("=" * 65)
    print(f"Top-1 Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1 Score:    {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted-F1 Score: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:      {metrics['macro_recall'] * 100:.2f}%")
    print("\nDetailed Per-Class Classification Report:")
    print(metrics["report_str"])

    # Save reports and figures
    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    cm_path = results_dir / f"{backbone_name}_{args.split}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_path,
        title=f"{backbone_name} - Confusion Matrix ({args.split.capitalize()} Set)",
    )
    print(f"[Artifact] Saved Confusion Matrix to: {cm_path}")

    # Save CSV report
    report_df = pd.DataFrame(metrics["report_dict"]).transpose()
    report_csv = results_dir / f"{backbone_name}_{args.split}_classification_report.csv"
    report_df.to_csv(report_csv)
    print(f"[Artifact] Saved Classification Report to: {report_csv}")


if __name__ == "__main__":
    main()
