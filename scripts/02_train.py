import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.device import get_device, set_seed
from src.data.splitter import prepare_group_stratified_splits
from src.data.dataset import get_dataloaders
from src.models.backbones import create_model
from src.losses.focal_loss import compute_class_weights, get_loss_function
from src.training.trainer import Trainer
from src.evaluation.visualization import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Train multi-crop leaf disease classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Backbone model override (e.g., efficientnet_b0, resnet50)")
    parser.add_argument("--epochs-p1", type=int, default=None, help="Phase 1 (frozen backbone) epochs")
    parser.add_argument("--epochs-p2", type=int, default=None, help="Phase 2 (fine-tuning) epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--loss-type", type=str, default=None, help="Loss type: weighted_ce, focal, ce")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Sample fraction for quick smoke testing (0.0 to 1.0)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["outputs"]

    # Parameter overrides
    backbone_name = args.model or model_cfg.get("backbone", "efficientnet_b0")
    phase1_epochs = args.epochs_p1 if args.epochs_p1 is not None else train_cfg.get("phase1_epochs", 3)
    phase2_epochs = args.epochs_p2 if args.epochs_p2 is not None else train_cfg.get("phase2_epochs", 15)
    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    loss_type = args.loss_type or train_cfg.get("loss_type", "weighted_ce")
    device_name = args.device or train_cfg.get("device", "auto")

    # Set seeds & resolve device
    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(device_name)
    print(f"[Device] Using compute device: {device}")

    # Ensure splits exist
    splits_dir = Path(data_cfg["splits_dir"])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    if not (train_csv.exists() and val_csv.exists()):
        print("Splits not found. Generating leakage-free group-stratified splits...")
        prepare_group_stratified_splits(
            csv_path=data_cfg["csv_path"],
            output_dir=splits_dir,
            train_ratio=data_cfg.get("train_split", 0.70),
            val_ratio=data_cfg.get("val_split", 0.15),
            test_ratio=data_cfg.get("test_split", 0.15),
            seed=seed,
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Optional subset sampling for fast verification
    if args.subset_fraction and 0.0 < args.subset_fraction < 1.0:
        print(f"[Smoke Testing Mode] Subsetting data to fraction: {args.subset_fraction}")
        train_subsets = [
            grp.sample(max(2, int(len(grp) * args.subset_fraction)), random_state=seed)
            for _, grp in train_df.groupby("class_name")
        ]
        train_df = pd.concat(train_subsets, ignore_index=True)

        val_subsets = [
            grp.sample(max(2, int(len(grp) * args.subset_fraction)), random_state=seed)
            for _, grp in val_df.groupby("class_name")
        ]
        val_df = pd.concat(val_subsets, ignore_index=True)

    class_names = sorted(train_df["class_name"].unique())
    num_classes = len(class_names)
    print(f"[Dataset] Train samples: {len(train_df)}, Val samples: {len(val_df)}, Classes: {num_classes}")

    # Build DataLoaders
    dataloaders = get_dataloaders(
        train_df=train_df,
        val_df=val_df,
        image_root=data_cfg.get("image_root", "data"),
        image_size=data_cfg.get("image_size", 224),
        batch_size=batch_size,
        num_workers=data_cfg.get("num_workers", 2),
    )

    # Compute class weights & loss
    class_weights = compute_class_weights(train_df, num_classes=num_classes).to(device)
    criterion = get_loss_function(
        loss_type=loss_type,
        class_weights=class_weights if "weighted" in loss_type or loss_type == "focal" else None,
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
        gamma=train_cfg.get("focal_gamma", 2.0),
    )
    print(f"[Loss] Using {loss_type} loss with label smoothing {train_cfg.get('label_smoothing', 0.1)}")

    # Create model
    print(f"[Model] Initializing {backbone_name} with ImageNet pretrained weights...")
    model = create_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )

    # Train
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        device=device,
        class_names=class_names,
        models_dir=out_cfg.get("models_dir", "models"),
    )

    history = trainer.fit(
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=train_cfg.get("phase1_lr", 1e-3),
        phase2_backbone_lr=train_cfg.get("phase2_backbone_lr", 2e-5),
        phase2_head_lr=train_cfg.get("phase2_head_lr", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
    )

    # Save training curves plot
    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    curve_path = results_dir / f"{backbone_name}_training_curves.png"
    plot_training_curves(history, output_path=curve_path)
    print(f"\n[Artifact] Saved training curves to {curve_path}")


if __name__ == "__main__":
    main()
