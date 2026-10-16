"""
Reusable Training & Evaluation Harness for Multi-Crop Leaf Disease Baselines.

Requirements:
- Fixed seeds for python, numpy, torch; log environment & package versions.
- Reads splits/*.csv (train.csv, val.csv, test.csv).
- Pretrained baselines: ResNet18, MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0.
- Loss: Cross-entropy with class weights, label smoothing 0.1 (focal loss flag supported).
- Two-phase transfer learning:
  Phase 1: Head only (backbone frozen) for 3 epochs.
  Phase 2: Fine-tune all layers at lr 1e-4 with AdamW and cosine decay.
- Early stopping monitored on validation macro-F1.
- Augmentation: Resize preserving aspect ratio (to 256), center-crop to 224, flip, rotation 15, color jitter.
- Test split evaluation saved to results/<model>_seed<k>.json:
  macro-F1, accuracy, per-class P/R/F1, confusion matrix, parameter count, model size in MB, FLOPs, CPU latency.
- Summary written to results/baseline_summary.csv with mean and std across seeds.
"""

import os
import sys
import time
import json
import random
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import torchvision
import timm

from src.models.backbones import create_model, normalize_model_name
from src.data.dataset import CropDiseaseDataset, get_transforms
from src.losses.focal_loss import compute_class_weights, get_loss_function


# Supported canonical architectures
SUPPORTED_MODELS = [
    "resnet18",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "efficientnet_b0",
]


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_environment_versions() -> Dict[str, str]:
    """Log environment, hardware, and library versions."""
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    print("=" * 80)
    print("ENVIRONMENT & DEPENDENCY VERSIONS:")
    for k, v in versions.items():
        print(f"  - {k:<14}: {v}")
    print("=" * 80 + "\n")
    return versions


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Profile parameter count, model size in MB, FLOPs, and CPU latency."""
    model_cpu = model.to("cpu").eval()

    # 1. Parameter count
    total_params = sum(p.numel() for p in model_cpu.parameters())
    trainable_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)

    # 2. Model size in MB (float32 parameter bytes)
    model_size_mb = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / (1024 * 1024)

    # 3. FLOPs via PyTorch FlopCounterMode
    x_sample = torch.randn(*input_shape, device="cpu")
    with FlopCounterMode(display=False) as fcm:
        _ = model_cpu(x_sample)
    total_flops = fcm.get_total_flops()

    # 4. CPU Latency (batch size 1, 224x224)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(x_sample)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model_cpu(x_sample)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))

    return {
        "parameter_count": int(total_params),
        "trainable_parameters": int(trainable_params),
        "model_size_mb": round(float(model_size_mb), 2),
        "flops": int(total_flops),
        "gflops": round(float(total_flops / 1e9), 3),
        "cpu_latency_ms": round(mean_latency, 2),
        "cpu_latency_std_ms": round(std_latency, 2),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Run evaluation and return macro-F1, accuracy, per-class metrics, and confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        support = int((y_true == idx).sum())
        per_class_dict[cname] = {
            "precision": round(float(per_class_prec[idx]), 4),
            "recall": round(float(per_class_rec[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": support,
        }

    return {
        "loss": round(float(running_loss / max(1, total_samples)), 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "report_str": classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0),
    }


def train_single_model(
    model_name: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str],
    device: torch.device,
    batch_size: int = 32,
    phase1_epochs: int = 3,
    phase2_epochs: int = 15,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    weight_decay: float = 1e-2,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    early_stopping_patience: int = 5,
    models_dir: str | Path = "models",
    results_dir: str | Path = "results",
    data_root: str | Path = "data",
) -> Dict[str, Any]:
    """Train and evaluate a single model run with fixed seed and 2-phase transfer learning."""
    set_seed(seed)
    canonical_name = normalize_model_name(model_name)
    num_classes = len(class_names)

    print("\n" + "=" * 80)
    print(f"TRAINING: {canonical_name} | Seed: {seed} | Device: {device} | Classes: {num_classes}")
    print(f"Loss: {'Focal (gamma=2.0)' if use_focal else 'Weighted Cross-Entropy'} | Label Smoothing: {label_smoothing}")
    print(f"Phase 1: {phase1_epochs} epochs (Head only) | Phase 2: {phase2_epochs} epochs (All layers, LR={phase2_lr})")
    print("=" * 80)

    # 1. Transforms & DataLoaders
    all_transforms = get_transforms(image_size=224, resize_dim=256)
    train_ds = CropDiseaseDataset(train_df, image_root=data_root, transform=all_transforms["train"])
    val_ds = CropDiseaseDataset(val_df, image_root=data_root, transform=all_transforms["val"])
    test_ds = CropDiseaseDataset(test_df, image_root=data_root, transform=all_transforms["test"])

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    pin_mem = False if device.type == "mps" else (device.type == "cuda")
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # 2. Loss Criterion with Class Weights
    class_weights = compute_class_weights(train_df, num_classes=num_classes, method="inverse").to(device)
    loss_type = "focal" if use_focal else "weighted_ce"
    criterion = get_loss_function(
        loss_type=loss_type,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        gamma=focal_gamma,
    )

    # 3. Model Creation
    model = create_model(
        backbone_name=canonical_name,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.3,
    ).to(device)

    best_macro_f1 = -1.0
    patience_counter = 0
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_path / f"{canonical_name}_seed{seed}_best.pth"

    # ==========================================================
    # PHASE 1: Head Only (Backbone Frozen) for 3 epochs
    # ==========================================================
    if phase1_epochs > 0:
        print("\n--- PHASE 1: Head Only (Backbone Frozen) ---")
        model.freeze_backbone()
        trainable_p1 = [p for p in model.parameters() if p.requires_grad]
        optimizer_p1 = torch.optim.AdamW(trainable_p1, lr=phase1_lr, weight_decay=weight_decay)

        for epoch in range(1, phase1_epochs + 1):
            model.train()
            p1_loss = 0.0
            p1_samples = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer_p1.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_p1, max_norm=1.0)
                optimizer_p1.step()

                bs = images.size(0)
                p1_loss += loss.item() * bs
                p1_samples += bs

            val_metrics = evaluate_model(model, val_loader, criterion, device, class_names)
            tr_loss = p1_loss / max(1, p1_samples)
            print(
                f"[P1 Epoch {epoch:02d}/{phase1_epochs:02d}] "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                torch.save(model.state_dict(), best_ckpt_path)

    # ==========================================================
    # PHASE 2: Fine-Tune All Layers at lr 1e-4 with AdamW & Cosine
    # ==========================================================
    completed_p2_epochs = 0
    if phase2_epochs > 0:
        print(f"\n--- PHASE 2: Fine-Tuning All Layers (lr={phase2_lr}, Cosine Decay) ---")
        model.unfreeze_all()
        optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=phase2_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p2, T_max=phase2_epochs, eta_min=1e-6
        )

        for epoch in range(1, phase2_epochs + 1):
            completed_p2_epochs = epoch
            model.train()
            p2_loss = 0.0
            p2_samples = 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                optimizer_p2.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_p2.step()

                bs = images.size(0)
                p2_loss += loss.item() * bs
                p2_samples += bs

            scheduler.step()
            val_metrics = evaluate_model(model, val_loader, criterion, device, class_names)
            tr_loss = p2_loss / max(1, p2_samples)
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"[P2 Epoch {epoch:02d}/{phase2_epochs:02d}] "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  --> Saved new best checkpoint (Val Macro-F1: {best_macro_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement in validation macro-F1 for {early_stopping_patience} epochs.")
                    break

    # ==========================================================
    # FINAL EVALUATION ON TEST SPLIT
    # ==========================================================
    print("\n--- Evaluating Best Model Checkpoint on Test Split ---")
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_ckpt_path)

    test_metrics = evaluate_model(model, test_loader, criterion, device, class_names)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

    # ==========================================================
    # PROFILING: Parameters, Size in MB, FLOPs, CPU Latency
    # ==========================================================
    print("Profiling model efficiency (FLOPs, parameters, CPU latency)...")
    profile_data = profile_model_efficiency(model)
    print(
        f"Params: {profile_data['parameter_count']:,} | "
        f"Size: {profile_data['model_size_mb']:.2f} MB | "
        f"GFLOPs: {profile_data['gflops']} | "
        f"CPU Latency: {profile_data['cpu_latency_ms']:.2f} ms +/- {profile_data['cpu_latency_std_ms']:.2f} ms"
    )

    # Save results to JSON
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    json_path = results_path / f"{canonical_name}_seed{seed}.json"

    result_payload = {
        "model": canonical_name,
        "seed": seed,
        "macro_f1": test_metrics["macro_f1"],
        "accuracy": test_metrics["accuracy"],
        "weighted_f1": test_metrics["weighted_f1"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "parameter_count": profile_data["parameter_count"],
        "trainable_parameters": profile_data["trainable_parameters"],
        "model_size_mb": profile_data["model_size_mb"],
        "flops": profile_data["flops"],
        "gflops": profile_data["gflops"],
        "cpu_latency_ms": profile_data["cpu_latency_ms"],
        "cpu_latency_std_ms": profile_data["cpu_latency_std_ms"],
        "best_val_macro_f1": round(float(best_macro_f1), 4),
        "phase1_epochs": phase1_epochs,
        "phase2_epochs_completed": completed_p2_epochs,
        "loss_type": loss_type,
        "label_smoothing": label_smoothing,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Saved evaluation results to: {json_path}")
    print("\nClassification Report (Test Split):")
    print(test_metrics["report_str"])

    return result_payload


def write_baseline_summary(results_dir: str | Path = "results") -> Optional[pd.DataFrame]:
    """Aggregate all results/<model>_seed<k>.json and write results/baseline_summary.csv."""
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*_seed*.json"))

    if not json_files:
        print("No result JSON files found to summarize.")
        return None

    records = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"Warning: could not read {jf}: {e}")

    if not records:
        return None

    df = pd.DataFrame(records)
    summary_rows = []

    for model_name, grp in df.groupby("model"):
        n_seeds = len(grp)
        acc_mean, acc_std = grp["accuracy"].mean(), grp["accuracy"].std(ddof=1) if n_seeds > 1 else 0.0
        f1_mean, f1_std = grp["macro_f1"].mean(), grp["macro_f1"].std(ddof=1) if n_seeds > 1 else 0.0
        params = grp["parameter_count"].iloc[0]
        size_mb = grp["model_size_mb"].iloc[0]
        gflops = grp["gflops"].iloc[0]
        lat_mean = grp["cpu_latency_ms"].mean()

        summary_rows.append({
            "model": model_name,
            "num_seeds": n_seeds,
            "macro_f1_mean": round(float(f1_mean), 4),
            "macro_f1_std": round(float(f1_std), 4),
            "accuracy_mean": round(float(acc_mean), 4),
            "accuracy_std": round(float(acc_std), 4),
            "parameter_count": int(params),
            "model_size_mb": round(float(size_mb), 2),
            "gflops": round(float(gflops), 3),
            "cpu_latency_ms": round(float(lat_mean), 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = results_path / "baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print("\n" + "=" * 90)
    print(f"BASELINE SUMMARY TABLE (Saved to {csv_path}):")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate pretrained leaf disease classification baselines."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model architecture (resnet18, mobilenetv3_small_100, mobilenetv3_large_100, efficientnet_b0)",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="splits",
        help="Directory containing train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds for multi-run evaluations (default: 42 43 44)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training & evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=3,
        help="Phase 1 (head only, frozen backbone) epochs (default: 3)",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=15,
        help="Phase 2 (fine-tuning all layers) epochs (default: 15)",
    )
    parser.add_argument(
        "--phase2-lr",
        type=float,
        default=1e-4,
        help="Phase 2 learning rate with AdamW & CosineAnnealing (default: 1e-4)",
    )
    parser.add_argument(
        "--use-focal",
        action="store_true",
        default=False,
        help="Use Focal Loss instead of weighted Cross-Entropy (default: False)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run 1-epoch CPU smoke test on a subset to verify pipeline without long training",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for dataset images (default: data)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save evaluation JSONs and summary CSV (default: results)",
    )
    args = parser.parse_args()

    log_environment_versions()

    # Load splits
    splits_path = Path(args.splits_dir)
    train_df = pd.read_csv(splits_path / "train.csv")
    val_df = pd.read_csv(splits_path / "val.csv")
    test_df = pd.read_csv(splits_path / "test.csv")

    class_names = sorted(train_df["class_name"].unique())

    # Resolve device
    if args.smoke_test or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Smoke Test Mode (writes only to results_dry, checkpoints_dry)
    if args.smoke_test:
        print("[SMOKE TEST MODE]: Running 1-epoch CPU smoke test on a small subset (120 samples)...")
        # Sample 10 images per class for fast verification
        smoke_train = train_df.groupby("class_name").head(10).reset_index(drop=True)
        smoke_val = val_df.groupby("class_name").head(5).reset_index(drop=True)
        smoke_test = test_df.groupby("class_name").head(5).reset_index(drop=True)

        res = train_single_model(
            model_name=args.model,
            seed=args.seeds[0] if args.seeds else 42,
            train_df=smoke_train,
            val_df=smoke_val,
            test_df=smoke_test,
            class_names=class_names,
            device=torch.device("cpu"),
            batch_size=16,
            phase1_epochs=1,
            phase2_epochs=0,
            use_focal=args.use_focal,
            models_dir="checkpoints_dry",
            results_dir="results_dry",
            data_root=args.data_root,
        )
        print("\n[SMOKE TEST COMPLETED SUCCESSFULLY]: Model forward, backward, eval, and profiling passed!")
        return

    # Full Runs (multi-seed)
    for seed in args.seeds:
        train_single_model(
            model_name=args.model,
            seed=seed,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_names=class_names,
            device=device,
            batch_size=args.batch_size,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
            phase2_lr=args.phase2_lr,
            use_focal=args.use_focal,
            results_dir=args.results_dir,
            data_root=args.data_root,
        )

    # Summarize across seeds
    write_baseline_summary(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
