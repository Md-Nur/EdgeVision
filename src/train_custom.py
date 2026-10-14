"""
Training Harness for CustomNet — Lightweight CNN trained from scratch.

Key differences from src/train.py (pretrained baselines):
  - No 2-phase transfer learning; model trains from random init for up to 80 epochs.
  - Stronger augmentation: RandomResizedCrop, shadow, blur (get_strong_transforms).
  - Optional MixUp / CutMix (off by default, enabled by flags).
  - OneCycleLR scheduler.
  - Dual loss: crop head CE + disease head CE, lambda-weighted.
  - Early stopping patience = 10 epochs on val macro-F1 (flat 12-class).
  - Inference via predict_with_masking() -> also computes within-crop macro-F1.
  - Saves results/custom_<attention>_seed<k>.json (superset of baseline format).
  - Appends/merges with results/baseline_summary.csv via shared write_summary().

Usage:
  # Smoke test (1 epoch, 10 imgs/class, CPU)
  PYTHONPATH=. .venv/bin/python src/train_custom.py --attention none --smoke-test

  # Full run (80 epochs, 3 seeds, all 3 attention variants)
  PYTHONPATH=. .venv/bin/python src/train_custom.py --attention none
  PYTHONPATH=. .venv/bin/python src/train_custom.py --attention se
  PYTHONPATH=. .venv/bin/python src/train_custom.py --attention cbam

  # With MixUp + CutMix
  PYTHONPATH=. .venv/bin/python src/train_custom.py --attention cbam --mixup --cutmix
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
import torch.nn.functional as F
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

from src.models.custom_net import build_custom_net, CROP_TO_CLASSES
from src.data.dataset import CropDiseaseDataset, get_strong_transforms
from src.losses.focal_loss import compute_class_weights, get_loss_function


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries."""
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


# ──────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix helpers
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y_disease: torch.Tensor,
    y_crop: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch.

    Returns mixed_x, y_a_disease, y_b_disease, y_a_crop, y_b_crop, lam.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y_disease, y_disease[idx], y_crop, y_crop[idx], lam


def cutmix_data(
    x: torch.Tensor,
    y_disease: torch.Tensor,
    y_crop: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch.

    Returns mixed_x, y_a_disease, y_b_disease, y_a_crop, y_b_crop, lam.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.shape
    idx = torch.randperm(batch_size, device=x.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    # Adjust lam to actual area ratio
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (h * w)
    return mixed_x, y_disease, y_disease[idx], y_crop, y_crop[idx], lam


def mixed_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for MixUp / CutMix."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Profiling
# ──────────────────────────────────────────────────────────────────────────────

def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Profile parameter count, model size MB, FLOPs, CPU latency."""
    model_cpu = model.to("cpu").eval()

    total_params = sum(p.numel() for p in model_cpu.parameters())
    trainable_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)
    model_size_mb = sum(
        p.numel() * p.element_size() for p in model_cpu.parameters()
    ) / (1024 * 1024)

    x_sample = torch.randn(*input_shape, device="cpu")
    with FlopCounterMode(display=False) as fcm:
        _ = model_cpu(x_sample)
    total_flops = fcm.get_total_flops()

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_cpu(x_sample)
        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model_cpu(x_sample)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "parameter_count": int(total_params),
        "trainable_parameters": int(trainable_params),
        "model_size_mb": round(float(model_size_mb), 2),
        "flops": int(total_flops),
        "gflops": round(float(total_flops / 1e9), 3),
        "cpu_latency_ms": round(float(np.mean(latencies)), 2),
        "cpu_latency_std_ms": round(float(np.std(latencies)), 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_custom(
    model: nn.Module,
    dataloader: DataLoader,
    disease_criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """Evaluate CustomNet on disease task with both flat and within-crop masked metrics.

    Returns a dict with:
      - flat: macro-F1, accuracy, per-class P/R/F1, confusion matrix (standard 12-class)
      - crop_accuracy: how often crop head is correct
      - within_crop_macro_f1: mean over crops of per-crop 4-class macro-F1 on masked preds
    """
    model.eval()
    all_flat_preds = []
    all_masked_preds = []
    all_targets = []
    all_crop_preds = []
    all_crop_targets = []
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        crop_labels = batch["crop_label"].to(device, non_blocking=True)

        crop_logits, disease_logits = model(images)

        loss = disease_criterion(disease_logits, labels)
        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        pred_crops, flat_preds, masked_preds = model.predict_with_masking(
            crop_logits, disease_logits
        )

        all_flat_preds.extend(flat_preds.cpu().numpy())
        all_masked_preds.extend(masked_preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_crop_preds.extend(pred_crops.cpu().numpy())
        all_crop_targets.extend(crop_labels.cpu().numpy())

    y_true = np.array(all_targets)
    y_flat = np.array(all_flat_preds)
    y_masked = np.array(all_masked_preds)
    y_crop_pred = np.array(all_crop_preds)
    y_crop_true = np.array(all_crop_targets)

    # ── Flat (12-class) metrics ──────────────────────────────────────────────
    flat_acc = float(accuracy_score(y_true, y_flat))
    flat_macro_f1 = float(f1_score(y_true, y_flat, average="macro", zero_division=0))
    flat_weighted_f1 = float(f1_score(y_true, y_flat, average="weighted", zero_division=0))

    per_class_f1 = f1_score(y_true, y_flat, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_flat, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_flat, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_flat)

    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        support = int((y_true == idx).sum())
        per_class_dict[cname] = {
            "precision": round(float(per_class_prec[idx]), 4),
            "recall": round(float(per_class_rec[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": support,
        }

    # ── Crop accuracy ────────────────────────────────────────────────────────
    crop_acc = float(accuracy_score(y_crop_true, y_crop_pred))

    # ── Within-crop macro-F1 (using masked predictions) ─────────────────────
    # For each crop, compute macro-F1 on its 4 disease classes
    # Only evaluate samples that belong to that crop
    crop_f1_scores = []
    for crop_idx, class_indices in CROP_TO_CLASSES.items():
        mask = y_crop_true == crop_idx
        if mask.sum() == 0:
            continue
        # Ground-truth and masked predictions for this crop's samples
        gt_crop = y_true[mask]
        pred_crop = y_masked[mask]
        # Compute macro-F1 restricted to this crop's label range
        crop_f1 = float(f1_score(
            gt_crop, pred_crop,
            labels=class_indices,
            average="macro",
            zero_division=0,
        ))
        crop_f1_scores.append(crop_f1)

    within_crop_macro_f1 = float(np.mean(crop_f1_scores)) if crop_f1_scores else 0.0

    return {
        "loss": round(float(running_loss / max(1, total_samples)), 4),
        "accuracy": round(flat_acc, 4),
        "macro_f1": round(flat_macro_f1, 4),
        "weighted_f1": round(flat_weighted_f1, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "report_str": classification_report(
            y_true, y_flat, target_names=class_names, digits=4, zero_division=0
        ),
        "crop_accuracy": round(crop_acc, 4),
        "within_crop_macro_f1": round(within_crop_macro_f1, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core training function
# ──────────────────────────────────────────────────────────────────────────────

def train_custom_net(
    attention: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str],
    device: torch.device,
    batch_size: int = 32,
    num_epochs: int = 80,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    crop_loss_lambda: float = 0.3,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    early_stopping_patience: int = 10,
    use_mixup: bool = False,
    use_cutmix: bool = False,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    dropout: float = 0.3,
    models_dir: str | Path = "models",
    results_dir: str | Path = "results",
) -> Dict[str, Any]:
    """Train CustomNet from scratch and evaluate on the test split.

    Dual loss: disease_loss + crop_loss_lambda * crop_loss.
    OneCycleLR scheduler.
    Early stopping on val flat macro-F1.
    Saves results/custom_<attention>_seed<k>.json.
    """
    set_seed(seed)
    model_key = f"custom_{attention}"
    num_classes = len(class_names)

    print("\n" + "=" * 80)
    print(f"TRAINING: CustomNet (attention={attention}) | Seed: {seed} | Device: {device}")
    print(f"Epochs: {num_epochs} | LR: {base_lr} | Crop loss λ: {crop_loss_lambda}")
    print(f"Loss: {'Focal' if use_focal else 'Weighted-CE'} | LS: {label_smoothing}")
    print(f"MixUp: {use_mixup} (α={mixup_alpha}) | CutMix: {use_cutmix} (α={cutmix_alpha})")
    print("=" * 80)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    all_tfms = get_strong_transforms(image_size=224)
    train_ds = CropDiseaseDataset(train_df, transform=all_tfms["train"])
    val_ds   = CropDiseaseDataset(val_df,   transform=all_tfms["val"])
    test_ds  = CropDiseaseDataset(test_df,  transform=all_tfms["test"])

    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Loss functions ───────────────────────────────────────────────────────
    disease_weights = compute_class_weights(train_df, num_classes=12, method="inverse").to(device)
    crop_weights    = compute_class_weights(
        train_df.assign(label=train_df["label"] // 4), num_classes=3, method="inverse"
    ).to(device)

    loss_type = "focal" if use_focal else "weighted_ce"
    disease_criterion = get_loss_function(
        loss_type=loss_type,
        class_weights=disease_weights,
        label_smoothing=label_smoothing,
        gamma=focal_gamma,
    )
    # Always use weighted CE for the (simple 3-class) crop head
    crop_criterion = nn.CrossEntropyLoss(weight=crop_weights, label_smoothing=label_smoothing)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_custom_net(attention=attention, dropout=dropout).to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # OneCycleLR: peaks at base_lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=base_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,          # 10% warm-up
        anneal_strategy="cos",
        div_factor=25.0,        # initial lr = max_lr/25
        final_div_factor=1e4,   # final lr = max_lr/1e4
    )

    # ── Checkpoint path ──────────────────────────────────────────────────────
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = models_path / f"{model_key}_seed{seed}_best.pth"

    best_val_macro_f1 = -1.0
    patience_counter  = 0
    completed_epochs  = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        completed_epochs = epoch
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in train_loader:
            images       = batch["image"].to(device, non_blocking=True)
            labels       = batch["label"].to(device, non_blocking=True)
            crop_labels  = batch["crop_label"].to(device, non_blocking=True)

            # ── Optional MixUp / CutMix (coin-flip if both enabled) ──────────
            aug_used = None
            if use_mixup or use_cutmix:
                apply_mix = use_mixup and (not use_cutmix or random.random() < 0.5)
                if apply_mix:
                    images, la_d, lb_d, la_c, lb_c, lam = mixup_data(
                        images, labels, crop_labels, alpha=mixup_alpha
                    )
                    aug_used = ("mixup", la_d, lb_d, la_c, lb_c, lam)
                elif use_cutmix:
                    images, la_d, lb_d, la_c, lb_c, lam = cutmix_data(
                        images, labels, crop_labels, alpha=cutmix_alpha
                    )
                    aug_used = ("cutmix", la_d, lb_d, la_c, lb_c, lam)

            optimizer.zero_grad()
            crop_logits, disease_logits = model(images)

            if aug_used is not None:
                _, la_d, lb_d, la_c, lb_c, lam = aug_used
                d_loss = mixed_criterion(disease_criterion, disease_logits, la_d, lb_d, lam)
                c_loss = mixed_criterion(crop_criterion,    crop_logits,    la_c, lb_c, lam)
            else:
                d_loss = disease_criterion(disease_logits, labels)
                c_loss = crop_criterion(crop_logits, crop_labels)

            loss = d_loss + crop_loss_lambda * c_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per batch

            bs = images.size(0)
            epoch_loss += loss.item() * bs
            epoch_samples += bs

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = evaluate_custom(
            model, val_loader, disease_criterion, device, class_names
        )
        tr_loss = epoch_loss / max(1, epoch_samples)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Within-Crop F1: {val_metrics['within_crop_macro_f1']:.4f} | "
            f"Crop Acc: {val_metrics['crop_accuracy']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> Saved best checkpoint (Val Macro-F1: {best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(
                    f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs "
                    f"(best val macro-F1: {best_val_macro_f1:.4f}). Stopping at epoch {epoch}."
                )
                break

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n--- Evaluating Best Checkpoint on Test Split ---")
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), best_ckpt_path)

    test_metrics = evaluate_custom(
        model, test_loader, disease_criterion, device, class_names
    )
    print(
        f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Test Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Within-Crop Macro-F1: {test_metrics['within_crop_macro_f1']:.4f} | "
        f"Crop Accuracy: {test_metrics['crop_accuracy']:.4f}"
    )

    # ── Profiling ────────────────────────────────────────────────────────────
    print("Profiling model efficiency (FLOPs, parameters, CPU latency)...")
    profile_data = profile_model_efficiency(model)
    print(
        f"Params: {profile_data['parameter_count']:,} | "
        f"Size: {profile_data['model_size_mb']:.2f} MB | "
        f"GFLOPs: {profile_data['gflops']} | "
        f"CPU Latency: {profile_data['cpu_latency_ms']:.2f} ± {profile_data['cpu_latency_std_ms']:.2f} ms"
    )

    # ── Save JSON result ─────────────────────────────────────────────────────
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    json_path = results_path / f"{model_key}_seed{seed}.json"

    result_payload: Dict[str, Any] = {
        # ── identifiers ──
        "model": model_key,
        "attention": attention,
        "seed": seed,
        # ── flat 12-class metrics ──
        "macro_f1": test_metrics["macro_f1"],
        "accuracy": test_metrics["accuracy"],
        "weighted_f1": test_metrics["weighted_f1"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        # ── custom metrics ──
        "within_crop_macro_f1": test_metrics["within_crop_macro_f1"],
        "crop_accuracy": test_metrics["crop_accuracy"],
        # ── efficiency ──
        "parameter_count": profile_data["parameter_count"],
        "trainable_parameters": profile_data["trainable_parameters"],
        "model_size_mb": profile_data["model_size_mb"],
        "flops": profile_data["flops"],
        "gflops": profile_data["gflops"],
        "cpu_latency_ms": profile_data["cpu_latency_ms"],
        "cpu_latency_std_ms": profile_data["cpu_latency_std_ms"],
        # ── training meta ──
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "epochs_completed": completed_epochs,
        "loss_type": loss_type,
        "label_smoothing": label_smoothing,
        "crop_loss_lambda": crop_loss_lambda,
        "use_mixup": use_mixup,
        "use_cutmix": use_cutmix,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Saved evaluation results to: {json_path}")
    print("\nClassification Report (Test Split):")
    print(test_metrics["report_str"])

    return result_payload


# ──────────────────────────────────────────────────────────────────────────────
# Summary CSV: merges custom + pretrained results
# ──────────────────────────────────────────────────────────────────────────────

def write_full_summary(results_dir: str | Path = "results") -> Optional[pd.DataFrame]:
    """Aggregate ALL results/*_seed*.json (baselines + custom) and write summary CSV.

    Handles the superset of fields; missing fields filled with NaN.
    """
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*_seed*.json"))

    if not json_files:
        print("No result JSON files found.")
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
        acc_mean = grp["accuracy"].mean()
        acc_std  = grp["accuracy"].std(ddof=1) if n_seeds > 1 else 0.0
        f1_mean  = grp["macro_f1"].mean()
        f1_std   = grp["macro_f1"].std(ddof=1) if n_seeds > 1 else 0.0

        # Custom-only fields (NaN for pretrained)
        wcf1_mean = grp["within_crop_macro_f1"].mean() if "within_crop_macro_f1" in grp.columns else float("nan")
        crop_acc_mean = grp["crop_accuracy"].mean() if "crop_accuracy" in grp.columns else float("nan")

        row = {
            "model": model_name,
            "num_seeds": n_seeds,
            "macro_f1_mean": round(float(f1_mean), 4),
            "macro_f1_std": round(float(f1_std), 4),
            "accuracy_mean": round(float(acc_mean), 4),
            "accuracy_std": round(float(acc_std), 4),
            "parameter_count": int(grp["parameter_count"].iloc[0]),
            "model_size_mb": round(float(grp["model_size_mb"].iloc[0]), 2),
            "gflops": round(float(grp["gflops"].iloc[0]), 3),
            "cpu_latency_ms": round(float(grp["cpu_latency_ms"].mean()), 2),
            "within_crop_macro_f1_mean": round(float(wcf1_mean), 4) if not np.isnan(wcf1_mean) else "",
            "crop_accuracy_mean": round(float(crop_acc_mean), 4) if not np.isnan(crop_acc_mean) else "",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = results_path / "baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print(f"FULL SUMMARY TABLE (Saved to {csv_path}):")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100 + "\n")

    return summary_df


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train CustomNet (from-scratch lightweight CNN) for leaf disease classification."
    )
    parser.add_argument(
        "--attention", type=str, default="none", choices=["none", "se", "cbam"],
        help="Attention module type: none | se | cbam (default: none)",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="splits",
        help="Directory with train.csv, val.csv, test.csv (default: splits)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="Random seeds (default: 42 43 44)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: auto | cpu | cuda | mps (default: auto)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Max training epochs (default: 80)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Max learning rate for OneCycleLR (default: 1e-3)",
    )
    parser.add_argument(
        "--crop-lambda", type=float, default=0.3,
        help="Weight for auxiliary crop head loss (default: 0.3)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate before classifier heads (default: 0.3)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--use-focal", action="store_true", default=False,
        help="Use Focal Loss instead of Weighted-CE (default: off)",
    )
    parser.add_argument(
        "--mixup", action="store_true", default=False,
        help="Enable MixUp augmentation (default: off)",
    )
    parser.add_argument(
        "--cutmix", action="store_true", default=False,
        help="Enable CutMix augmentation (default: off)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help="1-epoch CPU smoke test on 10 imgs/class subset",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory for evaluation JSONs and summary CSV (default: results)",
    )
    args = parser.parse_args()

    log_environment_versions()

    # ── Load splits ───────────────────────────────────────────────────────────
    splits_path = Path(args.splits_dir)
    train_df = pd.read_csv(splits_path / "train.csv")
    val_df   = pd.read_csv(splits_path / "val.csv")
    test_df  = pd.read_csv(splits_path / "test.csv")
    class_names = sorted(train_df["class_name"].unique())

    # ── Device resolution ─────────────────────────────────────────────────────
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

    # ── Smoke test mode ───────────────────────────────────────────────────────
    if args.smoke_test:
        print(
            "[SMOKE TEST MODE] 1-epoch CPU run | 10 imgs/class train | 5 imgs/class val+test | "
            f"attention={args.attention} | no MixUp/CutMix"
        )
        smoke_train = train_df.groupby("class_name").head(10).reset_index(drop=True)
        smoke_val   = val_df.groupby("class_name").head(5).reset_index(drop=True)
        smoke_test  = test_df.groupby("class_name").head(5).reset_index(drop=True)

        res = train_custom_net(
            attention=args.attention,
            seed=args.seeds[0] if args.seeds else 42,
            train_df=smoke_train,
            val_df=smoke_val,
            test_df=smoke_test,
            class_names=class_names,
            device=torch.device("cpu"),
            batch_size=16,
            num_epochs=1,
            base_lr=args.lr,
            crop_loss_lambda=args.crop_lambda,
            dropout=args.dropout,
            use_focal=args.use_focal,
            use_mixup=False,   # always off in smoke test
            use_cutmix=False,  # always off in smoke test
            early_stopping_patience=999,  # no early stop in 1-epoch test
            results_dir=args.results_dir,
        )
        print(
            f"\n[SMOKE TEST PASSED] "
            f"Macro-F1: {res['macro_f1']} | "
            f"Within-Crop F1: {res['within_crop_macro_f1']} | "
            f"Crop Acc: {res['crop_accuracy']} | "
            f"Params: {res['parameter_count']:,}"
        )
        return

    # ── Full multi-seed runs ──────────────────────────────────────────────────
    for seed in args.seeds:
        train_custom_net(
            attention=args.attention,
            seed=seed,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_names=class_names,
            device=device,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            base_lr=args.lr,
            crop_loss_lambda=args.crop_lambda,
            dropout=args.dropout,
            use_focal=args.use_focal,
            use_mixup=args.mixup,
            use_cutmix=args.cutmix,
            early_stopping_patience=args.patience,
            results_dir=args.results_dir,
        )

    # ── Write merged summary ──────────────────────────────────────────────────
    write_full_summary(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
