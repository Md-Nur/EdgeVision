"""
Stage A Python Orchestrator for Multi-Crop Disease Diagnosis.

Runs sequentially and resumably (skipping any run whose results/<run>.json already exists).
Seed 42 on fixed splits:
- Pretrained: resnet18, mobilenetv3_small, mobilenetv3_large, efficientnet_b0
  (two-phase: 3 epochs head-only, then up to 30 epochs fine-tune, early stopping on val macro-F1, patience 6)
- Custom (from scratch, 80 epochs, cosine LR, label smoothing 0.1, MixUp/CutMix OFF):
  attention: none, se, cbam

Features:
- Best available device (MPS if present, CUDA if present, CPU fallback).
- Saves best checkpoints to checkpoints/<run>_best.pth.
- Logs per-epoch train/val loss and macro-F1 to logs/<run>.csv.
- If a run's validation macro-F1 has not improved after 15 epochs at any point, stop it and mark it in the log.
- For every run evaluates on BOTH standard test split and exact-MD5-dedup test split.
- Reports macro-F1, accuracy, per-class F1, parameters, model size in MB, GFLOPs, and CPU latency
  (10 warm-up runs, 100 timed runs, torch.set_num_threads(4), batch size 1).
- Triggers src/aggregate_results.py to produce results/stage_a_table.csv and reports/<model>_confusion_matrix.png.
- Supports --dry-run for a fast 2-epoch end-to-end verification.
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

from src.models.backbones import create_model, normalize_model_name
from src.models.custom_net import build_custom_net, CROP_TO_CLASSES
from src.data.dataset import CropDiseaseDataset, get_transforms
from src.losses.focal_loss import compute_class_weights, get_loss_function
from src.aggregate_results import aggregate_stage_a


STAGE_A_MODELS = [
    # Pretrained models
    {"name": "resnet18", "type": "pretrained", "arch": "resnet18"},
    {"name": "mobilenetv3_small", "type": "pretrained", "arch": "mobilenetv3_small_100"},
    {"name": "mobilenetv3_large", "type": "pretrained", "arch": "mobilenetv3_large_100"},
    {"name": "efficientnet_b0", "type": "pretrained", "arch": "efficientnet_b0"},
    # Custom models (trained from scratch)
    {"name": "custom_none", "type": "custom", "attention": "none"},
    {"name": "custom_se", "type": "custom", "attention": "se"},
    {"name": "custom_cbam", "type": "custom", "attention": "cbam"},
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


def get_best_device() -> torch.device:
    """Auto-detect and return best available compute device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 100,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """
    Profile parameter count, model size in MB, GFLOPs, and CPU latency.
    Batch size 1, torch.set_num_threads(4), 10 warm-up runs, 100 timed runs.
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        model_cpu = model.to("cpu").eval()

        # 1. Parameter count
        total_params = sum(p.numel() for p in model_cpu.parameters())
        trainable_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)

        # 2. Model size in MB (float32 bytes)
        model_size_mb = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / (1024 * 1024)

        # 3. GFLOPs via PyTorch FlopCounterMode
        x_sample = torch.randn(*input_shape, device="cpu")
        with FlopCounterMode(display=False) as fcm:
            _ = model_cpu(x_sample)
        total_flops = fcm.get_total_flops()
        gflops = total_flops / 1e9

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
            "gflops": round(float(gflops), 3),
            "cpu_latency_ms": round(mean_latency, 2),
            "cpu_latency_std_ms": round(std_latency, 2),
            "num_threads": num_threads,
            "warmup_runs": num_warmup,
            "timed_runs": num_runs,
        }
    finally:
        torch.set_num_threads(old_threads)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    is_custom: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on a DataLoader split and return detailed metrics.
    Includes predictions, ground-truth targets, and group_ids for bootstrap CIs.
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_group_ids: List[Any] = []
    all_crop_preds: List[int] = []
    all_crop_targets: List[int] = []
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        gids = batch.get("group_id", [0] * len(labels))
        if isinstance(gids, torch.Tensor):
            all_group_ids.extend(gids.cpu().tolist())
        else:
            all_group_ids.extend(list(gids))

        if is_custom:
            crop_labels = batch["crop_label"].to(device, non_blocking=True)
            crop_logits, disease_logits = model(images)
            loss = criterion(disease_logits, labels)
            pred_crops, flat_preds, masked_preds = model.predict_with_masking(crop_logits, disease_logits)
            preds = flat_preds.cpu().numpy()
            all_crop_preds.extend(pred_crops.cpu().numpy())
            all_crop_targets.extend(crop_labels.cpu().numpy())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        targets = labels.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

        bs = images.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

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

    res: Dict[str, Any] = {
        "loss": round(float(running_loss / max(1, total_samples)), 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "y_true": [int(x) for x in all_targets],
        "y_pred": [int(x) for x in all_preds],
        "group_ids": [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in all_group_ids],
    }

    if is_custom and all_crop_targets:
        y_crop_true = np.array(all_crop_targets)
        y_crop_pred = np.array(all_crop_preds)
        crop_acc = float(accuracy_score(y_crop_true, y_crop_pred))
        res["crop_accuracy"] = round(crop_acc, 4)

    return res


def is_run_completed(json_path: Path, dry_run: bool = False) -> bool:
    """Check if result JSON represents a complete, valid run."""
    if not json_path.exists():
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "standard_test" not in data or "dedup_test" not in data:
            return False
        if "cpu_latency_ms" not in data or "gflops" not in data:
            return False
        # If running full mode, dry-run JSONs must be re-executed
        if not dry_run and data.get("dry_run", False):
            return False
        return True
    except Exception:
        return False


def run_pretrained_model(
    model_name: str,
    arch: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    std_test_df: pd.DataFrame,
    dedup_test_df: pd.DataFrame,
    class_names: List[str],
    device: torch.device,
    batch_size: int = 64,
    phase1_epochs: int = 3,
    phase2_epochs: int = 30,
    early_stopping_patience: int = 6,
    overall_patience: int = 15,
    checkpoints_dir: Path = Path("checkpoints"),
    logs_dir: Path = Path("logs"),
    results_dir: Path = Path("results"),
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a pretrained baseline model."""
    run_id = f"{model_name}_seed{seed}"
    ckpt_path = checkpoints_dir / f"{run_id}_best.pth"
    log_csv_path = logs_dir / f"{run_id}.csv"
    result_json_path = results_dir / f"{run_id}.json"

    print("\n" + "=" * 80)
    print(f"RUNNING PRETRAINED: {model_name} ({arch}) | Run ID: {run_id}")
    print(f"Device: {device} | Dry Run: {dry_run} | Batch Size: {batch_size}")
    print(f"Phase 1: {phase1_epochs} epochs | Phase 2: {phase2_epochs} epochs | ES Patience: {early_stopping_patience}")
    print("=" * 80)

    # 1. Transforms & Datasets
    transforms = get_transforms(image_size=224, resize_dim=256)
    train_ds = CropDiseaseDataset(train_df, transform=transforms["train"])
    val_ds = CropDiseaseDataset(val_df, transform=transforms["val"])
    std_test_ds = CropDiseaseDataset(std_test_df, transform=transforms["test"])
    dedup_test_ds = CropDiseaseDataset(dedup_test_df, transform=transforms["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    std_test_loader = DataLoader(std_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    dedup_test_loader = DataLoader(dedup_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Loss with inverse class weights & label smoothing
    class_weights = compute_class_weights(train_df, num_classes=len(class_names), method="inverse").to(device)
    criterion = get_loss_function(loss_type="weighted_ce", class_weights=class_weights, label_smoothing=0.1)

    # 3. Model creation
    model = create_model(backbone_name=arch, num_classes=len(class_names), pretrained=True, dropout=0.3).to(device)

    best_val_macro_f1 = -1.0
    patience_counter = 0
    consecutive_no_improve = 0
    stopped_early = False
    stop_reason = ""
    epoch_logs = []

    current_epoch = 0

    # ==========================================================
    # PHASE 1: Head-Only (Backbone Frozen)
    # ==========================================================
    if phase1_epochs > 0:
        print(f"\n--- Phase 1: Head Only Training ({phase1_epochs} epochs, lr=1e-3) ---")
        model.freeze_backbone()
        trainable_p1 = [p for p in model.parameters() if p.requires_grad]
        opt_p1 = torch.optim.AdamW(trainable_p1, lr=1e-3, weight_decay=1e-2)

        for ep in range(1, phase1_epochs + 1):
            current_epoch += 1
            model.train()
            p1_loss = 0.0
            p1_count = 0

            for batch in train_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                lbls = batch["label"].to(device, non_blocking=True)
                opt_p1.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_p1, max_norm=1.0)
                opt_p1.step()

                bs = imgs.size(0)
                p1_loss += loss.item() * bs
                p1_count += bs

            val_res = evaluate_split(model, val_loader, criterion, device, class_names, is_custom=False)
            tr_loss = p1_loss / max(1, p1_count)
            is_best = val_res["macro_f1"] > best_val_macro_f1

            if is_best:
                best_val_macro_f1 = val_res["macro_f1"]
                torch.save(model.state_dict(), ckpt_path)
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1

            print(
                f"[P1 Ep {ep}/{phase1_epochs} (Total Ep {current_epoch})] "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
                f"Val Acc: {val_res['accuracy']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f}"
                + (" [BEST]" if is_best else "")
            )

            epoch_logs.append({
                "epoch": current_epoch,
                "phase": "phase1",
                "train_loss": round(tr_loss, 4),
                "val_loss": val_res["loss"],
                "val_macro_f1": val_res["macro_f1"],
                "val_accuracy": val_res["accuracy"],
                "lr": 1e-3,
                "is_best": is_best,
                "early_stop_marked": False,
            })
            # Flush CSV incrementally so progress survives interruption
            pd.DataFrame(epoch_logs).to_csv(log_csv_path, index=False)

    # ==========================================================
    # PHASE 2: Fine-Tuning All Layers with Cosine Decay
    # ==========================================================
    if phase2_epochs > 0 and not stopped_early:
        print(f"\n--- Phase 2: Fine-Tuning All Layers ({phase2_epochs} epochs, lr=1e-4, Cosine Decay) ---")
        model.unfreeze_all()
        opt_p2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        sched_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=phase2_epochs, eta_min=1e-6)

        for ep in range(1, phase2_epochs + 1):
            current_epoch += 1
            model.train()
            p2_loss = 0.0
            p2_count = 0

            for batch in train_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                lbls = batch["label"].to(device, non_blocking=True)
                opt_p2.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt_p2.step()

                bs = imgs.size(0)
                p2_loss += loss.item() * bs
                p2_count += bs

            sched_p2.step()
            val_res = evaluate_split(model, val_loader, criterion, device, class_names, is_custom=False)
            tr_loss = p2_loss / max(1, p2_count)
            current_lr = sched_p2.get_last_lr()[0]
            is_best = val_res["macro_f1"] > best_val_macro_f1

            if is_best:
                best_val_macro_f1 = val_res["macro_f1"]
                patience_counter = 0
                consecutive_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                consecutive_no_improve += 1

            stop_marked_str = ""
            # Check 15-epoch no improvement rule
            if consecutive_no_improve >= overall_patience:
                stopped_early = True
                stop_reason = f"No improvement in validation macro-F1 for {overall_patience} epochs"
                stop_marked_str = f"STOPPED_EARLY: {stop_reason}"
            elif patience_counter >= early_stopping_patience and not dry_run:
                stopped_early = True
                stop_reason = f"Early stopping patience ({early_stopping_patience} epochs) reached in Phase 2"
                stop_marked_str = f"STOPPED_EARLY: {stop_reason}"

            print(
                f"[P2 Ep {ep:02d}/{phase2_epochs:02d} (Total Ep {current_epoch:02d})] "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
                f"Val Acc: {val_res['accuracy']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f} | "
                f"LR: {current_lr:.2e}"
                + (" [BEST]" if is_best else "")
                + (f" -> {stop_marked_str}" if stop_marked_str else "")
            )

            epoch_logs.append({
                "epoch": current_epoch,
                "phase": "phase2",
                "train_loss": round(tr_loss, 4),
                "val_loss": val_res["loss"],
                "val_macro_f1": val_res["macro_f1"],
                "val_accuracy": val_res["accuracy"],
                "lr": round(current_lr, 8),
                "is_best": is_best,
                "early_stop_marked": stop_marked_str if stop_marked_str else False,
            })
            # Flush CSV incrementally so progress survives interruption
            pd.DataFrame(epoch_logs).to_csv(log_csv_path, index=False)

            if stopped_early:
                print(f"\n[EARLY STOPPING TRIGGERED]: {stop_reason}")
                break

    # Save per-epoch logs to CSV
    log_df = pd.DataFrame(epoch_logs)
    log_df.to_csv(log_csv_path, index=False)
    print(f"\n[LOG SAVED]: Per-epoch logs written to: {log_csv_path}")

    # ==========================================================
    # EVALUATION ON BOTH TEST SPLITS
    # ==========================================================
    print("\n--- Evaluating Best Checkpoint on Test Splits ---")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), ckpt_path)

    std_test_metrics = evaluate_split(model, std_test_loader, criterion, device, class_names, is_custom=False)
    print(f"Standard Test Split -> Macro-F1: {std_test_metrics['macro_f1']:.4f} | Acc: {std_test_metrics['accuracy']:.4f}")

    dedup_test_metrics = evaluate_split(model, dedup_test_loader, criterion, device, class_names, is_custom=False)
    print(f"Dedup Test Split    -> Macro-F1: {dedup_test_metrics['macro_f1']:.4f} | Acc: {dedup_test_metrics['accuracy']:.4f}")

    # ==========================================================
    # EFFICIENCY PROFILING (batch size 1, 10 warm-up, 100 runs, torch.set_num_threads(4))
    # ==========================================================
    print("Profiling model efficiency (Parameters, MB, GFLOPs, CPU Latency with 4 threads)...")
    profile_data = profile_model_efficiency(model, num_warmup=10, num_runs=100, num_threads=4)
    print(
        f"Params: {profile_data['parameter_count']:,} | "
        f"Size: {profile_data['model_size_mb']} MB | "
        f"GFLOPs: {profile_data['gflops']} | "
        f"CPU Latency: {profile_data['cpu_latency_ms']} ms +/- {profile_data['cpu_latency_std_ms']} ms"
    )

    payload = {
        "model": model_name,
        "arch": arch,
        "type": "pretrained",
        "seed": seed,
        "dry_run": dry_run,
        "macro_f1": std_test_metrics["macro_f1"],
        "accuracy": std_test_metrics["accuracy"],
        "standard_test": std_test_metrics,
        "dedup_test": dedup_test_metrics,
        "parameter_count": profile_data["parameter_count"],
        "trainable_parameters": profile_data["trainable_parameters"],
        "model_size_mb": profile_data["model_size_mb"],
        "flops": profile_data["flops"],
        "gflops": profile_data["gflops"],
        "cpu_latency_ms": profile_data["cpu_latency_ms"],
        "cpu_latency_std_ms": profile_data["cpu_latency_std_ms"],
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "class_names": class_names,
    }

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[RESULT SAVED]: Evaluation payload saved to: {result_json_path}")
    return payload


def run_custom_model(
    model_name: str,
    attention: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    std_test_df: pd.DataFrame,
    dedup_test_df: pd.DataFrame,
    class_names: List[str],
    device: torch.device,
    batch_size: int = 64,
    num_epochs: int = 80,
    overall_patience: int = 15,
    base_lr: float = 1e-3,
    crop_loss_lambda: float = 0.3,
    checkpoints_dir: Path = Path("checkpoints"),
    logs_dir: Path = Path("logs"),
    results_dir: Path = Path("results"),
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a CustomNet model from scratch."""
    run_id = f"{model_name}_seed{seed}"
    ckpt_path = checkpoints_dir / f"{run_id}_best.pth"
    log_csv_path = logs_dir / f"{run_id}.csv"
    result_json_path = results_dir / f"{run_id}.json"

    print("\n" + "=" * 80)
    print(f"RUNNING CUSTOM FROM SCRATCH: {model_name} (attention={attention}) | Run ID: {run_id}")
    print(f"Device: {device} | Dry Run: {dry_run} | Batch Size: {batch_size} | Epochs: {num_epochs}")
    print(f"Scheduler: CosineAnnealingLR (lr={base_lr}) | Label Smoothing: 0.1 | MixUp/CutMix: OFF")
    print("=" * 80)

    # 1. Transforms & Datasets
    transforms = get_transforms(image_size=224, resize_dim=256)
    train_ds = CropDiseaseDataset(train_df, transform=transforms["train"])
    val_ds = CropDiseaseDataset(val_df, transform=transforms["val"])
    std_test_ds = CropDiseaseDataset(std_test_df, transform=transforms["test"])
    dedup_test_ds = CropDiseaseDataset(dedup_test_df, transform=transforms["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    std_test_loader = DataLoader(std_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    dedup_test_loader = DataLoader(dedup_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Criteria: weighted disease CE + unweighted crop CE
    disease_weights = compute_class_weights(train_df, num_classes=len(class_names), method="inverse").to(device)
    disease_criterion = get_loss_function(loss_type="weighted_ce", class_weights=disease_weights, label_smoothing=0.1)
    crop_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 3. Model creation (from scratch, random init)
    model = build_custom_net(attention=attention, dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_macro_f1 = -1.0
    consecutive_no_improve = 0
    stopped_early = False
    stop_reason = ""
    epoch_logs = []

    for ep in range(1, num_epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_count = 0

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            crop_labels = batch["crop_label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            crop_logits, disease_logits = model(imgs)

            loss_disease = disease_criterion(disease_logits, labels)
            loss_crop = crop_criterion(crop_logits, crop_labels)
            total_loss = loss_disease + crop_loss_lambda * loss_crop

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = imgs.size(0)
            ep_loss += total_loss.item() * bs
            ep_count += bs

        scheduler.step()
        val_res = evaluate_split(model, val_loader, disease_criterion, device, class_names, is_custom=True)
        tr_loss = ep_loss / max(1, ep_count)
        current_lr = scheduler.get_last_lr()[0]
        is_best = val_res["macro_f1"] > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = val_res["macro_f1"]
            consecutive_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            consecutive_no_improve += 1

        stop_marked_str = ""
        # 15-epoch no improvement check
        if consecutive_no_improve >= overall_patience and not dry_run:
            stopped_early = True
            stop_reason = f"No improvement in validation macro-F1 for {overall_patience} epochs"
            stop_marked_str = f"STOPPED_EARLY: {stop_reason}"

        print(
            f"[Custom Ep {ep:02d}/{num_epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
            f"Val Acc: {val_res['accuracy']:.4f} | Val Macro-F1: {val_res['macro_f1']:.4f} | "
            f"Crop Acc: {val_res.get('crop_accuracy', 0.0):.4f} | LR: {current_lr:.2e}"
            + (" [BEST]" if is_best else "")
            + (f" -> {stop_marked_str}" if stop_marked_str else "")
        )

        epoch_logs.append({
            "epoch": ep,
            "phase": "train_from_scratch",
            "train_loss": round(tr_loss, 4),
            "val_loss": val_res["loss"],
            "val_macro_f1": val_res["macro_f1"],
            "val_accuracy": val_res["accuracy"],
            "crop_accuracy": val_res.get("crop_accuracy", 0.0),
            "lr": round(current_lr, 8),
            "is_best": is_best,
            "early_stop_marked": stop_marked_str if stop_marked_str else False,
        })
        # Flush CSV incrementally so progress survives interruption
        pd.DataFrame(epoch_logs).to_csv(log_csv_path, index=False)

        if stopped_early:
            print(f"\n[EARLY STOPPING TRIGGERED]: {stop_reason}")
            break

    # Save per-epoch logs to CSV
    log_df = pd.DataFrame(epoch_logs)
    log_df.to_csv(log_csv_path, index=False)
    print(f"\n[LOG SAVED]: Per-epoch logs written to: {log_csv_path}")

    # ==========================================================
    # EVALUATION ON BOTH TEST SPLITS
    # ==========================================================
    print("\n--- Evaluating Best Checkpoint on Test Splits ---")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), ckpt_path)

    std_test_metrics = evaluate_split(model, std_test_loader, disease_criterion, device, class_names, is_custom=True)
    print(f"Standard Test Split -> Macro-F1: {std_test_metrics['macro_f1']:.4f} | Acc: {std_test_metrics['accuracy']:.4f}")

    dedup_test_metrics = evaluate_split(model, dedup_test_loader, disease_criterion, device, class_names, is_custom=True)
    print(f"Dedup Test Split    -> Macro-F1: {dedup_test_metrics['macro_f1']:.4f} | Acc: {dedup_test_metrics['accuracy']:.4f}")

    # ==========================================================
    # EFFICIENCY PROFILING (batch size 1, 10 warm-up, 100 runs, torch.set_num_threads(4))
    # ==========================================================
    print("Profiling model efficiency (Parameters, MB, GFLOPs, CPU Latency with 4 threads)...")
    profile_data = profile_model_efficiency(model, num_warmup=10, num_runs=100, num_threads=4)
    print(
        f"Params: {profile_data['parameter_count']:,} | "
        f"Size: {profile_data['model_size_mb']} MB | "
        f"GFLOPs: {profile_data['gflops']} | "
        f"CPU Latency: {profile_data['cpu_latency_ms']} ms +/- {profile_data['cpu_latency_std_ms']} ms"
    )

    payload = {
        "model": model_name,
        "attention": attention,
        "type": "custom",
        "seed": seed,
        "dry_run": dry_run,
        "macro_f1": std_test_metrics["macro_f1"],
        "accuracy": std_test_metrics["accuracy"],
        "standard_test": std_test_metrics,
        "dedup_test": dedup_test_metrics,
        "parameter_count": profile_data["parameter_count"],
        "trainable_parameters": profile_data["trainable_parameters"],
        "model_size_mb": profile_data["model_size_mb"],
        "flops": profile_data["flops"],
        "gflops": profile_data["gflops"],
        "cpu_latency_ms": profile_data["cpu_latency_ms"],
        "cpu_latency_std_ms": profile_data["cpu_latency_std_ms"],
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "class_names": class_names,
    }

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[RESULT SAVED]: Evaluation payload saved to: {result_json_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A Multi-Crop Disease Orchestrator.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run 2-epoch dry run of the pipeline")
    parser.add_argument("--dry-run-samples-per-class", type=int, default=50, help="Train samples per class for dry run (default: 50)")
    parser.add_argument("--force", action="store_true", default=False, help="Force rerun even if results exist")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and eval (default: 64)")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Directory containing split CSVs")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for JSON results")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for training logs")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for report artifacts")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to run (default: all 7 Stage A models)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_best_device()

    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    logs_dir = Path(args.logs_dir)
    reports_dir = Path(args.reports_dir)
    splits_dir = Path(args.splits_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE A: MULTI-CROP LEAF DISEASE DIAGNOSIS ORCHESTRATOR")
    print(f"Seed: {args.seed} | Device: {device} | Dry Run: {args.dry_run} | Batch Size: {args.batch_size}")
    print(f"Results: {results_dir} | Checkpoints: {checkpoints_dir} | Logs: {logs_dir}")
    print("=" * 80)

    # Load fixed splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    std_test_df = pd.read_csv(splits_dir / "test.csv")
    dedup_test_df = pd.read_csv(splits_dir / "dedup" / "test.csv")

    class_names = sorted(train_df["class_name"].unique())
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Data Splits -> Train: {len(train_df):,} | Val: {len(val_df):,} | Std Test: {len(std_test_df):,} | Dedup Test: {len(dedup_test_df):,}")

    # Prepare datasets for run
    if args.dry_run and args.dry_run_samples_per_class > 0:
        print(f"\n[DRY RUN MODE]: Stratifying training to {args.dry_run_samples_per_class} images/class for fast 2-epoch verification...")
        run_train_df = train_df.groupby("class_name").head(args.dry_run_samples_per_class).reset_index(drop=True)
        run_val_df = val_df.groupby("class_name").head(20).reset_index(drop=True)
    else:
        run_train_df = train_df
        run_val_df = val_df

    # Filter models if requested
    models_to_run = STAGE_A_MODELS
    if args.models:
        selected_names = [m.lower().replace("-", "_") for m in args.models]
        models_to_run = [m for m in STAGE_A_MODELS if m["name"] in selected_names]

    for model_cfg in models_to_run:
        model_name = model_cfg["name"]
        run_id = f"{model_name}_seed{args.seed}"
        result_json = results_dir / f"{run_id}.json"

        # Check resumability
        if not args.force and is_run_completed(result_json, dry_run=args.dry_run):
            print(f"\n[SKIP] Run {run_id} already exists at {result_json}. Skipping to next...")
            continue

        if model_cfg["type"] == "pretrained":
            p1_ep = 1 if args.dry_run else 3
            p2_ep = 1 if args.dry_run else 30
            run_pretrained_model(
                model_name=model_name,
                arch=model_cfg["arch"],
                seed=args.seed,
                train_df=run_train_df,
                val_df=run_val_df,
                std_test_df=std_test_df,
                dedup_test_df=dedup_test_df,
                class_names=class_names,
                device=device,
                batch_size=args.batch_size,
                phase1_epochs=p1_ep,
                phase2_epochs=p2_ep,
                early_stopping_patience=6,
                overall_patience=15,
                checkpoints_dir=checkpoints_dir,
                logs_dir=logs_dir,
                results_dir=results_dir,
                dry_run=args.dry_run,
            )
        elif model_cfg["type"] == "custom":
            custom_ep = 2 if args.dry_run else 80
            run_custom_model(
                model_name=model_name,
                attention=model_cfg["attention"],
                seed=args.seed,
                train_df=run_train_df,
                val_df=run_val_df,
                std_test_df=std_test_df,
                dedup_test_df=dedup_test_df,
                class_names=class_names,
                device=device,
                batch_size=args.batch_size,
                num_epochs=custom_ep,
                overall_patience=15,
                checkpoints_dir=checkpoints_dir,
                logs_dir=logs_dir,
                results_dir=results_dir,
                dry_run=args.dry_run,
            )

    # Aggregate all completed results
    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE. BUILDING STAGE A TABLE & CONFUSION MATRICES...")
    print("=" * 80)
    aggregate_stage_a(
        results_dir=results_dir,
        output_table_path=results_dir / "stage_a_table.csv",
        reports_dir=reports_dir,
        n_bootstraps=1000,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
