"""
Speed profiler for multi-crop disease classification models.

Times 20 training batches (batch size 64, device mps) for specified models (default: resnet18 and custom_none).
Measures per batch:
- Data loading wait time
- Host-to-device copy time
- Forward pass time
- Backward pass time
- Optimizer step time
Reports current DataLoader settings and identifies whether data loading or compute is the bottleneck.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.backbones import create_model
from src.models.custom_net import build_custom_net
from src.data.dataset import CropDiseaseDataset, get_transforms
from src.losses.focal_loss import compute_class_weights, get_loss_function


def sync_device(device: torch.device) -> None:
    """Synchronize device execution queues for accurate timing."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_batches(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_custom: bool = False,
    crop_criterion: Any = None,
    crop_loss_lambda: float = 0.3,
    num_warmup: int = 2,
    num_batches: int = 20,
) -> Dict[str, Any]:
    """Run warmup and time num_batches batches measuring each pipeline stage."""
    model.train()
    loader_iter = iter(dataloader)

    # Warmup
    for _ in range(num_warmup):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch = next(loader_iter)

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        if is_custom:
            crop_labels = batch["crop_label"].to(device, non_blocking=True)
            crop_logits, disease_logits = model(images)
            loss = criterion(disease_logits, labels) + crop_loss_lambda * crop_criterion(crop_logits, crop_labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        sync_device(device)

    # Timed runs
    data_times = []
    copy_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_batch_times = []

    for _ in range(num_batches):
        t_batch_start = time.perf_counter()

        # 1. Data loading wait time
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch = next(loader_iter)
        t_data = time.perf_counter() - t0

        # 2. Host-to-device copy
        t1 = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if is_custom:
            crop_labels = batch["crop_label"].to(device, non_blocking=True)
        sync_device(device)
        t_copy = time.perf_counter() - t1

        # 3. Forward pass
        optimizer.zero_grad()
        t2 = time.perf_counter()
        if is_custom:
            crop_logits, disease_logits = model(images)
            loss = criterion(disease_logits, labels) + crop_loss_lambda * crop_criterion(crop_logits, crop_labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        sync_device(device)
        t_fwd = time.perf_counter() - t2

        # 4. Backward pass
        t3 = time.perf_counter()
        loss.backward()
        sync_device(device)
        t_bwd = time.perf_counter() - t3

        # 5. Optimizer step
        t4 = time.perf_counter()
        optimizer.step()
        sync_device(device)
        t_opt = time.perf_counter() - t4

        t_batch_total = time.perf_counter() - t_batch_start

        data_times.append(t_data)
        copy_times.append(t_copy)
        forward_times.append(t_fwd)
        backward_times.append(t_bwd)
        optimizer_times.append(t_opt)
        total_batch_times.append(t_batch_total)

    compute_times = [f + b + o for f, b, o in zip(forward_times, backward_times, optimizer_times)]

    return {
        "num_batches": num_batches,
        "data_loading_mean_s": float(np.mean(data_times)),
        "data_loading_std_s": float(np.std(data_times)),
        "copy_mean_s": float(np.mean(copy_times)),
        "copy_std_s": float(np.std(copy_times)),
        "forward_mean_s": float(np.mean(forward_times)),
        "forward_std_s": float(np.std(forward_times)),
        "backward_mean_s": float(np.mean(backward_times)),
        "backward_std_s": float(np.std(backward_times)),
        "optimizer_mean_s": float(np.mean(optimizer_times)),
        "optimizer_std_s": float(np.std(optimizer_times)),
        "compute_mean_s": float(np.mean(compute_times)),
        "total_batch_mean_s": float(np.mean(total_batch_times)),
        "total_batch_std_s": float(np.std(total_batch_times)),
        "per_batch": [
            {
                "batch_idx": i + 1,
                "data_loading_ms": round(data_times[i] * 1000, 2),
                "copy_ms": round(copy_times[i] * 1000, 2),
                "forward_ms": round(forward_times[i] * 1000, 2),
                "backward_ms": round(backward_times[i] * 1000, 2),
                "optimizer_ms": round(optimizer_times[i] * 1000, 2),
                "total_ms": round(total_batch_times[i] * 1000, 2),
            }
            for i in range(num_batches)
        ],
    }


def run_profiling(
    models: List[str],
    batch_size: int = 64,
    device_name: str = "mps",
    splits_dir: str = "splits",
    data_root: str = "data",
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: Any = None,
    pin_memory: bool = False,
    num_batches: int = 20,
) -> Dict[str, Any]:
    # Determine device
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_csv = Path(splits_dir) / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train split not found at: {train_csv}")

    train_df = pd.read_csv(train_csv)
    class_names = sorted(train_df["class_name"].unique())
    num_classes = len(class_names)
    total_samples = len(train_df)
    batches_per_epoch = int(np.ceil(total_samples / batch_size))

    # Construct transforms and dataset
    transforms = get_transforms(image_size=224, resize_dim=256)
    train_ds = CropDiseaseDataset(train_df, image_root=data_root, transform=transforms["train"])

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, **loader_kwargs)

    # Loss functions
    class_weights = compute_class_weights(train_df, num_classes=num_classes, method="inverse").to(device)
    disease_criterion = get_loss_function(loss_type="weighted_ce", class_weights=class_weights, label_smoothing=0.1)
    crop_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("=" * 80)
    print("SPEED PROFILING RUN")
    print(f"Device: {device} | Batch Size: {batch_size} | Timed Batches: {num_batches}")
    print(f"Splits Dir: {splits_dir} | Data Root: {data_root}")
    print(f"Dataset Size: {total_samples} images ({batches_per_epoch} batches / epoch)")
    print("DataLoader Settings:")
    print(f"  - num_workers:        {num_workers}")
    print(f"  - persistent_workers: {persistent_workers if num_workers > 0 else False}")
    print(f"  - prefetch_factor:    {prefetch_factor if num_workers > 0 else None}")
    print(f"  - pin_memory:         {pin_memory}")
    print("=" * 80)

    results = {
        "dataloader_settings": {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
            "prefetch_factor": prefetch_factor if num_workers > 0 else None,
            "pin_memory": pin_memory,
        },
        "models": {},
        "batches_per_epoch": batches_per_epoch,
        "total_samples": total_samples,
    }

    for model_name in models:
        print(f"\nProfiling model: {model_name}...")
        is_custom = model_name.startswith("custom")

        if is_custom:
            attention = model_name.split("_")[-1] if "_" in model_name else "none"
            model = build_custom_net(attention=attention, dropout=0.3).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            res = time_batches(
                model=model,
                dataloader=train_loader,
                criterion=disease_criterion,
                optimizer=optimizer,
                device=device,
                is_custom=True,
                crop_criterion=crop_criterion,
                crop_loss_lambda=0.3,
                num_batches=num_batches,
            )
        else:
            model = create_model(backbone_name=model_name, num_classes=num_classes, pretrained=True, dropout=0.3).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            res = time_batches(
                model=model,
                dataloader=train_loader,
                criterion=disease_criterion,
                optimizer=optimizer,
                device=device,
                is_custom=False,
                num_batches=num_batches,
            )

        # Bottleneck diagnosis
        data_s = res["data_loading_mean_s"]
        compute_s = res["compute_mean_s"]
        copy_s = res["copy_mean_s"]
        data_pct = (data_s / res["total_batch_mean_s"]) * 100
        compute_pct = (compute_s / res["total_batch_mean_s"]) * 100
        copy_pct = (copy_s / res["total_batch_mean_s"]) * 100

        if data_s > compute_s:
            bottleneck = "DATA LOADING"
            bottleneck_desc = f"Data loading dominates ({data_pct:.1f}% of batch time vs compute {compute_pct:.1f}%)."
        else:
            bottleneck = "COMPUTE"
            bottleneck_desc = f"Compute dominates ({compute_pct:.1f}% of batch time vs data loading {data_pct:.1f}%)."

        res["bottleneck"] = bottleneck
        res["bottleneck_description"] = bottleneck_desc
        res["data_pct"] = round(data_pct, 1)
        res["compute_pct"] = round(compute_pct, 1)
        res["copy_pct"] = round(copy_pct, 1)
        res["estimated_seconds_per_epoch"] = round(res["total_batch_mean_s"] * batches_per_epoch, 2)

        results["models"][model_name] = res

        print("-" * 60)
        print(f"Results for {model_name}:")
        print(f"  Data-loading wait:    {res['data_loading_mean_s']*1000:8.2f} ms ({data_pct:5.1f}%)")
        print(f"  Host-to-device copy:  {res['copy_mean_s']*1000:8.2f} ms ({copy_pct:5.1f}%)")
        print(f"  Forward pass:         {res['forward_mean_s']*1000:8.2f} ms")
        print(f"  Backward pass:        {res['backward_mean_s']*1000:8.2f} ms")
        print(f"  Optimizer step:       {res['optimizer_mean_s']*1000:8.2f} ms")
        print(f"  Total compute (f+b+o):{res['compute_mean_s']*1000:8.2f} ms ({compute_pct:5.1f}%)")
        print(f"  Total per batch:      {res['total_batch_mean_s']*1000:8.2f} ms")
        print(f"  Est. sec per epoch:   {res['estimated_seconds_per_epoch']:8.2f} s ({res['estimated_seconds_per_epoch']/60:.2f} min)")
        print(f"  --> BOTTLENECK: {bottleneck} ({bottleneck_desc})")
        print("-" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile training speed across models.")
    parser.add_argument("--models", nargs="+", default=["resnet18", "custom_none"], help="Models to profile")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of timed batches (default: 20)")
    parser.add_argument("--device", type=str, default="mps", help="Device to run on (default: mps)")
    parser.add_argument("--splits-dir", type=str, default="splits", help="Splits directory (default: splits)")
    parser.add_argument("--data-root", type=str, default="data", help="Data root path (default: data)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (default: 0)")
    parser.add_argument("--persistent-workers", action="store_true", default=False, help="Use persistent_workers")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor")
    parser.add_argument("--pin-memory", action="store_true", default=False, help="DataLoader pin_memory")
    args = parser.parse_args()

    run_profiling(
        models=args.models,
        batch_size=args.batch_size,
        device_name=args.device,
        splits_dir=args.splits_dir,
        data_root=args.data_root,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    main()
