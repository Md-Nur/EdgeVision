from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.training.scheduler import get_cosine_schedule_with_warmup


class Trainer:
    """Two-phase transfer learning trainer for crop disease classification."""

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        device: torch.device,
        class_names: List[str],
        models_dir: str | Path = "models",
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.device = device
        self.class_names = class_names
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
        }

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Run a single training epoch."""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.dataloaders["train"], desc="Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> Dict[str, Any]:
        """Run evaluation without gradients."""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []

        loader = self.dataloaders[split]
        for batch in tqdm(loader, desc=f"Evaluating ({split})", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = compute_metrics(y_true, y_pred, class_names=self.class_names)
        metrics["loss"] = running_loss / max(1, total_samples)
        return metrics

    def fit(
        self,
        phase1_epochs: int = 3,
        phase2_epochs: int = 15,
        phase1_lr: float = 1e-3,
        phase2_backbone_lr: float = 1e-5,
        phase2_head_lr: float = 1e-4,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute two-phase transfer learning:
          Phase 1: Frozen backbone (train classifier head only)
          Phase 2: Discriminative fine-tuning with Cosine Annealing
        """
        best_macro_f1 = -1.0
        patience_counter = 0

        # ==========================================
        # PHASE 1: Frozen Backbone (Warmup Head)
        # ==========================================
        if phase1_epochs > 0:
            print("\n" + "=" * 60)
            print(f"STARTING PHASE 1: Frozen Backbone ({phase1_epochs} epochs, LR={phase1_lr})")
            print("=" * 60)
            self.model.freeze_backbone()

            # Optimizer only for trainable parameters (head)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_p1 = torch.optim.AdamW(
                trainable_params,
                lr=phase1_lr,
                weight_decay=weight_decay,
            )

            for epoch in range(1, phase1_epochs + 1):
                train_loss = self.train_epoch(optimizer_p1)
                val_metrics = self.evaluate("val")

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_macro_f1"].append(val_metrics["macro_f1"])

                print(
                    f"[P1 Epoch {epoch:02d}/{phase1_epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
                )

                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    self._save_checkpoint("best_model.pth", val_metrics, epoch=epoch, phase=1)

        # ==========================================
        # PHASE 2: Discriminative Fine-Tuning
        # ==========================================
        if phase2_epochs > 0:
            print("\n" + "=" * 60)
            print(f"STARTING PHASE 2: Unfreezing Top Layers ({phase2_epochs} epochs)")
            print(f"Backbone LR: {phase2_backbone_lr}, Head LR: {phase2_head_lr}")
            print("=" * 60)
            self.model.unfreeze_top_layers(ratio=0.30)

            param_groups = self.model.get_parameter_groups(
                backbone_lr=phase2_backbone_lr,
                head_lr=phase2_head_lr,
                weight_decay=weight_decay,
            )
            optimizer_p2 = torch.optim.AdamW(param_groups)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer_p2,
                warmup_epochs=1,
                total_epochs=phase2_epochs,
                min_lr=1e-7,
            )

            for epoch in range(1, phase2_epochs + 1):
                train_loss = self.train_epoch(optimizer_p2)
                scheduler.step()
                val_metrics = self.evaluate("val")

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_accuracy"].append(val_metrics["accuracy"])
                self.history["val_macro_f1"].append(val_metrics["macro_f1"])

                total_epoch = phase1_epochs + epoch
                print(
                    f"[P2 Epoch {epoch:02d}/{phase2_epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
                )

                if val_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = val_metrics["macro_f1"]
                    patience_counter = 0
                    self._save_checkpoint("best_model.pth", val_metrics, epoch=total_epoch, phase=2)
                    print(f"  --> Saved new best model checkpoint (Val Macro-F1: {best_macro_f1:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\n[Early Stopping] No Macro-F1 improvement for {early_stopping_patience} epochs.")
                        break

        # Always save last checkpoint
        self._save_checkpoint("last_model.pth", val_metrics, epoch=len(self.history["train_loss"]), phase=2)
        return self.history

    def _save_checkpoint(
        self,
        filename: str,
        val_metrics: Dict[str, Any],
        epoch: int,
        phase: int,
    ) -> None:
        """Save weights and training state."""
        save_path = self.models_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": self.model.state_dict(),
                "val_metrics": {
                    "accuracy": val_metrics.get("accuracy"),
                    "macro_f1": val_metrics.get("macro_f1"),
                    "weighted_f1": val_metrics.get("weighted_f1"),
                },
                "class_names": self.class_names,
                "backbone_name": getattr(self.model, "backbone_name", "unknown"),
            },
            save_path,
        )
