from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str | Path = "results/confusion_matrix.png",
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save publication-grade confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype("float"), row_sums, where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=16, pad=15)
    plt.ylabel("True Label", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_curves(
    history: Dict[str, Any],
    output_path: str | Path = "results/training_curves.png",
) -> None:
    """Plot training and validation loss and macro-F1 curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Cross-Entropy Loss Across Epochs", fontsize=13)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Macro-F1 plot
    axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="green", marker="^")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", color="orange", linestyle="--", marker="x")
    axes[1].set_title("Validation Macro-F1 & Accuracy Across Epochs", fontsize=13)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
