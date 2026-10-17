"""
Unit tests for weighted_ce_v2 loss function.

Requirements:
(a) For each of the 12 classes, minimise the loss over a free logit vector for a single sample
    and assert the optimal true-class probability is 0.908 +/- 0.01 under v2, and print the
    same numbers for v1 so the user can inspect the 0.64-0.97 spread.
(b) Assert v2 equals standard label-smoothed CE when all weights are 1.
"""

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.focal_loss import (
    compute_class_weights,
    get_loss_function,
    LabelSmoothedWeightedCrossEntropy,
)


def get_dataset_class_weights() -> torch.Tensor:
    """Compute exact inverse class weights from the project's train split."""
    train_df = pd.read_csv("splits/dedup/train.csv")
    return compute_class_weights(train_df, num_classes=12, method="inverse")


def test_loss_v2_optimal_true_class_probability():
    """
    (a) For each of the 12 classes, minimise the loss over a free logit vector for a single sample
    and assert the optimal true-class probability is 0.908 +/- 0.01 under v2.
    Also print the optimal probabilities under v1 to show the 0.64 - 0.97 spread caused by
    PyTorch's class-weighted smoothing term.
    """
    weights = get_dataset_class_weights()
    label_smoothing = 0.10
    num_classes = 12
    # Theoretical optimal target probability: (1 - eps) + eps / C = 0.90 + 0.10 / 12 = 0.908333...
    theoretical_optimal = (1.0 - label_smoothing) + label_smoothing / num_classes

    crit_v1 = get_loss_function("weighted_ce_v1", class_weights=weights, label_smoothing=label_smoothing)
    crit_v2 = get_loss_function("weighted_ce_v2", class_weights=weights, label_smoothing=label_smoothing)

    v1_probs = []
    v2_probs = []

    print("\n" + "=" * 80)
    print("TEST: Optimal True-Class Probability per Class (Single Sample Logit Optimization)")
    print("=" * 80)
    print(f"{'Class':<8} {'Weight':<10} {'v1 (PyTorch)':<16} {'v2 (Unweighted Smooth)':<22} {'Target':<10}")
    print("-" * 80)

    for c in range(num_classes):
        target = torch.tensor([c], dtype=torch.long)

        # Optimize v1
        logits_v1 = torch.zeros(1, num_classes, requires_grad=True)
        opt_v1 = torch.optim.Adam([logits_v1], lr=0.1)
        for _ in range(800):
            opt_v1.zero_grad()
            loss1 = crit_v1(logits_v1, target)
            loss1.backward()
            opt_v1.step()
        p1 = float(F.softmax(logits_v1, dim=-1)[0, c].item())
        v1_probs.append(p1)

        # Optimize v2
        logits_v2 = torch.zeros(1, num_classes, requires_grad=True)
        opt_v2 = torch.optim.Adam([logits_v2], lr=0.1)
        for _ in range(800):
            opt_v2.zero_grad()
            loss2 = crit_v2(logits_v2, target)
            loss2.backward()
            opt_v2.step()
        p2 = float(F.softmax(logits_v2, dim=-1)[0, c].item())
        v2_probs.append(p2)

        print(f"Class {c:<2d}   {weights[c].item():<10.4f} {p1:<16.4f} {p2:<22.4f} {theoretical_optimal:.4f}")

        # Assert v2 is 0.908 +/- 0.01
        assert abs(p2 - 0.908) <= 0.01, f"Class {c} v2 optimal prob {p2:.4f} is outside 0.908 +/- 0.01"

    print("-" * 80)
    print(f"Summary: v1 spread = [{min(v1_probs):.4f}, {max(v1_probs):.4f}] (range = {max(v1_probs)-min(v1_probs):.4f})")
    print(f"Summary: v2 spread = [{min(v2_probs):.4f}, {max(v2_probs):.4f}] (constant = {theoretical_optimal:.4f})")
    print("=" * 80)

    # Confirm v1 shows the documented 0.64 - 0.97 spread
    assert min(v1_probs) < 0.65, f"Expected v1 min prob < 0.65, got {min(v1_probs):.4f}"
    assert max(v1_probs) > 0.95, f"Expected v1 max prob > 0.95, got {max(v1_probs):.4f}"


def test_loss_v2_equals_standard_ce_when_weights_are_ones():
    """
    (b) Assert v2 equals standard label-smoothed CE when all weights are 1 (and when weights=None).
    """
    torch.manual_seed(42)
    num_classes = 12
    batch_size = 16
    label_smoothing = 0.10

    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    # Standard PyTorch label-smoothed CE
    standard_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="mean")
    loss_std = standard_ce(logits, targets)

    # v2 with explicit ones weights
    weights_ones = torch.ones(num_classes, dtype=torch.float32)
    crit_v2_ones = LabelSmoothedWeightedCrossEntropy(weight=weights_ones, label_smoothing=label_smoothing, reduction="mean")
    loss_v2_ones = crit_v2_ones(logits, targets)

    # v2 with weight=None
    crit_v2_none = LabelSmoothedWeightedCrossEntropy(weight=None, label_smoothing=label_smoothing, reduction="mean")
    loss_v2_none = crit_v2_none(logits, targets)

    assert torch.allclose(loss_v2_ones, loss_std, atol=1e-6), (
        f"v2 with weights=1 ({loss_v2_ones.item()}) does not match standard CE ({loss_std.item()})"
    )
    assert torch.allclose(loss_v2_none, loss_std, atol=1e-6), (
        f"v2 with weights=None ({loss_v2_none.item()}) does not match standard CE ({loss_std.item()})"
    )
