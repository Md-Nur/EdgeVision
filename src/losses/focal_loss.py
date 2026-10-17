from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = 12,
    method: str = "inverse",
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Calculate class weights to balance loss for minority classes.
    Methods:
      - 'inverse': w_c = N / (C * count_c), normalized so mean is 1.0
      - 'effective': Class-Balanced Loss based on Effective Number of Samples (Cui et al., CVPR 2019)
    """
    counts = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    weights = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        cnt = counts.get(c, 1)
        if method == "effective":
            effective_num = 1.0 - np.power(beta, cnt)
            weights[c] = (1.0 - beta) / max(effective_num, 1e-8)
        else:  # 'inverse'
            weights[c] = total_samples / (num_classes * cnt)

    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with correctly computed target probability p_t,
    class weighting (alpha_t), and optional label smoothing.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * CE_smoothed(p, y)
    where p_t is the model's estimated probability for the true class y.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, num_classes) raw unnormalized logits
        targets: (batch_size) ground truth class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        # 1. Correctly compute p_t (predicted probability of ground-truth class)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 2. Modulating factor (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)

        # 3. Class weights (alpha_t)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_factor = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_factor

        # 4. Cross-entropy with optional label smoothing
        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)
            ce_loss = (1.0 - self.label_smoothing) * (-target_log_probs) + self.label_smoothing * smooth_loss
        else:
            ce_loss = -target_log_probs

        loss = focal_weight * ce_loss


        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothedWeightedCrossEntropy(nn.Module):
    """
    Weighted Cross-Entropy Loss with unweighted label smoothing (v2).
    
    Per-sample loss:
        loss_i = w[y_i] * CE_smoothed(logits_i, y_i)
    where label smoothing acts on unweighted log-probabilities:
        CE_smoothed(logits_i, y_i) = -sum_c y_smooth_c * log p_c
                                   = (1 - eps) * (-log p_{y_i}) + eps * (-1/C * sum_c log p_c)
    Averaged over the batch as:
        sum(w[y_i] * loss_i) / sum(w[y_i]) (when reduction == 'mean')
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, num_classes) raw unnormalized logits
        targets: (batch_size) ground truth class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)
            ce_loss = (1.0 - self.label_smoothing) * (-target_log_probs) + self.label_smoothing * smooth_loss
        else:
            ce_loss = -target_log_probs

        if self.weight is not None:
            w = self.weight.to(inputs.device).gather(0, targets)
            loss_i = w * ce_loss
            if self.reduction == "mean":
                return loss_i.sum() / (w.sum() + self.eps)
            elif self.reduction == "sum":
                return loss_i.sum()
            return loss_i
        else:
            if self.reduction == "mean":
                return ce_loss.mean()
            elif self.reduction == "sum":
                return ce_loss.sum()
            return ce_loss


def get_loss_function(
    loss_type: str = "weighted_ce_v2",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory to create appropriate loss criterion."""
    if loss_type == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    elif loss_type in ["weighted_ce", "weighted_ce_v1"]:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "weighted_ce_v2":
        return LabelSmoothedWeightedCrossEntropy(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

