import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
):
    """Create Cosine Annealing LR scheduler with optional initial linear warmup."""
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
