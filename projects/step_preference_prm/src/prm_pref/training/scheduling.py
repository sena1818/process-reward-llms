from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_linear_warmup_decay_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    """Build the experiment's step-based linear warmup/decay schedule.

    ``total_steps`` is the complete fixed training budget.  Consequently a
    one-epoch protocol must pass precisely its one-epoch optimizer-update
    count; then its final scheduled update has learning rate zero.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - step, 0)
        decay_steps = max(total_steps - warmup_steps, 1)
        return float(remaining) / float(decay_steps)

    return LambdaLR(optimizer, multiplier)
