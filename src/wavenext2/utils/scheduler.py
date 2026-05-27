"""scheduler.py — InverseLR scheduler (T-M2.5).

指数 warmup × inverse-power-law decay の合成 (WaveFit-PT 流):

    lr(step) = base_lr * (1 - warmup ** (step + 1)) * (1 + step / inv_gamma) ** (-power)

- 第 1 項 `(1 - warmup**(step+1))`: step=0 で `1-warmup` (≈0)、step→∞ で 1 に漸近する緩やかな warmup。
- 第 2 項 `(1 + step/inv_gamma)**(-power)`: inverse power-law decay。

Default: inv_gamma=200000, power=0.5, warmup=0.999 (docs/training.md §2.4)。
G/D 別 optimizer にそれぞれアタッチする (base_lr=1e-4 / 2e-4)。
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["InverseLR"]


class InverseLR(LRScheduler):
    """指数 warmup + inverse power-law decay の LR scheduler.

    Args:
        optimizer: wrap する optimizer。
        inv_gamma: decay スケール (step)。既定 200000。
        power: decay 指数。既定 0.5。
        warmup: 指数 warmup の底 (∈ (0, 1))。既定 0.999。
        last_epoch: resume 用 step (既定 -1 = fresh)。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        inv_gamma: float = 200000.0,
        power: float = 0.5,
        warmup: float = 0.999,
        last_epoch: int = -1,
    ) -> None:
        if not 0.0 < warmup < 1.0:
            raise ValueError(f"warmup must be in (0, 1), got {warmup}")
        if inv_gamma <= 0:
            raise ValueError(f"inv_gamma must be positive, got {inv_gamma}")
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.warmup = float(warmup)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch, 0)
        warmup_mult = 1.0 - self.warmup ** (step + 1)  # step=0 → 1-warmup, step→∞ → 1
        decay_mult = (1.0 + step / self.inv_gamma) ** (-self.power)
        return [base_lr * warmup_mult * decay_mult for base_lr in self.base_lrs]
