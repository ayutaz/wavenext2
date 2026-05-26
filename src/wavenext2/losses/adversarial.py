"""Hinge GAN loss (GAN-WaveNeXt 2、WaveFit-PT 準拠)。

D loss: mean over sub-D of [relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()]
G loss: mean over sub-D of [-D(fake).mean()]

Discriminator は activation なしの logit を返す前提 (T-M2.2、open-questions §C6)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["HingeGANLoss"]


class HingeGANLoss(nn.Module):
    """Hinge formulation の GAN loss (sub-D 平均)。"""

    def d_loss(
        self, d_real_list: list[torch.Tensor], d_fake_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Args: 各 sub-D の real/fake logit list. Returns scalar (sub-D 平均)."""
        loss = d_real_list[0].new_zeros(())
        for d_real, d_fake in zip(d_real_list, d_fake_list):
            loss = loss + F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        return loss / len(d_real_list)

    def g_loss(self, d_fake_list: list[torch.Tensor]) -> torch.Tensor:
        """Generator 側: -D(fake).mean() の sub-D 平均。"""
        loss = d_fake_list[0].new_zeros(())
        for d_fake in d_fake_list:
            loss = loss - d_fake.mean()
        return loss / len(d_fake_list)
