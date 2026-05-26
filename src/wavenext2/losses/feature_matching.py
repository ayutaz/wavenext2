"""Feature matching loss (MSD ×3 の中間特徴 L1 平均、WaveFit-PT/HiFi-GAN 準拠)。

3 sub-D × N layer の中間特徴 (T-M2.2 が返す) の L1 距離を全 feature 数で平均。
real 側は target なので detach し、G 更新時に fake 側のみ勾配を流す。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FeatureMatchingLoss"]


class FeatureMatchingLoss(nn.Module):
    """3 sub-D × N layer の中間特徴 L1 平均。"""

    def forward(
        self,
        feats_real: list[list[torch.Tensor]],
        feats_fake: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Args: [[(B,C,T) ...] per sub-D] real/fake. Returns scalar (全 feature 平均)."""
        total = feats_fake[0][0].new_zeros(())
        n = 0
        for fr_d, ff_d in zip(feats_real, feats_fake):
            for fr_l, ff_l in zip(fr_d, ff_d):
                total = total + (fr_l.detach() - ff_l).abs().mean()  # real は target → detach
                n += 1
        return total / max(n, 1)
