"""Multi-Scale Discriminator (MSD ×3、MPD なし、WaveFit-PT 準拠)。

論文 §4.1「Discriminator は WaveFit と同一」より、MelGAN 流の 3 スケール MSD を採用。
隣接 sub-D 間で AvgPool1d で audio を downsample。MPD は不使用 (open-questions §C5)。

各 sub-D (NLayerDiscriminator): ReflectionPad+Conv(1→16,k15) + downsample conv ×4
(k41,s4,grouped) + channel conv(k5) + logits conv(k3→1)、全 Conv1d に weight_norm、
LeakyReLU(0.2)。出力は 3 sub-D × (logits, 中間特徴 list) で FM loss / hinge GAN に供給。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__ = ["MultiScaleDiscriminator", "NLayerDiscriminator", "SubDiscOutput"]


class SubDiscOutput(NamedTuple):
    """単一 sub-discriminator の出力。"""

    logits: torch.Tensor  # (B, 1, T') — activation なし (hinge GAN 用)
    features: list[torch.Tensor]  # 各中間層の特徴 (FM loss 用)


class NLayerDiscriminator(nn.Module):
    """単一 scale 用 discriminator (MelGAN 流、7 層 Conv1d + LeakyReLU)。"""

    def __init__(
        self,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4,
        max_channels: int = 1024,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        layers: list[nn.Module] = [
            nn.Sequential(nn.ReflectionPad1d(7), weight_norm(nn.Conv1d(1, ndf, kernel_size=15)))
        ]
        nf = ndf
        for _ in range(n_layers):
            nf_prev = nf
            nf = min(nf_prev * downsampling_factor, max_channels)
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        nf_prev,
                        nf,
                        kernel_size=downsampling_factor * 10 + 1,
                        stride=downsampling_factor,
                        padding=downsampling_factor * 5,
                        groups=nf_prev // 4,
                    )
                )
            )
        nf_prev = nf
        nf = min(nf_prev * 2, max_channels)
        layers.append(weight_norm(nn.Conv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2)))
        layers.append(weight_norm(nn.Conv1d(nf, 1, kernel_size=3, stride=1, padding=1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> SubDiscOutput:
        features: list[torch.Tensor] = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # 最終 logits 以外は LeakyReLU + 特徴収集
                h = F.leaky_relu(h, self.leaky_relu_slope)
                features.append(h)
        return SubDiscOutput(logits=h, features=features)


class MultiScaleDiscriminator(nn.Module):
    """MSD ×3 (WaveFit-PT 準拠、MPD なし)。出力は list[SubDiscOutput] (len=num_D)。"""

    def __init__(
        self,
        num_D: int = 3,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4,
        max_channels: int = 1024,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        self.sub_discriminators = nn.ModuleList(
            NLayerDiscriminator(ndf, n_layers, downsampling_factor, max_channels, leaky_relu_slope)
            for _ in range(num_D)
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[SubDiscOutput]:
        """Args: x (B, 1, T) or (B, T) 波形. Returns list[SubDiscOutput] (len num_D)."""
        if x.dim() == 2:  # (B, T) → (B, 1, T) defensive
            x = x.unsqueeze(1)
        outputs: list[SubDiscOutput] = []
        h = x
        for k, sub_d in enumerate(self.sub_discriminators):
            outputs.append(sub_d(h))
            if k < self.num_D - 1:
                h = self.pool(h)
        return outputs

    @classmethod
    def from_config(cls, cfg: dict) -> MultiScaleDiscriminator:
        allowed = {
            "num_D",
            "ndf",
            "n_layers",
            "downsampling_factor",
            "max_channels",
            "leaky_relu_slope",
        }
        return cls(**{k: v for k, v in cfg.items() if k in allowed})
