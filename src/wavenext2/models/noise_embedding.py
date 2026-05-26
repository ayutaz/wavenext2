"""Diff-WaveNeXt 2 の noise-level embedding (FastDiff 流 sinusoidal + FC×2 SiLU).

連続値 noise level `c = √(1-ᾱ)` (∈ [0,1]) を `(B, 512)` の埋め込みに写像する
**shared head**。この 512次元出力は各 ConvNeXt block (T-M1.1) の入口で **射影なしの additive bias**
として直接加算される。最終 `Linear(512,512)+SiLU` が実質の共有 projection を担うため、block 側に
per-block 射影層は持たない (Table 1 整合; 詳細は下記)。

**per-block fc_t 撤去 (2026-05-27)**: 当初は各 block が独立 `Linear(512,512)` を持つ FastDiff/DiffWave
流だったが、8 block 計 2.1M が論文 Table 1 の Diff sub-model=14.42M を +14% 超過するため撤去。
共有 head 直接加算で sub-model=14.354M (−0.46%)。注入形式は DiffWave/Okamoto21 の「共有 step
embedding を各層で additive」骨格に一致。

discrete index lookup の `nn.Embedding` ではなく、連続値を埋め込む点に注意。
確定根拠: docs/architecture.md §5.4、docs/open-questions.md §C7。
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["NoiseEmbedding", "sinusoidal_embedding"]


def sinusoidal_embedding(c: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """連続値 c の sinusoidal embedding。

    log-spaced freq: ``freq[i] = exp(-i * log(10000) / (half-1))`` (half = dim//2)。
    `freq[0]=1`, `freq[-1]=1e-4` で比は厳密に 10000。

    Args:
        c: (B,) 連続 noise level。
        dim: 埋め込み次元 (偶数)。

    Returns:
        (B, dim) = ``[sin(c·freq); cos(c·freq)]``。
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    log_scale = math.log(10000.0) / (half - 1)
    freq = torch.exp(-log_scale * torch.arange(half, dtype=c.dtype, device=c.device))  # (half,)
    args = c.unsqueeze(-1) * freq.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class NoiseEmbedding(nn.Module):
    """sinusoidal(128) → Linear(128,512) → SiLU → Linear(512,512) → SiLU.

    Args:
        sinusoidal_dim: sinusoidal 次元 (既定 128)。
        mid_dim: FC1 出力 (既定 512)。
        out_dim: 出力次元。ConvNeXtBlock の conditioning_dim と一致させる (既定 512)。
        input_rescale: sinusoidal 前に c に掛ける係数。既定 1.0 (FastDiff 忠実)。
            c∈[0,1] は周期を使い切れないため、M3.5 smoke 発散時に 1000.0 (DDPM step 相当)
            への切替を ablation する想定 (docs §8.1)。
    """

    def __init__(
        self,
        sinusoidal_dim: int = 128,
        mid_dim: int = 512,
        out_dim: int = 512,
        input_rescale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.input_rescale = input_rescale
        self.fc1 = nn.Linear(sinusoidal_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.act = nn.SiLU()  # = Swish

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B,) noise level → (B, out_dim)."""
        e = sinusoidal_embedding(c * self.input_rescale, dim=self.sinusoidal_dim)
        e = self.act(self.fc1(e))
        return self.act(self.fc2(e))
