"""ConvNeXt block — WaveNeXt 2 の核ビルディングブロック (GAN/Diff 両対応).

GAN モード (`conditioning_dim=None`) では純粋な ConvNeXt block。
Diff モード (`conditioning_dim>0`) では block 入口で noise-level conditioning を
**additive bias** として 1 回注入する (FastDiff 流、FiLM ではない)。

入出力 shape は常に (B, dim, T)。重み初期化は本ブロックでは行わず、利用側
(Generator) が `apply()` で一括 init する (Vocos と同方針)。

詳細: docs/architecture.md §2 / §5.4、docs/open-questions.md §C1 / §C7。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ConvNeXtBlock"]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (depthwise conv → LN → MLP → LayerScale → residual).

    Args:
        dim: チャネル次元 (embed_dim)。
        intermediate_dim: MLP 中間次元 (通常 dim の 3 倍)。
        kernel_size: depthwise conv のカーネル長 (奇数)。
        layer_scale_init_value: LayerScale γ の初期値。
        conditioning_dim: Diff の noise-level 埋め込み次元。None で GAN モード。
    """

    def __init__(
        self,
        dim: int = 512,
        intermediate_dim: int = 1536,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        conditioning_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or intermediate_dim <= 0:
            raise ValueError(
                f"dim and intermediate_dim must be positive, got {dim}, {intermediate_dim}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if conditioning_dim is not None and conditioning_dim <= 0:
            raise ValueError(f"conditioning_dim must be positive or None, got {conditioning_dim}")

        self.dim = dim
        self.conditioning_dim = conditioning_dim

        # Diff のみ: per-block 独立な additive-bias projection (block 間で共有しない)
        self.fc_t = nn.Linear(conditioning_dim, dim) if conditioning_dim is not None else None

        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Args: x (B, dim, T); cond (B, conditioning_dim) — Diff のみ. Returns (B, dim, T)."""
        if self.fc_t is not None and cond is None:
            raise ValueError(
                f"block built with conditioning_dim={self.conditioning_dim} "
                "but forward() got cond=None"
            )
        if self.fc_t is None and cond is not None:
            raise ValueError("block built without conditioning but forward() received cond")

        # Diff: block 入口で T 方向 broadcast の additive bias を 1 回注入
        if self.fc_t is not None:
            x = x + self.fc_t(cond).unsqueeze(-1)  # (B, dim, 1)

        residual = x
        x = self.dwconv(x)  # (B, dim, T)
        x = x.transpose(1, 2)  # (B, T, dim) — LayerNorm / Linear は channels_last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x  # LayerScale
        x = x.transpose(1, 2)  # (B, dim, T)
        return residual + x
