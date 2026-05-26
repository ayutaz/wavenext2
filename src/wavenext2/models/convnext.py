"""ConvNeXt block — WaveNeXt 2 の核ビルディングブロック (GAN/Diff 両対応).

GAN モード (`conditioning_dim=None`) では純粋な ConvNeXt block。
Diff モード (`conditioning_dim=dim`) では block 入口で共有 NoiseEmbedding 出力 (512次元) を
**parameter-free な additive bias** として注入する (per-block の射影層を持たない)。

**設計判断 (2026-05-27, エージェントチーム調査 + Table 1 連立復元)**: 当初は FastDiff 流に
per-block 独立 `Linear(512,512)` (8 block 計 2.1M) を持っていたが、論文 Table 1 の
Diff sub-model=14.42M を **+14% 超過**し丸め誤差で説明不可能なため棄却。projection なし
(共有 head を各 block で直接加算) なら sub-model=14.354M (−0.46%) で最も整合する。注入形式は
DiffWave/Okamoto21 (WaveNeXt 2 の sub-modeling 直系祖先) の「共有 step embedding を各層で
additive」の骨格に一致 (FiLM ではない)。詳細: docs/open-questions.md §C7。

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
        conditioning_dim: Diff の noise-level conditioning 次元。None で GAN モード。
            指定時は dim と一致必須 (共有 NoiseEmbedding 出力を射影なしで直接加算するため)。
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
        if conditioning_dim is not None and conditioning_dim != dim:
            # 射影なし直接加算のため cond 次元は feature 次元と一致必須 (per-block fc_t 撤去, §C7)
            raise ValueError(
                f"conditioning_dim must equal dim ({dim}) for parameter-free additive "
                f"injection, got {conditioning_dim}"
            )

        self.dim = dim
        self.conditioning_dim = conditioning_dim
        # Diff conditioning は parameter-free な additive bias (per-block 射影層は持たない, §C7)

        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Args: x (B, dim, T); cond (B, dim) — Diff のみ (共有 NoiseEmbedding 出力). Returns (B, dim, T)."""
        if self.conditioning_dim is not None and cond is None:
            raise ValueError(
                f"block built with conditioning_dim={self.conditioning_dim} "
                "but forward() got cond=None"
            )
        if self.conditioning_dim is None and cond is not None:
            raise ValueError("block built without conditioning but forward() received cond")

        # Diff: block 入口で共有 cond を T 方向 broadcast の additive bias として直接注入 (射影なし)
        if cond is not None:
            x = x + cond.unsqueeze(-1)  # (B, dim, 1)

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
