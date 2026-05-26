"""WaveNeXt-based generator (Fig. 2a).

embed Conv1d → LayerNorm → ConvNeXt × 8 → LayerNorm → Linear(dim, n_fft+2)
→ Linear(n_fft+2, hop, bias=False) → reshape → clip(-1, 1)。

**embed kernel size の確定 (2026-05-27 調査)**: architecture.md 構造図は kernel=7 だが、
それだと concat 入力 (GAN 2176ch) に対し embed が 7.8M となり sub-model 計 ~22M。
論文 Table 1 は GAN sub-model = 14.985M (2 iter=29.97M=2×, 5 iter=74.93M=5× の厳密倍数)
かつ WaveNeXt baseline (14.98M) とほぼ同一。kernel=7 over 2176 は数学的に不可能。
**embed kernel=1** (2176→512 の 1×1 射影 = 1.11M) なら GAN 計 15.43M で Table 1 +2.9% に収まり、
mel+STFT concat 設計も尊重する。時間方向の文脈は後段 ConvNeXt block (depthwise kernel=7) が担う。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from wavenext2.models.convnext import ConvNeXtBlock

__all__ = ["WaveNextGenerator"]


class WaveNextGenerator(nn.Module):
    """GAN/Diff 両対応の WaveNeXt generator。

    Args:
        input_channels: embed 入力 ch。GAN=128+2048=2176, Diff=128+1024=1152。
        n_fft: STFT size。GAN=2048, Diff=1024。
        hop_length: hop。GAN=300, Diff=256。
        dim: ConvNeXt embed dim (512)。
        intermediate_dim: ConvNeXt MLP hidden (1536)。
        n_blocks: ConvNeXt block 数 (8)。
        kernel_size: ConvNeXt depthwise kernel (7)。
        conditioning_dim: None=GAN, 512=Diff。指定時は各 block で cond を射影なし additive 注入
            (per-block fc_t は Table 1 と +14% 乖離のため撤去、open-questions §C7)。
        layer_scale_init: ConvNeXt LayerScale 初期値 (1e-6)。
        embed_kernel_size: 入力 embed Conv1d の kernel。既定 1 (Table 1 整合、上記参照)。
        final_activation: "clip" (既定、GAN の波形/残差出力) / "tanh" (M3 compile fallback 予約) /
            "none" (Diff の ε 予測など unbounded 出力。ε~N(0,1) は clip すると破壊的)。
        block_factory: ConvNeXtBlock 差し替え用 DI (None で既定)。
    """

    def __init__(
        self,
        input_channels: int,
        n_fft: int,
        hop_length: int,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
        conditioning_dim: int | None = None,
        layer_scale_init: float = 1e-6,
        embed_kernel_size: int = 1,
        final_activation: str = "clip",
        block_factory: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if final_activation not in ("clip", "tanh", "none"):
            raise ValueError(
                f"final_activation must be 'clip', 'tanh' or 'none', got {final_activation}"
            )
        self.input_channels = input_channels
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conditioning_dim = conditioning_dim
        self.final_activation = final_activation

        block_cls = block_factory or ConvNeXtBlock

        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=embed_kernel_size,
            padding=embed_kernel_size // 2,
            bias=True,
        )
        self.norm_in = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    layer_scale_init_value=layer_scale_init,
                    conditioning_dim=conditioning_dim,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.linear_1 = nn.Linear(dim, n_fft + 2, bias=True)
        self.linear_2 = nn.Linear(n_fft + 2, hop_length, bias=False)  # bias=False は確定仕様

        self._init_weights()

    def _init_weights(self) -> None:
        """全 Conv1d/Linear を trunc_normal_(std=0.02), bias=0 で初期化。

        ConvNeXtBlock は自前 init を持たないため、本 walk が block 内部の
        fc_t/dwconv/pwconv1/pwconv2 も初期化する (T-M1.1 §9 申し送り)。LayerNorm は既定のまま。
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Args: x (B, input_channels, T_mel); cond (B, conditioning_dim) or None.
        Returns: (B, T_mel*hop_length) — [-1,1] のノイズ成分。"""
        if (self.conditioning_dim is None) != (cond is None):
            raise ValueError(
                "cond must be provided iff conditioning_dim is not None "
                f"(conditioning_dim={self.conditioning_dim}, cond is None={cond is None})"
            )
        b, _, t_mel = x.shape
        h = self.embed(x)  # (B, dim, T_mel)
        h = self.norm_in(h.transpose(1, 2)).transpose(1, 2)  # channels_last で LN
        for block in self.blocks:
            h = block(h, cond=cond) if self.conditioning_dim is not None else block(h)
        h = self.norm_out(h.transpose(1, 2))  # (B, T_mel, dim)
        h = self.linear_1(h)  # (B, T_mel, n_fft+2)
        h = self.linear_2(h)  # (B, T_mel, hop_length)
        h = h.reshape(b, t_mel * self.hop_length)
        if self.final_activation == "tanh":
            return torch.tanh(h)
        if self.final_activation == "none":
            return h  # unbounded (Diff の ε 予測など)
        return torch.clip(h, min=-1.0, max=1.0)
