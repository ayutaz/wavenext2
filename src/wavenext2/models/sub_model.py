"""Sub-model wrapper (論文 Fig 2b) — STFTModule + WaveNextGenerator (+ NoiseEmbedding)。

GAN/Diff の差異を吸収し、T-M2.4 / T-M3.1 が `nn.ModuleList([SubModel... ])` で直列/独立に
並べるだけで iteration / reverse sampling を組めるようにする thin layer。

mel と STFT-spec の channel concat 順序は **[mel (128ch), stft_spec (2F-2ch)]** で固定
(`CONCAT_ORDER`)。順序は引数化せず破壊的変更を防ぐ。

戻り値の意味 (T-M1.6 §6.1 CRITICAL を解決):
- SubModelGAN.forward → **n_t** (残差ノイズ成分、generator で clip[-1,1])。減算 `y_{t-1}=y_t-n_t`
  は呼び出し側 (T-M2.4) が行う (architecture.md §4)。
- SubModelDiff.forward → **ε_pred** (DDPM のノイズ予測、**clip しない**)。ε~N(0,1) は |ε|>1 が
  ~32% で clip すると破壊的なため generator は final_activation="none"。波形の clamp は
  reverse sampling (training.md §4.2) 側で x0_hat に対して行う。
"""

from __future__ import annotations

import torch
from torch import nn

from wavenext2.models.generator import WaveNextGenerator
from wavenext2.models.noise_embedding import NoiseEmbedding
from wavenext2.models.stft import STFTModule

__all__ = ["CONCAT_ORDER", "SubModelDiff", "SubModelGAN"]

# channel concat 順序の単一情報源 (SoT)。T-M2.4 / T-M3.1 が import 可能。
CONCAT_ORDER: tuple[str, str] = ("mel", "stft_spec")


def _stft_spec_channels(n_fft: int) -> int:
    return 2 * (n_fft // 2 + 1) - 2  # = 2F - 2 = n_fft


class SubModelGAN(nn.Module):
    """GAN-WaveNeXt 2 用 sub-model (conditioning なし)。out = n_t (clip[-1,1])。"""

    CONCAT_ORDER = CONCAT_ORDER

    def __init__(
        self,
        mel_channels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 300,
        win_length: int = 1200,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.mel_channels = mel_channels
        self.hop_length = hop_length
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.generator = WaveNextGenerator(
            input_channels=mel_channels + _stft_spec_channels(n_fft),
            n_fft=n_fft,
            hop_length=hop_length,
            dim=dim,
            intermediate_dim=intermediate_dim,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            conditioning_dim=None,
            final_activation="clip",
        )

    def forward(self, mel: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """mel (B,128,T_mel), y_prev (B, T_mel*hop) → n_t (B, T_mel*hop), clip[-1,1]."""
        t_mel = mel.shape[2]
        stft_spec = self.stft_module(y_prev, t_mel)  # (B, 2F-2, T_mel)
        x = torch.cat([mel, stft_spec], dim=1)  # [mel, stft_spec]
        return self.generator(x)

    @classmethod
    def from_config(cls, cfg: dict) -> SubModelGAN:
        allowed = {
            "mel_channels",
            "n_fft",
            "hop_length",
            "win_length",
            "dim",
            "intermediate_dim",
            "n_blocks",
            "kernel_size",
        }
        return cls(**{k: v for k, v in cfg.items() if k in allowed})


class SubModelDiff(nn.Module):
    """Diff-WaveNeXt 2 用 sub-model (noise conditioning あり)。out = ε_pred (clip なし)。"""

    CONCAT_ORDER = CONCAT_ORDER

    def __init__(
        self,
        mel_channels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
        sinusoidal_dim: int = 128,
        cond_dim: int = 512,
    ) -> None:
        super().__init__()
        self.mel_channels = mel_channels
        self.hop_length = hop_length
        self.cond_dim = cond_dim
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.noise_embedding = NoiseEmbedding(
            sinusoidal_dim=sinusoidal_dim, mid_dim=cond_dim, out_dim=cond_dim
        )
        self.generator = WaveNextGenerator(
            input_channels=mel_channels + _stft_spec_channels(n_fft),
            n_fft=n_fft,
            hop_length=hop_length,
            dim=dim,
            intermediate_dim=intermediate_dim,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            conditioning_dim=cond_dim,
            final_activation="none",  # ε 予測は clip しない
        )

    def forward(self, mel: torch.Tensor, x_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """mel (B,128,T_mel), x_t (B, T_mel*hop), c (B,) noise level → ε_pred (B, T_mel*hop)."""
        if c.dim() != 1:
            raise ValueError(f"c must be shape (B,), got {tuple(c.shape)}")
        cond = self.noise_embedding(c)  # (B, cond_dim)
        t_mel = mel.shape[2]
        stft_spec = self.stft_module(x_t, t_mel)
        x = torch.cat([mel, stft_spec], dim=1)
        return self.generator(x, cond=cond)

    @classmethod
    def from_config(cls, cfg: dict) -> SubModelDiff:
        allowed = {
            "mel_channels",
            "n_fft",
            "hop_length",
            "win_length",
            "dim",
            "intermediate_dim",
            "n_blocks",
            "kernel_size",
            "sinusoidal_dim",
            "cond_dim",
        }
        return cls(**{k: v for k, v in cfg.items() if k in allowed})
