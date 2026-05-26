"""STFT module — 前ステップ波形を STFT して 2F-2 ch の STFT-spec に変換 (論文 §3.1).

GAN では `y_{t-1}`、Diff では `x_t` を入力波形として受け取り、Hann window STFT
(center=True, onesided=True) を mel-spec の時間長 T_mel に truncate し、実部 F bin と
虚部 F-2 bin (DC/Nyquist の常時 0 な虚部を削除) を concat して返す。

出力チャネル数は `2F-2 = n_fft` (F = n_fft//2 + 1)。mel (128 ch) と concat して
generator 入力 (GAN 2176 / Diff 1152 ch) を構成する。詳細は docs/architecture.md §3。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["STFTModule"]


class STFTModule(nn.Module):
    """波形 (B, T_audio) → STFT-spec (B, 2F-2, T_mel)。学習パラメータを持たない。

    Args:
        n_fft: STFT 点数。F = n_fft // 2 + 1。
        hop_length: STFT hop。mel-spec の hop と一致させること。
        win_length: Hann 窓長 (<= n_fft)。
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        if win_length > n_fft:
            raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # window は学習しない buffer (device 追従)。決定的に再生成可能なので persistent=False。
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y: torch.Tensor, t_mel: int) -> torch.Tensor:
        """Args: y (B, T_audio); t_mel: mel-spec 側の時間長. Returns (B, 2F-2, t_mel)."""
        if y.dim() != 2:
            raise ValueError(f"expected y of shape (B, T_audio), got {tuple(y.shape)}")
        if t_mel <= 0:
            raise ValueError(f"t_mel must be positive, got {t_mel}")

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (B, F, T_stft) complex
        if spec.shape[-1] < t_mel:
            raise RuntimeError(
                f"STFT produced {spec.shape[-1]} frames but t_mel={t_mel} required; "
                "ensure T_audio >= t_mel * hop_length"
            )
        spec = spec[..., :t_mel]  # mel-spec の時間長に truncate

        real = spec.real  # (B, F, t_mel)
        imag = spec.imag[:, 1:-1, :]  # (B, F-2, t_mel) — DC/Nyquist 虚部を削除
        return torch.cat([real, imag], dim=1)  # (B, 2F-2, t_mel)
