"""Log-mel-spectrogram 抽出モジュール.

SoT (single source of truth): mel 抽出パラメータ (n_fft / hop_length / win_length /
n_mels / f_min / f_max / sample_rate) は config 経由のみで受け取る。本モジュール内に
これらのマジック値のデフォルトを持たせない (GAN hop=300 / Diff hop=256 の取り違え防止)。

`torchaudio.transforms.MelSpectrogram(power=1.0, mel_scale="slaney", norm="slaney",
center=True)` を内部利用し、forward 末尾で `torch.log(torch.clamp(mel, min=eps))`
(自然対数 + 下限 clamp) を適用する。

確定根拠: docs/architecture.md §6.5、docs/training.md §1.2、docs/open-questions.md §C3。
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

__all__ = ["LogMelSpectrogram"]


class LogMelSpectrogram(nn.Module):
    """波形 (B, T_audio) → log-mel (B, n_mels, T_mel)。学習パラメータを持たない。

    Args:
        sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max:
            mel 抽出パラメータ。全て必須 (SoT: config から明示的に渡す)。
        eps: log clamp の下限。既定 1e-5 (docs/open-questions.md §C3 / CLAUDE.md 確定値)。
            ※ 本実装は scratch 学習で warm-start しないため Vocos の 1e-7 は採用しない。
        mel_transform: テスト用に注入可能な mel 変換 (None なら torchaudio で構築)。
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        eps: float = 1e-5,
        mel_transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps

        self.mel = mel_transform or torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,  # magnitude (Vocos)。power=2.0 (Tacotron2) ではない
            mel_scale="slaney",
            norm="slaney",
            center=True,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Args: audio (B, T_audio) or (T_audio,). Returns log-mel (B, n_mels, T_mel)."""
        mel = self.mel(audio)
        return torch.log(torch.clamp(mel, min=self.eps))  # 自然対数

    @classmethod
    def from_config(cls, cfg: dict) -> LogMelSpectrogram:
        """config dict から構築する factory (M1 全モジュール共通の factory パターン)。

        必須キーは __init__ の引数名 (sample_rate, n_fft, hop_length, win_length,
        n_mels, f_min, f_max)。YAML の構造 (sub_model 直下 + mel サブセクション) から
        この flat dict への写像は呼び出し側 (T-M2.5 / T-M3.2 の config loader) が行う。
        eps は "eps" または "log_eps" キーで受ける (config は log_eps)。
        """
        required = {"sample_rate", "n_fft", "hop_length", "win_length", "n_mels", "f_min", "f_max"}
        missing = required - cfg.keys()
        if missing:
            raise ValueError(f"missing required mel config keys: {sorted(missing)} (SoT)")
        kwargs = {k: cfg[k] for k in required}
        if "eps" in cfg:
            kwargs["eps"] = cfg["eps"]
        elif "log_eps" in cfg:
            kwargs["eps"] = cfg["log_eps"]
        return cls(**kwargs)
