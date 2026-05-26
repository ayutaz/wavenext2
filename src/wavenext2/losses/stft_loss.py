"""Multi-resolution STFT loss (Spectral Convergence + log-magnitude L1)。

3 resolution: n_fft [512,1024,2048] / win [360,900,1800] / hop [80,150,300]。
各 resolution で SC = ||S_t - S_p||_F / (||S_t||_F + eps)、Mag = |log(S_t+eps) - log(S_p+eps)|.mean()。
最終は 3 resolution 平均で (sc, mag) を別々に返す (重み付けは compute_total_loss)。
M3 (Diff) / M4 (eval) でも再利用。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MultiResolutionSTFTLoss"]

DEFAULT_FFTS = (512, 1024, 2048)
DEFAULT_WIN_LENGTHS = (360, 900, 1800)
DEFAULT_HOP_SIZES = (80, 150, 300)
DEFAULT_EPS = 1.0e-5


class _STFTLossSingleResolution(nn.Module):
    def __init__(self, n_fft: int, win_length: int, hop_size: int, eps: float = DEFAULT_EPS):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_size = hop_size
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mag_pred = self._stft_mag(y_pred)
        mag_true = self._stft_mag(y_true)
        sc = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + self.eps)
        mag = (torch.log(mag_true + self.eps) - torch.log(mag_pred + self.eps)).abs().mean()
        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """3 resolution の SC + log-mag L1 を平均し (sc, mag) を返す。"""

    def __init__(
        self,
        n_ffts: tuple[int, ...] = DEFAULT_FFTS,
        win_lengths: tuple[int, ...] = DEFAULT_WIN_LENGTHS,
        hop_sizes: tuple[int, ...] = DEFAULT_HOP_SIZES,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if not (len(n_ffts) == len(win_lengths) == len(hop_sizes)):
            raise ValueError("n_ffts / win_lengths / hop_sizes must have same length")
        self.resolutions = nn.ModuleList(
            _STFTLossSingleResolution(n, w, h, eps=eps)
            for n, w, h in zip(n_ffts, win_lengths, hop_sizes)
        )

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: (B, T) 波形ペア. Returns (sc_mean, mag_mean) scalar."""
        sc_sum = y_pred.new_zeros(())
        mag_sum = y_pred.new_zeros(())
        for res in self.resolutions:
            sc, mag = res(y_pred, y_true)
            sc_sum = sc_sum + sc
            mag_sum = mag_sum + mag
        n = len(self.resolutions)
        return sc_sum / n, mag_sum / n
