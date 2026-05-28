"""post_filter.py — Time-invariant spectral enhancement FIR の apply.

論文 §3.3 / docs/architecture.md §5 / Okamoto21 §3.3 の time-invariant post-filter。
dev set で 1 度だけ fit した linear-phase FIR (`post_filter/fir.npy`、512 tap) を、推論時に全発話で
同じく畳み込む。低 iteration diffusion vocoder で失われがちな高域 detail を加算補償する。

畳み込みの慣例 (CRITICAL、T-M3.4 §6.1):
- FIR は fftshift 済みで **中央 tap (index N//2 = 256) が delta 中心** の linear-phase。
- torch / numpy 両パスとも **full 畳み込み → `[N//2 : N//2 + T]` 切り出し** で統一する。
  これにより delta@256 FIR は厳密な identity (zero-shift) となり、長さも保存される。
- np.convolve(mode="same") は偶数長 FIR で start=(N-1)//2 を使い 1-sample 非対称になるため不採用
  (チケット §2 擬似コードの start=(N-1)//2 もこの理由で N//2 に訂正、§8.3)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import oaconvolve

__all__ = ["apply_post_filter", "load_post_filter"]

FIR_LENGTH = 512  # Okamoto21 §3.3 (= n_fft、linear-phase 用)


def load_post_filter(fir_path: str | Path) -> np.ndarray:
    """`post_filter/fir.npy` を読み込み、長さ / dtype を検証して返す."""
    fir = np.load(fir_path).astype(np.float32)
    if fir.ndim != 1:
        raise ValueError(f"FIR must be 1-D, got shape {fir.shape}")
    if fir.shape[0] != FIR_LENGTH:
        raise ValueError(f"Expected FIR length {FIR_LENGTH}, got {fir.shape[0]}")
    return fir


def apply_post_filter(
    audio: torch.Tensor | np.ndarray,
    fir: np.ndarray,
) -> torch.Tensor | np.ndarray:
    """合成波形に time-invariant FIR を畳み込む (linear-phase, length-preserving).

    torch.Tensor / np.ndarray 両受け (GPU 上 post-filter 適用 / RTF 測定対応)。入力と同じ型・
    dtype・長さで返す。fftshift 済 FIR の中央 tap (index N//2) を基準に full 畳み込みを切り出すため、
    delta@(N//2) の FIR は厳密に identity になる。

    Args:
        audio: (T,) torch.Tensor または np.ndarray, dtype=float32, 範囲 [-1, 1]。
        fir:   (512,) np.ndarray, fftshift 済み linear-phase FIR。

    Returns:
        入力と同じ型で同じ長さ (T,)。
    """
    k = fir.shape[0]
    start = k // 2  # 中央 tap 位置を基準に full conv を切り出す (delta@256 → identity)

    if isinstance(audio, torch.Tensor):
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        t = audio.shape[0]
        fir_t = torch.from_numpy(np.ascontiguousarray(fir)).to(audio.device, dtype=audio.dtype)
        # conv1d は相関なので kernel を flip して畳み込み化、padding=k-1 で full conv を得る。
        full = torch.nn.functional.conv1d(
            audio.view(1, 1, -1),
            fir_t.flip(0).view(1, 1, -1),
            padding=k - 1,
        ).view(-1)  # length = t + k - 1
        return full[start : start + t]

    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
    t = audio.shape[0]
    # scipy.signal.oaconvolve (overlap-add): np.convolve(mode="same") の左右非対称を回避。
    full = oaconvolve(audio, fir, mode="full").astype(np.float32)  # length = t + k - 1
    return full[start : start + t]
