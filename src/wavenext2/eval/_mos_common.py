"""_mos_common.py — UTMOS / NISQA 連携の共通ユーティリティ (T-M4.2).

本体 (Python 3.13 + torch>=2.10) は UTMOS22 の重い fairseq 依存や NISQA の旧 torch 依存を
**top-level で import しない** (lazy import に徹する)。checkpoint / モデル未 setup 時は silent NaN を
出さず `EvalModelNotFoundError` を raise する。実モデルの clone/download は scripts/setup_eval_models.py
(ネットワーク + 別 venv 構築要、ユーザー実行) が担う。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["EvalModelNotFoundError", "chunked", "to_wav_list"]

EVAL_ROOT = Path("eval_models")


class EvalModelNotFoundError(RuntimeError):
    """UTMOS / NISQA のモデル・パッケージが見つからないとき raise (silent NaN を出さない)."""


def to_wav_list(wavs: Any, sr: int) -> list[np.ndarray]:
    """入力 (path / np.ndarray / torch.Tensor のリスト or 単体) を mono float32 numpy の list に正規化."""
    if isinstance(wavs, (str, Path)):
        wavs = [wavs]
    elif hasattr(wavs, "ndim") and not isinstance(wavs, list):  # 単一配列/tensor
        wavs = [wavs]
    out: list[np.ndarray] = []
    for w in wavs:
        if isinstance(w, (str, Path)):
            import soundfile as sf

            y, _ = sf.read(str(w), dtype="float32")
            y = y.mean(axis=1) if y.ndim > 1 else y
        else:
            if hasattr(w, "detach"):
                w = w.detach().cpu().numpy()
            y = np.asarray(w, dtype=np.float32).reshape(-1)
        out.append(y.astype(np.float32))
    return out


def chunked(items: list[Any], size: int):
    """list を size ごとに分割 (バッチ推論用)."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
