"""compute_metrics.py — 客観評価指標 (MCD + log F0 RMSE) (T-M4.1).

- MCD (Mel-Cepstral Distortion): librosa MFCC + 自前 DTW alignment (ticket §6.1 最終手段 C)。
  pymcd / mel-cepstral-distance は Windows cp313 prebuilt wheel が無いため外部依存を排除し、
  librosa.feature.mfcc (mel-spectrogram → log → DCT) を mel-cepstrum の proxy として使う。
  **絶対値は SPTK mel-cepstrum ベースの論文値と系統差が出るため相対比較主軸** (§8.2)。
- log F0 RMSE: pyworld DIO + StoneMask で F0 抽出 → 両者 voiced のフレームのみ natural-log RMSE。

入力はすべて mono 波形 [-1, 1] (numpy or torch.Tensor)。F0 抽出失敗 (無音/極短) は例外でなく
np.nan を返し、集計側 (evaluate_dataset) が np.nanmean で除外する。
参考: docs/training.md §5.2 / docs/milestones.md §M4.1。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyworld

__all__ = [
    "compute_log_f0_rmse",
    "compute_mcd",
    "evaluate_dataset",
]

SR_DEFAULT = 24000
F0_FLOOR = 71.0  # pyworld DIO デフォルト (Hz)
F0_CEIL = 800.0
FRAME_PERIOD = 5.0  # ms
MCD_N_MFCC = 25  # c0..c24 を取り c1..c24 を使う (c0=energy 除外)
EPS = 1e-8
_MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)  # MCD dB スケール係数


def _to_numpy_mono(y: Any) -> np.ndarray:
    """torch.Tensor / np.ndarray → mono float64 numpy 1-D に正規化."""
    if hasattr(y, "detach"):  # torch.Tensor
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=tuple(range(y.ndim - 1))) if y.shape[-1] > y.shape[0] else y.mean(axis=-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
    return y.reshape(-1)


def _mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """librosa MFCC (c1..c24、c0=energy 除外) を (n_frames, 24) で返す."""
    m = librosa.feature.mfcc(y=y.astype(np.float32), sr=sr, n_mfcc=MCD_N_MFCC)
    return m[1:, :].T  # (n_frames, 24)


def compute_mcd(y_true: Any, y_pred: Any, sr: int = SR_DEFAULT) -> float:
    """Mel-Cepstral Distortion (dB) を DTW alignment ありで計算.

    librosa MFCC (c1..c24) を抽出し DTW で整列、整列フレーム上の cepstrum L2 を dB 化。
    同一音声で ≈ 0、歪み増で単調増加。
    """
    a = _to_numpy_mono(y_true)
    b = _to_numpy_mono(y_pred)
    mfcc_a = _mfcc(a, sr)  # (Ta, 24)
    mfcc_b = _mfcc(b, sr)  # (Tb, 24)
    if mfcc_a.shape[0] == 0 or mfcc_b.shape[0] == 0:
        return float("nan")
    # DTW alignment (librosa は (D, wp) を返す。wp は終点→始点の path)。
    _, wp = librosa.sequence.dtw(X=mfcc_a.T, Y=mfcc_b.T, metric="euclidean")
    diff = mfcc_a[wp[:, 0]] - mfcc_b[wp[:, 1]]  # (path_len, 24)
    sq = np.sum(diff**2, axis=1)  # (path_len,)
    return float(_MCD_K * np.mean(np.sqrt(sq)))


def compute_log_f0_rmse(
    y_true: Any,
    y_pred: Any,
    sr: int = SR_DEFAULT,
    f0_floor: float = F0_FLOOR,
    f0_ceil: float = F0_CEIL,
    frame_period: float = FRAME_PERIOD,
) -> float:
    """log F0 RMSE (両者 voiced のフレームのみで評価). 無音/極短は np.nan."""
    a = _to_numpy_mono(y_true)
    b = _to_numpy_mono(y_pred)

    def _f0(y: np.ndarray) -> np.ndarray:
        f0, t = pyworld.dio(
            y, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period
        )
        return pyworld.stonemask(y, f0, t, sr)

    f0_a = _f0(a)
    f0_b = _f0(b)
    n = min(len(f0_a), len(f0_b))
    if n == 0:
        return float("nan")
    f0_a, f0_b = f0_a[:n], f0_b[:n]
    voiced = (f0_a > 0) & (f0_b > 0)
    if not np.any(voiced):
        return float("nan")
    log_diff = np.log(f0_a[voiced]) - np.log(f0_b[voiced])
    return float(np.sqrt(np.mean(log_diff**2)))


def _load_wav(path: Any, sr: int) -> np.ndarray:
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def evaluate_dataset(
    pairs: list[tuple[Any, Any]],
    sr: int = SR_DEFAULT,
    num_workers: int = 0,  # noqa: ARG001 (process pool は M6 full eval で有効化、現状 serial)
    metrics: tuple[str, ...] = ("mcd", "log_f0_rmse"),
) -> tuple[dict[str, float], list[dict]]:
    """(gt, synth) のリスト (path または波形配列) を評価して (summary, per_utterance) を返す.

    F0 抽出失敗は np.nan として個別記録し、summary は np.nanmean / np.nanstd で除外集計
    (n_skipped に件数を計上)。決定性確保のため per-utterance は pairs 順を保持する。
    summary の key は prefix 規約 `{metric}_mean` / `{metric}_std` + `n` / `n_skipped` + `sr`。
    """
    per_utt: list[dict] = []
    for idx, (gt, synth) in enumerate(pairs):
        y_true = _load_wav(gt, sr) if isinstance(gt, (str, Path)) else _to_numpy_mono(gt)
        y_pred = _load_wav(synth, sr) if isinstance(synth, (str, Path)) else _to_numpy_mono(synth)
        row: dict = {"index": idx}
        if "mcd" in metrics:
            row["mcd"] = compute_mcd(y_true, y_pred, sr)
        if "log_f0_rmse" in metrics:
            row["log_f0_rmse"] = compute_log_f0_rmse(y_true, y_pred, sr)
        per_utt.append(row)

    summary: dict[str, float] = {"n": float(len(pairs)), "sr": float(sr)}
    for metric in metrics:
        if metric in ("mcd", "log_f0_rmse"):
            vals = np.array([r[metric] for r in per_utt], dtype=np.float64)
            # 全 nan (全 utterance で F0 失敗) は警告を出さず nan を返す。
            has_valid = len(vals) > 0 and bool(np.any(~np.isnan(vals)))
            summary[f"{metric}_mean"] = float(np.nanmean(vals)) if has_valid else float("nan")
            summary[f"{metric}_std"] = float(np.nanstd(vals)) if has_valid else float("nan")
    # skip 数 = どれか 1 指標が nan の utterance 件数 (F0 失敗 or MFCC 空)。各指標は独立に
    # nanmean 集約済なので集計自体は汚染されない。per-metric 内訳が要れば per_utt を参照。
    n_skipped = sum(
        1 for r in per_utt if any(isinstance(r.get(m), float) and np.isnan(r[m]) for m in metrics)
    )
    summary["n_skipped"] = float(n_skipped)
    return summary, per_utt
