"""T-M4.1: 客観評価指標 (MCD / log F0 RMSE / evaluate_dataset) の unit テスト.

docs/tickets/T-M4.1-objective-metrics.md §5。MCD は librosa MFCC + 自前 DTW (proxy) のため
絶対値は論文と系統差があるが (§8.2)、identical≈0 / 歪み単調増加 / voiced マスク / NaN セーフを
検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wavenext2.eval.compute_metrics import (
    compute_log_f0_rmse,
    compute_mcd,
    evaluate_dataset,
)

SR = 24000


def _voiced_tone(f0: float = 150.0, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)).astype(
        np.float64
    )


# --------------------------------------------------------------------------- #
# MCD
# --------------------------------------------------------------------------- #
def test_mcd_identical_zero() -> None:
    y = _voiced_tone()
    assert compute_mcd(y, y) == pytest.approx(0.0, abs=1e-3)


def test_mcd_increases_with_noise() -> None:
    y = _voiced_tone()
    rng = np.random.default_rng(0)
    m_small = compute_mcd(y, y + 0.01 * rng.standard_normal(len(y)))
    m_large = compute_mcd(y, y + 0.2 * rng.standard_normal(len(y)))
    assert 0 < m_small < m_large  # 歪み増で単調増加


def test_mcd_accepts_torch_and_numpy() -> None:
    y = _voiced_tone()
    assert compute_mcd(torch.tensor(y), torch.tensor(y)) == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# log F0 RMSE
# --------------------------------------------------------------------------- #
def test_log_f0_rmse_identical_zero() -> None:
    y = _voiced_tone()
    assert compute_log_f0_rmse(y, y) == pytest.approx(0.0, abs=1e-4)


def test_log_f0_rmse_pitch_shift() -> None:
    # 150→170 Hz の log RMSE は ln(170/150) ≈ 0.1252 に近い (voiced 全フレーム)。
    rmse = compute_log_f0_rmse(_voiced_tone(150.0), _voiced_tone(170.0))
    assert rmse == pytest.approx(np.log(170 / 150), abs=0.03)


def test_log_f0_rmse_silent_returns_nan() -> None:
    assert np.isnan(compute_log_f0_rmse(np.zeros(SR), np.zeros(SR)))


def test_log_f0_rmse_very_short_returns_nan() -> None:
    # 1 frame 未満 (極短) で NaN、例外を投げない。
    assert np.isnan(compute_log_f0_rmse(np.zeros(64), np.zeros(64)))


# --------------------------------------------------------------------------- #
# evaluate_dataset
# --------------------------------------------------------------------------- #
def test_evaluate_dataset_nanmean_skips_failures() -> None:
    y = _voiced_tone()
    pairs = [(y, y), (y, _voiced_tone(170.0)), (np.zeros(SR), np.zeros(SR))]
    summary, per_utt = evaluate_dataset(pairs)
    assert len(per_utt) == 3
    assert summary["n"] == 3.0
    assert summary["n_skipped"] == 1.0  # 無音ペアのみ skip
    assert not np.isnan(summary["log_f0_rmse_mean"])  # nanmean で汚染されない


def test_evaluate_dataset_schema_prefix() -> None:
    y = _voiced_tone()
    summary, _ = evaluate_dataset([(y, y)])
    for key in ("mcd_mean", "mcd_std", "log_f0_rmse_mean", "log_f0_rmse_std", "n", "n_skipped"):
        assert key in summary


def test_evaluate_dataset_deterministic() -> None:
    y = _voiced_tone()
    pairs = [(y, _voiced_tone(160.0)), (y, y)]
    s1, _ = evaluate_dataset(pairs)
    s2, _ = evaluate_dataset(pairs)
    assert s1 == s2  # 同一入力 → 同一出力


def test_evaluate_dataset_metrics_subset() -> None:
    y = _voiced_tone()
    summary, per_utt = evaluate_dataset([(y, y)], metrics=("log_f0_rmse",))
    assert "log_f0_rmse_mean" in summary
    assert "mcd_mean" not in summary
    assert "mcd" not in per_utt[0]
