"""T-M4.1: 統一 eval facade (runner.evaluate / EvalResult / backend registry) のテスト.

docs/tickets/T-M4.1-objective-metrics.md §5.1 facade テスト。tiny DiffWaveNext2 + 合成 dataset で
facade 動作・schema・JSON 永続化往復・backend dispatch を検証。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wavenext2.eval.runner import EvalResult, evaluate, register_metric_backend
from wavenext2.models.diff_wavenext2 import DiffWaveNext2

TINY = {
    "mel_channels": 32,
    "n_fft": 128,
    "hop_length": 64,
    "win_length": 128,
    "dim": 64,
    "intermediate_dim": 128,
    "n_blocks": 2,
    "kernel_size": 3,
    "sinusoidal_dim": 16,
    "cond_dim": 64,
}


@pytest.fixture(scope="module")
def model() -> DiffWaveNext2:
    torch.manual_seed(0)
    return DiffWaveNext2(sub_model_cfg=TINY)


def _dataset(n: int = 2, t_mel: int = 40) -> list[dict]:
    # voiced-ish GT audio (F0 抽出可能な長さ) + 32ch mel。
    sr = 24000
    out = []
    for _ in range(n):
        t = np.linspace(0, t_mel * 64 / sr, t_mel * 64, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        out.append({"mel": torch.randn(1, 32, t_mel), "audio": torch.tensor(audio)})
    return out


# --------------------------------------------------------------------------- #
# facade 動作 + schema
# --------------------------------------------------------------------------- #
def test_evaluate_facade_returns_eval_result(model: DiffWaveNext2) -> None:
    res = evaluate(model, _dataset(), metrics=("mcd", "log_f0_rmse"))
    assert isinstance(res, EvalResult)
    assert res.summary["n"] == 2.0
    assert len(res.per_utterance) == 2


def test_eval_result_schema_prefix(model: DiffWaveNext2) -> None:
    res = evaluate(model, _dataset(), metrics=("mcd", "log_f0_rmse"))
    for key in ("mcd_mean", "mcd_std", "log_f0_rmse_mean", "n", "n_skipped", "sr"):
        assert key in res.summary


def test_eval_result_json_roundtrip(model: DiffWaveNext2, tmp_path) -> None:
    # MCD は (voiced 要件なしで) 常に finite なので nan!=nan を避けて厳密往復を検証。
    res = evaluate(model, _dataset(), metrics=("mcd",), save_to=tmp_path / "r.json")
    assert (tmp_path / "r.json").exists()
    loaded = EvalResult.from_json(tmp_path / "r.json")
    assert loaded.summary == res.summary
    assert loaded.per_utterance == res.per_utterance


def test_evaluate_unknown_metric_raises(model: DiffWaveNext2) -> None:
    with pytest.raises(NotImplementedError, match="utmos"):
        evaluate(model, _dataset(), metrics=("utmos",))


def test_register_metric_backend_dispatch(model: DiffWaveNext2) -> None:
    # T-M4.2/4.3 が backend を登録する経路を検証 (dummy backend)。
    def _dummy(pairs, sr):  # noqa: ANN001, ARG001
        per = [{"dummy": 1.0} for _ in pairs]
        return {"dummy_mean": 1.0, "dummy_std": 0.0}, per

    register_metric_backend("dummy", _dummy)
    res = evaluate(model, _dataset(), metrics=("dummy",))
    assert res.summary["dummy_mean"] == 1.0
    assert all("dummy" in row for row in res.per_utterance)


def test_evaluate_post_filter_switch(model: DiffWaveNext2) -> None:
    # post_filter=None / delta FIR の 2 系列が出る (T-M3.4 連携)。
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0
    plain = evaluate(model, _dataset(), metrics=("log_f0_rmse",), post_filter=None)
    filtered = evaluate(model, _dataset(), metrics=("log_f0_rmse",), post_filter=fir)
    assert "log_f0_rmse_mean" in plain.summary
    assert "log_f0_rmse_mean" in filtered.summary
