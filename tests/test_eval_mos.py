"""T-M4.2: UTMOS / NISQA 連携の unit テスト.

docs/tickets/T-M4.2-utmos-nisqa.md §5。実モデル (speechmos / NISQA venv) は CI 不在のため、
未 setup 時の EvalModelNotFoundError、to_wav_list 正規化、backend 登録、monkeypatch backend
での schema を検証する。実モデル経路は @pytest.mark.slow / skipif。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from wavenext2.eval import run_nisqa, run_utmos
from wavenext2.eval._mos_common import EvalModelNotFoundError, chunked, to_wav_list
from wavenext2.eval.runner import _METRIC_BACKENDS

_HAS_SPEECHMOS = importlib.util.find_spec("speechmos") is not None


# --------------------------------------------------------------------------- #
# 共通ユーティリティ
# --------------------------------------------------------------------------- #
def test_to_wav_list_numpy_and_torch() -> None:
    a = np.zeros(100, dtype=np.float32)
    out = to_wav_list([a, torch.zeros(100)], sr=24000)
    assert len(out) == 2
    assert all(isinstance(w, np.ndarray) and w.dtype == np.float32 for w in out)


def test_to_wav_list_single_array() -> None:
    out = to_wav_list(np.zeros(100, dtype=np.float32), sr=24000)
    assert len(out) == 1


def test_chunked() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


# --------------------------------------------------------------------------- #
# 未 setup 時の明示エラー (silent NaN を出さない)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_HAS_SPEECHMOS, reason="speechmos 利用可能環境ではエラーにならない")
def test_utmos_missing_backend_raises() -> None:
    with pytest.raises(EvalModelNotFoundError, match="speechmos"):
        run_utmos.utmos_scores([np.zeros(24000, dtype=np.float32)])


def test_utmos_invalid_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        run_utmos.utmos_scores([np.zeros(100, dtype=np.float32)], backend="foo")


def test_nisqa_missing_venv_raises() -> None:
    # eval_models/.venv-nisqa が無い環境では EvalModelNotFoundError。
    with pytest.raises(EvalModelNotFoundError, match="NISQA"):
        run_nisqa.nisqa_scores([np.zeros(24000, dtype=np.float32)])


# --------------------------------------------------------------------------- #
# facade backend 登録 (import 副作用)
# --------------------------------------------------------------------------- #
def test_utmos_nisqa_backends_registered() -> None:
    assert "utmos" in _METRIC_BACKENDS
    assert "nisqa" in _METRIC_BACKENDS


# --------------------------------------------------------------------------- #
# monkeypatch backend で score / schema 検証 (実モデル非依存)
# --------------------------------------------------------------------------- #
def test_score_utmos_with_fake_speechmos(monkeypatch) -> None:
    # speechmos.utmos22_strong.run を fake してスコア集計を検証。
    import sys
    import types

    fake_mod = types.ModuleType("speechmos")
    fake_sub = types.SimpleNamespace(run=lambda y, sr: {"utmos22_strong": 4.0})  # noqa: ARG005
    fake_mod.utmos22_strong = fake_sub
    monkeypatch.setitem(sys.modules, "speechmos", fake_mod)

    out = run_utmos.score_utmos([np.zeros(100, dtype=np.float32) for _ in range(3)])
    assert out["utmos_mean"] == pytest.approx(4.0)
    assert out["utmos_std"] == pytest.approx(0.0)
    assert out["n"] == 3.0


def test_utmos_backend_facade_schema(monkeypatch) -> None:
    import sys
    import types

    fake_mod = types.ModuleType("speechmos")
    fake_mod.utmos22_strong = types.SimpleNamespace(
        run=lambda y, sr: {"utmos22_strong": 3.5}  # noqa: ARG005
    )
    monkeypatch.setitem(sys.modules, "speechmos", fake_mod)

    pairs = [(np.zeros(100, dtype=np.float32), np.zeros(100, dtype=np.float32)) for _ in range(2)]
    summary, per = _METRIC_BACKENDS["utmos"](pairs, 24000)
    assert summary["utmos_mean"] == pytest.approx(3.5)
    assert len(per) == 2
    assert all("utmos" in row for row in per)
