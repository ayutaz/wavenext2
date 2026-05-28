"""T-M5.1 / T-M5.2: divergence gate + run_smoke orchestrator + 1epoch config のテスト.

実 1 epoch 訓練は GPU + 実 LibriTTS-R 要のユーザー操作 (M6 課金前 GO/NO-GO)。ここでは機械判定可能な
divergence gate ロジック・orchestrator の metrics→判定・中間 config のパースを検証する。
"""

from __future__ import annotations

import json

import pytest

from scripts.run_smoke import evaluate_gate, render_report
from wavenext2.eval.gate import (
    OutputStats,
    diff_divergence_gate,
    gan_divergence_gate,
)
from wavenext2.utils.config import load_config

_GOOD_OUT = OutputStats(max_abs=0.8, std=0.1, all_finite=True)


# --------------------------------------------------------------------------- #
# GAN divergence gate
# --------------------------------------------------------------------------- #
def test_gan_gate_pass_on_healthy() -> None:
    r = gan_divergence_gate(
        loss_g=[5.0, 4.0, 3.0, 2.0],
        loss_adv=[1.0, 1.0, 1.0, 1.0],
        loss_d=[1.0, 0.8, 0.7, 0.6],
        output=_GOOD_OUT,
    )
    assert r.passed
    assert all(r.checks.values())


def test_gan_gate_fail_on_nan_loss() -> None:
    r = gan_divergence_gate(
        loss_g=[5.0, float("nan"), 3.0, 2.0],
        loss_adv=[1.0] * 4,
        loss_d=[0.5] * 4,
        output=_GOOD_OUT,
    )
    assert not r.passed
    assert "loss_g_finite failed" in r.reasons


def test_gan_gate_fail_on_d_too_strong() -> None:
    # loss_d が floor (0.01) を割ると D 強すぎ検出。
    r = gan_divergence_gate(
        loss_g=[5.0, 4.0, 3.0, 2.0], loss_adv=[1.0] * 4, loss_d=[0.005] * 4, output=_GOOD_OUT
    )
    assert not r.passed
    assert not r.checks["loss_d_above_floor"]


def test_gan_gate_fail_on_silent_output() -> None:
    r = gan_divergence_gate(
        loss_g=[5.0, 4.0, 3.0, 2.0],
        loss_adv=[1.0] * 4,
        loss_d=[0.5] * 4,
        output=OutputStats(max_abs=1e-6, std=1e-7, all_finite=True),
    )
    assert not r.passed
    assert not r.checks["output_not_silent"]


def test_gan_gate_fail_on_loss_adv_out_of_range() -> None:
    # loss_adv が 0 に張り付く (D 暴走 failure mode)。
    r = gan_divergence_gate(
        loss_g=[5.0, 4.0, 3.0, 2.0], loss_adv=[0.0] * 4, loss_d=[0.5] * 4, output=_GOOD_OUT
    )
    assert not r.passed
    assert not r.checks["loss_adv_in_range"]


def test_gan_gate_fail_on_increasing_loss() -> None:
    r = gan_divergence_gate(
        loss_g=[2.0, 3.0, 4.0, 5.0], loss_adv=[1.0] * 4, loss_d=[0.5] * 4, output=_GOOD_OUT
    )
    assert not r.checks["loss_g_decreasing"]


# --------------------------------------------------------------------------- #
# Diff divergence gate
# --------------------------------------------------------------------------- #
def test_diff_gate_pass_on_healthy() -> None:
    r = diff_divergence_gate(loss_mse=[1.0, 0.8, 0.6, 0.4], output=_GOOD_OUT)
    assert r.passed


def test_diff_gate_fail_on_inf() -> None:
    r = diff_divergence_gate(loss_mse=[1.0, float("inf"), 0.6, 0.4], output=_GOOD_OUT)
    assert not r.passed
    assert not r.checks["loss_finite"]


def test_diff_gate_fail_on_out_of_range_output() -> None:
    r = diff_divergence_gate(
        loss_mse=[1.0, 0.8, 0.6, 0.4], output=OutputStats(max_abs=1.5, std=0.2, all_finite=True)
    )
    assert not r.checks["output_in_range"]


def test_gate_short_history_passes_trend() -> None:
    # 履歴が短い (< 4) と傾向判定は pass 寄り (短い smoke を弾かない)。
    r = diff_divergence_gate(loss_mse=[1.0, 0.9], output=_GOOD_OUT)
    assert r.checks["loss_decreasing"]


# --------------------------------------------------------------------------- #
# run_smoke orchestrator
# --------------------------------------------------------------------------- #
def test_evaluate_gate_gan() -> None:
    metrics = {
        "loss_g": [5.0, 4.0, 3.0, 2.0],
        "loss_adv": [1.0] * 4,
        "loss_d": [0.5] * 4,
        "output": {"max_abs": 0.8, "std": 0.1, "all_finite": True},
    }
    r = evaluate_gate("gan", metrics)
    assert r.passed


def test_evaluate_gate_diff() -> None:
    metrics = {
        "loss_mse": [1.0, 0.8, 0.6, 0.4],
        "output": {"max_abs": 0.8, "std": 0.1, "all_finite": True},
    }
    assert evaluate_gate("diff", metrics).passed


def test_evaluate_gate_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        evaluate_gate("foo", {})


def test_render_report_contains_verdict() -> None:
    r = diff_divergence_gate(loss_mse=[1.0, 0.8, 0.6, 0.4], output=_GOOD_OUT)
    report = render_report("diff", r)
    assert "GO" in report
    assert "PASS" in report


def test_run_smoke_main_writes_json(tmp_path) -> None:
    from scripts.run_smoke import main

    metrics = {
        "loss_mse": [1.0, 0.8, 0.6, 0.4],
        "output": {"max_abs": 0.8, "std": 0.1, "all_finite": True},
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(metrics), encoding="utf-8")
    out = tmp_path / "gate.json"
    rc = main(["--mode", "diff", "--metrics-json", str(mpath), "--out", str(out)])
    assert rc == 0  # pass → exit 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["mode"] == "diff"


def test_run_smoke_main_fail_returns_nonzero(tmp_path) -> None:
    from scripts.run_smoke import main

    metrics = {
        "loss_mse": [1.0, float("nan"), 0.6, 0.4],
        "output": {"max_abs": 0.8, "std": 0.1, "all_finite": True},
    }
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["--mode", "diff", "--metrics-json", str(mpath)]) == 1  # fail → exit 1


# --------------------------------------------------------------------------- #
# 中間 config のパース
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "max_steps", "ckpt_dir"),
    [
        ("configs/gan_wavenext2_1epoch.yaml", 33000, "checkpoints/gan_1epoch"),
        ("configs/diff_wavenext2_1epoch.yaml", 33000, "checkpoints/diff_1epoch"),
    ],
)
def test_1epoch_config_parses(path: str, max_steps: int, ckpt_dir: str) -> None:
    cfg = load_config(path)
    assert cfg["train"]["max_steps"] == max_steps
    assert cfg["checkpoint"]["dir"] == ckpt_dir
    for key in ("model", "train", "validation", "checkpoint", "logging", "data"):
        assert key in cfg
