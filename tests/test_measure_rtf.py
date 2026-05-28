"""T-M4.3: RTF 計測 (measure_rtf) の unit テスト.

docs/tickets/T-M4.3-rtf.md §5。CPU テストは default tier、GPU は @pytest.mark.gpu で分離。
軽量 mock model で計測ロジック (warmup 除外 / 1-thread 制限と復元 / dispatch / RTF 式 / schema)
を検証する。
"""

from __future__ import annotations

import time

import pytest
import torch
from torch import nn

from wavenext2.eval.measure_rtf import measure_rtf


class _MockSynth(nn.Module):
    """synthesize(mel) を持つ Diff-like mock (固定 sleep で RTF 式を検証可能)。"""

    hop_length = 256

    def __init__(self, sleep: float = 0.0) -> None:
        super().__init__()
        self.sleep = sleep
        self.calls = 0

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        return torch.zeros(mel.shape[0], mel.shape[-1] * self.hop_length)


class _MockForwardOnly(nn.Module):
    """synthesize を持たない GAN-like mock (forward dispatch を検証)。"""

    hop_length = 300

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.zeros(mel.shape[0], mel.shape[-1] * self.hop_length)


def _mels(n: int = 4, t_mel: int = 20) -> list[torch.Tensor]:
    return [torch.randn(1, 128, t_mel) for _ in range(n)]


# --------------------------------------------------------------------------- #
# schema / finite
# --------------------------------------------------------------------------- #
def test_rtf_schema() -> None:
    out = measure_rtf(_MockSynth(), _mels(), device="cpu", n_warmup=1, n_measure=5)
    assert set(out) == {"rtf_mean", "rtf_std", "rtf_median", "n", "device", "compiled"}
    assert out["n"] == 5
    assert out["device"] == "cpu"
    assert out["compiled"] is False


def test_rtf_median_finite_positive() -> None:
    out = measure_rtf(_MockSynth(sleep=0.001), _mels(), device="cpu", n_warmup=1, n_measure=10)
    assert out["rtf_mean"] > 0 and out["rtf_median"] > 0


# --------------------------------------------------------------------------- #
# CPU 1-thread 制限 + 復元
# --------------------------------------------------------------------------- #
def test_cpu_single_thread_during_measure() -> None:
    observed: list[int] = []

    class _ObserveThreads(nn.Module):
        hop_length = 256

        def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
            observed.append(torch.get_num_threads())
            return torch.zeros(mel.shape[0], mel.shape[-1] * self.hop_length)

    measure_rtf(_ObserveThreads(), _mels(), device="cpu", n_warmup=1, n_measure=3)
    assert all(t == 1 for t in observed)  # 計測中は 1-thread


def test_thread_count_restored() -> None:
    before = torch.get_num_threads()
    measure_rtf(_MockSynth(), _mels(), device="cpu", n_warmup=1, n_measure=3)
    assert torch.get_num_threads() == before  # finally で復元


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_diff_uses_synthesize() -> None:
    m = _MockSynth()
    measure_rtf(m, _mels(), device="cpu", n_warmup=2, n_measure=5)
    assert m.calls == 7  # warmup 2 + measure 5 すべて synthesize 経由


def test_dispatch_gan_falls_back_to_forward() -> None:
    # synthesize を持たない mock でも forward dispatch で計測完走。
    out = measure_rtf(_MockForwardOnly(), _mels(), device="cpu", n_warmup=1, n_measure=3)
    assert out["rtf_median"] >= 0


# --------------------------------------------------------------------------- #
# RTF 式 / warmup
# --------------------------------------------------------------------------- #
def test_rtf_formula() -> None:
    # sleep=t の mock で rtf ≈ sleep / audio_sec。audio_sec = t_mel*hop/sr。
    sleep = 0.02
    t_mel = 20
    sr = 24000
    out = measure_rtf(
        _MockSynth(sleep=sleep), _mels(n=1, t_mel=t_mel), device="cpu", n_warmup=1, n_measure=5,
        sample_rate=sr,
    )
    audio_sec = t_mel * 256 / sr
    expected = sleep / audio_sec
    assert out["rtf_median"] == pytest.approx(expected, rel=0.5)  # sleep の OS 揺らぎで緩め


def test_warmup_excluded_from_count() -> None:
    m = _MockSynth()
    measure_rtf(m, _mels(), device="cpu", n_warmup=3, n_measure=4)
    assert m.calls == 7  # warmup 3 は calls に含むが n_measure は 4 のみ集計
    # 戻り値 n は measure 回数のみ
    out = measure_rtf(_MockSynth(), _mels(), device="cpu", n_warmup=3, n_measure=4)
    assert out["n"] == 4


# --------------------------------------------------------------------------- #
# 実モデル結合 (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_gan_synthesize_alias() -> None:
    from wavenext2.models.gan_wavenext2 import GANWaveNext2

    g = GANWaveNext2(T=2, sub_model_cfg={"n_fft": 256, "hop_length": 64, "win_length": 256})
    out = measure_rtf(g, [torch.randn(1, 128, 10)], device="cpu", n_warmup=1, n_measure=2)
    assert out["rtf_median"] > 0
