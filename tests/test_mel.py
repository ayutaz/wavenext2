"""LogMelSpectrogram の unit テスト (T-M1.3)."""

from __future__ import annotations

import math

import pytest
import torch

from wavenext2.data.mel import LogMelSpectrogram

GAN = dict(
    sample_rate=24000,
    n_fft=2048,
    hop_length=300,
    win_length=1200,
    n_mels=128,
    f_min=20.0,
    f_max=12000.0,
)
DIFF = dict(
    sample_rate=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=128,
    f_min=20.0,
    f_max=12000.0,
)
LOG_EPS = math.log(1e-5)  # ≈ -11.5129


# --- shape --------------------------------------------------------------------
def test_gan_shape_1sec(device):
    m = LogMelSpectrogram(**GAN).to(device)
    out = m(torch.randn(2, 24000, device=device))
    # center=True: T_mel = 1 + 24000//300 = 81
    assert out.shape == (2, 128, 81)


def test_diff_shape_1sec(device):
    m = LogMelSpectrogram(**DIFF).to(device)
    out = m(torch.randn(2, 24000, device=device))
    # 1 + 24000//256 = 94
    assert out.shape == (2, 128, 94)


def test_batch_dim_preserved():
    assert LogMelSpectrogram(**GAN)(torch.randn(4, 24000)).shape == (4, 128, 81)


def test_tmel_proportional_to_hop():
    m = LogMelSpectrogram(**GAN)
    t1 = m(torch.randn(1, 24000)).shape[-1]
    t2 = m(torch.randn(1, 48000)).shape[-1]
    assert t2 == 2 * t1 - 1  # 1 + 2N//hop = 2*(1+N//hop) - 1


def test_gan_diff_distinct_shape():
    # 取り違え検出器: 同じ audio で GAN/Diff の T_mel が異なる
    audio = torch.randn(1, 24000)
    assert LogMelSpectrogram(**GAN)(audio).shape[-1] != LogMelSpectrogram(**DIFF)(audio).shape[-1]


# --- range / numerics ---------------------------------------------------------
def test_output_lower_bound():
    out = LogMelSpectrogram(**GAN)(torch.randn(2, 24000))
    assert (out >= LOG_EPS - 1e-6).all()


def test_output_finite():
    out = LogMelSpectrogram(**GAN)(torch.randn(2, 24000))
    assert torch.isfinite(out).all()


def test_silence_is_log_eps():
    out = LogMelSpectrogram(**GAN)(torch.zeros(2, 24000))
    assert torch.allclose(out, torch.full_like(out, LOG_EPS), atol=1e-5)


def test_sine_frequency_ordering():
    # 1000Hz と 5000Hz の sine → ピーク mel band index は 1000 < 5000 の順
    sr = 24000
    t = torch.arange(sr, dtype=torch.float32) / sr
    m = LogMelSpectrogram(**GAN)
    peak_1k = int(m(torch.sin(2 * math.pi * 1000 * t).unsqueeze(0)).mean(dim=2).argmax())
    peak_5k = int(m(torch.sin(2 * math.pi * 5000 * t).unsqueeze(0)).mean(dim=2).argmax())
    assert peak_1k < peak_5k


def test_power_is_one():
    # HiFi-GAN/Vocos は power=1.0 (magnitude)。Tacotron2 の 2.0 と混同しないこと
    assert LogMelSpectrogram(**GAN).mel.spectrogram.power == 1.0


def test_deterministic():
    m = LogMelSpectrogram(**GAN)
    audio = torch.randn(2, 24000)
    assert torch.equal(m(audio), m(audio))


def test_no_learnable_params():
    assert sum(p.numel() for p in LogMelSpectrogram(**GAN).parameters()) == 0


# --- config / error -----------------------------------------------------------
def test_from_config_roundtrip():
    m = LogMelSpectrogram.from_config({**GAN, "log_eps": 1e-5})
    assert m.n_fft == 2048 and m.hop_length == 300 and m.eps == 1e-5


def test_from_config_eps_alias():
    # config の "log_eps" が eps にマップされる
    assert LogMelSpectrogram.from_config({**DIFF, "log_eps": 1e-6}).eps == 1e-6
    # "eps" 明示も優先して受ける
    assert LogMelSpectrogram.from_config({**DIFF, "eps": 1e-4}).eps == 1e-4


def test_from_config_missing_keys():
    with pytest.raises(ValueError, match="missing required mel config keys"):
        LogMelSpectrogram.from_config({"sample_rate": 24000})


def test_eps_nonpositive_raises():
    with pytest.raises(ValueError, match="eps must be positive"):
        LogMelSpectrogram(**GAN, eps=0.0)


def test_dependency_injection():
    # mel_transform を注入すると forward がそれを使う (torchaudio 非依存の高速 path)
    class _StubMel(torch.nn.Module):
        def forward(self, audio):
            return torch.full((audio.shape[0], 128, 10), 100.0)

    m = LogMelSpectrogram(**GAN, mel_transform=_StubMel())
    out = m(torch.randn(2, 24000))
    assert out.shape == (2, 128, 10)
    assert torch.allclose(out, torch.full_like(out, math.log(100.0)))  # log(clamp(100, eps))
