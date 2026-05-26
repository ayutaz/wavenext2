"""STFTModule の unit テスト (T-M1.2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from wavenext2.models.stft import STFTModule

SNAP_DIR = Path(__file__).parent / "snapshots"

GAN = dict(n_fft=2048, hop_length=300, win_length=1200)
DIFF = dict(n_fft=1024, hop_length=256, win_length=1024)


# --- shape / channel count -----------------------------------------------------
def test_shape_gan(device):
    m = STFTModule(**GAN).to(device)
    y = torch.randn(2, 80 * 300, device=device)
    # 2F-2 = n_fft = 2048
    assert m(y, 80).shape == (2, 2048, 80)


def test_shape_diff(device):
    m = STFTModule(**DIFF).to(device)
    y = torch.randn(2, 94 * 256, device=device)
    assert m(y, 94).shape == (2, 1024, 94)


def test_output_channels_equal_nfft():
    # 2F-2 = (n_fft//2+1) + (n_fft//2-1) = n_fft
    assert STFTModule(**GAN)(torch.randn(1, 24000), 80).shape[1] == 2048
    assert STFTModule(**DIFF)(torch.randn(1, 24064), 94).shape[1] == 1024


def test_truncation_exact():
    m = STFTModule(**GAN)
    # center=True で余剰フレームが出るが出力 T は厳密に t_mel
    assert m(torch.randn(2, 24000), 80).shape[-1] == 80


# --- 数値整合 -----------------------------------------------------------------
def test_real_imag_layout():
    m = STFTModule(**DIFF)
    y = torch.randn(2, 24064)
    out = m(y, 94)
    spec = torch.stft(
        y,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(1024),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True,
    )[..., :94]
    f = 1024 // 2 + 1  # 513
    assert torch.allclose(out[:, :f, :], spec.real, atol=1e-5)
    assert torch.allclose(out[:, f:, :], spec.imag[:, 1:-1, :], atol=1e-5)


def test_dc_nyquist_imag_are_zero():
    # DC/Nyquist bin の虚部はほぼ 0 (だから削除して良い)
    y = torch.randn(2, 24000)
    spec = torch.stft(
        y,
        n_fft=2048,
        hop_length=300,
        win_length=1200,
        window=torch.hann_window(1200),
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    assert spec.imag[:, 0, :].abs().max() < 1e-4
    assert spec.imag[:, -1, :].abs().max() < 1e-4


def test_roundtrip_sine_440hz():
    # 440Hz 正弦波の複素 magnitude ピーク bin が 440Hz 近傍に集中することを確認
    sr, freq, n_fft = 24000, 440.0, 2048
    t = torch.arange(sr, dtype=torch.float32) / sr
    y = torch.sin(2 * math.pi * freq * t).unsqueeze(0)
    out = STFTModule(**GAN)(y, 80)
    f = n_fft // 2 + 1  # 1025
    # output レイアウトから複素 magnitude を復元 (real=out[:F], imag は内側 F-2 bin)
    real = out[:, :f, :]
    imag_full = torch.zeros_like(real)
    imag_full[:, 1:-1, :] = out[:, f:, :]
    mag = torch.sqrt(real**2 + imag_full**2).mean(dim=(0, 2))  # (F,)
    bin_440 = round(freq / sr * n_fft)  # 38
    assert abs(int(mag.argmax()) - bin_440) <= 1  # 437〜445Hz に集中
    # 遠方 bin より少なくとも 20x 大きい (低/高周波にエネルギーが漏れていない)
    assert mag[bin_440] > 20 * mag[[10, 100, 500, 1000]].mean()


# --- 性質 ---------------------------------------------------------------------
def test_no_learnable_params():
    assert sum(p.numel() for p in STFTModule(**GAN).parameters()) == 0


def test_deterministic_cpu():
    m = STFTModule(**GAN)
    y = torch.randn(2, 24000)
    assert torch.equal(m(y, 80), m(y, 80))


def test_gradient_flow():
    m = STFTModule(**DIFF)
    y = torch.randn(2, 24064, requires_grad=True)
    m(y, 94).sum().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_preserved(dtype):
    m = STFTModule(**DIFF)
    out = m(torch.randn(2, 24064, dtype=dtype), 94)
    assert out.dtype == dtype


def test_device_buffer():
    m = STFTModule(**GAN)
    if torch.cuda.is_available():
        m = m.to("cuda")
        assert m.window.device.type == "cuda"
    else:
        assert m.window.device.type == "cpu"


# --- error handling ------------------------------------------------------------
def test_error_win_gt_nfft():
    with pytest.raises(ValueError, match="win_length"):
        STFTModule(n_fft=512, hop_length=128, win_length=1024)


def test_error_y_not_2d():
    with pytest.raises(ValueError, match="B, T_audio"):
        STFTModule(**GAN)(torch.randn(24000), 80)


def test_error_tmel_nonpositive():
    with pytest.raises(ValueError, match="t_mel"):
        STFTModule(**GAN)(torch.randn(2, 24000), 0)


def test_error_audio_too_short():
    # stft は走るが frame 数 (1+3000//300=11) が t_mel=80 に届かない → 自前 RuntimeError
    with pytest.raises(RuntimeError, match="ensure T_audio"):
        STFTModule(**GAN)(torch.randn(2, 3000), 80)


# --- concat 互換 (T-M1.6 前哨) -------------------------------------------------
def test_concat_compatibility():
    m = STFTModule(**GAN)
    stft_spec = m(torch.randn(2, 24000), 80)
    mel = torch.randn(2, 128, 80)
    x = torch.cat([mel, stft_spec], dim=1)
    assert x.shape == (2, 128 + 2048, 80)  # 2176


# --- dynamic range mismatch 計測 (§6.1 critical → T-M1.6 へ引き継ぎ) -----------
def test_input_scale_ratio_snapshot():
    """STFT-spec と log-mel のスケール比を記録 (T-M1.6 の LayerNorm 要否判断材料)."""
    import torchaudio

    sr = 24000
    t = torch.arange(sr, dtype=torch.float32) / sr
    y = torch.sin(2 * math.pi * 440.0 * t).unsqueeze(0)
    stft_spec = STFTModule(**GAN)(y, 80)

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=300,
        win_length=1200,
        n_mels=128,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
    )
    log_mel = torch.log(torch.clamp(mel_tf(y)[..., :80], min=1e-5))

    ratio = float(stft_spec.std() / log_mel.std())
    assert math.isfinite(ratio) and ratio > 0

    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / "stft_mel_scale_ratio.json"
    if not path.exists():
        path.write_text(json.dumps({"stft_std_over_logmel_std": round(ratio, 3)}, indent=2) + "\n")
        pytest.skip(f"baseline created: ratio={ratio:.3f}")
    ref = json.loads(path.read_text())["stft_std_over_logmel_std"]
    # ±50% 以内 (実装/torch のドリフト検知。比率自体の大きさは T-M1.6 で評価)
    assert abs(ratio - ref) <= 0.5 * ref
