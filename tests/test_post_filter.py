"""T-M3.4: post-filter (time-invariant FIR の apply / load) の unit テスト.

docs/tickets/T-M3.4-post-filter.md §5。畳み込みは torch / numpy 両パスとも「full conv →
[N//2 : N//2+T] 切り出し」で統一しているため、fftshift 済 delta@256 FIR は厳密な identity に
なる (チケット §2 擬似コードの crop start=(N-1)//2 は偶数長で 1-sample ズレるため N//2 に訂正、
§8.3)。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wavenext2.inference.infer_diff import reverse_sample
from wavenext2.inference.post_filter import apply_post_filter, load_post_filter
from wavenext2.models.diff_wavenext2 import DiffWaveNext2


def _delta_fir() -> np.ndarray:
    """fftshift 済 identity FIR (中央 tap index 256 に delta)。"""
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0
    return fir


# --------------------------------------------------------------------------- #
# apply (numpy)
# --------------------------------------------------------------------------- #
def test_apply_preserves_length() -> None:
    audio = np.random.randn(24000).astype(np.float32)
    out = apply_post_filter(audio, _delta_fir())
    assert out.shape == audio.shape
    # identity FIR なので全域で一致 (float32 oaconvolve 誤差のみ)。
    np.testing.assert_allclose(out, audio, atol=1e-5)


def test_apply_dtype_float32() -> None:
    audio = np.random.randn(24000).astype(np.float32)
    fir = (np.random.randn(512) * 1e-3).astype(np.float32)
    out = apply_post_filter(audio, fir)
    assert out.dtype == np.float32


def test_apply_rejects_2d_audio() -> None:
    audio = np.random.randn(2, 24000).astype(np.float32)
    with pytest.raises(ValueError, match="1-D"):
        apply_post_filter(audio, _delta_fir())


def test_apply_numpy_input_returns_numpy() -> None:
    out = apply_post_filter(np.random.randn(24000).astype(np.float32), _delta_fir())
    assert isinstance(out, np.ndarray)


def test_apply_no_nan_inf() -> None:
    audio = np.random.randn(24000).astype(np.float32)
    fir = (np.random.randn(512) * 1e-3).astype(np.float32)
    out = apply_post_filter(audio, fir)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# apply (torch)
# --------------------------------------------------------------------------- #
def test_apply_torch_input_returns_torch() -> None:
    audio = torch.randn(24000, dtype=torch.float32)
    out = apply_post_filter(audio, _delta_fir())
    assert isinstance(out, torch.Tensor)
    assert out.shape == audio.shape
    torch.testing.assert_close(out, audio, atol=1e-5, rtol=1e-5)


def test_apply_torch_rejects_2d() -> None:
    audio = torch.randn(2, 24000)
    with pytest.raises(ValueError, match="1-D"):
        apply_post_filter(audio, _delta_fir())


def test_apply_torch_matches_numpy() -> None:
    # torch / numpy パスが同一結果 (両受け実装の等価性、§2.5)。
    audio = np.random.randn(8000).astype(np.float32)
    fir = (np.random.randn(512) * 1e-2).astype(np.float32)
    out_np = apply_post_filter(audio, fir)
    out_t = apply_post_filter(torch.from_numpy(audio), fir).numpy()
    np.testing.assert_allclose(out_np, out_t, atol=1e-4)


# --------------------------------------------------------------------------- #
# linear-phase (symmetric FIR → 対称遅延、左右非対称なし)
# --------------------------------------------------------------------------- #
def test_apply_linear_phase_symmetric() -> None:
    # 中央 tap 対称な FIR で impulse 応答が impulse 位置を中心に対称 (遅延 0)。
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0
    fir[250] = fir[262] = 0.3  # 256 対称
    imp = np.zeros(2000, dtype=np.float32)
    imp[1000] = 1.0
    out = apply_post_filter(imp, fir)
    nz = np.nonzero(np.abs(out) > 1e-6)[0]
    center = (nz.min() + nz.max()) / 2
    assert center == pytest.approx(1000.0, abs=0.5)  # 左右非対称遅延なし


# --------------------------------------------------------------------------- #
# load_post_filter
# --------------------------------------------------------------------------- #
def test_load_post_filter_roundtrip(tmp_path) -> None:
    fir = _delta_fir()
    p = tmp_path / "fir.npy"
    np.save(p, fir)
    loaded = load_post_filter(p)
    np.testing.assert_array_equal(loaded, fir)


def test_load_post_filter_validates_length(tmp_path) -> None:
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros(256, dtype=np.float32))
    with pytest.raises(ValueError, match="length 512"):
        load_post_filter(p)


def test_load_post_filter_validates_ndim(tmp_path) -> None:
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((2, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="1-D"):
        load_post_filter(p)


# --------------------------------------------------------------------------- #
# reverse_sample との統合 (post_filter 引数 switch)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def diff_model() -> DiffWaveNext2:
    torch.manual_seed(0)
    return DiffWaveNext2(
        sub_model_cfg={"mel_channels": 128, "n_fft": 1024, "hop_length": 256, "win_length": 1024}
    )


def test_reverse_sample_post_filter_identity(diff_model: DiffWaveNext2) -> None:
    # delta FIR (identity) を post_filter に渡すと、無指定とほぼ同じ波形になる。
    mel = torch.randn(1, 128, 16)
    plain = reverse_sample(diff_model, mel, seed=7)
    filtered = reverse_sample(diff_model, mel, seed=7, post_filter=_delta_fir())
    assert filtered.shape == plain.shape
    torch.testing.assert_close(filtered, plain, atol=1e-4, rtol=1e-4)


def test_reverse_sample_post_filter_none_is_plain(diff_model: DiffWaveNext2) -> None:
    mel = torch.randn(1, 128, 16)
    a = reverse_sample(diff_model, mel, seed=3, post_filter=None)
    b = reverse_sample(diff_model, mel, seed=3)
    torch.testing.assert_close(a, b)


def test_importable() -> None:
    from wavenext2.inference import apply_post_filter as imported

    assert imported is apply_post_filter
