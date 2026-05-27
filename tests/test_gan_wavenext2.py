"""GANWaveNext2 (T sub-model 直列 + fixed-point iteration) の unit テスト (T-M2.4)."""

from __future__ import annotations

import pytest
import torch

from wavenext2.models import GANWaveNext2
from wavenext2.models.gan_wavenext2 import GANWaveNext2 as DirectImport

# SubModelGAN 1 個の厳密パラメータ数 (test_sub_model.py と一致)。Table 1 14.99M に対し +2.9%。
SINGLE = 15_427_674
HOP = 300


@pytest.fixture(scope="module")
def gan4():
    return GANWaveNext2(T=4)


def _mel(t_mel: int = 20, b: int = 2, device="cpu") -> torch.Tensor:
    return torch.randn(b, 128, t_mel, device=device)


# --- import / construction -----------------------------------------------------
def test_import():
    assert DirectImport is GANWaveNext2


def test_hop_length_cached(gan4):
    assert gan4.hop_length == HOP


def test_module_list_independence():
    # sub_models は別オブジェクトで weight 非共有
    m = GANWaveNext2(T=2)
    assert m.sub_models[0] is not m.sub_models[1]
    with torch.no_grad():
        m.sub_models[0].generator.linear_1.weight.add_(1.0)
    w0 = m.sub_models[0].generator.linear_1.weight
    w1 = m.sub_models[1].generator.linear_1.weight
    assert not torch.allclose(w0, w1)


# --- parameter count -----------------------------------------------------------
def test_param_count_linear_and_table1():
    table1 = {1: 14.99e6, 2: 29.97e6, 3: 44.96e6, 4: 59.94e6, 5: 74.93e6}
    for T, ref in table1.items():
        n = sum(p.numel() for p in GANWaveNext2(T=T).parameters())
        assert n == T * SINGLE, f"T={T}: {n} != {T}*{SINGLE}"  # 厳密 T 線形 (重み非共有)
        assert abs(n - ref) / ref < 0.05, f"T={T}: {n} vs Table1 {ref} (>5%)"  # Table 1 ±5%


# --- forward / backward --------------------------------------------------------
def test_forward_shape(gan4):
    out = gan4(_mel(t_mel=80))  # 80 * 300 = 24000
    assert out.shape == (2, 24000)


def test_forward_finite(gan4):
    # 未訓練 T=4 は残差累積で [-1,1] を超えうる (収束時に範囲内) → finite のみ要求
    out = gan4(_mel())
    assert torch.isfinite(out).all()


def test_forward_auto_infer_equals_explicit(gan4):
    gan4.eval()
    mel = _mel(t_mel=30)
    with torch.no_grad():
        a = gan4(mel)
        b = gan4(mel, audio_length=30 * HOP)
    assert torch.allclose(a, b)


def test_backward_all_4_submodels_grad():
    model = GANWaveNext2(T=4)
    model(_mel()).sum().backward()
    for i, sub in enumerate(model.sub_models):
        for name, p in sub.named_parameters():
            assert p.grad is not None, f"sub_models[{i}].{name} grad is None"


def test_deterministic(gan4):
    gan4.eval()
    mel = _mel(t_mel=24)
    with torch.no_grad():
        assert torch.allclose(gan4(mel), gan4(mel))  # dropout なし + y_T=zeros 固定


# --- T=1 degenerate case -------------------------------------------------------
def test_T1_equals_neg_submodel():
    # パターン (A): y_0 = zeros - n_1 = -sub_model(mel, zeros)
    model = GANWaveNext2(T=1).eval()
    mel = _mel(t_mel=20)
    with torch.no_grad():
        y0 = model(mel)
        n1 = model.sub_models[0](mel, torch.zeros(2, 20 * HOP))
    assert torch.allclose(y0, -n1, atol=1e-6)


def test_T1_output_in_range():
    # T=1 は y_0 = -n_1、n_1 は sub-model 内 clip[-1,1] のため範囲内
    model = GANWaveNext2(T=1).eval()
    with torch.no_grad():
        y0 = model(_mel(t_mel=20))
    assert y0.min() >= -1.0 and y0.max() <= 1.0


# --- residual update semantics -------------------------------------------------
def test_residual_update_semantics():
    y = torch.randn(2, 100)
    n = torch.randn(2, 100)
    assert torch.equal(GANWaveNext2._residual_update(y, n), y - n)


# --- return_intermediates ------------------------------------------------------
def test_return_intermediates_structure(gan4):
    gan4.eval()
    mel = _mel(t_mel=20)
    with torch.no_grad():
        inter = gan4(mel, return_intermediates=True)
    assert isinstance(inter, list) and len(inter) == gan4.T + 1  # [y_T, ..., y_0]
    for y in inter:
        assert y.shape == (2, 20 * HOP)
    assert torch.equal(inter[0], torch.zeros(2, 20 * HOP))  # y_T = zeros
    with torch.no_grad():
        y0 = gan4(mel)
    assert torch.allclose(inter[-1], y0)  # 最終要素 = forward(False) の出力


def test_return_intermediates_grad_flow():
    model = GANWaveNext2(T=3)
    inter = model(_mel(t_mel=16), return_intermediates=True)
    inter[-1].sum().backward()  # y_0 から全 sub-model に grad
    for sub in model.sub_models:
        assert all(p.grad is not None for p in sub.parameters())


# --- enable_grad_ckpt ----------------------------------------------------------
def test_enable_grad_ckpt_flag():
    assert GANWaveNext2(T=2, enable_grad_ckpt=True).enable_grad_ckpt is True


def test_enable_grad_ckpt_output_equivalence():
    # grad checkpointing は数値的 noop: 同じ重み + 入力で出力一致
    torch.manual_seed(0)
    model = GANWaveNext2(T=2, enable_grad_ckpt=True)
    model.train()
    mel = torch.randn(2, 128, 16, requires_grad=True)  # checkpoint の grad 入力警告を回避
    out_ckpt = model(mel)
    model.enable_grad_ckpt = False
    out_plain = model(mel)
    assert torch.allclose(out_ckpt, out_plain, atol=1e-5)


# --- shape / arg validation ----------------------------------------------------
def test_audio_length_mismatch_raises(gan4):
    with pytest.raises(ValueError, match="must equal T_mel"):
        gan4(_mel(t_mel=20), audio_length=20 * HOP + 1)


def test_invalid_mel_dim_raises(gan4):
    with pytest.raises(ValueError, match=r"must be \(B, 128, T_mel\)"):
        gan4(torch.randn(2, 128))


@pytest.mark.parametrize("bad_T", [0, -1])
def test_invalid_T_raises(bad_T):
    with pytest.raises(ValueError, match="T must be >= 1"):
        GANWaveNext2(T=bad_T)


# --- from_config ---------------------------------------------------------------
def test_from_config():
    m = GANWaveNext2.from_config(
        {
            "T": 2,
            "sub_model_cfg": {"n_fft": 2048, "hop_length": 300, "win_length": 1200},
            "enable_grad_ckpt": True,
        }
    )
    assert m.T == 2 and m.enable_grad_ckpt is True and len(m.sub_models) == 2
    assert m.hop_length == 300


def test_from_config_gan_values():
    # configs/gan_wavenext2.yaml の sub_model 値 (n_fft=2048, hop=300, win=1200) で生成・forward
    m = GANWaveNext2.from_config(
        {"T": 2, "sub_model": {"n_fft": 2048, "hop_length": 300, "win_length": 1200}}
    )
    assert m.sub_models[0].generator.input_channels == 128 + 2048
    out = m(_mel(t_mel=20))
    assert out.shape == (2, 20 * 300)


def test_from_config_ignores_extra_keys():
    m = GANWaveNext2.from_config({"T": 1, "unknown": 123})
    assert m.T == 1
