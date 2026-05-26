"""SubModelGAN / SubModelDiff の unit テスト (T-M1.6)."""

from __future__ import annotations

import pytest
import torch

from wavenext2.models import CONCAT_ORDER, SubModelDiff, SubModelGAN

# 厳密パラメータ数 (T-M1.4 の generator + Diff は NoiseEmbedding)。Table 1 比較はコメント。
GAN_PARAMS = 15_427_674  # = generator (Table 1 14.99M +2.9%)
DIFF_PARAMS = 16_126_978 + 328_704  # generator + NoiseEmbedding = 16,455,682


@pytest.fixture(scope="module")
def gan():
    return SubModelGAN()


@pytest.fixture(scope="module")
def diff():
    return SubModelDiff()


# --- GAN -----------------------------------------------------------------------
def test_gan_forward_shape(gan):
    out = gan(torch.randn(2, 128, 80), torch.randn(2, 24000))  # 80*300
    assert out.shape == (2, 24000)


def test_gan_output_clipped(gan):
    # n_t は generator の clip[-1,1] が効く
    out = gan(torch.randn(2, 128, 80), torch.randn(2, 24000) * 50)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_gan_param_count(gan):
    n = sum(p.numel() for p in gan.parameters())
    assert n == GAN_PARAMS
    assert abs(n - 14.99e6) / 14.99e6 < 0.05  # Table 1 +2.9%


def test_gan_input_channels(gan):
    assert gan.generator.input_channels == 128 + 2048  # 2176


def test_gan_backward(gan):
    gan.zero_grad()
    gan(torch.randn(2, 128, 40), torch.randn(2, 12000)).sum().backward()
    for name, p in gan.named_parameters():
        assert p.grad is not None, name


# --- Diff ----------------------------------------------------------------------
def test_diff_forward_shape(diff):
    out = diff(torch.randn(2, 128, 94), torch.randn(2, 24064), torch.rand(2))  # 94*256
    assert out.shape == (2, 24064)


def test_diff_output_not_clipped(diff):
    # ε 予測は clip しない (final_activation="none")。大入力で |out|>1 を許容
    out = diff(torch.randn(2, 128, 40), torch.randn(2, 10240) * 1e4, torch.rand(2))
    assert torch.isfinite(out).all()
    assert out.abs().max() > 1.0


def test_diff_param_count(diff):
    # generator (per-block fc_t 含む 16.13M) + NoiseEmbedding 0.33M = 16.46M。
    # Table 1 Diff 14.42M を +14% 超過 (per-block fc_t 起因)。fc_t 要否は M1 phase review。
    assert sum(p.numel() for p in diff.parameters()) == DIFF_PARAMS


def test_diff_input_channels(diff):
    assert diff.generator.input_channels == 128 + 1024  # 1152


def test_diff_cond_propagation(diff):
    # 異なる noise level c で出力が変わること (conditioning が伝播)。
    # 未訓練時は LayerScale γ=1e-6 で block が near-identity のため effect は減衰するが、
    # additive bias は residual に乗るので出力は確実に変化する。
    diff.eval()
    mel, x_t = torch.randn(2, 128, 40), torch.randn(2, 10240)
    with torch.no_grad():
        o0 = diff(mel, x_t, torch.zeros(2))
        o1 = diff(mel, x_t, torch.ones(2))
    assert (o0 - o1).abs().max() > 1e-3


def test_diff_c_shape_check(diff):
    with pytest.raises(ValueError, match=r"c must be shape"):
        diff(torch.randn(2, 128, 40), torch.randn(2, 10240), torch.rand(2, 1))


def test_diff_backward(diff):
    diff.zero_grad()
    diff(torch.randn(2, 128, 40), torch.randn(2, 10240), torch.rand(2)).sum().backward()
    for name, p in diff.named_parameters():
        assert p.grad is not None, name


# --- 共通 ----------------------------------------------------------------------
def test_concat_order():
    assert CONCAT_ORDER == ("mel", "stft_spec")
    assert SubModelGAN.CONCAT_ORDER == ("mel", "stft_spec")
    assert SubModelDiff.CONCAT_ORDER == ("mel", "stft_spec")


def test_from_config_ignores_extra_keys():
    # config drift 耐性: 余分な key を無視する
    cfg = {"n_fft": 2048, "hop_length": 300, "win_length": 1200, "unknown_key": 999}
    m = SubModelGAN.from_config(cfg)
    assert m.generator.input_channels == 128 + 2048


def test_from_config_diff():
    m = SubModelDiff.from_config({"n_fft": 1024, "hop_length": 256, "win_length": 1024})
    assert m.generator.input_channels == 128 + 1024


def test_modulelist_stackable():
    # T-M2.4 / T-M3.1 で ModuleList に積めることを確認
    gan_stack = torch.nn.ModuleList([SubModelGAN() for _ in range(2)])
    diff_stack = torch.nn.ModuleList([SubModelDiff() for _ in range(2)])
    assert len(gan_stack) == 2 and len(diff_stack) == 2
