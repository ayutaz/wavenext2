"""WaveNextGenerator の unit テスト (T-M1.4)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from wavenext2.models.generator import WaveNextGenerator

# 実装 (embed kernel=1) の厳密パラメータ数。Table 1 との比較はコメント参照。
GAN_PARAMS = 15_427_674  # ≈ Table 1 GAN sub-model 14.99M +2.9% (embed kernel=1, concat 2176)
# per-block fc_t 撤去後 (§C7)。embed(1152→512)+8 block+linear×2+LN。+NoiseEmbedding 0.33M で sub-model 計
DIFF_GEN_PARAMS = 14_025_730


@pytest.fixture(scope="module")
def gan_model():
    return WaveNextGenerator(input_channels=2176, n_fft=2048, hop_length=300, conditioning_dim=None)


@pytest.fixture(scope="module")
def diff_model():
    return WaveNextGenerator(input_channels=1152, n_fft=1024, hop_length=256, conditioning_dim=512)


# --- shape --------------------------------------------------------------------
def test_gan_shape(gan_model):
    out = gan_model(torch.randn(2, 2176, 80))
    assert out.shape == (2, 24000)  # 80 * 300
    assert out.dtype == torch.float32


def test_diff_shape(diff_model):
    out = diff_model(torch.randn(2, 1152, 94), cond=torch.randn(2, 512))
    assert out.shape == (2, 24064)  # 94 * 256


def test_batch_one(gan_model):
    assert gan_model(torch.randn(1, 2176, 40)).shape == (1, 12000)


# --- output range -------------------------------------------------------------
def test_output_range(gan_model):
    out = gan_model(torch.randn(2, 2176, 80) * 100)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_output_range_extreme(gan_model):
    out = gan_model(torch.randn(2, 2176, 40) * 1e6)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0


# --- parameter count ----------------------------------------------------------
def test_param_count_gan(gan_model):
    n = sum(p.numel() for p in gan_model.parameters())
    assert n == GAN_PARAMS
    # Table 1 GAN sub-model 14.99M との比較 (±5% 内)
    assert abs(n - 14.99e6) / 14.99e6 < 0.05


def test_param_count_diff_generator(diff_model):
    # Generator 本体 (per-block fc_t 撤去後)。14.03M。
    # +NoiseEmbedding 0.33M (T-M1.6) で sub-model 計 14.354M vs Table 1 14.42M (−0.46%)。
    assert sum(p.numel() for p in diff_model.parameters()) == DIFF_GEN_PARAMS


# --- weight init --------------------------------------------------------------
def test_linear_bias_init(gan_model):
    assert torch.allclose(gan_model.linear_1.bias, torch.zeros_like(gan_model.linear_1.bias))
    assert gan_model.linear_2.bias is None  # bias=False は確定仕様


def test_weight_init_std(gan_model):
    # trunc_normal_(std=0.02) → 実測 std はおよそ 0.02 (truncation でやや小)
    assert 0.01 < float(gan_model.embed.weight.detach().std()) < 0.03


def test_embed_kernel_is_one(gan_model):
    assert gan_model.embed.kernel_size == (1,)


# --- gradient / determinism ---------------------------------------------------
def test_gradient_flow_gan(gan_model):
    gan_model.zero_grad()
    gan_model(torch.randn(2, 2176, 40)).abs().mean().backward()
    for name, p in gan_model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_gradient_flow_diff(diff_model):
    diff_model.zero_grad()
    diff_model(torch.randn(2, 1152, 40), cond=torch.randn(2, 512)).abs().mean().backward()
    for name, p in diff_model.named_parameters():
        assert p.grad is not None, name


def test_deterministic(gan_model):
    gan_model.eval()
    x = torch.randn(2, 2176, 40)
    with torch.no_grad():
        assert torch.equal(gan_model(x), gan_model(x))


# --- conditioning 整合 --------------------------------------------------------
def test_cond_mismatch_gan_with_cond(gan_model):
    with pytest.raises(ValueError, match="cond must be provided iff"):
        gan_model(torch.randn(2, 2176, 40), cond=torch.randn(2, 512))


def test_cond_mismatch_diff_without_cond(diff_model):
    with pytest.raises(ValueError, match="cond must be provided iff"):
        diff_model(torch.randn(2, 1152, 40))


# --- options ------------------------------------------------------------------
def test_final_activation_tanh():
    g = WaveNextGenerator(
        input_channels=1152,
        n_fft=1024,
        hop_length=256,
        conditioning_dim=None,
        final_activation="tanh",
    )
    out = g(torch.randn(2, 1152, 40) * 100)
    assert out.min() >= -1.0 and out.max() <= 1.0  # tanh も [-1,1]


def test_final_activation_none_unbounded():
    # Diff の ε 予測用: clip しないので |out|>1 を許容する
    g = WaveNextGenerator(
        input_channels=1152,
        n_fft=1024,
        hop_length=256,
        conditioning_dim=None,
        final_activation="none",
    )
    g._init_weights()  # 既定 init は std 小さく出力が小さいので、大入力で範囲を出す
    out = g(torch.randn(2, 1152, 40) * 1e4)
    assert torch.isfinite(out).all()
    assert out.abs().max() > 1.0  # clip されていない


def test_invalid_final_activation():
    with pytest.raises(ValueError, match="final_activation"):
        WaveNextGenerator(input_channels=1152, n_fft=1024, hop_length=256, final_activation="relu")


def test_block_factory_di():
    # 軽量 MockBlock を注入して高速化 / ablation できることを確認
    class MockBlock(nn.Module):
        def __init__(
            self, dim, intermediate_dim, kernel_size, layer_scale_init_value, conditioning_dim
        ):
            super().__init__()
            self.lin = nn.Linear(dim, dim)
            self.conditioning_dim = conditioning_dim

        def forward(self, x, *, cond=None):
            return x  # identity (shape 保存のみ確認)

    g = WaveNextGenerator(
        input_channels=1152,
        n_fft=1024,
        hop_length=256,
        conditioning_dim=None,
        block_factory=MockBlock,
    )
    assert isinstance(g.blocks[0], MockBlock)
    assert g(torch.randn(2, 1152, 40)).shape == (2, 10240)
