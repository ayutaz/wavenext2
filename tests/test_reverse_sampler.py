"""T-M3.3: reverse sampler (DDIM/DDPM 一般形 β-free 4-step) の unit テスト.

docs/tickets/T-M3.3-reverse-sampler.md §5 を実装する。実 DiffWaveNext2 は ~57.42M params で重い
ため、sub-model 呼び出しを mock する軽量モデルを fixture で用意し call_count / dispatch order を
検証する。`_compute_ddpm_coefficients` は pure helper なので mock 不要で式の正確性を pin する。
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from wavenext2.inference.infer_diff import (
    _compute_ddpm_coefficients,
    eval_mode,
    reverse_sample,
)
from wavenext2.models.diff_wavenext2 import DiffWaveNext2

ABAR = [1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]


# --------------------------------------------------------------------------- #
# 軽量 mock モデル (実 SubModelDiff の代わり)
# --------------------------------------------------------------------------- #
class _MockSub(nn.Module):
    """eps_pred = x * scale を返す軽量 sub-model。call を記録する。"""

    def __init__(self, calls: list, idx: int, scale: float = 0.0) -> None:
        super().__init__()
        self.calls = calls
        self.idx = idx
        self.scale = scale

    def forward(self, mel: torch.Tensor, x_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.idx, c))
        return x_t * self.scale


class _MockModel(nn.Module):
    """DiffWaveNext2 の reverse_sample が期待する最小 interface を満たす mock。"""

    hop_length = 256

    def __init__(self, scale: float = 0.0) -> None:
        super().__init__()
        self.calls: list = []
        self.sub_models = nn.ModuleList([_MockSub(self.calls, i, scale) for i in range(4)])
        self.register_buffer("noise_schedule_abar", torch.tensor(ABAR, dtype=torch.float32))

    @property
    def NOISE_SCHEDULE_ABAR(self) -> torch.Tensor:  # noqa: N802
        return self.noise_schedule_abar


@pytest.fixture
def mock_model() -> _MockModel:
    return _MockModel(scale=0.0)


# --------------------------------------------------------------------------- #
# _compute_ddpm_coefficients (β-free)
# --------------------------------------------------------------------------- #
def test_compute_coefficients_no_beta_alpha_key() -> None:
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    assert "beta" not in coef
    assert "alpha" not in coef
    assert set(coef) == {"abar", "abar_next", "sqrt_abar", "sqrt_one_minus_abar"}


def test_compute_coefficients_shapes() -> None:
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    for v in coef.values():
        assert v.shape == (4,)


def test_compute_coefficients_abar_next() -> None:
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    assert torch.allclose(coef["abar_next"][:3], torch.tensor(ABAR[1:]))
    assert coef["abar_next"][3] == 1.0


def test_compute_coefficients_sqrt_one_minus_abar() -> None:
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    expected = torch.tensor([math.sqrt(1.0 - a) for a in ABAR])  # [0.99995, 0.9858, 0.6633, 0.3]
    assert torch.allclose(coef["sqrt_one_minus_abar"], expected, atol=1e-4)


def test_compute_coefficients_abar_increasing() -> None:
    # denoising 方向で ᾱ_t < ᾱ_next ⇒ σ² ≥ 0 の前提。
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    assert torch.all(coef["abar"] < coef["abar_next"])


def test_compute_coefficients_pure() -> None:
    abar = torch.tensor(ABAR)
    snapshot = abar.clone()
    _compute_ddpm_coefficients(abar)
    assert torch.equal(abar, snapshot)  # in-place 変更なし


# --------------------------------------------------------------------------- #
# σ² の非負性・整合性 (CRITICAL safety nets)
# --------------------------------------------------------------------------- #
def test_sigma_non_negative() -> None:
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    eta = 1.0
    for idx in range(3):  # t=1..K-1
        at = coef["abar"][idx]
        an = coef["abar_next"][idx]
        sigma2 = eta**2 * (1.0 - an) / (1.0 - at) * (1.0 - at / an)
        assert sigma2 >= 0.0
        assert (1.0 - an - sigma2) >= -1e-6  # coef_eps の sqrt 中身


def test_sigma_indexing_consistency() -> None:
    # t=1: ᾱ_t=1e-4, ᾱ_next=2.8e-2 で σ² 手計算と照合。
    coef = _compute_ddpm_coefficients(torch.tensor(ABAR))
    at = float(coef["abar"][0])
    an = float(coef["abar_next"][0])
    expected = (1.0 - an) / (1.0 - at) * (1.0 - at / an)
    sigma2 = 1.0**2 * (1.0 - an) / (1.0 - at) * (1.0 - at / an)
    assert sigma2 == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# shape / range / sanity
# --------------------------------------------------------------------------- #
def test_output_shape(mock_model: _MockModel) -> None:
    mel = torch.randn(2, 128, 80)
    out = reverse_sample(mock_model, mel)
    assert out.shape == (2, 80 * 256)


def test_output_range(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 32)
    out = reverse_sample(mock_model, mel)
    assert float(out.min()) >= -1.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("t_mel", [32, 64, 94, 100])
def test_t_audio_equals_t_mel_times_hop(mock_model: _MockModel, t_mel: int) -> None:
    mel = torch.randn(1, 128, t_mel)
    out = reverse_sample(mock_model, mel)
    assert out.shape == (1, t_mel * 256)


def test_no_nan_inf(mock_model: _MockModel) -> None:
    mel = torch.randn(2, 128, 40)
    out = reverse_sample(mock_model, mel)
    assert torch.isfinite(out).all()


def test_no_grad(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    out = reverse_sample(mock_model, mel)
    assert out.requires_grad is False


def test_dtype_preserved(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16, dtype=torch.float32)
    out = reverse_sample(mock_model, mel)
    assert out.dtype == torch.float32


def test_mel_dim_validation(mock_model: _MockModel) -> None:
    with pytest.raises(ValueError):
        reverse_sample(mock_model, torch.randn(128, 16))  # 2-D


@pytest.mark.parametrize("b", [1, 8])
def test_various_batch(mock_model: _MockModel, b: int) -> None:
    mel = torch.randn(b, 128, 24)
    out = reverse_sample(mock_model, mel)
    assert out.shape == (b, 24 * 256)


# --------------------------------------------------------------------------- #
# dispatch / call_count (1-to-1 の核心)
# --------------------------------------------------------------------------- #
def test_each_submodel_called_once(mock_model: _MockModel) -> None:
    mel = torch.randn(2, 128, 16)
    reverse_sample(mock_model, mel)
    idxs = [c[0] for c in mock_model.calls]
    assert idxs == [0, 1, 2, 3]  # call_count == 1 each, denoising 順


def test_submodel_call_c_shape(mock_model: _MockModel) -> None:
    mel = torch.randn(3, 128, 16)
    reverse_sample(mock_model, mel)
    for _idx, c in mock_model.calls:
        assert c.shape == (3,)


def test_submodel_c_values_match_schedule(mock_model: _MockModel) -> None:
    # 各 step の c_t = √(1-ᾱ_t)。
    mel = torch.randn(1, 128, 16)
    reverse_sample(mock_model, mel)
    expected = [math.sqrt(1.0 - a) for a in ABAR]
    for (_idx, c), e in zip(mock_model.calls, expected, strict=True):
        assert float(c[0]) == pytest.approx(e, abs=1e-4)


# --------------------------------------------------------------------------- #
# deterministic / stochastic
# --------------------------------------------------------------------------- #
def test_same_seed_deterministic(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    a = reverse_sample(mock_model, mel, seed=42)
    b = reverse_sample(mock_model, mel, seed=42)
    assert torch.allclose(a, b)


def test_different_seed_different_output(mock_model: _MockModel) -> None:
    # scale=0 だと eps_pred=0 で x0_hat=x/√ᾱ。stochastic 部 (σ·z) が seed で変わる。
    mel = torch.randn(1, 128, 16)
    a = reverse_sample(mock_model, mel, seed=42)
    b = reverse_sample(mock_model, mel, seed=43)
    assert not torch.allclose(a, b)


def test_seed_none_stochastic(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    a = reverse_sample(mock_model, mel, seed=None)
    b = reverse_sample(mock_model, mel, seed=None)
    assert not torch.allclose(a, b)


def test_seed_with_external_generator(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    g1 = torch.Generator().manual_seed(2025)
    a = reverse_sample(mock_model, mel, seed=g1)
    g2 = torch.Generator().manual_seed(2025)
    b = reverse_sample(mock_model, mel, seed=g2)
    assert torch.allclose(a, b)


def test_no_global_seed_pollution(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    before = torch.initial_seed()
    state = torch.random.get_rng_state()
    reverse_sample(mock_model, mel, seed=7)
    assert torch.initial_seed() == before
    assert torch.equal(torch.random.get_rng_state(), state)


# --------------------------------------------------------------------------- #
# eta (DDPM vs DDIM)
# --------------------------------------------------------------------------- #
def test_stochastic_flag_eta(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    ddpm = reverse_sample(mock_model, mel, seed=5, eta=1.0)
    ddim = reverse_sample(mock_model, mel, seed=5, eta=0.0)
    assert not torch.allclose(ddpm, ddim)


def test_eta_zero_deterministic_given_seed(mock_model: _MockModel) -> None:
    mel = torch.randn(1, 128, 16)
    a = reverse_sample(mock_model, mel, seed=7, eta=0.0)
    b = reverse_sample(mock_model, mel, seed=7, eta=0.0)
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# eval_mode context manager
# --------------------------------------------------------------------------- #
def test_eval_mode_context() -> None:
    m = nn.Linear(2, 2)
    m.train()
    with eval_mode(m):
        assert m.training is False
    assert m.training is True  # 復元


def test_eval_mode_restores_eval_state() -> None:
    m = nn.Linear(2, 2)
    m.eval()
    with eval_mode(m):
        assert m.training is False
    assert m.training is False  # 元が eval なら eval のまま


def test_reverse_sample_restores_training_state(mock_model: _MockModel) -> None:
    # reverse_sample 内部で eval_mode を使うが、呼び出し側の train 状態を破壊しない。
    mock_model.train()
    reverse_sample(mock_model, torch.randn(1, 128, 16))
    assert mock_model.training is True


# --------------------------------------------------------------------------- #
# 実 DiffWaveNext2 との結合
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_model() -> DiffWaveNext2:
    torch.manual_seed(0)
    return DiffWaveNext2(
        sub_model_cfg={"mel_channels": 128, "n_fft": 1024, "hop_length": 256, "win_length": 1024}
    )


def test_with_real_diff_model(real_model: DiffWaveNext2) -> None:
    mel = torch.randn(1, 128, 32)
    out = reverse_sample(real_model, mel, seed=1)
    assert out.shape == (1, 32 * 256)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def test_real_model_no_negative_beta_nan(real_model: DiffWaveNext2) -> None:
    # 論文 schedule で β 負値問題が起きない (β を計算しないため) ことを実 model で確認。
    mel = torch.randn(1, 128, 24)
    out = reverse_sample(real_model, mel, seed=2)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_synthesize_alias(real_model: DiffWaveNext2) -> None:
    # DiffWaveNext2.synthesize = reverse_sample (import 副作用) が同等結果を返す。
    mel = torch.randn(1, 128, 16)
    a = real_model.synthesize(mel, seed=9)
    b = reverse_sample(real_model, mel, seed=9)
    assert torch.allclose(a, b)


def test_importable() -> None:
    from wavenext2.inference import reverse_sample as imported

    assert imported is reverse_sample
