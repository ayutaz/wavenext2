"""Loss 関数 (Hinge GAN / FM / MR-STFT) + compute_total_loss の unit テスト (T-M2.3)."""

from __future__ import annotations

import pytest
import torch

from wavenext2.losses import (
    DEFAULT_WEIGHTS,
    FeatureMatchingLoss,
    HingeGANLoss,
    MultiResolutionSTFTLoss,
    compute_total_loss,
)


# --- HingeGANLoss -------------------------------------------------------------
def test_hinge_d_loss_at_zero():
    # D(real)=D(fake)=0 → relu(1-0)+relu(1+0)=2 (sub-D 平均で 2)
    loss = HingeGANLoss()
    z = [torch.zeros(2, 1, 10) for _ in range(3)]
    assert abs(float(loss.d_loss(z, z)) - 2.0) < 1e-5


def test_hinge_d_loss_grad_direction():
    # D(real) を増やすと d_loss が減る → ∂loss/∂d_real < 0
    loss = HingeGANLoss()
    d_real = [torch.zeros(2, 1, 10, requires_grad=True) for _ in range(3)]
    d_fake = [torch.zeros(2, 1, 10) for _ in range(3)]
    loss.d_loss(d_real, d_fake).backward()
    assert d_real[0].grad.mean() < 0


def test_hinge_g_loss():
    loss = HingeGANLoss()
    d_fake = [torch.full((2, 1, 10), 0.5) for _ in range(3)]
    assert abs(float(loss.g_loss(d_fake)) - (-0.5)) < 1e-5  # -mean(0.5)


# --- FeatureMatchingLoss ------------------------------------------------------
def test_fm_identical_is_zero():
    fm = FeatureMatchingLoss()
    feats = [[torch.randn(2, 16, 50), torch.randn(2, 64, 25)] for _ in range(3)]
    assert float(fm(feats, feats)) == 0.0


def test_fm_different_positive():
    fm = FeatureMatchingLoss()
    fr = [[torch.zeros(2, 16, 50)] for _ in range(3)]
    ff = [[torch.ones(2, 16, 50)] for _ in range(3)]
    assert abs(float(fm(fr, ff)) - 1.0) < 1e-6  # |0-1|=1


def test_fm_detaches_real_side():
    # real は target → 勾配は fake 側のみ
    fm = FeatureMatchingLoss()
    fr = [[torch.randn(2, 16, 50, requires_grad=True)]]
    ff = [[torch.randn(2, 16, 50, requires_grad=True)]]
    fm(fr, ff).backward()
    assert fr[0][0].grad is None
    assert ff[0][0].grad is not None


# --- MultiResolutionSTFTLoss --------------------------------------------------
def test_mrstft_identical_near_zero():
    mr = MultiResolutionSTFTLoss()
    y = torch.randn(2, 16384)
    sc, mag = mr(y, y)
    assert float(sc) < 1e-4 and float(mag) < 1e-4


def test_mrstft_different_positive():
    mr = MultiResolutionSTFTLoss()
    sc, mag = mr(torch.randn(2, 16384), torch.randn(2, 16384))
    assert float(sc) > 0 and float(mag) > 0


def test_mrstft_gradient_flow():
    mr = MultiResolutionSTFTLoss()
    y_pred = torch.randn(2, 16384, requires_grad=True)
    sc, mag = mr(y_pred, torch.randn(2, 16384))
    (sc + mag).backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


def test_mrstft_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        MultiResolutionSTFTLoss(n_ffts=(512, 1024), win_lengths=(360,), hop_sizes=(80, 150))


# --- compute_total_loss -------------------------------------------------------
def test_compute_total_loss_weighted_sum():
    losses = {
        "d_gan": torch.tensor(1.0),
        "d_fm": torch.tensor(2.0),
        "mrstft_sc": torch.tensor(3.0),
        "mrstft_mag": torch.tensor(4.0),
    }
    total, unw = compute_total_loss(losses)
    # 1*1 + 10*2 + 2.5*3 + 2.5*4 = 1 + 20 + 7.5 + 10 = 38.5
    assert abs(float(total) - 38.5) < 1e-5
    assert unw == {"d_gan": 1.0, "d_fm": 2.0, "mrstft_sc": 3.0, "mrstft_mag": 4.0}
    assert all(isinstance(v, float) for v in unw.values())


def test_compute_total_loss_backward():
    losses = {k: torch.tensor(1.0, requires_grad=True) for k in DEFAULT_WEIGHTS}
    total, _ = compute_total_loss(losses)
    total.backward()
    assert all(losses[k].grad is not None for k in DEFAULT_WEIGHTS)


def test_compute_total_loss_missing_weight():
    with pytest.raises(KeyError, match="no weight"):
        compute_total_loss({"unknown": torch.tensor(1.0)})


def test_compute_total_loss_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_total_loss({})


def test_compute_total_loss_custom_weights():
    total, _ = compute_total_loss({"a": torch.tensor(2.0)}, weights={"a": 3.0})
    assert abs(float(total) - 6.0) < 1e-5
