"""InverseLR scheduler の数式 pin テスト (T-M2.5)."""

from __future__ import annotations

import math

import pytest
import torch

from wavenext2.utils import InverseLR


def _opt(lr: float = 1e-4):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


# --- 数式 pin (step=0 / inv_gamma / 10*inv_gamma) ------------------------------
def test_lr_at_step_zero():
    # lr = base * (1 - warmup^1) * (1+0)^-0.5 = base * 0.001
    sch = InverseLR(_opt(1e-4), inv_gamma=200000, power=0.5, warmup=0.999)
    assert math.isclose(sch.get_last_lr()[0], 1e-4 * 0.001, rel_tol=1e-6)


def test_lr_at_step_inv_gamma():
    sch = InverseLR(_opt(1e-4), inv_gamma=200000, power=0.5, warmup=0.999)
    sch.last_epoch = 200000
    # (1 - 0.999^200001) ≈ 1、(1+1)^-0.5 = 0.7071
    assert math.isclose(sch.get_lr()[0], 1e-4 * (2.0**-0.5), rel_tol=1e-4)


def test_lr_at_step_10x_inv_gamma():
    sch = InverseLR(_opt(1e-4), inv_gamma=200000, power=0.5, warmup=0.999)
    sch.last_epoch = 2_000_000
    assert math.isclose(sch.get_lr()[0], 1e-4 * (11.0**-0.5), rel_tol=1e-4)


def test_lr_monotonic_after_warmup():
    sch = InverseLR(_opt(1e-4), inv_gamma=200000, power=0.5, warmup=0.999)
    prev = None
    for step in (200000, 400000, 800000, 2_000_000):
        sch.last_epoch = step
        lr = sch.get_lr()[0]
        if prev is not None:
            assert lr < prev  # warmup 完了域では単調減少
        prev = lr


def test_get_last_lr_consistency():
    opt = _opt(1e-4)
    sch = InverseLR(opt, inv_gamma=200000, power=0.5, warmup=0.999)
    opt.step()
    sch.step()
    assert math.isclose(sch.get_last_lr()[0], opt.param_groups[0]["lr"], rel_tol=1e-9)


def test_state_dict_roundtrip():
    opt = _opt(1e-4)
    sch = InverseLR(opt)
    for _ in range(5):
        opt.step()  # optimizer.step() を先に呼び scheduler 警告を回避
        sch.step()
    sd = sch.state_dict()
    sch2 = InverseLR(_opt(1e-4))
    sch2.load_state_dict(sd)
    assert sch2.last_epoch == sch.last_epoch
    assert sch2.base_lrs == sch.base_lrs
    assert math.isclose(sch2.get_last_lr()[0], sch.get_last_lr()[0], rel_tol=1e-9)


def test_two_optimizers_ratio_2_to_1():
    sch_g = InverseLR(_opt(1e-4))
    sch_d = InverseLR(_opt(2e-4))
    for step in (0, 1000, 200000, 2_000_000):
        sch_g.last_epoch = step
        sch_d.last_epoch = step
        assert math.isclose(sch_d.get_lr()[0] / sch_g.get_lr()[0], 2.0, rel_tol=1e-9)


# --- validation ----------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_warmup_raises(bad):
    with pytest.raises(ValueError, match="warmup"):
        InverseLR(_opt(), warmup=bad)


def test_invalid_inv_gamma_raises():
    with pytest.raises(ValueError, match="inv_gamma"):
        InverseLR(_opt(), inv_gamma=0)
