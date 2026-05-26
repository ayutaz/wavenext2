"""MultiScaleDiscriminator の unit テスト (T-M2.2)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from wavenext2.models.discriminator import (
    MultiScaleDiscriminator,
    NLayerDiscriminator,
    SubDiscOutput,
)


@pytest.fixture(scope="module")
def msd():
    return MultiScaleDiscriminator()


# --- 構造 ----------------------------------------------------------------------
def test_returns_3_subd(msd):
    out = msd(torch.randn(2, 1, 8192))
    assert len(out) == 3
    assert all(isinstance(o, SubDiscOutput) for o in out)


def test_logits_and_feature_count(msd):
    out = msd(torch.randn(2, 1, 8192))
    for o in out:
        assert o.logits.dim() == 3 and o.logits.shape[:2] == (2, 1)
        assert len(o.features) == 6  # 7 層 - logits 層


def test_defensive_unsqueeze(msd):
    # (B, T) を渡しても (B, 1, T) に補正される
    out = msd(torch.randn(2, 8192))
    assert len(out) == 3


def test_downsampling_progression(msd):
    # 後段 sub-D ほど入力が短くなる → logits の時間長も短くなる
    out = msd(torch.randn(2, 1, 16384))
    t = [o.logits.shape[-1] for o in out]
    assert t[0] > t[1] > t[2]


def test_channel_progression(msd):
    out = msd(torch.randn(2, 1, 8192))
    chans = [f.shape[1] for f in out[0].features]
    assert chans == [16, 64, 256, 1024, 1024, 1024]


# --- weight_norm (API 非依存) -------------------------------------------------
def test_weight_norm_applied(msd):
    # 最初の sub-D の入口 Conv1d が weight について parametrize されている
    sub = msd.sub_discriminators[0]
    conv = sub.layers[0][1]  # Sequential(ReflectionPad1d, weight_norm(Conv1d)) の Conv
    assert isinstance(conv, nn.Conv1d)
    assert parametrize.is_parametrized(conv, "weight")


# --- 数値性質 ------------------------------------------------------------------
def test_no_bounded_activation_hinge_compatible(msd):
    # logits を [-1,1]/[0,1] に潰す Tanh/Sigmoid が無いこと (hinge GAN は unbounded logits 前提)
    assert not any(isinstance(m, (nn.Tanh, nn.Sigmoid)) for m in msd.modules())


def test_gradient_flow(msd):
    msd.zero_grad()
    out = msd(torch.randn(2, 1, 8192))
    loss = sum(o.logits.pow(2).mean() for o in out)
    loss = loss + sum(f.mean() for o in out for f in o.features)
    loss.backward()
    for name, p in msd.named_parameters():
        assert p.grad is not None, name


def test_deterministic(msd):
    msd.eval()
    x = torch.randn(2, 1, 8192)
    with torch.no_grad():
        a = msd(x)
        b = msd(x)
    assert all(torch.equal(p.logits, q.logits) for p, q in zip(a, b))


def test_param_count_positive(msd):
    n = sum(p.numel() for p in msd.parameters())
    assert n > 1_000_000  # 3 sub-D で数 M


# --- config / 単体 sub-D ------------------------------------------------------
def test_from_config():
    msd = MultiScaleDiscriminator.from_config({"num_D": 2, "ndf": 16, "extra": 99})
    assert msd.num_D == 2


def test_single_nlayer():
    out = NLayerDiscriminator()(torch.randn(2, 1, 8192))
    assert isinstance(out, SubDiscOutput)
    assert len(out.features) == 6
