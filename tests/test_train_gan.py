"""GAN 訓練ループ (train_gan_step / checkpoint / config) の unit テスト (T-M2.5)."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW

from wavenext2.models.discriminator import MultiScaleDiscriminator
from wavenext2.models.gan_wavenext2 import GANWaveNext2
from wavenext2.train.train_gan import (
    TrainState,
    _Criteria,
    load_checkpoint,
    save_checkpoint,
    train_gan_step,
)
from wavenext2.utils import InverseLR, load_config

CFG = {
    "train": {"grad_clip_norm": 1.0},
    "loss": {"weights": {"d_gan": 1.0, "d_fm": 10.0, "mrstft_sc": 2.5, "mrstft_mag": 2.5}},
}
HOP = 300
T_MEL = 16
SEG = T_MEL * HOP  # 4800


def _build():
    G = GANWaveNext2(T=2)
    D = MultiScaleDiscriminator()
    opt_G = AdamW(G.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-3)
    opt_D = AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-3)
    return G, D, opt_G, opt_D, InverseLR(opt_G), InverseLR(opt_D)


def _batch():
    return torch.randn(2, 128, T_MEL), torch.randn(2, SEG)


_STEP_KEYS = {
    "loss_G",
    "loss_D",
    "loss_g_gan",
    "loss_g_fm",
    "loss_g_mrstft_sc",
    "loss_g_mrstft_mag",
    "loss_d_real",
    "loss_d_fake",
    "grad_norm_G",
    "grad_norm_D",
}


# --- train_gan_step ------------------------------------------------------------
def test_train_gan_step_keys_and_finite():
    G, D, oG, oD, sG, sD = _build()
    mel, audio = _batch()
    logs = train_gan_step(G, D, oG, oD, sG, sD, mel, audio, CFG, _Criteria())
    assert set(logs) == _STEP_KEYS
    assert all(math.isfinite(v) for v in logs.values())


def test_train_gan_step_updates_both():
    G, D, oG, oD, sG, sD = _build()
    g_before = G.sub_models[0].generator.linear_1.weight.detach().clone()
    d_before = D.sub_discriminators[0].layers[-1].weight.detach().clone()
    mel, audio = _batch()
    train_gan_step(G, D, oG, oD, sG, sD, mel, audio, CFG, _Criteria())
    assert not torch.equal(g_before, G.sub_models[0].generator.linear_1.weight)
    assert not torch.equal(d_before, D.sub_discriminators[0].layers[-1].weight)


def test_grad_norms_nonnegative():
    G, D, oG, oD, sG, sD = _build()
    mel, audio = _batch()
    logs = train_gan_step(G, D, oG, oD, sG, sD, mel, audio, CFG, _Criteria())
    assert logs["grad_norm_G"] >= 0 and logs["grad_norm_D"] >= 0


def test_lr_ratio_2_to_1_after_step():
    G, D, oG, oD, sG, sD = _build()
    mel, audio = _batch()
    train_gan_step(G, D, oG, oD, sG, sD, mel, audio, CFG, _Criteria())
    assert math.isclose(sD.get_last_lr()[0] / sG.get_last_lr()[0], 2.0, rel_tol=1e-6)


def test_no_ema_buffers():
    G = GANWaveNext2(T=2)
    assert not any("ema" in n.lower() for n, _ in G.named_buffers())
    assert not any("ema" in n.lower() for n, _ in G.named_parameters())


# --- TrainState ----------------------------------------------------------------
def test_train_state_d_loss_counter():
    s = TrainState()
    for _ in range(5):
        s.update_d_loss_history(0.001)  # < 0.01
    assert s.d_loss_below_threshold_steps == 5
    s.update_d_loss_history(0.5)  # reset
    assert s.d_loss_below_threshold_steps == 0


# --- checkpoint ----------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    G, D, oG, oD, sG, sD = _build()
    mel, audio = _batch()
    train_gan_step(G, D, oG, oD, sG, sD, mel, audio, CFG, _Criteria())
    state = TrainState(step=7, best_val_mrstft=1.23)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, G, D, oG, oD, sG, sD, state, CFG)

    G2, D2, oG2, oD2, sG2, sD2 = _build()
    restored = load_checkpoint(path, G2, D2, oG2, oD2, sG2, sD2)
    assert restored.step == 7
    assert math.isclose(restored.best_val_mrstft, 1.23)
    for (k1, v1), (k2, v2) in zip(G.state_dict().items(), G2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_atomic_best_pt(tmp_path):
    G, D, oG, oD, sG, sD = _build()
    path = tmp_path / "best.pt"
    save_checkpoint(path, G, D, oG, oD, sG, sD, TrainState(step=1), CFG, atomic=True)
    assert path.exists()
    assert not (tmp_path / "best.pt.tmp").exists()  # tmp は os.replace 後に消える
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 1


# --- config --------------------------------------------------------------------
def test_load_config_gan_yaml():
    cfg = load_config("configs/gan_wavenext2.yaml")
    assert cfg["model"]["T"] == 4
    for key in ("validation", "checkpoint", "logging"):
        assert key in cfg, f"top-level {key} missing"
    assert cfg["train"]["optimizer"]["lr_g"] == 1.0e-4
    assert cfg["train"]["optimizer"]["lr_d"] == 2.0e-4
