"""T-M3.2: Diff 訓練スクリプト (train_diff_step / build_sub_model_cfg / checkpoint) の unit テスト.

docs/tickets/T-M3.2-train-diff.md §5。実 LibriTTSRDataset は使わず合成 batch で step を検証。
モデルは tiny config で高速化 (default 14.354M は CPU backward が遅い)。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.optim import Adam

from wavenext2.models.diff_wavenext2 import DiffWaveNext2
from wavenext2.train.train_diff import (
    TrainStateDiff,
    build_sub_model_cfg,
    load_checkpoint,
    run_validation_diff,
    save_checkpoint,
    train_diff_step,
)

TINY = {
    "mel_channels": 32,
    "n_fft": 128,
    "hop_length": 64,
    "win_length": 128,
    "dim": 64,
    "intermediate_dim": 128,
    "n_blocks": 2,
    "kernel_size": 3,
    "sinusoidal_dim": 16,
    "cond_dim": 64,
}
CFG = {"train": {"grad_clip_norm": 1.0}, "model": {"noise_emb": {"c_rescale": 1.0}}}


def _batch(b: int = 2, t_mel: int = 10) -> dict:
    return {"mel": torch.randn(b, 32, t_mel), "audio": torch.randn(b, t_mel * 64)}


@pytest.fixture
def model() -> DiffWaveNext2:
    torch.manual_seed(0)
    return DiffWaveNext2(sub_model_cfg=TINY)


# --------------------------------------------------------------------------- #
# build_sub_model_cfg (nested → flat マッピング)
# --------------------------------------------------------------------------- #
def test_build_sub_model_cfg_mapping() -> None:
    nested = {
        "sub_model": {
            "n_mels": 128,
            "n_fft": 1024,
            "hop": 256,
            "win_length": 1024,
            "convnext": {
                "embed_dim": 512,
                "intermediate_dim": 1536,
                "n_blocks": 8,
                "kernel_size": 7,
            },
            "noise_level_embedding": {"sinusoidal_dim": 128, "fc2": [512, 512]},
        }
    }
    flat = build_sub_model_cfg(nested)
    assert flat == {
        "mel_channels": 128,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "dim": 512,
        "intermediate_dim": 1536,
        "n_blocks": 8,
        "kernel_size": 7,
        "sinusoidal_dim": 128,
        "cond_dim": 512,
    }


def test_build_sub_model_cfg_produces_table1_model() -> None:
    # 実 config の flat マッピングが Table 1 整合の 14.354M sub-model を作る。
    flat = build_sub_model_cfg(
        {
            "sub_model": {
                "n_mels": 128,
                "n_fft": 1024,
                "hop": 256,
                "win_length": 1024,
                "convnext": {
                    "embed_dim": 512,
                    "intermediate_dim": 1536,
                    "n_blocks": 8,
                    "kernel_size": 7,
                },
                "noise_level_embedding": {"sinusoidal_dim": 128, "fc2": [512, 512]},
            }
        }
    )
    m = DiffWaveNext2.from_config({"sub_model_cfg": flat}, only_sub_model=1)
    n = sum(p.numel() for p in m.parameters())
    assert n == 14_354_434


def test_build_sub_model_cfg_empty() -> None:
    # sub_model 欠損でも空 dict (SubModelDiff default に委ねる)。
    assert build_sub_model_cfg({}) == {}


# --------------------------------------------------------------------------- #
# train_diff_step
# --------------------------------------------------------------------------- #
def test_train_diff_step_finite(model: DiffWaveNext2) -> None:
    opt = Adam(model.sub_models[1].parameters(), lr=2e-4)
    logs = train_diff_step(model, opt, _batch(), k=2, cfg=CFG)
    assert np.isfinite(logs["loss"])
    for key in ("loss", "grad_norm", "c_mean", "c_std", "abar_mean", "eps_norm", "x_t_norm"):
        assert key in logs


def test_train_diff_step_grad_isolation(model: DiffWaveNext2) -> None:
    opt = Adam(model.sub_models[1].parameters(), lr=2e-4)
    train_diff_step(model, opt, _batch(), k=2, cfg=CFG)
    assert any(p.grad is not None for p in model.sub_models[1].parameters())
    for j in (0, 2, 3):
        assert all(p.grad is None for p in model.sub_models[j].parameters())


def test_train_diff_step_invalid_k(model: DiffWaveNext2) -> None:
    opt = Adam(model.sub_models[0].parameters(), lr=2e-4)
    with pytest.raises(ValueError):
        train_diff_step(model, opt, _batch(), k=5, cfg=CFG)


@pytest.mark.parametrize(("k", "lo", "hi"), [(1, 0.9929, 1.0), (4, 0.0, 0.4817)])
def test_c_within_band(model: DiffWaveNext2, k: int, lo: float, hi: float) -> None:
    opt = Adam(model.sub_models[k - 1].parameters(), lr=2e-4)
    logs = train_diff_step(model, opt, _batch(b=64), k=k, cfg=CFG)
    assert lo <= logs["c_mean"] <= hi


def test_c_rescale_applied(model: DiffWaveNext2) -> None:
    # c_rescale=1000 でも step が finite に動く (conditioning 入力 scale、diffusion math は raw c)。
    cfg = {"train": {"grad_clip_norm": 1.0}, "model": {"noise_emb": {"c_rescale": 1000.0}}}
    opt = Adam(model.sub_models[0].parameters(), lr=2e-4)
    logs = train_diff_step(model, opt, _batch(), k=1, cfg=cfg)
    assert np.isfinite(logs["loss"])


# --------------------------------------------------------------------------- #
# overfit (loss 減少、slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_overfit_loss_decreases(model: DiffWaveNext2) -> None:
    torch.manual_seed(0)
    opt = Adam(model.sub_models[3].parameters(), lr=1e-3)
    batch = _batch(b=2, t_mel=10)
    losses = [train_diff_step(model, opt, batch, k=4, cfg=CFG)["loss"] for _ in range(200)]
    assert np.mean(losses[-20:]) < np.mean(losses[:20])  # 学習が進む


# --------------------------------------------------------------------------- #
# validation (3 点 evaluation)
# --------------------------------------------------------------------------- #
def test_run_validation_diff_3points(model: DiffWaveNext2) -> None:
    val_loader = [_batch(b=1) for _ in range(3)]  # DataLoader 互換の iterable
    out = run_validation_diff(model, val_loader, k=2, cfg=CFG, device=torch.device("cpu"))
    assert set(out) == {"mse_at_L", "mse_at_mid", "mse_at_U", "mse"}
    assert all(np.isfinite(v) for v in out.values())
    assert out["mse"] == pytest.approx(
        (out["mse_at_L"] + out["mse_at_mid"] + out["mse_at_U"]) / 3.0
    )


def test_run_validation_diff_deterministic(model: DiffWaveNext2) -> None:
    val_loader = [_batch(b=1) for _ in range(2)]
    a = run_validation_diff(model, val_loader, k=1, cfg=CFG, device=torch.device("cpu"), seed=7)
    b = run_validation_diff(model, val_loader, k=1, cfg=CFG, device=torch.device("cpu"), seed=7)
    assert a["mse"] == pytest.approx(b["mse"])


# --------------------------------------------------------------------------- #
# checkpoint roundtrip + 4 sub-model 独立保存
# --------------------------------------------------------------------------- #
def test_save_load_checkpoint_roundtrip(model: DiffWaveNext2, tmp_path) -> None:
    opt = Adam(model.sub_models[0].parameters(), lr=2e-4)
    train_diff_step(model, opt, _batch(), k=1, cfg=CFG)
    state = TrainStateDiff(step=42, best_val_mse=0.5, sub_model_k=1)
    path = tmp_path / "sub_1.pt"
    save_checkpoint(path, model, opt, state, cfg=CFG, atomic=True)

    model2 = DiffWaveNext2(sub_model_cfg=TINY)
    opt2 = Adam(model2.sub_models[0].parameters(), lr=2e-4)
    restored = load_checkpoint(path, model2, opt2)
    assert restored.step == 42
    assert restored.best_val_mse == 0.5
    assert restored.sub_model_k == 1
    # weight 一致
    for p1, p2 in zip(
        model.sub_models[0].parameters(), model2.sub_models[0].parameters(), strict=True
    ):
        assert torch.allclose(p1, p2)


def test_independent_sub_model_checkpoints(tmp_path) -> None:
    # lazy instantiation の sub_1 / sub_2 が独立ファイルに別々の weight で保存される。
    torch.manual_seed(1)
    m1 = DiffWaveNext2.from_config({"sub_model_cfg": TINY}, only_sub_model=1)
    m2 = DiffWaveNext2.from_config({"sub_model_cfg": TINY}, only_sub_model=2)
    o1 = Adam(m1.sub_models[0].parameters(), lr=2e-4)
    o2 = Adam(m2.sub_models[1].parameters(), lr=2e-4)
    save_checkpoint(tmp_path / "sub_1.pt", m1, o1, TrainStateDiff(sub_model_k=1), atomic=True)
    save_checkpoint(tmp_path / "sub_2.pt", m2, o2, TrainStateDiff(sub_model_k=2), atomic=True)
    assert (tmp_path / "sub_1.pt").exists()
    assert (tmp_path / "sub_2.pt").exists()
    # sub_1.pt は sub_models.0.* のみ、sub_2.pt は sub_models.1.* のみ含む (lazy)。
    s1 = torch.load(tmp_path / "sub_1.pt", weights_only=False)["model_state_dict"]
    s2 = torch.load(tmp_path / "sub_2.pt", weights_only=False)["model_state_dict"]
    assert any(k.startswith("sub_models.0.") for k in s1)
    assert not any(k.startswith("sub_models.1.") for k in s1)
    assert any(k.startswith("sub_models.1.") for k in s2)
    assert not any(k.startswith("sub_models.0.") for k in s2)


def test_lazy_only_target_instantiated() -> None:
    m = DiffWaveNext2.from_config({"sub_model_cfg": TINY}, only_sub_model=3)
    assert m.sub_models[2] is not None
    for j in (0, 1, 3):
        assert m.sub_models[j] is None
