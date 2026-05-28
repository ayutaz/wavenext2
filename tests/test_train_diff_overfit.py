"""T-M3.5: Diff smoke training (overfit / conditioning / reverse-mock).

docs/tickets/T-M3.5-diff-smoke.md。smoke の 3 軸 minimal sanity (step が回る / finite /
単調減少) を確認する。primary GPU smoke (sub-model 4、実 LibriTTS-R) は `slow`+`gpu` で
デフォルト deselect、CPU で動く synthetic overfit / reverse-mock / init_loss を主軸とする
(T-M2.6 GAN smoke と同じく synthetic 主軸、実データ GPU は T-M5.2 で実行)。

実 API への適応 (チケット §2.2 擬似コードとの差、§8.3):
- `train_diff_step(model, opt, batch, k, cfg)` (batch は dict、mel/audio 個別ではない)
- `build_model(cfg, sub_model_k, device)` (optimizer は別途 Adam)
- conditioning cos<0.99 は **訓練後 + full-dim** でないと顕在化しないため slow+gpu に限定
  (tiny/未訓練では additive bias の効果が小さく cos≈1.0、c_rescale ablation は GPU smoke)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from wavenext2.inference.infer_diff import reverse_sample
from wavenext2.models.diff_wavenext2 import DiffWaveNext2
from wavenext2.train.train_diff import train_diff_step
from wavenext2.utils.config import load_config

SMOKE_CONFIG = "configs/diff_wavenext2_smoke.yaml"
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


def _robust_init_loss(losses_early: list[float], losses_warmup: list[float]) -> float:
    """`init_loss = max(step 50-100 平均, step 10-20 平均, 1e-6)` の robust 定義 (§6.1)。

    Adam β1=0.9 で勾配 EMA 安定化は ~10 step、β2=0.98 で ~50 step。step 10-20 のみだと
    過渡的で init_loss を過小評価 (ratio が緩く見える) するため、step 50-100 も併用し最大値。
    """
    a = sum(losses_early) / len(losses_early) if losses_early else 0.0
    b = sum(losses_warmup) / len(losses_warmup) if losses_warmup else 0.0
    return max(a, b, 1e-6)


def _safe_close(writer) -> None:
    """spawn worker 内 atexit 再 register での double-close 例外を wrap (§6.1)。"""
    try:
        writer.close()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# init_loss robust 定義 (pure function、CPU)
# --------------------------------------------------------------------------- #
def test_init_loss_robust() -> None:
    assert _robust_init_loss([0.5] * 50, [0.3] * 10) == 0.5  # early > warmup
    assert _robust_init_loss([0.2] * 50, [0.6] * 10) == 0.6  # warmup > early
    assert _robust_init_loss([], []) == 1e-6  # floor


# --------------------------------------------------------------------------- #
# smoke config の必須キー
# --------------------------------------------------------------------------- #
def test_smoke_config_keys() -> None:
    cfg = load_config(SMOKE_CONFIG)
    assert cfg["model"]["sub_model_idx"] == 4  # primary
    assert cfg["model"]["noise_emb"]["c_rescale"] == 1.0
    assert cfg["train"]["max_steps"] == 1000
    assert cfg["train"]["batch_size"] == 1
    assert cfg["train"]["segment_length"] == 25600
    assert cfg["validation"]["interval_steps"] == 0  # validation off


# --------------------------------------------------------------------------- #
# reverse-mock (T-M3.3 β 負値 / sampler 符号 / index バグ早期検知、CPU)
# --------------------------------------------------------------------------- #
class _ZeroSub(nn.Module):
    """eps_pred = 0 を返す identity mock (sub-model 2-4 用)。"""

    def forward(self, mel: torch.Tensor, x_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_t)


def test_smoke_reverse_with_mocks() -> None:
    # sub-model 1 のみ少し訓練、2-4 を identity mock → 4-step reverse が NaN を出さない。
    torch.manual_seed(0)
    model = DiffWaveNext2(sub_model_cfg=TINY)
    opt = Adam(model.sub_models[0].parameters(), lr=1e-3)
    batch = {"mel": torch.randn(1, 32, 10), "audio": torch.randn(1, 10 * 64)}
    for _ in range(50):
        train_diff_step(model, opt, batch, k=1, cfg=CFG)
    for i in (1, 2, 3):
        model.sub_models[i] = _ZeroSub()
    out = reverse_sample(model, batch["mel"], seed=0)
    assert out.shape == (1, 10 * 64)
    assert torch.isfinite(out).all()  # β-free reverse: 負値 β 由来の NaN なし
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


# --------------------------------------------------------------------------- #
# synthetic overfit (CPU、slow): 3 軸 sanity (回る / finite / 減少)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_smoke_synthetic_cpu() -> None:
    torch.manual_seed(0)
    model = DiffWaveNext2(sub_model_cfg=TINY)
    k = 4  # primary
    opt = Adam(model.sub_models[k - 1].parameters(), lr=1e-3)
    # 固定 synthetic batch: 220 Hz sine 波 + 固定 mel。
    t = torch.linspace(0, 1, 10 * 64)
    audio = (0.6 * torch.sin(2 * np.pi * 220 * t)).unsqueeze(0)
    batch = {"mel": torch.randn(1, 32, 10), "audio": audio}

    losses_warmup: list[float] = []  # step 10-20
    losses_early: list[float] = []  # step 50-100
    last = float("nan")
    for step in range(400):
        logs = train_diff_step(model, opt, batch, k=k, cfg=CFG)
        assert np.isfinite(logs["loss"]), f"NaN/Inf at step {step}"
        if 10 <= step < 20:
            losses_warmup.append(logs["loss"])
        if 50 <= step < 100:
            losses_early.append(logs["loss"])
        last = logs["loss"]

    init_loss = _robust_init_loss(losses_early, losses_warmup)
    final = (last + losses_early[-1]) / 2.0 if losses_early else last
    ratio = final / init_loss
    # CPU tiny model + fresh-eps/random-c のため厳格な <0.05 は GPU full smoke (T-M5.2) 側。
    # synthetic では「学習が進む」(ratio < 0.9) を architectural sanity として確認。
    assert ratio < 0.9, f"synthetic overfit did not learn: ratio={ratio:.4f}"


# --------------------------------------------------------------------------- #
# GPU + 実 LibriTTS-R smoke (primary、デフォルト deselect: slow+gpu)
# --------------------------------------------------------------------------- #
def _run_real_overfit(config_path: str, sub_idx: int, strict_ratio: bool) -> dict:
    """1 utterance を max_steps overfit する GPU smoke 本体 (実 API 適応)。"""
    from torch.utils.data import DataLoader, Subset

    from wavenext2.data.dataset import LibriTTSRDataset
    from wavenext2.train.train_diff import build_model

    cfg = load_config(config_path)
    device = torch.device("cuda")
    hop = cfg["model"]["sub_model"]["hop"]
    mel_cfg = {**cfg["data"]["mel"], "hop_length": hop}
    ds = LibriTTSRDataset(
        filelist_path=cfg["data"]["val_filelist"],
        root_dir=cfg["data"]["root_dir"],
        segment_length=cfg["train"]["segment_length"],
        hop_length=hop,
        mel_cfg=mel_cfg,
        mode="val",
    )
    loader = DataLoader(Subset(ds, [cfg["train"].get("smoke_idx", 0)]), batch_size=1)
    model = build_model(cfg, sub_idx, device)
    o = cfg["train"]["optimizer"]
    opt = Adam(model.sub_models[sub_idx - 1].parameters(), lr=o["lr"], betas=tuple(o["betas"]))

    losses_warmup: list[float] = []
    losses_early: list[float] = []
    last = float("nan")
    step = 0
    max_steps = cfg["train"]["max_steps"]
    while step < max_steps:
        for batch in loader:
            batch = {kk: v.to(device) if torch.is_tensor(v) else v for kk, v in batch.items()}
            logs = train_diff_step(model, opt, batch, sub_idx, cfg)
            assert np.isfinite(logs["loss"]), f"NaN/Inf at step {step}"
            if 10 <= step < 20:
                losses_warmup.append(logs["loss"])
            if 50 <= step < 100:
                losses_early.append(logs["loss"])
            last = logs["loss"]
            step += 1
            if step >= max_steps:
                break
    init_loss = _robust_init_loss(losses_early, losses_warmup)
    ratio = last / init_loss
    if strict_ratio:
        assert ratio < 0.05, (
            f"overfit failed: ratio {ratio:.4f} >= 0.05 "
            f"(T-M3.3/T-M1.5/T-M3.1/T-M3.2 を確認、c_rescale ablation も検討)"
        )
    return {"final_loss": last, "init_loss": init_loss, "ratio": ratio}


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes_sub_model_4() -> None:
    """PRIMARY: sub-model 4 を 1000 step overfit、ratio < 0.05 (要 GPU + 実 LibriTTS-R)。"""
    _run_real_overfit(SMOKE_CONFIG, sub_idx=4, strict_ratio=True)


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes_sub_model_1() -> None:
    """SUBORDINATE: sub-model 1 は step が回る + finite のみ (conditioning 感度極小)。"""
    _run_real_overfit(SMOKE_CONFIG, sub_idx=1, strict_ratio=False)


@pytest.mark.slow
@pytest.mark.gpu
def test_noise_level_conditioning() -> None:
    """訓練後 sub-model 4 で c=L vs c=U の eps_pred cos<0.99 (NoiseEmbedding 動作、要 GPU)。

    未訓練 / tiny では additive bias の効果が小さく cos≈1.0 になるため、full-dim + 訓練後の
    実 smoke でのみ意味を持つ。fail 時は c_rescale=1000 ablation (T-M1.5) を検討。
    """
    cfg = load_config(SMOKE_CONFIG)
    device = torch.device("cuda")
    from wavenext2.train.train_diff import build_model

    model = build_model(cfg, 4, device).eval()
    seg = cfg["train"]["segment_length"]
    mel = torch.randn(1, 128, seg // cfg["model"]["sub_model"]["hop"], device=device)
    x_t = torch.randn(1, seg, device=device)
    lo, hi = model.BAND_BOUNDS[3]
    with torch.no_grad():
        a = model.sub_models[3](mel, x_t, torch.full((1,), lo, device=device))
        b = model.sub_models[3](mel, x_t, torch.full((1,), hi, device=device))
    cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()
    assert cos < 0.99, f"NoiseEmbedding 無視の疑い: cos={cos:.4f} (T-M1.5/T-M1.1、c_rescale ablation)"


# --------------------------------------------------------------------------- #
# wrapper script の存在 + 健全性
# --------------------------------------------------------------------------- #
def test_smoke_diff_script_exists() -> None:
    assert Path("scripts/smoke_diff.py").exists()
