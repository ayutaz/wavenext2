"""train_gan.py — GAN-WaveNeXt 2 の訓練ループ (T-M2.5).

論文 §3.2 / docs/training.md §2 を実装する。
- Optimizer: AdamW (G/D 別 lr、betas=[0.8, 0.99], weight_decay=1e-3)
- Scheduler: InverseLR (inv_gamma=200000, power=0.5, warmup=0.999) を G/D 別々に
- Loss: hinge GAN (D: relu(1-D(real))+relu(1+D(fake)) / G: -D(fake)) + FM L1 + MR-STFT (SC+Mag)
- Grad clip: max_norm=1.0、EMA: 不使用 (open-questions 確定)
- Logging: TensorBoard (step ごとに loss、validation で MR-STFT total)

公開 API:
- `main()`: CLI entry point (argparse; click は依存に無いため stdlib を採用、ticket §8.3)。
- `train_gan_step(...) -> dict[str, float]`: 1 step alternating update。T-M2.6 smoke /
  T-M3.2 (train_diff_step) / 将来の Lightning 移行で再利用される唯一の公開関数。

**生成波形のクロップ**: `GANWaveNext2(mel)` は `T_mel*hop` サンプル (center=True で
segment_length より hop ぶん長い) を出すため、GT `audio` 長に crop してから loss を計算する
(T-M2.1/T-M2.4 申し送り。gen 長 > segment_length は常に成立するため pad は不要)。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wavenext2.data.dataset import LibriTTSRDataset, seed_worker
from wavenext2.losses.adversarial import HingeGANLoss
from wavenext2.losses.feature_matching import FeatureMatchingLoss
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss
from wavenext2.models.discriminator import MultiScaleDiscriminator
from wavenext2.models.gan_wavenext2 import GANWaveNext2
from wavenext2.utils.config import load_config
from wavenext2.utils.scheduler import InverseLR
from wavenext2.utils.seed import set_seed
from wavenext2.utils.training_loop import (
    atomic_save,
    capture_rng_state,
    iter_forever,
    register_sigterm_handler,
    restore_rng_state,
)

__all__ = ["TrainState", "main", "train_gan_step"]

logger = logging.getLogger("wavenext2.train_gan")

# loss_D がこの値を連続で下回ったら D が強すぎると警告 (§6.1)
_D_LOSS_WEAK_THRESHOLD = 0.01
_D_LOSS_WEAK_PATIENCE = 1000


@dataclass
class TrainState:
    """訓練の進行状態 (checkpoint で保存・復元する)。"""

    step: int = 0
    best_val_mrstft: float = float("inf")
    d_loss_below_threshold_steps: int = 0

    def update_d_loss_history(self, loss_d: float) -> None:
        if loss_d < _D_LOSS_WEAK_THRESHOLD:
            self.d_loss_below_threshold_steps += 1
        else:
            self.d_loss_below_threshold_steps = 0


@dataclass
class _Criteria:
    """loss モジュール束 (train_gan_step に渡す)。"""

    gan: HingeGANLoss = field(default_factory=HingeGANLoss)
    fm: FeatureMatchingLoss = field(default_factory=FeatureMatchingLoss)
    mrstft: MultiResolutionSTFTLoss = field(default_factory=MultiResolutionSTFTLoss)


def train_gan_step(
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sch_G: torch.optim.lr_scheduler.LRScheduler,
    sch_D: torch.optim.lr_scheduler.LRScheduler,
    mel: torch.Tensor,
    audio: torch.Tensor,
    cfg: dict,
    crit: _Criteria,
    amp: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """1 step の hinge GAN 交互更新を実行し loss scalar の dict を返す。

    autocast 境界は G/D forward のみ。hinge/FM/MR-STFT loss は fp32 で計算 (bf16 underflow 回避)。

    Args:
        G, D: generator / discriminator。
        opt_G, opt_D, sch_G, sch_D: optimizer / scheduler (G/D 別)。
        mel: (B, 128, T_mel)。audio: (B, segment_length) GT 波形。
        cfg: `cfg["loss"]["weights"]` と `cfg["train"]["grad_clip_norm"]` を参照。
        crit: loss モジュール束。
        amp: True で bf16 autocast。dtype: autocast dtype。

    Returns:
        loss_G/loss_D/各 sub-loss/grad_norm を含む float dict。
    """
    device_type = mel.device.type
    w = cfg["loss"]["weights"]
    clip = cfg["train"]["grad_clip_norm"]
    seg = audio.shape[-1]

    # --- Generator forward (autocast 内) → GT 長に crop ---
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=amp):
        y_hat = G(mel)[..., :seg]

    # --- Discriminator 更新 (forward は autocast、hinge は fp32) ---
    opt_D.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=amp):
        d_real = D(audio)
        d_fake = D(y_hat.detach())
    with torch.autocast(device_type=device_type, enabled=False):
        d_real_logits = [o.logits.float() for o in d_real]
        d_fake_logits = [o.logits.float() for o in d_fake]
        loss_d_real = sum(F.relu(1.0 - lg).mean() for lg in d_real_logits) / len(d_real_logits)
        loss_d_fake = sum(F.relu(1.0 + lg).mean() for lg in d_fake_logits) / len(d_fake_logits)
        loss_D = loss_d_real + loss_d_fake
    loss_D.backward()
    grad_norm_D = torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=clip)
    opt_D.step()
    sch_D.step()

    # --- Generator 更新 (forward は autocast、loss は fp32) ---
    opt_G.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=amp):
        d_fake_g = D(y_hat)
        d_real_g = D(audio)
    with torch.autocast(device_type=device_type, enabled=False):
        loss_g_gan = crit.gan.g_loss([o.logits.float() for o in d_fake_g])
        loss_g_fm = crit.fm(
            [[f.float() for f in o.features] for o in d_real_g],
            [[f.float() for f in o.features] for o in d_fake_g],
        )
        sc, mag = crit.mrstft(y_hat.float(), audio.float())
        loss_g_gan_w = w["d_gan"] * loss_g_gan
        loss_g_fm_w = w["d_fm"] * loss_g_fm
        loss_g_sc_w = w["mrstft_sc"] * sc
        loss_g_mag_w = w["mrstft_mag"] * mag
        loss_G = loss_g_gan_w + loss_g_fm_w + loss_g_sc_w + loss_g_mag_w
    loss_G.backward()
    grad_norm_G = torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=clip)
    opt_G.step()
    sch_G.step()

    return {
        "loss_G": loss_G.item(),
        "loss_D": loss_D.item(),
        "loss_g_gan": loss_g_gan.item(),
        "loss_g_fm": loss_g_fm.item(),
        "loss_g_mrstft_sc": sc.item(),
        "loss_g_mrstft_mag": mag.item(),
        "loss_d_real": loss_d_real.item(),
        "loss_d_fake": loss_d_fake.item(),
        "grad_norm_G": float(grad_norm_G),
        "grad_norm_D": float(grad_norm_D),
    }


# --- validation ----------------------------------------------------------------
@torch.no_grad()
def run_validation(
    G: nn.Module,
    val_loader: DataLoader,
    crit: _Criteria,
    device: torch.device,
) -> float:
    """val_loader 全体で MR-STFT total (sc+mag) の平均を返す (best ckpt 基準)。"""
    G.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        mel = batch["mel"].to(device)
        audio = batch["audio"].to(device)
        y_hat = G(mel)[..., : audio.shape[-1]]
        sc, mag = crit.mrstft(y_hat.float(), audio.float())
        total += float(sc + mag) * mel.shape[0]
        n += mel.shape[0]
    G.train()
    return total / max(n, 1)


# --- checkpoint ----------------------------------------------------------------
def save_checkpoint(
    path: Path,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sch_G: torch.optim.lr_scheduler.LRScheduler,
    sch_D: torch.optim.lr_scheduler.LRScheduler,
    state: TrainState,
    cfg: dict | None = None,
    atomic: bool = False,
) -> None:
    """resume に必要な全 state を保存。atomic=True で `.tmp`→os.replace の原子的書き込み。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": state.step,
        "best_val_mrstft": state.best_val_mrstft,
        "G_state_dict": G.state_dict(),
        "D_state_dict": D.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_D_state_dict": opt_D.state_dict(),
        "sch_G_state_dict": sch_G.state_dict(),
        "sch_D_state_dict": sch_D.state_dict(),
        "rng_state": capture_rng_state(),
        "config": cfg,
    }
    if atomic:
        atomic_save(ckpt, path)  # `.tmp` → os.replace (utils.training_loop)
    else:
        torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    G: nn.Module,
    D: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    sch_G: torch.optim.lr_scheduler.LRScheduler,
    sch_D: torch.optim.lr_scheduler.LRScheduler,
) -> TrainState:
    """checkpoint から全 state を復元し TrainState を返す。"""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    G.load_state_dict(ckpt["G_state_dict"])
    D.load_state_dict(ckpt["D_state_dict"])
    opt_G.load_state_dict(ckpt["opt_G_state_dict"])
    opt_D.load_state_dict(ckpt["opt_D_state_dict"])
    sch_G.load_state_dict(ckpt["sch_G_state_dict"])
    sch_D.load_state_dict(ckpt["sch_D_state_dict"])
    restore_rng_state(ckpt.get("rng_state"))
    return TrainState(
        step=ckpt["step"],
        best_val_mrstft=ckpt.get("best_val_mrstft", float("inf")),
    )


# --- data ----------------------------------------------------------------------
def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """train / val の DataLoader を構築 (実データ駆動、real audio が前提)。"""
    data, train = cfg["data"], cfg["train"]
    hop = cfg["model"]["sub_model"]["hop"]
    mel_cfg = {**data["mel"], "hop_length": hop}
    common = {
        "root_dir": data["root_dir"],
        "segment_length": train["segment_length"],
        "hop_length": hop,
        "mel_cfg": mel_cfg,
    }
    train_ds = LibriTTSRDataset(filelist_path=data["train_filelist"], mode="train", **common)
    val_ds = LibriTTSRDataset(filelist_path=data["val_filelist"], mode="val", **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["num_workers"],
        worker_init_fn=seed_worker,
        persistent_workers=train["num_workers"] > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, val_loader


def log_scalars(
    writer: SummaryWriter, step: int, logs: dict[str, float], lr_g: float, lr_d: float
) -> None:
    for k, v in logs.items():
        writer.add_scalar(f"train/{k}", v, step)
    writer.add_scalar("train/lr_G", lr_g, step)
    writer.add_scalar("train/lr_D", lr_d, step)


# --- main ----------------------------------------------------------------------
def _build_optim(
    cfg: dict, G: nn.Module, D: nn.Module
) -> tuple[AdamW, AdamW, InverseLR, InverseLR]:
    o = cfg["train"]["optimizer"]
    betas = tuple(o["betas"])
    opt_G = AdamW(G.parameters(), lr=o["lr_g"], betas=betas, weight_decay=o["weight_decay"])
    opt_D = AdamW(D.parameters(), lr=o["lr_d"], betas=betas, weight_decay=o["weight_decay"])
    s = {k: v for k, v in cfg["train"]["scheduler"].items() if k != "type"}
    return opt_G, opt_D, InverseLR(opt_G, **s), InverseLR(opt_D, **s)


def main(argv: list[str] | None = None) -> None:
    """GAN-WaveNeXt 2 の訓練 entry point (argparse)."""
    parser = argparse.ArgumentParser(description="Train GAN-WaveNeXt 2")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--debug", action="store_true", help="1 step だけ実行する smoke")
    parser.add_argument("--amp", action="store_true", help="bf16 mixed precision (既定 fp32)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    seed = cfg.get("seed")
    if seed is None:
        seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.amp else torch.float32

    G = GANWaveNext2.from_config(cfg["model"]).to(device)
    D = MultiScaleDiscriminator.from_config(cfg["discriminator"]).to(device)
    opt_G, opt_D, sch_G, sch_D = _build_optim(cfg, G, D)
    crit = _Criteria(mrstft=MultiResolutionSTFTLoss(**cfg["loss"]["mrstft"]))
    crit.mrstft.to(device)  # MR-STFT の window buffer を device へ (stft の device 不一致回避)

    state = TrainState()
    if args.resume:
        state = load_checkpoint(Path(args.resume), G, D, opt_G, opt_D, sch_G, sch_D)
        logger.info("resumed from %s at step %d", args.resume, state.step)

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])

    register_sigterm_handler(
        lambda: save_checkpoint(
            ckpt_dir / f"emergency_step_{state.step}.pt",
            G,
            D,
            opt_G,
            opt_D,
            sch_G,
            sch_D,
            state,
            cfg,
            atomic=True,
        )
    )

    train_loader, val_loader = build_loaders(cfg)
    G.train()
    D.train()
    for batch in iter_forever(train_loader):
        if state.step >= cfg["train"]["max_steps"]:
            break
        mel = batch["mel"].to(device)
        audio = batch["audio"].to(device)
        logs = train_gan_step(
            G, D, opt_G, opt_D, sch_G, sch_D, mel, audio, cfg, crit, amp=args.amp, dtype=dtype
        )
        state.update_d_loss_history(logs["loss_D"])
        if state.step % cfg["logging"]["scalar_interval_steps"] == 0:
            log_scalars(writer, state.step, logs, sch_G.get_last_lr()[0], sch_D.get_last_lr()[0])
        if state.d_loss_below_threshold_steps >= _D_LOSS_WEAK_PATIENCE:
            logger.warning(
                "loss_D < %.3g for %d steps: D may be too strong",
                _D_LOSS_WEAK_THRESHOLD,
                _D_LOSS_WEAK_PATIENCE,
            )

        val_iv = cfg["validation"]["interval_steps"]
        if val_iv and state.step > 0 and state.step % val_iv == 0:
            val = run_validation(G, val_loader, crit, device)
            writer.add_scalar("val/mrstft_total", val, state.step)
            if val < state.best_val_mrstft:
                state.best_val_mrstft = val
                save_checkpoint(
                    ckpt_dir / "best.pt", G, D, opt_G, opt_D, sch_G, sch_D, state, cfg, atomic=True
                )

        ckpt_iv = cfg["checkpoint"]["interval_steps"]
        if ckpt_iv and state.step > 0 and state.step % ckpt_iv == 0:
            save_checkpoint(
                ckpt_dir / f"step_{state.step}.pt", G, D, opt_G, opt_D, sch_G, sch_D, state, cfg
            )

        state.step += 1
        if args.debug:
            break

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
