"""train_diff.py — Diff-WaveNeXt 2 の訓練ループ (T-M3.2).

論文 §3.3 / docs/training.md §3 を実装する。
- 4 sub-model 独立訓練 (CLI 引数 `--sub-model {1,2,3,4}` で 1 つずつ指定)
- lazy instantiation: `DiffWaveNext2.from_config(model_cfg, only_sub_model=k)` で k 番目のみ
- Optimizer: Adam (lr=2e-4, betas=[0.9, 0.98], weight_decay=0.0)。Scheduler なし (固定 lr)
- Loss: F.mse_loss(eps_pred, eps_gt) on noise prediction (Fig 1b)
- Forward: x_t = √ᾱ * x_0 + √(1-ᾱ) * ε (DDPM 標準形、band 内 uniform sampling)
  c / abar の計算は **fp32 強制** (bf16 で c≈0.99995 が 1.0 に丸まり √ᾱ=0 となる罠を回避、§6.1)
- Grad clip: max_norm=1.0、EMA: 不使用
- Validation: c ∈ {L, mid, U} の 3 点 fixed evaluation (band 境界の過適合検知)

公開 API:
- `main()`: argparse CLI entry point (click は非依存のため train_gan と同じく argparse を採用)。
- `train_diff_step(model, opt, batch, k, cfg, amp, dtype) -> dict[str, float]`:
  T-M2.5 `train_gan_step` と「dict 返却 step 関数」原則を共有する公開関数 (signature 自体は
  GAN と異なる。M2 review で統一不可と確定)。T-M3.5 smoke / fixture / Lightning 移行で再利用。

checkpoint payload は GAN と異なる (単一 model/opt/sub_model_k) ため save/load は本ファイルに置き、
共通プリミティブ (atomic_save / capture_rng_state / restore_rng_state / iter_forever /
register_sigterm_handler) は `utils.training_loop` から import する。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wavenext2.data.dataset import LibriTTSRDataset, seed_worker
from wavenext2.models.diff_wavenext2 import DiffWaveNext2
from wavenext2.utils.config import load_config
from wavenext2.utils.seed import set_seed
from wavenext2.utils.training_loop import (
    atomic_save,
    capture_rng_state,
    iter_forever,
    register_sigterm_handler,
    restore_rng_state,
)

__all__ = ["TrainStateDiff", "build_sub_model_cfg", "main", "train_diff_step"]

logger = logging.getLogger("wavenext2.train_diff")


@dataclass
class TrainStateDiff:
    """Diff 訓練の進行状態 (checkpoint で保存・復元)。sub_model_k を保持。"""

    step: int = 0
    best_val_mse: float = float("inf")
    sub_model_k: int | None = None


def build_sub_model_cfg(model_cfg: dict) -> dict:
    """YAML の nested `model.sub_model` を SubModelDiff の flat kwargs へ写像する (SoT)。

    M2 review 申し送り (config nested→flat マッピング): `n_mels`→`mel_channels`、`hop`→
    `hop_length`、`convnext.*`→`dim`/`intermediate_dim`/`n_blocks`/`kernel_size`、
    `noise_level_embedding.*`→`sinusoidal_dim`/`cond_dim` を明示マップ。偶然の default 一致に
    依存しない。欠損キーは SubModelDiff の default に委ねる。
    """
    sm = model_cfg.get("sub_model", {})
    convnext = sm.get("convnext", {})
    noise_emb = sm.get("noise_level_embedding", {})
    flat: dict = {}
    if "n_mels" in sm:
        flat["mel_channels"] = sm["n_mels"]
    if "n_fft" in sm:
        flat["n_fft"] = sm["n_fft"]
    if "hop" in sm:
        flat["hop_length"] = sm["hop"]
    if "win_length" in sm:
        flat["win_length"] = sm["win_length"]
    if "embed_dim" in convnext:
        flat["dim"] = convnext["embed_dim"]
    if "intermediate_dim" in convnext:
        flat["intermediate_dim"] = convnext["intermediate_dim"]
    if "n_blocks" in convnext:
        flat["n_blocks"] = convnext["n_blocks"]
    if "kernel_size" in convnext:
        flat["kernel_size"] = convnext["kernel_size"]
    if "sinusoidal_dim" in noise_emb:
        flat["sinusoidal_dim"] = noise_emb["sinusoidal_dim"]
    # cond_dim は noise embedding 出力次元 = ConvNeXt embed_dim に揃える (additive bias)。
    if "fc2" in noise_emb and isinstance(noise_emb["fc2"], (list, tuple)):
        flat["cond_dim"] = noise_emb["fc2"][-1]
    elif "embed_dim" in convnext:
        flat["cond_dim"] = convnext["embed_dim"]
    return flat


def train_diff_step(
    model: DiffWaveNext2,
    opt: torch.optim.Optimizer,
    batch: dict,
    k: int,
    cfg: dict,
    amp: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """1 step backward を実行し loss scalar の dict を返す (公開関数)。

    Args:
        model: DiffWaveNext2 (lazy instantiation で sub_models[k-1] のみ実体)。
        opt:   Adam (sub_models[k-1].parameters() に紐付け済)。
        batch: {"mel": (B,128,T_mel), "audio": (B,segment_length)}。
        k:     訓練対象 sub-model index (1..4)。
        cfg:   YAML config dict (`model.noise_emb.c_rescale` で conditioning 入力 scale)。
        amp:   bf16 mixed precision (c/abar は fp32 強制で underflow 回避)。
        dtype: torch.bfloat16 if amp else torch.float32。

    Returns:
        loss / grad_norm / c_mean / c_std / abar_mean / eps_norm / eps_pred_norm / x_t_norm。
        `_c_batch` (histogram 用 tensor、consumer 側で pop)。
    """
    if not 1 <= k <= model.K:
        raise ValueError(f"sub-model index must be in [1, {model.K}], got {k}")
    sub = model.sub_models[k - 1]
    mel = batch["mel"]
    x_gt = batch["audio"]
    b = x_gt.shape[0]
    device = x_gt.device

    eps = torch.randn_like(x_gt)
    c = model.sample_noise_level(k, b).to(device)  # (B,) = √(1-ᾱ_t)、band uniform

    # c / abar は fp32 強制 (bf16 で c≈0.99995 が 1.0 に丸まり √ᾱ=0 になる罠、§6.1 Critical)。
    with torch.autocast(device_type=device.type, enabled=False):
        c_fp32 = c.float()
        abar = 1.0 - c_fp32**2
        sqrt_abar = torch.sqrt(abar)
        sqrt_one_minus_abar = c_fp32

    # x_t = √ᾱ * x_0 + √(1-ᾱ) * ε (DDPM 標準形)
    x_t = (
        sqrt_abar.to(x_gt.dtype).unsqueeze(-1) * x_gt
        + sqrt_one_minus_abar.to(x_gt.dtype).unsqueeze(-1) * eps
    )

    # conditioning 入力のみ c_rescale を乗算 (diffusion math は raw c、T-M1.5 §9.1 ablation)。
    c_rescale = cfg.get("model", {}).get("noise_emb", {}).get("c_rescale", 1.0)
    c_cond = c * c_rescale

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
        eps_pred = sub(mel, x_t, c_cond)
    with torch.autocast(device_type=device.type, enabled=False):
        loss = F.mse_loss(eps_pred.float(), eps.float())

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        sub.parameters(), max_norm=cfg["train"]["grad_clip_norm"]
    )
    opt.step()

    return {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "c_mean": c.mean().item(),
        "c_std": c.std().item() if b > 1 else 0.0,
        "abar_mean": abar.mean().item(),
        "eps_norm": eps.norm().item(),
        "eps_pred_norm": eps_pred.detach().float().norm().item(),
        "x_t_norm": x_t.detach().float().norm().item(),
        "_c_batch": c.detach().cpu(),
    }


@torch.no_grad()
def run_validation_diff(
    model: DiffWaveNext2,
    val_loader: DataLoader,
    k: int,
    cfg: dict,
    device: torch.device,
    seed: int = 42,
) -> dict[str, float]:
    """c ∈ {L, mid, U} の 3 点 fixed evaluation で MSE を計算 (band 境界の過適合検知)。

    val seed を固定して eps を deterministic 化。戻り値は 3 点 + 平均 `mse` の dict。
    """
    sub = model.sub_models[k - 1]
    lo, hi = model.BAND_BOUNDS[k - 1]
    c_points = {"mse_at_L": float(lo), "mse_at_mid": float((lo + hi) / 2.0), "mse_at_U": float(hi)}
    c_rescale = cfg.get("model", {}).get("noise_emb", {}).get("c_rescale", 1.0)

    model.eval()
    sums = dict.fromkeys(c_points, 0.0)
    n = 0
    gen = torch.Generator(device=device).manual_seed(seed)
    for batch in val_loader:
        mel = batch["mel"].to(device)
        x_gt = batch["audio"].to(device)
        b = x_gt.shape[0]
        eps = torch.randn(x_gt.shape, generator=gen, device=device, dtype=x_gt.dtype)
        for tag, c_scalar in c_points.items():
            c_val = torch.full((b,), c_scalar, device=device, dtype=torch.float32)
            abar = 1.0 - c_val**2
            sqrt_abar = torch.sqrt(abar)
            x_t = (
                sqrt_abar.to(x_gt.dtype).unsqueeze(-1) * x_gt
                + c_val.to(x_gt.dtype).unsqueeze(-1) * eps
            )
            eps_pred = sub(mel, x_t, (c_val * c_rescale).to(x_gt.dtype))
            sums[tag] += F.mse_loss(eps_pred.float(), eps.float()).item() * b
        n += b
    model.train()

    out = {tag: s / max(n, 1) for tag, s in sums.items()}
    out["mse"] = (out["mse_at_L"] + out["mse_at_mid"] + out["mse_at_U"]) / 3.0
    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    state: TrainStateDiff,
    cfg: dict | None = None,
    atomic: bool = False,
) -> None:
    """Diff sub-model の resume に必要な全 state を保存 (payload は GAN と別)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": state.step,
        "best_val_mse": state.best_val_mse,
        "sub_model_k": state.sub_model_k,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "rng_state": capture_rng_state(),
        "config": cfg,
    }
    if atomic:
        atomic_save(ckpt, path)
    else:
        torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
) -> TrainStateDiff:
    """checkpoint から model/opt/RNG state を復元し TrainStateDiff を返す。"""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["opt_state_dict"])
    restore_rng_state(ckpt.get("rng_state"))
    return TrainStateDiff(
        step=ckpt["step"],
        best_val_mse=ckpt.get("best_val_mse", float("inf")),
        sub_model_k=ckpt.get("sub_model_k"),
    )


def build_loaders(cfg: dict, sub_model_k: int) -> tuple[DataLoader, DataLoader]:
    """train / val の DataLoader を構築 (T-M2.1 LibriTTSRDataset、実 audio 前提)。

    sub-model 別 seed offset で `--sub-model k` 並列起動時の worker seed 衝突を回避
    (base_seed + k*1_000_000 + worker_id)。
    """
    data, train = cfg["data"], cfg["train"]
    hop = cfg["model"]["sub_model"]["hop"]
    mel_cfg = {**data["mel"], "hop_length": hop}
    common = {
        "root_dir": data["root_dir"],
        "segment_length": train["segment_length"],
        "hop_length": hop,
        "mel_cfg": mel_cfg,
    }
    base_seed = train.get("seed", 0)
    sub_offset = sub_model_k * 1_000_000

    def _seed_worker(worker_id: int) -> None:
        # sub-model 別 offset を加えてから T-M2.1 標準 seed_worker を呼ぶ。
        seed_worker(worker_id + sub_offset + base_seed)

    train_ds = LibriTTSRDataset(filelist_path=data["train_filelist"], mode="train", **common)
    val_ds = LibriTTSRDataset(filelist_path=data["val_filelist"], mode="val", **common)
    train_loader = DataLoader(
        train_ds,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train["num_workers"],
        worker_init_fn=_seed_worker,
        persistent_workers=train["num_workers"] > 0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, val_loader


def build_model(cfg: dict, sub_model_k: int, device: torch.device) -> DiffWaveNext2:
    """lazy instantiation で sub-model k のみを持つ DiffWaveNext2 を構築。"""
    flat = build_sub_model_cfg(cfg["model"])
    model = DiffWaveNext2.from_config({"sub_model_cfg": flat}, only_sub_model=sub_model_k)
    return model.to(device)


def main(argv: list[str] | None = None) -> None:
    """Diff-WaveNeXt 2 sub-model k の訓練 entry point (argparse)."""
    parser = argparse.ArgumentParser(description="Train Diff-WaveNeXt 2 (1 sub-model)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--sub-model", type=int, choices=[1, 2, 3, 4], required=True, dest="sub_model_k"
    )
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--debug", action="store_true", help="1 step だけ実行する smoke")
    parser.add_argument("--amp", action="store_true", help="bf16 mixed precision (既定 fp32)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    # top-level seed が無ければ train.seed を使う (None チェックで 0 を握り潰さない、ML review C1)。
    seed = cfg.get("seed")
    if seed is None:
        seed = cfg.get("train", {}).get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.amp else torch.float32
    k = args.sub_model_k

    model = build_model(cfg, k, device)
    sub = model.sub_models[k - 1]
    o = cfg["train"]["optimizer"]
    opt = Adam(
        sub.parameters(),
        lr=o["lr"],
        betas=tuple(o["betas"]),
        weight_decay=o["weight_decay"],
    )

    state = TrainStateDiff(sub_model_k=k)
    if args.resume:
        state = load_checkpoint(Path(args.resume), model, opt)
        logger.info("resumed sub_%d from %s at step %d", k, args.resume, state.step)

    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    writer = SummaryWriter(str(Path(cfg["logging"]["tensorboard_dir"]) / f"sub_{k}"))
    register_sigterm_handler(
        lambda: save_checkpoint(
            ckpt_dir / f"emergency_step_{state.step}_sub_{k}.pt",
            model,
            opt,
            state,
            cfg,
            atomic=True,
        )
    )

    train_loader, val_loader = build_loaders(cfg, k)
    model.train()
    for batch in iter_forever(train_loader):
        if state.step >= cfg["train"]["max_steps"]:
            break
        batch = {kk: v.to(device) if torch.is_tensor(v) else v for kk, v in batch.items()}
        logs = train_diff_step(model, opt, batch, k, cfg, amp=args.amp, dtype=dtype)

        if state.step % cfg["logging"]["scalar_interval_steps"] == 0:
            for name, val in logs.items():
                if name.startswith("_"):
                    continue
                writer.add_scalar(f"sub_{k}/{name}", val, state.step)
            writer.add_scalar(f"sub_{k}/lr", opt.param_groups[0]["lr"], state.step)
        hist_iv = cfg["logging"].get("histogram_interval_steps", 0)
        if hist_iv and state.step % hist_iv == 0:
            writer.add_histogram(f"sub_{k}/noise_level_c", logs["_c_batch"], state.step)

        # interval_steps=0 は「無効」(smoke 等)。`step % 0` の ZeroDivisionError を回避 (ML review I1)。
        val_iv = cfg["validation"]["interval_steps"]
        if val_iv and state.step > 0 and state.step % val_iv == 0:
            val = run_validation_diff(model, val_loader, k, cfg, device)
            for tag, v in val.items():
                writer.add_scalar(f"sub_{k}/val/{tag}", v, state.step)
            if val["mse"] < state.best_val_mse:
                state.best_val_mse = val["mse"]
                save_checkpoint(ckpt_dir / f"sub_{k}.pt", model, opt, state, cfg, atomic=True)

        ckpt_iv = cfg["checkpoint"]["interval_steps"]
        if ckpt_iv and state.step > 0 and state.step % ckpt_iv == 0:
            save_checkpoint(ckpt_dir / f"step_{state.step}_sub_{k}.pt", model, opt, state, cfg)

        state.step += 1
        if args.debug:
            break

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
