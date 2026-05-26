---
id: T-M3.2
title: Diff 訓練スクリプト (4 sub-model 独立 MSE 訓練、CLI 経由で sub-model 指定)
milestone: M3
phase: M3
status: pending
size: L
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M2.1, T-M3.1]
blocks: [T-M3.5, T-M5.2]
related_docs:
  - docs/milestones.md#m32-training-script-srcwavenext2traintrain_diffpy
  - docs/training.md
  - docs/architecture.md
  - docs/open-questions.md
  - docs/tickets/T-M3.1-diff-model.md
  - docs/tickets/T-M2.5-train-gan.md
  - docs/tickets/T-M2.1-dataset.md
---

# T-M3.2: Diff 訓練スクリプト (4 sub-model 独立 MSE 訓練、CLI 経由で sub-model 指定)

> **マイルストーン**: [M3](../milestones.md#m3-diff-wavenext-2-作業量-large5-サブタスク) / **サブタスク**: [M3.2](../milestones.md#m32-training-script-srcwavenext2traintrain_diffpy)
> **依存**: [T-M2.1](T-M2.1-dataset.md), [T-M3.1](T-M3.1-diff-model.md) / **後続**: [T-M3.5](T-M3.5-diff-smoke.md), [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
Diff-WaveNeXt 2 の **訓練ループ本体** を 1 ファイル (`src/wavenext2/train/train_diff.py`) に閉じ込め、T-M2.1 (Dataset) / T-M3.1 (DiffWaveNext2) で実装した部品を **Adam + 固定 lr + MSE(noise prediction) + band uniform sampling** で接続する。論文 §3.3 / `docs/training.md` §3 が要請する **4 sub-model 独立訓練** を CLI 引数 `--sub-model {1,2,3,4}` で 1 つずつ起動する形態で実装する。

GAN 側 T-M2.5 と異なる点:
- **4 sub-model それぞれに別 checkpoint** (`checkpoints/diff/sub_{1,2,3,4}.pt`)
- 訓練対象は **1 sub-model のみ** (CLI で指定、他 3 sub-model はメモリにロードしない)
- Optimizer は固定 lr (InverseLR scheduler は不採用)
- Loss は **MSE(eps_pred, eps_gt)** のみ (GAN 系 hinge / FM / MR-STFT は使わない)
- 拡散式 `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` を **on-the-fly でバッチ毎に適用** (band 内で noise level を uniform sampling)

本チケットは「実際に loss が finite で 1 step 動く」+「4 sub-model 独立 ckpt が分離保存される」ところまで責務を持ち、`docs/training.md` §3.4 の訓練設定 (lr=2e-4, β=[0.9,0.98], wd=0, grad_clip=1.0, segment_length=25600, batch_size=20, max_steps=1M) を **1 行も曖昧なく** コード化することで、後続 T-M3.5 (smoke) / T-M5.2 (1 epoch) / T-M6.2 (4 sub-model 本格訓練 32h on A100) の **唯一の入口** とする。

T-M2.5 で確立した `train_gan_step(...) -> dict[str, float]` 公開関数化 + `utils/training_loop.py` 共通化 (SIGTERM handler, atomic ckpt rename, worker_init_fn, validation 別 DataLoader) の方針を **完全踏襲** し、`train_diff_step(...)` を同 signature で実装する。

### ゴール
- [ ] `src/wavenext2/train/train_diff.py` に `main()` (click CLI entry point) が実装され `uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml --sub-model {1,2,3,4}` で起動可能
- [ ] `--sub-model` 引数で **1 sub-model 単独訓練** (他 3 sub-model は instantiate しない、メモリ効率 / 訓練独立性のため)
- [ ] **公開関数 `train_diff_step(model, opt, batch, k, cfg) -> dict[str, float]`**: T-M2.5 `train_gan_step` と signature 統一 (1 step backward + loss dict 返却、T-M3.5 smoke / `tests/conftest.py` fixture 共用 / 将来 Lightning 移行で再利用)
- [ ] Optimizer: `Adam(sub_model_k.parameters(), lr=2e-4, betas=[0.9, 0.98], weight_decay=0.0)` (AdamW ではなく Adam、wd=0 は論文準拠)
- [ ] **Scheduler 不使用** (Diff 側は固定 lr。T-M2.5 の `InverseLR` は import しない)
- [ ] Loss: `F.mse_loss(eps_pred, eps_gt)` (noise prediction、Fig 1b)
- [ ] Forward フロー:
  1. `eps ~ N(0, I)` (shape: `(B, segment_length)`)
  2. `c = model.sample_noise_level(k, batch_size)` → shape `(B,)`、`= √(1-ᾱ_t)`、band `[L_k, U_k]` uniform sampling (T-M3.1 §B1)
  3. `abar = 1.0 - c**2` (= `ᾱ_t`)
  4. `x_t = sqrt(abar).view(-1, 1) * x_gt + c.view(-1, 1) * eps` (DDPM 標準形)
  5. `eps_pred = model.sub_models[k-1](mel, x_t, c)` (1-indexed → 0-indexed)
  6. `loss = F.mse_loss(eps_pred, eps)`
- [ ] 4 sub-model それぞれ独立 checkpoint: `checkpoints/diff/sub_{1,2,3,4}.pt`、別々の `--sub-model k` invocation で独立保存
- [ ] Gradient clip: `clip_grad_norm_(max_norm=1.0)` (`docs/training.md` §3.4)
- [ ] EMA: 不使用 (T-M2.5 と同様)
- [ ] Mixed precision: fp32 default、`--amp` で bf16 切替 (fp16 不採用 ← noise level conditioning の sinusoidal embedding underflow リスク、T-M1.5 §6.1)
- [ ] Logging: TensorBoard に `loss`, `lr`, `noise_level_histogram` (band 内 uniform sampling 確認用), `mel`, `x_t`, `eps_pred`, `eps_gt`, `audio_sample` (validation 時のみ)、scalar tag は `sub_{k}/loss` 等で sub-model 別に分離
- [ ] Validation (T-M2.5 から流用): 10k step ごとに 100 utterances (`val.tsv` speaker-balanced) で MSE 計算
- [ ] Checkpoint: 10k step ごと `step_{step}_sub_{k}.pt`、best validation MSE で `sub_{k}.pt` (= `best.pt` 相当) 更新、atomic rename
- [ ] Resume: `--resume <path>` で step / optimizer state / RNG state を完全復元
- [ ] **`utils/training_loop.py` を T-M2.5 から共有 import** (SIGTERM handler / atomic save / iter_forever / log_scalars / run_validation)
- [ ] **`worker_init_fn` = `wavenext2.data.dataset.seed_worker`** (T-M2.1 §9.1 で提供される snippet を直接 import)
- [ ] `tests/test_train_diff.py` で smoke step (1 step 実行) + 100 step loss 単調減少 + band uniform sampling 検証 + 4 sub-model 独立 ckpt 検証 が pass
- [ ] `docs/milestones.md` §M3.2 Acceptance criteria 3 項目クリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規実装 (T-M0.2 で空 stub として配置済み or 未配置):
  - `src/wavenext2/train/train_diff.py` (本実装、`main()` + `train_diff_step()` 公開関数)
  - `tests/test_train_diff.py` (smoke step / 100 step / band sampling / 独立 ckpt の各テスト)
- 編集 (T-M2.5 で生成、本チケットで拡張):
  - `src/wavenext2/utils/training_loop.py` (T-M2.5 §8.1 採用昇格、M2.6 smoke pass 直後に T-M2.5 から refactor 済み想定。本チケットで Diff 側からも import 可能であることを検証)
  - `src/wavenext2/train/__init__.py` (`__all__` に `main_diff`, `train_diff_step` を追加、`from .train_diff import main as main_diff, train_diff_step`)
  - `configs/diff_wavenext2.yaml` (T-M0.2 で雛形生成済、本チケットで `train`/`validation`/`checkpoint`/`logging` セクション値確定)
  - `pyproject.toml` `[project.scripts]` に `train-diff = "wavenext2.train.train_diff:main"` 追加 (任意)
  - `docs/milestones.md` §M3.2 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M3.2 ステータス更新

> **重要 (T-M2.5 §8.1 採用昇格踏襲)**: `train_diff_step(model, opt, batch, k, cfg) -> dict[str, float]` を **公開関数** として切り出す。理由:
> - T-M3.5 smoke が `from train_diff import train_diff_step` で再利用 (`scripts/smoke_diff.py` は薄い wrapper にできる)
> - T-M2.5 `train_gan_step` と signature 統一で `tests/conftest.py` の fixture (model / opt / batch / cfg) を GAN/Diff で共用可能
> - 将来 `pytorch-lightning` 移行時は `LightningModule.training_step` に移すだけで済む (Lightning-ready architecture)

### 2.2 主要構造

#### `train_diff.py` のスケルトン

```python
"""train_diff.py — Diff-WaveNeXt 2 の訓練ループ.

論文 §3.3 / docs/training.md §3 を実装する。
- 4 sub-model 独立訓練 (CLI 引数 `--sub-model {1,2,3,4}` で 1 つずつ指定)
- Optimizer: Adam (lr=2e-4, betas=[0.9, 0.98], weight_decay=0.0)
- Scheduler: なし (固定 lr、`InverseLR` は GAN 側のみ)
- Loss: F.mse_loss(eps_pred, eps_gt) on noise prediction
- Forward: x_t = √ᾱ * x_0 + √(1-ᾱ) * ε (DDPM 標準形、band 内 uniform sampling)
- Grad clip: max_norm=1.0
- EMA: 不使用 (docs/open-questions.md 確定)
- Logging: TensorBoard (step ごとに loss / lr / noise_level histogram、10k step ごとに
  validation + sample audio / x_t / eps_pred 可視化)

公開 API:
- `main(...)`: click CLI entry point (loop / checkpoint / logging を統括)
- `train_diff_step(model, opt, batch, k, cfg) -> dict[str, float]`:
  1 step backward を実行し loss scalar の dict を返す。
  T-M2.5 `train_gan_step` と signature 統一、T-M3.5 smoke / fixture 共用 /
  Lightning 移行で再利用される公開関数。
  返り値 dict のキー: `loss`, `grad_norm`, `c_mean`, `c_std`, `abar_mean`, `eps_norm`,
  `eps_pred_norm`, `x_t_norm`
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wavenext2.data.dataset import LibriTTSRDataset, Batch, seed_worker
from wavenext2.models.diff_wavenext2 import DiffWaveNext2
from wavenext2.utils.config import load_config
from wavenext2.utils.logging import setup_logger
from wavenext2.utils.training_loop import (
    TrainState,
    atomic_save,
    iter_forever,
    log_scalars,
    register_sigterm_handler,
    run_validation,
    save_checkpoint,
    load_checkpoint,
)


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--sub-model", "sub_model_k", type=click.IntRange(1, 4), required=True,
              help="訓練する sub-model index (1..4)")
@click.option("--resume", "resume_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--debug", is_flag=True, help="小規模 smoke (1 batch / 1 step / no validation)")
@click.option("--amp", is_flag=True, help="bf16 mixed precision (default: fp32)")
def main(config_path: str, sub_model_k: int, resume_path: str | None,
         debug: bool, amp: bool) -> None:
    """Diff-WaveNeXt 2 sub-model k の訓練 entry point."""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if amp else torch.float32

    # ===== Model =====
    # 重要: DiffWaveNext2 全体を instantiate するが、訓練対象は sub_models[k-1] のみ
    # 他 3 sub-model は parameter が opt に渡らないため update されない
    # メモリ節約のため `DiffWaveNext2.from_config(cfg["model"], only_sub_model=k)` で
    # 1 sub-model のみ instantiate するオプションを T-M3.1 へ申し送り (§9.1)
    model = DiffWaveNext2.from_config(cfg["model"]).to(device)
    sub_model_k_module = model.sub_models[sub_model_k - 1]  # 1-indexed → 0-indexed

    # ===== Optimizer (k-th sub-model のみ) =====
    opt = Adam(
        sub_model_k_module.parameters(),
        lr=cfg["train"]["optimizer"]["lr"],         # 2e-4
        betas=tuple(cfg["train"]["optimizer"]["betas"]),  # [0.9, 0.98]
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],  # 0.0
    )
    # Scheduler は不使用 (固定 lr)

    # ===== Data =====
    train_loader, val_loader = build_loaders(cfg, sub_model_k)

    # ===== State =====
    state = TrainState(
        step=0,
        best_val_metric=float("inf"),    # MSE 最小化
        rng_state=None,
        sub_model_k=sub_model_k,         # checkpoint ファイル名に埋め込む
    )
    if resume_path:
        state = load_checkpoint(resume_path, model, opt, scheduler=None, state=state)

    # ===== Logging =====
    writer = SummaryWriter(Path(cfg["logging"]["tensorboard_dir"]) / f"sub_{sub_model_k}")

    # ===== Emergency save on SIGTERM/SIGINT (T-M2.5 §8.1 採用) =====
    ckpt_dir = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    register_sigterm_handler(
        lambda: save_checkpoint(
            model, opt, scheduler=None, state=state,
            path=ckpt_dir / f"emergency_step_{state.step}_sub_{sub_model_k}.pt",
            atomic=True,
        )
    )

    # ===== Training loop =====
    model.train()
    for batch in iter_forever(train_loader):
        if state.step >= cfg["train"]["max_steps"]:
            break

        batch = {k_: v.to(device) if torch.is_tensor(v) else v for k_, v in batch.items()}

        # --- 1 step backward (公開関数) ---
        logs = train_diff_step(
            model=model,
            opt=opt,
            batch=batch,
            k=sub_model_k,
            cfg=cfg,
            amp=amp,
            dtype=dtype,
        )

        # --- Logging ---
        if state.step % cfg["logging"]["scalar_interval_steps"] == 0:
            log_scalars(writer, state.step, logs,
                        lr=opt.param_groups[0]["lr"],
                        sub_model_k=sub_model_k)
        # band 内 uniform sampling 検証用: c の値を histogram で記録
        if state.step % cfg["logging"]["histogram_interval_steps"] == 0:
            writer.add_histogram(f"sub_{sub_model_k}/noise_level_c",
                                 logs["_c_batch"], state.step)

        if state.step % 10000 == 0:
            writer.flush()

        # --- Validation ---
        if state.step > 0 and state.step % cfg["validation"]["interval_steps"] == 0:
            val_mse = run_validation(
                model_fn=lambda b: _validation_forward(model, b, sub_model_k, cfg),
                loader=val_loader,
                metric_fn=F.mse_loss,
                device=device,
                writer=writer,
                step=state.step,
                tag_prefix=f"sub_{sub_model_k}/val",
            )
            if val_mse < state.best_val_metric:
                state.best_val_metric = val_mse
                # best ckpt: `sub_{k}.pt` (atomic rename)
                save_checkpoint(
                    model, opt, scheduler=None, state=state,
                    path=ckpt_dir / f"sub_{sub_model_k}.pt",
                    atomic=True,
                )

        # --- Checkpoint ---
        if state.step > 0 and state.step % cfg["checkpoint"]["interval_steps"] == 0:
            save_checkpoint(
                model, opt, scheduler=None, state=state,
                path=ckpt_dir / f"step_{state.step}_sub_{sub_model_k}.pt",
            )

        state.step += 1

        if debug and state.step >= 1:
            break  # smoke

    writer.flush()
    writer.close()


def train_diff_step(
    model: DiffWaveNext2,
    opt: torch.optim.Optimizer,
    batch: dict,
    k: int,
    cfg: dict,
    amp: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """1 step backward を実行し loss scalar の dict を返す.

    T-M2.5 `train_gan_step` と signature 統一の公開関数。
    T-M3.5 smoke / 将来の Lightning 移行で再利用される。

    Args:
        model: DiffWaveNext2 (全 4 sub-model 保持、訓練は k-th のみ)
        opt: Adam (sub_models[k-1].parameters() に紐付け済)
        batch: {"mel": (B, 128, T_mel), "audio": (B, segment_length), ...}
        k: 訓練対象 sub-model index (1..4、1-indexed)
        cfg: YAML config dict
        amp: bf16 mixed precision 切替
        dtype: torch.bfloat16 if amp else torch.float32

    Returns:
        dict with keys:
            - loss (float): MSE(eps_pred, eps)
            - grad_norm (float): clip_grad_norm_ の返り値
            - c_mean, c_std (float): band 内 sampling の統計 (band 内 uniform 確認)
            - abar_mean (float): cumulative noise level の平均
            - eps_norm, eps_pred_norm, x_t_norm (float): 数値安定性監視
            - _c_batch (torch.Tensor): histogram 用 (private、log 直後に破棄)
    """
    assert 1 <= k <= 4, f"sub-model index must be in [1, 4] (got {k})"
    sub_model_k = model.sub_models[k - 1]

    mel = batch["mel"]
    x_gt = batch["audio"]
    B = x_gt.shape[0]
    device = x_gt.device

    # --- 1. eps ~ N(0, I) ---
    eps = torch.randn_like(x_gt)

    # --- 2. band 内 uniform sampling: c = √(1-ᾱ_t) ---
    # T-M3.1 が提供する `sample_noise_level(k, batch_size)` を使用
    # 戻り値は (B,) on CPU、device 転送はここで行う
    c = model.sample_noise_level(k, B).to(device)         # (B,)

    # --- 3. abar = 1 - c**2 = ᾱ_t ---
    abar = 1.0 - c ** 2                                    # (B,)
    sqrt_abar = torch.sqrt(abar)                           # (B,)
    sqrt_one_minus_abar = c                                # (B,) (= √(1-ᾱ))

    # --- 4. x_t = √ᾱ * x_0 + √(1-ᾱ) * ε (DDPM 標準形) ---
    # broadcast: (B,) -> (B, 1)
    x_t = sqrt_abar.unsqueeze(-1) * x_gt + sqrt_one_minus_abar.unsqueeze(-1) * eps

    # --- 5. sub-model forward (autocast 境界) ---
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
        eps_pred = sub_model_k(mel, x_t, c)                # (B, T_audio)

    # --- 6. MSE loss (fp32 で計算、bf16 underflow 回避) ---
    with torch.autocast(device_type=device.type, enabled=False):
        loss = F.mse_loss(eps_pred.float(), eps.float())

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        sub_model_k.parameters(),
        max_norm=cfg["train"]["grad_clip_norm"],
    )
    opt.step()

    return {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "c_mean": c.mean().item(),
        "c_std": c.std().item() if B > 1 else 0.0,
        "abar_mean": abar.mean().item(),
        "eps_norm": eps.norm().item(),
        "eps_pred_norm": eps_pred.detach().norm().item(),
        "x_t_norm": x_t.detach().norm().item(),
        "_c_batch": c.detach().cpu(),   # histogram log 用 (consumer 側で pop)
    }


def _validation_forward(
    model: DiffWaveNext2,
    batch: dict,
    k: int,
    cfg: dict,
) -> torch.Tensor:
    """Validation 時の 1 step forward (MSE のみ計算、parameter 更新なし).

    Train 時の sampling と異なり、validation では **band の中点** で
    deterministic に c を固定する (品質 fluctuation 除外、§6 通常項目)。
    """
    sub_model_k = model.sub_models[k - 1]
    mel = batch["mel"]
    x_gt = batch["audio"]
    B = x_gt.shape[0]
    device = x_gt.device

    # validation は band 中点で deterministic
    L, U = model.BAND_BOUNDS[k - 1]
    c_val = torch.full((B,), (L + U) / 2.0, device=device, dtype=x_gt.dtype)
    abar = 1.0 - c_val ** 2
    eps = torch.randn_like(x_gt)   # validation 用 seed は run_validation 側で固定
    x_t = torch.sqrt(abar).unsqueeze(-1) * x_gt + c_val.unsqueeze(-1) * eps
    eps_pred = sub_model_k(mel, x_t, c_val)
    return F.mse_loss(eps_pred, eps)


def build_loaders(cfg: dict, sub_model_k: int) -> tuple[DataLoader, DataLoader]:
    """train/val DataLoader を構築 (T-M2.1 LibriTTSRDataset.from_config を使用).

    Diff 側は segment_length=25600, hop_length=256 を config 経由で渡す。
    T-M2.1 §9.1 `seed_worker` を `worker_init_fn` に渡す。
    """
    train_dataset = LibriTTSRDataset.from_config(cfg["data_train"], mode="train")
    val_dataset   = LibriTTSRDataset.from_config(cfg["data_val"],   mode="val", seed=42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],     # 20 (FastDiff default)
        num_workers=cfg["train"]["num_workers"],   # 8
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,                # T-M2.1 §9.1
        prefetch_factor=cfg["train"].get("prefetch_factor", 2),
        persistent_workers=True,
        multiprocessing_context=cfg["train"].get("multiprocessing_context", "spawn"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False,
        drop_last=False,
        persistent_workers=False,  # validation 用は leak 防止のため毎回 close
    )
    return train_loader, val_loader
```

#### `utils/training_loop.py` の共通 API (T-M2.5 §8.1 で M2.6 smoke pass 直後に切り出し済前提)

本チケットで Diff 側からも import する関数群 (T-M2.5 §9.1 「T-M3.2 へ」で申し送られた共通化):

```python
# src/wavenext2/utils/training_loop.py (T-M2.5 §8.1 で先行実装)

from dataclasses import dataclass, field
from pathlib import Path
import os
import signal
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainState:
    """訓練ループの状態 (GAN/Diff 共用).

    Diff 側は `sub_model_k` を追加情報として保持。
    """
    step: int = 0
    best_val_metric: float = float("inf")
    rng_state: dict | None = None
    sub_model_k: int | None = None   # Diff 用 (GAN では None)

    # D 強すぎ問題の検知 (GAN 側のみ使用、Diff では参照されない)
    d_loss_below_threshold_steps: int = 0
    _d_loss_history: list[float] = field(default_factory=list)

    def update_d_loss_history(self, loss_d: float, threshold: float = 0.01) -> None:
        """GAN 専用: D loss が threshold を下回り続けた step 数をカウント."""
        ...


def iter_forever(loader: DataLoader):
    """Infinite iterator over DataLoader (epoch 境界で StopIteration を吸収).

    `persistent_workers=True` の DataLoader で epoch 間 worker 再起動コスト回避。
    """
    while True:
        for batch in loader:
            yield batch


def register_sigterm_handler(save_fn: Callable[[], None]) -> None:
    """SIGTERM / SIGINT 受信時に emergency save を呼ぶ.

    M6 cluster preemption (Slurm / Kubernetes) 対策。
    """
    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        save_fn()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def atomic_save(obj: dict, path: Path) -> None:
    """`path.tmp` に保存後 `os.replace` で原子的に rename.

    途中で kill されても `path` 自体が中途半端な状態にならないことを保証。
    T-M2.4 申し送り (best.pt atomic rename) の generic 化。
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    state: TrainState,
    path: Path,
    atomic: bool = False,
) -> None:
    """checkpoint を保存 (atomic オプション付き)."""
    ckpt = {
        "step": state.step,
        "best_val_metric": state.best_val_metric,
        "sub_model_k": state.sub_model_k,
        "model_state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    if atomic:
        atomic_save(ckpt, path)
    else:
        torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    state: TrainState,
) -> TrainState:
    """checkpoint を復元 (step / opt / sch / RNG state)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["opt_state_dict"])
    if scheduler is not None and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    state.step = ckpt["step"]
    state.best_val_metric = ckpt["best_val_metric"]
    state.sub_model_k = ckpt.get("sub_model_k")
    torch.set_rng_state(ckpt["rng_state"]["torch"])
    if torch.cuda.is_available() and ckpt["rng_state"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
    np.random.set_state(ckpt["rng_state"]["numpy"])
    return state


def log_scalars(writer: SummaryWriter, step: int, logs: dict,
                lr: float | None = None, sub_model_k: int | None = None,
                **extra) -> None:
    """logs dict を TensorBoard へ scalar で書き出す.

    `_` 始まりキー (例: `_c_batch`) は histogram 用 raw tensor のためスキップ。
    """
    tag_prefix = f"sub_{sub_model_k}/" if sub_model_k is not None else ""
    for k, v in logs.items():
        if k.startswith("_"):
            continue
        writer.add_scalar(f"{tag_prefix}{k}", v, step)
    if lr is not None:
        writer.add_scalar(f"{tag_prefix}lr", lr, step)
    for k, v in extra.items():
        writer.add_scalar(f"{tag_prefix}{k}", v, step)


def run_validation(
    model_fn: Callable[[dict], torch.Tensor],
    loader: DataLoader,
    metric_fn: Callable | None,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    tag_prefix: str = "val",
) -> float:
    """100 utterances で validation metric を計算し平均を返す.

    model_fn(batch) は metric tensor (scalar) を返すクロージャ。
    GAN/Diff 共用、内部で `torch.no_grad` + `model.eval()` 相当の管理は呼び出し側。
    """
    ...  # 実装は T-M2.5 で行う
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `lr` | 2.0e-4 | docs/training.md §3.4 (FastDiff 慣例) |
| `betas` | [0.9, 0.98] | docs/training.md §3.4 |
| `weight_decay` | 0.0 | docs/training.md §3.4 (Diff 側は wd=0) |
| `grad_clip_norm` | 1.0 | docs/training.md §3.4 |
| `max_steps` | 1,000,000 | docs/training.md §3.4 (FastDiff `max_updates`) |
| `batch_size` | 20 | docs/training.md §3.4 (FastDiff default) |
| `num_workers` | 8 | docs/implementation-plan.md §5 |
| `segment_length` | 25600 | docs/training.md §3.4, T-M2.1 で確定 (Diff 側はぴったり整合) |
| `hop_length` | 256 | docs/training.md §1.2 |
| EMA | 不使用 | docs/open-questions.md (確定) |
| scheduler | **不使用** (固定 lr) | docs/training.md §3.4 (FastDiff 慣例、InverseLR は GAN のみ) |
| validation.interval_steps | 10000 | T-M2.5 と同期 |
| validation.num_utterances | 100 | docs/training.md §6 |
| validation.metric | MSE (deterministic c at band 中点) | 本チケットで決定 (§6.2) |
| checkpoint.interval_steps | 10000 | validation と同期 |
| amp dtype | bf16 (`--amp`) / fp32 (default) | T-M1.5 §6.1 (fp16 sinusoidal underflow 回避) |
| `BAND_BOUNDS[k-1]` | T-M3.1 で確定 | docs/architecture.md §5, docs/open-questions.md §B1 |
| `NOISE_SCHEDULE_ABAR` | `[1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` | docs/training.md §3.3 |

### 2.4 アルゴリズム / 処理フロー

#### 1 step backward (擬似コード)

```
batch = next(train_iter)
mel, x_gt = batch["mel"], batch["audio"]    # (B, 128, T_mel), (B, segment_length)

# 1. eps ~ N(0, I)
eps = torch.randn_like(x_gt)

# 2. band 内 uniform sampling: c = √(1-ᾱ_t)
c = model.sample_noise_level(k, B)           # (B,) on CPU → device
# k-th band [L_k, U_k] (T-M3.1) で uniform

# 3. ᾱ_t, √ᾱ, √(1-ᾱ) を計算
abar = 1.0 - c ** 2                           # (B,)
sqrt_abar = torch.sqrt(abar)                  # (B,)
sqrt_one_minus_abar = c                       # = √(1-ᾱ_t)

# 4. DDPM 標準形 x_t を生成
x_t = sqrt_abar.unsqueeze(-1) * x_gt \
    + sqrt_one_minus_abar.unsqueeze(-1) * eps  # (B, segment_length)

# 5. sub-model forward
eps_pred = model.sub_models[k-1](mel, x_t, c)  # (B, segment_length)

# 6. MSE loss
loss = F.mse_loss(eps_pred, eps)
loss.backward()
clip_grad_norm_(sub_models[k-1].params, 1.0)
opt.step()

# 7. Logging
writer.add_scalar(f"sub_{k}/loss", loss.item(), step)
writer.add_scalar(f"sub_{k}/c_mean", c.mean().item(), step)
writer.add_histogram(f"sub_{k}/noise_level_c", c, step)
```

#### Validation (10k step ごと、deterministic c at band 中点)

1. `model.eval()` / `torch.no_grad()` で val_loader (100 utterances) を走査
2. 各 utterance で:
   - `c_val = (L_k + U_k) / 2.0` (band 中点固定、validation seed 固定)
   - `eps_val = torch.randn_like(audio)` (validation seed 固定で deterministic)
   - `x_t_val = sqrt(1 - c_val^2) * audio + c_val * eps_val`
   - `eps_pred_val = sub_models[k-1](mel, x_t_val, c_val)`
   - `mse = F.mse_loss(eps_pred_val, eps_val)`
3. 100 utterance の MSE 平均で best 判定
4. `best_val_metric` 更新時に `sub_{k}.pt` を atomic save
5. 最初の 4 utterance で `add_audio(x_gt)`, `add_audio(x_t)`, `add_image(mel_visualization)`, `add_histogram(eps_pred)` を TensorBoard 出力

#### Checkpoint state (resume 完全復元)

GAN 側 T-M2.5 と同じ key を共有 (`utils/training_loop.py` で統一済):
- `step`, `best_val_metric`, `sub_model_k`, `model_state_dict`, `opt_state_dict`, `scheduler_state_dict` (=None), `rng_state` (torch / cuda / numpy)

#### 4 sub-model 独立訓練の起動シーケンス (T-M6.2 用)

```bash
# 順次起動 (single GPU の場合)
for k in 1 2 3 4; do
    uv run python -m wavenext2.train.train_diff \
        --config configs/diff_wavenext2.yaml \
        --sub-model $k \
        --amp
done

# multi-GPU 並列 (T-M6.2 / §8.1 採用検討)
CUDA_VISIBLE_DEVICES=0 train-diff --sub-model 1 --amp &
CUDA_VISIBLE_DEVICES=1 train-diff --sub-model 2 --amp &
CUDA_VISIBLE_DEVICES=2 train-diff --sub-model 3 --amp &
CUDA_VISIBLE_DEVICES=3 train-diff --sub-model 4 --amp &
wait
```

### 2.5 CLI / config 連携

`configs/diff_wavenext2.yaml` に追加するキー (T-M0.2 雛形 + 本チケットで確定):

```yaml
model:
  type: diff_wavenext2
  sub_model_cfg:
    mel_channels: 128
    n_fft: 1024
    hop: 256
    win_length: 1024
    dim: 512
    intermediate_dim: 1536
    n_blocks: 8
    conditioning_dim: 512

data_train:
  filelist: data/filelists/train.tsv
  root_dir: ${LIBRITTSR_ROOT}
  segment_length: 25600
  hop_length: 256
  mel:
    sample_rate: 24000
    n_fft: 1024
    hop_length: 256
    win_length: 1024
    n_mels: 128
    f_min: 20.0
    f_max: 12000.0
    eps: 1.0e-5

data_val:
  filelist: data/filelists/val.tsv
  root_dir: ${LIBRITTSR_ROOT}
  segment_length: 25600
  hop_length: 256
  mel: { /* same as data_train.mel */ }

train:
  max_steps: 1_000_000
  batch_size: 20
  num_workers: 8
  prefetch_factor: 2
  multiprocessing_context: spawn
  grad_clip_norm: 1.0
  optimizer:
    type: Adam
    lr: 2.0e-4
    betas: [0.9, 0.98]
    weight_decay: 0.0
  amp:
    enabled: false
    dtype: bfloat16

validation:
  interval_steps: 10000
  num_utterances: 100
  metric: mse
  c_at_band_midpoint: true   # 本チケット §6.2 で決定
  num_audio_samples: 4

checkpoint:
  dir: checkpoints/diff
  interval_steps: 10000
  keep_last_n: 5
  best_filename_pattern: sub_{k}.pt   # {k} は --sub-model で置換

logging:
  tensorboard_dir: logs/diff
  scalar_interval_steps: 100
  histogram_interval_steps: 500       # noise_level_c の histogram を 500 step ごと
```

CLI:
```bash
# 通常訓練 (sub-model 1)
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml --sub-model 1

# Resume
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml \
    --sub-model 1 --resume checkpoints/diff/sub_1.pt

# Smoke (1 step だけ)
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml \
    --sub-model 1 --debug

# bf16 mixed precision
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml \
    --sub-model 1 --amp
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer (train) | 1 | `train_diff.py` の訓練ループ本体 + `train_diff_step` + validation forward + checkpoint/resume + logging | general-purpose |
| Implementer (training_loop) | 1 | T-M2.5 で先行実装した `utils/training_loop.py` の Diff 用拡張 (`sub_model_k` field 追加 / `TrainState.best_val_metric` の汎用化) | general-purpose |
| Reviewer | 1 | docs/training.md §3 とコード対応確認 + T-M2.5 との API symmetry 確認 + FastDiff 非コピー確認 + DDPM 標準形 (x_t 式) の数式整合性確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_train_diff.py -v` + smoke (`--debug` で 1 step) + 100 step loss curve + 4 sub-model 独立 ckpt 動作確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M2.1 / T-M3.1 完了 + T-M2.5 で `utils/training_loop.py` 切り出し済が前提)
- 並列実行する場合の最大並列数: 1
- 後続 T-M3.5 (smoke) / T-M5.2 (1 epoch) / T-M6.2 (本格訓練) は本チケット完了後にそれぞれ別タスクとして起動

## 4. 提供範囲 (Scope)

### In Scope
- `train_diff.py`: `main()` (click CLI、`--sub-model` 必須) + `train_diff_step` (1 step backward 公開関数) + `_validation_forward` + `build_loaders`
- `tests/test_train_diff.py`: smoke step / 100 step loss / band sampling / 4 sub-model 独立 ckpt の各テスト
- `configs/diff_wavenext2.yaml` への `model` / `data_train` / `data_val` / `train` / `validation` / `checkpoint` / `logging` セクション値確定
- docstring (英文 + 日本語混在、`docs/training.md` §3.4 を明示)
- `pyproject.toml` の `[project.scripts]` 追加 (任意)

### Out of Scope
- **DiffWaveNext2 / sub-model / noise embedding の実装** (T-M3.1 / T-M1.6 / T-M1.5 で完成済前提)
- **Dataset / sox norm / 反射 pad の実装** (T-M2.1 で完成済前提)
- **`utils/training_loop.py` の新規実装** (T-M2.5 §8.1 採用昇格で M2.6 smoke pass 直後に refactor 済、本チケットでは Diff 側用 field 追加のみ)
- **Reverse sampler (推論)** (T-M3.3)
- **Post-filter** (T-M3.4)
- **1000 step smoke training** (T-M3.5)
- **1 epoch 訓練** (T-M5.2)
- **4 sub-model 本格訓練 32h** (T-M6.2)
- **BDDM noise schedule predictor** (再現不要、固定 4 値直接使用、docs/open-questions.md 確定)
- **Multi-GPU / DDP** (single GPU 前提、M6 で必要なら別 PR)
- **fp16 mixed precision** (sinusoidal embedding underflow リスクのため bf16 のみ)
- **EMA** (docs/open-questions.md 確定)
- **`accelerate` / `pytorch-lightning` 採用** (素 PyTorch + click)
- **wandb** (TensorBoard のみ。M5.1 phase review で再評価)
- **GAN 訓練スクリプト** (T-M2.5 で完成済前提)
- **`InverseLR` の Diff 側採用** (固定 lr、scheduler 不使用)
- **v-prediction parameterization** (eps-prediction 採用、§8.1 代替案)
- **L1 loss (MAE) 代替** (MSE 採用、§8.1 代替案)
- **Loss normalization (c で割る)** (§6.1 懸念事項、再評価トリガーは M5.2 / M6.2 発散時)

### Deliverable
- ファイル:
  - `src/wavenext2/train/train_diff.py` (新規実装)
  - `tests/test_train_diff.py` (新規実装)
  - `src/wavenext2/train/__init__.py` (re-export `main as main_diff`, `train_diff_step` 追加)
- 編集:
  - `src/wavenext2/utils/training_loop.py` (`TrainState` に `sub_model_k` field 追加、Diff 用 `best_val_metric` の generic 化)
  - `configs/diff_wavenext2.yaml` (model / data / train / validation / checkpoint / logging セクション値確定)
- 関数 / クラス:
  - `def main(config_path, sub_model_k, resume_path, debug, amp) -> None` (click CLI entry point)
  - `def train_diff_step(model, opt, batch, k, cfg, amp, dtype) -> dict[str, float]` (公開関数、T-M3.5 / 将来 Lightning で再利用)
  - 補助関数: `_validation_forward`, `build_loaders`
  - `TrainState.sub_model_k: int | None` field 追加
- ドキュメント差分:
  - `docs/milestones.md` §M3.2 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M3.2 ステータス更新
  - (該当時) `docs/training.md` §3 にコード参照リンク (`src/wavenext2/train/train_diff.py:train_diff_step`) を追記

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_train_diff.py`)

#### 公開関数 API 検証
- [ ] `test_train_diff_step_smoke`: **`train_diff_step` を直接呼ぶ smoke** で 1 step backward が動き、戻り値 dict のキー (`loss`, `grad_norm`, `c_mean`, `c_std`, `abar_mean`, `eps_norm`, `eps_pred_norm`, `x_t_norm`) が揃っていて全 value が `isfinite` (Acceptance #4 本チケット§5.5)
- [ ] `test_train_diff_step_signature_matches_gan`: `inspect.signature(train_diff_step)` の必須引数 `model, opt, batch, k, cfg` が T-M2.5 `train_gan_step` の `G, D, opt_G, opt_D, sch_G, sch_D, mel, audio, cfg, ...` と **同階層の dict-based / cfg 渡し** で signature 統一されている (T-M2.5 §8.1 採用昇格との整合)
- [ ] `test_train_diff_step_returns_finite_loss`: 1 step 実行 → `logs["loss"] > 0` かつ `isfinite`

#### Loss 単調減少 (Acceptance #1)
- [ ] `test_train_diff_step_loss_decreasing_100step`: 1 sub-model を 1 utterance over-fit で 100 step 訓練 → 初期 loss と 100 step 後 loss を比較し `loss[99] < loss[0] * 0.5` (単調減少、milestones.md §M3.2 Acceptance)
- [ ] `test_train_diff_step_loss_decreasing_monotonic_trend`: 10-step moving average が `step=[0..10], [40..50], [90..100]` の 3 区間で **狭義単調減少** (短期ノイズ吸収)

#### Band uniform sampling (Acceptance #2)
- [ ] `test_noise_level_band_uniform_k1`: sub-model 1 で 1000 batch sampling 後の `c` の min/max/mean が `[0.9929, 1.0]` band 内、平均 `≈ (0.9929 + 1.0) / 2 = 0.99645 ± 0.01`
- [ ] `test_noise_level_band_uniform_k2`: 同様に sub-model 2 で `[0.8246, 0.9929)`, 平均 `≈ 0.9088`
- [ ] `test_noise_level_band_uniform_k3`: `[0.4817, 0.8246)`, 平均 `≈ 0.6532`
- [ ] `test_noise_level_band_uniform_k4`: `[0.0, 0.4817)`, 平均 `≈ 0.2409`
- [ ] `test_noise_level_histogram_logged`: TensorBoard event file 内に `sub_{k}/noise_level_c` histogram tag が存在 (Acceptance #2 milestones.md §M3.2 「TensorBoard histogram で確認」)

#### 4 sub-model 独立 checkpoint (Acceptance #3)
- [ ] `test_independent_checkpoint_per_sub_model`: `--sub-model 1` と `--sub-model 2` を別々に起動し、`checkpoints/diff/sub_1.pt` / `sub_2.pt` が **別 file** として保存され、`torch.load` で読んだ `state.sub_model_k` が それぞれ 1, 2 になっている
- [ ] `test_only_target_sub_model_updates`: sub-model 1 のみ 5 step 訓練後、`model.sub_models[0].state_dict()` は **変化**、`model.sub_models[1..3].state_dict()` は **不変** (parameter freezing 確認: opt が sub_models[0] のみに紐付いている検証)
- [ ] `test_checkpoint_filename_pattern`: `cfg["checkpoint"]["best_filename_pattern"] = "sub_{k}.pt"` の `{k}` が `--sub-model` で正しく置換される

#### DDPM 拡散式 (Forward フロー)
- [ ] `test_diffusion_forward_x_t_formula`: 固定 `eps`, `c`, `x_0` で `x_t = sqrt(1-c^2) * x_0 + c * eps` の値が手計算と一致 (DDPM 標準形 PDF Fig 1b)
- [ ] `test_eps_pred_shape`: `eps_pred.shape == eps.shape == x_gt.shape == (B, segment_length=25600)`
- [ ] `test_abar_in_unit_range`: `c.min() >= 0.0` かつ `c.max() <= 1.0`、`abar.min() >= 0.0` かつ `abar.max() <= 1.0`

#### Optimizer / Loss 設定
- [ ] `test_optimizer_is_adam_not_adamw`: `type(opt) is torch.optim.Adam` (AdamW ではない)
- [ ] `test_optimizer_lr_2e_4`: `opt.param_groups[0]["lr"] == 2e-4`
- [ ] `test_optimizer_betas_0_9_0_98`: `opt.param_groups[0]["betas"] == (0.9, 0.98)`
- [ ] `test_optimizer_weight_decay_zero`: `opt.param_groups[0]["weight_decay"] == 0.0`
- [ ] `test_no_scheduler`: `main()` 内に `LRScheduler` の instance が存在しない (Diff は固定 lr)
- [ ] `test_loss_is_mse`: `train_diff_step` の loss が `F.mse_loss(eps_pred, eps)` (MAE / Huber でない)

#### Grad clip / EMA
- [ ] `test_grad_clip_applied`: `clip_grad_norm_` が呼ばれた直後の grad norm が `<= 1.0 + ε`、戻り値 `grad_norm` が clip 前 norm
- [ ] `test_no_ema`: model 内に EMA shadow parameter が存在しない

#### Checkpoint / Resume
- [ ] `test_checkpoint_save_load_roundtrip`: 1 step → save → 新 model + load → step / opt state / RNG state / model state が完全一致
- [ ] `test_atomic_save_rename`: save 中に kill しても `sub_{k}.pt` が中途半端な状態にならない (`sub_{k}.pt.tmp` → `os.replace`)
- [ ] `test_resume_continues_step`: `--resume` で `state.step` がチェックポイントから再開、optimizer の momentum buffer も復元

#### TensorBoard logging (Acceptance #2 / #3)
- [ ] `test_tensorboard_event_written`: 1 step 後に `logs/diff/sub_{k}/events.out.tfevents.*` が生成され、`sub_{k}/loss`, `sub_{k}/lr`, `sub_{k}/c_mean`, `sub_{k}/grad_norm` の scalar が存在
- [ ] `test_tensorboard_histogram_noise_level`: 500 step 後に `sub_{k}/noise_level_c` histogram tag が存在 (Acceptance #2)
- [ ] `test_validation_audio_logged`: validation 実行後に `sub_{k}/val/audio_*`, `sub_{k}/val/mel_*` tag が存在

#### CLI / config
- [ ] `test_cli_sub_model_required`: `--sub-model` 引数なしで起動 → click が `Missing option` で fail
- [ ] `test_cli_sub_model_range_validation`: `--sub-model 5` で起動 → click `IntRange(1, 4)` で fail
- [ ] `test_cli_sub_model_range_validation_zero`: `--sub-model 0` で起動 → 同上
- [ ] `test_amp_bf16`: `--amp` 指定時に `eps_pred.dtype in (torch.float32, torch.bfloat16)`
- [ ] `test_amp_autocast_boundary`: **MSE loss が fp32 で計算される** ことを assert (`--amp` 有効でも `loss.dtype == torch.float32`、§6.1 通常項目「mixed precision」)

#### Seed determinism / Worker init
- [ ] `test_seed_determinism`: 同 seed + 同 config + 同 batch で 1 step 後の `sub_models[k-1].state_dict()` が deterministic
- [ ] `test_seed_worker_imported_from_dataset`: `from wavenext2.data.dataset import seed_worker` で import 可能 (T-M2.1 §9.1 から流用)
- [ ] `test_worker_init_fn_isolates_rng`: DataLoader を 2 worker で起動して、各 worker の `np.random.rand()` が異なることを確認 (T-M2.1 §6.1 通常項目 worker seed 分散)

#### SIGTERM / preemption
- [ ] `test_sigterm_emergency_save`: `os.kill(os.getpid(), SIGTERM)` を別スレッドから送信し、`emergency_step_N_sub_{k}.pt` が生成されることを確認

### 5.2 e2e / 結合テスト
- [ ] `test_real_audio` (T-M0.3 完了後、`@pytest.mark.slow` で skip 可): LibriTTS-R 1 utterance で sub-model 1 を 5 step 実行、loss が `isfinite`、`x_t` が `[-segment_length^0.5, segment_length^0.5]` の数値範囲
- [ ] T-M3.5 (smoke training 1000 step) の入口として `main()` が CLI から呼べる (`subprocess.run(["uv", "run", "python", "-m", "wavenext2.train.train_diff", "--debug", "--config", tmp_cfg, "--sub-model", "1"])`)
- [ ] T-M5.2 (1 epoch) の準備として `train_loader` が StopIteration せず 7k step 回せる (`iter_forever` の正常動作)

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.2 より転記)
- [ ] sub-model 1 を 100 step 訓練して loss が単調減少
- [ ] noise level が band 内で uniform sampling されていることを TensorBoard で確認
- [ ] 4 つの sub-model それぞれ独立に checkpoint 保存

### 5.4 追加 acceptance (本チケット独自)
- [ ] **`train_diff_step` を直接呼ぶ smoke で 1 step backward が動く** (Acceptance #4、`tests/test_train_diff.py::test_train_diff_step_smoke`)
- [ ] CLI の `--sub-model {1..4}` 範囲外 / 欠落でエラー (`tests/test_train_diff.py::test_cli_sub_model_*`)
- [ ] `uv run pytest tests/test_train_diff.py -v` が **exit code 0** で完了
- [ ] `utils/training_loop.py` の API (`TrainState`, `iter_forever`, `register_sigterm_handler`, `atomic_save`, `save_checkpoint`, `load_checkpoint`, `log_scalars`, `run_validation`) が GAN/Diff 双方から import 可能

### 5.5 テスト戦略

- **GPU テスト分離**: `@pytest.mark.gpu` で GPU 必須テストを CI runner 別に分離。`test_amp_bf16` は GPU マーカー必須
- **`@pytest.mark.slow` マーカー**: `test_real_audio` 等 LibriTTS-R 実音声を使うテストは slow でデフォルト除外
- **CI 時間目標**: `tests/test_train_diff.py` 全体 **40 秒以内** (`test_train_diff_step_loss_decreasing_100step` が 100 step × forward+backward で支配的、batch_size=1, T_audio=4096 で最小化)
  - **必須最適化**: `scope="module"` fixture で `model`, `opt`, `batch` を再利用
  - **batch_size=1, segment_length=4096 で最小化**: smoke では segment_length=25600 の 1/6 でメモリ・時間削減
- **fixture 共用 (T-M2.5 と連携)**: `tests/conftest.py` の `gan_smoke_batch` を流用、`diff_smoke_batch` で segment_length=4096 版を追加。`model` fixture は GAN/Diff で別 (型が違うため)
- **coverage 目標**: 本チケットのカバレッジ目標 **80%** (Resume の異常系 / config drift warning は手動テスト)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (実装着手前に必ず解決)

- **CRITICAL: T-M3.1 `sample_noise_level(k, batch_size)` の戻り値 device / dtype 仕様**:
  - 本チケットは `c = model.sample_noise_level(k, B).to(device)` で device 転送を行うが、T-M3.1 が **GPU 上で生成して返す** 仕様なら無駄な転送が発生
  - **本チケット実装着手時の MUST DO**: T-M3.1 §9.1 / API doc を確認し、戻り値 device 仕様を pin
  - 提案: T-M3.1 は `sample_noise_level(k, B, device=None) -> torch.Tensor (B,) float32` の signature を採用 (本チケット §9.1 で T-M3.1 へ申し送り)
  - **未解決のままだと**: 性能劣化 (microbenchmark で確認)、または dtype mismatch で sub_model forward が落ちる
  - 検知: `tests/test_train_diff.py::test_noise_level_band_uniform_k*` で `c.device.type == "cuda"` を assert

- **CRITICAL: `train_diff_step` の MSE loss scale 依存** (本チケット §6.1 / 仕様の曖昧さで再評価):
  - DDPM 標準形 `x_t = √ᾱ * x_0 + √(1-ᾱ) * ε` で `c = √(1-ᾱ)` が小さい (sub-model 1: `c ≈ 1.0`、sub-model 4: `c ≈ 0.3`)
  - `eps` 自体は `N(0, 1)` だが、sub-model 4 では `x_t` のほとんどが `x_0` 由来となり、small noise を予測する難しさが loss 値に反映されにくい (= loss が小さくなりがち)
  - **対応**: 本チケットでは **normalization なし** で MSE をそのまま使用 (FastDiff 慣例)。再評価トリガーは M5.2 smoke で sub-model 4 が学習しない場合
  - **緩和案** (§8.1): `loss = F.mse_loss(eps_pred, eps) / (c ** 2 + ε)` の scale 補正、または v-prediction parameterization
  - **検知**: TensorBoard で sub_model 1〜4 の loss curve を並べて、sub-model 4 だけが極端に flat なら警告

- **CRITICAL: `DiffWaveNext2` 全体 instantiate で OOM** (T-M3.1 §9.1 で要請):
  - 4 sub-model × 14.42M = 57.68M params + activation 4 倍 で 24GB GPU でも OOM 可能性
  - **緩和**: T-M3.1 に `from_config(cfg, only_sub_model: int | None = None)` の追加を要請 (§9.1)
  - **当面の workaround**: 全 4 sub-model instantiate するが、3 個は `.eval()` + `requires_grad_(False)` で activation も保持しない設計 (但し forward を呼ばないため activation は実質生成されない)
  - **検知**: `test_train_diff_step_smoke` で `torch.cuda.max_memory_allocated()` を pin (例: T_audio=4096 batch=1 fp32 で < 4GB)

#### 通常項目

- **mixed precision (bf16) で MSE が underflow するリスク**:
  - `(eps_pred - eps)^2` の差が小さい (≈ 1e-3 オーダー) と bf16 精度では 0 に丸まる
  - **対応**: MSE 計算は **fp32 で実行** (autocast 境界で float32 戻し、§2.2 `train_diff_step` 実装参照)
  - **検証**: `test_amp_autocast_boundary` で `--amp` 時も `loss.dtype == torch.float32`

- **`sample_noise_level` のグローバル RNG 汚染**:
  - T-M3.1 が `torch.empty(B).uniform_(L, U)` でグローバル RNG を消費すると `worker_init_fn` の効果が打ち消される
  - **対応**: T-M3.1 §9.1 で `sample_noise_level(k, B, generator: torch.Generator | None = None)` を採用するよう申し送り
  - **本チケット側**: `model.sample_noise_level(k, B)` 呼び出し時に generator を渡せる設計を予約 (但し本チケットでは generator=None で実装)

- **`y_prev` でなく `x_t` 入力の T-M3.1 SubModelDiff インターフェース**:
  - T-M3.1 `SubModelDiff.forward(mel, x_t, c)` の signature 確認必須 (GAN 側は `forward(mel, y_prev)`)
  - **対応**: `train_diff_step` は `eps_pred = sub_model_k(mel, x_t, c)` で呼ぶ、T-M3.1 と signature 合致を tests/test_sub_model.py で確認済前提
  - **検証**: `test_eps_pred_shape` で shape `(B, segment_length)` を確認

- **Validation の `c` 固定値の妥当性**:
  - 本チケットは validation で band 中点 `c_val = (L_k + U_k) / 2.0` を使用
  - **代替案**: band 内ランダム + validation seed 固定でも deterministic 化可能 (実装は単純)
  - **採用根拠**: band 中点は schedule 点と離れている可能性があるが、訓練分布の中心なので representative
  - **再評価**: M5.2 smoke で validation MSE と推論時の品質の相関が薄ければ band 中点ではなく **schedule 点固定** (`c_val = SCHEDULE_C[k-1]`) に変更

- **`iter_forever` + `persistent_workers=True` の seed 罠 (T-M2.1 §6.1 から伝搬)**:
  - DataLoader が worker 再起動なしに新 epoch に入ると random crop seed が更新されず **同じ crop が繰り返される** 可能性
  - **対応**: T-M2.1 §9.1 `seed_worker` を `worker_init_fn` に渡す (本チケットで import)
  - **検証**: `test_worker_init_fn_isolates_rng` で 2 worker の seed が異なる

- **TensorBoard `add_histogram` の容量爆発**:
  - 500 step ごと × 1M step = 2000 histogram × 各 20 個 bin × 数 KB = 数十 MB / sub-model × 4 = 数 GB
  - **対応**: `histogram_interval_steps=500` (default、本チケットで決定)。1M step で 2000 histogram は許容範囲
  - **代替**: `histogram_interval_steps=10000` (validation と同期) に下げても sampling 検証は十分

- **`prefetch_factor` × `persistent_workers=True` の validation leak (T-M2.5 §6.1 と同じ罠)**:
  - validation の DataLoader を train とは別 instance にして `persistent_workers=False` (`build_loaders` で実装)
  - **検証**: validation 完了後に GPU memory が train batch 1 個分以下に戻ることを `torch.cuda.memory_allocated` で観測

- **disk full 監視**:
  - `checkpoint.keep_last_n=5` でも `logs/diff/sub_{k}/events.out.tfevents.*` が 1M step × 4 sub-model で数十 GB
  - **対応**: `writer.flush()` を 10k step ごと明示、`max_queue=1000` に縮小 (T-M2.5 と同様の方針)

- **D 強すぎ問題は Diff では発生しない** (`docs/training.md` §3 で Diff は MSE のみ、Discriminator なし):
  - GAN 側 T-M2.5 §6.1 の `loss_D < 0.01` 検知ロジックは Diff には不要、`utils/training_loop.py` で GAN 専用に保持
  - **本チケット側**: `TrainState.d_loss_below_threshold_steps` は GAN 側のみ更新、Diff では参照しない

- **再現性 (`torch.use_deterministic_algorithms`)**:
  - T-M2.5 と同じく `--deterministic` フラグで `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True)` を提供 (default off、ablation 時のみ on)

- **Loss curve の log scale 表示**:
  - MSE は `1e-3` 〜 `1e-1` の range で動く可能性が高い、線形 scale だと初期の急減が読めない
  - **対応**: TensorBoard で `loss` を log scale 表示 (configure 側、コード変更不要)

- **`sample_noise_level` で BAND_BOUNDS が境界を包含するかの仕様**:
  - 例: sub-model 1 の band `[0.9929, 1.0]` は **両端を含む閉区間** か **半開区間 `[0.9929, 1.0)`** か
  - `torch.empty(B).uniform_(L, U)` は `[L, U)` (右端を含まない) で実装される
  - **対応**: 本チケットでは半開区間採用 (PyTorch 標準動作)、ただし sub-model 1 で `c = 1.0` が出ないため、推論時の `c = 0.99995` (schedule 点) が含まれることを確認
  - **検証**: `test_noise_level_band_uniform_k1` で `c.max() <= 1.0` および schedule 点 `0.99995` が範囲内であることを assert

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み**
  - DDPM 標準形 (`x_t` 式) → PDF Fig 1b で確定
  - 4-step noise schedule (`ᾱ = [1e-4, 2.8e-2, 5.6e-1, 9.1e-1]`) → 確定
  - point-specialized partition (band 境界、1-to-1 dispatch) → 確定
  - additive bias conditioning (per-block `Linear(512, 512)`) → 確定
  - BDDM noise predictor 不採用 (固定 4 値直接使用) → 確定
  - Optimizer / lr / betas / wd → docs/training.md §3.4 で確定
  - EMA 不使用 → 確定

- 本チケットで決定する事項:
  - **Validation 時の `c` 値**: **band 中点** (`(L_k + U_k) / 2`) を採用。schedule 点固定との trade-off は §6.1 で決定 (band 中点は train 分布の中心、schedule 点は推論時の実値)。**再評価トリガー: M5.2 smoke**
  - **Loss normalization (`c` で割る) は採用しない**: FastDiff 慣例。**再評価トリガー: M5.2 / M6.2 で sub-model 4 が学習しない場合**
  - **`sub_model_k` を CLI 必須引数化**: config の `train.sub_model_k` ではなく CLI 必須にすることで、4 sub-model を **並列起動 (multi-GPU) 可能** な設計を担保 (T-M6.2 用)
  - **Checkpoint ファイル名規約**: best = `sub_{k}.pt`、interval = `step_{step}_sub_{k}.pt`、emergency = `emergency_step_{step}_sub_{k}.pt`、すべて sub-model index を suffix に含む
  - **TensorBoard log directory**: `logs/diff/sub_{k}/` で sub-model 別に分離 (multi-GPU 並列起動時の collision 防止)
  - **Validation eps の seed 固定**: `torch.Generator(device).manual_seed(state.step // validation.interval_steps * 1000 + k)` で 10k step ごとに固定値を変えながらも deterministic
  - **Histogram interval**: `histogram_interval_steps=500` (scalar の 5 倍粗い)。容量と sampling 検証のバランス

### 6.3 他チケットとの整合性

- **T-M2.1 (Dataset)** との整合:
  - 期待 signature: `LibriTTSRDataset.from_config(cfg["data_train"], mode="train")` で segment_length=25600, hop_length=256 を受け取る (`docs/training.md` §3.4)
  - `__getitem__` 戻り値: **`Batch` TypedDict** (`{"mel", "audio", "n_samples"}`、T-M2.1 §8.1 採用昇格)
  - `seed_worker` を `worker_init_fn` に渡す (T-M2.1 §9.1)
  - 不整合があった場合は T-M2.1 側を修正

- **T-M3.1 (DiffWaveNext2)** との整合:
  - 期待 signature: `DiffWaveNext2.from_config(cfg["model"])` で 4 sub-model instantiate (本チケット要請: `only_sub_model: int | None` 追加)
  - 期待 method: `model.sample_noise_level(k, B) -> torch.Tensor (B,) float32`、`model.BAND_BOUNDS[k-1] -> (L, U)`
  - 期待 attribute: `model.sub_models: nn.ModuleList[SubModelDiff]` (1-indexed → 0-indexed)
  - **重要**: `model.sample_noise_level(k, B)` の戻り値 device は `cpu` (本チケットで `.to(device)` 転送) または config で `device` 指定可能 → T-M3.1 §9.1 へ申し送り
  - `SubModelDiff.forward(mel, x_t, c) -> eps_pred (B, T_audio)` (T-M1.6 で確定)
  - 不整合があった場合は T-M3.1 側を修正

- **T-M2.5 (train_gan)** との整合:
  - `utils/training_loop.py` (T-M2.5 §8.1 採用昇格、M2.6 smoke pass 直後に切り出し) から `TrainState`, `iter_forever`, `register_sigterm_handler`, `atomic_save`, `save_checkpoint`, `load_checkpoint`, `log_scalars`, `run_validation` を import
  - `train_diff_step` の signature は `train_gan_step(G, D, opt_G, opt_D, sch_G, sch_D, mel, audio, cfg, ...)` と並列構造 (Diff 側は `model, opt, batch, k, cfg`)
  - **重要**: T-M2.5 が `tests/conftest.py` に共通 fixture (`tiny_config`, `tmp_checkpoint_dir`) を置いた場合、本チケットでも import して再利用
  - 不整合があった場合は T-M2.5 側を修正 (training_loop.py の API 変更要請)

- **T-M3.5 (Diff smoke)** へ渡す情報:
  - `train_diff_step` を import して 1000 回呼ぶ薄い loop で実装
  - `scripts/smoke_diff.py` は薄い wrapper (`train_diff_step` を直接呼ぶ)
  - 1 utterance を `train_loader` から繰り返し読む `OverfitDataset` を T-M3.5 で別途実装 (本チケットでは扱わない)
  - 1000 step 完了で `loss` が **初期値の 5%** 以下になることを期待 (milestones.md §M3.5)

- **T-M5.2 (1 sub-model 1 epoch)** へ渡す情報:
  - `uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2.yaml --sub-model 1 --amp` で起動
  - 1 epoch (train-clean-100 で約 7k step at batch_size=20) は YAML の `train.max_steps` を 7000 に上書きするか、別途 `configs/diff_1epoch.yaml` を T-M5.2 で作成
  - 1 epoch 後の validation MSE が **初期値の 30%** 以下を期待 (milestones.md §M5.2)

- **T-M6.2 (4 sub-model 32h 本格訓練)** へ渡す情報:
  - 順次起動 (single GPU) または並列起動 (4 GPU) のスクリプトを T-M6.2 で実装
  - 1M step / sub-model × 4 = 4M step、A100 単体で約 32 時間 (合計)
  - Multi-GPU 並列で 4 sub-model を同時起動する場合、`CUDA_VISIBLE_DEVICES` で分離、`logs/diff/sub_{k}/` で TensorBoard 分離

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §3 (訓練フロー、optimizer、loss、segment_length) と整合
- [ ] `docs/open-questions.md` 関連項目 (DDPM 標準形、4-step schedule、point-specialized partition、BDDM 不採用、EMA 不使用) と整合
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 3 項目クリア、5.4 追加 acceptance 全クリア
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 必須 CLI 引数欠落 (`--sub-model`)、`--sub-model 0 / 5`、checkpoint ファイル不存在、config drift 警告が適切に出る
- [ ] **参考実装 (FastDiff / BDDM / Vocos / WaveFit-PT) をコピーしていない** (CLAUDE.md 末尾、`docs/training.md` §3 記述から再構成)
- [ ] DDPM 拡散式 `x_t = √ᾱ * x_0 + √(1-ᾱ) * ε` のコード実装が PDF Fig 1b と数式一致 (`tests/test_train_diff.py::test_diffusion_forward_x_t_formula` で 3 点 pin)
- [ ] Optimizer が `Adam` (AdamW ではない)、`betas=(0.9, 0.98)`、`weight_decay=0.0`
- [ ] Scheduler が **存在しない** (固定 lr、`InverseLR` の import なし)
- [ ] Loss が `F.mse_loss` (MAE / Huber でない)
- [ ] `clip_grad_norm_` が `backward()` 後 / `step()` 前のタイミング
- [ ] checkpoint resume で step / opt / RNG state が完全復元 (test_resume_continues_step で 2 step 目 loss 一致)
- [ ] TensorBoard が `sub_{k}/loss`, `sub_{k}/lr`, `sub_{k}/grad_norm`, `sub_{k}/c_mean`, `sub_{k}/noise_level_c (histogram)` を sub-model 別 directory で出力
- [ ] EMA 関連コードが **一切存在しない**
- [ ] `--amp` 時に MSE 計算が fp32 で実行され bf16 underflow を回避
- [ ] `iter_forever` が `persistent_workers=True` で worker 再起動コストを回避
- [ ] CPU でテストが pass (CUDA 不要、ただし `--amp` は GPU 必須 → `@pytest.mark.gpu`)
- [ ] `utils/training_loop.py` を T-M2.5 と共有していて重複実装がない
- [ ] `seed_worker` を `wavenext2.data.dataset` から import (T-M2.1 §9.1 で提供)
- [ ] `--sub-model` 必須化で 4 sub-model 並列起動が可能 (multi-GPU で `CUDA_VISIBLE_DEVICES` 分離可能)
- [ ] `pyproject.toml` の `[project.scripts]` 追加 (任意、追加した場合は `uv run train-diff --help` が動作)
- [ ] **`sample_noise_level` 戻り値 device / dtype** が tests で pin されている (T-M3.1 §9.1 申し送り事項の確認)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M3) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計

- **素 PyTorch + click CLI で 1 ファイル (`train_diff.py`) に集約** (T-M2.5 §8.1 と同じ理由):
  - (a) `accelerate` / `pytorch-lightning` は抽象レイヤが厚すぎ、4 sub-model 独立訓練のような非標準パターンで hook が増えて可読性低下
  - (b) 1 ファイルに収めると `main()` を読むだけで訓練フロー全体が把握でき、再現実装の本来目的に合致
  - (c) T-M2.5 (GAN) との API symmetry (`train_*_step` 同 signature) で fixture 共用と将来 Lightning 移行が容易

- **【採用】`train_diff_step(model, opt, batch, k, cfg) -> dict[str, float]` を公開関数として切り出し** (T-M2.5 §8.1 採用昇格踏襲):
  - 理由 1: T-M3.5 smoke が `from train_diff import train_diff_step` で再利用、`scripts/smoke_diff.py` は薄い wrapper に
  - 理由 2: T-M2.5 `train_gan_step` と signature 統一で fixture 共用が可能
  - 理由 3: 将来 Lightning 移行時にロジックを `LightningModule.training_step` に移すだけで済む

- **【採用】SIGTERM/SIGINT handler で emergency checkpoint 保存** (T-M2.5 §8.1 採用):
  - `register_sigterm_handler` を `utils/training_loop.py` から import
  - M6.2 cluster preemption (32h 訓練、preempt 確率高) に対応必須

- **【採用】`utils/training_loop.py` を T-M2.5 から共有 import**:
  - T-M2.5 §8.1 採用昇格で M2.6 smoke pass 直後に切り出し済 (前提)
  - 本チケットでは `TrainState` に `sub_model_k` field を追加するのみ
  - GAN 側の `d_loss_below_threshold_steps` ロジックは Diff では参照しない (`utils/training_loop.py` 内で GAN/Diff 共存)

- **【採用】CLI 必須引数 `--sub-model k`** (config の `train.sub_model_k` ではない):
  - 4 sub-model を multi-GPU 並列起動可能 (`CUDA_VISIBLE_DEVICES=0..3` × `--sub-model 1..4` を bash で並列実行)
  - config に sub_model_k を埋め込むと **4 config file が必要** で煩雑、CLI 引数化で 1 config 共用
  - T-M6.2 で `for k in 1..4; do ...; done` の bash one-liner 起動が可能

- **【採用】固定 lr (scheduler 不採用)**:
  - `docs/training.md` §3.4 で FastDiff 慣例。`InverseLR` は GAN 側のみ
  - 理由: Diff は loss が fluctuate しやすく LR scheduling の効果が小さい (FastDiff 報告)

- **【採用】`--amp` は bf16 のみ (fp16 不採用)**:
  - T-M1.5 §6.1 で sinusoidal embedding の小さい値が fp16 で underflow するリスク
  - MSE loss も `(eps_pred - eps)^2 ≈ 1e-3` で fp16 underflow リスク
  - 検証: `test_amp_autocast_boundary` で MSE が fp32 で計算されることを assert

- **【採用】Validation で band 中点 `c_val = (L_k + U_k) / 2.0` 固定**:
  - 訓練分布の中心、representative
  - 代替案 (schedule 点固定) は推論時実値だが分布の端、validation の安定性低下

- **【採用】TensorBoard log directory を sub-model 別に分離** (`logs/diff/sub_{k}/`):
  - multi-GPU 並列起動時の collision 防止
  - `tensorboard --logdir logs/diff` で 4 sub-model を 1 つの UI で比較可能

#### Deprecated (旧案、却下根拠)

1. **`pytorch-lightning` Trainer 採用**: T-M2.5 §8.1 と同じ理由で却下
2. **`accelerate` library 採用**: T-M2.5 §8.1 と同じ理由で却下
3. **4 sub-model を 1 つの train run で並列訓練 (single CLI invocation で 4 sub-model 同時)**:
   - メリット: 1 回の起動で完結、SIGTERM handler / TensorBoard が 1 つ
   - 却下: 4 sub-model 同時 forward でメモリ × 4、勾配計算も × 4 で OOM 確実
   - 代替 (multi-GPU 並列): `CUDA_VISIBLE_DEVICES=0..3` × `--sub-model 1..4` の bash 並列起動で達成可能
4. **noise level sampling を log-uniform にする**:
   - メリット: band 内の boundary 付近 (`c ≈ U_k`) で confusing が発生しにくい
   - 却下: 論文 / FastDiff は uniform U(L, U) のみ。**再評価トリガー: M5.2 smoke で sub-model 1 (c ≈ 1.0 域) の loss が高止まりした場合**
5. **MSE → MAE (L1) 代替**:
   - メリット: outlier に頑健
   - 却下: 論文 Fig 1b で MSE と明示、再現性優先。**再評価トリガー: M5.2 / M6.2 で発散時**
6. **v-prediction parameterization (`v = sqrt(1-abar)*x_0 + sqrt(abar)*eps`、`x_0 prediction` の安定版)**:
   - メリット: noise level 依存の loss scale 不均衡を緩和 (sub-model 4 で `c ≈ 0.3` の問題を解消)
   - 却下: 論文は eps-prediction、再現性優先
   - **再評価トリガー: M5.2 smoke で sub-model 4 が学習しない場合、または M6.2 本格訓練で品質劣化時**
7. **`x_0 prediction` parameterization**:
   - メリット: sub-model が直接 clean 波形を出力するため interpretable
   - 却下: 論文は eps-prediction、再現性優先
8. **Loss normalization (`loss = MSE / (c^2 + ε)`)**:
   - メリット: sub-model 1〜4 の loss scale を揃える
   - 却下: FastDiff 慣例なし。**再評価トリガー: M5.2 で sub-model 4 が学習しない場合**
9. **wandb 採用**:
   - メリット: cloud sync、複数 sub-model の比較が UI で容易
   - 却下: TensorBoard で十分 (M5.1 phase review で再評価予定、T-M2.5 §8.1 と同期)
10. **`omegaconf` / `hydra` 採用**:
    - メリット: config compose で GAN/Diff/4-sub-model 切替が容易
    - 却下: YAML 直読み + click で十分
11. **Loss weight を `nn.Parameter` 化して学習**: T-M2.5 §8.1 と同じ理由で却下 (固定値で再現性優先)
12. **訓練ループを `Trainer` クラスにカプセル化**: M3 段階では `main()` 関数 1 つで十分
13. **CLI 引数を `--sub-models 1,2,3,4` で複数指定可能化** (single CLI で順次 4 sub-model 訓練):
    - メリット: 1 回の起動で 4 sub-model 順次訓練
    - 却下: SIGTERM 時に途中 sub-model の resume が困難 (どの sub-model まで進んだか state 管理が複雑)、multi-GPU 並列の利益喪失
14. **`sample_noise_level` を `train_diff.py` 内で直接実装** (T-M3.1 に委任しない):
    - メリット: T-M3.1 への依存削減
    - 却下: BAND_BOUNDS は `DiffWaveNext2` の責務 (T-M3.1)、train script からは API 呼ぶだけが SoT に整合
15. **Validation で実 reverse sampling (T-M3.3) を使う**:
    - メリット: 推論時品質と直結
    - 却下: T-M3.3 完了後でないと使えない (本チケットは T-M3.3 と並列実装可能な設計を優先)、reverse sampling は 4 sub-model 必要だが本チケットは 1 sub-model 訓練

#### 追加検討した設計案

- **`noise_level_sampling` を log-uniform にする**:
  - 採用しない (uniform で十分、§8.1 Deprecated 4)
- **`max_steps` を sub-model 別に変える** (e.g., sub-model 1 は 500k step、sub-model 4 は 1.5M step):
  - 採用しない (FastDiff 慣例で 1M step 一律、再評価トリガー: M6.2 で sub-model 4 が高 loss)
- **Validation の eps を fixed (`torch.zeros_like` or固定 seed の `randn`)**:
  - 採用 (validation seed 固定の `randn` で deterministic)
- **`grad_clip_norm` を sub-model 別に変える**:
  - 採用しない (一律 1.0、論文準拠)
- **Mixed precision を per-loss 制御 (MSE だけ fp32)**:
  - 採用 (§6.1 通常項目「mixed precision」)
- **【検討追加】pydantic / dataclass-based config schema** (T-M2.5 §8.1 と同期):
  - 採用しない (現時点)、**再評価トリガー: M5.1 phase review で config 変更頻発時**

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| `pytorch-lightning` 不採用 | M6 完了時 (T-M2.5 と同期) | DDP / multi-node が必要になったら採用検討 |
| TensorBoard 単独 (wandb 不採用) | M5.1 phase review (T-M2.5 と同期) | 複数 sub-model 比較で wandb 採用判断 |
| `accelerate` 不採用 | M6.2 着手前 (必須レビュー) | multi-GPU 並列訓練のコスト比較 |
| MSE (vs MAE / Huber) | M5.2 smoke 発散 / M6.2 品質劣化 | MAE or Huber 採用 |
| eps-prediction (vs v-prediction / x_0 prediction) | M5.2 で sub-model 4 学習しない / M6.2 品質劣化 | v-prediction parameterization 採用 |
| Loss normalization なし (vs `MSE/(c^2+ε)`) | M5.2 で sub-model 4 学習しない | normalization 採用 |
| Uniform sampling (vs log-uniform) | M5.2 で sub-model 1 (c ≈ 1.0) 高 loss | log-uniform 採用 |
| Validation `c` at band 中点 | M5.2 で validation MSE と推論品質の相関が薄い | schedule 点固定に変更 |
| `max_steps=1M` 一律 | M6.2 で sub-model 別 plateau 時刻が大幅に異なる | sub-model 別 max_steps |
| Single sub-model per CLI invocation | M6.2 wall-clock 短縮が必要 | multi-sub-model parallel in single CLI (multi-GPU) |
| `histogram_interval_steps=500` | logs disk 容量超過 | 10000 に間引き |
| pydantic config schema | M5.1 phase review (T-M2.5 と同期) | 採用判断 |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切 (size=L)。
  - 4 sub-model 独立訓練は 1 CLI invocation = 1 sub-model で thin に保つ
  - T-M2.5 (GAN) と並ぶ訓練ループの集約点
  - 1 ファイル + tests 1 ファイルで完結し、後続 T-M3.5 / T-M5.2 / T-M6.2 はこの entry point を呼ぶだけ
- **別マイルストーンに移すべき部分はないか**: なし。
  - Validation forward は `train_diff_step` と密結合のため同ファイル内で OK
  - `utils/training_loop.py` の共通化は T-M2.5 で実施済 (本チケットでは Diff 用 field 追加のみ)
- **インターフェース定義の見直し余地**:
  - **`train_diff_step` を class method (`DiffTrainer.step`) にカプセル化**: 採用しない (関数のシンプルさ優先、Lightning 移行時に `LightningModule.training_step` に移植容易)
  - **`build_loaders` を `utils/data_loader.py` に分離**: 採用しない (本チケット private 関数で十分、GAN 側と異なる config キー (`data_train`, `data_val`) を持つため共通化のメリット小)

- **【新原則】「training_step 関数」と「main loop」の分離** (T-M2.5 §8.2 と同じ):
  - `train_diff_step` 単体で 1 step 検証可能、`main()` 全体を呼ぶ必要なし
  - T-M3.5 smoke は `train_diff_step` を直接 1000 回呼ぶ薄い loop
  - Lightning 移行時は `train_diff_step` を `LightningModule.training_step` に移すだけ

- **【新原則】「4 sub-model 訓練 = 4 CLI invocation」**:
  - 1 sub-model = 1 訓練プロセス = 1 checkpoint = 1 TensorBoard directory
  - 設計の単純さ最優先、resume / preemption / multi-GPU 並列のすべてで利点
  - 代替案 (single CLI で 4 sub-model 順次 / 並列) は state 管理複雑化のため却下

#### M1 / M2 phase review で確立した原則の適用

- **factory パターン**: T-M1.6 / T-M2.5 で確立した factory 一貫化に従い、`DiffWaveNext2.from_config(cfg["model"])` / `LibriTTSRDataset.from_config(cfg["data_train"])` 経由で生成
- **config drift 耐性**: `from_config` 内で `allowed_keys` 絞り込みにより、新 key 追加でクラスが壊れない
- **`scope="module"` fixture**: M1 で確立したテスト最適化を本チケットでも採用 (`tests/test_train_diff.py` で `model`, `opt` を再利用)
- **`Batch` TypedDict 共有**: T-M2.1 §8.1 採用昇格、本チケットでも `from wavenext2.data.dataset import Batch` で import

#### 再評価トリガー (M3 phase review 時)

- T-M3.5 (smoke 1000 step) で sub-model 1 の loss が **初期値の 5%** を切らなければ、本チケットの Adam lr / batch_size / segment_length を見直す
- T-M5.2 (1 sub-model 1 epoch) で sub-model 別の validation MSE が plateau に達さなければ、Loss normalization / v-prediction を検討
- T-M6.2 (4 sub-model 本格訓練) で sub-model 4 が他より loss が flat なら、Loss normalization / log-uniform sampling を採用

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M3.5 (1000 step smoke) へ
- **使用方法**:
  ```python
  # tests/integration/test_diff_overfit_1000step.py or scripts/smoke_diff.py
  from wavenext2.train.train_diff import train_diff_step
  # 1000 回呼ぶ薄い loop で実装
  for step in range(1000):
      logs = train_diff_step(model, opt, batch_overfit, k=1, cfg=cfg)
  ```
- **重要事項**:
  - **`scripts/smoke_diff.py` は `train_diff_step` を import する薄い wrapper にする**
  - 1 utterance を `train_loader` から繰り返し読む `OverfitDataset` を T-M3.5 で別途実装 (本チケットでは扱わない)
  - 1000 step 完了で `logs["loss"]` が **初期値の 5%** 以下になることを期待 (milestones.md §M3.5)
  - 本チケットで実装した `--debug` フラグは 1 step だけのため T-M3.5 では使わない
  - smoke 完了後に `sub_1.pt` 1 ファイルのみ生成 (10k step に達しないため interval ckpt は生成されない)
  - 同じ utterance に対する reverse sample が GT に近い (MR-STFT loss で比較) は T-M3.3 (reverse sampler) 完了後に実施

#### T-M5.2 (1 sub-model 1 epoch 訓練) へ
- **使用方法**:
  ```bash
  uv run python -m wavenext2.train.train_diff \
    --config configs/diff_wavenext2.yaml \
    --sub-model 1 \
    --amp  # 推奨
  # 7k step まで自動で回る (configs/diff_wavenext2.yaml の max_steps を 7000 に修正)
  ```
- **重要事項**:
  - 1 epoch (train-clean-100 で約 7k step at batch_size=20) は YAML の `train.max_steps` を 7000 に上書きするか、別途 `configs/diff_1epoch.yaml` を T-M5.2 で作成
  - validation を 70k step → 700 / 1400 / ... に詰める (interval を 1000 に下げる) で MSE 降下を確認
  - 1 epoch 後の validation MSE が **初期値の 30%** 以下を期待
  - **noise level conditioning が機能** (異なる noise level で異なる出力) の確認 (milestones.md §M5.2 Acceptance)
  - `--amp` 推奨 (24GB GPU で OOM 回避)

#### T-M6.2 (4 sub-model 32h 本格訓練) へ
- **使用方法** (順次起動、single GPU):
  ```bash
  for k in 1 2 3 4; do
      uv run python -m wavenext2.train.train_diff \
          --config configs/diff_wavenext2.yaml \
          --sub-model $k --amp
  done
  ```
- **使用方法** (並列起動、4 GPU):
  ```bash
  CUDA_VISIBLE_DEVICES=0 uv run python -m wavenext2.train.train_diff --sub-model 1 --amp --config configs/diff_wavenext2.yaml &
  CUDA_VISIBLE_DEVICES=1 uv run python -m wavenext2.train.train_diff --sub-model 2 --amp --config configs/diff_wavenext2.yaml &
  CUDA_VISIBLE_DEVICES=2 uv run python -m wavenext2.train.train_diff --sub-model 3 --amp --config configs/diff_wavenext2.yaml &
  CUDA_VISIBLE_DEVICES=3 uv run python -m wavenext2.train.train_diff --sub-model 4 --amp --config configs/diff_wavenext2.yaml &
  wait
  ```
- **重要事項**:
  - A100 単体で 1 sub-model あたり約 8 時間、4 sub-model 合計 32 時間
  - Claude Code は `bash` で `run_in_background=true` で起動 → 数時間〜数日おきに `tensorboard --inspect logs/diff/sub_{k}` で監視
  - Divergence / OOM 検知時は自動 `--resume checkpoints/diff/sub_{k}.pt` で再開
  - `checkpoint.keep_last_n=5` で rolling delete を有効化、`sub_{k}.pt` (best) のみ常時保持
  - 訓練完了後に T-M3.4 post-filter fit (`scripts/fit_post_filter.py`) を自動実行
  - 推論パイプライン (T-M3.3) と組み合わせて test-clean-100 全 4824 utterance を評価

#### T-M3.3 (Reverse sampler) へ
- **使用方法**:
  - 本チケット完了後に `checkpoints/diff/sub_{1,2,3,4}.pt` の 4 つを `DiffWaveNext2.from_pretrained_subs(...)` 等で 1 つの model に統合する API を T-M3.1 / T-M3.3 で検討
  - reverse sampling 自体は T-M3.3 の責務 (本チケットは訓練のみ)

#### T-M3.4 (Post-filter) へ
- **使用方法**:
  - 4 sub-model 訓練完了後、`sub_{1,2,3,4}.pt` をロード → reverse sample → dev set 200 utterances で FIR fit
  - 本チケットは 4 sub-model checkpoint を提供するだけ、post-filter fit は T-M3.4 の責務

#### T-M2.5 (train_gan) から受領
- **`utils/training_loop.py` の共通 API** (T-M2.5 §8.1 採用昇格で M2.6 smoke pass 直後に切り出し済):
  - `TrainState` dataclass (`step`, `best_val_metric`, `rng_state`, `sub_model_k`)
  - `iter_forever(loader)` generator
  - `register_sigterm_handler(save_fn)` 関数
  - `atomic_save(obj, path)` 関数
  - `save_checkpoint(model, opt, scheduler, state, path, atomic=False)` 関数
  - `load_checkpoint(path, model, opt, scheduler, state) -> TrainState` 関数
  - `log_scalars(writer, step, logs, lr=None, sub_model_k=None, **extra)` 関数
  - `run_validation(model_fn, loader, metric_fn, device, writer, step, tag_prefix)` 関数
- **`train_gan_step` 同 signature**: `train_diff_step(model, opt, batch, k, cfg)` で並列構造
- **AMP autocast 境界**: GAN 同様、forward は autocast / loss 計算は fp32 (`with torch.autocast(..., enabled=False)`)

#### T-M2.1 (Dataset) から受領
- **`Batch` TypedDict**: `from wavenext2.data.dataset import Batch` で `{"mel", "audio", "n_samples"}` キーを共有
- **`seed_worker` 関数**: `from wavenext2.data.dataset import seed_worker` で `worker_init_fn` 用 (T-M2.1 §9.1 で提供)
- **`LibriTTSRDataset.from_config(cfg, mode="train")`** で Diff モード (segment_length=25600, hop=256) のデータ取得
- **256 が 25600 の約数** で `(T_mel-1)*hop == segment_length` ぴったり整合 (GAN 側の 184 sample 余剰問題なし)
- **`multiprocessing_context="spawn"`**: Windows / Linux 共通で明示

#### T-M3.1 (DiffWaveNext2) から受領
- **`DiffWaveNext2.from_config(cfg["model"])`**: 4 sub-model instantiate
- **`model.sample_noise_level(k, B) -> torch.Tensor (B,) float32`**: band 内 uniform sampling
- **`model.BAND_BOUNDS[k-1] -> (L, U)`**: band 境界
- **`model.sub_models: nn.ModuleList[SubModelDiff]`** (1-indexed → 0-indexed)
- **`SubModelDiff.forward(mel, x_t, c) -> eps_pred (B, T_audio)`** (T-M1.6 で確定)

#### T-M3.1 への申し送り (本チケット §6.1 critical)
- **`sample_noise_level(k, B, device=None, generator=None) -> torch.Tensor (B,) float32`** の signature 採用要請:
  - `device` 指定でデバイス転送コスト削減
  - `generator` 指定でグローバル RNG 汚染回避 (`worker_init_fn` の効果を保つ)
- **`from_config(cfg, only_sub_model: int | None = None)` の追加要請**:
  - 1 sub-model のみ instantiate するメモリ節約モード
  - 本チケットでは default (`only_sub_model=None`) で全 4 sub-model instantiate するが、メモリ逼迫時は `only_sub_model=k` で `--sub-model k` 専用ロード
- **`BAND_BOUNDS` の閉区間 / 半開区間仕様**:
  - 本チケットは `torch.empty(B).uniform_(L, U)` (半開区間 `[L, U)`) を採用
  - sub-model 1 で `c = 1.0` が出ないため、推論時の `c = 0.99995` (schedule 点) が範囲内であることを T-M3.1 で確認

#### T-M0.2 申し送り (CI 整備、T-M2.5 §9.1 と同期)
- **`.github/workflows/test.yml` で `pytest -n auto` 並列化**: 本チケットの 100 step loss curve テスト (~5 秒/run × 4 sub-model 検証 = 20 秒) で全体テスト時間に影響
- `pytest-xdist` の依存追加を `pyproject.toml` の `[dependency-groups.dev]` に明示
- GPU テスト (`@pytest.mark.gpu`) は別 job で run、CPU テストのみ並列化

#### 共通の注意事項
- **EMA 関連 code を一切残さない** (T-M2.5 §9.1 と同期、docs/open-questions.md 確定)
- **autocast の境界**: `--amp` 時、autocast は **sub-model forward のみ** に限定。MSE loss 計算は fp32 で実行
- **`persistent_workers=True`**: train DataLoader に必須。epoch 境界での worker 再起動コスト回避
- **`persistent_workers=False`**: validation DataLoader は leak 防止のため毎回 close
- **Checkpoint state の完全性**: model (sub_models[k-1]) + opt + step + best metric + sub_model_k + RNG state (torch / cuda / numpy) の 6 グループを保存
- **4 sub-model 訓練の独立性**: 各 sub-model は別 process で訓練、checkpoint も別 file、TensorBoard log directory も別。**1 sub-model の失敗が他 3 つに影響しない設計**
- **`--sub-model` は必須引数**: 省略不可、CLI で必ず指定 (multi-GPU 並列起動可能化)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M3.2 の Acceptance チェックボックス 3 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M3.2 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/training.md` §3 にコード参照リンク (`src/wavenext2/train/train_diff.py:train_diff_step`) を追記
  - [ ] (該当時) `docs/implementation-plan.md` の `configs/diff_wavenext2.yaml` 雛形を実装と一致するよう更新 (model / data / train / validation / checkpoint / logging キー追加)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` で全 100% 確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - MSE → MAE / Huber (M5.2 発散時)
  - eps-prediction → v-prediction / x_0 prediction (M5.2 で sub-model 4 学習しない場合)
  - Loss normalization (`MSE / (c^2 + ε)`) 採用 (M5.2 で sub-model 4 学習しない場合)
  - Uniform → log-uniform sampling (M5.2 で sub-model 1 高 loss)
  - Validation `c` を band 中点 → schedule 点固定 (M5.2 で相関薄い)
  - `max_steps` を sub-model 別 (M6.2 で plateau 時刻が大幅に異なる)
  - Single CLI で multi-sub-model parallel (M6.2 wall-clock 短縮)
  - wandb 採用 (M5.1 phase review、T-M2.5 §8.1 と同期)
  - `accelerate` 採用 (M6.2 multi-GPU 化時)
- **`docs/open-questions.md` への追記要否**: 不要
