---
id: T-M2.5
title: GAN 訓練スクリプト (AdamW + InverseLR + 交互更新 + grad clip + 監視)
milestone: M2
phase: M2
status: completed
size: L
owner: -
created: 2026-05-26
updated: 2026-05-28
status_note: completed 2026-05-28 (tests/test_train_gan.py + test_inverse_lr.py 21 pass)
depends_on: [T-M2.1, T-M2.2, T-M2.3, T-M2.4]
blocks: [T-M2.6, T-M5.1]
related_docs:
  - docs/milestones.md#m25-training-script-srcwavenext2traintrain_ganpy
  - docs/training.md
  - docs/implementation-plan.md
  - docs/open-questions.md
---

# T-M2.5: GAN 訓練スクリプト (AdamW + InverseLR + 交互更新 + grad clip + 監視)

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.5](../milestones.md#m25-training-script-srcwavenext2traintrain_ganpy)
> **依存**: [T-M2.1](T-M2.1-dataset.md), [T-M2.2](T-M2.2-discriminator.md), [T-M2.3](T-M2.3-losses.md), [T-M2.4](T-M2.4-gan-model.md) / **後続**: [T-M2.6](T-M2.6-gan-smoke.md), [T-M5.1](T-M5.1-gan-1epoch.md)

## 1. タスク目的とゴール

### 目的
GAN-WaveNeXt 2 の **訓練ループ本体** を 1 ファイル (`src/wavenext2/train/train_gan.py`) に閉じ込め、T-M2.1〜T-M2.4 で実装した部品 (Dataset, Discriminator, Loss, GANWaveNext2) を **AdamW + InverseLR + hinge GAN 交互更新** で接続する。

本チケットは「実際に loss が finite で 1 step 動く」ところまで責務を持ち、`docs/training.md` §2.4 の訓練設定 (lr_g=1e-4, lr_d=2e-4, β=[0.8, 0.99], wd=1e-3, InverseLR (inv_gamma=200000, power=0.5, warmup=0.999), grad_clip=1.0, EMA 不使用) を **1 行も曖昧なく** コード化することで、後続 T-M2.6 (1000 step overfit) / T-M5.1 (1 epoch) / T-M6.1 (2M step 本番) の **唯一の入口** とする。

### ゴール
- [ ] `src/wavenext2/train/train_gan.py` に `main()` 関数 (click CLI entry point) が実装され `python -m wavenext2.train.train_gan --config ...` で起動可能
- [ ] Optimizer: G=AdamW(lr=1e-4, betas=[0.8, 0.99], wd=1e-3), D=AdamW(lr=2e-4, betas=[0.8, 0.99], wd=1e-3)
- [ ] Scheduler: 自作 `InverseLR` (inv_gamma=200000, power=0.5, warmup=0.999) を G/D 別々にアタッチ
- [ ] 交互更新: D 更新 (hinge `relu(1-D(real))+relu(1+D(fake))`) → G 更新 (hinge `-D(fake)`+FM L1+MR-STFT(SC+Mag))
- [ ] Gradient clip: `clip_grad_norm_(max_norm=1.0)` を G/D それぞれの backward 直後に適用
- [ ] EMA: **不使用** (`docs/open-questions.md` 確定、コード上に EMA 関連の dead code を残さない)
- [ ] Mixed precision: fp32 デフォルト、`--amp` フラグで bf16 に切替可能 (fp16 は T-M1.5 §6.1 リスクのため採用しない)
- [ ] Logging: TensorBoard に `loss_G`, `loss_D`, `loss_g_gan`, `loss_g_fm`, `loss_g_mrstft_sc`, `loss_g_mrstft_mag`, `lr_G`, `lr_D`, sample audio (validation), mel spectrogram 可視化が書ける
- [ ] Validation: 10k step ごとに 100 utterances (T-M0.3 で生成済み `val.tsv`、speaker-balanced) で MR-STFT total を計算
- [ ] Checkpoint: 10k step ごとに `checkpoints/gan/step_{step}.pt` 保存、best validation MR-STFT で `best.pt` 更新
- [ ] Resume: `--resume <path>` で step / optimizer state / scheduler state / RNG state を完全復元
- [ ] `tests/test_train_gan.py` で smoke step (1 step 実行) と `tests/test_inverse_lr.py` で InverseLR の数式検証が pass
- [ ] `docs/milestones.md` §M2.5 Acceptance criteria 3 項目クリア (loss finite / checkpoint 保存復元 / TensorBoard ログ)

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規実装 (T-M0.2 で空 stub として配置済み or 未配置):
  - `src/wavenext2/train/train_gan.py` (本実装、`main()` + `train_gan_step()` 公開関数)
  - `src/wavenext2/utils/scheduler.py` (`InverseLR` を実装、Diff 側 T-M3.2 でも将来再利用可能性あり)
  - `tests/test_train_gan.py` (smoke step テスト、`train_gan_step` を直接呼ぶ)
  - `tests/test_inverse_lr.py` (scheduler 数式テスト)
- 編集:
  - `src/wavenext2/train/__init__.py` (`__all__` に `main`, `train_gan_step` を追加、`from .train_gan import main, train_gan_step`)
  - `src/wavenext2/utils/__init__.py` (`__all__` に `InverseLR` を追加)
  - `configs/gan_wavenext2.yaml` (T-M0.2 で生成済の雛形を本実装用にキー追加: `validation.interval_steps`, `checkpoint.dir`, `checkpoint.interval_steps`, `logging.tensorboard_dir`)
  - `pyproject.toml` の `[project.scripts]` に `train-gan = "wavenext2.train.train_gan:main"` を追加 (任意、後続 M5/M6 の便宜のため)
  - `docs/milestones.md` §M2.5 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M2.5 ステータス更新

> **重要 (§8.1 採用昇格)**: `train_gan_step(G, D, opt_G, opt_D, mel, audio, cfg) -> dict[str, float]` を **公開関数** として切り出す。理由:
> - T-M2.6 smoke が `from train_gan import train_gan_step` で再利用 (`scripts/smoke_gan.py` は薄い wrapper にできる)
> - T-M3.2 (train_diff) で `train_diff_step` を同 signature にすると `tests/conftest.py` の fixture 共用が可能
> - 将来 `pytorch-lightning` 移行時はロジックを `LightningModule.training_step` に移すだけで済む (Lightning-ready architecture)

### 2.2 主要構造

#### import 形式 (T-M0.2 で確定)
```python
from wavenext2.train.train_gan import main
from wavenext2.utils.scheduler import InverseLR
```

#### `train_gan.py` のスケルトン

```python
"""train_gan.py — GAN-WaveNeXt 2 の訓練ループ.

論文 §3.2 / docs/training.md §2 を実装する。
- Optimizer: AdamW (G/D 別 lr、betas=[0.8, 0.99], weight_decay=1e-3)
- Scheduler: InverseLR (inv_gamma=200000, power=0.5, warmup=0.999) を G/D 別々に
- Loss: hinge GAN (D: relu(1-D(real)) + relu(1+D(fake)) / G: -D(fake)) + FM L1 + MR-STFT (SC + Mag)
- Grad clip: max_norm=1.0
- EMA: 不使用 (docs/open-questions.md 確定)
- Logging: TensorBoard (step ごとに loss、10k step ごとに validation + sample audio)

公開 API:
- `main(...)`: click CLI entry point (loop / checkpoint / logging を統括)
- `train_gan_step(G, D, opt_G, opt_D, mel, audio, cfg) -> dict[str, float]`:
  1 step alternating update を実行し loss scalar の dict を返す。
  T-M2.6 smoke / T-M3.2 (train_diff_step) / 将来の Lightning 移行で再利用される唯一の公開関数。
  返り値 dict のキー: `loss_G`, `loss_D`, `loss_g_gan`, `loss_g_fm`, `loss_g_mrstft_sc`,
  `loss_g_mrstft_mag`, `loss_d_real`, `loss_d_fake`, `grad_norm_G`, `grad_norm_D`
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wavenext2.data.dataset import LibriTTSRDataset
from wavenext2.losses.adversarial import HingeGANLoss
from wavenext2.losses.feature_matching import FeatureMatchingLoss
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss
from wavenext2.models.discriminator import MultiScaleDiscriminator
from wavenext2.models.gan_wavenext2 import GANWaveNext2
from wavenext2.utils.config import load_config
from wavenext2.utils.logging import setup_logger
from wavenext2.utils.scheduler import InverseLR


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--resume", "resume_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--debug", is_flag=True, help="小規模 smoke (1 batch / 1 step / no validation)")
@click.option("--amp", is_flag=True, help="bf16 mixed precision (default: fp32)")
def main(config_path: str, resume_path: str | None, debug: bool, amp: bool) -> None:
    """GAN-WaveNeXt 2 の訓練 entry point."""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if amp else torch.float32

    # ===== Models =====
    G = GANWaveNext2.from_config(cfg["model"]).to(device)
    D = MultiScaleDiscriminator.from_config(cfg["discriminator"]).to(device)

    # ===== Optimizers =====
    opt_G = AdamW(
        G.parameters(),
        lr=cfg["train"]["optimizer"]["lr_g"],     # 1e-4
        betas=tuple(cfg["train"]["optimizer"]["betas"]),  # [0.8, 0.99]
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],  # 1e-3
    )
    opt_D = AdamW(
        D.parameters(),
        lr=cfg["train"]["optimizer"]["lr_d"],     # 2e-4
        betas=tuple(cfg["train"]["optimizer"]["betas"]),
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],
    )

    # ===== Schedulers =====
    sch_G = InverseLR(opt_G, **cfg["train"]["scheduler"])  # inv_gamma=200000, power=0.5, warmup=0.999
    sch_D = InverseLR(opt_D, **cfg["train"]["scheduler"])

    # ===== Losses =====
    crit_gan   = HingeGANLoss()                      # T-M2.3
    crit_fm    = FeatureMatchingLoss(loss_type="l1") # T-M2.3
    crit_mrstft = MultiResolutionSTFTLoss(**cfg["loss"]["mrstft"])  # T-M2.3

    # ===== Data =====
    train_loader, val_loader = build_loaders(cfg)

    # ===== State =====
    state = TrainState(
        step=0,
        best_val_mrstft=float("inf"),
        rng_state=None,
    )
    if resume_path:
        state = load_checkpoint(resume_path, G, D, opt_G, opt_D, sch_G, sch_D)

    # ===== Logging =====
    writer = SummaryWriter(cfg["logging"]["tensorboard_dir"])

    # ===== Emergency save on SIGTERM/SIGINT (M6 cluster preemption 対策) =====
    def _emergency_save(signum: int, frame: Any) -> None:  # noqa: ARG001
        save_checkpoint(G, D, opt_G, opt_D, sch_G, sch_D, state,
                        Path(cfg["checkpoint"]["dir"]) / f"emergency_step_{state.step}.pt")
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT,  _emergency_save)

    # ===== Training loop =====
    G.train()
    D.train()
    for batch in iter_forever(train_loader):
        if state.step >= cfg["train"]["max_steps"]:
            break

        mel, x_gt = batch["mel"].to(device), batch["audio"].to(device)

        # --- 1 step alternating update (公開関数) ---
        logs = train_gan_step(G, D, opt_G, opt_D, sch_G, sch_D,
                              mel, x_gt, cfg,
                              crit_gan, crit_fm, crit_mrstft,
                              amp=amp, dtype=dtype)

        # --- Logging ---
        if state.step % cfg["logging"]["scalar_interval_steps"] == 0:
            log_scalars(writer, state.step, logs,
                         lr_G=sch_G.get_last_lr()[0], lr_D=sch_D.get_last_lr()[0])
        # SummaryWriter buffer leak 対策: 10k step ごとに flush
        if state.step % 10000 == 0:
            writer.flush()

        # --- D 強すぎ問題の検知 ---
        # `loss_D < 0.01` が 1000 step 連続なら警告ログ (§6.1 critical 項目)
        state.update_d_loss_history(logs["loss_D"])
        if state.d_loss_below_threshold_steps >= 1000:
            logger.warning("loss_D < 0.01 for 1000 steps: D may be too strong (consider 1:2 update)")

        # --- Validation ---
        if state.step > 0 and state.step % cfg["validation"]["interval_steps"] == 0:
            val_mrstft = run_validation(G, val_loader, crit_mrstft, device, writer, state.step)
            if val_mrstft < state.best_val_mrstft:
                state.best_val_mrstft = val_mrstft
                # atomic rename で best.pt が中途半端な状態にならないことを保証 (T-M2.4 申し送り)
                save_checkpoint(G, D, opt_G, opt_D, sch_G, sch_D, state,
                                Path(cfg["checkpoint"]["dir"]) / "best.pt",
                                atomic=True)

        # --- Checkpoint ---
        if state.step > 0 and state.step % cfg["checkpoint"]["interval_steps"] == 0:
            save_checkpoint(G, D, opt_G, opt_D, sch_G, sch_D, state,
                            Path(cfg["checkpoint"]["dir"]) / f"step_{state.step}.pt")

        state.step += 1

        if debug and state.step >= 1:
            break  # smoke

    writer.flush()
    writer.close()


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
    crit_gan: nn.Module,
    crit_fm: nn.Module,
    crit_mrstft: nn.Module,
    amp: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """1 step alternating update を実行し loss scalar の dict を返す.

    重要: autocast 境界は G forward + D forward のみ。hinge GAN / FM / MR-STFT loss は
    fp32 で計算 (bf16 underflow 回避、§6.1 通常項目「mixed precision」)。
    T-M2.6 smoke / T-M3.2 (train_diff_step) / 将来の Lightning 移行で再利用される公開関数。

    Returns:
        dict with keys:
            - loss_G, loss_D (total)
            - loss_g_gan, loss_g_fm, loss_g_mrstft_sc, loss_g_mrstft_mag (G の sub-loss)
            - loss_d_real, loss_d_fake (D の sub-loss、§6.1 通常項目で別ログ要請)
            - grad_norm_G, grad_norm_D (clip_grad_norm_ の返り値)
    """
    # --- Generator forward (autocast 内) ---
    with torch.autocast(device_type=mel.device.type, dtype=dtype, enabled=amp):
        y_hat = G(mel, audio_length=audio.shape[-1])

    # --- Discriminator update (forward は autocast、loss は fp32) ---
    opt_D.zero_grad(set_to_none=True)
    with torch.autocast(device_type=mel.device.type, dtype=dtype, enabled=amp):
        d_real = D(audio.unsqueeze(1))
        d_fake = D(y_hat.detach().unsqueeze(1))
    # hinge GAN は fp32 (underflow 回避)
    with torch.autocast(device_type=mel.device.type, enabled=False):
        loss_d_real = sum((F.relu(1.0 - r[0].float())).mean() for r in d_real) / len(d_real)
        loss_d_fake = sum((F.relu(1.0 + f[0].float())).mean() for f in d_fake) / len(d_fake)
        loss_D = loss_d_real + loss_d_fake
    loss_D.backward()
    grad_norm_D = torch.nn.utils.clip_grad_norm_(
        D.parameters(), max_norm=cfg["train"]["grad_clip_norm"]
    )
    opt_D.step()
    sch_D.step()

    # --- Generator update (forward は autocast、loss は fp32) ---
    opt_G.zero_grad(set_to_none=True)
    with torch.autocast(device_type=mel.device.type, dtype=dtype, enabled=amp):
        d_fake_for_g  = D(y_hat.unsqueeze(1))
        d_real_for_fm = D(audio.unsqueeze(1))
    with torch.autocast(device_type=mel.device.type, enabled=False):
        loss_g_gan = crit_gan.g_loss(d_fake_for_g)
        loss_g_fm  = crit_fm(d_real_for_fm, d_fake_for_g) * cfg["loss"]["weights"]["d_fm"]
        # MR-STFT は (total, unweighted_dict) tuple (T-M2.3 申し送り)
        loss_g_mrstft_total, mrstft_unweighted = crit_mrstft(y_hat, audio)
        loss_g_sc  = mrstft_unweighted["sc"]  * cfg["loss"]["weights"]["mrstft_sc"]
        loss_g_mag = mrstft_unweighted["mag"] * cfg["loss"]["weights"]["mrstft_mag"]
        loss_G = loss_g_gan + loss_g_fm + loss_g_sc + loss_g_mag
    loss_G.backward()
    grad_norm_G = torch.nn.utils.clip_grad_norm_(
        G.parameters(), max_norm=cfg["train"]["grad_clip_norm"]
    )
    opt_G.step()
    sch_G.step()

    return {
        "loss_G": loss_G.item(),
        "loss_D": loss_D.item(),
        "loss_g_gan":         loss_g_gan.item(),
        "loss_g_fm":          loss_g_fm.item(),
        "loss_g_mrstft_sc":   loss_g_sc.item(),
        "loss_g_mrstft_mag":  loss_g_mag.item(),
        "loss_d_real":        loss_d_real.item(),
        "loss_d_fake":        loss_d_fake.item(),
        "grad_norm_G":        float(grad_norm_G),
        "grad_norm_D":        float(grad_norm_D),
        # unweighted (T-M2.3 申し送り、TensorBoard 強制 log)
        "loss_g_mrstft_sc_unweighted":  mrstft_unweighted["sc"].item(),
        "loss_g_mrstft_mag_unweighted": mrstft_unweighted["mag"].item(),
    }
```

#### `scheduler.py` (`InverseLR`)

```python
"""scheduler.py — InverseLR scheduler.

lr(step) = base_lr * warmup ** (max(inv_gamma - step, 0))
                   * (1 + step / inv_gamma) ** (-power)

WaveFit-PT が採用する exponential warmup + inverse-power-law decay。

Default: inv_gamma=200000, power=0.5, warmup=0.999
"""

from __future__ import annotations
import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class InverseLR(LRScheduler):
    """Inverse power-law LR scheduler with exponential warmup.

    Args:
        optimizer: Wrapped optimizer.
        inv_gamma: Decay scale (steps). Default 200000.
        power: Decay exponent. Default 0.5.
        warmup: Exponential warmup base (∈ (0, 1)). Default 0.999.
            Effectively: lr starts at base_lr * warmup ** inv_gamma and approaches base_lr.
        last_epoch: Resume step (default -1 = fresh start).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        inv_gamma: float = 200000.0,
        power: float = 0.5,
        warmup: float = 0.999,
        last_epoch: int = -1,
    ) -> None:
        self.inv_gamma = float(inv_gamma)
        self.power     = float(power)
        self.warmup    = float(warmup)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(self.last_epoch, 0)
        warmup_mult = 1.0 - self.warmup ** (1 + step)        # 0 → 1 as step → ∞
        decay_mult  = (1.0 + step / self.inv_gamma) ** (-self.power)
        return [base_lr * warmup_mult * decay_mult for base_lr in self.base_lrs]
```

> **注**: WaveFit-PT の InverseLR は厳密には `lr = base_lr * (1 - warmup ** (step+1)) * (1 + step/inv_gamma)**(-power)` の形をとる (`yukara-ikemiya/wavefit-pytorch/src/scheduler.py` 参照のうえで論文記述と整合確認)。実装時に再確認し、`tests/test_inverse_lr.py` で **step=0 / step=inv_gamma / step=10*inv_gamma での lr 値を pin** する。

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `lr_g` | 1.0e-4 | docs/training.md §2.4, docs/open-questions.md (確定) |
| `lr_d` | 2.0e-4 | docs/training.md §2.4 (G の 2 倍) |
| `betas` | [0.8, 0.99] | docs/training.md §2.4 |
| `weight_decay` | 1.0e-3 | docs/training.md §2.4 |
| `inv_gamma` | 200000 | docs/training.md §2.4 |
| `power` | 0.5 | docs/training.md §2.4 |
| `warmup` | 0.999 | docs/training.md §2.4 |
| `grad_clip_norm` | 1.0 | docs/training.md §2.4 |
| `max_steps` | 2,000,000 | docs/implementation-plan.md §5 (M6 で 410h × A100 想定) |
| `batch_size` | 16 | docs/implementation-plan.md §5 |
| `segment_length` | 16384 | docs/implementation-plan.md §5 (Vocos 流) |
| `num_workers` | 8 | docs/implementation-plan.md §5 |
| EMA | 不使用 | docs/open-questions.md (確定: Vocos / WaveFit-PT 共に未使用) |
| validation.interval_steps | 10000 | docs/training.md §6 (WaveFit-PT `n_step_test=10000`) |
| validation.num_utterances | 100 | docs/training.md §6 |
| validation.metric | mrstft_total (sc + mag) | docs/training.md §6 |
| checkpoint.interval_steps | 10000 | validation と同期 |
| loss weights | d_gan=1.0, d_fm=10.0, mrstft_sc=2.5, mrstft_mag=2.5 | docs/training.md §2.4 |
| amp dtype | bf16 (`--amp` 指定時) / fp32 (default) | T-M1.5 §6.1 (fp16 underflow リスク回避) |

### 2.4 アルゴリズム / 処理フロー

#### 1 step の交互更新 (擬似コード)

```
mel, x_gt = batch
# 1. Generator forward (with autocast if --amp)
y_hat = G(mel, audio_length=x_gt.shape[-1])   # T 個 sub-model の fixed-point iteration、初期 y_T=zeros

# 2. Discriminator update
d_real = D(x_gt)
d_fake = D(y_hat.detach())                     # G への grad を遮断
loss_D = relu(1 - d_real).mean() + relu(1 + d_fake).mean()  # hinge
loss_D.backward()
clip_grad_norm_(D.params, 1.0)
opt_D.step()
sch_D.step()

# 3. Generator update
d_fake_for_g = D(y_hat)                        # detach しない
d_real_for_fm = D(x_gt)                        # FM 用 (中間特徴を取り出す)
loss_g_gan  = -d_fake_for_g.mean()             # hinge G
loss_g_fm   = L1(d_real_for_fm.features, d_fake_for_g.features) * 10.0
loss_g_sc, loss_g_mag = mrstft(y_hat, x_gt)
loss_G = loss_g_gan + loss_g_fm + 2.5 * loss_g_sc + 2.5 * loss_g_mag
loss_G.backward()
clip_grad_norm_(G.params, 1.0)
opt_G.step()
sch_G.step()

# 4. Logging
if step % log_interval == 0: writer.add_scalar(...)
if step % validation.interval_steps == 0: run_validation()
if step % checkpoint.interval_steps == 0: save_checkpoint()
```

#### Validation (10k step ごと)

1. `G.eval()` / `torch.no_grad()` で `val_loader` (100 utterances) を走査
2. 各 utterance で `y_hat = G(mel, audio_length=audio.shape[-1])` を計算
3. MR-STFT (SC + Mag) を合計して utterance 平均
4. `state.best_val_mrstft` を更新 → `best.pt` 保存
5. 最初の 4 utterance を `writer.add_audio()` / `writer.add_image(mel_visualization)` で TensorBoard に出力
6. `G.train()` に戻す

#### Checkpoint (10k step ごと + best 更新時)

保存する key (resume 完全復元のため):
```python
ckpt = {
    "step": state.step,
    "best_val_mrstft": state.best_val_mrstft,
    "G_state_dict": G.state_dict(),
    "D_state_dict": D.state_dict(),
    "opt_G_state_dict": opt_G.state_dict(),
    "opt_D_state_dict": opt_D.state_dict(),
    "sch_G_state_dict": sch_G.state_dict(),
    "sch_D_state_dict": sch_D.state_dict(),
    "rng_state": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    },
    "config": cfg,  # 再現性チェック用
}
```

#### Resume (`--resume`)

1. `ckpt = torch.load(path, map_location="cpu")`
2. `G.load_state_dict(ckpt["G_state_dict"])` 以下 4 module + 4 optimizer/scheduler を復元
3. `torch.set_rng_state(ckpt["rng_state"]["torch"])` 等で RNG state も復元 (validation 100 utterances の再現性のため)
4. `state.step = ckpt["step"]` から再開
5. config drift 検知: `ckpt["config"]` と現在の `cfg` を diff し、不整合があれば warning (ただし続行)

### 2.5 CLI / config 連携

`configs/gan_wavenext2.yaml` に追加するキー (T-M0.2 で生成済みのものに加える):

```yaml
train:
  max_steps: 2_000_000
  grad_clip_norm: 1.0
  optimizer:
    type: AdamW
    lr_g: 1.0e-4
    lr_d: 2.0e-4
    betas: [0.8, 0.99]
    weight_decay: 1.0e-3
  scheduler:
    type: InverseLR
    inv_gamma: 200000
    power: 0.5
    warmup: 0.999
  amp:
    enabled: false       # --amp フラグで上書き可能
    dtype: bfloat16
validation:
  interval_steps: 10000
  num_utterances: 100
  metric: mrstft_total
  num_audio_samples: 4   # TensorBoard に書き出す utterance 数
checkpoint:
  dir: checkpoints/gan
  interval_steps: 10000
  keep_last_n: 5         # rolling delete (best.pt は除外)
logging:
  tensorboard_dir: logs/gan
  scalar_interval_steps: 100
```

CLI:
```bash
# 通常訓練
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml

# Resume
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml --resume checkpoints/gan/best.pt

# Smoke (1 step だけ)
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml --debug

# bf16 mixed precision
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml --amp
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer (train) | 1 | `train_gan.py` の訓練ループ本体 + checkpoint/resume + logging | general-purpose |
| Implementer (scheduler) | 1 | `scheduler.py` の `InverseLR` + 数式 pin テスト | general-purpose |
| Reviewer | 1 | docs/training.md §2 とのコード対応確認 + WaveFit-PT 非コピー確認 + hinge GAN 実装の symmetry 確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_train_gan.py tests/test_inverse_lr.py -v` + smoke (`--debug` で 1 step) 動作確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M2.1〜T-M2.4 すべての完了が前提、最重量の集約点)
- 並列実行する場合の最大並列数: 1 (Implementer が train_gan.py と scheduler.py を並行作業する場合のみ別 worktree で 2 並列可能、ただし `tests/test_train_gan.py::test_smoke_step` で scheduler を呼ぶため最終マージ前に scheduler が確定している必要あり)
- 後続 T-M2.6 (1000 step) / T-M5.1 (1 epoch) は本チケット完了後にそれぞれ別タスクとして起動

## 4. 提供範囲 (Scope)

### In Scope
- `train_gan.py`: `main()` (click CLI) + 1 step alternating training loop + checkpoint/resume + TensorBoard logging + validation 機能
- `scheduler.py`: `InverseLR` (warmup * decay の合成、`get_lr()` で `step=0/inv_gamma/10*inv_gamma` の値を pin)
- `tests/test_train_gan.py`: smoke step (1 step 実行 + loss finite + checkpoint 保存復元 + TensorBoard event file 生成確認)
- `tests/test_inverse_lr.py`: 数式 pin + warmup 域 / decay 域 / 漸近域の値検証
- `configs/gan_wavenext2.yaml` への train/validation/checkpoint/logging キー追加
- docstring (英文 + 日本語混在、`docs/training.md` §2.4 と weights を明示)
- `pyproject.toml` の `[project.scripts]` 追加 (任意)

### Out of Scope
- **1000 step overfitting テスト** (T-M2.6 で実施)
- **1 epoch 訓練** (T-M5.1 で実施)
- **2M step 本格訓練** (T-M6.1 で実施)
- **Discriminator / Loss の実装** (T-M2.2 / T-M2.3 で完成済前提)
- **Dataset の実装** (T-M2.1 で完成済前提)
- **GANWaveNext2 の実装** (T-M2.4 で完成済前提)
- **Multi-GPU / DDP** (single GPU 前提、M6.1 で必要なら DDP 化を別 PR)
- **fp16 mixed precision** (T-M1.5 §6.1 で hinge GAN の `relu(1-D)` underflow リスクが指摘されているため bf16 のみ)
- **EMA** (`docs/open-questions.md` 確定: Vocos / WaveFit-PT 共に未使用)
- **Lookahead optimizer / gradient penalty** (§8.1 代替案、本チケットでは採用しない)
- **`accelerate` / `pytorch-lightning` 採用** (素 PyTorch + click で実装)
- **wandb** (TensorBoard のみ。wandb は M6.1 で必要なら別途追加)
- **Hyperparameter search** (`docs/training.md` 値で固定)
- **Diff 用訓練スクリプト** (T-M3.2 で別途実装、ただし `InverseLR` は再利用される可能性あり)
- **TorchScript / `torch.compile`** (M5/M6 で必要なら別 PR)

### Deliverable
- ファイル:
  - `src/wavenext2/train/train_gan.py` (新規実装)
  - `src/wavenext2/utils/scheduler.py` (新規実装)
  - `tests/test_train_gan.py` (新規実装)
  - `tests/test_inverse_lr.py` (新規実装)
  - `src/wavenext2/train/__init__.py` (re-export `main` 追加)
  - `src/wavenext2/utils/__init__.py` (re-export `InverseLR` 追加)
- 関数 / クラス:
  - `def main(config_path, resume_path, debug, amp) -> None` (click CLI entry point)
  - `class InverseLR(LRScheduler)`
  - 補助関数: `build_loaders(cfg)`, `load_checkpoint(...)`, `save_checkpoint(...)`, `run_validation(...)`, `log_scalars(...)`, `iter_forever(loader)`, `TrainState` dataclass
- 設定:
  - `configs/gan_wavenext2.yaml` に `train`, `validation`, `checkpoint`, `logging` セクション追加
- ドキュメント差分:
  - `docs/milestones.md` §M2.5 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M2.5 ステータス更新
  - (該当時) `docs/training.md` §2 にコード参照リンクを追記

## 5. テスト項目

### 5.1 Unit テスト

#### `tests/test_inverse_lr.py`
- [ ] `test_lr_at_step_zero`: `InverseLR(opt, inv_gamma=200000, power=0.5, warmup=0.999)` で `step=0` のときの lr が `base_lr * (1 - 0.999) * 1.0 = base_lr * 0.001` (warmup 域)
- [ ] `test_lr_at_step_inv_gamma`: `step=200000` で `lr ≈ base_lr * (1 - 0.999^200001) * (1 + 1)^(-0.5) = base_lr * 1.0 * 0.707 ≈ base_lr * 0.707`
- [ ] `test_lr_at_step_10x_inv_gamma`: `step=2_000_000` で `lr ≈ base_lr * 1.0 * (1 + 10)^(-0.5) = base_lr * 0.302`
- [ ] `test_lr_monotonic_after_warmup`: `step >= inv_gamma` の領域で lr が **単調減少**
- [ ] `test_get_last_lr_consistency`: `get_lr()` と `get_last_lr()` が `step()` 直後で一致
- [ ] `test_state_dict_roundtrip`: `state_dict()` → 新 scheduler → `load_state_dict()` で `last_epoch` / `base_lrs` が完全復元
- [ ] `test_two_optimizers_independent`: G/D 別 scheduler が各 base_lr (1e-4, 2e-4) を独立に管理し、ratio 2:1 が全 step で保たれる

#### `tests/test_train_gan.py`
- [ ] `test_train_gan_step`: **`train_gan_step` を直接呼ぶ smoke** で 1 step backward が動き、戻り値 dict のキー (`loss_G`, `loss_D`, `loss_g_gan`, ...) が揃っていて全 value が `isfinite` (公開関数 API の検証、§5.5 / §8.1 採用昇格)
- [ ] `test_smoke_step`: tmp config + tmp dataset (T-M2.1 fixture or random tensor) で 1 step 実行 → `loss_G`, `loss_D` が `isfinite` (milestones.md §M2.5 Acceptance #1)
- [ ] `test_alternating_update_order`: D 更新後の D parameter と G 更新後の G parameter が **同 step 内で別々に変化** していることを `param.clone()` で前後比較
- [ ] `test_grad_clip_applied`: `clip_grad_norm_` が呼ばれた直後の grad norm が `<= 1.0 + ε`、戻り値の `grad_norm_G` / `grad_norm_D` が clip 前の norm を返すことも確認
- [ ] `test_no_ema`: model 内に EMA shadow parameter が存在しないこと (`docs/open-questions.md` 確定)
- [ ] `test_checkpoint_save_load_roundtrip`: 1 step 実行 → save → 新インスタンス + load → step / opt state / sch state / RNG state / model state が完全一致 (milestones.md §M2.5 Acceptance #2)
- [ ] `test_atomic_best_pt_rename`: `best.pt` 保存中に kill しても **中途半端な状態にならない** (`save_checkpoint(atomic=True)` で `best.pt.tmp` → `os.replace`、T-M2.4 申し送り)
- [ ] `test_resume_continues_step`: `--resume` で `state.step` がチェックポイントから再開、`sch_G.last_epoch` も一致
- [ ] `test_tensorboard_event_written`: 1 step 後に TensorBoard event file が `logs/gan/` に生成され、`loss_G`/`loss_D`/`lr_G`/`lr_D` の scalar が存在 (milestones.md §M2.5 Acceptance #3)
- [ ] `test_validation_runs`: `--debug` モードでも 1 回 validation を強制呼び出し可能なヘルパで MR-STFT total が `isfinite`
- [ ] `test_amp_bf16`: `--amp` 指定時に `y_hat.dtype in (torch.float32, torch.bfloat16)` (autocast 境界では float32 戻し)
- [ ] `test_amp_autocast_boundary`: **hinge GAN loss が fp32 で計算される** ことを assert (`--amp` 有効でも `loss_D.dtype == torch.float32` および `loss_g_gan.dtype == torch.float32`、§6.1 通常項目「mixed precision」)
- [ ] `test_hinge_d_loss_symmetry`: `D(real)=0, D(fake)=0` の場合 `loss_D = 2.0` (hinge formula の sanity check)
- [ ] `test_hinge_g_loss_sign`: `D(fake)` の値が大きいほど `loss_g_gan` が小さい (`-D(fake).mean()` の sign 確認)
- [ ] `test_seed_determinism`: 同 seed + 同 config + 同 batch で 1 step 後の `G.state_dict()` が deterministic
- [ ] `test_lr_ratio_2_to_1`: `sch_D.get_last_lr()[0] / sch_G.get_last_lr()[0] ≈ 2.0` を全 step で確認
- [ ] `test_sigterm_emergency_save`: `os.kill(os.getpid(), SIGTERM)` を別スレッドから送信し、`emergency_step_N.pt` が生成されることを確認 (§8.1 採用昇格)
- [ ] `test_d_loss_below_threshold_counter`: `train_gan_step` を mock D で 1000 回呼び `state.d_loss_below_threshold_steps == 1000` を確認、warning ログが出力されること (§6.1 「D 強すぎ問題」)
- [ ] `test_worker_init_fn_seed`: 2 epoch 目の最初の batch crop offset が 1 epoch 目と異なることを確認 (`worker_init_fn` で `epoch * num_workers + worker_id` を seed する罠の検証)

### 5.2 e2e / 結合テスト
- [ ] `test_real_audio` (T-M0.3 完了後、`@pytest.mark.slow` で skip 可): LibriTTS-R 1 utterance で 5 step 実行、loss が `isfinite`、出力 audio が `[-1, 1]`
- [ ] T-M2.6 (smoke training 1000 step) の入口として `main()` が CLI から呼べる (`subprocess.run(["uv", "run", "python", "-m", "wavenext2.train.train_gan", "--debug", "--config", tmp_cfg])`)
- [ ] T-M5.1 (1 epoch) の準備として `train_loader` が StopIteration せず 33k step 回せる (`iter_forever` の正常動作)

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.5 より転記)
- [ ] 1 step 実行で `loss_G`, `loss_D` が finite
- [ ] checkpoint 保存・復元が動作
- [ ] TensorBoard に loss / 各 sub-loss / sample audio が記録される

### 5.4 追加 acceptance (本チケット独自)
- [ ] `InverseLR` の数式が `tests/test_inverse_lr.py` で step=0 / step=inv_gamma / step=10*inv_gamma の 3 点で pin され、将来の式変更を CI が検知
- [ ] CLI の 3 フラグ (`--config` 必須 / `--resume` / `--debug` / `--amp`) が click ヘルプに表示
- [ ] `pyproject.toml` の `[project.scripts]` で `train-gan` コマンドが登録 (任意)
- [ ] `uv run pytest tests/test_train_gan.py tests/test_inverse_lr.py -v` が **exit code 0** で完了

### 5.5 テスト戦略 (M2 phase review 申し送り想定)

- **GPU テスト分離**: `@pytest.mark.gpu` で GPU 必須テストを CI runner 別に分離。`test_amp_bf16` は GPU マーカー必須
- **`@pytest.mark.slow` マーカー**: `test_real_audio` 等 LibriTTS-R 実音声を使うテストは slow でデフォルト除外
- **CI 時間目標**: `tests/test_train_gan.py` 全体 **30 秒以内** (smoke step 1 つで GANWaveNext2 (T=4) の forward+backward が ~5 秒、validation 1 utterance が ~3 秒、checkpoint save/load が ~2 秒)
  - **必須最適化**: `scope="module"` fixture で `G`, `D`, optimizer, scheduler を再利用 (各テストで init し直さない)
  - **batch_size=1, T=2 で最小化**: smoke では `T=2` (full 設定の半分) でメモリと時間を削減
- **coverage 目標**: 本チケットのカバレッジ目標 **80%** (Resume の異常系 / config drift warning は手動テスト)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (実装着手前に必ず解決)

- **CRITICAL: InverseLR の数式実装が WaveFit-PT 原典と一致するか**:
  - 本チケットの暫定式は `lr = base_lr * (1 - warmup^(step+1)) * (1 + step/inv_gamma)^(-power)` だが、WaveFit-PT (`yukara-ikemiya/wavefit-pytorch/src/scheduler.py`) の原実装が `warmup` を別解釈 (例: linear warmup を併用) している可能性
  - **本チケット実装着手時の MUST DO**:
    1. WaveFit-PT の InverseLR 実装を **目視確認** (コピーは禁止だが式は確認する)
    2. `tests/test_inverse_lr.py` で step=0, 1000, 200000, 2000000 の 4 点で WaveFit-PT 値と一致することを pin
    3. 不一致なら docs/training.md §2.4 と整合する側を採用、再評価ログを §8.3 に追記
  - **未解決のままだと**: M6 本格訓練で論文と異なる learning rate スケジュールになり 410h を無駄にする

- **CRITICAL: Discriminator 過学習 (D loss が極小化、G が学習しない)**:
  - hinge GAN は D が強くなりすぎると `loss_g_gan` の grad が消失するリスク
  - **緩和**: lr 比 G:D=1:2 はすでに `docs/training.md` で確定済 (G 側を 2 倍速くしない・D 側を 2 倍速くする) ← 一見逆だが論文 + WaveFit-PT 準拠
  - **検知**: TensorBoard で `loss_D < 0.01` が連続 1000 step 続いたら警告ログ
  - **対応 (M5/M6 で発覚した場合)**: §8.1 代替案 「D を 2 step に 1 回更新」へ切替

- **CRITICAL: T=4 × batch_size=16 × sub-model 4 段で OOM**:
  - GANWaveNext2 は T=4 で 60M params + activation 4 倍。fp32 で 24GB GPU (RTX 4090) では OOM の可能性
  - **緩和**: T-M2.4 で `enable_grad_ckpt=True` を YAML config 経由で propagation する設計を要請済 (T-M1.6 §9.1)
  - **本チケット側の責務**: `train.amp.enabled=true` / `model.enable_grad_ckpt=true` を YAML config に明示し、`--amp` フラグでの bf16 切替で activation memory 半減を保証
  - **検知**: `test_smoke_step` で `torch.cuda.max_memory_allocated()` を pin (例: T=4 batch=16 fp32 で < 22GB)

#### 通常項目

- **mixed precision (bf16) で hinge GAN の `relu(1 - D)` が underflow するリスク**:
  - `D(real)` が **正の大きな値** の場合 `1 - D(real) < 0` → relu でゼロ。bf16 では精度低下で `relu(1-D)=0` になりやすく D の学習が止まる
  - **対応**: hinge GAN の計算は **fp32 で実行** (autocast 境界で float32 戻し)、autocast は generator forward + discriminator forward のみに限定
  - **検証**: `test_hinge_d_loss_symmetry` を bf16 でも実行し loss 値が fp32 と ε 以内で一致
- **`InverseLR` の `last_epoch` が `optimizer.step()` 前の `scheduler.step()` で誤動作**:
  - PyTorch の慣例: `optimizer.step()` → `scheduler.step()` の順。逆順だと UserWarning が出る
  - **対応**: 訓練ループで明確にこの順序を守る (上記 2.4 の擬似コード)。test_alternating_update_order で順序検証
- **Resume 時の RNG state 復元の不完全性**:
  - `torch.set_rng_state` だけでは `cuda` rng / `numpy` rng / Python `random` rng が復元されない
  - **対応**: 上記 2.4 のチェックポイント仕様で 3 つ全部保存 + `random.setstate()` も追加
  - **検証**: `test_resume_continues_step` で 2 step 目の loss が resume の有無で完全一致
- **Validation utterance の選択非決定性**:
  - `val.tsv` (T-M0.3) から 100 utterances を fixed order でロードする必要がある
  - **対応**: `LibriTTSRDataset(filelist, mode="val")` で `shuffle=False` + sorted order を保証 (T-M2.1 §6 に申し送り)
  - validation の `DataLoader` は `shuffle=False`, `drop_last=False`, `batch_size=1`
- **TensorBoard の sample audio 容量**:
  - `add_audio` は 24kHz × 数秒 × 4 utterance × (10k step / 10k validation) = 数 GB に膨らむ可能性
  - **対応**: `validation.num_audio_samples=4` で限定、`writer.add_audio(..., global_step=state.step)` で TensorBoard 側で間引き
- **`iter_forever` の DataLoader worker 再起動コスト**:
  - epoch 境界で worker が再起動すると 1 epoch あたり数十秒ロス
  - **対応**: `persistent_workers=True` を `DataLoader` に渡す
- **CPU でのテスト実行**:
  - smoke step を CPU で実行すると GANWaveNext2 forward+backward が数十秒
  - **対応**: `T=2`, `batch_size=1`, `segment_length=4096` の最小 config を `tests/conftest.py` で fixture 化 (T-M0.2 / T-M1.6 §9.1 に申し送り済)
- **checkpoint 累積による disk full**:
  - 10k step ごとに save すると 2M step で 200 checkpoint × 数百 MB = 数十 GB
  - **対応**: `checkpoint.keep_last_n=5` で rolling delete (`best.pt` は除外)、save 後に古い `step_*.pt` を削除
- **D の hinge loss が batch ごとに不均衡**:
  - real / fake サンプルがバッチで非対称だと `loss_D` の magnitude が振動
  - **対応**: 各 batch で `loss_real = relu(1-d_real).mean()` と `loss_fake = relu(1+d_fake).mean()` を **別々に TensorBoard ログ** して観察可能にする
- **`iter_forever` + `persistent_workers=True` の組み合わせの seed 罠**:
  - DataLoader が worker 再起動なしに新 epoch に入ると random crop seed が更新されず **同じ crop が繰り返される** 可能性
  - **対応**: `worker_init_fn` で `epoch * num_workers + worker_id` を seed として渡す (T-M2.1 §9.1 のテンプレ snippet を採用)
  - **検証**: 2 epoch 目の最初の batch が 1 epoch 目の最初の batch と異なる crop offset であることを確認
- **`prefetch_factor` × `persistent_workers=True` の validation leak**:
  - `prefetch_factor=2` (default) で `num_workers=8` だと **16 batch** が常に prefetch される。validation 中も persistent_workers が leak して GPU memory を圧迫する可能性
  - **対応**: validation の `DataLoader` を **train とは別 instance** にして `persistent_workers=False`、validation 完了後に `del val_loader_iter` で明示的に閉じる
- **disk full 監視**:
  - `checkpoint.keep_last_n=5` でも `logs/gan/events.out.tfevents.*` が 2M step で数 GB に膨張
  - SummaryWriter 内 `max_queue=10000` の buffer も memory leak 候補
  - **対応**: `writer.flush()` を **10k step ごと明示**、`max_queue=1000` に縮小、`logs/` の disk usage を 100k step ごとに `shutil.disk_usage()` で確認しログ警告
- **D 強すぎ問題の検知 (実装 scope 明記)**:
  - `loss_D < 0.01 for 1000 steps` で alert/log を **本チケットの実装 scope に含める** (現状 §6.1 で言及あり、メトリクス TrainState に history buffer を保持して連続 step 数をカウント)
  - 検知時は warning ログのみ (自動切替はしない、§8.1 再評価トリガーで M5 に判断を持ち越し)
- **AMP autocast 境界の実装明示**:
  - hinge GAN を fp32 で計算する旨は §8.1 にあるが、コード骨格 §2.2 で `loss_D = crit_gan.d_loss(...)` が autocast context 外に出ていない実装は **混乱を招く**
  - **対応**: §2.2 擬似コード (`train_gan_step`) で `with torch.autocast(..., enabled=amp)` (G/D forward) と `with torch.autocast(..., enabled=False)` (loss 計算) を **明示的に nest** する構造に修正済 (上記擬似コード参照)
  - **検証**: `tests/test_train_gan.py::test_amp_autocast_boundary` で hinge GAN loss が `--amp` 時も fp32 で計算されることを assert (dtype 検査)
- **再現性 (`torch.use_deterministic_algorithms`)**:
  - `uv.lock` + `torch.manual_seed` + `np.random.seed` だけでは GPU の atomic operations や cuDNN の non-deterministic algorithm により完全な再現は不可
  - **対応**: M6 ablation で論文 MOS 差を測る際に必須、本チケットでは `--deterministic` フラグ (default off) で `torch.use_deterministic_algorithms(True)` + 環境変数 `CUBLAS_WORKSPACE_CONFIG=:4096:8` を提供
  - **注意**: 本フラグ有効時は throughput が 30% 程度低下するため、本格訓練では off、ablation 比較時のみ on
- **TensorBoard `add_audio` のストレージ膨張**:
  - `validation.num_audio_samples=4` × 10k step interval = 200 validation × 4 utterance = **800 audio sample** = 数 GB 規模
  - **対応**: `num_audio_samples=2` に半減 + mel spectrogram 画像のみで音声非保存 (`add_image` のみで `add_audio` 省略) の代替案を `configs/gan_wavenext2.yaml` でデフォルトに採用
  - 音声を残す場合は `validation.audio_interval_steps=100000` で `add_audio` 頻度を validation 自体より下げる

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み**。
  - Optimizer / lr / betas / wd / scheduler / grad clip / EMA 不使用 → docs/training.md §2.4 で確定
  - Loss 重み (d_gan=1.0, d_fm=10.0, mrstft_sc=2.5, mrstft_mag=2.5) → 確定
  - Validation metric (mrstft_total) / 頻度 (10k step) → docs/training.md §6 で確定
- 旧来の判断ポイント (本チケット作成時に確定):
  - **TensorBoard vs wandb**: TensorBoard を採用 (Vocos / WaveFit-PT 慣例)。wandb は M6.1 で要望があれば別 PR
  - **`accelerate` / `pytorch-lightning` 採用**: 採用しない (素 PyTorch + click)。理由は §8.1 参照
  - **D 更新と G 更新の順序**: D → G の順 (HiFi-GAN / WaveFit-PT 慣例、§2.4 擬似コード参照)
  - **`y_hat.detach()` のタイミング**: D update の入力で detach。G update 時は D に gradient を流すが D の重みは更新しない (`opt_D.zero_grad()` を G update 後に呼ばないことで保証)
  - **fp16 vs bf16**: bf16 のみ。fp16 は T-M1.5 §6.1 で underflow リスク (sinusoidal embedding の小さい値) が指摘されており本チケットでも hinge GAN の `relu(1-D)` underflow が新たな懸念
- `iter_forever` 実装方針: `while True: yield from loader` で十分 (DataLoader が `persistent_workers=True` なら StopIteration しない)

### 6.3 他チケットとの整合性

- **T-M2.1 (Dataset)** との整合:
  - 期待 signature: `LibriTTSRDataset(filelist, segment_length, mode)`、`__getitem__` 戻り値 `{"mel": (128, T_mel), "audio": (segment_length,)}` (dict-based)
  - 不整合があった場合は T-M2.1 側を修正
  - **validation 用**: `mode="val"` で gain=-3 dB 固定、`shuffle=False` を本チケット (DataLoader) で指定
- **T-M2.2 (Discriminator)** との整合:
  - 期待 signature: `D(audio_unsqueezed) -> list[(scalar_pred, list[features])]` (3 sub-discriminator)
  - FM loss はこの中間 features を使う
  - 不整合があった場合は T-M2.2 側を修正
- **T-M2.3 (Losses)** との整合:
  - 期待 class: `HingeGANLoss.d_loss(d_real, d_fake)`, `HingeGANLoss.g_loss(d_fake)`, `FeatureMatchingLoss(loss_type="l1")(d_real, d_fake)`, `MultiResolutionSTFTLoss(...)(y_hat, y_gt) -> (sc, mag)`
  - 不整合があった場合は T-M2.3 側を修正
- **T-M2.4 (GANWaveNext2)** との整合:
  - 期待 signature: `G(mel, audio_length: int) -> (B, audio_length)` (T-M1.6 §9.1 の暫定設計に依拠)
  - 期待 attribute: `G.T` (sub-model 数)、`from_config(cfg)` factory
  - **戻り値が clip(-1, 1) 済の波形** であること (T-M1.6 §6.1 critical の解決待ち、本チケット実装着手前に確認)
- **T-M2.6 (smoke training)** へ渡す情報:
  - `main()` を `subprocess.run([..., "--debug"])` で呼ぶ前提
  - 1000 step は本チケットでは扱わない (T-M2.6 で実装)
- **T-M3.2 (Diff 訓練スクリプト)** との整合:
  - `InverseLR` は M3 では使われない (Diff 側は固定 lr=2e-4) が、`scheduler.py` 自体は共通モジュール化しておく
  - 訓練ループ構造の **共通点 (logging / checkpoint / resume)** は本チケットで確立した API を T-M3.2 で再利用可能にする (例: `save_checkpoint(model, opt, sch, state, path)` を generic 化、ただし本チケットでは GAN 専用実装で十分、refactor は T-M3.2 の §8 で再検討)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §2 (訓練フロー、optimizer、scheduler、loss 重み) と整合
- [ ] `docs/open-questions.md` 関連項目 (EMA 不使用、optimizer 設定、validation 基準) と整合
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 3 項目クリア、5.4 追加 acceptance 全クリア
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 必須 CLI 引数欠落、checkpoint ファイル不存在、config drift 警告が適切に出る
- [ ] **参考実装 (WaveFit-PT / HiFi-GAN / Vocos) をコピーしていない** (CLAUDE.md 末尾、`docs/training.md` §2 記述から再構成しているか)
- [ ] InverseLR の数式 (`tests/test_inverse_lr.py` の 3 点 pin) が WaveFit-PT 原実装と一致
- [ ] hinge GAN 実装が `D: relu(1-D(real)).mean() + relu(1+D(fake)).mean() / G: -D(fake).mean()` で対称性が保たれている
- [ ] 交互更新の順序 (D → G) と `y_hat.detach()` のタイミング (D update のみ) が正しい
- [ ] `clip_grad_norm_` が `backward()` 後 / `step()` 前のタイミング
- [ ] checkpoint resume で step / opt / sch / RNG state が完全復元 (test_resume_continues_step で 2 step 目 loss 一致)
- [ ] TensorBoard が `loss_G`, `loss_D`, sub-loss 4 種, `lr_G`, `lr_D`, sample audio, mel spectrogram を出力
- [ ] EMA 関連コードが **一切存在しない** (dead code 含めて)
- [ ] `--amp` 時に hinge GAN 計算が fp32 で実行され bf16 underflow を回避
- [ ] `iter_forever` が `persistent_workers=True` で worker 再起動コストを回避
- [ ] CPU でテストが pass (CUDA 不要、ただし `--amp` は GPU 必須 → `@pytest.mark.gpu`)
- [ ] `pyproject.toml` の `[project.scripts]` 追加 (任意、追加した場合は `uv run train-gan --help` が動作)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M2) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **素 PyTorch + click CLI で 1 ファイル (`train_gan.py`) に集約**
  - 理由 (3 観点):
    - (a) `accelerate` / `pytorch-lightning` は **抽象レイヤが厚すぎ**、論文の交互更新 / fixed-point iteration / 4 sub-model 独立訓練のような **非標準パターン** で hook が増えて可読性低下
    - (b) 1 ファイルに収めると `main()` を読むだけで訓練フロー全体が把握でき、再現実装の本来目的 (論文記述とコードの 1 対 1 対応) に合致
    - (c) M3.2 (Diff 訓練) では訓練フローが大きく異なる (4 sub-model 独立訓練 + MSE only) ので、共通抽象化のメリットが薄い

- **【採用昇格】`train_gan_step(G, D, opt_G, opt_D, mel, audio, cfg) -> dict[str, float]` を公開関数として切り出し** (極めて重要、横断的):
  - 理由 1: T-M2.6 smoke が `from train_gan import train_gan_step` で再利用、`scripts/smoke_gan.py` は薄い wrapper に
  - 理由 2: T-M3.2 (train_diff) で `train_diff_step` を同 signature にすると `tests/conftest.py` の fixture 共用が可能 (`G`, `D`, optimizer, scheduler を再利用)
  - 理由 3: 将来 `pytorch-lightning` 移行時にロジックを `LightningModule.training_step` に移すだけで済む (Lightning-ready architecture)
  - 旧案 (`main()` 内インライン実装) は §1〜§8.0 で採用していたが、M2 phase review で採用昇格

- **【採用昇格】SIGTERM/SIGINT handler で emergency checkpoint 保存**:
  - `signal.signal(SIGTERM, lambda *_: save_emergency_ckpt())` を `main()` 起動直後に登録
  - 理由: M6 cluster preemption (Slurm / Kubernetes) で SIGTERM 受信 → 数十秒の grace period → SIGKILL のシーケンスに対応必須、現状完全欠落
  - 検証: `tests/test_train_gan.py::test_sigterm_emergency_save` で `os.kill(os.getpid(), SIGTERM)` 後に `emergency_step_N.pt` が生成されることを確認

- **【採用昇格 (M3.2 着手前必須)】`utils/training_loop.py` への共通化**:
  - 本チケットでは `train_gan.py` 内に `train_gan_step` を実装
  - **M3.2 着手前に** `save_checkpoint` / `load_checkpoint` / `iter_forever` / `log_scalars` / `run_validation` を `utils/training_loop.py` に切り出す refactor を実行
  - 旧 §6.3 「本チケットでは GAN 専用実装で十分」は撤回、M3.2 着手前必須に格上げ
  - 再評価トリガー: M2.6 smoke pass 直後

- **`InverseLR` を自作 (PyTorch 標準にないため)**
  - 理由: WaveFit-PT 慣例 + 論文記述。代替は `ExponentialLR` (warmup なし) / `CosineAnnealingWarmRestarts` (周期型) で論文 schedule に合致しない

- **TensorBoard を採用 (wandb ではない、ただし M5.1 で前倒し再評価)**
  - 理由: 個人開発 + offline 訓練前提で wandb account 不要
  - **【再評価トリガー前倒し】**: 旧案では M6.1 で再評価だったが、複数 ablation 比較が **M5 から発生想定** のため **M5.1 phase review で wandb 採用判断**
  - 実装は SummaryWriter の wrapper として追加可能 (`wandb.tensorboard.patch(...)` で既存コード変更最小)

- **`--amp` は bf16 のみ (fp16 不採用)**
  - 理由: T-M1.5 §6.1 で sinusoidal embedding の小さい値が fp16 で underflow するリスクが指摘済。hinge GAN の `relu(1-D)` も同じく underflow リスクで本チケットで再確認

- **`accelerate` 不採用判断を M6 DDP 時に必ず再評価 (固定)**:
  - 現状「必要なら別 PR」が緩い表現 → **M6.1 着手前のレビュー必須項目** として固定
  - 判断軸: DDP / FSDP / multi-node を素 PyTorch で書くコストと `accelerate.prepare(...)` の resume 互換性検証コストの比較

#### Deprecated (旧案、却下根拠)

1. **`pytorch-lightning` Trainer 採用**
   - メリット: ckpt 管理 / DDP / mixed precision / TensorBoard ロガーが標準提供
   - 却下: 上記 (a)(b)(c)。論文記述と Lightning Module の hook 名 (`training_step` / `validation_step`) の対応が抽象化されすぎて読みにくい

2. **`accelerate` library 採用**
   - メリット: DDP / mixed precision を 1 行で導入可能
   - 却下: 単 GPU 想定の M0〜M5 では恩恵が薄い、M6.1 で必要なら本チケット完了後に別 PR で追加。`accelerate.prepare(model, opt, dataloader)` が optimizer state を wrap する仕様で resume の検証が増える

3. **D を毎 2 step に 1 回更新 (1:2 → 1:1)**
   - メリット: D 過学習防止
   - 却下: 論文 / WaveFit-PT は 1:1 更新 (lr 比 1:2 で代替)。1:2 更新は §8.1 §再評価トリガー → M5 で D loss が極小化したら採用

4. **Gradient penalty (WGAN-GP 風)**
   - メリット: G 安定化
   - 却下: hinge GAN と無相関、論文 / WaveFit-PT に記述なし。M6 で divergence が出たら検討

5. **Lookahead optimizer (Adam の wrapper)**
   - メリット: 収束安定、SOTA 報告多数
   - 却下: 論文記述なし、AdamW で十分との論文記述に従う

6. **TensorBoard ではなく wandb**
   - メリット: cloud sync、team collab
   - 却下: 個人開発前提、M6.1 で必要なら別 PR

7. **fp16 mixed precision を採用**
   - メリット: メモリ半減、速度向上
   - 却下: hinge GAN の `relu(1-D)` underflow + sinusoidal embedding underflow リスク (M3 用 InverseLR は使わないが Diff 側 sinusoidal で同様の懸念)。bf16 で代替

8. **`omegaconf` / `hydra` 採用**
   - メリット: config compose、CLI override
   - 却下: YAML 直読み + click で十分、依存追加のメリットが薄い

9. **Loss weight を `nn.Parameter` 化して学習**
   - メリット: 自動 balance
   - 却下: 論文記述 (`d_fm=10.0`, `mrstft=2.5`) が明示されているため固定値で再現性優先

10. **訓練ループを `Trainer` クラスにカプセル化**
    - メリット: テスト容易性、機能拡張容易
    - 却下: M2 段階では `main()` 関数 1 つで十分、M5/M6 で hook 追加が必要なら refactor

#### 追加検討した設計案

- **`InverseLR` の `warmup` を **linear warmup** (`min(1.0, step / warmup_steps)`) に置換**: 採用しない (WaveFit-PT 原実装が exponential warmup の `0.999^step` 系列のため、原典準拠)
- **Validation を別 process で並列実行**: 採用しない (M2 では 10k step に 1 回で同期実行で十分、M6 で時間が問題になったら別 process 化)
- **Mixed precision を per-loss 制御 (hinge GAN だけ fp32)**: 採用 (§6.1 通常項目「mixed precision」参照、autocast を generator/discriminator forward のみに限定)
- **【検討追加】pydantic / dataclass-based config schema**:
  - 現状の `yaml.safe_load` だけでは key typo (例: `mrstft_sc` vs `mr_stft_sc`) が **runtime まで検出されない**
  - pre-commit `check-yaml` は YAML syntax のみで schema 不問
  - **採用判断**: 現時点では未採用 (依存追加のコスト > メリット)、**再評価トリガー: M5.1 で config 変更頻発 / typo が原因の 1 epoch ロスが発生したら採用**
  - 採用時は `pydantic.BaseModel` で `TrainConfig`, `LossWeightsConfig` 等の階層を定義、`load_config()` でバリデーション

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| `pytorch-lightning` 不採用 | M6 完了時 | DDP / multi-node が必要になったら採用検討 |
| TensorBoard 単独 (wandb 不採用) | **M5.1 phase review (前倒し)** | 複数 ablation 比較が M5 から発生想定 → wandb 採用 |
| `accelerate` 不採用 | **M6.1 着手前 (必須レビュー、固定)** | DDP / FSDP のコスト比較 |
| D 1:1 更新 | M5 / M6 中 | D loss が極小化し G が学習しない場合 1:2 化 |
| fp16 不採用 | M6 完了時 | bf16 が GPU で使えない (古い GPU) 場合 fp16 + loss scaling を検討 |
| Loss weight 固定 | M6 ablation 時 | weight 微調整で品質が上がる場合 |
| Validation 同期実行 | M6 完了時 | 10k step の停止時間が問題になったら別 process 化 |
| `InverseLR` exponential warmup | M5 phase review | linear warmup でも同等品質なら簡易な方を採用 |
| pydantic / dataclass config schema | M5.1 (config 変更頻発時) | typo 由来の 1 epoch ロスが発生したら採用 |
| `utils/training_loop.py` 共通化 | **M2.6 smoke pass 直後 (M3.2 着手前必須)** | `save_checkpoint` / `iter_forever` / `log_scalars` / `run_validation` を切り出し |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切。
  - 訓練ループは「Dataset / Discriminator / Loss / Model / Optimizer / Scheduler / Logging / Checkpoint」の集約点であり、独立チケット化することで責務が明確
  - 1 ファイル + 1 scheduler ファイルで完結し、後続 T-M2.6 / T-M5.1 / T-M6.1 はこの entry point を呼ぶだけ
- **別マイルストーンに移すべき部分はないか**: なし。
  - `InverseLR` は `utils/scheduler.py` に分離するが M3.2 (Diff) では使わないため M3 への移動はしない
  - Checkpoint / Resume は GAN 専用実装で十分、M3.2 で必要なら別実装
- **インターフェース定義の見直し余地**:
  - **`main()` を click CLI vs argparse**: click を採用 (Python ecosystem 慣例、`pyproject.toml` `[project.scripts]` で entry point 化が容易)
  - **TrainState dataclass の公開**: 公開する (`from wavenext2.train.train_gan import TrainState` で resume の確認テストから参照可能)
  - **`save_checkpoint` / `load_checkpoint` を `utils/checkpoint.py` に分離**: M3.2 で再利用するなら分離、本チケットでは `train_gan.py` 内 private 関数で十分 (Diff 側で必要になった時に refactor)

- **【新原則】「training_step 関数」と「main loop」の分離**:
  - 「1 step backward 動作 (`train_gan_step`)」と「loop / checkpoint / validation の管制 (`main`)」を **完全分離**
  - 利点 1: テストは `train_gan_step` 単体で 1 step 検証可能 (`main()` 全体を呼ぶ必要なし、CI 高速化)
  - 利点 2: 後で Lightning に移行する場合も `train_gan_step` を `LightningModule.training_step` に移すだけで済む
  - 利点 3: T-M3.2 で `train_diff_step` を同 signature にして `tests/conftest.py` の fixture 共用可能
  - この原則は M3.2 / 後続全ての訓練スクリプトで踏襲

#### M1 phase review で確立した原則の適用

- **factory パターン**: T-M1.6 §8.2 で確立した factory 一貫化に従い、`GANWaveNext2.from_config(cfg["model"])` / `MultiScaleDiscriminator.from_config(cfg["discriminator"])` 経由で生成 (T-M2.4 / T-M2.2 に申し送り)
- **config drift 耐性**: `from_config` 内で `allowed_keys` 絞り込みにより、新 key 追加でクラスが壊れない
- **`scope="module"` fixture**: M1 で確立したテスト最適化を本チケットでも採用 (`tests/test_train_gan.py` で `G`, `D` を再利用)

#### 再評価トリガー (M2 phase review 時)

- 1000 step smoke (T-M2.6) で loss curve が想定外 (NaN / divergence / D 極小化) なら、本チケットの hinge GAN / lr / scheduler を見直す
- 1 epoch (T-M5.1) で validation MR-STFT が初期値の 30% を切らなければ、loss weight / amp / grad clip を見直す

### 8.3 学んだこと (2026-05-28 完了後追記)
- **CLI は click ではなく argparse**: `click` は依存に無く、依存追加を避けるため stdlib argparse を採用。`main(argv=None)` で `parser.parse_args(argv)`、`[project.scripts] train-gan` も登録。
- **実 API と ticket 擬似コードの差異を吸収**:
  - `MultiResolutionSTFTLoss(y_hat, audio)` は **`(sc, mag)` の 2 tensor を返す** (ticket の `(total, unweighted_dict)` ではない)。重み付けは train_gan_step 側で実施。
  - Discriminator は `list[SubDiscOutput(logits, features)]` を返すため、hinge は `[o.logits for o in outs]`、FM は `[[f for f in o.features] for o in outs]` を抽出して渡す。`HingeGANLoss.d_loss/g_loss` と `FeatureMatchingLoss` は logit/feature の list を受ける。
  - `SubModelGAN.from_config` / `MultiScaleDiscriminator.from_config` は `mode=` 引数を持たない (ticket の `from_config(cfg, mode="gan")` は誤り) → `from_config(cfg)` で呼ぶ。
- **生成波形 crop (T-M2.4 申し送りの実装地点)**: `G(mel)` は `T_mel*hop` を出し常に `segment_length` より長い (center=True の +1 frame)。train_gan_step / run_validation で `y_hat = G(mel)[..., :audio.shape[-1]]` と GT 長に crop。pad は不要 (gen 長 > seg が常に成立)。
- **config schema**: `validation`/`checkpoint`/`logging` を **top-level** に置く設計に統一 (ticket §2.5 準拠)。既存 yaml は `train.validation` 入れ子だったため移動。`model.sub_model` の nested 値は `GANWaveNext2.from_config` 経由では既定値にフォールバックするが、GAN 既定 (n_fft=2048/hop=300/win=1200/mel=128) が config 値と一致するため正しく動作 (nested→flat の厳密マッピングは real-data 経路を実走する T-M5.1 で詰める)。
- 想定外: `train_gan_step` を public 関数として切り出した設計 (§8.1 採用昇格) が、synthetic tensor だけで CPU 上で全 loss 経路を unit テストできる利点を実証 (real-data 不要)。`build_loaders`/`main` の real-data 経路は T-M5.1 まで未実走。
- 教訓: 完了済 component の戻り値・signature を実装着手時に Read で確認してから配線する。ticket 擬似コードは作成時点の想定で、実装後の API と乖離する (今回 MR-STFT 戻り値・Discriminator 出力・from_config signature の 3 点)。
- **⚠️ M2 phase review (2026-05-28) 訂正**:
  - **`train_gan_step` と `train_diff_step` の signature 統一は不可能** (本 §8.1 の「同 signature / fixture 共用」記述は誤り)。GAN は `(G, D, opt_G, opt_D, sch_G, sch_D, mel, audio, cfg, crit, ...)` の敵対的 2 系統、Diff は `(model, opt, batch, k, cfg, ...)` の単一 model + MSE (scheduler/crit/D なし)。共通なのは「`-> dict[str, float]` を返す step 関数を切り出す原則」のみ。conftest fixture も GAN/Diff 別建て。→ T-M3.2 着手前にチケット記述を訂正すること。
  - **`utils/` への共通化 refactor が未実施** (本 §8.1 で「M3.2 着手前必須」と採用昇格済だが現状 `train_gan.py` 内インライン)。`save_checkpoint`/`load_checkpoint`/`iter_forever`/`run_validation`/`log_scalars`/emergency save は GAN/Diff 共通。**M3.2 で `train_diff.py` を書く前に `utils/training_loop.py` (or `utils/checkpoint.py`) へ括り出す**。同時に未実装の `keep_last_n` rolling delete / config-drift 検知 / RNG 再現テスト も共通 util 側で実装・テストする。
  - **MR-STFT config キー名の実バグを修正**: 本番 `gan_wavenext2.yaml` の `loss.mrstft.fft_sizes` は `MultiResolutionSTFTLoss(n_ffts=...)` と不一致で `**cfg` 展開時に TypeError だった → `n_ffts` に修正済。
  - **config nested→flat マッピングの脆弱性 (M3.2/T-M5.1 着手前に対応)**: `GANWaveNext2.from_config(cfg["model"])` は `sub_model` の `hop`/`n_mels` を `SubModelGAN.from_config` の allowed-key (`hop_length`/`mel_channels`) で drop し既定値にフォールバック (GAN/Diff とも既定が yaml 値と一致するため偶然動作)。`build_loaders` は `["sub_model"]["hop"]` 直参照で二重命名。yaml キーをモデル引数名に揃えるか from_config に rename マップを 1 箇所集約する。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.6 (1000 step overfitting test) へ
- **使用方法 (推奨)**:
  ```python
  # tests/integration/test_overfit_1000step.py
  from wavenext2.train.train_gan import train_gan_step
  # `train_gan_step` を import して 1000 回呼ぶ薄い loop で実装
  for step in range(1000):
      logs = train_gan_step(G, D, opt_G, opt_D, sch_G, sch_D,
                            mel_overfit, audio_overfit, cfg,
                            crit_gan, crit_fm, crit_mrstft)
  ```
- **重要事項**:
  - **`scripts/smoke_gan.py` は `train_gan_step` を import する薄い wrapper にする** (§8.1 採用昇格)
  - 1 utterance を `train_loader` から繰り返し読む `OverfitDataset` を T-M2.6 で別途実装 (本チケットでは扱わない)
  - 1000 step 完了で `loss_g_mrstft_sc + loss_g_mrstft_mag` が **初期値の 10%** 以下になることを期待 (milestones.md §M2.6)
  - 本チケットで実装した `--debug` フラグは 1 step だけのため T-M2.6 では使わない
  - checkpoint は 10k step ごとに保存だが、smoke では 1000 step で終了するので `step_1000.pt` は生成されない (`best.pt` のみ更新される設計)

#### T-M5.1 (1 epoch 訓練) へ
- **使用方法**:
  ```bash
  uv run python -m wavenext2.train.train_gan \
    --config configs/gan_wavenext2.yaml \
    --amp  # 推奨
  # 33k step まで自動で回る (configs/gan_wavenext2.yaml の max_steps を 33000 に修正)
  ```
- **重要事項**:
  - 1 epoch (train-clean-100 で約 33k step) は YAML の `train.max_steps` を 33000 に上書きするか、別途 `configs/gan_1epoch.yaml` を T-M5.1 で作成
  - validation を 4 回 (8k / 16k / 24k / 32k) 走らせて MR-STFT 降下を確認
  - 1 epoch 後の validation MR-STFT が **初期値の 30%** 以下を期待 (milestones.md §M5.1)
  - `--amp` 推奨 (24GB GPU で OOM 回避)
  - `T-M2.4` の `enable_grad_ckpt=True` を YAML config で有効化推奨 (T=4 × batch=16 では必須レベル)

#### T-M6.1 (2M step 本格訓練) へ
- **使用方法**:
  ```bash
  uv run python -m wavenext2.train.train_gan \
    --config configs/gan_wavenext2.yaml --amp \
    --resume checkpoints/gan/best.pt  # 中断時
  ```
- **重要事項**:
  - A100 単体で約 410 時間 (`docs/training.md` §2.4)
  - Claude Code は `bash` で `run_in_background=true` で起動 → 数時間〜数日おきに `tensorboard --inspect` で監視
  - Divergence / OOM 検知時は自動 `--resume` で再開
  - `checkpoint.keep_last_n=5` で rolling delete を有効化、`best.pt` のみ常時保持

#### T-M3.2 (Diff 訓練スクリプト) へ
- **共通モジュール (M3.2 着手前必須 refactor)**:
  - `wavenext2.utils.scheduler.InverseLR` は Diff では使わない (固定 lr) が、`save_checkpoint` / `load_checkpoint` / `iter_forever` / `log_scalars` / `run_validation` を **M2.6 smoke pass 直後に `utils/training_loop.py` に切り出す** (§8.1 採用昇格、M3.2 着手前必須)
  - **`train_diff_step(model, opt, sch, mel, audio, cfg) -> dict[str, float]` を同 signature で実装**: `tests/conftest.py` の fixture (G, D, opt, sch, crit) を GAN / Diff で共用可能、テスト時間短縮
  - 旧 §6.3 「本チケットでは GAN 専用実装で十分、refactor は T-M3.2 の §8 で再検討」は撤回
- **`TrainState` dataclass**: Diff 側でも `step` / `best_val_*` / `rng_state` を持つ同等の dataclass を作る (共通化はしない、Diff metric が違うため)
- **TensorBoard scalar 命名規約**: `loss_G`, `loss_D`, `loss_g_*`, `lr_*` は本チケットで確立。Diff 側は `loss`, `lr` の simple naming で良い (sub-model index を tag suffix)

#### T-M0.2 申し送り (CI 整備)
- **`.github/workflows/test.yml` で `pytest -n auto` 並列化**: M2 全体 unit test 110s 超過 → 並列化必須
- `pytest-xdist` の依存追加を `pyproject.toml` の `[dependency-groups.dev]` に明示
- GPU テスト (`@pytest.mark.gpu`) は別 job で run、CPU テストのみ並列化

#### T-M2.1 から受領
- `worker_init_fn` テンプレ snippet (`epoch * num_workers + worker_id` seed、§6.1 通常項目「`iter_forever` + `persistent_workers=True`」参照)
- `Batch` dataclass (`{"mel": Tensor, "audio": Tensor}` の代わりに dataclass 化されていれば import して使用)
- `prefetch_factor` 明示 (default=2 のため `DataLoader(..., prefetch_factor=2)` を config 経由で明示)

#### T-M2.3 から受領
- `compute_total_loss` の戻り値が `(total: Tensor, unweighted_dict: dict[str, Tensor])` tuple
- `losses_unweighted` を **TensorBoard log 強制** (weighted のみだと loss weight 変更時に履歴が比較不可)
- 上記 `train_gan_step` 擬似コードで `mrstft_unweighted` を dict として受け取り、return dict に `_unweighted` suffix で含めて TensorBoard log

#### T-M2.4 から受領
- `return_intermediates=True` で T 個の中間 `y_t` を返す signature (M2.6 で中間 loss curve 観察に使用、本チケットでは未使用だが将来 hook 用に config に keep)
- `best.pt` の **atomic rename** (`best.pt.tmp` → `os.replace(best.pt.tmp, best.pt)`)、`test_atomic_best_pt_rename` で検証
- `audio_length=None` で自動推定 (`mel.shape[-1] * hop`) → `train_gan_step` 内で `audio_length=audio.shape[-1]` を明示渡し

#### 共通の注意事項
- **EMA 関連 code を一切残さない**: `docs/open-questions.md` 確定の通り、Vocos / WaveFit-PT 共に EMA 未使用。本チケットでも EMA を 1 行も書かない (TODO コメント含めて)
- **`y_hat.detach()` のタイミング**: D update の入力でのみ detach。G update では D に grad を流す (D の重みは `opt_D.step()` を G update 中に呼ばないことで保護)
- **autocast の境界**: `--amp` 時、autocast は **G forward + D forward のみ** に限定。loss 計算 (hinge GAN / FM / MR-STFT) は fp32 で実行
- **`persistent_workers=True`**: DataLoader に必須。epoch 境界での worker 再起動コスト回避
- **Checkpoint state の完全性**: 4 module (G, D, opt_G, opt_D, sch_G, sch_D) + 1 step + 1 best metric + RNG state (torch / cuda / numpy / random) + config の 4 グループを保存

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.5 の Acceptance チェックボックス 3 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M2.5 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/training.md` §2 にコード参照リンク (`src/wavenext2/train/train_gan.py:main`) を追記
  - [ ] (該当時) `docs/implementation-plan.md` §5 の `configs/gan_wavenext2.yaml` 雛形を実装と一致するよう更新 (validation / checkpoint / logging キー追加)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` で全 100% 確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - `pytorch-lightning` 採用検討 (M6 完了時)
  - D 更新を 1:2 化 (M5 で D 極小化発生時)
  - wandb 併用 (M6 完了時、複数実験比較が必要な場合)
  - `accelerate` 採用 (M6 DDP 化時)
  - `InverseLR` の warmup を linear に変更 (M5 phase review、原典準拠性と簡易性のトレードオフ)
  - `save_checkpoint` / `load_checkpoint` の `utils/checkpoint.py` 分離 (T-M3.2 で再利用が必要なら)
- **`docs/open-questions.md` への追記要否**: 不要
