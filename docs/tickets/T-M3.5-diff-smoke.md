---
id: T-M3.5
title: Diff smoke (1 sub-model × 1000 step overfitting)
milestone: M3
phase: M3
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M3.2, T-M3.3]
blocks: [T-M5.2]
related_docs:
  - docs/milestones.md#m35-smoke-training
  - docs/training.md
---

# T-M3.5: Diff smoke training (1 sub-model × 1000 step overfitting)

> **マイルストーン**: [M3](../milestones.md#m3-diff-wavenext-2-作業量-large5-サブタスク) / **サブタスク**: [M3.5](../milestones.md#m35-smoke-training)
> **依存**: [T-M3.2](T-M3.2-train-diff.md), [T-M3.3](T-M3.3-reverse-sampler.md) (前提: [T-M3.1](T-M3.1-diff-model.md), [T-M1.5](T-M1.5-noise-embedding.md), [T-M1.6](T-M1.6-sub-model.md)) / **後続**: [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
Diff-WaveNeXt 2 の訓練 stack (Dataset / SubModelDiff / NoiseEmbedding / MSE loss / Training loop) が **end-to-end で動作し、勾配が正しく流れ、noise level conditioning が機能しているか** を、**primary に sub-model 4 (大ノイズ、c の値域が広く conditioning test の感度が ~70 倍高い、最も学習が難しい)** を 1 utterance で 1000 step 訓練して **overfitting させる** ことで最小コストで検証する。sub-model 1 は subordinate (nightly)。
これは M5.2 (1 sub-model 1 epoch) / M6.2 (4 sub-model 本格訓練) の **gate** であり、ここで失敗したら T-M3.3 (β 負値、σ_t index、reverse step 式) / T-M1.5 (NoiseEmbedding `c * 1000` rescale) / T-M3.1 (BAND_BOUNDS) / T-M3.2 (MSE 計算 / noise scaling / bf16 c 精度) / T-M1.4 (Generator) / T-M1.1 (additive bias 注入) まで遡って修正する。

### smoke の 3 軸 minimal sanity (T-M2.6 GAN smoke と整合)
smoke で確認するのは以下の 3 軸のみ:
1. **step が回る** (forward / backward / optimizer.step が exception 出さず完走)
2. **finite** (loss / parameter / gradient に NaN/Inf なし)
3. **単調減少** (overfit 設定で loss が下がる、architectural sanity 担保)

**確認しないこと**: 品質 (UTMOS/MCD), 収束性 (validation loss), 汎化 (100+ utterances) — すべて M5.2 移譲。

### ゴール
- [ ] `tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_4` が pytest single-source として実装され、`@pytest.mark.slow @pytest.mark.gpu` でデフォルト skip (**primary smoke**)
- [ ] `tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_1` を **subordinate** (nightly) として追加 — sub-model 1 は c ∈ [0.9929, 1.0) で conditioning test の感度が極端に低いため step が回る + finite のみを緩く検証
- [ ] `tests/test_train_diff_overfit.py::test_smoke_reverse_with_mocks` を追加 — sub-model 1 のみ訓練、2-4 を identity (`eps_pred = 0`) で mock した 4-step reverse sample が NaN を出さないことを検証 (T-M3.3 β 負値 / sampler 符号 / index バグの早期検知)
- [ ] `tests/test_train_diff_overfit.py::test_init_loss_robust` を追加 — `init_loss = max(step 50-100 平均, step 10-20 平均)` の計算が sub-model 1 / sub-model 4 両 case で stable
- [ ] `scripts/smoke_diff.py` は薄い wrapper として `subprocess.run(["pytest", "tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_4", "-v", "-m", "slow and gpu"])` を呼ぶだけ (DRY、T-M2.6 同様)
- [ ] `scripts/smoke_diff_synthetic.py` **は新規作成しない** — synthetic CPU smoke は `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` (`@pytest.mark.cpu_only`) に集約。CI .yml は `pytest -m cpu_only` 1 行で起動
- [ ] `configs/diff_wavenext2_smoke.yaml` で smoke 用 hyperparameter (max_steps=1000, batch=1, segment_length=25600, **sub_model_idx=4 (primary)**, validation off, `c_rescale` config 切替可能) を別 config 化
- [ ] 1000 step 後の MSE loss が初期値 (`init_loss = max(step 50-100 平均, step 10-20 平均)` の robust 定義) の **5% 以下** に低下 (Diff は GAN より overfit しやすい、milestones.md §M3.5)
- [ ] noise level conditioning 機能テスト: 同じ `x_t` + 同じ `mel` + **異なる `c`** で `eps_pred` の cosine similarity < 0.99 (条件付けが効いている証拠、sub-model 4 では `c` 幅 0.48 で感度高)
- [ ] `c_rescale` config 切替で smoke 内 ablation 可能化 — sub-model 4 で fail なら `c * 1000` rescale を切り替えて再試行 → 学習する場合 T-M1.5 NoiseEmbedding の根本問題確定 (T-M3.2 で `c_rescale` config を実装する前提)
- [ ] TensorBoard scalars (`loss_mse`, `lr`, `c_mean`, `c_std`) に NaN/Inf なし
- [ ] 全 1000 step を 1 GPU で **20 分以内** に完走 (1 sub-model のみ、GAN より軽量)
- [ ] `atexit.register(writer.close)` を `try/except OSError` で wrap (spawn worker 内で double-close 例外を吐くリスクを回避) + SIGINT/SIGTERM handler で event ファイルを確実に flush
- [ ] `docs/milestones.md` §M3.5 Acceptance チェックボックス更新 (本 ticket では reverse sample 整合は §6 で out of scope と明示)

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `tests/test_train_diff_overfit.py` (**pytest single-source**、primary は `sub_model_4`、subordinate `sub_model_1` / `synthetic_cpu` / `reverse_with_mocks` / `init_loss_robust` を含む)
  - `scripts/smoke_diff.py` (薄い wrapper、`subprocess.run(["pytest", ...])`、primary = sub-model 4)
  - `configs/diff_wavenext2_smoke.yaml` (sub_model_idx=4 primary、`c_rescale` flag 含む)
- **新規作成しない**:
  - ~~`scripts/smoke_diff_synthetic.py`~~ — `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` に集約
- 編集:
  - `docs/milestones.md` §M3.5 Acceptance チェックボックス
  - `docs/tickets/index.md` T-M3.5 ステータス

### 2.2 主要構造

```python
# tests/test_train_diff_overfit.py
"""Diff smoke: primary は sub-model 4 (大ノイズ、c 幅 0.48、conditioning test 感度 ~70x)。
sub-model 1 は subordinate (c 幅 7e-3 で感度極小、step 完走と finite のみ確認)。

pytest single-source 設計 (T-M2.6 と同じパターン)。`scripts/smoke_diff.py` は subprocess
wrapper として本 test を呼ぶ。T-M3.2 の `train_diff_step` を import 再利用。
noise level conditioning が機能していることを確認するため、同一 `x_t` + 同一 `mel` で
異なる `c` を入れて `eps_pred` の cosine similarity が < 0.99 になることを検証する。
"""
from __future__ import annotations
import atexit
import signal
import yaml
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from wavenext2.train.train_diff import train_diff_step, build_diff_model_and_optimizer
from wavenext2.data.dataset import LibriTTSRDataset


def _robust_init_loss(losses_early: list[float], losses_warmup: list[float]) -> float:
    """`init_loss = max(step 50-100 平均, step 10-20 平均)` の robust 定義。
    Adam β1=0.9 で勾配 EMA が安定するのは ~10 step、β2=0.98 で ~50 step。
    step 10-20 はまだ過渡的で init_loss が過小評価 (ratio が緩く見える) するため、
    step 50-100 の平均も並行計算し最大値を採用する。floor は 1e-6 で stable 化。"""
    a = sum(losses_early) / max(len(losses_early), 1) if losses_early else 0.0
    b = sum(losses_warmup) / max(len(losses_warmup), 1) if losses_warmup else 0.0
    return max(a, b, 1e-6)


def _safe_close(writer) -> None:
    """spawn worker 内で atexit が再 register され double-close 例外を吐くリスクを wrap。"""
    try:
        writer.close()
    except OSError:
        pass


def _run_overfit_loop(cfg: dict, sub_idx: int, max_steps: int, strict_ratio: bool) -> dict:
    """sub-model `sub_idx` を 1 utterance で max_steps step overfit する共通ループ。"""
    full_ds = LibriTTSRDataset(cfg["data"]["filelist"], segment_length=cfg["data"]["segment_length"], mode="val")
    single_ds = Subset(full_ds, [cfg["data"].get("smoke_idx", 0)])
    loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0)

    diff_model, opt, sched = build_diff_model_and_optimizer(cfg, sub_idx=sub_idx)
    writer = build_tensorboard_writer(cfg["logging"]["log_dir"])
    atexit.register(_safe_close, writer)
    signal.signal(signal.SIGINT, lambda *_: (_safe_close(writer), exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_safe_close(writer), exit(143)))

    losses_warmup: list[float] = []   # step 10-20
    losses_early: list[float] = []    # step 50-100
    step = 0
    while step < max_steps:
        for mel, audio in loader:
            mel, audio = mel.cuda(), audio.cuda()
            logs = train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg)
            assert torch.isfinite(torch.tensor(logs["loss_mse"])).all(), f"NaN/Inf at step {step}"
            if 10 <= step < 20:
                losses_warmup.append(logs["loss_mse"])
            if 50 <= step < 100:
                losses_early.append(logs["loss_mse"])
            for k, v in logs.items():
                writer.add_scalar(k, v, step)
            step += 1
            if step >= max_steps:
                break

    final_loss = logs["loss_mse"]
    init_loss = _robust_init_loss(losses_early, losses_warmup)
    ratio = final_loss / init_loss
    print(f"[smoke-diff sub={sub_idx}] init MSE = {init_loss:.6e}, final = {final_loss:.6e}, ratio = {ratio:.4f}")
    if strict_ratio:
        assert ratio < 0.05, f"Overfit failed: ratio {ratio:.4f} >= 0.05 (T-M3.3/T-M1.5/T-M3.1/T-M3.2 を確認、c_rescale ablation も検討)"
    return {"final_loss": final_loss, "init_loss": init_loss, "ratio": ratio}


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes_sub_model_4(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """PRIMARY smoke: sub-model 4 (c ∈ [0, 0.4817)、conditioning 感度 ~70x sub-model 1)。
    最も学習が難しい sub-model が pass すれば他は安全 (sub-model 1 学習が trivial で false positive リスク)。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["model"]["sub_model_idx"] = 4
    _run_overfit_loop(cfg, sub_idx=4, max_steps=cfg["train"]["max_steps"], strict_ratio=True)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.nightly
def test_smoke_completes_sub_model_1(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """SUBORDINATE (nightly): sub-model 1 (c ∈ [0.9929, 1.0)、幅 7e-3 で conditioning test 感度 極小)。
    step が回る + finite のみを緩く検証 (品質指標は緩く)。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["model"]["sub_model_idx"] = 1
    _run_overfit_loop(cfg, sub_idx=1, max_steps=cfg["train"]["max_steps"], strict_ratio=False)


@pytest.mark.cpu_only
def test_smoke_synthetic_cpu(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """synthetic mel + 440 Hz sine 波 1 sample × 500 step を CPU で 1〜2 分。
    LibriTTS-R 取得不可環境 / nightly CI 用 (PR CI は GPU smoke のみ)。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["model"]["sub_model_idx"] = 4
    cfg["train"]["max_steps"] = 500
    # synthetic data path injection (詳細は実装時)
    ...


@pytest.mark.slow
@pytest.mark.gpu
def test_noise_level_conditioning(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """同じ x_t + mel + 異なる c で eps_pred の cosine similarity < 0.99 を確認 (primary = sub-model 4)。
    sub-model 4 は c 幅 0.48 で感度高、NoiseEmbedding 動作確認に最適。
    NoiseEmbedding が無視されていれば cosine ≈ 1.0 になる (T-M1.5 / T-M1.1 のバグ検知)。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["model"]["sub_model_idx"] = 4
    diff_model, _, _ = build_diff_model_and_optimizer(cfg, sub_idx=4)
    diff_model.eval()
    B, T = 1, cfg["data"]["segment_length"]
    mel_dim, mel_T = cfg["model"]["sub_model"]["mel_channels"], T // cfg["model"]["sub_model"]["hop_length"]
    torch.manual_seed(0)
    mel = torch.randn(B, mel_dim, mel_T).cuda()
    x_t = torch.randn(B, T).cuda()
    L, U = diff_model.BAND_BOUNDS[3]  # sub-model 4 (0-indexed)
    c_a = torch.full((B,), L).cuda()
    c_b = torch.full((B,), U).cuda()
    with torch.no_grad():
        eps_a = diff_model.sub_models[3](mel, x_t, c_a)
        eps_b = diff_model.sub_models[3](mel, x_t, c_b)
    cos = F.cosine_similarity(eps_a.flatten(1), eps_b.flatten(1), dim=1).mean().item()
    print(f"[smoke-diff] cos(eps(c={L:.4f}), eps(c={U:.4f})) = {cos:.4f}")
    assert cos < 0.99, f"NoiseEmbedding が無視されている疑い: cos={cos:.4f} (T-M1.5 / T-M1.1 を確認、c_rescale ablation も検討)"


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_reverse_with_mocks(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """sub-model 1 のみ訓練、2-4 を identity (eps_pred = 0) で mock した 4-step reverse sample が NaN を出さない。
    sampler の符号 / index バグ、T-M3.3 β 負値問題の早期検知。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["model"]["sub_model_idx"] = 1
    diff_model, opt, _ = build_diff_model_and_optimizer(cfg, sub_idx=1)
    # 簡易訓練 (200 step で sub-model 1 を warmup)
    # sub-model 2-4 を eps_pred = torch.zeros_like で置換
    # reverse_sample を呼び、output が finite かを assert
    ...


@pytest.mark.cpu_only
def test_init_loss_robust() -> None:
    """`init_loss = max(step 50-100 平均, step 10-20 平均)` の計算が両方の case で安定。
    Adam β1=0.9 で勾配 EMA が安定するのは ~10 step、β2=0.98 で ~50 step。
    step 10-20 はまだ過渡的で init_loss が過小評価するため、step 50-100 も並行計算する。"""
    # Case 1: early (50-100) が大きい
    assert _robust_init_loss([0.5] * 50, [0.3] * 10) == 0.5
    # Case 2: warmup (10-20) が大きい
    assert _robust_init_loss([0.2] * 50, [0.6] * 10) == 0.6
    # Case 3: floor (両方 0)
    assert _robust_init_loss([], []) == 1e-6


# scripts/smoke_diff.py — 薄い wrapper (primary = sub-model 4)
# import subprocess, sys
# def main() -> None:
#     subprocess.run([sys.executable, "-m", "pytest",
#                     "tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_4",
#                     "-v", "-m", "slow and gpu"], check=True)
#
# sub-model 4 並列 smoke (option): pytest -k "smoke_diff" -n 2
#   k=1 と k=4 を 2 GPU 並列、CI コスト 1×→2× だが sub-model 4 失敗の早期検知価値が上回る
```

### 2.3 使用するハイパーパラメータ / 定数 (`configs/diff_wavenext2_smoke.yaml`)

```yaml
seed: 42
data:
  filelist: data/filelists/train.txt
  segment_length: 25600         # Diff 設定 (hop=256 × 100 frames)
  smoke_idx: 0
model:
  sub_model_idx: 4              # PRIMARY smoke: sub-model 4 (c 幅 0.48、conditioning 感度 ~70x、最難 sub-model)
  c_rescale: 1.0                # 1.0 / 1000.0 切替可能 (T-M3.2 で実装)、smoke 内 ablation 用
  sub_model:
    mel_channels: 128
    n_fft: 1024
    hop_length: 256             # Diff の hop (FastDiff 準拠)
    win_length: 1024
    dim: 512
    n_blocks: 8
train:
  max_steps: 1000
  batch_size: 1
  grad_clip: 1.0
  validation: false
optimizer:
  lr: 2.0e-4
  betas: [0.9, 0.98]
  weight_decay: 0.0
logging:
  log_dir: logs/smoke_diff
```

| 名前 | 値 | 出典 |
|---|---|---|
| `max_steps` | 1000 | milestones.md §M3.5 |
| `sub_model_idx` | **4** (primary) | M3 レビュー反映: sub-model 4 (c 幅 0.48) で conditioning 感度 ~70x、最難 sub-model 優先検証 |
| `c_rescale` | 1.0 (default) / 1000.0 (ablation) | T-M3.2 で実装、本 smoke で ablation 化 (T-M1.5 NoiseEmbedding 根本問題確定手段) |
| `segment_length` | 25600 | docs/training.md §1.3 (Diff hop=256) |
| 失敗判定閾値 | `final / init < 0.05` | milestones.md §M3.5 (Diff は GAN より overfit しやすい) |
| `init_loss` 定義 | `max(step 50-100 平均, step 10-20 平均, 1e-6)` | M3 レビュー反映: Adam β1/β2 EMA 安定性に基づく robust 定義 |
| Optimizer | Adam(lr=2e-4, β=[0.9,0.98], wd=0) | T-M3.2 / milestones.md §M3.2 |

### 2.4 アルゴリズム / 処理フロー
1. `configs/diff_wavenext2_smoke.yaml` 読み込み (**sub_model_idx=4 primary**)
2. `LibriTTSRDataset` から 1 utterance を `Subset([0])` で抽出
3. T-M3.2 の `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg)` を 1000 回呼ぶ
   - 内部で band 内 uniform sampling: `c ~ U(0, 0.4817)` (sub-model 4、幅 0.48)
   - `x_t = √abar · x_0 + √(1-abar) · ε`、abar = 1 - c²
   - `eps_pred = sub_model(mel, x_t, c)`、`loss = MSE(eps_pred, ε)`
4. **`init_loss` の robust 定義**: step 10-20 平均と step 50-100 平均を並行記録、`init_loss = max(両者, 1e-6)`
   - Adam β1=0.9 で勾配 EMA 安定化は ~10 step、β2=0.98 では ~50 step
   - step 10-20 のみだと過渡期で `init_loss` 過小評価 → ratio が緩く見える false negative リスク
5. NaN/Inf hook (`assert isfinite(loss_mse)`)
6. 1000 step 後に `final / init < 0.05` を assert
7. **別 test**: noise level conditioning (異なる `c` で eps_pred が異なるか) を独立検証 (sub-model 4 で感度 ~70x)
8. **別 test**: reverse with mocks (sub-model 1 のみ訓練、2-4 を `eps_pred=0` で mock、4-step reverse が NaN を出さない) で T-M3.3 β 負値 / sampler 符号 / index バグを早期検知
9. `atexit.register(_safe_close, writer)` + try/except OSError + SIGINT/SIGTERM で writer flush (spawn worker double-close 回避)
10. **smoke 内 ablation** (sub-model 4 で fail 時): `c_rescale: 1.0 → 1000.0` で再試行、学習する場合 T-M1.5 NoiseEmbedding 根本問題確定

**注記**: smoke では reverse sample (4 sub-model 揃う必要あり) は行わない (§6.1)。`milestones.md` §M3.5 の 2 つ目の acceptance「reverse sample が GT に近い」は本 ticket では out of scope とし、M5.2 (4 sub-model 揃った段階) で再評価する。
ただし `test_smoke_reverse_with_mocks` (sub-model 2-4 を identity mock) は **sampler の符号 / index バグ早期検知の手段** として acceptance に追加する (T-M3.3 β 負値問題の検証)。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | smoke test (sub-model 4 primary / sub-model 1 subordinate / synthetic / mock reverse / init_loss) + config 実装 | general-purpose |
| Reviewer | 1 | loss 推移 / cos similarity 確認 + 遡及調査 (T-M3.3 / T-M1.5 / T-M3.1 / T-M3.2) | general-purpose |
| Tester | 1 | smoke 実行 + TensorBoard 確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M3.2 / T-M3.3 完了が前提)
- 最大並列数: 1
- 後続 T-M5.2 とは順次。

## 4. 提供範囲 (Scope)

### In Scope
- `tests/test_train_diff_overfit.py` (pytest single-source、primary = sub-model 4)
- `scripts/smoke_diff.py` (薄い wrapper、primary = sub-model 4)
- `configs/diff_wavenext2_smoke.yaml` (sub_model_idx=4 primary、`c_rescale` flag)
- T-M3.2 から `train_diff_step` を import 再利用 (`c_rescale` config 切替対応前提)
- noise level conditioning 機能テスト (cosine similarity、sub-model 4 で感度 ~70x)
- **sub-model 1 subordinate test** (nightly、step が回る + finite のみ)
- **`test_smoke_reverse_with_mocks`** (sub-model 1 訓練、2-4 を identity mock した 4-step reverse、T-M3.3 β 負値検知)
- **`test_init_loss_robust`** (`init_loss = max(50-100, 10-20)` の単体テスト)
- **synthetic CPU smoke** を `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` に集約 (wrapper script なし)
- **smoke 内 ablation**: `c_rescale: 1.0 → 1000.0` 切替で再試行
- NaN/Inf hook、`atexit.register(_safe_close, writer)` (OSError wrap) + SIGINT/SIGTERM 保証
- 1 GPU 20 分以内完走

### Out of Scope
- **reverse sample テスト (本物の 4 sub-model)** (4 sub-model 必要、§6.1 / §8 で詳述)
  - ただし mock 版 reverse は in scope (T-M3.3 sampler 符号 / index バグ検知用)
- **MR-STFT loss で reverse sample 評価** (1 sub-model 欠落状態では無意味)
- 本格 1 epoch 訓練 (T-M5.2)
- 4 sub-model 統合訓練 (T-M6.2)
- UTMOS / NISQA / MCD (T-M4.*)
- multi-GPU / DDP / AMP (M6)
- GAN smoke (T-M2.6)
- post-filter (T-M3.4)
- GitHub Actions CI 化 (GPU runner 必須、M6 まで保留。synthetic CPU smoke のみ `pytest -m cpu_only` で nightly job に分離可)
- ~~`scripts/smoke_diff_synthetic.py`~~ (test 自体に集約、wrapper 削除)

### Deliverable
- ファイル: `tests/test_train_diff_overfit.py`, `scripts/smoke_diff.py`, `configs/diff_wavenext2_smoke.yaml`
- 関数: `test_smoke_completes_sub_model_4()` (primary), `test_smoke_completes_sub_model_1()` (subordinate/nightly), `test_smoke_synthetic_cpu()` (cpu_only), `test_noise_level_conditioning()`, `test_smoke_reverse_with_mocks()`, `test_init_loss_robust()`, `main()` (wrapper)
- ドキュメント差分: `docs/milestones.md` §M3.5、`docs/tickets/index.md`

## 5. テスト項目

### 5.1 Unit / 結合テスト
- [ ] **`tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_4`** (PRIMARY) — `@pytest.mark.slow @pytest.mark.gpu`、sub-model 4 で 1000 step 完走、MSE NaN/Inf なし、`final / init < 0.05`、`init_loss = max(50-100, 10-20)` の robust 定義
- [ ] **`tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_1`** (SUBORDINATE, nightly) — `@pytest.mark.slow @pytest.mark.gpu @pytest.mark.nightly`、sub-model 1 でも step が回る + finite (品質指標は緩く、`strict_ratio=False`)
- [ ] **`tests/test_train_diff_overfit.py::test_noise_level_conditioning`** — `@pytest.mark.slow @pytest.mark.gpu`、sub-model 4 で同じ `x_t`+`mel`、異なる `c` で `cosine_similarity(eps_a, eps_b) < 0.99` (感度 ~70x)
- [ ] **`tests/test_train_diff_overfit.py::test_smoke_reverse_with_mocks`** — `@pytest.mark.slow @pytest.mark.gpu`、sub-model 1 のみ訓練、2-4 を identity (`eps_pred = 0`) で mock した 4-step reverse sample が NaN を出さない (T-M3.3 β 負値 / sampler 符号 / index バグ検知)
- [ ] **`tests/test_train_diff_overfit.py::test_init_loss_robust`** — `@pytest.mark.cpu_only`、`init_loss = max(step 50-100 平均, step 10-20 平均, 1e-6)` の計算が両方の case (early > warmup / warmup > early / 両方 0) で安定
- [ ] `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` — `@pytest.mark.cpu_only`、synthetic mel + 440 Hz sine 波 1 sample × 500 step を CPU で 1〜2 分 (subordinate gate / nightly CI 候補)、`scripts/smoke_diff_synthetic.py` 不要 (test 集約)
- [ ] `tests/test_train_diff_overfit.py::test_smoke_config_keys` — smoke config の必須 key
- [ ] `tests/test_train_diff_overfit.py::test_smoke_output_finite` — 訓練中の `eps_pred` が finite

### 5.2 e2e テスト
- [ ] `uv run python scripts/smoke_diff.py` が exit code 0 で完了 (primary = sub-model 4)
- [ ] TensorBoard scalars (`loss_mse`, `lr`, `c_mean`, `c_std`) が単調減少傾向
- [ ] `c_mean` が band 内 (`[0, 0.4817]`) に収まる (sub-model 4)、TensorBoard で histogram 確認
- [ ] **smoke 内 ablation**: sub-model 4 で fail 時、`c_rescale: 1.0 → 1000.0` で再試行 → 学習する場合 T-M1.5 NoiseEmbedding 根本問題確定
- [ ] **CI 構成**: PR CI は `pytest -m "slow and gpu and not nightly"` で primary (sub-model 4) のみ、nightly job は `-m "slow and gpu"` + `-m cpu_only` で全テスト (synthetic は `pytest -m cpu_only` 別 job に分離)

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.5 より転記 + 本 ticket 修正)
- [ ] 1 utterance で 1000 step 後、MSE loss が初期値の 5% 以下 (primary = sub-model 4)
- [ ] ~~同じ utterance に対する reverse sample が GT に近い~~ → **本 ticket では out of scope** (4 sub-model 必要、§6.1 / §8 で詳述、M5.2 で再評価)
- [ ] **代替 acceptance**: noise level conditioning 機能テスト (cosine similarity < 0.99、sub-model 4 で感度 ~70x)
- [ ] **追加 acceptance**: reverse with mocks (sub-model 2-4 を identity で mock、NaN なし、T-M3.3 検証)

### 5.4 追加 acceptance (本 ticket 独自)
- [ ] TensorBoard に NaN/Inf なし
- [ ] 1 GPU で 20 分以内に完走 (primary = sub-model 4)
- [ ] CPU 環境では `@pytest.mark.gpu` で skip 可、synthetic / init_loss test は CPU で実行
- [ ] M2+M3 合計 CI 時間 220s 超過防止: T-M3.1 (30s) + T-M3.2 (60s) + T-M3.3 (30s) + T-M3.4 unit (10s) + T-M3.5 synthetic (90s CPU) = 220s → synthetic を `@pytest.mark.slow` 別 job に分離、**PR CI は 130s 目標**

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (失敗時の遡及調査優先順位)

- **CRITICAL: 1 sub-model だけで reverse sample テストは不可能**
  - 影響: `milestones.md` §M3.5 の 2 つ目 acceptance「reverse sample が GT に近い」は 4 sub-model が揃わないと評価できない (sub-model 2,3,4 が random init のままでは reverse 全 step が無意味)
  - 緩和: **本 smoke では loss curve + noise level conditioning のみで判定**、reverse sample は M5.2 / M6.2 で再評価する旨を §5.3 / §9.1 に明記
  - 代替案: 4 sub-model すべて smoke (時間 4 倍、§8 代替で再評価)

- **CRITICAL: 1 sub-model overfit と reverse sampling 品質は別問題**
  - 影響: smoke pass しても、4 sub-model 統合時に sampler の符号 / β 計算誤りで破綻する
  - 検知: M3.3 reverse sample の unit test (deterministic / `[-1,1]` 範囲) を独立に走らせる
  - 遡及調査: T-M3.3 (β/σ 計算、loop 方向)

- **RESOLVED (sub-model 4 primary 昇格で緩和): band 内 `c` の幅が狭い問題は sub-model 1 固有**
  - 背景: sub-model 1 (`[0.9929, 1.0]`、幅 7e-3) は実質 `c ≈ 0.999` 単一値で、conditioning test の感度が極端に低い (NoiseEmbedding が `linear(sinusoidal(c*1000))` で出力差が顕在化しない)
  - **解決**: **primary smoke を sub-model 4 (`[0, 0.4817]`、幅 0.48) に昇格** (§ゴール / §8.1)。conditioning test の感度が ~70 倍高く、NoiseEmbedding 動作を確実に検出
  - 検知: `test_noise_level_conditioning` (sub-model 4) で `c=L` vs `c=U` の cos similarity が ≥ 0.99 なら conditioning が無視されている
  - 緩和案 (sub-model 4 でも fail 時): **`c_rescale` config (1.0 → 1000.0) を smoke で切替** (T-M1.5 `c * 1000` rescale の必要性を直接判定、最優先 ablation)

- **CRITICAL: MSE は scale 依存、loss が小さくなりがち**
  - 影響: `init_loss` の絶対値が小さい場合 → `ratio < 0.05` が緩く見えるが実は学習していない可能性
  - 緩和: `init_loss = max(step 50-100 平均, step 10-20 平均, 1e-6)` の robust 定義 (floor 1e-6 で ratio 計算 unstable 回避)
  - 緩和: synthetic CPU smoke で baseline (random init vs converged) の MSE 絶対値を pin → M5.2 へ申し送り

- **RESOLVED (sub-model 4 primary 昇格で緩和): 学習速度の sub-model 依存**
  - 背景: sub-model 1 (small noise) は学習が早く trivial に pass、sub-model 4 (大ノイズ) は遅い → 旧設計 (sub-model 1 primary) では sub-model 4 fail を M6.2 まで見逃すリスク
  - **解決**: **最も学習が難しい sub-model 4 を primary に昇格** (pass すれば他は安全、false positive リスク最小化)
  - sub-model 1 は subordinate (nightly、step が回る + finite のみ)、M5.2 で全 sub-model smoke を再実行する申し送り

- **CRITICAL: `init_loss` の robust 定義が必要**
  - 影響: step 10-20 のみだと Adam の勾配 EMA (β1=0.9 で ~10 step、β2=0.98 で ~50 step) が過渡的で過小評価、ratio 計算 unstable、false negative
  - 緩和: `init_loss = max(step 50-100 平均, step 10-20 平均, 1e-6)`、`test_init_loss_robust` で両 case の安定性を検証、絶対値が下がっているかも併せて確認

#### 通常項目
- **GPU 環境が借りられない**: Colab T4 で 17 分想定、Lambda Labs spot で 1 時間 → smoke はそもそも軽い、subordinate synthetic smoke は CPU で 1〜2 分
- **`segment_length=25600` で 1 GPU OOM**: T-M1.4 `enable_grad_ckpt=True` を smoke config で有効化
- **TensorBoard log_dir 衝突**: `log_dir = f"logs/smoke_diff/{datetime.now()}"`
- **`Subset([0])` の deterministic 性**: `mode="val"` で固定 gain (-3 dB)、`smoke_idx` を config に明示
- **`atexit.register(writer.close)` の double-close リスク**: spawn worker 内 (pytest-xdist `-n 2` 並列実行時など) で atexit が再 register され、close 済み writer に再度 close を呼んで例外を吐くリスク → `_safe_close(writer)` で `try/except OSError` wrap (§ゴール / 2.2 参照)
- **M2+M3 合計 CI 時間 220s 超過**: §5.4 参照。synthetic を `@pytest.mark.slow` 別 job に分離、PR CI は 130s 目標

### 6.2 仕様の曖昧さ
- **閾値 0.05**: `milestones.md` §M3.5 から (Diff は GAN より overfit しやすい想定)、経験則的に妥当
- **`init_loss` 定義**: `max(step 50-100 平均, step 10-20 平均, 1e-6)` の robust 定義 (T-M2.6 の step 10-20 のみから強化)。Adam β1=0.9 で勾配 EMA 安定化は ~10 step、β2=0.98 では ~50 step。step 10-20 はまだ過渡的で `init_loss` 過小評価 (ratio が緩く見える) リスク
- **cosine similarity 閾値 0.99 — baseline 未確定 (要 T-M1.5 同期)**: 「経験則、M1 PoC で pin」とあるが、T-M1.5 (NoiseEmbedding) の unit test で **未訓練 model の cos(c=L, c=U) の baseline** を計測しないと smoke で false fail/pass の判別不能。**T-M1.5 着手前に「`cos < 0.99` は untrained でも満たすか」を要確認** (§6.3 で T-M1.5 と双方向同期化)。NoiseEmbedding が完全 zero 化されていれば cos ≈ 1.0、機能していれば <0.95 想定
- `docs/open-questions.md`: すべて確定済み

### 6.3 他チケットとの整合性
- **T-M3.2 (train_diff)**: `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg) -> dict[str, float]`、`build_diff_model_and_optimizer(cfg, sub_idx) -> (model, opt, sched)` を expose。**`c_rescale` config (1.0 / 1000.0) で smoke 内 ablation 可能化** (sub-model 4 fail 時の T-M1.5 根本問題切り分け手段)
- **T-M3.3 (reverse_sample)**: 本物の reverse (4 sub-model) は呼ばない (M5.2 持ち越し)。ただし **`test_smoke_reverse_with_mocks` で sub-model 2-4 を identity mock した reverse を呼び、β 負値 / σ_t index / reverse step 式の符号バグが NaN を引き起こすかを早期検知**
- **T-M3.1 (DiffWaveNeXt2)**: `BAND_BOUNDS[3]` (sub-model 4)、`sub_models[3]` を smoke から直接参照 (conditioning test)
- **T-M1.5 (NoiseEmbedding) — 双方向同期**: conditioning が "効く" こと自体は T-M1.5 unit test で先行検証、smoke は統合確認。**T-M1.5 側に「untrained model の cos(c=L, c=U) baseline 計測」を依頼** (本 ticket の `cos < 0.99` 閾値が untrained でも満たされないことを保証するため、T-M1.5 着手前に確認)
- **T-M2.6 (GAN smoke)**: pytest single-source パターンを踏襲、wrapper を 2 経路維持しない。`init_loss` robust 定義 (`max(50-100, 10-20)`) は T-M2.6 にもフィードバック可能 (本 ticket で先行採用)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `scripts/smoke_diff.py` が `train_diff_step` を **重複実装せず流用** している (DRY、primary = sub-model 4)
- [ ] `scripts/smoke_diff_synthetic.py` が **作成されていない** (synthetic は `test_smoke_synthetic_cpu` に集約、wrapper 2 経路維持しない)
- [ ] `configs/diff_wavenext2_smoke.yaml` が `diff_wavenext2.yaml` (本番) と明確に分離、`sub_model_idx=4` / `c_rescale` flag を含む
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)、reverse sample acceptance が out of scope に降格された justification が §6.1 / §9.1 に記載
- [ ] NaN/Inf hook が機能 (artificial NaN 注入で test 可)
- [ ] noise level conditioning test が **本物の bug を検知できる** (artificial に NoiseEmbedding を zero 化して fail することを確認、sub-model 4 で感度高)
- [ ] `test_smoke_reverse_with_mocks` が sub-model 2-4 を identity mock し、T-M3.3 β 負値で NaN が出ることを検知できる
- [ ] `test_init_loss_robust` が `max(50-100, 10-20)` の両 case をカバー
- [ ] `_safe_close(writer)` が `try/except OSError` で double-close を wrap (spawn worker 対策)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] `assert` メッセージが具体的 (どの critical 項目を確認すべきか明示、`c_rescale` ablation 言及)
- [ ] TensorBoard log_dir がタイムスタンプ付き
- [ ] 参考実装 (FastDiff) コピーしていない
- [ ] 1000 step 実行時間 20 分以内 (1 sub-model は GAN より軽量)、PR CI 130s 目標 (synthetic 別 job)

## 8. ゼロから作り直すとしたら

### 8.1 別の設計を採るとしたら

#### 採用設計 (M3 レビューで昇格)
- **sub-model 4 を primary smoke に昇格** (× 1 utterance × 1000 step):
  - 理由 1: sub-model 1 (c ∈ [0.9929, 1.0)、幅 7e-3) で noise level conditioning test は感度が極端に低い (c の値域差が小さすぎて NoiseEmbedding が `linear(sinusoidal(c*1000))` で出力差が顕在化しない可能性)
  - 理由 2: sub-model 4 (c ∈ [0, 0.4817)、幅 0.48) は conditioning test の感度が ~70 倍高く、`c * 1000` rescale (T-M1.5) の必要性も明確に判定できる
  - 理由 3: 最も学習が難しい sub-model が pass すれば他は安全 (sub-model 1 学習が trivial で false positive リスク)
  - sub-model 1 は nightly subordinate (CPU synthetic smoke で覚悟する)
- **`init_loss = max(step 50-100 平均, step 10-20 平均)` の robust 定義**: Adam の β1=0.9 で勾配 EMA が安定するのは ~10 step、β2=0.98 で ~50 step。step 10〜20 はまだ過渡的で `init_loss` が過小評価 (ratio が緩く見える) リスク
- **pytest single-source** (T-M2.6 と統一、DRY)
- **noise level conditioning test を独立に追加** (reverse sample acceptance の代替、sub-model 4 で感度高)
- **synthetic CPU smoke** を `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` (`@pytest.mark.cpu_only`) に集約 (wrapper script なし)

#### 検討した代替案 (M3 レビューで検討追加)
- **`c_rescale` config 切替で smoke 内 ablation 可能化** (T-M3.2 で `c_rescale` config を実装する前提): smoke が sub-model 4 で fail なら `c * 1000` rescale を切り替えて再試行 → 学習する場合 T-M1.5 NoiseEmbedding の根本問題確定
- **未訓練 sub-model 2-4 を identity (= eps_pred = 0) で mock し、sub-model 1 のみ訓練済 → 4-step reverse**: smoke acceptance に追加 (`test_smoke_reverse_with_mocks`)、sampler の符号 / index バグを早期検知 (T-M3.3 の β 負値問題の検証手段)
- **sub-model 4 並列 smoke** (`pytest -k "smoke_diff" -n 2` で k=1 と k=4 を 2 GPU 並列): CI コスト 1×→2× だが sub-model 4 失敗の早期検知価値が上回る (atexit double-close 対策 `_safe_close` 前提)
- **4 sub-model すべて smoke**: 時間 4 倍 (80 分) だが完全 sanity、reverse sample acceptance も検証可。**M5.2 直前で実施** を §9.1 に申し送り。
- **reverse sample test (本物) を smoke に含める**: 4 sub-model すべて random init で reverse → 波形の dynamic range のみ確認 (品質は無意味)。実装コスト低だが結果の信頼性低 → out of scope (mock 版のみ採用)。
- **GitHub Actions CI 化**: GPU runner 必須、M6 まで保留 (T-M2.6 と同じ)。synthetic CPU smoke のみ `pytest -m cpu_only` で nightly job に分離可。

#### 再評価トリガー
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| sub-model 4 primary | M5.2 結果 | 4 sub-model 揃ったら全 sub-model で smoke を再実行 |
| `final / init < 0.05` | M5.2 / M6.2 | 厳しすぎる場合 0.1 に緩和 |
| cos similarity < 0.99 | M1 PoC (T-M1.5 untrained baseline) | 初期実測値で pin、untrained で満たされる場合 < 0.95 に強化 |
| `c * 1000` rescale | sub-model 4 学習しない時 | T-M1.5 で rescale on/off を flag 化、本 smoke で `c_rescale` config 切替 ablation |
| sub-model 1 subordinate | sub-model 1 を primary に戻す必要が出た時 | 通常は不要 (sub-model 4 が最難で sufficient) |

### 8.2 思想 / 哲学の見直し
- **smoke の 3 軸 minimal sanity 確定**: T-M2.6 GAN smoke と並べて「step が回る + finite + 単調減少」の minimal sanity に固定。「品質」「収束性」「汎化」は smoke では確認しない (M5.2 移譲)
- **`scripts/smoke_diff_synthetic.py` を test 自体に集約**: wrapper script なしで `pytest tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` で起動、CI .yml で `pytest -m cpu_only` 1 行で十分
- **粒度**: 適切。smoke は「最小コストで最大検出力」、sub-model 4 (最難) のみ × 1000 step がその balance
- **別マイルストーン移行**: なし
- **インターフェース**: pytest single-source 統一 (T-M2.6 と整合)
- **smoke pass = sub-model 4 sanity のみ**:
  - reverse sample 品質 / 4 sub-model 統合品質は別評価 (M5.2 / M6.2)
  - sub-model 1-3 の学習可能性は別保証 (M5.2 で全 sub-model smoke 再実施を推奨)。ただし最難 sub-model 4 が pass していれば false positive リスクは小

### 8.3 学んだこと (チケット完了後に追記)
- TBD
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M3.2 / T-M3.3 から受領
- **T-M3.2 から受領**: `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg) -> dict[str, float]` 公開関数、`build_diff_model_and_optimizer(cfg, sub_idx) -> (model, opt, sched)`、**`c_rescale` config で 1.0 / 1000.0 切替** (smoke 内 ablation 用)
- **T-M3.3 から受領**: 本物の `reverse_sample` (4 sub-model 必要) は M5.2 で初使用。本 smoke では **β 負値問題が smoke で NaN を引き起こすかを早期検知** (`test_smoke_reverse_with_mocks` で sub-model 2-4 を identity mock した 4-step reverse)
- **TensorBoard flush 保証**: `atexit.register(_safe_close, writer)` (try/except OSError で double-close wrap) + SIGINT/SIGTERM (T-M2.6 と同じ)

#### T-M5.2 (Diff 1 sub-model 1 epoch) へ
- **使用方法**: smoke pass 後、1 sub-model × 1 epoch (全データ) へ昇格
- **smoke pass は architectural sanity のみ、収束性は別評価** (M5.2 で実測)
- **smoke で確認済み** (primary = sub-model 4):
  - sub-model 4 の forward + backward が finite
  - MSE loss が overfit する (architectural sanity)
  - noise level conditioning が機能 (cos < 0.99、sub-model 4 で感度 ~70x)
  - sampler の符号 / index バグなし (mock reverse で NaN なし)
- **smoke で確認できない / M5.2 で別途検証**:
  - sub-model 1/2/3 の学習可能性 (M5.2 で全 sub-model smoke を **再実行** することを強く推奨)
  - reverse sample 品質 (本物の 4 sub-model 必要)
  - 100 utterances 以上での汎化
  - validation loss
- **失敗時のフィードバック方向 / 遡及調査優先順位** (smoke が pass しなかった場合):
  - **subordinate gate**: LibriTTS-R 1 sample で fail → synthetic CPU smoke で再試行 → synthetic でも fail なら **architecture バグ確定**
  - **遡及調査優先順位** (M3 レビューで更新):
    1. **T-M3.3** (β 負値、σ_t index、reverse step 式 — mock reverse の NaN で早期検知可能)
    2. **T-M1.5** (NoiseEmbedding `c * 1000` rescale が必要かどうか、`c_rescale` config 切替 ablation)
    3. **T-M3.1** (BAND_BOUNDS 計算誤り、`c` の値域が想定外)
    4. **T-M3.2** (MSE 計算誤り、noise scaling `√abar` / `√(1-abar)` 取り違え、bf16 c 精度)
    5. **T-M1.4** (Generator output head、Linear → reshape → clip)
    6. **T-M1.1** (ConvNeXt block への additive bias 注入、FiLM ではなく per-block 独立 `Linear(512, 512)`)
- **`c * 1000` rescale 申し送り**: sub-model 4 が学習しない場合、本 smoke で `c_rescale` config (T-M3.2) を 1.0 → 1000.0 に切替えて ablation を最優先で実施

#### T-M6.2 (4 sub-model 本格訓練) へ
- **smoke で sub-model 4 (最難) のみ確認** したため、4 sub-model 統合時には:
  - sub-model 1-3 の学習可能性は M5.2 で全 sub-model smoke 再実行で確認 (sub-model 4 pass で false positive リスクは小)
  - reverse sample の `[-1, 1]` 範囲は M3.3 unit test で先行確認
  - post-filter (T-M3.4) は別途 fit 必須

### 9.2 ドキュメント更新
- [ ] `docs/milestones.md` §M3.5 Acceptance を **修正**:
  - 1 つ目 (MSE 5% 以下): そのまま採用
  - 2 つ目 (reverse sample): **M5.2 に移譲、本 ticket では out of scope** と明記、代わりに noise level conditioning test を採用
- [ ] `docs/tickets/index.md` T-M3.5 ステータス更新
- [ ] (該当時) `docs/training.md` §3 に smoke 実行手順を追記する余地

### 9.3 Open question として残ったもの
- **解決できなかった疑問**:
  - sub-model 4 (primary) と sub-model 1-3 で smoke の難易度差はどの程度か → M5.2 で全 sub-model 実測 (sub-model 4 が最難で pass すれば他は安全との前提)
  - `c * 1000` rescale が不要か必要か → 本 smoke で `c_rescale` config 切替 ablation で判定 (sub-model 4 fail 時)
  - `cos < 0.99` 閾値が untrained model でも満たされるか → **T-M1.5 着手前に untrained baseline 計測が必要** (§6.2 / §6.3)
- **`docs/open-questions.md` への追記要否**: smoke 失敗時のみ再オープン (現時点では追記不要)
