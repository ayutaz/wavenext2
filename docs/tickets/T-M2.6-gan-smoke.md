---
id: T-M2.6
title: GAN smoke (1 sample × 1000 step overfitting)
milestone: M2
phase: M2
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M2.5]
blocks: [T-M5.1]
related_docs:
  - docs/milestones.md#m26-smoke-training-overfitting-test
  - docs/training.md
---

# T-M2.6: GAN smoke training (1 sample × 1000 step overfitting test)

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.6](../milestones.md#m26-smoke-training-overfitting-test)
> **依存**: [T-M2.5](T-M2.5-train-gan.md) (前提: [T-M2.1](T-M2.1-dataset.md), [T-M2.4](T-M2.4-gan-model.md), [T-M1.6](T-M1.6-sub-model.md)) / **後続**: [T-M5.1](T-M5.1-gan-1epoch.md)

## 1. タスク目的とゴール

### 目的
GAN-WaveNeXt 2 の訓練 stack (Dataset / Generator / Discriminator / Loss / Training loop) が **end-to-end で動作し、勾配が正しく流れているか** を、1 utterance だけを 1000 step 訓練して **overfitting させる** ことで最小コストで検証する。
これは M5 の本格 1 epoch / M6 の本格訓練 (410h) に進む前の **gate** であり、ここで失敗したら必ず M2.4 (戻り値 n_t vs y_{t-1}) / M2.3 (hinge loss 符号) / M1.6 / M1.4 まで遡って修正する。

### ゴール
- [ ] `tests/test_train_gan_overfit.py::test_smoke_completes` が pytest single-source として実装され、`@pytest.mark.slow @pytest.mark.gpu` でデフォルト skip
- [ ] `scripts/smoke_gan.py` は薄い wrapper として `subprocess.run(["pytest", "tests/test_train_gan_overfit.py::test_smoke_completes", "-v", "-m", "slow and gpu"])` を呼ぶだけ (DRY 原則、2 経路維持しない)
- [ ] `scripts/smoke_gan_synthetic.py` (sine 波 1 sample × 500 step、CPU で 1〜2 分) を `@pytest.mark.cpu_only` で subordinate gate として追加 (nightly CI 候補、ただし GitHub Actions 化は M6 まで保留)
- [ ] `configs/gan_wavenext2_smoke.yaml` で smoke 用 hyperparameter (max_steps=1000, batch=1, segment_length=16384, validation off) を別 config 化
- [ ] 1000 step 後の MR-STFT loss が初期値 (step 10〜20 平均、warmup 後の最初の真の loss) の **10% 以下** に低下
- [ ] 1000 step 終了時点で生成波形 `y_0` の peak が `[-1, 1]` に収まる (clip が機能)
- [ ] TensorBoard scalars (`loss_G`, `loss_D`, `loss_stft`, `loss_adv`, `loss_fm`) に NaN/Inf 出現なし。**MR-STFT 閾値 0.1 のみ厳格適用、hinge GAN loss は finite + 振動許容を別検証**
- [ ] 全 1000 step を 1 GPU で **30 分以内** に完走 (A100 / RTX 3090 想定、CPU では skip 可)
- [ ] (オプション) 100 step ごとに `logs/smoke/step_{N}.wav` を保存して聴感確認できる
- [ ] `atexit.register(writer.close)` + SIGINT/SIGTERM handler で event ファイルを確実に flush (loss curve が消える失敗モード回避)
- [ ] `docs/milestones.md` §M2.6 Acceptance 2 項目をクリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `tests/test_train_gan_overfit.py` (**pytest single-source**、`@pytest.mark.slow @pytest.mark.gpu` で skip 可。`test_smoke_completes` 本体ロジックはここに集約)
  - `scripts/smoke_gan.py` (薄い wrapper、`subprocess.run(["pytest", ...])` を呼ぶだけ。2 経路維持を避ける DRY 設計)
  - `scripts/smoke_gan_synthetic.py` (sine 波 1 sample × 500 step、CPU で 1〜2 分。subordinate gate / nightly CI 候補)
  - `configs/gan_wavenext2_smoke.yaml` (smoke 専用 config)
- 編集:
  - `docs/milestones.md` §M2.6 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M2.6 ステータス更新

### 2.2 主要構造

```python
# tests/test_train_gan_overfit.py (pytest single-source)
"""GAN smoke training: 1 utterance × 1000 step overfitting test.

pytest single-source 設計。CLI 起動 (scripts/smoke_gan.py) はこの test を subprocess で呼ぶ薄い
wrapper として実装され、ロジック重複を避ける (DRY)。
T-M2.5 の `train_gan_step` を import して再利用、`atexit.register(writer.close)` で SIGINT 時にも
event ファイルを確実に flush する (loss curve が消える失敗モードを防ぐ)。
1000 step 後に MR-STFT loss が初期値の 10% 以下にならない場合は **Critical アラート** を出す。
"""

from __future__ import annotations
import argparse
import atexit
import signal
import yaml
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from wavenext2.train.train_gan import train_gan_step, build_models_and_optimizers
from wavenext2.data.dataset import LibriTTSRDataset


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes(config_path: str = "configs/gan_wavenext2_smoke.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # 1. Single-sample dataset (Subset で 1 utterance だけを永遠に loop)
    full_ds = LibriTTSRDataset(cfg["data"]["filelist"], segment_length=cfg["data"]["segment_length"], mode="val")
    single_ds = Subset(full_ds, [cfg["data"].get("smoke_idx", 0)])
    loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0)

    # 2. Models / optimizers (T-M2.5 流用)
    G, D, opt_G, opt_D, sched_G, sched_D = build_models_and_optimizers(cfg)
    writer = build_tensorboard_writer(cfg["logging"]["log_dir"])
    # atexit + SIGINT/SIGTERM handler で event ファイルを確実に flush
    atexit.register(writer.close)
    signal.signal(signal.SIGINT, lambda *_: (writer.close(), exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (writer.close(), exit(143)))

    # 3. 1000 step loop
    # `init_loss` は step 10〜20 (warmup 後の最初の真の loss) で測る
    # 理由: step 0〜10 は InverseLR warmup 域 (lr ≈ 1e-7) で実質まだ更新されておらず、
    # `init_loss` の意味が崩れるため。M1 PoC 後に実測値を pin する申し送り (§9.1) 参照。
    losses_init: list[float] = []
    step = 0
    while step < cfg["train"]["max_steps"]:
        for mel, audio in loader:
            mel, audio = mel.cuda(), audio.cuda()
            logs = train_gan_step(G, D, opt_G, opt_D, mel, audio, cfg)
            # NaN/Inf hook (Critical) — MR-STFT は厳格、hinge GAN は finite のみチェック (振動許容)
            assert torch.isfinite(torch.tensor(logs["loss_stft"])).all(), f"NaN/Inf at step {step}"
            assert torch.isfinite(torch.tensor(logs["loss_G"])).all(), f"loss_G NaN/Inf at step {step}"
            if 10 <= step < 20:
                losses_init.append(logs["loss_stft"])
            for k, v in logs.items():
                writer.add_scalar(k, v, step)
            if step % 100 == 0:
                save_sample_wav(G, mel, audio, step, cfg["logging"]["log_dir"])
            step += 1
            if step >= cfg["train"]["max_steps"]:
                break

    # 4. 最終検証 — MR-STFT 閾値 0.1 のみ厳格、hinge GAN は finite + 振動許容で別検証
    final_loss = logs["loss_stft"]
    init_loss = max(sum(losses_init) / len(losses_init), 1e-4)  # floor で unstable ratio 回避
    ratio = final_loss / init_loss
    print(f"[smoke] init MR-STFT = {init_loss:.4f}, final = {final_loss:.4f}, ratio = {ratio:.4f}")
    assert ratio < 0.1, f"Overfitting failed: ratio {ratio:.4f} >= 0.1 (T-M2.4 / T-M1.6 / T-M2.3 を確認)"

    # 出力範囲チェック
    with torch.no_grad():
        y_0 = G(mel, audio.shape[1])
    assert y_0.abs().max() <= 1.0 + 1e-5, f"Clip 違反: max abs = {y_0.abs().max()}"


# scripts/smoke_gan.py — 薄い wrapper (2 経路維持しない DRY 設計)
# import subprocess, sys
# def main() -> None:
#     subprocess.run(
#         [sys.executable, "-m", "pytest",
#          "tests/test_train_gan_overfit.py::test_smoke_completes",
#          "-v", "-m", "slow and gpu"],
#         check=True,
#     )
# if __name__ == "__main__":
#     main()
```

### 2.3 使用するハイパーパラメータ / 定数 (`configs/gan_wavenext2_smoke.yaml`)

```yaml
# configs/gan_wavenext2_smoke.yaml — smoke 用、本番設定 (gan_wavenext2.yaml) と異なる
seed: 42

data:
  filelist: data/filelists/train.txt
  segment_length: 16384         # ≈ 0.68s @ 24kHz
  smoke_idx: 0                  # 0 番目の utterance を使う (deterministic)

model:
  T: 4                          # sub-model 数 (T-M2.4)
  sub_model:
    mel_channels: 128
    n_fft: 2048
    hop_length: 300
    win_length: 1200
    dim: 512
    n_blocks: 8

train:
  max_steps: 1000               # smoke は 1000 step (本番は 2M)
  batch_size: 1                 # 1 sample overfit
  grad_clip: 1.0
  validation: false             # smoke では validation skip

optimizer:
  generator:
    lr: 1.0e-4
    betas: [0.8, 0.99]
    weight_decay: 1.0e-3
  discriminator:
    lr: 2.0e-4
    betas: [0.8, 0.99]
    weight_decay: 1.0e-3

scheduler:
  inv_gamma: 200000
  power: 0.5
  warmup: 0.999

loss:
  d_gan: 1.0
  d_fm: 10.0
  mrstft_sc: 2.5
  mrstft_mag: 2.5

logging:
  log_dir: logs/smoke_gan
  save_wav_every: 100           # 100 step ごとに wav 保存
```

| 名前 | 値 | 出典 |
|---|---|---|
| `max_steps` | 1000 | milestones.md §M2.6、本ticket §1 |
| `batch_size` | 1 | milestones.md §M2.6 |
| `segment_length` | 16384 | docs/training.md §1.3 (Vocos 値) |
| `smoke_idx` | 0 (deterministic) | 再現性のため |
| `validation` | false | smoke では skip (実行時間短縮) |
| `save_wav_every` | 100 | 聴感確認用 |
| 失敗判定閾値 | `final / init < 0.1` | 経験則 (overfitting なら 1〜2 桁低下) |

### 2.4 アルゴリズム / 処理フロー

1. `configs/gan_wavenext2_smoke.yaml` を読み込み
2. `LibriTTSRDataset` から **1 つの utterance だけ** を `Subset([idx])` で抽出
3. `DataLoader(batch_size=1, num_workers=0)` で無限ループに供給
4. T-M2.5 の `train_gan_step()` を 1000 回呼ぶ
5. **step 10〜20** (warmup 後の最初の真の loss) の `loss_stft` を `init_loss` として記録
   - step 0〜10 は `InverseLR` warmup 域 (lr ≈ 1e-7) で実質まだ更新されていないため除外
6. **NaN/Inf hook**: 毎 step `assert torch.isfinite(loss_stft)` (厳格) + `assert torch.isfinite(loss_G)` (finite のみ、振動許容)
7. **100 step ごと** に sample wav 保存 (`logs/smoke/step_{N}.wav`)
8. 1000 step 完了後に MR-STFT のみ `final_loss / init_loss < 0.1` を assert (hinge GAN は finite + 振動許容で別検証)
9. 生成波形 `y_0.abs().max() <= 1.0` を assert
10. `atexit.register(writer.close)` + SIGINT/SIGTERM handler で `step_1000.wav` だけ残って loss curve が消える失敗モードを防ぐ

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `scripts/smoke_gan.py` + `configs/gan_wavenext2_smoke.yaml` 実装 | general-purpose |
| Reviewer | 1 | loss 推移 / 出力 wav の聴感確認 + Critical アラート時の遡及調査 (T-M1.6 / T-M2.4 / T-M2.3) | general-purpose |
| Tester | 1 | smoke 実行 + TensorBoard 確認 + Acceptance 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M2.5 完了が前提)
- 並列実行する場合の最大並列数: 1
- 後続 T-M5.1 とは順次。

## 4. 提供範囲 (Scope)

### In Scope
- `tests/test_train_gan_overfit.py` 実装 (**pytest single-source**、`@pytest.mark.slow @pytest.mark.gpu`)
- `scripts/smoke_gan.py` 薄い wrapper (`subprocess.run(["pytest", ...])`、DRY 設計)
- `scripts/smoke_gan_synthetic.py` (sine 波 1 sample × 500 step、CPU で 1〜2 分、subordinate gate)
- `configs/gan_wavenext2_smoke.yaml` 別 config
- T-M2.5 から `train_gan_step` を import して再利用 (重複実装しない)
- `atexit.register(writer.close)` + SIGINT/SIGTERM handler (event ファイル flush 保証)
- NaN/Inf hook (MR-STFT 厳格、hinge GAN は finite のみ振動許容)
- `final_loss / init_loss < 0.1` assertion (MR-STFT 限定)
- 出力波形 `[-1, 1]` clip assertion
- 100 step ごとの wav 保存 (聴感確認用)

### Out of Scope
- 本格 1 epoch 訓練 (T-M5.1)
- 全 validation set での評価 (T-M5.1)
- UTMOS / NISQA / MCD 計算 (T-M4.*)
- 複数 utterances overfit (§8 代替案)
- multi-GPU / DDP (M6 で検討)
- AMP fp16 (M6 で検討、smoke は fp32)
- Diff smoke (T-M3.5)
- **GitHub Actions CI 化** (GPU runner 必須、self-hosted runner は M6 まで保留。synthetic smoke のみ CPU で nightly CI 候補、GPU smoke は手動 / Colab 経由のまま)

### Deliverable
- ファイル:
  - `tests/test_train_gan_overfit.py` (新規、**pytest single-source**、`@pytest.mark.slow @pytest.mark.gpu`)
  - `scripts/smoke_gan.py` (新規、薄い wrapper)
  - `scripts/smoke_gan_synthetic.py` (新規、`@pytest.mark.cpu_only`、CPU で 1〜2 分)
  - `configs/gan_wavenext2_smoke.yaml` (新規)
- 関数 / クラス:
  - `test_smoke_completes()` (pytest 本体)
  - `test_smoke_synthetic_cpu()` (subordinate gate)
  - `main()` (薄い wrapper)
- ドキュメント差分:
  - `docs/milestones.md` §M2.6 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M2.6 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト
- [ ] `tests/test_train_gan_overfit.py::test_smoke_completes` — **`@pytest.mark.slow @pytest.mark.gpu` でデフォルト skip**、1000 step で完走、MR-STFT NaN/Inf なし、hinge GAN は finite のみ確認 (振動許容)
- [ ] `tests/test_train_gan_overfit.py::test_smoke_synthetic_cpu` — **`@pytest.mark.cpu_only`**、sine 波 1 sample × 500 step を CPU で 1〜2 分以内 (subordinate gate、nightly CI 候補)
- [ ] `tests/test_train_gan_overfit.py::test_smoke_loss_decrease` — `final_loss / init_loss < 0.1` (**MR-STFT のみ厳格適用**、hinge GAN は別検証)
- [ ] `tests/test_train_gan_overfit.py::test_smoke_output_range` — 生成波形 `[-1, 1]` 範囲内
- [ ] `tests/test_train_gan_overfit.py::test_smoke_config_keys` — smoke config に必須 key が揃っている

### 5.2 e2e / 結合テスト
- [ ] `uv run python scripts/smoke_gan.py --config configs/gan_wavenext2_smoke.yaml` が exit code 0 で完了
- [ ] TensorBoard scalars (`loss_G`, `loss_D`, `loss_stft`, `loss_adv`, `loss_fm`) が単調減少傾向 (短期 spike は許容)
- [ ] `logs/smoke_gan/step_{0,100,...,1000}.wav` が 11 ファイル保存される
- [ ] (オプション) 聴感: step_1000.wav が GT に近い (人間レビュー)

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.6 より転記)
- [ ] 1 utterance で 1000 step 後、再構成音声が GT に近づく (MR-STFT loss が初期値の 10% 以下)
- [ ] 生成波形が `[-1, 1]` に収まる

### 5.4 追加 acceptance (本チケット独自)
- [ ] TensorBoard scalars に NaN/Inf 出現なし (NaN 検知 hook が機能)
- [ ] 1 GPU で **30 分以内** に完走 (A100 想定。RTX 3090 でも 1 時間以内)
- [ ] CPU 環境では `@pytest.mark.gpu` で skip 可能 (CI が CPU only でも fail しない)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (失敗時の遡及調査優先順位)

- **CRITICAL: 1 sample overfitting が起きない**
  - 影響: アーキテクチャ or loss にバグあり、M5 / M6 を起動しても無駄
  - 検知: `final_loss / init_loss >= 0.1` で即停止 + アラート
  - **subordinate gate**: LibriTTS-R 1 sample で fail → `scripts/smoke_gan_synthetic.py` で再試行 → synthetic でも fail なら **architecture バグ確定** (T-M1.4 / T-M1.6 / T-M2.2 / T-M2.4 を順次見直し)
  - 遡及調査 (優先順位順):
    1. **T-M1.6 戻り値仕様 (n_t vs y_{t-1})**: 最重要。`y = y - n_t` か `y = sub_model(...)` のどちらが正しいか確認 (T-M1.6 §6.1 critical 項目)
    2. **T-M2.4 fixed-point iteration の loss 戦略 / `return_intermediates`**: `y - n_t` の符号と初期化 `y_T = zeros` の整合性
    3. **T-M2.3 hinge GAN loss の符号 / FM weight scale**: `D_loss = relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()` の `+` / `-` 取り違え
    4. **T-M1.4 generator init / clip 飽和**: clip(-1, 1) が **学習を阻害** している可能性 (`tanh` fallback 検討、T-M1.4 §1 `final_activation` 引数で切り替え)
    5. **T-M2.5 D 強すぎ / AMP 境界**: G/D の lr 比 (`1e-4 : 2e-4`) が崩れていないか、AMP fp16 の境界条件

- **CRITICAL: `init_loss = mean(losses[0:10])` の意味が崩れる**
  - 影響: step 0〜10 は warmup 域 (`InverseLR` で lr ≈ 1e-7) で実質まだ更新されていない、`init_loss` の意味が崩れる
  - 修正案 1: **step 10〜20 にずらす** (warmup 後の最初の真の loss、本ticket では採用)
  - 修正案 2: scheduler を smoke 用に bypass
  - 修正案 3: `init_loss` を step 0 random init 直後の 1 forward のみ (train ループ前に分離)

- **CRITICAL: hinge GAN loss は overfit でも安定的に減らない**
  - 影響: 閾値 0.1 を hinge GAN にも適用すると false positive で smoke が無限に fail
  - 緩和: **閾値 0.1 は MR-STFT loss のみに適用**、hinge GAN loss は finite + 振動許容を別検証
  - 検知: `assert torch.isfinite(loss_G)` のみ、`loss_G / init_G_loss < 0.1` は判定しない

- **CRITICAL: `log_dir` タイムスタンプ衝突回避と SIGINT 時 event 未 flush**
  - 影響: SIGINT 時に `step_1000.wav` だけ残って loss curve が消える失敗モード
  - 緩和: `atexit.register(writer.close)` + SIGINT/SIGTERM handler で確実に flush

- **CRITICAL: MR-STFT 初期値の order が不明**
  - 影響: `docs/milestones.md` に書かれていない → smoke 失敗時に「閾値が厳しいだけか本当のバグか」判別困難
  - 緩和: M1 PoC 後に `init_loss` 実測値を pin する申し送りを §9.1 に追加

- **CRITICAL: 1000 step で十分か (5000 step 必要かも)**
  - 影響: 偽陰性 (本当はバグなしだが 1000 step では足りない) で M5 起動を遅らせる
  - 緩和: smoke を 2 phase で実行 — phase A (1000 step) で fail なら phase B (5000 step) で再試行
  - alternative: §8 代替案「5000 step まで延長」を採用 (実行時間 1.5h、A100 必要)

- **CRITICAL: TensorBoard scalars が NaN になる**
  - 影響: 学習が divergent (grad explosion / loss 設計バグ)
  - 検知: 毎 step `assert torch.isfinite(loss_G)` で即停止
  - 緩和: grad_clip=1.0 が機能しているか確認、lr を 1/2 に下げて再開

#### 通常項目

- **GPU 環境が借りられない (Windows + CPU only)**
  - 影響: smoke 実行が困難 (CPU で 1 step ≈ 10s → 1000 step ≈ 2.8h、borderline)
  - 緩和:
    - Colab Pro (T4) を利用 → 1 step ≈ 1s → 1000 step ≈ 17 分 (推奨)
    - Lambda Labs spot (A100) を 1 時間借りる
    - smoke の規模を 500 step に半減 (品質チェックは弱まる)
- **GPU メモリ不足 (1 GPU で OOM)**
  - 影響: smoke 完走できない (T=4 sub-model 並列保持で activation memory 大)
  - 緩和: T-M1.4 `enable_grad_ckpt=True` を smoke config で有効化
- **CPU で smoke を実行する場合の実行時間**
  - 影響: CI で 2.8h は許容外
  - 緩和: `tests/test_train_gan_overfit.py` は `@pytest.mark.gpu` で CPU CI skip、Bash 経由の手動実行を推奨
- **Subset([0]) が deterministic でないリスク**
  - 影響: 再現性低下、smoke 結果が毎回違う
  - 緩和: `LibriTTSRDataset` の `mode="val"` で固定 gain (-3 dB) を使う、`smoke_idx` を config に明示
- **生成 wav の channels 数 / sample rate**
  - 影響: `torchaudio.save` 失敗
  - 緩和: 24kHz mono を明示
- **TensorBoard log_dir 衝突**
  - 影響: 過去の smoke ログと混ざる
  - 緩和: `log_dir = f"logs/smoke_gan/{datetime.now()}"` でタイムスタンプ付与
- **`init_loss` が極端に小さい (e.g., 1e-10) と ratio 計算が unstable**
  - 影響: false negative (overfitting failed と誤判定)
  - 緩和: `init_loss = max(init_loss, 1e-4)` で floor を入れる

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: すべて確定済み
- 本チケット固有の判断:
  - **`final_loss / init_loss` の閾値**: `0.1` (milestones.md §M2.6 「初期値の 10% 以下」より)。経験則的に overfitting なら 1〜2 桁低下するため妥当
  - **`init_loss` の平均区間**: 「step 0〜10 の平均」を採用 (step 0 単独だと variance 大)。`init_loss = mean(losses[0:10])`
  - **失敗時の停止方針**: `assert` で即停止 (続けても無駄、Critical アラート優先)
  - **wav 保存頻度**: 100 step ごと (1000 step / 100 = 11 ファイル、storage は ~1MB/file × 11 = 11MB で軽量)

### 6.3 他チケットとの整合性

- **T-M2.5 (train_gan)** との整合:
  - 期待: `train_gan_step(G, D, opt_G, opt_D, mel, audio, cfg) -> dict[str, float]`
  - 期待: `build_models_and_optimizers(cfg) -> (G, D, opt_G, opt_D, sched_G, sched_D)`
  - 不整合があれば T-M2.5 にフィードバック
- **T-M2.1 (Dataset)** との整合:
  - 期待: `LibriTTSRDataset(filelist, segment_length, mode)` で `__getitem__` が `(mel, audio)` を返す
  - mode="val" で固定 gain (-3 dB) → smoke の再現性確保
- **T-M2.4 (GAN モデル)** との整合:
  - 期待: `G(mel, audio_length)` で `(B, audio_length)` を返す (forward の signature が T-M2.4 §9.1 と一致)
  - **戻り値仕様 (n_t vs y_{t-1})** が T-M1.6 §6.1 で確定した実装を踏襲しているか確認
- **T-M2.3 (Loss)** との整合:
  - 期待: hinge GAN loss、MR-STFT (SC + Mag L1)、FM loss が全て import 可能で finite な値を返す

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `scripts/smoke_gan.py` が T-M2.5 の関数を **重複実装せず流用** している (DRY 原則)
- [ ] `configs/gan_wavenext2_smoke.yaml` が `gan_wavenext2.yaml` 本番 config と **明確に分離** されている (max_steps / batch / validation の差分が明示)
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)
- [ ] NaN/Inf hook が機能 (artificial に NaN を注入してテスト可能か)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] エラー処理: `assert` メッセージが具体的 (どの critical 項目を確認すべきか明示)
- [ ] TensorBoard log_dir がタイムスタンプ付きで衝突しない
- [ ] 参考実装をコピーしていない (一から書いたか)
- [ ] 1000 step の実行時間が想定内 (A100 30 分、Colab T4 17 分)

## 8. ゼロから作り直すとしたら

### 8.1 別の設計を採るとしたら

#### 採用設計
- **1 sample × 1000 step overfitting** (最小コストで最大検出力)
  - 1 sample だから overfitting が起きるべき (起きないなら critical)
  - 1000 step なら CPU でも borderline 実行可
- **pytest single-source に統一** (採用昇格)
  - `scripts/smoke_gan.py` と `tests/test_train_gan_overfit.py` の 2 経路維持はメンテコスト
  - `@pytest.mark.slow @pytest.mark.gpu` でデフォルト skip、CLI 起動は `pytest tests/test_train_gan_overfit.py::test_smoke_completes -v -m "slow and gpu"` で代替する DRY 設計
  - `scripts/smoke_gan.py` は薄い wrapper として `subprocess.run(["pytest", ...])` を呼ぶだけ
- **synthetic data smoke (sine 波) を subordinate gate として追加** (採用昇格)
  - 1 sample LibriTTS-R で fail → synthetic で再試行 → synthetic でも fail なら architecture バグ確定
  - `scripts/smoke_gan_synthetic.py` を `@pytest.mark.cpu_only` で nightly CI に追加 (GPU runner 不要)

#### 検討した代替案
- **5 utterances × 1000 step**: より厳しい (overfitting しにくい)、smoke の意義が薄れる
- **100 utterances (full val set)**: smoke と言わない、T-M5.1 と被る
- **1 utterance × 5000 step**: full convergence 確認、A100 で 2.5h
- **Synthetic data (440 Hz sine + 880 Hz sine)**: 数値検証に強い、LibriTTS-R 不要 (T-M2.1 と decouple 可能)、ただし聴感確認できない → **subordinate gate として採用** (上記)
- **1 sample × 33k step (1 epoch)**: T-M5.1 と被る、smoke の粒度ではない
- **GitHub Actions CI 化**: **明示的に却下** (GPU runner 必須、self-hosted runner は M6 まで保留)。synthetic smoke のみ CPU で nightly CI 候補、GPU smoke は手動 / Colab 経由のまま

#### 再評価トリガー
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| 1 sample × 1000 step | M5.1 smoke 結果 | M5.1 で 1 epoch が fail なら smoke の閾値 (10%) を緩める or 5000 step に延長 |
| `final/init < 0.1` 閾値 | M5.1 / M6.1 結果 | 経験則的に厳しすぎる場合は 0.2 に緩和 |
| 100 step ごとの wav 保存 | M5 phase review | storage が問題なら 250 step ごとに減らす |

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切。smoke は「最小コストで最大の検出力」を目指すべきで、1 sample × 1000 step がその balance
- **別マイルストーン移行**: なし。M2 (GAN 実装) の最終 gate として M2 に置くのが自然
- **インターフェース**: **pytest single-source に統一** (採用昇格)。`scripts/smoke_gan.py` は薄い wrapper として `subprocess.run(["pytest", ...])` を呼ぶだけ。2 経路維持を避けることで DRY 原則を遵守
- **smoke が pass しても M5.1 (1 epoch) で fail する可能性を明示**:
  - Smoke は「architectural sanity」(forward/backward/loss sign) を見るもので、「収束性」「汎化」は見ない
  - 期待値の上限を明示することで M5.1 fail 時の誤った遡及調査 (T-M1.6 / T-M2.4 への過剰な疑い) を防ぐ
  - M5.1 で fail した場合、まず以下を疑う: hyperparameter (lr, batch_size, warmup), データ前処理 (mel 正規化、audio gain), 訓練 schedule (epoch 数、scheduler)。smoke が pass しているので architectural バグの優先度は低い

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.5 (train_gan) から受領
- **import 再利用**: `train_gan_step` を import して再利用 (重複実装しない、DRY 原則)
- **必要 API**: `build_models_and_optimizers(cfg) -> (G, D, opt_G, opt_D, sched_G, sched_D)`
- **TensorBoard flush 保証**: `atexit.register(writer.close)` + SIGINT/SIGTERM handler
- **scheduler bypass オプション**: T-M2.5 側で smoke 用に scheduler を bypass できる引数を提供すること (修正案 2)

#### T-M5.1 (GAN 1 epoch 訓練) へ
- **使用方法**: smoke が pass したら本格 1 epoch (33k step) へ昇格
  ```bash
  # T-M5.1 で実行
  uv run python scripts/train_gan.py --config configs/gan_wavenext2.yaml
  ```
- **smoke で確認済みの事項** (architectural sanity のみ):
  - Generator / Discriminator の forward + backward が finite で動作する
  - hinge GAN loss の符号が正しい
  - MR-STFT loss が低下することを確認済 (1 sample overfit に限る)
  - 出力波形が `[-1, 1]` に収まる
- **smoke で確認できない事項** (T-M5.1 で別途確認):
  - **smoke pass = architectural sanity のみで収束性は別評価** (M5.1 で fail しても smoke 設計を疑わない)
  - 100 utterances 以上での汎化性能
  - validation MR-STFT loss
  - 聴感の自然さ
- **smoke `init_loss` 実測値を M1 PoC 後に pin する申し送り**:
  - `docs/milestones.md` §M2.6 に「MR-STFT 初期値 ≈ X.X」と pin
  - 閾値 0.1 が「厳しすぎるだけか本当のバグか」判別するために実測 baseline が必要
  - M1 PoC (synthetic / single utterance) で先行測定し pin する
- **失敗時のフィードバック方向** (smoke が pass しなかった場合):
  - **subordinate gate**: LibriTTS-R 1 sample で fail → synthetic smoke で再試行 → synthetic でも fail なら **architecture バグ確定** (T-M1.4 / T-M1.6 / T-M2.2 / T-M2.4 を順次見直し)
  - **遡及調査優先順位**:
    1. T-M1.6 (戻り値仕様 n_t vs y_{t-1})
    2. T-M2.4 (fixed-point iteration の loss 戦略、`return_intermediates`)
    3. T-M2.3 (hinge GAN 符号、FM weight scale)
    4. T-M1.4 (Generator init / clip 飽和)
    5. T-M2.5 (D 強すぎ、AMP 境界)

#### T-M2.4 / T-M1.6 / T-M2.3 へ (Critical 失敗時のフィードバック先)
- smoke が fail した場合に発覚する典型バグ:
  - **T-M1.6**: `SubModelGAN.forward(mel, y_prev)` が `n_t = y_t - y_{t-1}` (residual) を返すか `y_{t-1}` (denoised) を返すか
  - **T-M2.4**: fixed-point iteration の符号 (`y = y - n_t` か `y = n_t` か)、`return_intermediates` 仕様
  - **T-M2.3**: hinge GAN loss `D_loss = relu(1 - D(real)) + relu(1 + D(fake))` の `1 ±` 符号、FM weight scale
- **synthetic smoke から fail なら architecture バグ確定**: T-M1.4 / T-M1.6 / T-M2.2 / T-M2.4 を順次見直し

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.6 Acceptance チェックボックス 2 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M2.6 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/training.md` §2 に smoke 実行手順を追記する余地あり (レビュー時判断)
  - [ ] (失敗時) `docs/open-questions.md` に再オープンする question を追記

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (smoke はそのもの自体が "answer")
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - 1 sample × 1000 step の閾値 `0.1` が経験的に妥当か (M5 / M6 結果で確認)
  - CPU only 環境での smoke 代替 (synthetic data) 採用要否 (M5 phase review)
- **`docs/open-questions.md` への追記要否**: 不要 (smoke は実験そのもの)
