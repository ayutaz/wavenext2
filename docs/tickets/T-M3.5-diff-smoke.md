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
Diff-WaveNeXt 2 の訓練 stack (Dataset / SubModelDiff / NoiseEmbedding / MSE loss / Training loop) が **end-to-end で動作し、勾配が正しく流れ、noise level conditioning が機能しているか** を、sub-model 1 のみを 1 utterance で 1000 step 訓練して **overfitting させる** ことで最小コストで検証する。
これは M5.2 (1 sub-model 1 epoch) / M6.2 (4 sub-model 本格訓練) の **gate** であり、ここで失敗したら T-M1.5 (NoiseEmbedding `c * 1000` rescale) / T-M3.1 (BAND_BOUNDS) / T-M3.2 (MSE 計算 / noise scaling) / T-M1.4 (Generator) / T-M1.1 (additive bias 注入) まで遡って修正する。

### ゴール
- [ ] `tests/test_train_diff_overfit.py::test_smoke_completes` が pytest single-source として実装され、`@pytest.mark.slow @pytest.mark.gpu` でデフォルト skip
- [ ] `scripts/smoke_diff.py` は薄い wrapper として `subprocess.run(["pytest", "tests/test_train_diff_overfit.py::test_smoke_completes", "-v", "-m", "slow and gpu"])` を呼ぶだけ (DRY、T-M2.6 同様)
- [ ] `scripts/smoke_diff_synthetic.py` (synthetic mel + sine 波 1 sample × 500 step、CPU で 1〜2 分) を `@pytest.mark.cpu_only` で subordinate gate として追加
- [ ] `configs/diff_wavenext2_smoke.yaml` で smoke 用 hyperparameter (max_steps=1000, batch=1, segment_length=25600, sub_model_idx=1, validation off) を別 config 化
- [ ] 1000 step 後の MSE loss が初期値 (step 10〜20 平均、warmup 後) の **5% 以下** に低下 (Diff は GAN より overfit しやすい、milestones.md §M3.5)
- [ ] noise level conditioning 機能テスト: 同じ `x_t` + 同じ `mel` + **異なる `c`** で `eps_pred` の cosine similarity < 0.99 (条件付けが効いている証拠)
- [ ] TensorBoard scalars (`loss_mse`, `lr`, `c_mean`, `c_std`) に NaN/Inf なし
- [ ] 全 1000 step を 1 GPU で **20 分以内** に完走 (1 sub-model のみ、GAN より軽量)
- [ ] `atexit.register(writer.close)` + SIGINT/SIGTERM handler で event ファイルを確実に flush
- [ ] `docs/milestones.md` §M3.5 Acceptance チェックボックス更新 (本 ticket では reverse sample 整合は §6 で out of scope と明示)

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `tests/test_train_diff_overfit.py` (**pytest single-source**、`@pytest.mark.slow @pytest.mark.gpu`)
  - `scripts/smoke_diff.py` (薄い wrapper、`subprocess.run(["pytest", ...])`)
  - `scripts/smoke_diff_synthetic.py` (synthetic mel + sine 波、`@pytest.mark.cpu_only`、CPU で 1〜2 分)
  - `configs/diff_wavenext2_smoke.yaml`
- 編集:
  - `docs/milestones.md` §M3.5 Acceptance チェックボックス
  - `docs/tickets/index.md` T-M3.5 ステータス

### 2.2 主要構造

```python
# tests/test_train_diff_overfit.py
"""Diff smoke: sub-model 1 × 1 utterance × 1000 step overfitting test.

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


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    sub_idx = cfg["model"]["sub_model_idx"]  # 1 (smoke では sub-model 1 のみ)

    full_ds = LibriTTSRDataset(cfg["data"]["filelist"], segment_length=cfg["data"]["segment_length"], mode="val")
    single_ds = Subset(full_ds, [cfg["data"].get("smoke_idx", 0)])
    loader = DataLoader(single_ds, batch_size=1, shuffle=False, num_workers=0)

    diff_model, opt, sched = build_diff_model_and_optimizer(cfg, sub_idx=sub_idx)
    writer = build_tensorboard_writer(cfg["logging"]["log_dir"])
    atexit.register(writer.close)
    signal.signal(signal.SIGINT, lambda *_: (writer.close(), exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (writer.close(), exit(143)))

    losses_init: list[float] = []
    step = 0
    while step < cfg["train"]["max_steps"]:
        for mel, audio in loader:
            mel, audio = mel.cuda(), audio.cuda()
            logs = train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg)
            assert torch.isfinite(torch.tensor(logs["loss_mse"])).all(), f"NaN/Inf at step {step}"
            if 10 <= step < 20:
                losses_init.append(logs["loss_mse"])
            for k, v in logs.items():
                writer.add_scalar(k, v, step)
            step += 1
            if step >= cfg["train"]["max_steps"]:
                break

    final_loss = logs["loss_mse"]
    init_loss = max(sum(losses_init) / len(losses_init), 1e-6)
    ratio = final_loss / init_loss
    print(f"[smoke-diff] init MSE = {init_loss:.6e}, final = {final_loss:.6e}, ratio = {ratio:.4f}")
    assert ratio < 0.05, f"Overfit failed: ratio {ratio:.4f} >= 0.05 (T-M1.5 / T-M3.1 / T-M3.2 を確認)"


@pytest.mark.slow
@pytest.mark.gpu
def test_noise_level_conditioning(config_path: str = "configs/diff_wavenext2_smoke.yaml") -> None:
    """同じ x_t + mel + 異なる c で eps_pred の cosine similarity < 0.99 を確認。
    NoiseEmbedding が無視されていれば cosine ≈ 1.0 になる (T-M1.5 / T-M1.1 のバグ検知)。"""
    cfg = yaml.safe_load(Path(config_path).read_text())
    diff_model, _, _ = build_diff_model_and_optimizer(cfg, sub_idx=cfg["model"]["sub_model_idx"])
    diff_model.eval()
    B, T = 1, cfg["data"]["segment_length"]
    mel_dim, mel_T = cfg["model"]["sub_model"]["mel_channels"], T // cfg["model"]["sub_model"]["hop_length"]
    torch.manual_seed(0)
    mel = torch.randn(B, mel_dim, mel_T).cuda()
    x_t = torch.randn(B, T).cuda()
    L, U = diff_model.BAND_BOUNDS[cfg["model"]["sub_model_idx"] - 1]
    c_a = torch.full((B,), L).cuda()
    c_b = torch.full((B,), U).cuda()
    with torch.no_grad():
        eps_a = diff_model.sub_models[cfg["model"]["sub_model_idx"] - 1](mel, x_t, c_a)
        eps_b = diff_model.sub_models[cfg["model"]["sub_model_idx"] - 1](mel, x_t, c_b)
    cos = F.cosine_similarity(eps_a.flatten(1), eps_b.flatten(1), dim=1).mean().item()
    print(f"[smoke-diff] cos(eps(c={L:.4f}), eps(c={U:.4f})) = {cos:.4f}")
    assert cos < 0.99, f"NoiseEmbedding が無視されている疑い: cos={cos:.4f} (T-M1.5 / T-M1.1 を確認)"


# scripts/smoke_diff.py — 薄い wrapper
# import subprocess, sys
# def main() -> None:
#     subprocess.run([sys.executable, "-m", "pytest",
#                     "tests/test_train_diff_overfit.py::test_smoke_completes",
#                     "-v", "-m", "slow and gpu"], check=True)
```

### 2.3 使用するハイパーパラメータ / 定数 (`configs/diff_wavenext2_smoke.yaml`)

```yaml
seed: 42
data:
  filelist: data/filelists/train.txt
  segment_length: 25600         # Diff 設定 (hop=256 × 100 frames)
  smoke_idx: 0
model:
  sub_model_idx: 1              # smoke では sub-model 1 のみ (最も small noise, 学習が早い)
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
| `sub_model_idx` | 1 | milestones.md §M3.5 (1 sub-model のみ) |
| `segment_length` | 25600 | docs/training.md §1.3 (Diff hop=256) |
| 失敗判定閾値 | `final / init < 0.05` | milestones.md §M3.5 (Diff は GAN より overfit しやすい) |
| Optimizer | Adam(lr=2e-4, β=[0.9,0.98], wd=0) | T-M3.2 / milestones.md §M3.2 |

### 2.4 アルゴリズム / 処理フロー
1. `configs/diff_wavenext2_smoke.yaml` 読み込み (sub_model_idx=1)
2. `LibriTTSRDataset` から 1 utterance を `Subset([0])` で抽出
3. T-M3.2 の `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg)` を 1000 回呼ぶ
   - 内部で band 内 uniform sampling: `c ~ U(0.9929, 1.0)` (sub-model 1)
   - `x_t = √abar · x_0 + √(1-abar) · ε`、abar = 1 - c²
   - `eps_pred = sub_model(mel, x_t, c)`、`loss = MSE(eps_pred, ε)`
4. step 10〜20 を `init_loss` として記録 (warmup 後)
5. NaN/Inf hook (`assert isfinite(loss_mse)`)
6. 1000 step 後に `final / init < 0.05` を assert
7. **別 test**: noise level conditioning (異なる `c` で eps_pred が異なるか) を独立検証
8. `atexit` + SIGINT/SIGTERM で writer flush

**注記**: smoke では reverse sample (4 sub-model 必要) は行わない (§6.1)。`milestones.md` §M3.5 の 2 つ目の acceptance「reverse sample が GT に近い」は本 ticket では out of scope とし、M5.2 (4 sub-model 揃った段階) で再評価する。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | smoke script + synthetic + config 実装 | general-purpose |
| Reviewer | 1 | loss 推移 / cos similarity 確認 + 遡及調査 (T-M1.5 / T-M3.1 / T-M3.2) | general-purpose |
| Tester | 1 | smoke 実行 + TensorBoard 確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M3.2 / T-M3.3 完了が前提)
- 最大並列数: 1
- 後続 T-M5.2 とは順次。

## 4. 提供範囲 (Scope)

### In Scope
- `tests/test_train_diff_overfit.py` (pytest single-source、`@pytest.mark.slow @pytest.mark.gpu`)
- `scripts/smoke_diff.py` (薄い wrapper)
- `scripts/smoke_diff_synthetic.py` (synthetic mel + sine 波、`@pytest.mark.cpu_only`)
- `configs/diff_wavenext2_smoke.yaml`
- T-M3.2 から `train_diff_step` を import 再利用
- noise level conditioning 機能テスト (cosine similarity)
- NaN/Inf hook、`atexit` + SIGINT/SIGTERM 保証
- 1 GPU 20 分以内完走

### Out of Scope
- **reverse sample テスト** (4 sub-model 必要、§6.1 / §8 で詳述)
- **MR-STFT loss で reverse sample 評価** (1 sub-model 欠落状態では無意味)
- sub-model 2/3/4 の smoke (§8 代替案、本 ticket は sub-model 1 のみ)
- 本格 1 epoch 訓練 (T-M5.2)
- 4 sub-model 統合訓練 (T-M6.2)
- UTMOS / NISQA / MCD (T-M4.*)
- multi-GPU / DDP / AMP (M6)
- GAN smoke (T-M2.6)
- post-filter (T-M3.4)
- GitHub Actions CI 化 (GPU runner 必須、M6 まで保留)

### Deliverable
- ファイル: `tests/test_train_diff_overfit.py`, `scripts/smoke_diff.py`, `scripts/smoke_diff_synthetic.py`, `configs/diff_wavenext2_smoke.yaml`
- 関数: `test_smoke_completes()`, `test_noise_level_conditioning()`, `test_smoke_synthetic_cpu()`, `main()` (wrapper)
- ドキュメント差分: `docs/milestones.md` §M3.5、`docs/tickets/index.md`

## 5. テスト項目

### 5.1 Unit / 結合テスト
- [ ] `tests/test_train_diff_overfit.py::test_smoke_completes` — `@pytest.mark.slow @pytest.mark.gpu`、1000 step 完走、MSE NaN/Inf なし、`final / init < 0.05`
- [ ] `tests/test_train_diff_overfit.py::test_noise_level_conditioning` — `@pytest.mark.slow @pytest.mark.gpu`、同じ `x_t`+`mel`、異なる `c` で `cosine_similarity(eps_a, eps_b) < 0.99`
- [ ] `tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu` — `@pytest.mark.cpu_only`、synthetic mel + 440 Hz sine 波 1 sample × 500 step を CPU で 1〜2 分 (subordinate gate / nightly CI 候補)
- [ ] `tests/test_train_diff_overfit.py::test_smoke_config_keys` — smoke config の必須 key
- [ ] `tests/test_train_diff_overfit.py::test_smoke_output_finite` — 訓練中の `eps_pred` が finite

### 5.2 e2e テスト
- [ ] `uv run python scripts/smoke_diff.py` が exit code 0 で完了
- [ ] TensorBoard scalars (`loss_mse`, `lr`, `c_mean`, `c_std`) が単調減少傾向
- [ ] `c_mean` が band 内 (`[0.9929, 1.0]`) に収まる (sub-model 1)、TensorBoard で histogram 確認

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.5 より転記 + 本 ticket 修正)
- [ ] 1 utterance で 1000 step 後、MSE loss が初期値の 5% 以下
- [ ] ~~同じ utterance に対する reverse sample が GT に近い~~ → **本 ticket では out of scope** (4 sub-model 必要、§6.1 / §8 で詳述、M5.2 で再評価)
- [ ] **代替 acceptance**: noise level conditioning 機能テスト (cosine similarity < 0.99)

### 5.4 追加 acceptance (本 ticket 独自)
- [ ] TensorBoard に NaN/Inf なし
- [ ] 1 GPU で 20 分以内に完走
- [ ] CPU 環境では `@pytest.mark.gpu` で skip 可

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

- **CRITICAL: band 内 `c` の幅が狭い (sub-model 1: `[0.9929, 1.0]`)**
  - 影響: 実質 `c ≈ 0.999` 単一値で、学習信号が単調、conditioning が "学習されない" 可能性
  - 検知: `test_noise_level_conditioning` で `c=L` vs `c=U` の cos similarity が ≈ 1.0 (≥ 0.99) なら conditioning が無視されている
  - 緩和案 1: **T-M1.5 の `c * 1000` rescale を smoke で再評価** (`c=0.9929` vs `c=1.0` の差を amplify、最優先 ablation)
  - 緩和案 2: smoke を sub-model 4 (`[0.0, 0.4817]`、`c` の幅が広い) で実施し直す (§8 代替案)
  - 緩和案 3: smoke 専用に band 内ではなく `c ∈ U(0, 1)` で一時的に学習 → conditioning 単体検証

- **CRITICAL: MSE は scale 依存、sub-model 1 では loss が小さくなりがち**
  - 影響: `init_loss ≈ 1e-3` 規模で、絶対値が小さい → `ratio < 0.05` が緩く見えるが実は学習していない可能性
  - 緩和: `init_loss` の floor を `1e-6` に設定 (ratio 計算 unstable 回避)
  - 緩和: synthetic CPU smoke で baseline (random init vs converged) の MSE 絶対値を pin → M5.2 へ申し送り

- **CRITICAL: sub-model 1 (small noise) は学習が早い、sub-model 4 (大ノイズ) は遅い**
  - 影響: smoke pass = sub-model 1 で pass、sub-model 4 で fail の可能性 (M6.2 で発覚すれば手戻り大)
  - 緩和: §8 代替案「sub-model 4 で smoke」を M5.2 直前に追加実施 (option)
  - M5.2 で 4 sub-model 揃ったら全 sub-model で smoke を再実行する申し送り

- **CRITICAL: `init_loss` が極端に小さい (sub-model 1 の特性)**
  - 影響: ratio 計算 unstable、false negative
  - 緩和: floor = `1e-6`、絶対値が下がっているかも併せて確認

#### 通常項目
- **GPU 環境が借りられない**: Colab T4 で 17 分想定、Lambda Labs spot で 1 時間 → smoke はそもそも軽い、subordinate synthetic smoke は CPU で 1〜2 分
- **`segment_length=25600` で 1 GPU OOM**: T-M1.4 `enable_grad_ckpt=True` を smoke config で有効化
- **TensorBoard log_dir 衝突**: `log_dir = f"logs/smoke_diff/{datetime.now()}"`
- **`Subset([0])` の deterministic 性**: `mode="val"` で固定 gain (-3 dB)、`smoke_idx` を config に明示

### 6.2 仕様の曖昧さ
- **閾値 0.05**: `milestones.md` §M3.5 から (Diff は GAN より overfit しやすい想定)、経験則的に妥当
- **`init_loss` 平均区間**: step 10〜20 (warmup 後、T-M2.6 と統一)
- **cosine similarity 閾値 0.99**: 経験則 (NoiseEmbedding が完全 zero 化されていれば cos ≈ 1.0、機能していれば <0.95 想定)、initial 値は M1 PoC で pin
- `docs/open-questions.md`: すべて確定済み

### 6.3 他チケットとの整合性
- **T-M3.2 (train_diff)**: `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg) -> dict[str, float]`、`build_diff_model_and_optimizer(cfg, sub_idx) -> (model, opt, sched)` を expose
- **T-M3.3 (reverse_sample)**: 本 smoke では呼ばない (4 sub-model 必要)、M5.2 に持ち越し
- **T-M3.1 (DiffWaveNeXt2)**: `BAND_BOUNDS[k-1]`、`sub_models[k-1]` を smoke から直接参照 (conditioning test)
- **T-M1.5 (NoiseEmbedding)**: conditioning が "効く" こと自体は T-M1.5 unit test で先行検証、smoke は統合確認
- **T-M2.6 (GAN smoke)**: pytest single-source パターンを踏襲、wrapper を 2 経路維持しない

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `scripts/smoke_diff.py` が `train_diff_step` を **重複実装せず流用** している (DRY)
- [ ] `configs/diff_wavenext2_smoke.yaml` が `diff_wavenext2.yaml` (本番) と明確に分離
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)、reverse sample acceptance が out of scope に降格された justification が §6.1 / §9.1 に記載
- [ ] NaN/Inf hook が機能 (artificial NaN 注入で test 可)
- [ ] noise level conditioning test が **本物の bug を検知できる** (artificial に NoiseEmbedding を zero 化して fail することを確認)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] `assert` メッセージが具体的 (どの critical 項目を確認すべきか明示)
- [ ] TensorBoard log_dir がタイムスタンプ付き
- [ ] 参考実装 (FastDiff) コピーしていない
- [ ] 1000 step 実行時間 20 分以内 (1 sub-model は GAN より軽量)

## 8. ゼロから作り直すとしたら

### 8.1 別の設計を採るとしたら

#### 採用設計
- **sub-model 1 のみ × 1 utterance × 1000 step** (最小コスト、`milestones.md` §M3.5 準拠)
- **pytest single-source** (T-M2.6 と統一、DRY)
- **noise level conditioning test を独立に追加** (reverse sample acceptance の代替)
- **synthetic CPU smoke** を `@pytest.mark.cpu_only` で nightly CI 候補に

#### 検討した代替案
- **4 sub-model すべて smoke**: 時間 4 倍 (80 分) だが完全 sanity、reverse sample acceptance も検証可。**M5.2 直前で実施** を §9.1 に申し送り。
- **sub-model 1 の代わりに sub-model 4 で smoke**: 大ノイズで難しい task をテスト、`c` の幅も広い → conditioning test の感度高。sub-model 1 で学習しない場合の **最優先 ablation**。
- **reverse sample test を smoke に含める**: 4 sub-model すべて random init で reverse → 波形の dynamic range のみ確認 (品質は無意味)。実装コスト低だが結果の信頼性低 → out of scope。
- **`c * 1000` rescale ablation を smoke 内に組み込む**: smoke 自体を ablation 化、sub-model 1 が学習しない場合の最優先施策。M5.2 へ申し送り。
- **GitHub Actions CI 化**: GPU runner 必須、M6 まで保留 (T-M2.6 と同じ)。

#### 再評価トリガー
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| sub-model 1 のみ | M5.2 結果 | 4 sub-model 揃ったら全 sub-model で smoke を再実行 |
| `final / init < 0.05` | M5.2 / M6.2 | 厳しすぎる場合 0.1 に緩和 |
| cos similarity < 0.99 | M1 PoC | 初期実測値で pin、緩い場合 < 0.95 に強化 |
| `c * 1000` rescale | sub-model 1 学習しない時 | T-M1.5 で rescale on/off を flag 化、本 smoke で ablation |
| sub-model 4 で smoke | sub-model 1 pass / sub-model 4 fail で発覚時 | sub-model 4 を primary smoke に昇格 |

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切。smoke は「最小コストで最大検出力」、sub-model 1 のみ × 1000 step がその balance
- **別マイルストーン移行**: なし
- **インターフェース**: pytest single-source 統一 (T-M2.6 と整合)
- **smoke pass = sub-model 1 sanity のみ**:
  - reverse sample 品質 / 4 sub-model 統合品質は別評価 (M5.2 / M6.2)
  - sub-model 4 の学習可能性は別保証 (M5.2 で全 sub-model smoke 再実施を推奨)

### 8.3 学んだこと (チケット完了後に追記)
- TBD
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M3.2 / T-M3.3 から受領
- **T-M3.2 から import**: `train_diff_step(diff_model, opt, mel, audio, sub_idx, cfg) -> dict[str, float]`、`build_diff_model_and_optimizer(cfg, sub_idx) -> (model, opt, sched)`
- **T-M3.3 から受領**: `reverse_sample` は本 smoke では **使わない** (4 sub-model 必要)、M5.2 で初使用
- **TensorBoard flush 保証**: `atexit.register(writer.close)` + SIGINT/SIGTERM (T-M2.6 と同じ)

#### T-M5.2 (Diff 1 sub-model 1 epoch) へ
- **使用方法**: smoke pass 後、1 sub-model × 1 epoch (全データ) へ昇格
- **smoke で確認済み**:
  - sub-model 1 の forward + backward が finite
  - MSE loss が overfit する (architectural sanity)
  - noise level conditioning が機能 (cos < 0.99)
- **smoke で確認できない / M5.2 で別途検証**:
  - sub-model 2/3/4 の学習可能性 (M5.2 で全 sub-model smoke を **再実行** することを強く推奨)
  - reverse sample 品質 (4 sub-model 必要)
  - 100 utterances 以上での汎化
  - validation loss
- **失敗時のフィードバック方向** (smoke が pass しなかった場合):
  - **subordinate gate**: LibriTTS-R 1 sample で fail → synthetic smoke で再試行 → synthetic でも fail なら **architecture バグ確定**
  - **遡及調査優先順位**:
    1. **T-M1.5** (NoiseEmbedding `c * 1000` rescale が必要かどうか、最優先 ablation)
    2. **T-M3.1** (BAND_BOUNDS 計算誤り、`c` の値域が想定外)
    3. **T-M3.2** (MSE 計算誤り、noise scaling `√abar` / `√(1-abar)` 取り違え)
    4. **T-M1.4** (Generator output head、Linear → reshape → clip)
    5. **T-M1.1** (ConvNeXt block への additive bias 注入、FiLM ではなく per-block 独立 `Linear(512, 512)`)
- **`c * 1000` rescale 申し送り**: sub-model 1 が学習しない場合、本 smoke で T-M1.5 `c * 1000` rescale on/off の ablation を最優先で実施

#### T-M6.2 (4 sub-model 本格訓練) へ
- **smoke で sub-model 1 のみ確認** したため、4 sub-model 統合時には:
  - sub-model 4 (大ノイズ) の収束遅延を想定 (sub-model 1 の 2〜3 倍 step が必要かも)
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
  - sub-model 1 と sub-model 4 で smoke の難易度差はどの程度か (sub-model 4 で別途検証が必要か) → M5.2 で実測
  - `c * 1000` rescale が不要か必要か → 本 smoke で発覚すれば最優先 ablation
- **`docs/open-questions.md` への追記要否**: smoke 失敗時のみ再オープン (現時点では追記不要)
