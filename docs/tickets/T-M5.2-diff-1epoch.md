---
id: T-M5.2
title: Diff 統合スモーク (1 sub-model train-clean-100 1 epoch)
milestone: M5
phase: M5
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M3.5]
blocks: [T-M6.2]
related_docs:
  - docs/milestones.md#m52-diff-wavenext-2-1-sub-model-1-epoch
  - docs/training.md
---

# T-M5.2: Diff 統合スモーク (1 sub-model train-clean-100 1 epoch)

> **マイルストーン**: [M5](../milestones.md#m5-統合スモークテスト-作業量-smallwall-clock-は-gpu-数時間) / **サブタスク**: [M5.2](../milestones.md#m52-diff-wavenext-2-1-sub-model-1-epoch)
> **依存**: [T-M3.5](T-M3.5-diff-smoke.md) (前提: [T-M3.2](T-M3.2-train-diff.md), [T-M3.3](T-M3.3-reverse-sampler.md), [T-M4.1](T-M4.1-objective-metrics.md)) / **後続**: [T-M6.2](T-M6.2-diff-full-training.md)

## 1. タスク目的とゴール

### 目的
T-M3.5 smoke (1 utterance × 1000 step overfit) では確認できない **実データでの学習可能性・収束性・OOM 耐性** を、既存の `train_diff.py --sub-model {1 or 4}` を **train-clean-100 全データで 1 epoch** 訓練することで検証する。これは M6.2 (4 sub-model 本格訓練、各 1M step、A100 32h) への **唯一の gate** であり、ここで OOM / divergence / conditioning 無効化が出たら M6 に進まず T-M3.x / T-M1.5 / T-M1.1 へ遡る。

T-M3.5 から昇格する確認軸:
- smoke: 「step が回る + finite + overfit する」(architectural sanity)
- **M5.2: 「実データで OOM なし完走 + validation MSE が単調減少しプラトー到達 + conditioning が実訓練後も機能」(汎化・収束性・実メモリ)**

### ゴール
- [ ] `train_diff.py --sub-model 4` (**primary**) を train-clean-100 (約 145k/T_epoch utterance) で 1 epoch、batch_size=20 で **OOM なく完走** (wall-clock: GPU 数時間)
- [ ] noise level conditioning が **1 epoch 訓練後も機能** (同 `x_t`+`mel`、異なる `c` で `eps_pred` の cosine similarity < 0.99、sub-model 4 で感度 ~70x)
- [ ] validation MSE (`evaluate()` facade、3 点 `c ∈ {L, mid, U}`) が **単調減少しプラトーに到達** (発散・停滞・NaN なし)
- [ ] TensorBoard に MSE loss / noise level histogram / lr が NaN/Inf なく記録され、`c` が band 内 (`[0, 0.4817]`、sub-model 4) に収まる
- [ ] **β 負値問題 (T-M3.3 CRITICAL) の解決確認**: reverse sampler が NaN を出さない (sub-model 2-4 を identity mock した 4-step reverse、または 4 sub-model 全部 1 epoch 訓練して full reverse)
- [ ] `docs/milestones.md` §M5.2 Acceptance 2 項目クリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `scripts/run_diff_1epoch.py` (薄い driver: `train_diff.main` を 1 epoch / sub-model 4 設定で起動 → `evaluate()` で validation)、`configs/diff_wavenext2_1epoch.yaml`
- 編集: `docs/milestones.md` §M5.2、`docs/tickets/index.md` T-M5.2 ステータス
- **新規実装コードなし** — T-M3.2 (`train_diff.py`) / T-M3.3 (`reverse_sample`) / T-M4.1 (`evaluate`) を流用するのが本チケットの主旨

### 2.2 主要構造
```python
# scripts/run_diff_1epoch.py — 薄い driver
# 1. configs/diff_wavenext2_1epoch.yaml で max_steps = ceil(N_train / batch_size) (= 1 epoch)
# 2. train_diff.main(config, sub_model_k=4) を起動 (lazy: from_config(only_sub_model=4))
# 3. 完走後 evaluate(model, val_dataset, metrics=["mrstft"]) で validation
#    (reverse sample は 4 sub-model 必要なので 1 sub-model では mock or skip、§5)
# 4. conditioning sanity (T-M3.5 test_noise_level_conditioning を 1 epoch ckpt に対し再実行)
```
- `train_diff_step` / `build_diff_model_and_optimizer` / `run_validation` (3 点 evaluation) は **T-M3.2 のものをそのまま使用**
- `c_rescale` flag は T-M3.2 config 経由で引き継ぎ (sub-model 4 で conditioning 効かなければ 1.0 → 1000.0 に切替)

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| `sub_model_idx` | **4** (primary) | T-M3.5 (sub-model 4 を smoke primary に昇格、conditioning 感度 ~70x) |
| `max_steps` | `ceil(N_train / 20)` (= 1 epoch) | docs/milestones.md §M5.2 |
| `batch_size` | 20 | docs/training.md §3.4 (FastDiff default、本番と同じ) |
| `segment_length` | 25600 | docs/training.md §3.4 (Diff hop=256) |
| Optimizer | Adam(lr=2e-4, β=[0.9,0.98], wd=0) | T-M3.2 / docs/training.md §3.4 |
| validation.c_points | `[L, mid, U]` (3 点) | T-M3.2 §6.2 (M3 phase review) |
| `c_rescale` | 1.0 (default) / 1000.0 (conditioning 効かない時) | T-M1.5 / T-M3.2 |
| amp dtype | fp32 (default) / bf16 (`--amp`) | T-M3.2 (fp16 不採用、c 精度は fp32 強制) |
| validation metric | MSE (3 点平均) + MR-STFT | T-M4.1 `evaluate()` facade |

### 2.4 アルゴリズム / 処理フロー
1. `configs/diff_wavenext2_1epoch.yaml` を読み込み (`sub_model_idx=4`、`max_steps` = 1 epoch 分)
2. `train_diff.main(config, sub_model_k=4)` を起動 (T-M3.2、lazy instantiation で sub-model 4 のみ)
3. 1 epoch 訓練ループ完走、TensorBoard に loss / lr / `c` histogram を記録
4. 10k step ごと validation (`run_validation`、3 点 `c`)、`sub_4/val/mse_at_{L,mid,U}` を監視
5. 完走後: 1 epoch ckpt をロードし conditioning sanity test (T-M3.5 流用、cos < 0.99) を再実行
6. β 負値 gate: sub-model 2-4 を identity mock した 4-step reverse (`reverse_sample`) が NaN を出さないことを確認
7. OOM 検知時: `enable_grad_ckpt=True` / batch_size 半減 → 再試行 (M6.2 申し送り)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | driver script + 1epoch config、起動・監視 | general-purpose |
| Reviewer | 1 | loss curve / conditioning / β gate 確認 + 遡及調査 (T-M3.3 / T-M1.5) | general-purpose |
| Tester | 1 | 1 epoch 起動 + TensorBoard 確認 + `evaluate()` validation | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M3.5 smoke pass が前提、T-M5.1 GAN 1 epoch とは GPU 競合のため逐次推奨)
- 最大並列数: 1 (4 sub-model 全部 1 epoch を選ぶ場合は §8 の通り 4 GPU で並列化可)

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/run_diff_1epoch.py` (薄い driver、T-M3.2 `train_diff.main` 流用)
- `configs/diff_wavenext2_1epoch.yaml` (sub_model_idx=4 primary、`c_rescale` flag)
- sub-model 4 を train-clean-100 で 1 epoch 訓練 (OOM 耐性 / 収束性 / 汎化の確認)
- `evaluate()` facade (T-M4.1) で validation 評価
- conditioning sanity の **1 epoch 後 再確認** (cos < 0.99)
- β 負値問題の gate 確認 (mock reverse で NaN なし、または 4 sub-model full reverse)
- TensorBoard で MSE loss / noise level histogram / lr 監視

### Out of Scope
- **本物の reverse sample 品質評価 (UTMOS/MCD)** — 4 sub-model 揃う必要あり、M6.2 へ (1 sub-model では mock reverse のみ、§6)
- post-filter fit / apply (T-M3.4、M6.2 で fit)
- 4 sub-model 1M step 本格訓練 (T-M6.2)
- multi-GPU / DDP (M6)
- GAN 1 epoch (T-M5.1)

### Deliverable
- ファイル: `scripts/run_diff_1epoch.py`, `configs/diff_wavenext2_1epoch.yaml`
- ドキュメント差分: `docs/milestones.md` §M5.2、`docs/tickets/index.md`

## 5. テスト項目

### 5.1 結合テスト / 検証項目
- [ ] **1 epoch を OOM なし完走** (batch_size=20、segment_length=25600)
- [ ] noise level conditioning が機能 (1 epoch ckpt で `cos(eps(c=L), eps(c=U)) < 0.99`、sub-model 4)
- [ ] validation MSE が **単調減少、プラトーに到達** (3 点 `c ∈ {L, mid, U}` の `sub_4/val/mse_*`)
- [ ] TensorBoard に MSE loss / noise level histogram / lr が NaN/Inf なく記録、`c` が band 内
- [ ] (reverse sample は 4 sub-model 必要なので、**1 sub-model では eval しない、または他 3 を mock**)

### 5.2 e2e
- [ ] `uv run python scripts/run_diff_1epoch.py` が OOM / NaN なく exit code 0
- [ ] `evaluate(model, val_dataset, metrics=["mrstft"])` が `EvalResult` を返す (T-M4.1 facade)

### 5.3 Acceptance criteria (`docs/milestones.md` §M5.2 より転記)
- [ ] OOM なしで完走
- [ ] noise level conditioning が機能 (異なる noise level で異なる出力)

## 6. 懸念事項 (M6 前の gate)

### 6.1 技術的リスク (Critical)
- **β 負値問題の解決確認 (T-M3.3 CRITICAL)**: `β[1]=1-2.8e-2/1e-4=-279` → `1/√(1-β)` で NaN 必発の問題が T-M3.3 で α_t + skip-aware σ により解決済みか、reverse sampler が NaN を出さないことを本 gate で確認 (**sub-model 2-4 を identity mock**、または 4 sub-model 全部 1 epoch なら full reverse)。NaN が出たら M6 に進まず T-M3.3 を最優先で再オープン
- **`c * 1000` rescale の要否 (T-M1.5)**: sub-model 4 (c 幅 0.48) で 1 epoch 訓練後も conditioning が効かない (cos ≥ 0.99) なら `c_rescale` を 1.0 → 1000.0 に切替えて再訓練。学習する場合 T-M1.5 NoiseEmbedding の根本問題確定 → M6.2 へ rescale 値を引き継ぐ
- **per-sub-model loss scale 不均衡 (T-M3.1)**: sub-model 1 (低ノイズ) と 4 (高ノイズ) で MSE の桁が大きく異なる懸念。TensorBoard で sub-model 別 loss 桁を確認し、M6.2 で 4 sub-model を比較する際の baseline を pin
- **bf16 で c 精度 (T-M3.2)**: `--amp` 時 sub-model 1 の `c≈0.9999` が fp16/bf16 で 1.0 に丸まり `abar=0` → x_gt 成分消失。T-M3.2 で c/abar 演算を fp32 強制済みか確認 (sub-model 4 primary では c が小さく顕在化しにくいので、sub-model 1 を 1 epoch する場合は特に注意)
- **OOM**: batch_size=20 + segment_length=25600 で 1 GPU OOM の懸念。`enable_grad_ckpt=True` (T-M1.4) を 1epoch config で有効化、それでも OOM なら batch_size 半減 → M6.2 へ申し送り

### 6.2 設計上の判断
- **lazy instantiation (T-M3.1)**: `from_config(only_sub_model=k)` で k 番目 sub-model のみ instantiate (メモリ 1/4)。本 gate で OOM 余裕を稼ぐため必須
- **GPU リソース**: A100 数時間。クラスタ未確保なら T4/spot でも可だが wall-clock 増 (ユーザー操作、§9)

### 6.3 仕様の曖昧さ
- `docs/open-questions.md`: すべて確定済み。本 gate は smoke (T-M3.5) で残した「収束性・汎化・OOM」を実測するのみ

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `train_diff.py` を **重複実装せず流用** (driver は薄い wrapper、DRY)
- [ ] Acceptance criteria 全項目クリア (§5.3)、OOM なし完走 + conditioning 機能
- [ ] validation MSE curve が単調減少・プラトー到達 (TensorBoard で確認)
- [ ] **β 負値 gate**: mock reverse (or full reverse) が NaN を出さないことを確認 (T-M3.3 解決の証跡)
- [ ] conditioning sanity が 1 epoch 後も pass (cos < 0.99)、効かない場合 `c_rescale` ablation を実施
- [ ] `evaluate()` facade (T-M4.1) を使用 (グルーコードを毎回書いていない)
- [ ] noise level histogram が band 内 (`[0, 0.4817]`、sub-model 4)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] 参考実装 (FastDiff) コピーしていない

## 8. ゼロから作り直すとしたら

### 8.1 別の設計を採るとしたら
- **1 sub-model ではなく 4 sub-model 全部 1 epoch**: full reverse sample 評価が可能になり β 負値 gate を mock なしで確認できる。GPU 4 枚あれば並列、1 枚なら wall-clock 4 倍。**M6.2 直前の最終 gate として推奨** (本チケット v1 は sub-model 4 のみ、mock reverse で β を確認するコスト最小設計)
- **sub-model 1 と 4 の両方を 1 epoch**: 学習難易度の両端 (sub-model 1 = 低ノイズ trivial、sub-model 4 = 高ノイズ最難) を確認。sub-model 1 では bf16 c 精度 (§6.1) を、sub-model 4 では conditioning 感度を検証できる
- **fixed step gate (10k step)**: 1 epoch ではなく固定 10k step で gate。wall-clock 短縮できるが「1 epoch 完走の OOM 耐性」が確認できないため不採用

### 8.2 思想 / 哲学の見直し
- **smoke (T-M3.5) と 1epoch (本チケット) の責務分離**: smoke = architectural sanity (1 utterance overfit)、1epoch = 収束性・汎化・OOM (実データ)。粒度は適切
- **M6 への gate という位置づけ**: 本 gate が pass = sub-model 4 が実データで学習する証跡。4 sub-model 統合品質 (reverse sample) は M6.2 で別評価

### 8.3 学んだこと (チケット完了後に追記)
- TBD
- 再評価トリガー: **M6.2 本格訓練前** (4 sub-model 全部 1 epoch に拡張するか、本 sub-model 4 のみで gate するかを再判断)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **T-M6.2 (Diff 4 sub-model 訓練) へ**:
  - smoke / 1 epoch を pass した config (`diff_wavenext2_1epoch.yaml`) を **1M step × 4 sub-model** に拡張
  - **`c * 1000` rescale の要否を引き継ぐ** (本 gate で確定した `c_rescale` 値をそのまま使用)
  - OOM 回避策 (`enable_grad_ckpt` / batch_size) を引き継ぐ
  - per-sub-model loss scale の baseline (sub-model 1 vs 4 の MSE 桁) を pin
  - reverse sample 品質評価 (UTMOS/MCD) は M6.2 で 4 sub-model 揃った段階で `evaluate()` facade で実施、post-filter (T-M3.4) も fit
- **ユーザー操作**: GPU リソース確保 (A100 数時間、未確保なら T4/spot でも可)
- **失敗時のフィードバック方向 / 遡及調査優先順位** (gate が pass しなかった場合、最優先順):
  1. **β 負値問題 (T-M3.3)** — reverse sampler が NaN を出す場合 (mock reverse で早期検知)
  2. **noise embedding rescale (T-M1.5)** — conditioning が効かない場合 (`c_rescale` 1.0 → 1000.0 ablation)
  3. **T-M3.1** (BAND_BOUNDS / lazy instantiation)、**T-M3.2** (MSE / noise scaling / bf16 c 精度)、**T-M1.1** (additive bias 注入)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M5.2 Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M5.2 ステータス更新 + M5 進捗サマリ
  - [ ] (該当時) `docs/training.md` に 1 epoch 訓練の wall-clock 実測値を追記

### 9.3 Open question として残ったもの
- sub-model 4 (primary) のみで gate 十分か、4 sub-model 全部 1 epoch が必要か → M6.2 直前で再判断 (§8.1)
- `c * 1000` rescale が最終的に必要か → 本 gate の conditioning 結果で確定し M6.2 へ引き継ぎ
- `docs/open-questions.md` への追記要否: gate 失敗時のみ再オープン (現時点では不要)
