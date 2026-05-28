---
id: T-M5.2
title: Diff 統合スモーク (1 sub-model train-clean-100 1 epoch)
milestone: M5
phase: M5
status: in_review
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-28
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
- [ ] `train_diff.py --sub-model 4` (**primary**) を train-clean-100 (約 145k/T_epoch utterance) で 1 epoch、batch_size=20 で **OOM なく完走** (wall-clock: GPU 数時間)。加えて β gate 用に `--sub-model 1` も 1 epoch (§6.1)
- [ ] noise level conditioning が **1 epoch 訓練後も機能** (同 `x_t`+`mel`、異なる `c` で `eps_pred` の cosine similarity が 最低線 < 0.99 / sub-model 4 期待値 < 0.5、感度 ~70x 想定、§6.3)
- [ ] validation MSE (`evaluate()` facade、3 点 `c ∈ {L, mid, U}`) が **単調減少しプラトーに到達** (発散・停滞・NaN なし)
- [ ] TensorBoard に MSE loss / noise level histogram / lr が NaN/Inf なく記録され、`c` が band 内 (`[0, 0.4817]`、sub-model 4) に収まる
- [ ] **β 負値問題 (T-M3.3 CRITICAL) の解決確認 (本 gate の核)**: reverse sampler が NaN を出さない。**sub-model 1 を 1 epoch 加え、実 eps_pred で t=1→2 の 2-step reverse を回して確認** (sub-model 2-4 を identity mock した reverse は `sigma[t]/√(1-β)` を実 eps_pred で通さず確証にならない、§6.1)
- [ ] `docs/milestones.md` §M5.2 Acceptance 2 項目クリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `scripts/run_diff_1epoch.py` (薄い driver: `train_diff.main` を 1 epoch / sub-model 4 (primary) + sub-model 1 (β gate) 設定で起動 → `evaluate()` で validation)、`configs/diff_wavenext2_1epoch.yaml`
- 編集: `docs/milestones.md` §M5.2、`docs/tickets/index.md` T-M5.2 ステータス
- **新規実装コードなし** — T-M3.2 (`train_diff.py`) / T-M3.3 (`reverse_sample`) / T-M4.1 (`evaluate`) を流用するのが本チケットの主旨

### 2.2 主要構造
```python
# scripts/run_diff_1epoch.py — 薄い driver
# 1. configs/diff_wavenext2_1epoch.yaml で max_steps = ceil(N_train / batch_size) (= 1 epoch)
# 2. train_diff.main(config, sub_model_k=4) を起動 (lazy: from_config(only_sub_model=4))
#    + sub-model 1 も 1 epoch (β gate 用、sub-model 別 namespace + state file で逐次管理、§6.2)
# 3. 完走後 evaluate(model, val_dataset, metrics=["mrstft"]) で validation
#    (full reverse は 4 sub-model 必要なので M6.2 へ。本 gate は sub-model 1 の実 eps_pred t=1→2 2-step reverse で β を確認、§5)
# 4. conditioning sanity (T-M3.5 test_noise_level_conditioning を 1 epoch ckpt に対し再実行)
```
- `train_diff_step` / `build_diff_model_and_optimizer` / `run_validation` (3 点 evaluation) は **T-M3.2 のものをそのまま使用**
- `c_rescale` flag は T-M3.2 config 経由で引き継ぎ (sub-model 4 で conditioning 効かなければ 1.0 → 1000.0 に切替)

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| `sub_model_idx` | **4** (primary、conditioning 感度) + **1** (β gate 用、実 eps_pred 2-step reverse) | T-M3.5 / §6.1 (β 負値 NaN は sub-model 1→2 遷移で発生) |
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
5. 完走後: 1 epoch ckpt をロードし conditioning sanity test (T-M3.5 流用、最低線 cos<0.99 / sub-model 4 期待値 cos<0.5) を再実行
6. β 負値 gate: **sub-model 1 を 1 epoch 加え、実 eps_pred で t=1→2 の 2-step reverse** (`reverse_sample`) が NaN を出さないことを確認 (mock では `sigma[t]/√(1-β)` の数値経路を実 eps_pred で通さず確証にならない、§6.1)
7. OOM 検知時: `enable_grad_ckpt=True` / batch_size 半減 → 再試行 (M6.2 申し送り)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | driver script + 1epoch config、起動・監視 | general-purpose |
| Reviewer | 1 | loss curve / conditioning / β gate 確認 + 遡及調査 (T-M3.3 / T-M1.5) | general-purpose |
| Tester | 1 | 1 epoch 起動 + TensorBoard 確認 + `evaluate()` validation | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M3.5 smoke pass が前提、T-M5.1 GAN 1 epoch とは GPU 競合のため逐次推奨)
- 最大並列数: 1 (v1 は sub-model 4 + sub-model 1 を single GPU で逐次実行、§6.2 の通り sub-model 別 namespace + state file で再開管理。4 sub-model 全部 1 epoch を選ぶ場合は §8.1 の通り 4 GPU で並列化可)

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/run_diff_1epoch.py` (薄い driver、T-M3.2 `train_diff.main` 流用)
- `configs/diff_wavenext2_1epoch.yaml` (sub_model_idx=4 primary + sub_model_idx=1 (β gate 用)、`c_rescale` flag)
- sub-model 4 を train-clean-100 で 1 epoch 訓練 (OOM 耐性 / 収束性 / 汎化の確認) + sub-model 1 を 1 epoch (β gate 用)
- `evaluate()` facade (T-M4.1) で validation 評価
- conditioning sanity の **1 epoch 後 再確認** (最低線 cos<0.99 / sub-model 4 期待値 cos<0.5)
- β 負値問題の gate 確認 (**実 eps_pred の t=1→2 2-step reverse で NaN なし**、sub-model 1 を使用)
- TensorBoard で MSE loss / noise level histogram / lr 監視

### Out of Scope
- **full reverse sample 品質評価 (UTMOS/MCD)** — 4 sub-model 揃う必要あり、M6.2 へ (本 gate は β 用 2-step reverse のみ、§6)
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
- [ ] **β 負値 gate (最優先)**: 実 eps_pred の 2-step reverse (t=1→2、sub-model 1 を 1 epoch して使用) が **NaN を出さない** (mock では確証にならない、§6.1)
- [ ] noise level conditioning **最低線 gate**: 1 epoch ckpt で `cos(eps(c=L), eps(c=U)) < 0.99` (conditioning が死んでいない)
- [ ] noise level conditioning **期待値 gate (sub-model 4、§6.3)**: `cos(eps(c=L), eps(c=U)) < 0.5` (感度 ~70x 想定の期待値)
- [ ] validation MSE が **単調減少、プラトーに到達** (3 点 `c ∈ {L, mid, U}` の `sub_4/val/mse_*`、直近 N step の傾き < ε で自動判定、§6.3)
- [ ] TensorBoard に MSE loss / noise level histogram / lr が NaN/Inf なく記録、`c` が band 内
- [ ] (full reverse sample 品質は 4 sub-model 必要なので M6.2 へ。本 gate は β 用 2-step reverse のみ)

### 5.2 e2e
- [ ] `uv run python scripts/run_diff_1epoch.py` が OOM / NaN なく exit code 0
- [ ] `evaluate(model, val_dataset, metrics=["mrstft"])` が `EvalResult` を返す (T-M4.1 facade)

### 5.3 Acceptance criteria (`docs/milestones.md` §M5.2 より転記)
- [ ] OOM なしで完走
- [ ] noise level conditioning が機能 (異なる noise level で異なる出力)

## 6. 懸念事項 (M6 前の gate)

### 6.1 技術的リスク (Critical)
- **β 負値問題の解決確認 (T-M3.3 CRITICAL、本 gate の核)**: `β[1]=1-2.8e-2/1e-4=-279` → `1/√(1-β)` で NaN 必発の問題が T-M3.3 で α_t + skip-aware σ により解決済みか、reverse sampler が NaN を出さないことを本 gate で確認。**ただし β 負値の NaN は sub-model 1→2 遷移 (high noise) で起きるため、sub-model 2-4 を identity mock した reverse は `sigma[t]/√(1-β)` の数値経路を実 eps_pred で通さず、NaN なしの確証にならない**。最小でも **sub-model 1 を 1 epoch 加え、実 eps_pred で t=1→2 の 2-step reverse を回す** (§8.1 で v1 推奨に昇格)。NaN が出たら M6 に進まず T-M3.3 を最優先で再オープン
- **gate 判定の優先順序**: ① reverse NaN なし (β 負値解決の確認、NaN 必発の最優先 blocker) → ② conditioning 生存 (cos が死んでいない) → ③ MSE 収束。**β 負値は収束性より先に潰すべき blocker** であり、収束していても NaN が出れば即 NO-GO
- **bf16 fp32 強制が gate 前提 (T-M3.2、格上げ)**: `--amp` 時 sub-model 1 の `c≈0.9999` が fp16/bf16 で 1.0 に丸まり `abar≈0` → x_gt 成分消失。**c/abar 演算の fp32 強制 (T-M3.2) が未済なら、bf16 で sub-model 1 を 1 epoch すること自体が無意味** (上記 β gate のため sub-model 1 を回す前提では特に致命的)。よって fp32 強制の有無は本 gate の前提条件に格上げ。sub-model 4 primary では c が小さく顕在化しにくいが、sub-model 1 を加えるなら必須確認
- **`c * 1000` rescale の要否 (T-M1.5)**: sub-model 4 (c 幅 0.48) で 1 epoch 訓練後も conditioning が効かない (cos ≥ 0.99) なら `c_rescale` を 1.0 → 1000.0 に切替えて再訓練。学習する場合 T-M1.5 NoiseEmbedding の根本問題確定 → M6.2 へ rescale 値を引き継ぐ
- **per-sub-model loss scale は桁差でなく学習曲線単調性で判定 (T-M3.1)**: sub-model 1 (`c≈1`、`ε≈x_t`) と 4 (`c≈0.3`) で MSE が一桁違うのは **正常**。「桁差そのもの」を異常とせず、**各 sub-model の学習曲線の単調性 (単調減少・プラトー到達)** で baseline を pin する。M6.2 で 4 sub-model を比較する際もこの単調性基準を引き継ぐ
- **OOM**: batch_size=20 + segment_length=25600 で 1 GPU OOM の懸念。`enable_grad_ckpt=True` (T-M1.4) を 1epoch config で有効化、それでも OOM なら batch_size 半減 → M6.2 へ申し送り
- **resume 完全性 (T-M5.1 と共通)**: 途中再開時に Adam state / step counter / RNG state が完全復元されること。これが破れると 1 epoch 完走が偽の連続性になる。checkpoint に optimizer/step/RNG を含め、resume で復元することを確認

### 6.2 設計上の判断
- **lazy instantiation (T-M3.1)**: `from_config(only_sub_model=k)` で k 番目 sub-model のみ instantiate (メモリ 1/4)。本 gate で OOM 余裕を稼ぐため必須
- **sub-model ごと別 log dir / checkpoint namespace + state file 再開管理**: single GPU 上で sub-model を逐次実行する (sub-model 4 + sub-model 1 等) 場合、TensorBoard ログと checkpoint を sub-model 別の namespace (`logs/diff_1epoch/sub_{k}/`, `checkpoints/diff_1epoch/sub_{k}/`) に分離し、どの sub-model まで完了したかを state file で管理。途中再開時にどの sub-model から再開するかを明示できる
- **wall-clock budget gate + checkpoint 間隔短縮 (T-M5.1 と共通)**: 1 epoch 完走に固執せず wall-clock budget (例: 数時間) を上限に設定し、超過時は budget 内で gate 判定。途中 OOM/divergence からの素早い resume のため checkpoint 間隔を短縮 (10k → 2k step 等)
- **GPU リソース**: A100 数時間。クラスタ未確保なら T4/spot でも可だが wall-clock 増 (ユーザー操作、§9)

### 6.3 gate 判定の追加観点
- **cos<0.99 は緩すぎ、別 gate に分離**: sub-model 4 は conditioning 感度 ~70x 想定なので、**`cos(eps(c=L), eps(c=U)) < 0.5` を期待値**とする。`cos < 0.99` は「conditioning が死んでいない」最低線にすぎないため、§5.1 の最低線 gate (cos<0.99) と期待値 gate (cos<0.5、sub-model 4) を分離して判定
- **wall-clock 実レンジの実測**: `ceil(N_train/20)` step が batch=20 × segment=25600 で A100 / T4 でそれぞれ何時間かを実測。M6.2 (各 sub-model 1M step、A100 32h) との比例から GPU 確保判断の材料にする
- **TensorBoard ログ保管・提示 (T-M5.1 と共通)**: loss curve / noise level histogram / lr のログを保管し、gate 判定者 (user) に提示できる形で残す
- **プラトー自動検知**: 「MSE 単調減少・プラトー到達」を人間が曲線を見る定性判定に委ねず、**直近 N step の傾き < ε で自動判定** (`run_validation` 内で slope を計算しログ)。gate 判定の再現性を高める

### 6.4 仕様の曖昧さ
- `docs/open-questions.md`: すべて確定済み。本 gate は smoke (T-M3.5) で残した「収束性・汎化・OOM」を実測するのみ

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `train_diff.py` を **重複実装せず流用** (driver は薄い wrapper、DRY)
- [ ] Acceptance criteria 全項目クリア (§5.3)、OOM なし完走 + conditioning 機能
- [ ] **gate 判定順序の遵守 (§6.1)**: ① reverse NaN なし → ② conditioning 生存 → ③ MSE 収束 の順で判定し、NaN が出れば収束していても NO-GO
- [ ] **β 負値 gate**: **実 eps_pred の 2-step reverse (t=1→2、sub-model 1 を 1 epoch して使用) が NaN を出さない** ことを確認 (mock では確証にならない、T-M3.3 解決の証跡)
- [ ] **bf16 fp32 強制が確認済み (§6.1 前提)**: sub-model 1 を回す場合、c/abar 演算が fp32 強制されている (未済なら 1 epoch 自体が無意味)
- [ ] conditioning sanity が 1 epoch 後も pass: 最低線 `cos < 0.99`、sub-model 4 期待値 `cos < 0.5` (§6.3)。効かない場合 `c_rescale` ablation を実施
- [ ] validation MSE curve が単調減少・プラトー到達 (直近 N step の傾き < ε で自動判定、TensorBoard で確認、§6.3)
- [ ] per-sub-model loss scale は桁差でなく学習曲線の単調性で判定 (§6.1)
- [ ] resume 完全性: Adam state / step / RNG 復元が保証されている (§6.1)
- [ ] `evaluate()` facade (T-M4.1) を使用 (グルーコードを毎回書いていない)
- [ ] noise level histogram が band 内 (`[0, 0.4817]`、sub-model 4)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] 参考実装 (FastDiff) コピーしていない

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M5 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **sub-model 4 + sub-model 1 を 1 epoch (v1 推奨)** | sub-model 1 を加えることで **実 eps_pred の 2-step reverse (t=1→2)** が回せ、β 負値 gate を実数値経路で確認できる (mock では確証にならない、§6.1)。sub-model 1 で bf16 c 精度、sub-model 4 で conditioning 感度も両端検証 | sub-model 4 単体より wall-clock 2 倍、single GPU では逐次 | — (**採用**。β 負値が NaN 必発の最優先 blocker であり、最小でも sub-model 1 を加える必要がある) | sub-model 1 でも reverse NaN なし・conditioning 機能が確認できれば、M6.2 直前で 4 sub-model full に拡張するか再判断 |
| **sub-model 4 のみ 1 epoch + mock reverse** | wall-clock 最小、conditioning 感度が最大の sub-model に集中 | **mock (2-4 を identity) は `sigma[t]/√(1-β)` を実 eps_pred で通さず β gate の確証にならない** (§6.1)。sub-model 1→2 遷移の high-noise NaN を検出できない | β gate の核を満たせないため v1 主案から降格 (mock reverse のコスト最小設計は β 確認を犠牲にする) | wall-clock budget が逼迫し、β 解決が別経路 (T-M3.3 unit test 等) で確認済みなら mock のみへ縮退 |
| **4 sub-model 全部 1 epoch** | full reverse sample 評価が可能、β gate を mock なしで完全確認、reverse 品質も測れる | GPU 4 枚あれば並列、1 枚なら wall-clock 4 倍。config が 1 sub-model 形状から逸脱 (§8.2) | M5 は M6.2 への gate であり 4 sub-model 統合品質は M6.2 で別評価。コスト過大 | **M6.2 直前の最終 gate**として推奨。GPU 4 枚確保できる、または β/conditioning に強い懸念が残るとき |
| **fixed step gate (10k step)** | wall-clock 短縮、gate 基準が epoch 長に依存せず安定 | 「1 epoch 完走の OOM 耐性」が確認できない | milestones.md が「1 epoch」と明記、OOM 耐性が gate の一要素 | `ceil(N_train/20)` step の wall-clock が過大なら 10k step gate へ切替 (§6.3 で実測) |

#### 採用設計
- **sub-model 4 (conditioning 感度 primary) + sub-model 1 (β gate 用) を 1 epoch** — β 負値 NaN は sub-model 1→2 遷移で起きるため、実 eps_pred の 2-step reverse で確認するには sub-model 1 が必須 (mock では確証にならない)
- **既存 `train_diff.py` を 1epoch config で起動するだけ** (新規 train コードゼロ、最小コストの gate、DRY)
- **`evaluate()` facade 1 entry point 評価** (グルーコードを書かない、T-M4.1 の設計意図に乗る)
- **gate 判定順序**: ① reverse NaN なし → ② conditioning 生存 → ③ MSE 収束 (§6.1)

#### 再評価トリガー
- **M6.2 本格訓練前**: sub-model 4+1 のみで gate 十分か、4 sub-model 全部 1 epoch (full reverse) に拡張するかを再判断 (§9.3)。`ceil(N_train/20)` step の wall-clock が過大なら fixed 10k step gate へ切替

### 8.2 思想 / 哲学の見直し
- **gate 判定者 = user の GO/NO-GO (T-M5.1 と整合)**: M6.2 は A100 32h の課金。**「mock reverse で β gate を通すコスト最小設計の妥当性」は user に GPU 予算とセットで提示して決める**。本 gate が技術的に pass しても、4 sub-model full まで回すか sub-model 4+1 で止めるかは予算依存の判断であり user の承認事項
- **config-shape mismatch の責務境界 (重要)**: §9.1 は `diff_wavenext2_1epoch.yaml` を「1M step × 4 sub-model に拡張」と書くが、**1epoch config は `sub_model_idx=4` 単体・lazy instantiation 前提**であり 4 sub-model 一括 config への変換は暗黙。**「本 gate で検証した config は 1 sub-model 形状であり、M6.2 の 4 sub-model config は別物 = 4 sub-model 統合は M6.2 で初検証」**と明記する。1epoch config がそのまま M6.2 に通る保証はない
- **gate 順序哲学**: smoke (T-M3.5) = architectural sanity (1 utterance overfit) / 1epoch (本チケット) = 収束性・汎化・OOM (実データ) の分離は妥当。ただし本 gate の核は MSE 収束ではなく **β 負値 (NaN 必発) の解決確認** であり、収束していても reverse が NaN を出せば NO-GO (§6.1 gate 順序)
- **smoke (T-M3.5) と 1epoch (本チケット) の責務分離**: smoke = architectural sanity (1 utterance overfit)、1epoch = 収束性・汎化・OOM (実データ)。粒度は適切
- **M6 への gate という位置づけ**: 本 gate が pass = sub-model 4 (+1) が実データで学習し reverse が NaN を出さない証跡。4 sub-model 統合品質 (full reverse sample) は M6.2 で別評価

### 8.3 学んだこと (2026-05-28 足場実装、実行は user-gated)

- **T-M5.1 と共通 orchestrator で DRY 実装**: `eval/gate.py::diff_divergence_gate(loss_mse, output) -> GateResult` (loss 単調減少傾向・波形 finite/[-1,1]/非無音・NaN なし) + `scripts/run_smoke.py --mode diff` + `configs/diff_wavenext2_1epoch.yaml` + `scripts/eval_diff_checkpoint.py`。GAN/Diff の gate は閾値 (Diff は loss_adv/loss_d なし) のみ差分、orchestrator は `--mode` で分岐。
- **実 1 epoch + conditioning 検証は GPU + データ必須**: noise level conditioning (cos<0.99)・β-free reverse の実 NaN 確認・c_rescale ablation はいずれも GPU smoke (T-M3.5 で実装済の slow+gpu テスト) + 1 epoch 実訓練で決着する項目で、user 実行環境に委ねる。
- 詳細な学びは T-M5.1 §8.3 と共通 (M5 = M6/M7 同様のユーザー操作必須境界、divergence gate の純関数化)。
- **β 負値 gate 判断**: TBD (実 eps_pred 2-step reverse で NaN 有無を確認、T-M3.5 reverse-mock で構造的に NaN なしは確認済)
- **`c * 1000` rescale 判断**: TBD (sub-model 4 conditioning が cos<0.99/<0.5 を満たすかで確定)
- **bf16 fp32 強制判断**: TBD (sub-model 1 の c 精度が保たれるかで確定)
- **wall-clock 実測**: TBD (`ceil(N_train/20)` step の A100 / T4 実時間、M6.2 32h との比例)
- 想定外: TBD
- 教訓: TBD
- 再評価トリガー: **M6.2 本格訓練前** (4 sub-model 全部 1 epoch に拡張するか、本 sub-model 4+1 のみで gate するかを再判断)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M6.2 (Diff 4 sub-model 訓練) へ
- **divergence gate pass**: 本 gate (β 負値 NaN なし + conditioning 生存 + MSE 収束) が pass したことが M6.2 起動の前提
- **config-shape mismatch (重要、§8.2)**: 本 gate で検証した `diff_wavenext2_1epoch.yaml` は **`sub_model_idx=4` (+1) 単体・lazy instantiation 形状**。M6.2 の **4 sub-model 一括 config は別物** であり、1epoch config から「1M step × 4 sub-model」への変換は **M6.2 で初統合・初検証**となる (1epoch config がそのまま通る保証はない)
- **`c * 1000` rescale の要否を引き継ぐ** (本 gate で確定した `c_rescale` 値をそのまま使用、効かなければ 1.0 → 1000.0)
- **resume 完全性 (§6.1)**: Adam state / step counter / RNG state の復元保証を M6.2 (各 1M step、長時間) でも維持
- OOM 回避策 (`enable_grad_ckpt` / batch_size) を引き継ぐ
- per-sub-model loss scale の baseline (sub-model 1 vs 4 の MSE は桁差でなく **学習曲線の単調性**で pin、§6.1)
- reverse sample 品質評価 (UTMOS/MCD) は M6.2 で 4 sub-model 揃った段階で `evaluate()` facade で実施、post-filter (T-M3.4) も fit

#### T-M5.1 と共通の申し送り
- **`run_smoke.py` orchestrator**: GAN/Diff の 1 epoch gate を共通 driver から起動する場合、T-M5.1 と orchestrator を共有 (起動・監視・evaluate 呼び出しの共通化)
- **`EvalResult.gate_passed`**: gate 判定結果 (NaN/収束/conditioning) を `EvalResult` に `gate_passed` フラグとして載せ、T-M5.1 と同じ判定 API で扱う

#### ユーザー操作 (必須)
- **GPU リソース確保 + gate GO/NO-GO 承認**: A100 数時間 (未確保なら T4/spot でも可、wall-clock 増)。本 gate pass 後に M6.2 (32h 課金) へ進むかは user の GO/NO-GO 承認事項 (§8.2、GPU 予算とセットで判断)

#### 失敗時のフィードバック方向 / 遡及調査優先順位 (gate が pass しなかった場合、最優先順)
1. **β 負値問題 (T-M3.3)** — reverse sampler が NaN を出す場合。**mock ではなく実 eps_pred の 2-step reverse (t=1→2) で確認**したうえで遡及 (mock は確証にならない、§6.1)
2. **bf16 c 精度の fp32 強制 (T-M3.2)** — sub-model 1 の `c≈0.9999→abar≈0` 丸めが起きる場合、c/abar 演算の fp32 強制を確認
3. **noise embedding rescale (T-M1.5)** — conditioning が効かない場合 (`c_rescale` 1.0 → 1000.0 ablation)
4. **T-M3.1** (BAND_BOUNDS / lazy instantiation)
5. **T-M1.4** (enable_grad_ckpt、OOM 時) / **T-M1.1** (additive bias 注入)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M5.2 Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M5.2 ステータス更新 + M5 進捗サマリ
  - [ ] (該当時) `docs/training.md` に 1 epoch 訓練の wall-clock 実測値を追記

### 9.3 Open question として残ったもの
- sub-model 4+1 (v1) で gate 十分か、4 sub-model 全部 1 epoch (full reverse) が必要か → M6.2 直前で再判断 (§8.1)
- `c * 1000` rescale が最終的に必要か → 本 gate の conditioning 結果で確定し M6.2 へ引き継ぎ
- 1epoch config (1 sub-model 形状) → M6.2 の 4 sub-model 一括 config への変換が問題なく通るか → M6.2 で初統合・初検証 (§8.2)
- `docs/open-questions.md` への追記要否: gate 失敗時のみ再オープン (現時点では不要)
