---
id: T-M6.3
title: Ablation 比較 (T=2/3/4/5、post-filter、sub-modeling) — 任意
milestone: M6
phase: M6
status: pending
size: L
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M6.1, T-M6.2]
blocks: []
related_docs:
  - docs/milestones.md#m63-ablation-任意
  - docs/paper-summary.md
---

# T-M6.3: Ablation 比較 (T=2/3/4/5、post-filter、sub-modeling) — 任意

> **マイルストーン**: [M6](../milestones.md#m6-本格訓練-claude-code-は起動監視のみ-wall-clock-a100-で約-442-時間) / **サブタスク**: [M6.3](../milestones.md#m63-ablation-任意)
> **依存**: [T-M6.1](T-M6.1-gan-full-training.md), [T-M6.2](T-M6.2-diff-full-training.md) / **後続**: [T-M7.1](T-M7.1-mos-test.md) (best config を MOS へ)
>
> **⚠️ 任意タスク**: 計算コストが大きく (GAN フル訓練 × 4 + Diff single model 追加)、優先度は低い。GPU 予算が確定した場合のみ着手。予算次第でスキップ可。

## 1. タスク目的とゴール

### 目的
論文 Table 1〜3 の主要な ablation trend を再現し、設計選択 (GAN の iteration 数 T、Diff の post-filter、Diff の sub-modeling) が品質・速度に与える影響を**定量比較表 + プロット**で示す。各 config を T-M4.1 の `evaluate()` facade で同一指標で評価し、論文との対比を Markdown レポートとして自動生成する。

### ゴール
完了したと判断できる具体的な状態:
- [ ] Ablation matrix (GAN: T=2/3/4/5、Diff: w/wo post-filter、Diff: w/wo sub-modeling) を生成し、各 config で訓練を起動・完走
- [ ] 全 config を `evaluate()` facade で同一指標 (UTMOS / NISQA / MCD / log F0 RMSE / RTF) で評価し、比較表を自動生成
- [ ] 比較表が論文 Table 1〜3 の **定性 trend** を再現 (T 増加で品質向上・RTF 悪化、post-filter で高域改善、sub-modeling で品質向上)
- [ ] matplotlib プロット (T vs UTMOS, T vs RTF 等) を生成
- [ ] 論文との対比 Markdown レポートを `eval_results/` (or `docs/`) に出力
- [ ] `docs/milestones.md` §M6.3 Deliverable、`docs/tickets/index.md` の T-M6.3 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `scripts/run_ablation.py` (ablation matrix 生成 → 各 config 訓練起動 → `evaluate()` 評価 → 比較表 + プロット + レポート生成のオーケストレータ。薄い driver、訓練/評価ロジックは再利用)
  - `configs/ablation/` (matrix 各セルの config: `gan_T2.yaml` 〜 `gan_T5.yaml`, `diff_single.yaml` 等。既存 `gan_wavenext2.yaml` / `diff_wavenext2.yaml` からの差分 override)
- 編集:
  - `docs/milestones.md` §M6.3、`docs/tickets/index.md` T-M6.3 ステータス + M6 フェーズレビューログ

> **重要 (DRY)**: 訓練は T-M6.1 / T-M6.2 と同じ `train_gan.py` / `train_diff.py` を config 差分で起動。評価は T-M4.1 `evaluate()` を呼ぶだけ。本チケットの新規コードは matrix 生成・config override・比較表/プロット/レポート集約に限定する。

### 2.2 主要構造

#### Ablation matrix と評価 (既存 facade 再利用)
```python
# scripts/run_ablation.py (薄い orchestrator、Windows spawn 対策で if __name__ ガード下)
from wavenext2.eval.runner import evaluate, EvalResult

# matrix 定義 (config override の集合)
GAN_ABLATION   = {"T=2": "configs/ablation/gan_T2.yaml", ..., "T=5": "configs/ablation/gan_T5.yaml"}
DIFF_ABLATION  = {
    "w/ post-filter":   ("checkpoints/diff/", "fir.npy"),   # post_filter=fir
    "wo post-filter":   ("checkpoints/diff/", None),        # post_filter=None
    "w/ sub-modeling":  ("checkpoints/diff/", ...),         # 4 point-specialized sub-model
    "wo sub-modeling":  ("checkpoints/diff_single/", ...),  # single model を別途訓練
}

# 各セルを同一 dataset・同一指標で評価 (グルーコード不要)
results: dict[str, EvalResult] = {}
for name, cfg in {**GAN_ABLATION, **DIFF_ABLATION}.items():
    model = load_model(cfg)
    results[name] = evaluate(model, test_ds,
                             metrics=["utmos", "nisqa", "mcd", "log_f0_rmse", "rtf"],
                             post_filter=fir_or_none,
                             save_to=f"eval_results/ablation_{name}.json")

build_comparison_table(results)   # → Markdown 比較表
plot_ablation(results)            # → T vs UTMOS / T vs RTF 等の PNG
write_paper_comparison(results)   # → 論文 Table 1〜3 との対比 Markdown
```

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| GAN ablation T | 2, 3, 4, 5 (iteration / sub-model 直列数) | docs/milestones.md §M6.3 / paper-summary §5 |
| Diff post-filter | with (`fir.npy`) / without (`None`) | T-M3.4 / paper-summary §3 |
| Diff sub-modeling | with (4 point-specialized) / without (single model) | T-M3.1 / paper-summary §3 |
| 評価指標 | UTMOS, NISQA, MCD, log F0 RMSE, RTF | T-M4.1 §9.1 / paper-summary §4 |
| 評価データ | LibriTTS-R test-clean 4,824 utt | paper-summary §4 |
| 部分訓練 step (コスト削減時) | 500k step で打ち切り (§6) | 本チケットで確定 |
| 論文 RTF 参照値 | GAN T=5: GPU 0.0066 / CPU 0.20、Diff w/ sub: GPU 0.0282 / CPU 0.16 | paper-summary §5 表 |
| 論文パラメータ参照値 | GAN T=5: 74.93M、Diff w/ sub: 57.68M | paper-summary §5 |

### 2.4 アルゴリズム / 処理フロー
1. `scripts/run_ablation.py` が matrix を展開し、各セルの config を `configs/ablation/` に生成 (既存 config からの override)
2. 各 GAN config (T=2〜5) を `train_gan.py` で `run_in_background` 起動。フル訓練が非現実的なら **fixed 500k step で打ち切り** (§6)
3. Diff sub-modeling なし (single model) を `train_diff.py` で別途訓練 (約 32h 追加、§6)
4. post-filter は **訓練不要**: 学習済み Diff の `reverse_sample(post_filter=fir / None)` を呼び分けるだけ (T-M3.4)
5. 全 config を `evaluate()` facade で同一 metrics 評価、結果を `eval_results/ablation_*.json` に永続化
6. 比較表 (Markdown) を組み立て、論文 Table 1〜3 の参照値を併記
7. matplotlib で trend プロット (T vs UTMOS, T vs RTF, T vs param 数 等) を PNG 出力
8. 論文 trend との一致を **定性判定** (順序関係) でレポートに記述

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `run_ablation.py` + `configs/ablation/*.yaml` (matrix 生成・比較表・プロット・レポート) | general-purpose |
| Operator/Monitor | 1 | 各 config 訓練の background 起動・監視・OOM/divergence 時 resume | general-purpose |
| Reviewer | 1 | 論文 Table 1〜3 trend との対比検証、best config 選定、T-M7.1 申し送り | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M6.1 / T-M6.2 完了が前提。学習済み checkpoint がないと開始できない)
- 並列実行する場合の最大並列数: matrix 各セルの訓練は GPU が複数あれば並列可 (例 4 セル → 4 GPU)。単一 GPU なら順次
- 任意タスクのため、予算が無ければスキップ

## 4. 提供範囲 (Scope)

### In Scope
- Ablation matrix の生成 (GAN T=2/3/4/5、Diff w/wo post-filter、Diff w/wo sub-modeling)
- 各 config の訓練起動・監視 (既存 `train_gan.py` / `train_diff.py` を config 差分で起動)
- sub-modeling なし (single model) の Diff を別途訓練 (32h 追加)
- 全 config を `evaluate()` facade で同一指標評価
- 比較表 (Markdown) + matplotlib プロット (T vs UTMOS / RTF / param 数) の自動生成
- 論文 Table 1〜3 trend との対比レポート (定性判定主軸)

### Out of Scope
- **新規 train / model / 指標ロジック** (すべて T-M2.x / T-M3.x / T-M4.x で完成済、本チケットは起動・評価のみ)
- **論文との定量完全一致** (backend / 環境差で絶対値はずれる、§6。定性 trend 再現で十分)
- **主観評価 (MOS)** → T-M7.1 (本チケットは best config を選定して申し送るのみ)
- **hyperparameter sweep / Optuna / W&B** (§8 で代替案、本チケットは離散 matrix のみ)
- **論文に無い ablation 軸** (segment_length, eps, noise schedule 等) → 別途必要なら §8

### Deliverable
- ファイル:
  - `scripts/run_ablation.py` (新規、orchestrator)
  - `configs/ablation/*.yaml` (新規、matrix 各セル)
  - `eval_results/ablation_*.json` (評価結果、`.gitignore` 済)
  - `eval_results/ablation_comparison.md` (比較表 + 論文対比レポート)
  - `eval_results/ablation_*.png` (trend プロット)
- 関数 / クラス: なし (既存 `train_*.main` / `evaluate` を再利用、本スクリプトは集約のみ)
- ドキュメント差分: `docs/milestones.md` §M6.3 / `docs/tickets/index.md` 更新

## 5. テスト項目

### 5.1 Unit テスト
- [ ] `configs/ablation/*.yaml` が `load_config` でパース可能、各 GAN config の T が 2/3/4/5 に正しく設定される
- [ ] `run_ablation.py` が `if __name__ == "__main__"` ガード下で `evaluate()` を呼ぶ (Windows spawn 安全)
- [ ] 比較表生成関数が dummy `EvalResult` の dict から Markdown テーブルを正しく整形 (行=config、列=指標)
- [ ] プロット関数が dummy データで PNG を出力 (matplotlib backend 非対話)

### 5.2 e2e / 結合テスト (GPU 必須・`@pytest.mark.slow @pytest.mark.gpu`)
- [ ] 各 ablation config が **完走** (OOM / divergence なし、fixed step 打ち切り含む)
- [ ] 全 config の `evaluate()` が finite な指標を返し、`eval_results/ablation_*.json` に永続化
- [ ] 比較表 Markdown + プロット PNG + 対比レポートが生成される

### 5.3 Acceptance criteria (`docs/milestones.md` §M6.3 より転記)
- [ ] 比較表 + プロット (Deliverable) が生成される
- [ ] T=2,3,4,5 (GAN)、w/wo post-filter (Diff)、w/wo sub-modeling (Diff) を網羅
- [ ] 論文 Table 1〜3 の trend を再現

### 5.4 追加 acceptance (本チケット独自)
- [ ] T 増加で品質 (UTMOS/MCD) 向上・RTF 悪化の trend が比較表で確認できる
- [ ] post-filter ありで高域改善 (NISQA or スペクトル) が確認できる
- [ ] sub-modeling ありが single model より品質が高い trend が確認できる
- [ ] best config が選定され T-M7.1 へ申し送られる

## 6. 懸念事項

### 6.1 技術的リスク
- **【最大】計算コスト**: GAN T=2/3/4/5 をフル訓練 (各 410h) = 410h × 4 ≈ **非現実的**。
  - **緩和**: fixed step (500k step) で打ち切り、または部分訓練で trend のみ確認。500k step でも T 間の**相対**順序 (品質・RTF の trend) は出る見込み
  - **検知**: wall-clock 予算超過。予算が尽きたら matrix を間引く (T=2/4 のみ等) か本チケット自体をスキップ
- **sub-modeling なし (single model) の追加訓練**: Diff の単一モデルを別途訓練 = 約 **32h 追加**。GPU 予算に含める
- **論文との定量一致は期待しない**: MCD/UTMOS/RTF は backend・GPU・MFCC order・DTW mode で系統差 (T-M4.1 §8.2)。RTF も A100 世代・CPU (AMD EPYC 7542) 差で絶対値がずれる → **定性 trend (順序関係) の再現で合格**とする
- **post-filter は訓練不要だが評価のみ**: `reverse_sample(post_filter=fir / None)` の呼び分けで apply/no-apply を出す (T-M3.4)。新規訓練コスト 0

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 関連: すべて確定済み (Diff partition は point-specialized、§architecture.md §5)
- 本チケット固有の判断:
  - **論文 trend の定性再現で十分か、定量一致まで求めるか** → **定性再現を採用** (§6.1、§8.2)。絶対値は参考程度
  - **fixed step 値** → 500k を初期値とし、trend が出なければ延長 (§8)

### 6.3 他チケットとの整合性
- **T-M6.1 / T-M6.2 (本格訓練)**: 学習済み checkpoint (`checkpoints/gan/`, `checkpoints/diff/`) と config を流用。T=4 (GAN) / w/ sub-modeling (Diff) は M6.1/M6.2 の成果物をそのまま比較対象に使える (再訓練不要)
- **T-M4.1 (`evaluate()` facade)**: 全 config を 1 entry point で同一指標評価。`eval_results/*.json` 永続化・相対比較主軸の方針に従う
- **T-M3.4 (post-filter)**: `reverse_sample(post_filter=...)` の switch で apply/no-apply を比較 (chain しない)
- **T-M7.1 (MOS)**: 本チケットで選定した best config を主観評価対象に申し送る (§9)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 訓練/評価ロジックを新規実装せず既存 `train_*.py` / `evaluate()` を起動・呼び出している (DRY)
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)
- [ ] Ablation matrix (GAN T=2〜5、Diff post-filter、Diff sub-modeling) を網羅
- [ ] 比較表が論文 Table 1〜3 の **定性 trend** を再現 (T↑で品質↑・RTF↑、post-filter で高域改善、sub-modeling で品質↑)
- [ ] fixed step 打ち切りの場合、その旨が比較表・レポートに明記され absolute 値は参考扱い
- [ ] プロット (T vs UTMOS / RTF 等) が生成され読み取れる
- [ ] best config が選定され T-M7.1 へ申し送られている
- [ ] 参考実装をコピーしていない (matrix 生成・集約のみで新規ロジックなし)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M6 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **ablation を最初から matrix 設計し M6.1/M6.2 と統合** | 重複訓練を排除 (T=4 / w-sub は本訓練と共有)、起動を 1 系統に | M6.1/M6.2 の責務が膨らむ | M6.1/M6.2 を独立 gate に保つため分離。本チケットは差分セルのみ追加訓練 | 予算確定時に matrix 全体を一括設計し直す |
| **部分訓練 (500k step) で trend のみ確認** | フル訓練 (410h×4) を回避、相対順序は出る | 絶対値が論文と乖離、未収束 config で trend 逆転リスク | (採用候補) コスト最優先なら本案。trend が不安定なら step 延長 | フル訓練予算が無いとき (= ほぼ常時) |
| **Optuna / W&B sweep で hyperparameter 探索** | 離散 matrix を超えた最適化、自動 logging | 論文 Table の離散軸 (T=2/3/4/5) と対応しない、計算コスト増 | 本チケットは論文 trend 再現が目的で sweep ではない | 論文外の最適化が必要になったとき |
| **論文 Table と定量一致を目標化** | 再現の説得力が増す | backend/GPU/MFCC/DTW 差で絶対値一致は非現実的 (T-M4.1 §8.2) | 定性 trend (順序) で十分、絶対値は参考 | 論文著者から測定条件の詳細が得られたとき |

#### 採用設計
- **既存 `train_*.py` / `evaluate()` を matrix で起動するだけ** (新規ロジックゼロ)
- **コスト削減時は fixed 500k step 打ち切り** + 定性 trend (順序関係) で合格判定
- **論文 Table は参考値**、再現の合否は自系列内の相対 trend (T↑で品質↑・RTF↑ 等)

#### 再評価トリガー
- **予算確定時**: GPU 予算が付いた時点で着手。付かなければ本チケットをスキップし T-M7.1 は M6.1/M6.2 の default config (GAN T=4 / Diff w/ sub-modeling) を best とする
- **trend が不安定**: 500k step で順序が論文と逆転したら step 延長 or matrix 間引き

### 8.2 思想 / 哲学の見直し
- **粒度**: 任意タスクとして独立は妥当。ただし matrix の T=4 / w-sub は M6.1/M6.2 と共有できるため、追加訓練は T=2/3/5 (GAN) と single model (Diff) の差分セルに限定するのが最小コスト
- **定性 trend 主軸**: 個人 GPU では論文の絶対値再現は非現実的 (CLAUDE.md「個人 GPU での全モデル本格訓練は非現実的」)。ablation は「論文の主張する trend が再現するか」の定性確認に徹し、絶対値一致は求めない
- **任意性**: 予算次第でスキップ可能。M7 (MOS) は M6.1/M6.2 の default config だけでも実施できるため、本チケットは M7 の前提ではない (best config 選定を高度化するのみ)

### 8.3 学んだこと (チケット完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M7.1 (内部 MOS テスト) へ
- **best config の申し送り**: ablation で選定した best config (品質・RTF のバランスで最良。論文に倣えば GAN T=4〜5 / Diff w/ sub-modeling + post-filter) を主観評価の対象に。比較表 (`eval_results/ablation_comparison.md`) を判断材料として渡す
- ablation をスキップした場合は M6.1/M6.2 の default config (GAN T=4 / Diff w/ sub-modeling) を best とする

#### ユーザー操作 (必須)
- **GPU 予算**: ablation は計算コストが大きく (GAN T=2/3/5 追加訓練 + Diff single model 32h)、**優先度は低い**。予算が無ければスキップ。フル訓練が無理なら fixed 500k step 打ち切りを user に提示して GO/NO-GO を仰ぐ

#### レポート出力
- 論文 Table 1〜3 との対比レポートを `eval_results/ablation_comparison.md` (または `docs/`) に出力。fixed step 打ち切りの場合はその旨を明記し absolute 値は参考扱い

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M6.3 の Deliverable を実績で更新
  - [ ] `docs/tickets/index.md` の T-M6.3 ステータスを `📝 pending` → `✅ completed`、M6 進捗サマリ + フェーズレビューログ M6 行を更新
  - [ ] §8.3 に ablation で判明した trend / 想定外を追記

### 9.3 Open question として残ったもの
- fixed step (500k) で論文 trend が安定再現するか (未収束 config で順序逆転リスク、step 延長で確認)
- 論文 RTF (A100 / AMD EPYC 7542) との絶対値差が許容範囲か (環境差で系統的にずれる、相対比較主軸)
- post-filter の高域改善が NISQA で定量化できるか (聴感では明確でも指標に出ないことがある)
