---
id: T-M7.1
title: 内部 MOS テスト (主観評価、人間必須、Claude Code は集計のみ)
milestone: M7
phase: M7
status: pending
size: M
owner: -
created: 2026-05-27
updated: 2026-05-27
depends_on: [T-M6.1, T-M6.2]
blocks: []
related_docs:
  - docs/milestones.md#m71-内部-mos-テスト
  - docs/paper-summary.md
---

# T-M7.1: 内部 MOS テスト (主観評価、人間必須、Claude Code は集計のみ)

> **マイルストーン**: [M7](../milestones.md#m7-主観評価-任意-人間が必須-claude-code-は集計のみ) / **サブタスク**: [M7.1](../milestones.md#m71-内部-mos-テスト)
> **依存**: [T-M6.1](T-M6.1-gan-full-training.md) (GAN best.pt), [T-M6.2](T-M6.2-diff-full-training.md) (Diff 4 sub-model + FIR) / **後続**: なし (再現実装の最終成果)

## 1. タスク目的とゴール

### 目的
論文 §4.2 / §5 (`docs/paper-summary.md` L64,L75,L98) の **主観評価 (5 段階 MOS)** を内部・少人数 (5〜10 名) で実施し、**GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN**、**Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)** という論文の相対序列 (paper-summary L98「4 iter で MOS は WaveFit 5 iter および HiFi-GAN と同等」) を再現できるかを確認する。これは客観評価 (M4) では捉えきれない知覚品質を人間が判定する **再現実装の最終品質証明** であり、客観評価との相関を取ることで再現性を多角的に裏付ける。

論文 (20 サンプル × ネイティブ英語話者 20 名) に対し、本チケットは **内部・少人数** で行うため、**MOS の絶対値は評価者・環境依存で論文と一致しない前提**に立ち、**自系列内の相対比較 (model 間の順序関係)** を合否の主軸とする (T-M4.1 §8.2 と整合)。Claude Code は **(a) 質問票・評価用 web app の自動生成 (sample シャッフル + model 名匿名化)**、**(b) 評価結果 CSV を受けての統計検定 (paired t-test / Mann-Whitney) + 95% CI 計算**、**(c) 論文 MOS との比較表 + 客観指標 (UTMOS/NISQA) との相関係数の算出** のみを担う。**評価者の手配・実評価・倫理審査・CSV 提供は user 操作** (§9.3)。

M7 は **任意** (`docs/milestones.md` L589「主観評価 (任意、人間が必須)」)。M6 の客観評価で論文相対整合が確認できていれば再現性は概ね担保されており、本チケットは主観面での追加検証。倫理審査・評価者手配のコストを user が許容する場合のみ実施する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M7.1 Acceptance を内包):
- [ ] `scripts/build_mos_test.py` が **20 utterances × 6 models = 120 sample** の質問票 (評価用 web app or Google Forms 用) を **sample シャッフル + model 名匿名化 (例: A〜F のランダム割当) + 各評価者で順序ランダム化** して自動生成する
- [ ] 6 models = **GT, GAN-WaveNeXt 2 (T=4), Diff-WaveNeXt 2 (w/ sub-model), HiFi-GAN, WaveFit, FastDiff (w/ sub-model)** のサンプルが揃い、本命 2 系列は T-M6.1 `checkpoints/gan/best.pt` / T-M6.2 `reverse_sample(post_filter=fir, seed=43)` 由来 (deterministic、§6.1)
- [ ] `src/wavenext2/eval/mos_analysis.py` が **評価結果 CSV を入力**として **paired t-test / Mann-Whitney U** と **95% CI (信頼区間)** を計算し、`MOSResult` (per-model mean/std/CI + 検定 p 値の pairwise 行列) を返す
- [ ] **論文 MOS との比較表** が生成される (`eval/report.py` 一本化、`docs/paper-summary.md` の MOS 序列と対比、絶対値でなく相対序列で整合判定)
- [ ] **MOS と UTMOS/NISQA (客観、M4.2) の相関係数** (Pearson / Spearman) が算出され、客観 ↔ 主観の整合度が定量化される
- [ ] **Acceptance**: GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN、Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)。少人数のため統計的有意差まで要求せず **点推定の順序 + CI 重なり** で判定 (§6.1)
- [ ] `mos_results/*.json` に per-evaluator 生スコア + 集計 + 検定結果を永続化 (`.gitignore` 登録、M4 の `eval_results/` と同方針)
- [ ] `tests/test_mos_analysis.py` の全テスト pass (合成 CSV で検定 / CI / 相関が計算される)
- [ ] `docs/milestones.md` §M7.1 Acceptance 2 項目クリア、`docs/tickets/index.md` の T-M7.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/eval/mos_analysis.py` (本実装: CSV 入力 → `paired_t_test` / `mann_whitney` / `confidence_interval` / `correlate_with_objective` / `MOSResult` dataclass)
  - `scripts/build_mos_test.py` (質問票生成: 120 sample の sample シャッフル + model 名匿名化 + 評価用 web app / Google Forms CSV / webMUSHRA config の生成)
  - `tests/test_mos_analysis.py` (合成 CSV で検定 / CI / 相関 / 匿名化往復のテスト)
- 編集:
  - `src/wavenext2/eval/__init__.py` (`__all__` に `mos_analysis` の公開関数 / `MOSResult` を追加)
  - `src/wavenext2/eval/report.py` (M6 で一本化済の対比レポート生成器に **MOS 比較表 + 相関表** を追加、`eval_results/*.json` + `mos_results/*.json` を読む)
  - `.gitignore` (`mos_results/` を追加。`eval_results/` と同方針、§6.3)
  - `docs/milestones.md` §M7.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M7.1 ステータス + M7 フェーズレビューログ行
- **新規モデル / 訓練ロジックなし** — 本チケットは生成サンプル (T-M6.1 / T-M6.2) を入力に質問票生成と統計集計のみ。`evaluate()` facade (T-M4.1)・`reverse_sample` (T-M3.3)・`eval/report.py` (M6) を流用する。

### 2.2 主要構造

#### `mos_analysis.py` — CSV 入力 → 統計検定 + CI + 相関 (本実装)

```python
"""mos_analysis.py — 内部 MOS テストの統計集計.

評価結果 CSV (evaluator × sample の 5 段階スコア) を入力に、model ごとの
平均 MOS / 95% CI と model 間の paired t-test / Mann-Whitney U を計算する。
客観指標 (UTMOS/NISQA、M4.2) との相関係数も算出する。

MOS の絶対値は評価者・環境依存のため、合否は自系列内の相対序列 (T-M4.1 §8.2)。
参考: docs/paper-summary.md §4-5 / docs/milestones.md §M7.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MOSResult:
    """MOS 集計結果. per-model 統計 + pairwise 検定 + 客観相関."""

    per_model: dict[str, dict[str, float]]   # {model: {"mean", "std", "ci_low", "ci_high", "n"}}
    pairwise: dict[tuple[str, str], dict[str, float]]  # {(A,B): {"t_p", "mw_p", "mean_diff", "ci_low", "ci_high"}}
    correlation: dict[str, float]            # {"pearson_utmos", "spearman_utmos", "pearson_nisqa", ...}
    n_evaluators: int

    def to_json(self, path: "str | Path") -> None: ...
    @classmethod
    def from_json(cls, path: "str | Path") -> "MOSResult": ...


def load_mos_csv(path: "str | Path") -> "pd.DataFrame":
    """評価結果 CSV を long-form DataFrame に正規化.

    想定 CSV: evaluator_id, sample_id, model (匿名 code を真名へ復号), utterance_id, score(1-5)。
    Google Forms の wide-form (1 行 = 1 評価者) も build_mos_test.py の mapping で復号して受ける。
    """
    ...


def confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """平均 MOS の 95% CI (t 分布、少人数 n<30 前提で normal でなく t)."""
    ...


def paired_t_test(model_a: np.ndarray, model_b: np.ndarray) -> dict[str, float]:
    """同一 (evaluator, utterance) ペアでの paired t-test (scipy.stats.ttest_rel).

    MOS は同じ評価者が全 model を採点する paired 設計のため paired を既定とする。
    戻り値: {"t_stat", "t_p", "mean_diff", "ci_low", "ci_high"}。
    """
    ...


def mann_whitney(model_a: np.ndarray, model_b: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U (順序尺度 MOS の正規性を仮定しないノンパラ、scipy.stats.mannwhitneyu).

    MOS は 5 段階順序尺度なので、paired t-test と併記して頑健性を担保 (§6.1)。
    """
    ...


def correlate_with_objective(
    mos_per_utt: "pd.DataFrame",       # utterance × model の平均 MOS
    objective: "pd.DataFrame",         # eval_results/*.json 由来 (UTMOS/NISQA per utterance)
) -> dict[str, float]:
    """MOS と UTMOS/NISQA の Pearson / Spearman 相関 (主観 ↔ 客観の整合度)."""
    ...


def analyze(
    csv_path: "str | Path",
    code_to_model: dict[str, str],     # build_mos_test.py が出力した匿名 code → 真名
    objective_json: "str | Path | None" = None,
    *,
    save_to: "str | Path | None" = None,
) -> MOSResult:
    """CSV → MOSResult。全 pairwise 検定 + CI + (objective_json があれば) 相関。"""
    ...
```

#### `scripts/build_mos_test.py` — 120 sample 質問票生成 (sample シャッフル + 匿名化)

```python
"""build_mos_test.py — 内部 MOS テストの質問票生成.

20 utterances × 6 models = 120 sample を、(a) model 名を匿名 code (A-F) に
ランダム割当、(b) 各評価者ごとに sample 順序をランダム化 して出力する。
出力形式は --format で web app (JSON) / google_forms (CSV+説明) / webmushra (config) を選択。

6 models のサンプル所在:
  - GT                          : test-clean GT 波形 (sox norm -3dB、T-M2.1)
  - GAN-WaveNeXt 2 (T=4)        : T-M6.1 checkpoints/gan/best.pt の合成出力
  - Diff-WaveNeXt 2 (w/ sub-model): T-M6.2 reverse_sample(post_filter=fir, seed=43)
  - HiFi-GAN / WaveFit / FastDiff: 比較対象 (公式 checkpoint or 自前訓練、§6.1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_questionnaire(
    sample_dir: Path,                  # 6 model × 20 utt の wav が置かれたディレクトリ
    n_utterances: int = 20,
    models: tuple[str, ...] = (
        "GT", "GAN-WaveNeXt2-T4", "Diff-WaveNeXt2-submodel",
        "HiFi-GAN", "WaveFit", "FastDiff-submodel",
    ),
    n_evaluators: int = 8,             # 5〜10 名
    seed: int = 43,                    # deterministic な匿名化 / シャッフル
    out_format: str = "web",           # "web" | "google_forms" | "webmushra"
    out_dir: Path = Path("mos_test/"),
) -> dict[str, str]:
    """120 sample の質問票を生成し、匿名 code → 真名の mapping を返す.

    - 匿名化: model → ランダム code (A-F)。mapping は out_dir/code_map.json に保存
      (集計時に mos_analysis.analyze へ渡す。被験者には絶対に渡さない)
    - シャッフル: 各評価者ごとに 120 sample の提示順をランダム化 (順序 bias 排除)
    - 出力: web app 用 playlist JSON / Google Forms インポート CSV / webMUSHRA config
    """
    ...
```

#### CLI (Claude Code が起動。生成と集計の 2 フェーズ)

```bash
# フェーズ 1: 生成サンプル準備 (T-M6.1/T-M6.2 の本命 + 比較対象を sample_dir に集約)
uv run python scripts/build_mos_test.py --sample-dir mos_test/samples --n-utterances 20 \
    --n-evaluators 8 --format web --out-dir mos_test/

# (ここで user が評価者に web app / Google Forms を配布し、5 段階で採点。CSV を回収)

# フェーズ 2: 回収 CSV を統計集計 (paired t-test / Mann-Whitney / CI / 相関)
uv run python -m wavenext2.eval.mos_analysis --csv mos_test/responses.csv \
    --code-map mos_test/code_map.json --objective eval_results/gan_full.json \
    --save-to mos_results/mos.json

# フェーズ 3: 論文 MOS 比較表 + 相関表を生成 (eval/report.py 一本化)
uv run python -m wavenext2.eval.report --mos mos_results/mos.json --paper-compare
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| MOS scale | **5 段階** (1=Bad 〜 5=Excellent、0.5 刻みは不可とし整数) | docs/paper-summary.md L75 |
| utterances 数 | **20** (test-clean からサブセット) | docs/paper-summary.md L64 / docs/milestones.md L594 |
| models 数 | **6** (GT + 本命 2 + 比較 3) | docs/milestones.md L594 |
| sample 総数 | **120** (= 20 × 6) | docs/milestones.md L594 |
| 評価者数 | **5〜10 名** (内部・少人数) | docs/milestones.md L595 |
| 論文評価者数 (参考) | 20 名 (ネイティブ英語話者) | docs/paper-summary.md L64 |
| 6 models | GT / GAN-WaveNeXt 2 (T=4) / Diff-WaveNeXt 2 (w/ sub-model) / HiFi-GAN / WaveFit / FastDiff (w/ sub-model) | チケット仕様 / docs/paper-summary.md L98-100 |
| 本命 GAN サンプル | T-M6.1 `checkpoints/gan/best.pt` (T=4) | T-M6.1 §9.1 |
| 本命 Diff サンプル | T-M6.2 `reverse_sample(post_filter=fir, seed=43)` | T-M6.2 §9.1 (T-M7.1 へ) |
| seed (匿名化 / シャッフル) | 43 (deterministic、M4/M6 と統一) | T-M4.1 / T-M6.2 |
| CI 信頼水準 | 95% (t 分布、n<30) | 慣例 / §6.1 少人数 |
| 検定 (主) | paired t-test (`scipy.stats.ttest_rel`) | MOS の paired 設計 |
| 検定 (頑健性) | Mann-Whitney U (`scipy.stats.mannwhitneyu`) | 順序尺度・非正規性 |
| 相関 | Pearson + Spearman (vs UTMOS/NISQA) | 主観 ↔ 客観整合 |
| Acceptance (相対序列) | GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub) | docs/milestones.md L599-600 |

### 2.4 アルゴリズム / 処理フロー

1. **(前提) M6 完了確認**: T-M6.1 `checkpoints/gan/best.pt` + T-M6.2 4 sub-model + `post_filter/fir.npy` が揃い、客観評価 (`eval_results/*.json`) が論文相対整合済であること。M7 は任意なので user が実施判断 (§9.3)
2. **比較対象サンプルの準備** (§6.1 の判断): HiFi-GAN / WaveFit / FastDiff の 20 utterance 分の合成を **公式 checkpoint 推論** または **自前訓練 (M6.3 ablation の副産物)** で用意し、本命 2 系列 (T-M6.1/T-M6.2) + GT と合わせ `mos_test/samples/` に 6 model × 20 utt = 120 wav を集約。**全 model で同一の sox norm -3dB 正規化** を適用し音量 bias を排除
3. **質問票生成** (`scripts/build_mos_test.py`、Claude Code): model → 匿名 code (A-F) ランダム割当、各評価者ごとに 120 sample の提示順ランダム化、`--format` で web app JSON / Google Forms CSV / webMUSHRA config を出力。匿名 mapping を `code_map.json` に保存 (被験者非公開)
4. **(user 操作) 倫理審査 + 評価者手配**: 各実施機関の規程に従いインフォームドコンセントを取得 (README L130)。内部メンバー 5〜10 名 (or 外部クラウドソーシング) を手配し、可能ならヘッドホン + 静音環境で評価を依頼
5. **(user 操作) 評価実施 + CSV 回収**: web app / Google Forms で 5 段階採点。結果を `responses.csv` (evaluator_id, sample_id, score, ...) として Claude Code に提供
6. **統計集計** (`mos_analysis.analyze`、Claude Code): 匿名 code を復号 → per-model 平均 MOS + 95% CI (t 分布)、全 model pairwise の paired t-test + Mann-Whitney、`mos_results/mos.json` に永続化
7. **論文 MOS 比較 + 客観相関** (`eval/report.py`、Claude Code): 論文の MOS 序列 (paper-summary L98) と対比表を生成 (絶対値でなく相対序列)、`eval_results/*.json` の UTMOS/NISQA per-utterance と MOS の Pearson/Spearman 相関を算出
8. **Acceptance 判定**: GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub) を **点推定の順序 + CI 重なり** で判定 (少人数のため有意差は努力目標、§6.1)
9. **完了報告**: 変更ファイル / 通過 acceptance / 論文 MOS 比較表 / 客観相関 / 既知懸念 (少人数の検出力・環境統制) を user に提示。M7 が再現実装の最終成果

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `mos_analysis.py` (検定/CI/相関) + `build_mos_test.py` (質問票生成/匿名化) 実装 + テスト記述、`eval/report.py` への MOS 比較表追加 | general-purpose |
| Reviewer | 1 | 統計手法の妥当性 (paired vs unpaired, t-test vs Mann-Whitney, CI の t 分布)、匿名化/シャッフルの bias 排除、論文 MOS 序列との整合確認、倫理審査要件の明記確認 | general-purpose |
| Analyst | 1 | 回収 CSV (user 提供) での集計実行 + 論文比較表 + 客観相関の算出 + Acceptance 判定 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **N/A** (M7 は本チケット 1 つのみ)
- 並列実行する場合の最大並列数: 1
- **本チケットは user 操作 (倫理審査・評価者手配・CSV 回収) を挟むため、Claude Code の作業は「生成フェーズ」と「集計フェーズ」に分断される** (§2.4)。生成フェーズ完了後は user の評価完了を待つ非同期タスク

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/build_mos_test.py`: 120 sample (20 utt × 6 model) 質問票生成、**model 名匿名化 (A-F ランダム)** + **評価者ごとの sample 順序ランダム化**、web app / Google Forms / webMUSHRA の 3 形式出力、`code_map.json` (匿名 mapping) 保存
- `src/wavenext2/eval/mos_analysis.py`: 評価結果 CSV → **paired t-test / Mann-Whitney U** + **95% CI** + **UTMOS/NISQA との相関 (Pearson/Spearman)**、`MOSResult` dataclass + `mos_results/*.json` 永続化
- `eval/report.py` への **論文 MOS 比較表** + **客観相関表** の追加 (M6 一本化済レポート生成器を流用)
- 本命 2 系列の deterministic サンプル取得 (T-M6.1 `best.pt` / T-M6.2 `reverse_sample(seed=43)`)
- Acceptance 判定 (相対序列 + CI 重なり): GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub)
- `tests/test_mos_analysis.py` (合成 CSV で検定/CI/相関/匿名化往復)
- `.gitignore` への `mos_results/` 登録

### Out of Scope
- **評価者の手配・実評価・スコア入力** → user 操作 (§9.3、Claude Code は評価できない)
- **倫理審査・インフォームドコンセントの取得・被験者保護** → user / 各実施機関の責務 (README L130、Claude Code は要件を質問票に明記するのみ)
- **比較対象 (HiFi-GAN/WaveFit/FastDiff) の訓練本体** → 公式 checkpoint 推論 or T-M6.3 ablation の副産物を流用 (本チケットは合成済みサンプルを受けるだけ。§6.1 で調達方針を判断)
- **客観指標 (UTMOS/NISQA/MCD/log F0 RMSE) の計算本体** → T-M4.1/T-M4.2 (本チケットは `eval_results/*.json` を読んで相関を取るのみ)
- **MUSHRA / CMOS / AB preference の本実装** → §8.1 代替案 (MOS が論文準拠。検出力が不足する場合に検討)
- **大規模クラウドソーシング (MTurk / Prolific)** → §8.1 (内部少人数が本チケットの前提)
- **論文と同規模 (20 名) の評価** → 内部少人数 (5〜10 名) が前提、規模差は §6.1 で明示

### Deliverable
- ファイル: `src/wavenext2/eval/mos_analysis.py`, `scripts/build_mos_test.py`, `tests/test_mos_analysis.py`
- 成果物: `mos_test/` (質問票 + `code_map.json`)、`mos_results/mos.json` (集計 + 検定結果)、論文 MOS 比較表 + 客観相関表 (`eval/report.py` 出力)
- 関数 / クラス: `analyze(csv_path, code_to_model, objective_json) -> MOSResult`, `MOSResult` dataclass, `build_questionnaire(...)`, `paired_t_test` / `mann_whitney` / `confidence_interval` / `correlate_with_objective`
- ドキュメント差分: `docs/milestones.md` §M7.1、`docs/tickets/index.md`、§8.3 に MOS 結果 (論文対比 / 相関 / 検出力所感) を追記

## 5. テスト項目

> **本チケットの実評価は user (人間の評価者) が行うため、Claude Code 側のテストは「質問票生成」と「集計ロジック」を合成データで検証する**。MOS スコア自体は CI に乗らない (人間依存)。

### 5.1 Unit テスト (`tests/test_mos_analysis.py`)
- [ ] `test_build_120_samples` — `build_questionnaire(n_utterances=20, models=<6>)` が **正確に 120 sample** を生成する (20 × 6)
- [ ] `test_anonymize_roundtrip` — model 名が匿名 code (A-F) に置換され、`code_map.json` で復号すると真名に戻る (被験者向け出力に真名が混入しない)
- [ ] `test_shuffle_per_evaluator` — 各評価者で sample 提示順が異なり (順序ランダム化)、同一 seed では再現する (deterministic)
- [ ] `test_paired_t_test` — 合成 CSV (model A > B が既知) で `paired_t_test` が正しい符号の `mean_diff` と妥当な p 値を返す (`scipy.stats.ttest_rel` 整合)
- [ ] `test_mann_whitney` — 同合成 CSV で Mann-Whitney U が paired t-test と整合する向きの p 値を返す (順序尺度頑健性)
- [ ] `test_confidence_interval_95` — 既知の平均/分散の合成スコアで 95% CI が t 分布の理論値と一致 (n<30 で normal でなく t を使う)
- [ ] `test_correlation_with_objective` — MOS と UTMOS が相関する合成データで Pearson/Spearman が高相関を返し、無相関データで ≈ 0 を返す
- [ ] `test_acceptance_ordering` — GAN(T=4) > HiFi-GAN / Diff(w/sub) > FastDiff(w/sub) の合成 CSV で Acceptance 判定が pass、逆順で fail
- [ ] `test_mos_result_json_roundtrip` — `MOSResult.to_json` → `from_json` で per_model / pairwise / correlation が往復一致 (`mos_results/*.json` 永続化)
- [ ] `test_load_csv_wide_and_long` — Google Forms wide-form / long-form の両 CSV を `load_mos_csv` が同一 DataFrame に正規化
- [ ] `test_missing_scores_handled` — 一部 sample 未採点 (欠損) の評価者がいても NaN セーフに集計 (paired は共通 sample のみ、CI は有効 n で)

### 5.2 e2e / 結合テスト
- [ ] **生成 → (ダミー回答) → 集計** の一周: `build_mos_test.py` で生成した質問票に対し合成回答 CSV を作り、`analyze` が `MOSResult` を返し `eval/report.py` が比較表を出す (人間評価をダミーで代替した API 動作確認)
- [ ] **本命サンプル取得**: T-M6.1 `best.pt` / T-M6.2 `reverse_sample(seed=43)` で 20 utt の wav が生成され `sample_dir` に揃う (deterministic 再生成可)
- [ ] **客観 JSON 連携**: `eval_results/gan_full.json` (T-M6.1) / `eval_results/diff_full_postfilter.json` (T-M6.2) の per-utterance UTMOS/NISQA を読んで MOS と相関が計算される

### 5.3 Acceptance criteria (`docs/milestones.md` §M7.1 より転記)
- [ ] GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN
- [ ] Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)

### 5.4 検証項目 (本チケット独自、チケット仕様 §5 より)
- [ ] **120 sample の質問票が正しく生成される** (sample のシャッフル、model 名匿名化) — `test_build_120_samples` / `test_anonymize_roundtrip` / `test_shuffle_per_evaluator`
- [ ] **CSV 入力で paired t-test / Mann-Whitney が計算される** — `test_paired_t_test` / `test_mann_whitney`
- [ ] **95% CI が計算される** — `test_confidence_interval_95`
- [ ] **論文 MOS との比較表** — `eval/report.py --paper-compare` が paper-summary L98 の序列と対比表を出す (相対序列で整合)
- [ ] **MOS と UTMOS/NISQA (客観) の相関係数** — `test_correlation_with_objective` + 実 `eval_results/*.json` 連携

## 6. 懸念事項

> **本チケットは人間の評価者が必須で、Claude Code は質問票生成と集計のみ**。最大の懸念は (a) 倫理審査・被験者保護、(b) 少人数の統計的検出力、(c) 比較対象サンプルの調達。

### 6.1 技術的リスク

#### 【最優先】被験者保護・倫理審査 (README L130)
- 主観評価は **人間を被験者とする実験**であり、各実施機関 (大学 / 企業) の倫理規程・IRB の対象になりうる。README L130 が「被験者保護・倫理審査は各実施機関の規程に従うこと」と明記
- **緩和**: ① 質問票冒頭に **インフォームドコンセント** (目的・所要時間・データの匿名化・撤回権・連絡先) を必須項目として `build_mos_test.py` が自動挿入、② 個人を特定する情報を収集しない (evaluator_id は無作為 ID)、③ user に「実施前に所属機関の倫理審査要否を確認する」ことを完了報告で明示
- **責務境界**: Claude Code は **要件を質問票に組み込む** ことまで。審査の申請・承認・被験者対応は **user の責務** (§9.3)。Claude Code が倫理審査を代行することはできない
- **検知**: 質問票にインフォームドコンセント節が無い / 個人情報項目が混入していれば fail

#### 【重要】少人数 (5〜10 名) で統計的検出力が足りない (論文は 20 名)
- 論文は 20 名だが本チケットは内部 5〜10 名。MOS の評価者間分散は大きく、**n=5〜10 では model 間の有意差が出ない (検出力不足)** 公算が高い
- **緩和**: ① **有意差 (p<0.05) を Acceptance の必須にせず、点推定の順序 + 95% CI の重なり** で判定 (CI が大きく重なっても点推定で GAN ≥ HiFi-GAN なら pass 寄り)、② **paired 設計** (同一評価者が全 model 採点) で評価者間分散を相殺し検出力を上げる、③ 1 評価者あたり 120 sample 全採点で per-pair の n を稼ぐ、④ 検出力不足を §8.3 / 完了報告に明記し「相対傾向の参考値」と位置づける
- **検知**: pairwise p 値が軒並み >0.05 でも順序が論文と整合すれば pass 寄り。CI 幅と n を report に必ず併記
- **再評価トリガー**: 評価者数確定時 (§8.1)。20 名以上確保できるなら有意差まで要求、クラウドソーシングで大規模化を検討

#### 【重要】model 名の匿名化と sample 順序のランダム化 (bias 排除)
- 評価者が「これは提案手法」と分かると **確証 bias** で高評価しがち。また先に聞いた sample が基準になる **順序 bias** も生じる
- **緩和**: ① model → 匿名 code (A-F) を **seed 固定でランダム割当** し被験者には真名を一切見せない (`code_map.json` は集計専用・非公開)、② **各評価者ごとに 120 sample の提示順を独立にランダム化** (順序 bias 排除)、③ GT も匿名 code に混ぜる (GT と分かると基準化される)
- **検知**: `test_anonymize_roundtrip` / `test_shuffle_per_evaluator`。被験者向け出力 (web app / Forms) に model 真名が grep で見つかれば fail

#### 【重要】評価環境の統制 (ヘッドホン・静音) が内部評価で担保できない
- 内部少人数だと評価者が各自の環境 (スピーカー / ノイズ環境 / 再生音量) で聞くため、環境差が MOS に混入する。論文は統制環境を想定
- **緩和**: ① 質問票に **推奨環境 (ヘッドホン推奨・静音・音量校正)** を明記、② 冒頭に **音量校正用リファレンス sample** を置く、③ web app なら再生環境 (デバイス種別) を任意申告させて report に記録、④ **絶対 MOS でなく相対序列を主軸**にすることで環境差の影響を相対的に吸収 (T-M4.1 §8.2)
- **検知**: 環境統制は完全には担保できない前提で、report に「環境非統制」を明示し絶対値の解釈に注意喚起

#### 【重要】比較対象 (HiFi-GAN/WaveFit/FastDiff) のサンプルをどう用意するか
- 6 models のうち 3 つ (HiFi-GAN/WaveFit/FastDiff) は本リポジトリの訓練対象外。サンプルを **(a) 公式 checkpoint で推論** するか **(b) 自前訓練 (T-M6.3 ablation の副産物)** するかで品質基準が変わる
- **緩和**: ① **公式 pretrained checkpoint** があれば第一候補 (HiFi-GAN/FastDiff は公開 checkpoint あり、参考実装 README L83-94)。ただし **LibriTTS-R 24kHz で学習済か** を確認 (別データ / 別 sr の checkpoint だと不公平)、② 無ければ **自前訓練** (T-M6.3 で baseline を訓練している場合は流用)、③ **どの checkpoint / 訓練で得たかを report に明記** し、論文の絶対値比較でなく自系列内の相対比較に徹する (§8.2)
- **検知**: 比較対象が別 sr / 別データの checkpoint だと MOS が不当に低く出て GAN-WaveNeXt 2 が「勝った」ように見える偽陽性 → checkpoint の素性を必ず記録
- **再評価トリガー**: 公式 checkpoint の LibriTTS-R 適合性が不明な場合、T-M6.3 で同一データ自前訓練に切替

#### 【重要】MOS の絶対値は評価者・環境依存 → 相対比較主軸 (T-M4.1 §8.2 と整合)
- MOS は評価者の採点癖 (辛い/甘い)・環境・サンプル集合で絶対値が大きく変わる。論文 (20 名・統制環境) の MOS 絶対値と内部少人数の値は一致しない前提
- **緩和**: **自系列内の model 間順序関係** を合否主軸とし、論文 MOS の絶対値一致は努力目標。paired 設計 + 同一サンプル集合で評価者の採点癖を相殺。`mos_results/*.json` に評価環境・評価者数を記録。これは M4 客観評価 (T-M4.1 §8.2) の「相対比較主軸」と完全に整合する哲学
- **検知**: 論文 MOS との絶対差が大きくても相対序列が整合すれば pass 寄り

#### その他リスク
- **GT が最高 MOS にならない異常**: 通常 GT (真の録音) が最高 MOS のはず。GT が vocoder 出力より低いと評価設計のバグ (匿名化ミス / サンプル取り違え / 正規化不整合) を疑う。GT を sanity anchor として report で確認
- **サンプル間の音量差による bias**: model ごとに出力音量が違うと音量で品質を誤判断される → **全 model を同一 sox norm -3dB** で正規化 (T-M2.1 / val と同じ)。GT も同様
- **CSV フォーマットの揺れ (Google Forms vs web app)**: 回収 CSV が wide-form (1 行 = 1 評価者 × 120 列) か long-form かで parser が変わる → `load_mos_csv` が両対応、`code_map.json` で復号。列名揺れは build 時に固定の sample_id を埋め込んで吸収
- **欠損・離脱評価者**: 全 sample を採点しない評価者がいると paired t-test の対が崩れる → 共通採点 sample のみで paired、CI は有効 n で計算、欠損率を report に記載
- **scipy バージョン依存の検定 API**: `mannwhitneyu` の `alternative` デフォルトが scipy バージョンで変わった経緯あり → `alternative="two-sided"` を明示。`ttest_rel` の `nan_policy` も明示
- **多重比較**: 6 model の全 pairwise = 15 検定で多重比較問題 (偽陽性増)。少人数で有意差を主張しない方針なので Bonferroni 等は必須としないが、report に「多重比較未補正」を注記 (主張する場合は補正を §8.1 で検討)
- **mos_results の cloud / commit 方針**: 被験者スコア (匿名でも) を含むため `mos_results/` は `.gitignore` で commit しない (個人情報保護)。`code_map.json` も同様に非公開

### 6.2 仕様の曖昧さ
- `docs/open-questions.md`: MOS 関連の項目は無い (論文の MOS 手法は標準的な 5 段階 ACR を想定)。本チケットで確定する判断:
  - **検定手法**: paired t-test を主、Mann-Whitney U を頑健性確認に併記 (MOS は paired 設計の順序尺度)
  - **CI**: 95%・t 分布 (n<30)
  - **Acceptance 判定基準**: 少人数のため有意差でなく **点推定の順序 + CI 重なり** (§6.1)
  - **比較対象サンプルの調達**: 公式 checkpoint 優先、LibriTTS-R 適合性確認、無ければ自前訓練 (§6.1)
- 論文が MOS で 0.5 刻みを使ったか整数かは不明 → **整数 5 段階 ACR** を採用 (最も標準的)

### 6.3 他チケットとの整合性
- **T-M6.1 (GAN フル訓練)** から受領: `checkpoints/gan/best.pt` (T=4) を本命 GAN サンプルの生成元に使う (§9.1)。`evaluate()` で得た `eval_results/gan_full.json` の UTMOS/NISQA を相関の客観側に使う
- **T-M6.2 (Diff フル訓練)** から受領: `reverse_sample(model, mel, seed=43, post_filter=fir)` で deterministic に Diff (w/ sub-model) サンプルを生成 (T-M6.2 §9.1 が T-M7.1 へ明記)。`eval_results/diff_full_postfilter.json` を相関に使う
- **T-M4.1 (evaluate facade)** と整合: **相対比較主軸** の哲学 (T-M4.1 §8.2) を MOS にも適用。客観 (M4) ↔ 主観 (M7) の相関で再現性を多角的に検証
- **T-M4.2 (UTMOS/NISQA)** から受領: UTMOS/NISQA の per-utterance 値 (`eval_results/*.json`) を MOS との相関の客観側に使う
- **`eval/report.py` (M6 で一本化)** を流用: 論文 Table 対比レポート生成器に MOS 比較表 + 相関表を追加 (GAN/Diff/ablation と同一フォーマット、M6 フェーズレビュー昇格)
- **`.gitignore`**: `eval_results/` (M4) と同方針で `mos_results/` + `mos_test/code_map.json` を除外 (被験者データ保護)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/milestones.md` §M7.1 / `docs/paper-summary.md` の MOS 手法 (5 段階・20 utt・6 model) と整合
- [ ] Acceptance criteria 全項目クリア (§5.3): GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub)
- [ ] 120 sample が正しく生成され、**model 名匿名化 + sample 順序ランダム化** が機能 (被験者向け出力に真名が漏れない)
- [ ] paired t-test / Mann-Whitney / 95% CI が `scipy.stats` と整合し合成データで検証済
- [ ] **少人数の検出力不足を明示** し、有意差でなく点推定順序 + CI 重なりで判定 (§6.1)、report に CI 幅と n を併記
- [ ] **倫理審査・インフォームドコンセント** が質問票に組み込まれ、審査は user 責務と明記 (README L130、§9.3)
- [ ] 比較対象 (HiFi-GAN/WaveFit/FastDiff) の checkpoint 素性 (公式 or 自前、LibriTTS-R 適合性) が report に記録され、論文絶対値でなく相対比較に徹している
- [ ] 全 model が同一 sox norm -3dB 正規化済 (音量 bias 排除)、GT が sanity anchor として最高 MOS 付近
- [ ] MOS と UTMOS/NISQA の相関 (Pearson/Spearman) が算出され、主観 ↔ 客観の整合度が示されている
- [ ] `mos_results/` / `code_map.json` が `.gitignore` 済 (被験者データ保護)、`MOSResult` JSON 往復一致
- [ ] M7 phase review の結果が §8.3 に記録
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] 参考実装 (webMUSHRA 等) をコピーしていない (質問票生成 / 集計ロジックは一から記述、webMUSHRA は config 出力先として利用するのみ)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M7 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **MUSHRA / AB preference test に変更** | MOS より検出力が高い (相対比較で評価者内分散が小さい)、少人数でも差が出やすい | 論文は MOS なので直接対比できない、UI・運用が複雑 | 論文 (paper-summary L75) が 5 段階 MOS。論文再現が第一目的 | **少人数 MOS で検出力が決定的に不足し論文序列を判定できないとき** (§6.1) |
| **CMOS (Comparative MOS) で直接比較** | 提案 vs baseline を直接比較し小さな差を検出、評価者の絶対採点癖を排除 | ペア提示で組合せ爆発 (6 model = 15 ペア)、論文は absolute MOS | 論文は ACR (absolute)。CMOS は対比が別軸 | **GAN-WaveNeXt 2 vs HiFi-GAN の僅差を検出したいとき** |
| **クラウドソーシング (Amazon MTurk / Prolific) で大規模化** | n を 20→100+ に増やし検出力・一般化を確保、論文同等以上 | 課金、評価者の質管理 (attention check 必須)、環境非統制が更に悪化 | 内部少人数が本チケットの前提 (milestones L595)。大規模化は予算依存 | **評価者を 20 名以上確保したい / 論文同規模で再現したいとき** |
| **webMUSHRA (既存 OSS) をそのまま使う** | UI 実装ゼロ、MUSHRA/MOS 両対応の枯れた web app、ローカルホスト可 | MUSHRA 寄りの UI、Google Forms より導入に技術知識が要る | `build_mos_test.py` で webMUSHRA config を出力する形で **部分採用** (§2.2)。集計は自前 | **web app の自前実装が重い / MUSHRA に移行するとき** (config 出力先として既に対応) |
| **倫理審査を最初から組み込む (IRB テンプレート同梱)** | 実施機関の審査をスムーズ化、被験者保護を設計段階で担保 | リポジトリに機関依存の審査書式を持つのは過剰、機関ごとに様式が違う | インフォームドコンセントの質問票挿入までは In Scope、審査書式は機関依存で user 責務 (§6.1) | **特定機関での反復実施が決まり様式を固定したいとき** |
| **2-stage: 客観 (M4) で足切り → 主観 (M7) は僅差ペアのみ** | 主観評価のコストを僅差 model 間に集中、評価者負担減 | 設計が複雑、論文は全 model MOS | 論文は全 6 model を MOS。本チケットも全 model | **評価者負担が過大で僅差ペアに絞りたいとき** |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M7.1 は質問票生成 + 統計集計 + 相関 + 論文対比を抱えるが、いずれも CSV ↔ 統計の薄い層で、`eval/report.py` (M6) と `eval_results/*.json` (M4) を流用するため size=M が妥当。実評価 (人間) を含まないので実装規模は中程度
- **責務境界 (Claude Code は評価できない)**: 本チケットの核は「**人間にしかできない評価を Claude Code が前後で支援する**」こと。質問票生成 (前) と統計集計 (後) は Claude Code、実評価・倫理審査・評価者手配は user。この分断は M6 (GPU は user、起動・監視は Claude Code) と同じ「人間必須作業の周辺を自動化する」思想
- **相対比較主軸の一貫性**: MOS の絶対値を諦め相対序列を主軸にするのは M4 客観評価 (T-M4.1 §8.2) と完全に同じ哲学。客観 (M4) と主観 (M7) を **相関** で結ぶことで、片方だけでは不十分な再現性検証を多角化する。これが M7 を「任意だが価値ある」最終段にしている
- **少人数の限界を隠さない**: 論文 20 名に対し 5〜10 名は検出力不足が構造的。これを「参考値」と明示し、有意差を偽装しない (多重比較未補正も注記)。再現実装の誠実さとして重要
- **最終成果としての位置づけ**: M7 は再現実装パイプラインの最終段。論文 MOS 比較表 + 客観相関が「論文を再現できた」ことの主観面の証明になる。ただし M7 が任意である以上、M6 客観評価で再現性が担保されていれば M7 未実施でもプロジェクトは成立する

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- **少人数 (5〜10 名) で論文序列を判定できたか**: TBD (有意差 or 点推定順序 + CI 重なりで判定)
- **GAN(T=4) MOS ≥ HiFi-GAN が再現したか**: TBD
- **Diff(w/sub) MOS ≥ FastDiff(w/sub) が再現したか**: TBD
- **MOS と UTMOS/NISQA の相関の強さ**: TBD (Pearson/Spearman、客観 ↔ 主観の整合度)
- **比較対象 checkpoint の素性が公平だったか**: TBD (公式 vs 自前、LibriTTS-R 適合性)
- **環境非統制 / 検出力不足の影響度**: TBD
- 想定外: TBD
- 教訓: TBD
- 再評価トリガー: **評価者数確定時** (5〜10 名 → 20 名以上に増やせるなら有意差要求 / MUSHRA / クラウドソーシングへ、§8.1)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **本チケットは M7 (最終段) で後続チケットなし**。最終成果は **論文との MOS 比較表 + 客観相関** で、これが再現実装の品質証明 (主観面)
- **最終成果物の所在**:
  - `mos_results/mos.json`: per-model MOS (mean/95% CI) + pairwise 検定 (paired t / Mann-Whitney) + UTMOS/NISQA 相関
  - `eval/report.py --paper-compare --mos`: 論文 MOS 序列との比較表 + 客観相関表
- **客観 (M4) と主観 (M7) の相関による多角検証**: M4 の `eval_results/*.json` (UTMOS/NISQA/MCD/log F0 RMSE) と M7 の MOS を相関で結び、「客観指標が良い ⇒ 主観も良い」が成立するかで再現性を裏付ける。これがプロジェクト全体の品質証明の締め
- **再現性の総括**: M6 (客観・自動) + M7 (主観・人間) の両輪で論文再現を主張。M7 が任意のため、M6 単独でも再現性は概ね担保される位置づけを明記

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M7.1 Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M7.1 ステータス更新 + M7 フェーズレビューログ行を更新
  - [ ] (該当時) `docs/paper-summary.md` に内部 MOS の再現結果 (相対序列 / 客観相関) を「再現実装の所見」として追記
  - [ ] §8.3 に MOS 結果 (論文対比 / 相関 / 検出力所感) を実測値付きで追記

### 9.3 ユーザー操作 (必須)
- **倫理審査・被験者保護**: 各実施機関の規程に従い、実施前に倫理審査の要否を確認 (README L130)。インフォームドコンセントの取得・被験者対応は user 責務 (Claude Code は質問票に同意節を組み込むのみ、§6.1)
- **評価者の手配**: 内部メンバー 5〜10 名 (or 外部クラウドソーシング) を手配。可能ならヘッドホン + 静音環境を依頼 (環境統制、§6.1)
- **評価実施 + CSV 提供**: web app / Google Forms で 5 段階採点を実施し、結果 CSV を Claude Code に提供 (Claude Code は人間評価を代行できない)
- **比較対象サンプルの素性確認**: HiFi-GAN/WaveFit/FastDiff の公式 checkpoint を使う場合、LibriTTS-R 24kHz 適合性を user が確認 (不適合なら自前訓練、§6.1)

### 9.4 Open question として残ったもの
- 少人数 (5〜10 名) で論文の MOS 序列を統計的に判定できるか → 検出力不足前提で点推定順序 + CI 重なりで判定、不足なら §8.1 (MUSHRA / クラウドソーシング) を評価者数確定時に再判断
- 比較対象 (HiFi-GAN/WaveFit/FastDiff) の公式 checkpoint が LibriTTS-R 24kHz に適合するか → 不明なら自前訓練 (T-M6.3 副産物) に切替、checkpoint 素性を report に記録
- MOS の絶対値が論文に届くか / 相対序列の再現で十分か → 相対比較主軸 (§6.1、T-M4.1 §8.2 と整合)
- 環境非統制 (内部評価) が MOS にどれだけ影響するか → 相対序列で吸収する前提、report に環境を記録
- `docs/open-questions.md` への追記要否: MOS 手法 (検定 / CI / 匿名化 / 倫理) は本チケットで確定。acceptance 未達時のみ再オープン (現時点では不要)
