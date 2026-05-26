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
# updated 2026-05-27: M7 phase review 反映 (Wilcoxon signed-rank への差し替え、CC BY 4.0 帰属義務を §6 最優先化)
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

論文 (20 サンプル × ネイティブ英語話者 20 名) に対し、本チケットは **内部・少人数** で行うため、**MOS の絶対値は評価者・環境依存で論文と一致しない前提**に立ち、**自系列内の相対比較 (model 間の順序関係)** を合否の主軸とする (T-M4.1 §8.2 と整合)。Claude Code は **(a) 質問票・評価用 web app の自動生成 (sample シャッフル + model 名匿名化、webMUSHRA config 生成を主軸)**、**(b) 評価結果 CSV を受けての統計検定 (一次判定 Wilcoxon signed-rank、paired t-test は論文対比の補助) + bootstrap (BCa) 95% CI 計算 + 評価者間信頼性 (ICC/Krippendorff's α)**、**(c) 論文 MOS との比較表 + 客観指標 (UTMOS/NISQA) との Spearman 相関 (主) の算出** のみを担う。**評価者の手配・実評価・倫理審査・CC BY 4.0 帰属表示・CSV 提供は user 操作** (§9.3)。

M7 は **任意** (`docs/milestones.md` L589「主観評価 (任意、人間が必須)」)。M6 の客観評価で論文相対整合が確認できていれば再現性は概ね担保されており、本チケットは主観面での追加検証。倫理審査・評価者手配のコストを user が許容する場合のみ実施する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M7.1 Acceptance を内包):
- [ ] `scripts/build_mos_test.py` が **20 utterances × 6 models = 120 sample** の質問票 (評価用 web app or Google Forms 用) を **sample シャッフル + model 名匿名化 (例: A〜F のランダム割当) + 各評価者で順序ランダム化** して自動生成する
- [ ] 6 models = **GT, GAN-WaveNeXt 2 (T=4), Diff-WaveNeXt 2 (w/ sub-model), HiFi-GAN, WaveFit, FastDiff (w/ sub-model)** のサンプルが揃い、本命 2 系列は T-M6.1 `checkpoints/gan/best.pt` / T-M6.2 `reverse_sample(post_filter=fir, seed=43)` 由来 (deterministic、§6.1)
- [ ] `src/wavenext2/eval/mos_analysis.py` が **評価結果 CSV を入力**として **Wilcoxon signed-rank (一次判定、対応あり順序尺度)** + **paired t-test (論文対比の補助)** + **Holm 補正後 p (15 pairwise)** と **bootstrap (BCa) 95% CI (既定、ordinal + 小 n で t 近似より頑健)** + **評価者間信頼性 (ICC(2,k) / Krippendorff's α)** を計算し、`MOSResult` (per-model mean/std/CI + 検定 p 値の pairwise 行列 + 信頼性指標) を返す
- [ ] **論文 MOS との比較表** が生成される (`eval/report.py` 一本化、`docs/paper-summary.md` の MOS 序列と対比、絶対値でなく相対序列で整合判定。MOS は ordinal なので順位ベースを正、ACR mean は慣例的近似)
- [ ] **MOS と UTMOS/NISQA (客観、M4.2) の相関係数** (Spearman 主 / Pearson 補助) が算出され、客観 ↔ 主観の整合度が定量化される
- [ ] **Acceptance**: GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN、Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)。少人数のため統計的有意差まで要求せず **点推定の順序 + bootstrap CI 重なり** で判定 (§6.1)。評価者間信頼性 α<0.6 なら序列判定自体を保留 (安全弁、§6.1)
- [ ] `mos_results/*.json` に per-evaluator 生スコア + 集計 + 検定結果を永続化 (`.gitignore` 登録。ただし **個人データとして `eval_results/` とは別格扱い** — 保持期間・削除責任・アクセス制御を伴う、§6.3 / §8.2)
- [ ] `tests/test_mos_analysis.py` の全テスト pass (合成 CSV で検定 / CI / 相関が計算される)
- [ ] `docs/milestones.md` §M7.1 Acceptance 2 項目クリア、`docs/tickets/index.md` の T-M7.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/eval/mos_analysis.py` (本実装: CSV 入力 → `wilcoxon_signed_rank` (一次判定) / `paired_t_test` (補助) / `holm_correct` / `bootstrap_ci` / `inter_rater_reliability` / `correlate_with_objective` / `MOSResult` dataclass。`sample_id` スキーマは eval 側 dataclass を import して権威を一元化、§8.2)
  - `scripts/build_mos_test.py` (質問票生成: **webMUSHRA config 生成器に絞る** (§8.1 昇格)。120 sample の sample シャッフル + model 名匿名化。自前 web app / Google Forms は deprecate 候補)
  - `tests/test_mos_analysis.py` (合成 CSV で Wilcoxon / paired t / Holm / bootstrap CI / 信頼性 / 相関 / 匿名化往復のテスト)
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
平均 MOS / bootstrap (BCa) 95% CI と model 間の Wilcoxon signed-rank (一次判定) を
計算する。paired t-test は論文対比の補助。15 pairwise は Holm 補正後 p を併記。
評価者間信頼性 (ICC(2,k) / Krippendorff's α) と客観指標 (UTMOS/NISQA、M4.2) との
Spearman 相関 (主) も算出する。

MOS は順序尺度なので一次判定は順位ベース (Wilcoxon)。1-5 を interval 扱いした
mean/t-test は慣例的近似であり厳密には不正のため、t-test は論文対比の補助に留める。
MOS の絶対値は評価者・環境依存のため、合否は自系列内の相対序列 (T-M4.1 §8.2)。
sample_id フォーマットは eval 側 dataclass を import して一元定義 (eval → scripts 単方向)。
参考: docs/paper-summary.md §4-5 / docs/milestones.md §M7.1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MOSResult:
    """MOS 集計結果. per-model 統計 + pairwise 検定 + 信頼性 + 客観相関."""

    per_model: dict[str, dict[str, float]]   # {model: {"mean", "std", "ci_low", "ci_high", "n"}} (CI は bootstrap BCa)
    pairwise: dict[tuple[str, str], dict[str, float]]  # {(A,B): {"wilcoxon_p" (一次), "wilcoxon_p_holm", "t_p" (補助), "median_diff", "ci_low", "ci_high"}}
    reliability: dict[str, float]            # {"icc_2k", "krippendorff_alpha"} — α<0.6 なら序列判定保留 (安全弁)
    correlation: dict[str, float]            # {"spearman_utmos" (主), "pearson_utmos" (補助), "spearman_nisqa", ...}
    n_evaluators: int
    schema_version: str                      # CSV/code_map スキーマ version (load 時に検証、§6.3 / §9.1)

    def to_json(self, path: "str | Path") -> None: ...
    @classmethod
    def from_json(cls, path: "str | Path") -> "MOSResult": ...


def load_mos_csv(path: "str | Path", *, expected_schema_version: str) -> "pd.DataFrame":
    """評価結果 CSV を long-form DataFrame に正規化.

    想定 CSV: evaluator_id, sample_id, model (匿名 code を真名へ復号), utterance_id, score(1-5)。
    CSV/code_map に埋め込まれた schema_version + seed + git_sha を検証 (version 不一致は raise、§6.3)。
    sample_id フォーマットは eval 側 dataclass の権威定義を import して parse (§8.2)。
    webMUSHRA の結果 CSV を build_mos_test.py の mapping で復号して受けるのを主経路とする。
    """
    ...


def bootstrap_ci(
    scores: np.ndarray, confidence: float = 0.95, *, n_resamples: int = 10000, seed: int = 43
) -> tuple[float, float]:
    """平均 MOS の bootstrap (BCa) 95% CI (既定).

    ordinal + 小 n では正規/t 近似が不正確なため bootstrap を既定とする
    (scipy.stats.bootstrap, method="BCa")。t 分布 CI は report の補助列。
    """
    ...


def wilcoxon_signed_rank(model_a: np.ndarray, model_b: np.ndarray) -> dict[str, float]:
    """同一 (evaluator, utterance) ペアの Wilcoxon signed-rank (一次判定、scipy.stats.wilcoxon).

    MOS は同一評価者が全 model を採点する paired 設計かつ順序尺度なので、対応あり
    ノンパラ検定の Wilcoxon signed-rank が正しい (Mann-Whitney は独立 2 標本検定で
    paired 設計に誤用だったため不採用、§6.1)。戻り値: {"stat", "wilcoxon_p", "median_diff"}。
    """
    ...


def paired_t_test(model_a: np.ndarray, model_b: np.ndarray) -> dict[str, float]:
    """同一 (evaluator, utterance) ペアでの paired t-test (補助、scipy.stats.ttest_rel).

    1-5 を interval 扱いする慣例的近似であり厳密には不正。一次判定は Wilcoxon、
    本検定は論文 (mean MOS) との対比の補助に留める。戻り値: {"t_stat", "t_p", "mean_diff"}。
    """
    ...


def holm_correct(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """15 pairwise (6 model 総当たり) の Holm-Bonferroni 補正後 p 値.

    多重比較の偽陽性を抑制。未補正 p と並べて Holm 補正後 p を必ず併記する (§6.1)。
    """
    ...


def inter_rater_reliability(long_df: "pd.DataFrame") -> dict[str, float]:
    """評価者間信頼性 ICC(2,k) と Krippendorff's α (順序).

    相関と並ぶ一級の再現性指標。α<0.6 なら評価者間一致が低く序列判定自体を保留
    する安全弁として report に明示する (§6.1)。
    """
    ...


def correlate_with_objective(
    mos_per_utt: "pd.DataFrame",       # utterance × model の平均 MOS
    objective: "pd.DataFrame",         # eval_results/*.json 由来 (UTMOS/NISQA per utterance)
) -> dict[str, float]:
    """MOS と UTMOS/NISQA の相関. MOS は ordinal なので Spearman を主、Pearson を補助."""
    ...


def analyze(
    csv_path: "str | Path",
    code_to_model: dict[str, str],     # build_mos_test.py が出力した匿名 code → 真名
    objective_json: "str | Path | None" = None,
    *,
    expected_schema_version: str,
    save_to: "str | Path | None" = None,
) -> MOSResult:
    """CSV → MOSResult。全 pairwise Wilcoxon (一次) + Holm + paired t (補助) +
    bootstrap CI + 信頼性 (ICC/α) + (objective_json があれば) Spearman 相関。"""
    ...
```

#### `scripts/build_mos_test.py` — 120 sample 質問票生成 (sample シャッフル + 匿名化)

```python
"""build_mos_test.py — 内部 MOS テストの webMUSHRA config 生成.

20 utterances × 6 models = 120 sample を、(a) model 名を匿名 code (A-F) に
ランダム割当、(b) 各評価者ごとに sample 順序をランダム化 して出力する。
出力は **webMUSHRA config を主軸** とする (匿名化・順序ランダム化・結果 CSV 出力を
内蔵する枯れた OSS に全面委譲。本スクリプトは config 生成器 1 本に絞り、自前 web app /
Google Forms は deprecate 候補、§8.1 昇格)。config に schema_version + seed + git_sha を埋込。

CC BY 4.0 帰属表示 (最優先・倫理、§6.1): LibriTTS-R は CC BY 4.0。GT/合成音声を web 配信
する際は出典・ライセンス・改変有無の attribution 表示が義務。合成音声は LibriTTS-R の
「改変物」に当たるため明記必須。config の説明文に帰属表記を必ず注入する。

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
    n_evaluators: int = 8,             # 5〜10 名 (事前検出力分析で逆算するのが本来、§8.1)
    seed: int = 43,                    # deterministic な匿名化 / シャッフル
    out_format: str = "webmushra",     # "webmushra" (主) | "web"/"google_forms" (deprecate 候補)
    out_dir: Path = Path("mos_test/"),
) -> dict[str, str]:
    """120 sample の webMUSHRA config を生成し、匿名 code → 真名の mapping を返す.

    - 匿名化: model → ランダム code (A-F)。mapping は out_dir/code_map.json に保存
      (集計時に mos_analysis.analyze へ渡す。被験者には絶対に渡さない)
    - シャッフル: 各評価者ごとに 120 sample の提示順をランダム化 (順序 bias 排除)
    - CC BY 4.0 帰属表記 (LibriTTS-R の出典・ライセンス・改変有無) を config 説明文に注入 (§6.1)
    - インフォームドコンセント + データ保持/削除フローへの参照も説明文に注入 (§6.1)
    - schema_version + seed + git_sha を code_map.json / config に埋込 (§6.3 / §9.1)
    - 出力: webMUSHRA config (主)。web app playlist / Google Forms CSV は deprecate 候補
    """
    ...
```

#### CLI (Claude Code が起動。生成と集計の 2 フェーズ)

```bash
# フェーズ 1: 生成サンプル準備 (T-M6.1/T-M6.2 の本命 + 比較対象を sample_dir に集約)
# webMUSHRA config を生成 (匿名化 + 順序ランダム化 + CC BY 4.0 帰属注入は config 内蔵)
uv run python scripts/build_mos_test.py --sample-dir mos_test/samples --n-utterances 20 \
    --n-evaluators 8 --format webmushra --out-dir mos_test/

# (ここで user が webMUSHRA をセルフホストし評価者に配布、5 段階で採点。結果 CSV を回収)

# フェーズ 2: 回収 CSV を統計集計 (Wilcoxon 一次 + Holm + paired t 補助 + bootstrap CI + 信頼性 + 相関)
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
| 評価者数 | **5〜10 名** (内部・少人数。本来は事前検出力分析で逆算: d=0.3 検出に n≈90 ペア要、§8.1) | docs/milestones.md L595 |
| 論文評価者数 (参考) | 20 名 (ネイティブ英語話者) | docs/paper-summary.md L64 |
| 6 models | GT / GAN-WaveNeXt 2 (T=4) / Diff-WaveNeXt 2 (w/ sub-model) / HiFi-GAN / WaveFit / FastDiff (w/ sub-model) | チケット仕様 / docs/paper-summary.md L98-100 |
| 本命 GAN サンプル | T-M6.1 `checkpoints/gan/best.pt` (T=4) | T-M6.1 §9.1 |
| 本命 Diff サンプル | T-M6.2 `reverse_sample(post_filter=fir, seed=43)` | T-M6.2 §9.1 (T-M7.1 へ) |
| seed (匿名化 / シャッフル / bootstrap) | 43 (deterministic、M4/M6 と統一) | T-M4.1 / T-M6.2 |
| CI 信頼水準 | 95% **bootstrap (BCa)** が既定、t 分布は補助 | §6.1 (ordinal + 小 n) |
| 検定 (一次判定) | **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`) | 対応あり順序尺度 (同一評価者が全 model 採点する paired 設計) |
| 検定 (補助) | paired t-test (`scipy.stats.ttest_rel`) | 論文 (mean MOS) 対比の補助。interval 近似は厳密には不正 |
| 多重比較補正 | **Holm-Bonferroni** (15 pairwise) | 未補正 p と併記必須 (§6.1) |
| 評価者間信頼性 | **ICC(2,k) / Krippendorff's α** | 再現性の一級指標。α<0.6 で序列判定保留 (§6.1) |
| 相関 | **Spearman 主** / Pearson 補助 (vs UTMOS/NISQA) | MOS は ordinal、主観 ↔ 客観整合 |
| Acceptance (相対序列) | GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub) | docs/milestones.md L599-600 |

### 2.4 アルゴリズム / 処理フロー

1. **(前提) M6 完了確認**: T-M6.1 `checkpoints/gan/best.pt` + T-M6.2 4 sub-model + `post_filter/fir.npy` が揃い、客観評価 (`eval_results/*.json`) が論文相対整合済であること。M7 は任意なので user が実施判断 (§9.3)
2. **比較対象サンプルの準備** (§6.1 の判断): HiFi-GAN / WaveFit / FastDiff の 20 utterance 分の合成を **公式 checkpoint 推論** または **自前訓練 (M6.3 ablation の副産物)** で用意し、本命 2 系列 (T-M6.1/T-M6.2) + GT と合わせ `mos_test/samples/` に 6 model × 20 utt = 120 wav を集約。**全 model で同一の sox norm -3dB 正規化** を適用し音量 bias を排除
3. **質問票生成** (`scripts/build_mos_test.py`、Claude Code): model → 匿名 code (A-F) ランダム割当、各評価者ごとに 120 sample の提示順ランダム化、`--format webmushra` で config を出力 (主軸、§8.1)。**CC BY 4.0 帰属 (LibriTTS-R 出典・ライセンス・改変有無) + インフォームドコンセント + データ削除フロー** を config 説明文に注入 (§6.1)。匿名 mapping + schema_version + seed + git_sha を `code_map.json` に保存 (被験者非公開)
4. **(user 操作) 倫理審査 + 評価者手配**: 各実施機関の規程に従いインフォームドコンセントを取得 (README L130)。**EU 評価者を含むクラウドソーシングなら GDPR 対象** (同意の法的根拠・越境移転・撤回時のデータ削除フロー、§6.1)。内部メンバー 5〜10 名を手配し、可能ならヘッドホン + 静音環境で評価を依頼
5. **(user 操作) 評価実施 + CSV 回収**: webMUSHRA で 5 段階採点。結果を `responses.csv` (evaluator_id, sample_id, score, ...) として **漏洩しない経路** で Claude Code に提供 (メール/クラウド共有経路自体が個人データ漏洩リスク、§6.1)
6. **統計集計** (`mos_analysis.analyze`、Claude Code): schema_version 検証 + 匿名 code を復号 → per-model 平均 MOS + bootstrap (BCa) 95% CI、全 model pairwise の **Wilcoxon signed-rank (一次) + Holm 補正後 p + paired t-test (補助)**、**評価者間信頼性 (ICC(2,k)/Krippendorff's α)**、`mos_results/mos.json` に永続化
7. **論文 MOS 比較 + 客観相関** (`eval/report.py`、Claude Code): 論文の MOS 序列 (paper-summary L98) と対比表を生成 (絶対値でなく相対序列、順位ベースを正)、`eval_results/*.json` の UTMOS/NISQA per-utterance と MOS の **Spearman (主) / Pearson (補助)** 相関を算出
8. **Acceptance 判定**: GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub) を **点推定の順序 + bootstrap CI 重なり** で判定 (少人数のため有意差は努力目標、§6.1)。**α<0.6 なら序列判定を保留** (安全弁)
9. **完了報告**: 変更ファイル / 通過 acceptance / 論文 MOS 比較表 / 客観相関 / 既知懸念 (少人数の検出力・環境統制) を user に提示。M7 が再現実装の最終成果

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `mos_analysis.py` (検定/CI/相関) + `build_mos_test.py` (質問票生成/匿名化) 実装 + テスト記述、`eval/report.py` への MOS 比較表追加 | general-purpose |
| Reviewer | 1 | 統計手法の妥当性 (一次判定が Wilcoxon signed-rank で paired 設計に正しいか、Mann-Whitney 誤用が排されているか、bootstrap CI、Holm 補正、ICC/α)、匿名化/シャッフルの bias 排除、論文 MOS 序列との整合確認、**CC BY 4.0 帰属表示 + 倫理審査要件 + GDPR/個人情報保護の明記確認** | general-purpose |
| Analyst | 1 | 回収 CSV (user 提供) での集計実行 + 論文比較表 + 客観相関の算出 + Acceptance 判定 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **N/A** (M7 は本チケット 1 つのみ)
- 並列実行する場合の最大並列数: 1
- **本チケットは user 操作 (倫理審査・評価者手配・CSV 回収) を挟むため、Claude Code の作業は「生成フェーズ」と「集計フェーズ」に分断される** (§2.4)。生成フェーズ完了後は user の評価完了を待つ非同期タスク

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/build_mos_test.py`: 120 sample (20 utt × 6 model) **webMUSHRA config 生成** (主軸、§8.1)、**model 名匿名化 (A-F ランダム)** + **評価者ごとの sample 順序ランダム化** + **CC BY 4.0 帰属 / インフォームドコンセント / データ削除フローの説明文注入**、`code_map.json` (匿名 mapping + schema_version + seed + git_sha) 保存。自前 web app / Google Forms は deprecate 候補
- `src/wavenext2/eval/mos_analysis.py`: 評価結果 CSV → **Wilcoxon signed-rank (一次) + Holm 補正 + paired t-test (補助)** + **bootstrap (BCa) 95% CI** + **評価者間信頼性 (ICC(2,k)/Krippendorff's α)** + **UTMOS/NISQA との Spearman (主)/Pearson (補助) 相関**、`MOSResult` dataclass + `mos_results/*.json` 永続化 (schema_version 検証付き)
- `eval/report.py` への **論文 MOS 比較表** + **客観相関表** の追加 (M6 一本化済レポート生成器を流用)
- 本命 2 系列の deterministic サンプル取得 (T-M6.1 `best.pt` / T-M6.2 `reverse_sample(seed=43)`)
- Acceptance 判定 (相対序列 + bootstrap CI 重なり、α<0.6 で保留): GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub)
- `tests/test_mos_analysis.py` (合成 CSV で Wilcoxon/Holm/bootstrap CI/信頼性/相関/匿名化往復)
- `.gitignore` への `mos_results/` 登録 (個人データとして別格扱い、§6.3)

### Out of Scope
- **評価者の手配・実評価・スコア入力** → user 操作 (§9.3、Claude Code は評価できない)
- **倫理審査・インフォームドコンセントの取得・被験者保護** → user / 各実施機関の責務 (README L130、Claude Code は要件を質問票に明記するのみ)
- **比較対象 (HiFi-GAN/WaveFit/FastDiff) の訓練本体** → 公式 checkpoint 推論 or T-M6.3 ablation の副産物を流用 (本チケットは合成済みサンプルを受けるだけ。§6.1 で調達方針を判断)
- **客観指標 (UTMOS/NISQA/MCD/log F0 RMSE) の計算本体** → T-M4.1/T-M4.2 (本チケットは `eval_results/*.json` を読んで相関を取るのみ)
- **CMOS / AB preference の本実装** → §8.1 代替案 (MOS が論文準拠。少人数なら CMOS を主・ACR を従に格上げ検討)。MUSHRA は webMUSHRA に config 出力する形で In Scope
- **大規模クラウドソーシング (MTurk / Prolific) + attention check (trap sample / gold standard) の本実装** → §8.1 (内部少人数が本チケットの前提)
- **論文と同規模 (20 名) の評価** → 内部少人数 (5〜10 名) が前提、規模差は §6.1 で明示

### Deliverable
- ファイル: `src/wavenext2/eval/mos_analysis.py`, `scripts/build_mos_test.py`, `tests/test_mos_analysis.py`
- 成果物: `mos_test/` (質問票 + `code_map.json`)、`mos_results/mos.json` (集計 + 検定結果)、論文 MOS 比較表 + 客観相関表 (`eval/report.py` 出力)
- 関数 / クラス: `analyze(csv_path, code_to_model, objective_json, *, expected_schema_version) -> MOSResult`, `MOSResult` dataclass, `build_questionnaire(...)`, `wilcoxon_signed_rank` (一次) / `paired_t_test` (補助) / `holm_correct` / `bootstrap_ci` / `inter_rater_reliability` / `correlate_with_objective`
- ドキュメント差分: `docs/milestones.md` §M7.1、`docs/tickets/index.md`、§8.3 に MOS 結果 (論文対比 / 相関 / 検出力所感) を追記

## 5. テスト項目

> **本チケットの実評価は user (人間の評価者) が行うため、Claude Code 側のテストは「質問票生成」と「集計ロジック」を合成データで検証する**。MOS スコア自体は CI に乗らない (人間依存)。

### 5.1 Unit テスト (`tests/test_mos_analysis.py`)
- [ ] `test_build_120_samples` — `build_questionnaire(n_utterances=20, models=<6>)` が **正確に 120 sample** を生成する (20 × 6)
- [ ] `test_anonymize_roundtrip` — model 名が匿名 code (A-F) に置換され、`code_map.json` で復号すると真名に戻る (被験者向け出力に真名が混入しない)
- [ ] `test_shuffle_per_evaluator` — 各評価者で sample 提示順が異なり (順序ランダム化)、同一 seed では再現する (deterministic)
- [ ] `test_wilcoxon_signed_rank` — 合成 CSV (model A > B が既知、paired) で `wilcoxon_signed_rank` が正しい符号の `median_diff` と妥当な p 値を返す (`scipy.stats.wilcoxon` 整合、一次判定)
- [ ] `test_paired_t_test_auxiliary` — 同合成 CSV で `paired_t_test` が Wilcoxon と整合する向きの p 値を返す (論文対比の補助、`scipy.stats.ttest_rel`)
- [ ] `test_holm_correction` — 15 pairwise の合成 p 値で Holm 補正後 p が未補正以上かつ単調、有意判定が縮むことを検証
- [ ] `test_bootstrap_ci_95` — 既知分布の合成スコアで bootstrap (BCa) 95% CI が妥当範囲に収まり、seed 固定で再現する (ordinal + 小 n で t 近似に依存しない)
- [ ] `test_inter_rater_reliability` — 一致度が高い/低い合成スコアで ICC(2,k) / Krippendorff's α がそれぞれ高/低を返し、α<0.6 で保留フラグが立つ
- [ ] `test_correlation_with_objective` — MOS と UTMOS が相関する合成データで Spearman (主)/Pearson (補助) が高相関を返し、無相関データで ≈ 0 を返す
- [ ] `test_acceptance_ordering` — GAN(T=4) > HiFi-GAN / Diff(w/sub) > FastDiff(w/sub) の合成 CSV で Acceptance 判定が pass、逆順で fail
- [ ] `test_mos_result_json_roundtrip` — `MOSResult.to_json` → `from_json` で per_model / pairwise / reliability / correlation / schema_version が往復一致 (`mos_results/*.json` 永続化)
- [ ] `test_load_csv_schema_version` — `load_mos_csv` が schema_version 一致で正規化し、version 不一致で raise する。webMUSHRA wide-form / long-form の両 CSV を同一 DataFrame に正規化
- [ ] `test_missing_scores_handled` — 一部 sample 未採点 (欠損) の評価者がいても NaN セーフに集計 (Wilcoxon は共通 sample のみ、bootstrap CI は有効 n で)

### 5.2 e2e / 結合テスト
- [ ] **生成 → (ダミー回答) → 集計** の一周: `build_mos_test.py` で生成した質問票に対し合成回答 CSV を作り、`analyze` が `MOSResult` を返し `eval/report.py` が比較表を出す (人間評価をダミーで代替した API 動作確認)
- [ ] **本命サンプル取得**: T-M6.1 `best.pt` / T-M6.2 `reverse_sample(seed=43)` で 20 utt の wav が生成され `sample_dir` に揃う (deterministic 再生成可)
- [ ] **客観 JSON 連携**: `eval_results/gan_full.json` (T-M6.1) / `eval_results/diff_full_postfilter.json` (T-M6.2) の per-utterance UTMOS/NISQA を読んで MOS と相関が計算される

### 5.3 Acceptance criteria (`docs/milestones.md` §M7.1 より転記)
- [ ] GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN
- [ ] Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)

### 5.4 検証項目 (本チケット独自、チケット仕様 §5 より)
- [ ] **120 sample の質問票が正しく生成される** (sample のシャッフル、model 名匿名化) — `test_build_120_samples` / `test_anonymize_roundtrip` / `test_shuffle_per_evaluator`
- [ ] **CSV 入力で Wilcoxon signed-rank (一次) + paired t-test (補助) + Holm 補正が計算される** — `test_wilcoxon_signed_rank` / `test_paired_t_test_auxiliary` / `test_holm_correction`
- [ ] **bootstrap (BCa) 95% CI が計算される** — `test_bootstrap_ci_95`
- [ ] **評価者間信頼性 (ICC/α) が計算され α<0.6 で保留フラグ** — `test_inter_rater_reliability`
- [ ] **論文 MOS との比較表** — `eval/report.py --paper-compare` が paper-summary L98 の序列と対比表を出す (相対序列で整合、順位ベースを正)
- [ ] **MOS と UTMOS/NISQA (客観) の相関係数 (Spearman 主)** — `test_correlation_with_objective` + 実 `eval_results/*.json` 連携

## 6. 懸念事項

> **本チケットは人間の評価者が必須で、Claude Code は質問票生成と集計のみ**。最大の懸念は (a) **LibriTTS-R (CC BY 4.0) の帰属表示義務 (倫理・最優先)**、(b) **統計検定の誤用 (Wilcoxon でなく Mann-Whitney は paired 設計に誤用)**、(c) 倫理審査・被験者保護・GDPR、(d) 少人数の統計的検出力、(e) 比較対象サンプルの調達。

### 6.1 技術的リスク

#### 【最優先・倫理】LibriTTS-R (CC BY 4.0) の帰属表示義務
- LibriTTS-R は **CC BY 4.0** ライセンス。評価 web app で **GT 音声と合成音声を公開配信** する際、CC BY 4.0 の **attribution (出典・ライセンス・改変有無) 表示が義務**。とくに **合成音声 (vocoder 出力) は LibriTTS-R の「改変物 (derivative/adapted)」に当たるため、改変した旨の明記が必須**
- **緩和**: ① `build_mos_test.py` が webMUSHRA config / Forms 説明文に **LibriTTS-R の出典・CC BY 4.0 ライセンス・改変有無の帰属表記を必ず注入** (テンプレート固定)、② GT も合成音も同一 attribution ブロックを表示、③ 帰属表記の文面を report に記録
- **検知**: 被験者向け出力 (webMUSHRA config / Forms 説明文) に **LibriTTS-R 帰属表記が無ければ fail**。grep で "LibriTTS-R" + "CC BY 4.0" + 改変有無の記載を確認
- **責務境界**: Claude Code は帰属表記を出力に組み込むまで。配信先・運用での表示確認は user 責務 (§9.3)

#### 【CRITICAL・統計】検定の誤用 — Mann-Whitney でなく Wilcoxon signed-rank
- 当初案の `mann_whitney` (Mann-Whitney U) は **独立 2 標本検定**。本テストは **同一評価者が全 model を採点する paired 設計** なので、独立検定の使用は **統計的誤用** (評価者内の対応関係を捨ててしまい検出力も失う)
- **正しい手法**: 対応あり順序尺度には **Wilcoxon signed-rank (`scipy.stats.wilcoxon`)** が一次判定。`paired t-test` は 1-5 を interval 扱いする慣例的近似で論文 (mean MOS) 対比の **補助** に留める。一次判定はあくまで順位ベース (Wilcoxon)
- **緩和**: ① `mos_analysis.py` の一次検定を `wilcoxon_signed_rank` に差し替え (Mann-Whitney は撤去)、② paired t-test は補助列として残し論文 mean MOS と対比、③ §2.2 / §2.3 / 検定の出典を全て Wilcoxon 主・t-test 補助で統一
- **検知**: `mannwhitneyu` 呼び出しが残っていれば fail。一次判定 p が `scipy.stats.wilcoxon` 由来であることを `test_wilcoxon_signed_rank` で確認

#### 【最優先】被験者保護・倫理審査 (README L130)
- 主観評価は **人間を被験者とする実験**であり、各実施機関 (大学 / 企業) の倫理規程・IRB の対象になりうる。README L130 が「被験者保護・倫理審査は各実施機関の規程に従うこと」と明記
- **緩和**: ① 質問票冒頭に **インフォームドコンセント** (目的・所要時間・データの匿名化・撤回権・連絡先) を必須項目として `build_mos_test.py` が自動挿入、② 個人を特定する情報を収集しない (evaluator_id は無作為 ID)、③ user に「実施前に所属機関の倫理審査要否を確認する」ことを完了報告で明示
- **責務境界**: Claude Code は **要件を質問票に組み込む** ことまで。審査の申請・承認・被験者対応は **user の責務** (§9.3)。Claude Code が倫理審査を代行することはできない
- **検知**: 質問票にインフォームドコンセント節が無い / 個人情報項目が混入していれば fail

#### 【重要・法令】GDPR / 個人情報保護法 (越境移転・撤回後の削除フロー)
- クラウドソーシング等で **EU 在住の評価者** を含むと **GDPR の対象**。日本国内なら個人情報保護法。評価スコアは匿名でも、デバイス申告やタイムスタンプ等と組み合わせると個人データになりうる
- **論点**: ① 同意取得の **法的根拠** (consent)、② データの **越境移転** (評価者 → user → Claude Code 経路)、③ **撤回権の実装** — 既に「撤回権」は質問票に記載予定だが、**撤回時に実際にどの evaluator_id のレコードをどう削除するか** の運用フローが未定義。撤回権は「記載」だけでなく「削除の実行手順」まで定義が必要
- **緩和**: ① `code_map.json` / `mos_results/*.json` を **evaluator_id で引いて該当レコードを物理削除** する手順を README / 完了報告に明記、② データ最小化 (収集項目を最小に、§8.2)、③ 越境移転の同意を consent 文面に含める。実際の削除実行・法令適合判断は user 責務 (§9.3)
- **検知**: 撤回フローが「権利の記載」のみで「削除手順」が未定義なら不十分。完了報告で削除対象 (どの ID・どのファイル) を具体化

#### 【重要・統計】MOS の ordinal 性 — 順位ベースを正、t-test は補助
- MOS の 1-5 は **順序尺度 (ordinal)**。等間隔 (1↔2 と 4↔5 が同価値) の保証はないため、**interval 扱いして mean / t-test を取るのは慣例的近似であり厳密には不正**
- **緩和**: ① 一次判定を順位ベース (Wilcoxon signed-rank)、② mean MOS / paired t-test は論文 (mean MOS 報告) との対比の補助に明確に格下げ、③ 点推定も median を併記、④ CI は bootstrap (BCa) を既定 (正規/t 近似に依存しない、後述)
- **検知**: report で一次判定が順位ベースか、t-test が「補助」と明記されているか

#### 【重要】少人数 (5〜10 名) で統計的検出力が足りない (論文は 20 名)
- 論文は 20 名だが本チケットは内部 5〜10 名。MOS の評価者間分散は大きく、**n=5〜10 では model 間の有意差が出ない (検出力不足)** 公算が高い
- **緩和**: ① **有意差 (p<0.05) を Acceptance の必須にせず、点推定の順序 + bootstrap (BCa) 95% CI の重なり** で判定 (CI が大きく重なっても点推定で GAN ≥ HiFi-GAN なら pass 寄り)、② **paired 設計 + Wilcoxon signed-rank** (同一評価者が全 model 採点) で評価者間分散を相殺し検出力を上げる、③ 1 評価者あたり 120 sample 全採点で per-pair の n を稼ぐ、④ 検出力不足を §8.3 / 完了報告に明記し「相対傾向の参考値」と位置づける、⑤ **本来は事前検出力分析で評価者数を逆算** すべき (論文 MOS 差 0.1〜0.3 は小さく、d=0.3 検出に n≈90 ペア要。5〜10 名は power<0.2 で恣意的、§8.1)
- **検知**: pairwise p 値が軒並み >0.05 でも順序が論文と整合すれば pass 寄り。bootstrap CI 幅と n を report に必ず併記
- **再評価トリガー**: 評価者数確定時 (§8.1)。20 名以上確保できるなら有意差まで要求、クラウドソーシングで大規模化を検討。少人数のままなら **CMOS を主・ACR を従に格上げ** を検討 (§8.1)

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
- **緩和**: **自系列内の model 間順序関係** を合否主軸とし、論文 MOS の絶対値一致は努力目標。paired 設計 + 同一サンプル集合で評価者の採点癖を相殺。一次判定は Wilcoxon (順位ベース)、ACR mean は慣例的近似で t-test は論文対比の補助。`mos_results/*.json` に評価環境・評価者数を記録。これは M4 客観評価 (T-M4.1 §8.2) の「相対比較主軸」と完全に整合する哲学
- **検知**: 論文 MOS との絶対差が大きくても相対序列が整合すれば pass 寄り

#### 【重要・統計】評価者間信頼性 (ICC / Krippendorff's α) を一級指標に
- 相関だけでなく **評価者間の一致度 (inter-rater reliability)** が低いと、序列そのものが評価者の偶然のばらつきに過ぎない可能性がある。少人数だと一致度が低くなりやすい
- **緩和**: ① **ICC(2,k) もしくは Krippendorff's α (順序)** を「再現性の証拠」として相関と並ぶ一級指標に格上げし `MOSResult.reliability` に格納・report に明示、② **α<0.6 なら評価者間一致が不十分として序列判定自体を保留** する安全弁を設ける (Acceptance を pass 扱いしない)
- **検知**: report に信頼性指標が無い / α<0.6 なのに序列を断定していれば fail (`test_inter_rater_reliability`)

#### その他リスク
- **GT が最高 MOS にならない異常**: 通常 GT (真の録音) が最高 MOS のはず。GT が vocoder 出力より低いと評価設計のバグ (匿名化ミス / サンプル取り違え / 正規化不整合) を疑う。GT を sanity anchor として report で確認
- **サンプル間の音量差による bias**: model ごとに出力音量が違うと音量で品質を誤判断される → **全 model を同一 sox norm -3dB** で正規化 (T-M2.1 / val と同じ)。GT も同様
- **CSV フォーマットの揺れ (webMUSHRA vs Forms)**: 回収 CSV が wide-form (1 行 = 1 評価者 × 120 列) か long-form かで parser が変わる → `load_mos_csv` が両対応、`code_map.json` で復号。列名揺れは build 時に固定の sample_id (eval 側 dataclass の権威定義、§8.2) を埋め込んで吸収
- **欠損・離脱評価者**: 全 sample を採点しない評価者がいると Wilcoxon の対が崩れる → 共通採点 sample のみで paired、bootstrap CI は有効 n で計算、欠損率を report に記載
- **scipy バージョン依存の検定 API**: `wilcoxon` の `zero_method` / `mode` や `ttest_rel` の `nan_policy`、`bootstrap` の `method="BCa"` を明示。相関は `spearmanr` を主に使用
- **多重比較 (補正必須に格上げ)**: 6 model の全 pairwise = 15 検定で多重比較問題 (偽陽性増)。「未補正注記」では弱いため、**Holm 補正後 p を未補正 p と併記必須** (`holm_correct`)。少人数で有意差を主張しない方針でも補正後 p は report に出す
- **mos_results は個人データとして別格扱い**: 被験者スコア (匿名でも) を含むため `mos_results/` / `code_map.json` は `.gitignore` で commit しない。ただし `eval_results/` (著作物なし) と **同一 `.gitignore` 方針で括らず**、**保持期間・削除責任・アクセス制御を伴う個人データ** として扱う (§8.2)
- **CSV 授受経路の個人データ漏洩**: 「user が CSV を Claude Code に渡す」経路 (メール / クラウド共有) **自体** が個人データ漏洩リスク。匿名化済 (逆引き不能 ID のみ) の CSV を渡す、共有リンクは期限付き、を user 側手順として明記 (§9.3)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md`: MOS 関連の項目は無い (論文の MOS 手法は標準的な 5 段階 ACR を想定)。本チケットで確定する判断:
  - **検定手法**: **Wilcoxon signed-rank を一次判定** (MOS は paired 設計の順序尺度)、paired t-test は論文 (mean MOS) 対比の補助。Mann-Whitney は独立検定で paired に誤用のため不採用 (§6.1)。15 pairwise は Holm 補正後 p を併記
  - **CI**: 95%・**bootstrap (BCa)** が既定 (ordinal + 小 n)、t 分布は補助
  - **評価者間信頼性**: ICC(2,k) / Krippendorff's α を一級指標に、α<0.6 で序列判定保留
  - **Acceptance 判定基準**: 少人数のため有意差でなく **点推定の順序 + bootstrap CI 重なり** (§6.1)、α<0.6 なら保留
  - **比較対象サンプルの調達**: 公式 checkpoint 優先、LibriTTS-R 適合性確認、無ければ自前訓練 (§6.1)
- 論文が MOS で 0.5 刻みを使ったか整数かは不明 → **整数 5 段階 ACR** を採用 (最も標準的)

### 6.3 他チケットとの整合性
- **T-M6.1 (GAN フル訓練)** から受領: `checkpoints/gan/best.pt` (T=4) を本命 GAN サンプルの生成元に使う (§9.1)。`evaluate()` で得た `eval_results/gan_full.json` の UTMOS/NISQA を相関の客観側に使う
- **T-M6.2 (Diff フル訓練)** から受領: `reverse_sample(model, mel, seed=43, post_filter=fir)` で deterministic に Diff (w/ sub-model) サンプルを生成 (T-M6.2 §9.1 が T-M7.1 へ明記)。`eval_results/diff_full_postfilter.json` を相関に使う
- **T-M4.1 (evaluate facade)** と整合: **相対比較主軸** の哲学 (T-M4.1 §8.2) を MOS にも適用。客観 (M4) ↔ 主観 (M7) の相関で再現性を多角的に検証
- **T-M4.2 (UTMOS/NISQA)** から受領: UTMOS/NISQA の per-utterance 値 (`eval_results/*.json`) を MOS との相関の客観側に使う
- **`eval/report.py` (M6 で一本化)** を流用: 論文 Table 対比レポート生成器に MOS 比較表 + 相関表を追加 (GAN/Diff/ablation と同一フォーマット、M6 フェーズレビュー昇格)
- **`.gitignore`**: `mos_results/` + `mos_test/code_map.json` を除外。ただし **`eval_results/` (M4、著作物なし) と同一方針で括らず**、個人データとして保持期間・削除責任・アクセス制御を伴う別格扱い (§8.2)
- **eval/ がスキーマ権威**: `sample_id` フォーマットは **eval 側 dataclass で一元定義** し、`scripts/build_mos_test.py` / `mos_analysis.py` はそれを import (**eval → scripts 単方向依存**)。M4 `eval_results/*.json` と一貫させる (§8.2)
- **CSV スキーマ version 管理**: `code_map.json` / 回収 CSV に **`schema_version` + `seed` + `git_sha`** を埋め込み、`load_mos_csv` が version を検証 (不一致は raise)。後続が CSV を解釈する際の互換性を保証 (§9.1)
- **evaluator_id 匿名化の責務境界**: evaluator 実名 ↔ 匿名 ID の **逆引き表は user 側に残し**、Claude Code は **逆引き不能な ID のみ受領** する。撤回・削除要求時も user が逆引きして該当 ID を特定 (§9.3)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/milestones.md` §M7.1 / `docs/paper-summary.md` の MOS 手法 (5 段階・20 utt・6 model) と整合
- [ ] Acceptance criteria 全項目クリア (§5.3): GAN(T=4) MOS ≥ HiFi-GAN、Diff(w/sub) MOS ≥ FastDiff(w/sub)
- [ ] 120 sample が正しく生成され、**model 名匿名化 + sample 順序ランダム化** が機能 (被験者向け出力に真名が漏れない)
- [ ] **Wilcoxon signed-rank が一次判定**で `scipy.stats.wilcoxon` と整合、**Mann-Whitney の誤用が撤去**され、paired t-test は補助、Holm 補正後 p 併記、bootstrap (BCa) CI が合成データで検証済 (§6.1)
- [ ] **評価者間信頼性 (ICC/Krippendorff's α)** が算出され、α<0.6 で序列判定保留の安全弁が機能 (§6.1)
- [ ] **少人数の検出力不足を明示** し、有意差でなく点推定順序 + bootstrap CI 重なりで判定 (§6.1)、report に CI 幅と n を併記。事前検出力分析の所見を記録
- [ ] **LibriTTS-R (CC BY 4.0) 帰属表示** (出典・ライセンス・改変有無) が被験者向け出力 (webMUSHRA config / Forms 説明文) に注入されている (§6.1、最優先・倫理)
- [ ] **倫理審査・インフォームドコンセント・GDPR/個人情報保護** (撤回時の削除手順含む) が質問票に組み込まれ、審査・削除実行は user 責務と明記 (README L130、§9.3)
- [ ] 比較対象 (HiFi-GAN/WaveFit/FastDiff) の checkpoint 素性 (公式 or 自前、LibriTTS-R 適合性) が report に記録され、論文絶対値でなく相対比較に徹している
- [ ] 全 model が同一 sox norm -3dB 正規化済 (音量 bias 排除)、GT が sanity anchor として最高 MOS 付近
- [ ] MOS と UTMOS/NISQA の相関 (**Spearman 主** / Pearson 補助) が算出され、主観 ↔ 客観の整合度が示されている
- [ ] `mos_results/` / `code_map.json` が `.gitignore` 済 (個人データとして `eval_results/` と別格扱い、保持期間・削除責任明記、§8.2)、`code_map.json`/CSV に `schema_version`+`seed`+`git_sha`、`MOSResult` JSON 往復一致
- [ ] M7 phase review の結果が §8.3 に記録
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] 参考実装 (webMUSHRA 等) をコピーしていない (質問票生成 / 集計ロジックは一から記述、webMUSHRA は config 出力先として利用するのみ)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M7 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採否 / 理由 | 再評価トリガー |
|---|---|---|---|---|
| **webMUSHRA (既存 OSS) に全面委譲** ✅採用昇格 | 匿名化・順序ランダム化・結果 CSV 出力を内蔵する枯れた OSS、UI 実装ゼロ、自前 web app の保守が消える | MUSHRA 寄りの UI、セルフホストに技術知識が要る | **採用**: `build_mos_test.py` を **webMUSHRA config 生成器 1 本に絞る** (web/Forms/webMUSHRA の 3 形式出力は保守過剰)。自前 web app は **deprecate 候補** (§2.2) | **webMUSHRA で UI 要件を満たせないとき** |
| **少人数なら CMOS を主・ACR を従に格上げ** ✅採用検討昇格 | 提案 vs 各 baseline 直接対比で **評価者の絶対採点癖を構造的に排除**、小 n で最も現実的 (論文差 0.1〜0.3 は小、d=0.3 検出に n≈90 ペア要) | ペア提示で組合せ増、論文は absolute MOS | **格上げ検討**: 5〜10 名は power<0.2。論文再現の一次は ACR だが、少人数では CMOS を主に切替えるのが現実的 (§6.1) | **少人数 ACR で検出力が決定的に不足し論文序列を判定できないとき** (§6.1) |
| **bootstrap (BCa) CI を既定に** ✅採用 | ordinal + 小 n では正規/t 近似より頑健 | 計算コスト (resample)、seed 管理 | **採用**: CI 既定を bootstrap (BCa)、t-CI は補助 (§2.3 / §6.1) | (常時適用) |
| **Spearman 相関を主に** ✅採用 | MOS は ordinal なので Pearson より Spearman が適切 | — | **採用**: 客観相関は Spearman 主・Pearson 補助 (§2.3) | (常時適用) |
| **事前検出力分析で評価者数を決定** ✅検討追加 | 効果量から逆算し恣意性を排除 (d=0.3 で n≈90 ペア) | 効果量の事前仮定が必要 | **検討**: 現状 5〜10 名は恣意的。効果量から必要 n を逆算し評価者数の根拠にする (§6.1) | **評価者数を確定する前** |
| **静的ホスティング選定 (GitHub Pages / Forms / webMUSHRA セルフホスト)** ✅検討追加 | 配信方式を要件で選べる、音声配布 URL を保護可 | 方式ごとに長短 (Forms は音声埋込が貧弱) | **検討**: GitHub Pages 静的 + CSV 回収 / Forms / webMUSHRA セルフホストを比較。**音声配布 URL の保護** (署名付き URL / Basic 認証 / 期限切れ) を併せて設計 (CC BY 帰属と両立) | **配信方式・URL 保護を確定するとき** |
| **クラウドソーシング (Prolific) + attention check** ✅検討追加 | n を 20→100+ に増やし検出力・一般化を確保 | 課金、環境非統制が悪化、GDPR 対象化 (§6.1) | **検討**: Prolific 等で大規模化する場合、**trap sample (既知劣化音声に低評価強制)・最低再生時間・gold standard** で品質管理、**最低賃金相当の報酬設計** | **評価者を 20 名以上確保 / 論文同規模で再現したいとき** |
| **倫理審査を最初から組み込む (IRB テンプレート同梱)** | 実施機関の審査をスムーズ化 | リポジトリに機関依存の審査書式を持つのは過剰 | インフォームドコンセントの質問票挿入までは In Scope、審査書式は機関依存で user 責務 (§6.1) | **特定機関での反復実施が決まり様式を固定したいとき** |
| **2-stage: 客観 (M4) で足切り → 主観 (M7) は僅差ペアのみ** | 主観評価のコストを僅差 model 間に集中 | 設計が複雑、論文は全 model MOS | 論文は全 6 model を MOS。本チケットも全 model | **評価者負担が過大で僅差ペアに絞りたいとき** |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M7.1 は質問票生成 + 統計集計 + 相関 + 論文対比を抱えるが、いずれも CSV ↔ 統計の薄い層で、`eval/report.py` (M6) と `eval_results/*.json` (M4) を流用するため size=M が妥当。実評価 (人間) を含まないので実装規模は中程度
- **責務境界 (Claude Code は評価できない)**: 本チケットの核は「**人間にしかできない評価を Claude Code が前後で支援する**」こと。質問票生成 (前) と統計集計 (後) は Claude Code、実評価・倫理審査・評価者手配は user。この分断は M6 (GPU は user、起動・監視は Claude Code) と同じ「人間必須作業の周辺を自動化する」思想
- **相対比較主軸の一貫性**: MOS の絶対値を諦め相対序列を主軸にするのは M4 客観評価 (T-M4.1 §8.2) と完全に同じ哲学。客観 (M4) と主観 (M7) を **相関** で結ぶことで、片方だけでは不十分な再現性検証を多角化する。これが M7 を「任意だが価値ある」最終段にしている
- **少人数の限界を隠さない**: 論文 20 名に対し 5〜10 名は検出力不足が構造的。これを「参考値」と明示し、有意差を偽装しない (**Holm 補正後 p も併記**)。再現実装の誠実さとして重要
- **MOS 絶対値は近似、順位ベースを正**: ACR mean は慣例的近似。一次判定は **Wilcoxon (順位ベース)**、paired t-test は論文 (mean MOS) 対比の補助という優先順位を哲学として固定する。検定の選択 (Wilcoxon vs Mann-Whitney) は paired 設計か独立かで決まり、本テストは paired なので Wilcoxon が必然
- **データガバナンス (data minimization)**: 匿名でも (デバイス申告 + スコアパターン) の組合せで再識別リスクがある。**収集項目を最小化** することを設計原則 (data minimization) に据え、必須でない属性 (詳細なデバイス情報・自由記述等) は集めない
- **`mos_results/` は個人データとして別格扱い**: `eval_results/` (M4、著作物なし) と **同一 `.gitignore` 方針で括らない**。被験者スコアは匿名でも個人データであり、**保持期間・削除責任・アクセス制御** を伴う管理対象。`.gitignore` 除外は最低限で、本質はライフサイクル管理
- **eval/ がスキーマ権威 (単方向依存)**: `sample_id` フォーマットは **eval 側 dataclass で一元定義** し、`scripts/` はそれを import する (**eval → scripts 単方向依存**)。M4 `eval_results/*.json` とフォーマットを一貫させ、スキーマの二重定義を避ける。`schema_version` を CSV/`code_map.json` に持たせ後方互換を検証
- **最終成果としての位置づけ**: M7 は再現実装パイプラインの最終段。論文 MOS 比較表 + 客観相関が「論文を再現できた」ことの主観面の証明になる。ただし M7 が任意である以上、M6 客観評価で再現性が担保されていれば M7 未実施でもプロジェクトは成立する

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- **少人数 (5〜10 名) で論文序列を判定できたか**: TBD (有意差 or 点推定順序 + CI 重なりで判定)
- **GAN(T=4) MOS ≥ HiFi-GAN が再現したか**: TBD
- **Diff(w/sub) MOS ≥ FastDiff(w/sub) が再現したか**: TBD
- **MOS と UTMOS/NISQA の相関の強さ**: TBD (Spearman 主 / Pearson 補助、客観 ↔ 主観の整合度)
- **評価者間信頼性 (ICC/Krippendorff's α)**: TBD (α<0.6 で序列判定を保留したか)
- **比較対象 checkpoint の素性が公平だったか**: TBD (公式 vs 自前、LibriTTS-R 適合性)
- **環境非統制 / 検出力不足の影響度**: TBD (事前検出力分析の所見、CMOS への切替判断)
- 想定外: TBD
- 教訓: TBD
- 再評価トリガー: **評価者数確定時** (5〜10 名 → 20 名以上に増やせるなら有意差要求 / MUSHRA / クラウドソーシングへ、§8.1)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **本チケットは M7 (最終段) で後続チケットなし**。**最終成果は 論文との MOS 比較表 + 客観 (M4) ↔ 主観 (M7) の Spearman 相関で再現性を多角検証** すること (主観面の品質証明)
- **最終成果物の所在**:
  - `mos_results/mos.json`: per-model MOS (mean/median/**bootstrap 95% CI**) + pairwise 検定 (**Wilcoxon 一次 + Holm 補正後 p** + paired t 補助) + **評価者間信頼性 (ICC/α)** + UTMOS/NISQA **Spearman** 相関 + `schema_version`
  - `eval/report.py --paper-compare --mos`: 論文 MOS 序列との比較表 + 客観相関表
- **CSV スキーマ version 管理**: `code_map.json`/CSV に `schema_version` + `seed` + `git_sha` を持たせ、`load_mos_csv` が version を検証 (§6.3)。後続が CSV を再解釈する際の互換性保証
- **evaluator_id 匿名化の責務境界**: 逆引き表 (実名 ↔ ID) は **user 側に残り**、Claude Code は **逆引き不能な ID のみ受領**。撤回・削除も user が逆引きして実行 (§9.3)
- **客観 (M4) と主観 (M7) の相関による多角検証**: M4 の `eval_results/*.json` (UTMOS/NISQA/MCD/log F0 RMSE) と M7 の MOS を **Spearman 相関** で結び、「客観指標が良い ⇒ 主観も良い」が成立するかで再現性を裏付ける。これがプロジェクト全体の品質証明の締め
- **再現性の総括**: M6 (客観・自動) + M7 (主観・人間) の両輪で論文再現を主張。M7 が任意のため、M6 単独でも再現性は概ね担保される位置づけを明記

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M7.1 Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M7.1 ステータス更新 + M7 フェーズレビューログ行を更新
  - [ ] (該当時) `docs/paper-summary.md` に内部 MOS の再現結果 (相対序列 / 客観相関) を「再現実装の所見」として追記
  - [ ] §8.3 に MOS 結果 (論文対比 / 相関 / 検出力所感) を実測値付きで追記

### 9.3 ユーザー操作 (必須)
- **LibriTTS-R (CC BY 4.0) 帰属表示** (最優先・倫理): 評価 web app で GT/合成音声を配信する際、**LibriTTS-R の出典・CC BY 4.0 ライセンス・改変有無の帰属表記** が表示されていることを配信先で確認 (合成音声は「改変物」、§6.1)。Claude Code は config 説明文に帰属を注入するが、配信運用での表示確認は user 責務
- **倫理審査・被験者保護**: 各実施機関の規程に従い、実施前に倫理審査の要否を確認 (README L130)。インフォームドコンセントの取得・被験者対応は user 責務 (Claude Code は質問票に同意節を組み込むのみ、§6.1)
- **GDPR / 個人情報保護 + 撤回時の削除実行**: EU 評価者を含むなら GDPR 対象 (同意の法的根拠・越境移転、§6.1)。**撤回要求時に逆引き表で evaluator_id を特定し、`mos_results/*.json` / `code_map.json` の該当レコードを物理削除する** のは user 責務 (Claude Code は逆引き不能、§6.3)
- **評価者の手配**: 内部メンバー 5〜10 名 (or 外部クラウドソーシング) を手配。可能ならヘッドホン + 静音環境を依頼 (環境統制、§6.1)
- **評価実施 + CSV 提供 (漏洩しない経路)**: webMUSHRA で 5 段階採点を実施し、結果 CSV を Claude Code に提供 (Claude Code は人間評価を代行できない)。**CSV 授受経路 (メール/クラウド共有) 自体が個人データ漏洩リスク** のため、逆引き不能 ID のみの匿名化済 CSV を、期限付き共有リンク等で渡す (§6.1)
- **比較対象サンプルの素性確認**: HiFi-GAN/WaveFit/FastDiff の公式 checkpoint を使う場合、LibriTTS-R 24kHz 適合性を user が確認 (不適合なら自前訓練、§6.1)

### 9.4 Open question として残ったもの
- 少人数 (5〜10 名) で論文の MOS 序列を統計的に判定できるか → 検出力不足前提で点推定順序 + CI 重なりで判定、不足なら §8.1 (MUSHRA / クラウドソーシング) を評価者数確定時に再判断
- 比較対象 (HiFi-GAN/WaveFit/FastDiff) の公式 checkpoint が LibriTTS-R 24kHz に適合するか → 不明なら自前訓練 (T-M6.3 副産物) に切替、checkpoint 素性を report に記録
- MOS の絶対値が論文に届くか / 相対序列の再現で十分か → 相対比較主軸 (§6.1、T-M4.1 §8.2 と整合)
- 環境非統制 (内部評価) が MOS にどれだけ影響するか → 相対序列で吸収する前提、report に環境を記録
- LibriTTS-R (CC BY 4.0) 帰属の文面・GDPR 撤回時の削除手順 → 本チケットで方針確定 (config 注入 + user 側削除実行、§6.1/§9.3)。実運用での表示・削除実行は user 責務
- `docs/open-questions.md` への追記要否: MOS 手法 (Wilcoxon 一次 / bootstrap CI / Holm / 信頼性 / 匿名化 / 倫理 / CC BY 帰属) は本チケットで確定。acceptance 未達時のみ再オープン (現時点では不要)
