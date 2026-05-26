---
id: T-M4.1
title: 客観評価スクリプト (MCD + log F0 RMSE)
milestone: M4
phase: M4
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.1]
blocks: [T-M5.1, T-M5.2]
related_docs:
  - docs/milestones.md#m41-客観評価スクリプト-srcwavenext2evalcompute_metricspy
  - docs/training.md
---

# T-M4.1: 客観評価スクリプト (MCD + log F0 RMSE)

> **マイルストーン**: [M4](../milestones.md#m4-評価インフラ-作業量-medium3-サブタスク) / **サブタスク**: [M4.1](../milestones.md#m41-客観評価スクリプト-srcwavenext2evalcompute_metricspy)
> **依存**: [T-M0.1](T-M0.1-python-env.md) / **後続**: [T-M5.1](T-M5.1-gan-1epoch.md), [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
論文 §4.2 (`docs/training.md` §5.2) の客観評価指標のうち、スペクトル類似度 **MCD (Mel-Cepstral Distortion)** と韻律精度 **log F0 RMSE** を `src/wavenext2/eval/compute_metrics.py` に実装する。GAN / Diff いずれの vocoder 出力に対しても、GT 波形との対で 1 GPU で LibriTTS-R test-clean 全 4,824 utterance を 30 分以内に処理できるバッチ評価パイプラインを提供する。これにより M5 smoke / M6 本格訓練の品質ゲート (論文 Table 1〜3 との対比) を自動化する。

UTMOS / NISQA (NN ベース MOS 推定) は T-M4.2、RTF (速度) は T-M4.3 で別途実装する。本チケットは MCD / log F0 RMSE の **波形ペア → スカラー指標** 計算を核に持つが、**加えて M4 全体の統一 eval facade `src/wavenext2/eval/runner.py` の受け皿を本チケットで作る** (フェーズレビューで 3 視点全員が指摘した最大の構造的空白)。M4.1 (MCD / log F0 RMSE)、M4.2 (UTMOS / NISQA)、M4.3 (RTF) を **1 entry point `evaluate(model, dataset, metrics=[...], post_filter=None) -> EvalResult` で束ねる**ことで、M5 / M6 caller が「合成 → ペア化 → 各指標」のグルーコードを毎回書くのを排除する (§2.2 / §8.1)。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M4.1 Acceptance を内包):
- [ ] `src/wavenext2/eval/compute_metrics.py` に `compute_mcd` / `compute_log_f0_rmse` が実装され、`from wavenext2.eval.compute_metrics import compute_mcd, compute_log_f0_rmse` で import 可能
- [ ] `compute_mcd(y_true, y_pred, sr=24000) -> float` が同一音声で `≈ 0`、ノイズ付加で値が増加する
- [ ] `compute_log_f0_rmse(y_true, y_pred, sr=24000) -> float` が pyworld で F0 抽出 → voiced frame のみで log RMSE を算出し、同一音声で `≈ 0`
- [ ] MCD バックエンドが `pymcd` (cp313 wheel あり) または `mel-cepstral-distance` (fallback) のいずれかに確定し、`pyproject.toml` の dependencies に反映 (T-M0.1 §6.1 から引き継ぎ)
- [ ] バッチ評価 API `evaluate_dataset(pairs, sr, num_workers) -> (summary_dict, per_utt_df)` が test-clean 全 4,824 utterance を **1 GPU で 30 分以内** に処理 (並列化込み)
- [ ] post-filter switch (`reverse_sample(model, mel, *, seed, post_filter=fir | None)`、T-M3.4 連携) で Diff の apply/no-apply の MCD / log F0 RMSE を比較できる評価 driver を提供
- [ ] validation metric として T-M2.3 の `MultiResolutionSTFTLoss` を eval でも再利用 (`from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss`)
- [ ] 無音 / 極短 utterance での F0 抽出失敗が `NaN` 伝播なく安全に skip される (edge case handling)
- [ ] **統一 eval facade `src/wavenext2/eval/runner.py` に `evaluate(model, dataset, metrics=[...], post_filter=None) -> EvalResult` を実装** (M4.1/M4.2/M4.3 を 1 entry point で束ねる)。`EvalResult` は `{summary: dict, per_utterance: list[dict] or DataFrame}` 構造
- [ ] `evaluate_dataset` の戻り値を `(summary_dict, per_utt_df)` tuple とし (§8.1 昇格)、per-utterance 生値を保持
- [ ] **`eval_results/*.json` 永続化を実装**: per-utterance 値を JSON dump し、論文 Table 対比時の再計算を回避。`.gitignore` に `eval_results/` 登録 (T-M4.2 と非対称を解消)
- [ ] `tests/test_compute_metrics.py` / `tests/test_eval_runner.py` の全テスト pass
- [ ] `docs/milestones.md` §M4.1 Acceptance 2 項目クリア、`docs/tickets/index.md` の T-M4.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル

- 新規:
  - `src/wavenext2/eval/compute_metrics.py` (本実装: `compute_mcd`, `compute_log_f0_rmse`, `evaluate_dataset`, MCD バックエンド抽象)
  - `src/wavenext2/eval/runner.py` (統一 eval facade: `evaluate(model, dataset, metrics, post_filter) -> EvalResult`, `EvalResult` dataclass, `eval_results/*.json` 永続化)
  - `tests/test_compute_metrics.py` (Unit + edge case + 速度回帰 + fallback 整合)
  - `tests/test_eval_runner.py` (facade 動作 / `EvalResult` schema / 永続化往復)
- 編集:
  - `src/wavenext2/eval/__init__.py` (`__all__` に `compute_mcd`, `compute_log_f0_rmse`, `evaluate_dataset`, `evaluate`, `EvalResult` を追加)
  - `pyproject.toml` (MCD バックエンドを `pymcd` / `mel-cepstral-distance` のいずれかに確定。T-M0.1 §6.1 の「M4 着手時に再確認」を実行。`uv add` で更新し `uv.lock` の diff を commit)
  - `.gitignore` (`eval_results/` を追加。既存行があれば確認のみ。T-M0.2 への申し送り)
  - `docs/milestones.md` §M4.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M4.1 ステータス更新

### 2.2 主要構造

#### import 形式 (後続 T-M5.1 / T-M5.2 で確定)

```python
# 低レベル指標計算器
from wavenext2.eval.compute_metrics import (
    compute_mcd,
    compute_log_f0_rmse,
    evaluate_dataset,
)
# 統一 eval facade (M5/M6 caller はこちらを使う)
from wavenext2.eval.runner import evaluate, EvalResult
```

#### `compute_metrics.py`

```python
"""compute_metrics.py — 客観評価指標 (MCD + log F0 RMSE).

MCD          : Mel-Cepstral Distortion。pymcd (DTW alignment) を第一候補、
               cp313 wheel 未提供時は mel-cepstral-distance に fallback (§6.1)。
log F0 RMSE  : pyworld DIO+StoneMask で F0 抽出 → voiced frame のみ log RMSE。

入力はすべて mono float32 波形 [-1, 1] (numpy or torch.Tensor)。
参考: docs/training.md §5.2 / docs/milestones.md §M4.1
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SR_DEFAULT = 24000
F0_FLOOR = 71.0      # pyworld DIO デフォルト (Hz)
F0_CEIL = 800.0      # pyworld DIO デフォルト (Hz)
FRAME_PERIOD = 5.0   # ms, pyworld デフォルト
MCD_N_MFCC = 24      # MCD 標準 (c1..c24、c0 は energy で除外)
EPS = 1e-8

_AudioLike = "np.ndarray | torch.Tensor"


def _to_numpy_mono(y) -> np.ndarray:
    """torch.Tensor / np.ndarray → mono float64 numpy 1-D に正規化."""
    ...


def compute_mcd(
    y_true,
    y_pred,
    sr: int = SR_DEFAULT,
    backend: Literal["pymcd", "mel-cepstral-distance"] | None = None,
) -> float:
    """Mel-Cepstral Distortion (dB) を DTW alignment ありで計算.

    backend=None で利用可能なバックエンドを自動選択 (pymcd → fallback)。
    同一音声で ≈ 0、歪み増で単調増加。
    """
    ...


def compute_log_f0_rmse(
    y_true,
    y_pred,
    sr: int = SR_DEFAULT,
    f0_floor: float = F0_FLOOR,
    f0_ceil: float = F0_CEIL,
    frame_period: float = FRAME_PERIOD,
) -> float:
    """log F0 RMSE (両者 voiced のフレームのみで評価).

    1. pyworld.dio + stonemask で F0 系列を抽出 (両波形)
    2. DTW or 長さ揃えでフレーム対応付け
    3. voiced ∧ voiced のフレームのみ残し log(f0) の RMSE
    4. voiced 共通フレームが 0 のとき NaN を返さず np.nan を返却 (集計側で除外)
    """
    ...


def evaluate_dataset(
    pairs: "list[tuple[PathLike, PathLike]]",
    sr: int = SR_DEFAULT,
    num_workers: int = 8,
    metrics: "tuple[str, ...]" = ("mcd", "log_f0_rmse", "mrstft_sc", "mrstft_mag"),
) -> "tuple[dict[str, float], pd.DataFrame]":
    """(gt_path, synth_path) のリストを並列評価して (summary, per_utt) を返す.

    - multiprocessing.Pool (num_workers) で utterance を分散 (DTW は CPU-bound)
    - mrstft_* は T-M2.3 MultiResolutionSTFTLoss を batch GPU で別途集計
    - NaN (F0 抽出失敗 utterance) は nanmean で除外、除外件数も返す
    - 戻り値: (summary_dict, per_utt_df)。summary は集約値 + n/n_skipped、
      per_utt_df は utterance ごとの生値 (§8.1 で flat dict から tuple へ昇格)。
      per-utterance 生値は runner.evaluate() が eval_results/*.json に永続化。
    """
    ...
```

#### 統一 eval facade `runner.py` (M4 全体の受け皿、§8.1 採用昇格)

```python
"""runner.py — M4 全体の統一 eval facade.

M4.1 (MCD / log F0 RMSE), M4.2 (UTMOS / NISQA), M4.3 (RTF) を 1 entry point で束ね、
M5 / M6 caller が「合成 → ペア化 → 各指標」のグルーコードを毎回書くのを排除する。
各指標の実装は M4.1/M4.2/M4.3 が backend として提供 (本 facade は dispatch のみ)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EvalResult:
    """統一 eval 戻り値. summary は集約値、per_utterance は生値."""

    summary: dict[str, float]                 # {"{metric}_mean", "{metric}_std", "n", "n_skipped", ...}
    per_utterance: "list[dict] | pd.DataFrame" # utterance ごとの生値 (論文 Table 対比の再計算回避)

    def to_json(self, path: "str | Path") -> None:
        """eval_results/*.json へ per-utterance + summary を dump (再計算回避)."""
        ...

    @classmethod
    def from_json(cls, path: "str | Path") -> "EvalResult":
        ...


def evaluate(
    model,
    dataset,
    metrics: "list[str]" = ("mcd", "log_f0_rmse", "mrstft"),
    post_filter: "np.ndarray | None" = None,
    *,
    seed: int = 43,
    save_to: "str | Path | None" = None,
) -> EvalResult:
    """統一 eval entry point.

    1. model で dataset 全 utterance を合成 (Diff は reverse_sample(post_filter=...) を呼ぶ)
    2. GT とペア化
    3. metrics を dispatch:
       - "mcd" / "log_f0_rmse" / "mrstft*" → compute_metrics.evaluate_dataset (M4.1)
       - "utmos" / "nisqa"                 → T-M4.2 backend
       - "rtf"                             → T-M4.3 backend
    4. summary (prefix 規約 {metric}_mean / {metric}_std / n / n_skipped) と
       per_utterance をまとめた EvalResult を返す
    5. save_to 指定時 (または既定 eval_results/) に JSON 永続化
    """
    ...
```

#### MCD バックエンド抽象 (cp313 リスク吸収)

```python
def _resolve_mcd_backend(name: str | None):
    """利用可能な MCD 実装を解決. pymcd 優先、無ければ mel-cepstral-distance.

    どちらも import できなければ ImportError を投げ、§8.1 の自前 DTW+MFCC へ誘導。
    両バックエンドは同じ MFCC order (c1..c24) で揃え、数値差を許容範囲内 (§5) に。
    """
    ...
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `sr` | 24000 Hz | docs/training.md §1.1 |
| MCD バックエンド (第一候補) | `pymcd` (`Calculate_MCD(mode="dtw")`) | docs/milestones.md §M4.1 |
| MCD バックエンド (fallback) | `mel-cepstral-distance>=0.0.4` | T-M0.1 §6.1 (cp313 wheel リスク) |
| MCD MFCC order | c1..c24 (c0 energy 除外) | MCD 標準慣例 (pymcd デフォルト) |
| MCD alignment | DTW (`mode="dtw"`) | docs/training.md §5.2 (発話間 alignment 必須) |
| F0 抽出器 | `pyworld.dio` + `pyworld.stonemask` (refine) | T-M0.1 dependencies |
| `f0_floor` | 71.0 Hz | pyworld DIO デフォルト |
| `f0_ceil` | 800.0 Hz | pyworld DIO デフォルト |
| `frame_period` | 5.0 ms | pyworld デフォルト |
| voiced/unvoiced 判定 | `f0 > 0` (両者 voiced のフレームのみ評価) | §6 韻律評価慣例 |
| log F0 の log | natural log (`np.log`) | 慣例 (MCD と同じ自然対数系) |
| `num_workers` | 8 (CPU コア数で調整) | DTW CPU-bound 並列化 |
| MR-STFT eval | T-M2.3 既定 (`n_ffts=[512,1024,2048]`, `win=[360,900,1800]`, `hop=[80,150,300]`, `eps=1e-5`) | T-M2.3 §2.3 再利用 |
| 処理時間予算 | test-clean 4,824 utt を 1 GPU + N CPU worker で **30 分以内** | docs/milestones.md §M4.1 Acceptance |

### 2.4 アルゴリズム / 処理フロー

#### `compute_mcd(y_true, y_pred, sr)`
1. 両波形を mono float64 に正規化 (`_to_numpy_mono`)
2. バックエンド解決 (`_resolve_mcd_backend`): pymcd 利用可なら `Calculate_MCD(MCD_mode="dtw")`、不可なら `mel-cepstral-distance`
3. pymcd は path 入力 API のため、tmp WAV に書き出すか in-memory API を使う (§6.1 で I/O オーバーヘッド検知)
4. DTW alignment 後の mel-cepstrum L2 距離 (dB scale) を返す
5. 同一波形で `≈ 0` (DTW 完全 alignment) を保証

#### `compute_log_f0_rmse(y_true, y_pred, sr)`
1. 両波形を mono float64 に正規化
2. `f0_t, t = pyworld.dio(y, sr, f0_floor, f0_ceil, frame_period)` → `pyworld.stonemask` で refine (両波形)
3. フレーム数を min に揃える (または DTW path で対応付け、§8.1 で簡易版は長さ揃えを採用)
4. `voiced = (f0_true > 0) & (f0_pred > 0)` のマスクを作成
5. voiced フレームが 0 件なら `np.nan` を返す (無音 utterance)
6. `rmse = sqrt(mean((log(f0_true[voiced]) - log(f0_pred[voiced]))**2))` を返す

#### `evaluate_dataset(pairs, sr, num_workers, metrics)`
1. `mcd` / `log_f0_rmse` を `multiprocessing.Pool(num_workers)` で utterance 並列 (DTW / F0 抽出は CPU-bound、GIL 回避のため process pool)
2. `mrstft_*` を要求された場合は GPU で batch 処理 (T-M2.3 `MultiResolutionSTFTLoss`、padding して mini-batch 化)
3. 各 utterance の結果を per-utterance DataFrame に格納し、`np.nanmean` / `np.nanstd` で集約 (NaN = F0 失敗を除外)
4. summary 側は prefix 規約 (§6.3) に従い `{"mcd_mean", "mcd_std", "log_f0_rmse_mean", "log_f0_rmse_std", "mrstft_sc_mean", "mrstft_mag_mean", "n", "n_skipped"}` を、生値側は per-utterance DataFrame を返す (`(summary_dict, per_utt_df)` tuple)
5. **決定性**: `np.nanmean` の順序依存と BLAS 非決定性を避けるため、per-utterance 結果は pairs 順に固定 sort してから集約。同一入力 → 同一出力を保証 (§6.1)

#### `evaluate(model, dataset, metrics, post_filter)` (統一 facade、`runner.py`)
1. `model` で `dataset` 全 utterance を合成 (Diff は `reverse_sample(model, mel, seed=43, post_filter=post_filter)`、GAN は forward)
2. 合成波形と GT をペア化
3. metrics を backend へ dispatch: `mcd`/`log_f0_rmse`/`mrstft*` → `compute_metrics.evaluate_dataset` (M4.1)、`utmos`/`nisqa` → T-M4.2、`rtf` → T-M4.3
4. 各 backend の `(summary, per_utt)` をマージして `EvalResult` を構築 (summary は prefix 規約で統一)
5. `eval_results/<run_id>.json` に per-utterance + summary を dump (論文 Table 対比時の再計算を回避)。TensorBoard scalar 記録もここで行う所在を明示

#### post-filter switch driver (T-M3.4 連携)
1. Diff 評価時は `evaluate(model, dataset, metrics=["mcd","log_f0_rmse"], post_filter=None)` と `evaluate(..., post_filter=fir)` の両方を呼ぶ (facade 内部で `reverse_sample(post_filter=...)` を switch)
2. 2 つの `EvalResult.summary` を対比し、MCD / log F0 RMSE の差分 (post-filter 改善幅) をレポート

### 2.5 設計上の重要決定

- **統一 eval facade `runner.evaluate()` を本チケットに集約** (§8.1 採用昇格、3 視点全員が指摘): M4.1/M4.2/M4.3 を 1 entry point で束ね、M5/M6 caller のグルーコードを排除。各指標の中身は M4.2/M4.3 が backend として埋めるが、`evaluate(model, dataset, metrics, post_filter) -> EvalResult` の API と dispatch・永続化の枠は本チケットで定義する
- **`compute_metrics` の低レベル関数は「波形ペア → スカラー」に責務限定**: 合成 (reverse_sample / GAN forward) は facade (`runner.evaluate`) が担い、`compute_mcd` / `compute_log_f0_rmse` は GT と合成済み波形の対だけを受ける純粋関数。facade と低レベル計算の 2 層構成で UTMOS/NISQA (T-M4.2)・RTF (T-M4.3) とも責務分離
- **`evaluate_dataset` 戻り値を `(summary_dict, per_utt_df)` tuple に** (§8.1 昇格): 統一 facade と per-utterance 永続化の前提。M6 ablation で per-utterance 分布が要るため flat dict では不足
- **eval 結果の永続化と論文 Table 対比を eval 層の責務に**: `eval_results/*.json` に per-utterance + summary を dump し、M6 で再計算・再発明しない。CSV/JSON dump + TensorBoard scalar 記録の所在を本チケットで明示。`.gitignore` に `eval_results/` 登録 (T-M4.2 と非対称を解消)
- **MCD バックエンドを差し替え可能に抽象化** (§6.1 cp313 リスク): `backend` 引数 + `_resolve_mcd_backend` で `pymcd` / `mel-cepstral-distance` を切り替え。T-M0.1 が `pyproject.toml` の MCD 依存を「M4 着手時に確定」と申し送っているので、本チケットで wheel 提供状況を確認して 1 つに pin する
- **F0 は両者 voiced のフレームのみで RMSE** (§6): unvoiced フレーム (f0=0) を含めると `log(0) = -inf` になり破綻するため、`(f0_true > 0) & (f0_pred > 0)` のマスクで除外。WORLD/論文の韻律評価慣例に合致
- **DTW は CPU-bound のため process pool で並列化**: pymcd の DTW alignment が 4,824 utt で律速 (§6.1)。`multiprocessing.Pool` で utterance を分散し 30 分予算を死守。MR-STFT のみ GPU batch
- **`MultiResolutionSTFTLoss` を eval metric として T-M2.3 から再利用** (T-M2.3 §9.1 / T-M3.4 と同方針): validation の MR-STFT (sc/mag) を `compute_metrics` でも算出し、checkpoint 選択 (WaveFit-PT は MR-STFT 最小で best) と整合
- **post-filter apply/no-apply の比較を `post_filter` 引数 switch で実現** (T-M3.4 §2.5 採用設計): 評価 driver が `reverse_sample(post_filter=fir)` と `None` を呼び分けるだけで apply/no-apply 両系列を出せる。`apply_post_filter` を caller で chain しない
- **NaN セーフな集約**: F0 抽出失敗 utterance は個別関数で `np.nan` を返し、`evaluate_dataset` は `np.nanmean` で平均 + skip 件数を別途報告。1 utterance の失敗で全体が `NaN` 汚染されない

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `compute_metrics.py` 本実装 (MCD / F0 / バッチ driver) + テスト記述、`pyproject.toml` の MCD 依存確定 | general-purpose |
| Reviewer | 1 | 論文 §5.2 整合確認、pymcd vs mel-cepstral-distance の数値差検証、voiced マスク / NaN セーフ性レビュー、コピー流用していないか | general-purpose |
| Tester | 1 | `uv run pytest tests/test_compute_metrics.py -v` 実行 + 速度回帰 (test-clean サブセットで 30 分予算の外挿検証) | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M0.1 のみに依存、T-M4.2 / T-M4.3 とは独立)
- 並列実行する場合の最大並列数: 3 (M4 の 3 チケット (T-M4.1 / T-M4.2 / T-M4.3) を別エージェントに分担可能)
- 後続 T-M5.1 / T-M5.2 (smoke) とは順次。

## 4. 提供範囲 (Scope)

### In Scope
- **統一 eval facade `evaluate(model, dataset, metrics=[...], post_filter=None) -> EvalResult`** (`runner.py`、M4.1/M4.2/M4.3 を束ねる 1 entry point。M4.2/M4.3 backend の dispatch 枠を含む)
- **`EvalResult` dataclass** (`{summary: dict, per_utterance: list[dict] or DataFrame}`) + `eval_results/*.json` 永続化 (`to_json` / `from_json`)
- `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float` (pymcd / fallback 抽象、DTW alignment)
- `compute_log_f0_rmse(y_true, y_pred, sr=24000, ...) -> float` (pyworld F0 抽出 + voiced マスク + NaN セーフ)
- `evaluate_dataset(pairs, sr, num_workers, metrics) -> (summary_dict, per_utt_df)` (並列バッチ評価、MR-STFT を GPU batch で同時集計、§8.1 で tuple 昇格)
- MCD バックエンド抽象 (`_resolve_mcd_backend`) と `pyproject.toml` 依存確定
- post-filter apply/no-apply の MCD / log F0 RMSE 比較 driver (T-M3.4 連携、`reverse_sample(post_filter=...)` 呼び分け)
- T-M2.3 `MultiResolutionSTFTLoss` の eval metric 再利用
- `.gitignore` への `eval_results/` 登録
- `tests/test_compute_metrics.py` (同一音声 ≈0 / 歪み増加 / 30 分予算 / F0 edge case / fallback 数値整合)
- `tests/test_eval_runner.py` (facade 動作 / `EvalResult` schema / JSON 永続化往復)

### Out of Scope
- UTMOS / NISQA の**指標計算本体** → **T-M4.2** (本チケットは `evaluate()` facade から呼ぶ dispatch 枠のみ定義、`utmos`/`nisqa` の実装は T-M4.2)
- RTF の**速度計測本体** → **T-M4.3** (同上、`rtf` backend は T-M4.3 が実装)
- PESQ / STOI / speaker similarity (resemblyzer) → §8.1 代替案 (M6.3 ablation で追加検討)
- CREPE / RMVPE (NN ベース F0) への置換 → §8.1 代替案
- 合成パイプライン本体 (GAN forward / `reverse_sample`) → T-M2.4 / T-M3.3 の責務、本チケットは波形ペアを受けるだけ
- 主観評価 (MOS) の集計・統計検定 → T-M7.1
- 論文 Table 1〜3 との具体的な数値対比レポート生成 → M6.1 / M6.2 (本チケットは指標計算器のみ)

### Deliverable
- ファイル:
  - `src/wavenext2/eval/compute_metrics.py` (新規)
  - `src/wavenext2/eval/runner.py` (新規、統一 facade)
  - `tests/test_compute_metrics.py` (新規)
  - `tests/test_eval_runner.py` (新規)
- 関数 / クラス:
  - `evaluate(model, dataset, metrics=[...], post_filter=None, *, seed=43, save_to=None) -> EvalResult` (統一 facade)
  - `EvalResult` dataclass (`summary`, `per_utterance`, `to_json`, `from_json`)
  - `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float`
  - `compute_log_f0_rmse(y_true, y_pred, sr=24000, f0_floor, f0_ceil, frame_period) -> float`
  - `evaluate_dataset(pairs, sr=24000, num_workers=8, metrics=...) -> tuple[dict[str, float], pd.DataFrame]`
  - `_resolve_mcd_backend(name) -> backend`, `_to_numpy_mono(y) -> np.ndarray` (内部)
- ドキュメント差分:
  - `docs/milestones.md` §M4.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M4.1 ステータス更新
  - (該当時) `pyproject.toml` の MCD 依存確定を `docs/open-questions.md` に追記
  - `.gitignore` に `eval_results/` 追加 (T-M0.2 への申し送り済、既存行があれば確認のみ)

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_compute_metrics.py`)

- [ ] `test_mcd_identical_zero` — 同一音声で `compute_mcd(y, y) ≈ 0` (atol=1e-3、DTW 完全 alignment)
- [ ] `test_log_f0_rmse_identical_zero` — 同一音声 (有声含む合成 sine / 実 utterance) で `compute_log_f0_rmse(y, y) ≈ 0`
- [ ] `test_mcd_increases_with_noise` — `y_pred = y + noise` で noise scale を上げると MCD が単調増加
- [ ] `test_log_f0_rmse_increases_with_pitch_shift` — pitch shift / F0 摂動で log F0 RMSE が増加
- [ ] `test_mcd_accepts_torch_and_numpy` — torch.Tensor / np.ndarray 両入力で同値 (相対誤差 1e-5)
- [ ] `test_f0_silent_returns_nan` — 無音 (全 0) utterance で `np.nan` を返す (NaN 伝播せず、例外も投げない)
- [ ] `test_f0_very_short_returns_nan` — 極短 (< 1 frame) utterance で `np.nan` を返す
- [ ] `test_f0_unvoiced_frames_excluded` — voiced フレームのみで RMSE 計算 (unvoiced を混ぜても `-inf` にならない)
- [ ] `test_mcd_backend_fallback_consistency` (`@pytest.mark.slow`) — `pymcd` と `mel-cepstral-distance` の MCD が同じ MFCC order で許容範囲内 (相対誤差 < 5% 目安、両 import 可能な環境のみ)
- [ ] `test_evaluate_dataset_nanmean_skips_failures` — F0 失敗 utterance を含む pairs で `n_skipped` が正、平均が `nanmean` で汚染されない
- [ ] `test_evaluate_dataset_returns_mrstft` — `metrics` に `mrstft_sc`/`mrstft_mag` を含めると T-M2.3 由来の値が summary に入る
- [ ] `test_evaluate_dataset_returns_tuple` — 戻り値が `(summary_dict, per_utt_df)` tuple で per-utterance 行数が pairs 数と一致
- [ ] `test_mcd_mode_consistency` — `adv_dtw` mode で同一音声 MCD ≈ 0 (mode A/B の整合、§8.1)
- [ ] `test_post_filter_tilt` — post-filter 適用で MCD↓・log F0 RMSE 不変を確認 (tilt 補正は spectral、F0 には効かない)

#### 統一 facade テスト (`tests/test_eval_runner.py`)

- [ ] `test_evaluate_facade` — `evaluate(model, dataset, metrics=["mcd", "log_f0_rmse"])` が `EvalResult` を返す (dummy model / 小 dataset で API 動作)
- [ ] `test_eval_result_schema` — `summary` の key 命名が prefix 規約 (`{metric}_mean`, `{metric}_std`, `n`, `n_skipped`) に従う
- [ ] `test_eval_result_json_roundtrip` — `to_json` → `from_json` で summary / per_utterance が往復一致 (`eval_results/*.json` 永続化)
- [ ] `test_evaluate_post_filter_switch` — `post_filter=fir` / `None` で 2 系列の `EvalResult` が出て差分を取れる (T-M3.4 連携)

### 5.2 e2e / 結合テスト

- [ ] LibriTTS-R test-clean のサブセット (例 50 utt) を実 GT で `evaluate_dataset` に通して dict が返る
- [ ] post-filter switch driver: `reverse_sample(model, mel, seed=43, post_filter=fir)` / `None` の 2 系列を評価し、MCD / log F0 RMSE の差分が出る (T-M3.4 連携、未学習 model では値の妥当性は不問・API 動作のみ)
- [ ] **速度回帰**: サブセット N utt の wall time から 4,824 utt を外挿し **30 分以内** に収まる見込みを確認 (本番は M5/M6 で full 実行)

### 5.3 Acceptance criteria (`docs/milestones.md` §M4.1 より転記)
- [ ] 同一音声に対して MCD ≈ 0, log F0 RMSE ≈ 0
- [ ] LibriTTS-R test-clean 全 4,824 utterance を 1 GPU で 30 分以内に処理

### 5.4 追加 acceptance (本チケット独自)
- [ ] `from wavenext2.eval.compute_metrics import compute_mcd, compute_log_f0_rmse, evaluate_dataset` が動作
- [ ] `from wavenext2.eval.runner import evaluate, EvalResult` が動作し、`evaluate(model, dataset, metrics=["mcd","log_f0_rmse"])` が `EvalResult` を返す
- [ ] `EvalResult.summary` の key が prefix 規約 (`{metric}_mean`, `{metric}_std`, `n`, `n_skipped`) に従う
- [ ] `EvalResult.to_json` / `from_json` の往復一致、`eval_results/*.json` が生成され `.gitignore` 済
- [ ] MCD バックエンドが `pyproject.toml` で確定 (pymcd or mel-cepstral-distance)、`uv sync` でエラーなく解決
- [ ] 既知の歪み (ノイズ付加) で MCD が増加することを定量確認
- [ ] F0 抽出失敗 (無音 / 極短) の edge case が `np.nan` で安全に処理される
- [ ] `mel-cepstral-distance` fallback が pymcd と同じ MFCC order で数値整合 (両 import 可能な環境)
- [ ] CPU のみでも全 Unit テストが pass (MR-STFT GPU batch は CUDA marker で分離)
- [ ] 同一入力で `evaluate_dataset` / `evaluate` が同一出力 (BLAS 非決定性 / nanmean 順序依存を排除)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (実装着手前に必ず確認)

- **CRITICAL: `pymcd` の cp313 wheel 未提供リスク** (T-M0.1 §6.1 から引き継ぎ):
  - `uv add pymcd` が cp313 wheel を見つけられず sdist からの local build に落ちる / 失敗する可能性
  - **対策 A** (第一候補): pymcd の PyPI に cp313 wheel があれば採用、`Calculate_MCD(MCD_mode="dtw")` を使う
  - **対策 B** (fallback): `mel-cepstral-distance>=0.0.4` (純 Python 寄りで枯れている) に切替。MFCC order を pymcd と揃える
  - **対策 C** (最終手段): §8.1 の自前 DTW + MFCC 実装 (`librosa.feature.mfcc` + `librosa.sequence.dtw`) で外部 MCD 依存を排除
  - **着手前 MUST DO**: M4 着手時点で `uv add pymcd` を dry-run し wheel 提供状況を確認 → A/B/C のいずれかに確定し `pyproject.toml` を pin

- **CRITICAL: pymcd の DTW alignment が 4,824 utt で 30 分予算を超過するリスク**:
  - DTW は `O(N*M)` で発話長に依存、pymcd は path 入力 (内部で librosa load) のため I/O + DTW で 1 utt 数百 ms〜数秒
  - **対策**: `multiprocessing.Pool(num_workers)` で utterance 並列 (DTW は CPU-bound で GIL を解放しないため thread でなく process)。N コアで線形スケール
  - **検知**: 5.2 速度回帰テストでサブセット外挿、超過時は §8.1 自前 MFCC + 高速 DTW (`fastdtw` / `librosa.sequence.dtw`) に切替、または並列度を上げる
  - **再評価トリガー**: M5 smoke で eval pipeline が律速になった場合 (§8 再評価トリガー)

#### 通常項目

- **pyworld の F0 抽出パラメータ選択** (§問題提起): `frame_period` / `f0_floor` / `f0_ceil` の値で F0 系列が変わる。本チケットは pyworld デフォルト (`frame_period=5.0`, `f0_floor=71`, `f0_ceil=800`) を採用。LibriTTS-R は読み上げ音声なのでデフォルトで概ね妥当だが、低音話者で `f0_floor=71` が高すぎる懸念 → 必要なら `f0_floor=40` に下げる ablation を §8 で検討
- **voiced/unvoiced 判定での log F0 RMSE の扱い**: `(f0_true > 0) & (f0_pred > 0)` の AND マスクで両者 voiced のみ評価 (採用)。代替に GT 基準 (`f0_true > 0` のみ) で pred の unvoiced を 0 埋めする案もあるが `log(0)` 破綻を招くため不採用。voiced 判定の不一致 (VUV error) 自体は本指標では測らない (§8.1 で VUV error 追加を検討)
- **pymcd vs mel-cepstral-distance の数値差** (異なる実装で値が変わる): MFCC の窓・order・liftering・DTW 距離定義が実装で微妙に異なる。両者を **同じ MFCC order (c1..c24)** に揃えても完全一致はしない → `test_mcd_backend_fallback_consistency` は相対誤差 < 5% 目安の緩い許容で pin。論文 Table との対比時はどちらのバックエンドで測ったか `eval_stats` に記録
- **無音 / 極短 utterance での F0 抽出失敗**: LibriTTS-R test-clean に極短セグメントは少ないが、先頭/末尾の silence trimming 由来で voiced 0 件になり得る → `np.nan` 返却 + `nanmean` 集約で対応。skip 件数を report
- **MCD の path 入力 API による tmp WAV I/O オーバーヘッド**: pymcd は file path を取る API が主のため、in-memory 配列を毎回 tmp WAV に書くと I/O が律速になり得る → pymcd の array API があれば優先、無ければ `tmpfs` / RAM disk に書く or §8.1 自前実装に切替
- **長さ不一致 (GAN hop=300 / Diff hop=256 で合成長が GT と異なる)**: F0 は frame 数を min に揃え、MCD は DTW が alignment するため問題は小さいが、極端な長さ差は warning を出す
- **`MultiResolutionSTFTLoss` の eval 再利用時の dtype / device**: T-M2.3 の `window` は `persistent=False` で dtype-aware 再生成 (T-M2.3 §6.1) のため eval で fp32 / GPU batch でも問題なし。長さの異なる utterance を batch 化する際は pad + 同一長 crop に注意
- **multiprocessing の Windows spawn 挙動**: Windows は `spawn` start method で worker が module を再 import するため、`compute_metrics.py` を `if __name__` ガード下で呼ぶか `Pool` 初期化を関数内に閉じる。pyworld / numpy は spawn-safe
- **sample rate 16k vs 24k で MCD / F0 値が変わる**: 論文は測定 sr を明記していない。本チケットは **24 kHz で測定** (合成出力と同じ sr、resample による歪み混入を避ける) と確定し、`eval_stats` / `EvalResult.summary` に `sr` を記録。M6 論文対比時に論文が 16k 測定だった疑いが出たら再評価
- **post-filter の群遅延 (group delay)**: T-M3.4 post-filter が `np.convolve(mode="same")` だと FIR の群遅延で波形が数 sample ずれる。MCD は DTW が吸収するが、log F0 RMSE は frame 対応がずれて過大評価になり得る → post-filter 側を **線形位相 FIR + `scipy.signal.oaconvolve`** にして遅延を補正、または eval 前に遅延分を揃える。`test_post_filter_tilt` で F0 RMSE 不変を確認
- **BLAS 非決定性 / `nanmean` の順序依存**: per-utterance 集約が float 加算順序で僅かに変わると論文 Table 対比の再現性が崩れる → per-utterance 結果を pairs 順に固定 sort してから `np.nanmean` / `np.nanstd`、MR-STFT batch も決定論的に。**同一入力 → 同一出力** を保証する記述をテスト (§5.4) で担保
- **CI に GPU が無い場合の skip 条件**: MR-STFT batch テストは `@pytest.mark.cuda` を付け GPU 無し CI では skip。速度回帰 (サブセット wall time → 4,824 utt 30 分外挿) は CPU でも走るが CI 時間を食うため `@pytest.mark.slow` で nightly のみ、PR CI では実行可否を明示 (デフォルト skip)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目: 評価指標の具体的パラメータ (F0 floor/ceil, MCD order, voiced マスク) は論文に明示がなく **本チケットで確定** する:
  - MCD: c1..c24, DTW alignment (採用)
  - log F0 RMSE: 両者 voiced AND マスク, natural log, pyworld デフォルトパラメータ (採用)
- MCD バックエンドの確定 (pymcd or mel-cepstral-distance) は T-M0.1 §6.1 / §9.3 の open question を本チケットで解消 → 結果を `pyproject.toml` と (必要なら) `docs/open-questions.md` に反映

### 6.3 他チケットとの整合性

- **3 チケット共通の dict schema 非対称を解消 (prefix 規約を M4 で合意)**: 現状 M4.1=`{mcd, log_f0_rmse, n_skipped_f0}`、M4.2=`{utmos_mean, utmos_std, n}`、M4.3=`{rtf_mean, ...}` と mean/std/n 命名・skip 報告が不統一。**本チケットで prefix 規約 `{metric}_mean` / `{metric}_std` / `n` / `n_skipped` を定義し、T-M4.2 / T-M4.3 へ申し送る** (§9.1)。`evaluate()` facade の `EvalResult.summary` はこの規約に従い 3 backend をマージするため、規約が前提になる。本チケットの `evaluate_dataset` も `n_skipped_f0` → `n_skipped`、`n_total` → `n` に統一
- **`eval_results/` の `.gitignore` 非対称を解消**: 永続化を本チケットの責務にするため `.gitignore` に `eval_results/` を追加 (T-M4.2 が既に書いている場合は重複確認のみ)。T-M0.2 へ既存 `eval_results/` 行の有無確認を申し送り (§9)
- **T-M0.1 (Python 環境)** から受領:
  - `pymcd` / `pyworld` / `librosa` が `pyproject.toml` の dependencies に含まれる
  - **本チケットからの逆連絡**: MCD バックエンドを確定 (T-M0.1 §6.1 「dependencies は最終的に pymcd / mel-cepstral-distance のいずれかに確定する (M4 評価インフラ着手時に再確認)」を実行)。fallback 採用時は `uv add mel-cepstral-distance` + `uv remove pymcd` の diff を commit
- **T-M2.3 (Loss)** から受領 (§9.1 申し送りを実行):
  - `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` を eval metric として import (GAN 専用と決めつけない汎用設計済、T-M2.3 §2.5)
  - 既定値 `n_ffts=[512,1024,2048]`, `win=[360,900,1800]`, `hop=[80,150,300]`, `eps=1e-5`
- **T-M3.4 (Post-filter)** から受領 (本チケットが「後続」):
  - `reverse_sample(model, mel, *, seed, post_filter: np.ndarray | None = None)` の引数統合 (T-M3.4 で実装済) を使い、apply/no-apply の MCD / log F0 RMSE を比較
  - `post_filter=None` で no-apply、`post_filter=fir` で apply。`apply_post_filter` を caller で直接 chain しない
- **T-M3.3 (Reverse sampler)** から間接受領:
  - Diff 評価時に `reverse_sample(model, mel, seed=43)` で deterministic 合成 → 本チケットがその出力を GT と対にして評価
- **T-M5.1 / T-M5.2 (smoke)** へ渡す情報:
  - `evaluate_dataset(pairs)` で 1 epoch 後の MCD / log F0 RMSE を取得し品質確認 (smoke の acceptance に使用)
- **T-M2.1 (Dataset)** との整合:
  - GT 波形は val/test の sox `norm -3.0 dB` 正規化済 (docs/training.md §1.2.5)。合成側も同じ正規化前提で MCD/F0 が公平になることを確認

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §5.2 / `docs/milestones.md` §M4.1 の指標定義と整合 (MCD = スペクトル類似度, log F0 RMSE = 韻律精度)
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 全クリア
- [ ] MCD バックエンドが `pyproject.toml` で 1 つに確定し、`uv sync` がエラーなく解決
- [ ] 同一音声で MCD ≈ 0 / log F0 RMSE ≈ 0、ノイズ付加で MCD 増加が定量確認されている
- [ ] log F0 RMSE が voiced ∧ voiced マスクで計算され、unvoiced で `-inf` / `NaN` 汚染しない
- [ ] 無音 / 極短 utterance が `np.nan` で安全に処理され、`evaluate_dataset` が `nanmean` + skip 件数報告
- [ ] post-filter switch (`reverse_sample(post_filter=fir / None)`) で apply/no-apply 比較ができ、群遅延で log F0 RMSE がずれない
- [ ] **統一 facade `evaluate(model, dataset, metrics, post_filter) -> EvalResult` が実装され、M4.2/M4.3 backend を dispatch する枠がある**
- [ ] **`EvalResult.summary` の key が prefix 規約 (`{metric}_mean`, `{metric}_std`, `n`, `n_skipped`) に従い、`eval_results/*.json` 永続化と往復一致**
- [ ] `evaluate_dataset` が `(summary_dict, per_utt_df)` tuple を返す
- [ ] `sr=24000` で測定し `EvalResult.summary` に sr 記録、同一入力 → 同一出力 (決定性) が担保
- [ ] T-M2.3 `MultiResolutionSTFTLoss` を再利用 (重複実装していない)
- [ ] 30 分予算が並列化 (process pool) で達成見込み、速度回帰テストで外挿確認
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 不正入力 (空 pairs, shape mismatch) で適切な例外、F0 失敗は例外でなく `np.nan`
- [ ] **参考実装 (pymcd / ParallelWaveGAN の eval) をコピーしていない**: バックエンドは pip ライブラリ呼び出しに留め、独自 driver は一から記述

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M4) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

> **フェーズ (M4) レビューで採用に昇格した設計** (旧 §8.2 検討事項からの格上げ):
> - **統一 eval facade `runner.evaluate(model, dataset, metrics=[...], post_filter=None) -> EvalResult`** を本チケットで定義 (§2.2 / §2.5)。M4.1/M4.2/M4.3 を 1 entry point で束ね、M5/M6 caller のグルーコードを排除。3 視点全員が指摘した最大の構造的空白の受け皿。
> - **`evaluate_dataset` の戻り値を `(summary_dict, per_utt_df)` tuple に昇格** (旧 §8.2 の「見直し余地」を採用)。統一 API と per-utterance 永続化の前提。
> - **`eval_results/*.json` 永続化を本チケットの責務に** (旧 §8.2 思想を採用)。per-utterance 値を JSON dump し論文 Table 対比時の再計算を回避、`.gitignore` に `eval_results/` 登録 (T-M4.2 と非対称を解消)。

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **MCD を自前 DTW + MFCC で実装** (`librosa.feature.mfcc` + `librosa.sequence.dtw`) | pymcd / mel-cepstral-distance の cp313 wheel 依存を完全排除、I/O オーバーヘッド (tmp WAV) 回避、in-memory 配列で高速 | DTW 実装の正しさ検証が必要、論文 MCD と数値が微妙にずれる可能性 | まず枯れた pip ライブラリ (pymcd) を試す。wheel / 速度問題が出たら本案へ | **§6.1 Critical: pymcd cp313 build 失敗かつ fallback も不可**、または **30 分予算超過** |
| **pymcd `mode` の A/B 比較** (`plain` / `dtw` / `adv_dtw`) | `adv_dtw` (端点固定 + 傾斜制限) は vocoder 評価の事実上の標準で alignment が安定 | mode で MCD 絶対値が変わり論文 Table 対比が難しくなる | `dtw` を既定にしつつ `adv_dtw` を A/B 検討。`plain` は alignment ずれで過大評価のため不採用 | **M6 で論文 MCD と乖離が大きく alignment 起因が疑われるとき** |
| **F0 抽出器を DIO+StoneMask から Harvest に変更** | Harvest は低音話者のオクターブエラーに強い | DIO+StoneMask より重い、論文の WORLD 系想定からは DIO が無難 | まず DIO+StoneMask (軽量・pyworld デフォルト) を採用、オクターブエラーが出れば Harvest へ | **M6 で低音話者の F0 にオクターブエラー多発時** |
| **utterance shard 分割で multi-GPU 並列** (M6 で full 4824 を複数 config 評価) | rank ごとに pairs を slice → 結果 JSON を merge で評価をスケール | 単一 GPU では不要、merge ロジックが要る | M4.1 単体は CPU process pool で 30 分予算を達成見込み。multi-GPU は M6 の full 評価で初めて必要 | **M6 で full 4824 × 複数 config 評価が CPU 並列でも遅いとき** |
| **log F0 RMSE を CREPE / RMVPE (NN ベース F0) で代替** | pyworld DIO より高精度、雑音耐性が高い | GPU 必須・重い、追加依存 (torchcrepe 等)、論文は WORLD 系想定 | pyworld で論文整合性を優先、軽量 | **M6 で DIO の F0 抽出が破綻 (オクターブエラー多発) する場合** |
| **PESQ / STOI を追加** (音質・了解度指標) | 知覚音質の補完指標、文献比較が容易 | 論文 Table に無い指標で対比できない、追加依存 (`pesq`, `pystoi`) | 論文 §5.2 は UTMOS/NISQA/MCD/logF0 の 4 指標、本チケットは MCD/logF0 に集中 | **M6.3 ablation で追加分析が必要なとき** |
| **speaker similarity (resemblyzer) を追加** | 話者性保持の評価 | 論文に無い、d-vector モデル依存 | スコープ外、vocoder は話者非依存タスク | **多話者 generalization 検証が必要なとき (M6 以降)** |
| **MCD/F0 を 1 GPU batch で完結 (DTW を GPU 実装)** | process pool 不要、GPU で一括 | GPU DTW 実装が複雑、pymcd は CPU 前提 | CPU process pool で 30 分予算は達成見込み | **CPU 並列でも予算超過するとき** |
| **VUV (voiced/unvoiced) error を追加指標化** | 韻律評価の補完 (voiced 判定不一致を測る) | 論文 Table に無い、本チケットは log F0 RMSE に集中 | スコープ外 | **M6 で F0 RMSE だけでは品質差が出ないとき** |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M4.1 は MCD + log F0 RMSE + バッチ driver + edge case に加え統一 facade / 永続化を抱えるため size=M〜L 寄り。ただし facade は dispatch + dataclass の薄い層で、指標本体は M4.2/M4.3 が埋めるため M4.1 に集約する方が「合成 → ペア化 → 各指標」の重複を避けられ自然
- **責務境界 (2 層構成)**: 低レベル `compute_metrics` は「波形ペア → スカラー」純粋関数、上位 `runner.evaluate` が合成 (reverse_sample / GAN forward) + dispatch + 永続化を担う。UTMOS/NISQA (T-M4.2)・RTF (T-M4.3) は facade の backend として刺さり、機能境界はきれいに分かれる
- **絶対値比較を諦め相対比較を主軸に**: MCD は backend・MFCC order・DTW mode で系統差が出るため、論文 Table の**絶対値**比較は困難。自系列内の**相対**比較 (GT≈0、GAN < Diff 等の順序関係) を品質ゲートの主軸とする。`docs/paper-summary.md` の論文値は参考程度に留め、再現の合否は相対傾向で判断
- **eval 結果の永続化と論文 Table 対比は eval 層の責務**: M6 で再発明しないよう、本チケットで `eval_results/*.json` (per-utterance + summary) dump + TensorBoard scalar 記録の所在を明示。M6 は永続化された JSON を読んで相対比較するだけにする
- **`evaluate_dataset` 戻り値は `(summary_dict, per_utt_df)` tuple (採用済)**: per-utterance 生値を保持し、M6 ablation の分布分析・JSON 永続化の入力にする (旧「見直し余地」を §8.1 で採用昇格)
- **MR-STFT を eval に含める是非**: validation の checkpoint 選択基準 (WaveFit-PT) と整合させるため含めるが、論文 Table 1〜3 は MR-STFT を主要指標にしていない → `metrics` 引数でオプトイン (デフォルト含めるが外せる)

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

- **インターフェース**:
  - 統一 facade: `evaluate(model, dataset, metrics=["mcd","log_f0_rmse","mrstft"], post_filter=None, *, seed=43, save_to=None) -> EvalResult` (M5/M6 caller の推奨 entry point)
  - `EvalResult` dataclass: `{summary: dict, per_utterance: list[dict] or DataFrame}` + `to_json` / `from_json`
  - `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float`
  - `compute_log_f0_rmse(y_true, y_pred, sr=24000, f0_floor=71, f0_ceil=800, frame_period=5.0) -> float`
  - `evaluate_dataset(pairs: list[tuple[PathLike, PathLike]], sr=24000, num_workers=8, metrics=("mcd","log_f0_rmse","mrstft_sc","mrstft_mag")) -> tuple[dict[str, float], pd.DataFrame]`
  - いずれも torch.Tensor / np.ndarray の mono float32 波形 [-1,1] を受ける
- **戻り値 dict schema (prefix 規約、M4 3 チケット共通)**: summary は `{metric}_mean` / `{metric}_std` / `n` / `n_skipped` の prefix 規約に従う。**T-M4.2 / T-M4.3 もこの規約に統一**し、`evaluate()` facade の backend として呼ばれる
- **設定値**:
  - MCD バックエンド (確定したもの: pymcd or mel-cepstral-distance) を `pyproject.toml` に pin、測定 `sr=24000` を `EvalResult.summary` に記録
  - F0: pyworld デフォルト (`frame_period=5.0`, `f0_floor=71`, `f0_ceil=800`)、voiced AND マスク, natural log
- **注意事項 (後続が踏みそうな罠)**:
  - F0 抽出失敗は **例外でなく `np.nan`** を返す。集計は必ず `np.nanmean`、skip 件数も確認
  - MCD は DTW alignment 込みなので長さ違いの波形でも測れるが、log F0 RMSE は frame 数を min に揃える
  - Windows では multiprocessing が `spawn` のため `evaluate_dataset` を `if __name__ == "__main__"` 下から呼ぶ (T-M5.1 / T-M5.2 / M6 評価 driver)
  - post-filter 比較は `reverse_sample(post_filter=fir / None)` の呼び分けで行う (T-M3.4 統合済 API、`apply_post_filter` を chain しない)
  - 論文 Table 対比は**相対比較主軸** (絶対値は backend / MFCC order / DTW mode で系統差)。`eval_results/*.json` を読んで自系列内の順序関係で合否判定

- **個別チケットへの申し送り**:
  - **T-M4.2 / T-M4.3 へ**: 戻り値 dict schema を統一 (`{metric}_mean`, `{metric}_std`, `n`, `n_skipped`)。`evaluate()` facade の backend として `metrics=["utmos"]` (M4.2) / `["rtf"]` (M4.3) で dispatch される設計に合わせる
  - **T-M5.1 / T-M5.2 (smoke) へ**: 1 epoch 後の品質確認は `evaluate(model, val_dataset, metrics=["mcd","log_f0_rmse","mrstft"])` の **1 entry point 呼び出し**で行う (グルーコード不要)
  - **T-M6.1 / T-M6.2 (本格訓練) へ**: `evaluate()` で full 4824 utt 評価、論文 Table 対比は**相対比較主軸**、結果は `eval_results/*.json` 永続化 (M6 で再計算しない)。full × 複数 config が遅ければ utterance shard multi-GPU (§8.1) を検討
  - **T-M3.4 から**: `reverse_sample(post_filter=fir)` で apply/no-apply の switch を受ける (facade が内部で呼び分け)
  - **T-M0.2 への申し送り**: `.gitignore` に `eval_results/` 追加 (既存 `eval_results/` 行があれば確認のみ、無ければ本チケットで追加)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M4.1 の Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M4.1 ステータスを `📝 pending` → `✅ completed`、M4 進捗サマリ更新
  - [ ] (該当時) `pyproject.toml` の MCD 依存確定を `docs/open-questions.md` の T-M0.1 §6.1 open question 解消として追記
  - [ ] `.gitignore` に `eval_results/` を追加 (T-M0.2 への申し送り、既存行があれば確認のみ)
  - [ ] (該当時) `docs/tickets/index.md` のフェーズレビューログ M4 行を更新

### 9.3 Open question として残ったもの
- pymcd vs mel-cepstral-distance のどちらが論文 Table の MCD 値に近いか — backend / MFCC order / DTW mode で系統差が出るため**絶対値の一致は期待せず相対比較を主軸**にする (§8.2)。M6 で乖離が大きければバックエンド / mode 再検討 (`eval_stats` にどちらで測ったか記録)
- pymcd の `mode` (`plain` / `dtw` / `adv_dtw`) のどれが論文・vocoder 評価慣例に最も近いか — `dtw` 既定、`adv_dtw` を A/B 検討 (§8.1)。alignment 起因の乖離が疑われたら切替
- pyworld DIO の F0 抽出パラメータ (`f0_floor=71`) が LibriTTS-R 全話者で妥当か — 低音話者でオクターブエラーが出れば `f0_floor` 調整、Harvest、or CREPE/RMVPE 切替 (§8.1)
- 30 分予算が実 4,824 utt + 実 GPU で本当に達成できるか — full 実行は M5/M6、超過時は §8.1 自前高速 DTW / 並列度増 / utterance shard multi-GPU へ
- 測定 `sr` が論文と一致するか (16k vs 24k) — 本チケットは 24k 測定、論文が 16k だった疑いが出たら再評価 (§6.1)
