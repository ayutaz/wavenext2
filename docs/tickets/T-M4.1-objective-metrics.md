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

UTMOS / NISQA (NN ベース MOS 推定) は T-M4.2、RTF (速度) は T-M4.3 で別途実装する。本チケットは **波形ペア → スカラー指標** の純粋計算に責務を限定する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M4.1 Acceptance を内包):
- [ ] `src/wavenext2/eval/compute_metrics.py` に `compute_mcd` / `compute_log_f0_rmse` が実装され、`from wavenext2.eval.compute_metrics import compute_mcd, compute_log_f0_rmse` で import 可能
- [ ] `compute_mcd(y_true, y_pred, sr=24000) -> float` が同一音声で `≈ 0`、ノイズ付加で値が増加する
- [ ] `compute_log_f0_rmse(y_true, y_pred, sr=24000) -> float` が pyworld で F0 抽出 → voiced frame のみで log RMSE を算出し、同一音声で `≈ 0`
- [ ] MCD バックエンドが `pymcd` (cp313 wheel あり) または `mel-cepstral-distance` (fallback) のいずれかに確定し、`pyproject.toml` の dependencies に反映 (T-M0.1 §6.1 から引き継ぎ)
- [ ] バッチ評価 API `evaluate_dataset(pairs, sr, num_workers) -> dict[str, float]` が test-clean 全 4,824 utterance を **1 GPU で 30 分以内** に処理 (並列化込み)
- [ ] post-filter switch (`reverse_sample(model, mel, *, seed, post_filter=fir | None)`、T-M3.4 連携) で Diff の apply/no-apply の MCD / log F0 RMSE を比較できる評価 driver を提供
- [ ] validation metric として T-M2.3 の `MultiResolutionSTFTLoss` を eval でも再利用 (`from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss`)
- [ ] 無音 / 極短 utterance での F0 抽出失敗が `NaN` 伝播なく安全に skip される (edge case handling)
- [ ] `tests/test_compute_metrics.py` の全テスト pass
- [ ] `docs/milestones.md` §M4.1 Acceptance 2 項目クリア、`docs/tickets/index.md` の T-M4.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル

- 新規:
  - `src/wavenext2/eval/compute_metrics.py` (本実装: `compute_mcd`, `compute_log_f0_rmse`, `evaluate_dataset`, MCD バックエンド抽象)
  - `tests/test_compute_metrics.py` (Unit + edge case + 速度回帰 + fallback 整合)
- 編集:
  - `src/wavenext2/eval/__init__.py` (`__all__` に `compute_mcd`, `compute_log_f0_rmse`, `evaluate_dataset` を追加)
  - `pyproject.toml` (MCD バックエンドを `pymcd` / `mel-cepstral-distance` のいずれかに確定。T-M0.1 §6.1 の「M4 着手時に再確認」を実行。`uv add` で更新し `uv.lock` の diff を commit)
  - `docs/milestones.md` §M4.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M4.1 ステータス更新

### 2.2 主要構造

#### import 形式 (後続 T-M5.1 / T-M5.2 で確定)

```python
from wavenext2.eval.compute_metrics import (
    compute_mcd,
    compute_log_f0_rmse,
    evaluate_dataset,
)
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
) -> dict[str, float]:
    """(gt_path, synth_path) のリストを並列評価して平均指標を返す.

    - multiprocessing.Pool (num_workers) で utterance を分散 (DTW は CPU-bound)
    - mrstft_* は T-M2.3 MultiResolutionSTFTLoss を batch GPU で別途集計
    - NaN (F0 抽出失敗 utterance) は nanmean で除外、除外件数も返す
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
3. 各 utterance の結果を集約し、`np.nanmean` で平均 (NaN = F0 失敗を除外)
4. `{"mcd": ..., "log_f0_rmse": ..., "mrstft_sc": ..., "mrstft_mag": ..., "n_skipped_f0": int, "n_total": int}` を返す

#### post-filter switch driver (T-M3.4 連携)
1. Diff 評価時は `reverse_sample(model, mel, seed=43, post_filter=None)` と `reverse_sample(..., post_filter=fir)` の両方で合成
2. 各々を GT と対にして `evaluate_dataset` に渡し、MCD / log F0 RMSE の差分 (post-filter 改善幅) をレポート

### 2.5 設計上の重要決定

- **波形ペア → スカラーの純粋関数に責務限定**: 合成 (reverse_sample / GAN forward) は呼び出し側、本チケットは GT と合成済み波形の対だけを受ける。UTMOS/NISQA (T-M4.2)・RTF (T-M4.3) とも責務分離
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
- `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float` (pymcd / fallback 抽象、DTW alignment)
- `compute_log_f0_rmse(y_true, y_pred, sr=24000, ...) -> float` (pyworld F0 抽出 + voiced マスク + NaN セーフ)
- `evaluate_dataset(pairs, sr, num_workers, metrics) -> dict` (並列バッチ評価、MR-STFT を GPU batch で同時集計)
- MCD バックエンド抽象 (`_resolve_mcd_backend`) と `pyproject.toml` 依存確定
- post-filter apply/no-apply の MCD / log F0 RMSE 比較 driver (T-M3.4 連携、`reverse_sample(post_filter=...)` 呼び分け)
- T-M2.3 `MultiResolutionSTFTLoss` の eval metric 再利用
- `tests/test_compute_metrics.py` (同一音声 ≈0 / 歪み増加 / 30 分予算 / F0 edge case / fallback 数値整合)

### Out of Scope
- UTMOS / NISQA (NN ベース MOS 推定) → **T-M4.2**
- RTF 速度計測 → **T-M4.3**
- PESQ / STOI / speaker similarity (resemblyzer) → §8.1 代替案 (M6.3 ablation で追加検討)
- CREPE / RMVPE (NN ベース F0) への置換 → §8.1 代替案
- 合成パイプライン本体 (GAN forward / `reverse_sample`) → T-M2.4 / T-M3.3 の責務、本チケットは波形ペアを受けるだけ
- 主観評価 (MOS) の集計・統計検定 → T-M7.1
- 論文 Table 1〜3 との具体的な数値対比レポート生成 → M6.1 / M6.2 (本チケットは指標計算器のみ)

### Deliverable
- ファイル:
  - `src/wavenext2/eval/compute_metrics.py` (新規)
  - `tests/test_compute_metrics.py` (新規)
- 関数 / クラス:
  - `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float`
  - `compute_log_f0_rmse(y_true, y_pred, sr=24000, f0_floor, f0_ceil, frame_period) -> float`
  - `evaluate_dataset(pairs, sr=24000, num_workers=8, metrics=...) -> dict[str, float]`
  - `_resolve_mcd_backend(name) -> backend`, `_to_numpy_mono(y) -> np.ndarray` (内部)
- ドキュメント差分:
  - `docs/milestones.md` §M4.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M4.1 ステータス更新
  - (該当時) `pyproject.toml` の MCD 依存確定を `docs/open-questions.md` に追記

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
- [ ] `test_evaluate_dataset_nanmean_skips_failures` — F0 失敗 utterance を含む pairs で `n_skipped_f0` が正、平均が `nanmean` で汚染されない
- [ ] `test_evaluate_dataset_returns_mrstft` — `metrics` に `mrstft_sc`/`mrstft_mag` を含めると T-M2.3 由来の値が dict に入る

### 5.2 e2e / 結合テスト

- [ ] LibriTTS-R test-clean のサブセット (例 50 utt) を実 GT で `evaluate_dataset` に通して dict が返る
- [ ] post-filter switch driver: `reverse_sample(model, mel, seed=43, post_filter=fir)` / `None` の 2 系列を評価し、MCD / log F0 RMSE の差分が出る (T-M3.4 連携、未学習 model では値の妥当性は不問・API 動作のみ)
- [ ] **速度回帰**: サブセット N utt の wall time から 4,824 utt を外挿し **30 分以内** に収まる見込みを確認 (本番は M5/M6 で full 実行)

### 5.3 Acceptance criteria (`docs/milestones.md` §M4.1 より転記)
- [ ] 同一音声に対して MCD ≈ 0, log F0 RMSE ≈ 0
- [ ] LibriTTS-R test-clean 全 4,824 utterance を 1 GPU で 30 分以内に処理

### 5.4 追加 acceptance (本チケット独自)
- [ ] `from wavenext2.eval.compute_metrics import compute_mcd, compute_log_f0_rmse, evaluate_dataset` が動作
- [ ] MCD バックエンドが `pyproject.toml` で確定 (pymcd or mel-cepstral-distance)、`uv sync` でエラーなく解決
- [ ] 既知の歪み (ノイズ付加) で MCD が増加することを定量確認
- [ ] F0 抽出失敗 (無音 / 極短) の edge case が `np.nan` で安全に処理される
- [ ] `mel-cepstral-distance` fallback が pymcd と同じ MFCC order で数値整合 (両 import 可能な環境)
- [ ] CPU のみでも全 Unit テストが pass (MR-STFT GPU batch は CUDA marker で分離)

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

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目: 評価指標の具体的パラメータ (F0 floor/ceil, MCD order, voiced マスク) は論文に明示がなく **本チケットで確定** する:
  - MCD: c1..c24, DTW alignment (採用)
  - log F0 RMSE: 両者 voiced AND マスク, natural log, pyworld デフォルトパラメータ (採用)
- MCD バックエンドの確定 (pymcd or mel-cepstral-distance) は T-M0.1 §6.1 / §9.3 の open question を本チケットで解消 → 結果を `pyproject.toml` と (必要なら) `docs/open-questions.md` に反映

### 6.3 他チケットとの整合性

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
- [ ] post-filter switch (`reverse_sample(post_filter=fir / None)`) で apply/no-apply 比較ができる
- [ ] T-M2.3 `MultiResolutionSTFTLoss` を再利用 (重複実装していない)
- [ ] 30 分予算が並列化 (process pool) で達成見込み、速度回帰テストで外挿確認
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 不正入力 (空 pairs, shape mismatch) で適切な例外、F0 失敗は例外でなく `np.nan`
- [ ] **参考実装 (pymcd / ParallelWaveGAN の eval) をコピーしていない**: バックエンドは pip ライブラリ呼び出しに留め、独自 driver は一から記述

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M4) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **MCD を自前 DTW + MFCC で実装** (`librosa.feature.mfcc` + `librosa.sequence.dtw`) | pymcd / mel-cepstral-distance の cp313 wheel 依存を完全排除、I/O オーバーヘッド (tmp WAV) 回避、in-memory 配列で高速 | DTW 実装の正しさ検証が必要、論文 MCD と数値が微妙にずれる可能性 | まず枯れた pip ライブラリ (pymcd) を試す。wheel / 速度問題が出たら本案へ | **§6.1 Critical: pymcd cp313 build 失敗かつ fallback も不可**、または **30 分予算超過** |
| **log F0 RMSE を CREPE / RMVPE (NN ベース F0) で代替** | pyworld DIO より高精度、雑音耐性が高い | GPU 必須・重い、追加依存 (torchcrepe 等)、論文は WORLD 系想定 | pyworld で論文整合性を優先、軽量 | **M6 で DIO の F0 抽出が破綻 (オクターブエラー多発) する場合** |
| **PESQ / STOI を追加** (音質・了解度指標) | 知覚音質の補完指標、文献比較が容易 | 論文 Table に無い指標で対比できない、追加依存 (`pesq`, `pystoi`) | 論文 §5.2 は UTMOS/NISQA/MCD/logF0 の 4 指標、本チケットは MCD/logF0 に集中 | **M6.3 ablation で追加分析が必要なとき** |
| **speaker similarity (resemblyzer) を追加** | 話者性保持の評価 | 論文に無い、d-vector モデル依存 | スコープ外、vocoder は話者非依存タスク | **多話者 generalization 検証が必要なとき (M6 以降)** |
| **MCD/F0 を 1 GPU batch で完結 (DTW を GPU 実装)** | process pool 不要、GPU で一括 | GPU DTW 実装が複雑、pymcd は CPU 前提 | CPU process pool で 30 分予算は達成見込み | **CPU 並列でも予算超過するとき** |
| **VUV (voiced/unvoiced) error を追加指標化** | 韻律評価の補完 (voiced 判定不一致を測る) | 論文 Table に無い、本チケットは log F0 RMSE に集中 | スコープ外 | **M6 で F0 RMSE だけでは品質差が出ないとき** |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M4.1 は size=M と妥当。MCD + log F0 RMSE + バッチ driver + edge case で 2〜4 時間規模。MCD と log F0 RMSE を別チケットに割るほどの分量はない (共通の波形正規化・並列 driver を共有するため統合が自然)
- **責務境界**: 「波形ペア → スカラー」に限定し、合成 (reverse_sample / GAN forward) を呼び出し側に置く設計は維持。UTMOS/NISQA (T-M4.2)・RTF (T-M4.3) と機能境界がきれいに分かれており、M4 の 3 チケットを並列実装可能
- **インターフェース定義の見直し余地**: `evaluate_dataset` の戻り値を flat dict にしているが、将来 per-utterance の生値も必要なら `(summary_dict, per_utt_df)` の tuple 化を検討 (M6 の ablation レポートで per-utterance 分布が要るかもしれない)
- **MR-STFT を eval に含める是非**: validation の checkpoint 選択基準 (WaveFit-PT) と整合させるため含めるが、論文 Table 1〜3 は MR-STFT を主要指標にしていない → `metrics` 引数でオプトイン (デフォルト含めるが外せる)

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

- **インターフェース**:
  - `compute_mcd(y_true, y_pred, sr=24000, backend=None) -> float`
  - `compute_log_f0_rmse(y_true, y_pred, sr=24000, f0_floor=71, f0_ceil=800, frame_period=5.0) -> float`
  - `evaluate_dataset(pairs: list[tuple[PathLike, PathLike]], sr=24000, num_workers=8, metrics=("mcd","log_f0_rmse","mrstft_sc","mrstft_mag")) -> dict[str, float]`
  - いずれも torch.Tensor / np.ndarray の mono float32 波形 [-1,1] を受ける
- **設定値**:
  - MCD バックエンド (確定したもの: pymcd or mel-cepstral-distance) を `pyproject.toml` に pin
  - F0: pyworld デフォルト (`frame_period=5.0`, `f0_floor=71`, `f0_ceil=800`)、voiced AND マスク, natural log
- **注意事項 (後続が踏みそうな罠)**:
  - F0 抽出失敗は **例外でなく `np.nan`** を返す。集計は必ず `np.nanmean`、skip 件数も確認
  - MCD は DTW alignment 込みなので長さ違いの波形でも測れるが、log F0 RMSE は frame 数を min に揃える
  - Windows では multiprocessing が `spawn` のため `evaluate_dataset` を `if __name__ == "__main__"` 下から呼ぶ (T-M5.1 / T-M5.2 / M6 評価 driver)
  - post-filter 比較は `reverse_sample(post_filter=fir / None)` の呼び分けで行う (T-M3.4 統合済 API、`apply_post_filter` を chain しない)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M4.1 の Acceptance チェックボックス 2 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M4.1 ステータスを `📝 pending` → `✅ completed`、M4 進捗サマリ更新
  - [ ] (該当時) `pyproject.toml` の MCD 依存確定を `docs/open-questions.md` の T-M0.1 §6.1 open question 解消として追記
  - [ ] (該当時) `docs/tickets/index.md` のフェーズレビューログ M4 行を更新

### 9.3 Open question として残ったもの
- pymcd vs mel-cepstral-distance のどちらが論文 Table の MCD 値に近いか — M6 で論文と対比したときに乖離が大きければバックエンド再検討 (`eval_stats` にどちらで測ったか記録しておく)
- pyworld DIO の F0 抽出パラメータ (`f0_floor=71`) が LibriTTS-R 全話者で妥当か — 低音話者でオクターブエラーが出れば `f0_floor` 調整 or CREPE/RMVPE 切替 (§8.1)
- 30 分予算が実 4,824 utt + 実 GPU で本当に達成できるか — full 実行は M5/M6、超過時は §8.1 自前高速 DTW or 並列度増へ
