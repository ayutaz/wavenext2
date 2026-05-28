---
id: T-M0.3
title: LibriTTS-R 取得と filelist 生成
milestone: M0
phase: M0
status: in_review
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M2.1, T-M5.1, T-M5.2]
related_docs:
  - docs/milestones.md#m03-libritts-r-取得
  - docs/implementation-plan.md
  - docs/training.md
---

# T-M0.3: LibriTTS-R 取得と filelist 生成

> **マイルストーン**: [M0](../milestones.md#m0-環境整備とデータ準備-作業量-small) / **サブタスク**: [M0.3](../milestones.md#m03-libritts-r-取得)
> **依存**: T-M0.1, T-M0.2 / **後続**: T-M2.1, [T-M5.1](T-M5.1-gan-1epoch.md), [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
LibriTTS-R (CC BY 4.0) の "train-clean-100" + "train-clean-360" + "test-clean" を取得・展開し、後続の Dataset 実装 (T-M2.1) が読み込める wav の filelist (`data/filelists/{train,val,test,dev_postfilter}.tsv`) と統計 (`stats.json`, `audio_info.tsv`) を生成する。同時に sample rate が 24 kHz であることを検証し、必要なら resample を案内する (LibriTTS-R は元から 24 kHz であるため通常 resample は不要)。

ユーザー操作が必須なマイルストーン (`docs/milestones.md` §M0.3) であり、ライセンス同意とダウンロード認証はユーザー側で実施する。Claude Code は展開・filelist 生成・整合性検証を担当する。

### 確定事項 (3 視点レビュー反映)
- **`data/filelists/*.tsv` は commit する** (~数 MB、再現性確保。CI smoke で `val.tsv` を直接使用可能)
- **`data/filelists/stats.json` を出力** (speaker / chapter / duration / dBFS 分布、M4 話者バイアス分析にも使用)
- **`data/filelists/audio_info.tsv` を出力** (全 wav の sr / channels / duration、filelist 生成の source of truth)
- **HuggingFace mirror の採否は本チケット実装時に確定** (実装時に `mythicinfinity/libritts_r` 等の存在を確認、認証不要なら自動 DL に切り替え、要認証なら openslr ユーザー操作を維持)
- **filelist フォーマット = TSV header 付き** (詳細は §2.2 / §2.5、後続 T-M2.1 の BucketSampler に直結)
- **validation = speaker-balanced sampling** (詳細は §2.4、ランダム抽出の話者偏りを回避)
- **`dev_postfilter.tsv` (200 utterances) も本チケットで生成** (T-M3.4 で同じスクリプトを再実行しない、責務集中)
- **`LICENSE-LibriTTS-R.md` をリポジトリ ROOT に配置** (attribution)

### ゴール
- [ ] `scripts/prepare_libritts.py` が完成し、`--src-dir` 引数で LibriTTS-R ルートを受け取って動作する
- [ ] `data/filelists/train.tsv` が約 145,000 行で生成される (train-clean-100 + 360 から val / dev_postfilter / 長さフィルタ除外を除いた残り)
- [ ] `data/filelists/val.tsv` が 100 行 (speaker-balanced sampling、seed=42)
- [ ] `data/filelists/test.tsv` が 4,824 行 (test-clean 全件、長さフィルタ非適用)
- [ ] `data/filelists/dev_postfilter.tsv` が 200 行 (train から hold-out、seed=43、val と非重複)
- [ ] `data/filelists/stats.json` (speaker / chapter / duration / dBFS 分布) が生成される
- [ ] `data/filelists/audio_info.tsv` (全 wav の sr / channels / duration) が生成される
- [ ] `LICENSE-LibriTTS-R.md` をリポジトリ ROOT に配置
- [ ] 任意の wav を `soundfile` (`sf.info` / `sf.read`) で読めて `sample_rate == 24000` であることを全件検証 (`audio_info.tsv` 経由)。**torchaudio 2.11 で `load`/`info` が廃止されたため soundfile に統一** (T-M0.1 §9.1)
- [x] train / val / dev_postfilter / test に重複がないこと (set 比較) をスクリプト内で assert (synthetic e2e で検証済)

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `scripts/prepare_libritts.py` (CLI スクリプト本体)
- 新規 (出力): `data/filelists/{train,val,test,dev_postfilter}.tsv`, `data/filelists/stats.json`, `data/filelists/audio_info.tsv`
- 新規 (リポジトリ ROOT): `LICENSE-LibriTTS-R.md` (attribution、最小限引用)
- 編集: -

### 2.2 主要構造
```python
# scripts/prepare_libritts.py
import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
# 注: torchaudio は使わない (2.11 で load/info 廃止)。音声 I/O は soundfile に統一


# TSV columns (header) — T-M2.1 BucketSampler が直接読む
FILELIST_HEADER = [
    "rel_path", "n_samples", "speaker_id", "chapter_id",
    "duration_sec", "peak_dbfs", "rms_dbfs",
]


def collect_wavs(root: Path, subsets: list[str]) -> list[Path]:
    """subsets 配下の wav を再帰収集して相対パスで返す"""
    ...


def build_audio_info(wav_paths: list[Path], src_dir: Path) -> list[dict]:
    """全 wav について soundfile.info で sr/channels/duration + peak/RMS dBFS を集計"""
    ...


def speaker_balanced_sample(
    pool: list[dict], n_speakers: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """n_speakers 話者を抽出し、各話者から発話数最多の wav を 1 件選択。
    Returns (selected, remaining)
    """
    ...


def filter_duration(rows: list[dict], min_sec: float, max_sec: float) -> tuple[list[dict], list[dict]]:
    """duration_sec が [min_sec, max_sec] の範囲外を除外。Returns (kept, filtered)"""
    ...


def write_tsv(path: Path, rows: list[dict]) -> None:
    """TSV (header 付き, UTF-8, LF 改行) で書き出し"""
    ...


def write_stats(path: Path, stats: dict) -> None:
    """speaker / chapter / duration / dBFS 分布を JSON で書き出し"""
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="LibriTTS-R 展開ルート (LibriTTS_R/train-clean-100/... を含む)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/filelists"),
                        help="filelist 出力先")
    parser.add_argument("--val-size", type=int, default=100,
                        help="train から speaker-balanced で hold-out する validation 数")
    parser.add_argument("--dev-postfilter-size", type=int, default=200,
                        help="train から hold-out する post-filter fit 用件数 (T-M3.4)")
    parser.add_argument("--val-seed", type=int, default=42,
                        help="validation hold-out の再現性確保")
    parser.add_argument("--postfilter-seed", type=int, default=43,
                        help="dev_postfilter hold-out (val と区別)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="train から除外する下限 (segment_length 16384/24000 ≈ 0.68s)")
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="train から除外する上限 (OOM リスク回避)")
    parser.add_argument("--verify-checksum", action="store_true",
                        help="openslr .tar.gz の MD5 を検証 (任意、--src-dir 直下に tar.gz が要)")
    parser.add_argument("--min-free-gb", type=float, default=100.0,
                        help="ディスク空き容量の事前チェック (DL + 解凍想定)")
    args = parser.parse_args()
    ...
```

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| sample_rate | 24000 Hz | `docs/training.md` §1.1, §1.2 |
| train subsets | `["train-clean-100", "train-clean-360"]` | `docs/training.md` §1.1 |
| test subset | `["test-clean"]` | `docs/milestones.md` §M0.3 |
| validation 件数 | 100 (speaker-balanced) | `docs/training.md` §6 |
| dev_postfilter 件数 | 200 (val と非重複) | `docs/training.md` §5.4 (T-M3.4 の入力) |
| min-duration | 1.0 秒 | segment_length 16384/24000 ≈ 0.68s より大きく取る |
| max-duration | 15.0 秒 | OOM リスク回避、val/test には非適用 |
| 想定 train 行数 | 約 145,000 (filter 後) | `docs/milestones.md` §M0.3 Acceptance |
| 想定 test 行数 | 4,824 | `docs/milestones.md` §M0.3 Acceptance |
| validation seed | 42 | speaker-balanced sampling 用 |
| dev_postfilter seed | 43 | val と区別、衝突回避 |
| min-free-gb | 100 GB | DL + 解凍想定 |

### 2.4 アルゴリズム / 処理フロー
1. CLI 引数をパース、`shutil.disk_usage(--src-dir)` で `--min-free-gb` 未満なら abort
2. `--src-dir` 配下に `train-clean-100/`, `train-clean-360/`, `test-clean/` が存在するか検証 (なければ error)。**`README.md` 等で "LibriTTS-R" 文字列を確認** (LibriTTS と LibriTTS-R 取り違え事故防止)
3. `--verify-checksum` が指定された場合、openslr の `.tar.gz` MD5 をスクリプト同梱 dict と照合
4. 各 subset を `pathlib.Path.rglob("*.wav")` で再帰収集し、`--src-dir` からの **相対パス** に変換 (POSIX 形式 `/` 統一)
5. **全 wav について `soundfile.info` で sr/channels/duration を取得、波形を読み peak/RMS dBFS を計算** → `audio_info.tsv` に出力 (~55 万 wav、数分想定)。**`sample_rate != 24000` を error log に列挙**
6. パスから `speaker_id`, `chapter_id` をパース (LibriTTS-R ディレクトリ規約: `<subset>/<speaker>/<chapter>/<utt>.wav`)
7. train pool に **音声長フィルタ** (`min-duration` / `max-duration`) を適用、filtered 件数を `stats.json` に記録。val / test / dev_postfilter には**非適用**
8. **dev_postfilter 抽出**: `random.Random(postfilter_seed=43)` で train pool から 200 件抽出
9. **val 抽出 (speaker-balanced)**:
   - dev_postfilter を除いた pool から speaker_id 一覧を作る (~1,151 話者)
   - `random.Random(val_seed=42)` で 100 speakers を抽出
   - 各 speaker から発話数最多の wav を 1 件選択 → val 100 utterances
10. train = train pool − val − dev_postfilter (set 差集合、`assert len(train) + len(val) + len(dev_postfilter) == len(train_pool_after_filter)`)
11. test = test-clean 全件 (sorted、長さフィルタ非適用)
12. `stats.json` に以下を出力:
    - `train.n_utterances`, `train.n_speakers`, `train.n_chapters`
    - `train.duration_hours_total`, duration histogram (bins: 0-1, 1-3, 3-5, 5-10, 10-15, 15+)
    - `train.peak_dbfs` / `train.rms_dbfs` の min/max/mean/p5/p50/p95
    - `val.speakers` (100 speaker id 一覧、検証用)
    - `filtered.n_too_short`, `filtered.n_too_long`
    - 同様に val / test / dev_postfilter
13. `--out-dir` を作成し 4 TSV + `stats.json` + `audio_info.tsv` を書き出し (UTF-8, LF 改行)
14. **重複検証**: `assert set(train) ∩ set(val) ∩ set(dev_postfilter) == ∅`
15. 完了サマリ (各行数、検証通過件数、filtered 件数) を stdout に出力

### 2.5 想定するファイル構成
```
<src-dir>/
  train-clean-100/
    103/  # speaker id
      1241/  # chapter id
        103_1241_000000_000001.wav
        ...
  train-clean-360/
    ...
  test-clean/
    ...
```

filelist は **TSV (header 付き)** 形式:
```
rel_path	n_samples	speaker_id	chapter_id	duration_sec	peak_dbfs	rms_dbfs
train-clean-100/103/1240/103_1240_000001_000000.wav	72000	103	1240	3.00	-2.15	-21.34
train-clean-100/103/1240/103_1240_000001_000001.wav	56400	103	1240	2.35	-3.02	-22.10
...
```

**列の意図**:
- `n_samples`: T-M2.1 BucketSampler / 長さ別 batch を作るとき、毎 epoch 全 wav を `torchaudio.info` で開く I/O ボトルネックを回避
- `speaker_id`: T-M2.6 / T-M3.5 smoke 評価で speaker-balance を検証
- `peak_dbfs` / `rms_dbfs`: T-M2.1 sox `norm` 正規化前後の peak validate

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `scripts/prepare_libritts.py` 実装 + Acceptance 自動検証 | general-purpose |
| Tester | 1 | ダウンロード成否確認、整合性 (重複なし / 24 kHz / speaker-balance) の e2e 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M0.1 環境構築と T-M0.2 scaffold が前提)
- 並列実行する場合の最大並列数: 1
- 注意: **ユーザー操作 (LibriTTS-R のライセンス同意 + ダウンロード) が必須** のため、Claude Code はユーザーがダウンロード先パスを伝えるまで待機する。
  - 代替案: huggingface mirror (例 `mythicinfinity/libritts_r` 等) があれば認証不要で `uv run` 経由の自動 DL も検討可能。**本チケット実装時に存在確認** (§6 参照)。

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/prepare_libritts.py` (CLI スクリプト)
- `data/filelists/{train,val,test,dev_postfilter}.tsv` の生成 (TSV header 付き)
- `data/filelists/stats.json` の生成 (speaker / chapter / duration / dBFS 分布)
- `data/filelists/audio_info.tsv` の生成 (全 wav の sr / channels / duration)
- 24 kHz 検証 (全件、`audio_info.tsv` 経由で `sample_rate != 24000` を 0 件確認)
- 音声長フィルタ (`min-duration` / `max-duration`, train のみ適用)
- speaker-balanced validation sampling (seed=42)
- dev_postfilter hold-out (seed=43, T-M3.4 用 200 utterances)
- 再現性確保のための seed 固定
- Windows / POSIX どちらでも動くパス処理 (出力は POSIX 形式)
- `LICENSE-LibriTTS-R.md` 配置 (attribution)
- ディスク容量の事前チェック (`shutil.disk_usage`, default 100 GB)
- データセット corruption 検知 (`--verify-checksum` 任意フラグ)

### Out of Scope
- LibriTTS-R 本体のダウンロード自動化 (ユーザーが手動 DL する前提。huggingface mirror による自動 DL 対応は §8 検討事項に残す)
- 24 kHz 以外からの resample (LibriTTS-R は元から 24 kHz なので不要)
- mel 抽出 (`scripts/extract_mel.py` は別タスク。本タスクでは filelist のみ生成)
- Dataset class 実装 (T-M2.1 で別途実装)
- ファイル単位の音質チェック (clipping 検出など。スコープ外)
- **webdataset / lmdb / tar shard 化** (M5 完了時に I/O 律速が判明したら再評価、§8.1 参照)
- **speaker-disjoint val** (論文は seen-speaker 合成品質を評価、§8.1 で却下根拠を明記)

### Deliverable
- ファイル: `scripts/prepare_libritts.py`, `data/filelists/{train,val,test,dev_postfilter}.tsv`, `data/filelists/stats.json`, `data/filelists/audio_info.tsv`, `LICENSE-LibriTTS-R.md`
- 関数 / クラス: `collect_wavs`, `build_audio_info`, `speaker_balanced_sample`, `filter_duration`, `write_tsv`, `write_stats`, `main`
- ドキュメント差分:
  - `docs/milestones.md` §M0.3 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M0.3 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト
最小限の動作確認は e2e で十分だが、**`speaker_balanced_sample` と `filter_duration` は単体テスト推奨** (`tests/test_prepare_libritts.py`)。理由: ランダム抽出は seed が同じでも実装差で結果が変わりやすく、後続スモークの再現性に直結する。

### 5.2 e2e / 結合テスト
- [ ] **サンプル wav 1 個での smoke**: `scripts/prepare_libritts.py --src-dir <tiny>` がエラーなく完走 (tiny は test-clean から数 wav だけ取り出した tmpdir でも可)
- [ ] **filelist 行数**: 生成された `train.tsv` が約 145,000 行、`val.tsv` が 100 行、`test.tsv` が 4,824 行、`dev_postfilter.tsv` が 200 行
- [ ] **24 kHz 検証**: `audio_info.tsv` 全行で `sample_rate == 24000` (0 件の混入があれば error)
- [ ] **重複なし**: `set(train) ∩ set(val) ∩ set(dev_postfilter) == ∅` を assert
- [ ] **相対パス**: 出力 filelist の各行が `--src-dir` からの相対パス (絶対パス混入なし)、`/` 区切り
- [ ] **val speaker-balanced**: `val.tsv` の speaker_id ユニーク数が 100 (1 話者 1 utterance)
- [ ] **スナップショット検証**: `sorted(val_paths_hash) == EXPECTED_HASH` (実装時に確定、変更検知用)
- [ ] **stats.json schema**: 必須キー (`train.n_utterances`, `train.n_speakers`, `filtered.n_too_short`, etc.) が存在

### 5.3 Acceptance criteria (3 視点レビュー反映後)
- [ ] `data/filelists/{train,val,test,dev_postfilter}.tsv` が生成される (TSV、header あり)
- [ ] `data/filelists/stats.json` が生成され、speaker / chapter / duration / dBFS 分布を含む
- [ ] `data/filelists/audio_info.tsv` が生成され、全 wav の sr / channels / duration を含む
- [ ] train: 145k 行前後 (filter 後の正確な行数を `stats.json` で確認)
- [ ] val: 100 行 (speaker-balanced、1 話者 1 utterance)
- [ ] test: 4,824 行 (公式 LibriTTS-R test-clean に一致)
- [ ] dev_postfilter: 200 行 (val と非重複)
- [ ] `sorted(val_paths_hash) == EXPECTED_HASH` でスナップショット検証 (テストで担保、変更検知)
- [ ] 全 wav のうち sample_rate=24000 でないものが 0 件 (`audio_info.tsv` から確認)
- [ ] `LICENSE-LibriTTS-R.md` がリポジトリ ROOT に存在
- [ ] 任意の wav ファイルを `soundfile` で読めて `sample_rate=24000` を確認 (torchaudio 2.11 で load/info 廃止のため soundfile に統一)

## 6. 懸念事項

### 6.1 技術的リスク

#### critical → ✅ 解決 (2026-05-27)
- **test-clean vs test-clean-100 の表記揺れ**: openslr/141 に "test-clean-100" split は存在せず、公式は `test-clean`。**実装は `TEST_SUBSET = "test-clean"` で確定**。論文の "test-clean-100" (4,824 utt) はこの `test-clean` を指すと解釈 (件数一致)。`docs/training.md` §5 冒頭に明確化注記を追加済み。paper-summary.md / open-questions.md の "test-clean-100" は論文引用なので原文保持。**論文 PDF が手元になく Table 1 の厳密確認は未了** → PDF 入手時に再確認 (現状の解釈で実害なし、§9.4 参照)。

#### 追加 (3 視点レビュー)
- **LibriTTS vs LibriTTS-R 混同リスク**: openslr/60 (LibriTTS) と openslr/141 (LibriTTS-R) を取り違える事故 → `--src-dir` バリデーションで `README.md` 内の "LibriTTS-R" 文字列確認
- **ディスク事前チェック**: `shutil.disk_usage()` で 100 GB (DL + 解凍) 未満なら abort する safety guard
- **データセット corruption**: `--verify-checksum` フラグ (任意、デフォルト OFF) で openslr `.tar.gz` の MD5 を検証
- **filelist の改行コード**: Windows で CRLF 化けるリスク → T-M0.2 の `.gitattributes` で `eol=lf` 強制 (T-M0.2 の責務、本チケットは確認のみ)
- **Speaker imbalance の可視化**: `stats.json` で対処、M2.1 sampler 設計に直結

#### 既存
- **ディスク容量**: LibriTTS-R 解凍後合計 ~50 GB。`--min-free-gb` (default 100 GB) で事前 abort
- **ネットワーク帯域**: openslr.org からの DL は数 GB × 3 ファイルで、低速回線だと数時間かかる。タイムアウト時は再開可能な `wget -c` / `curl -C -` の使用を推奨
- **Windows パスの `\` vs `/` 問題**: Windows の `pathlib.Path` は `\` で出力するが、filelist は POSIX 形式 `/` で統一する (`Path.as_posix()` を使用)
- **subset ディレクトリ名のゆれ**: openslr 版は `train-clean-100` (ハイフン)、huggingface mirror 版は `train_clean_100` (アンダースコア) など命名が違うことがある。`--src-dir` 配下を `rglob` で検出するロジックは subset 名指定ではなく、`*.wav` の親ディレクトリパターンで検出する方が頑健かもしれない (要設計判断)
- **読めない wav (corruption)**: 一部の wav が壊れていた場合、`torchaudio.info` で例外が出る。`build_audio_info` 全件走査で検出して error log に列挙

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` で LibriTTS-R に関する未解決事項はなし (データセット自体は標準的)
- ただし以下はこのチケット内で確定 (詳細は §1 確定事項):
  - **validation 100 utterances の選択方法**: speaker-balanced sampling (seed=42)。ランダム抽出の話者偏り回避
  - **dev_postfilter 200 utterances**: 本チケットで生成 (seed=43、val 非重複)
  - **filelist のパス形式**: 相対パス (POSIX 形式) + TSV header 付き
  - **train/val/test 分割**: openslr の標準 split に従う (test-clean は train から完全分離)
  - **音声長フィルタ**: train のみ `[1.0, 15.0]` 秒、val / test / dev_postfilter には非適用

### 6.3 他チケットとの整合性
- **T-M2.1 (Dataset)**: filelist の読み込み側。本チケットで決めた「TSV header 付き / 相対パス / POSIX 区切り」の規約を共有。BucketSampler は `n_samples` 列を直接使用。Dataset 側で `Path(root) / row["rel_path"]` のように結合する設計と整合
- **T-M2.6 (GAN smoke), T-M3.5 (Diff smoke)**: smoke で validation 100 utterances を seed 固定で参照するため、本チケットで生成した `val.tsv` を使用 (speaker-balanced 済み)
- **T-M3.4 (Post-filter)**: `dev_postfilter.tsv` (200 utterances) を使用。本チケットで生成済みのため M3.4 で同じスクリプトを再実行しない
- **T-M5.1, T-M5.2 (1 epoch smoke)**: train.tsv の全量で 1 epoch 訓練するため、行数 (約 145k) が大幅にズレるとステップ数見積もりに影響

## 7. レビュー観点

実装完了後、Reviewer (もしくは Self-review) が以下を確認:

- [ ] filelist のパスが **相対パス** で統一されている (絶対パス混入なし)
- [ ] パス区切り文字が `/` (POSIX 形式) で統一されている (Windows の `\` が残っていない)
- [ ] **TSV header** が `FILELIST_HEADER` 定数と一致
- [ ] **24 kHz 確認** が全件で抜けていない (`audio_info.tsv` 経由)
- [ ] val が train から正しく除外されている (`set(train) & set(val) == set()`)
- [ ] **val は speaker-balanced** (`val.tsv` の speaker_id ユニーク数 = 100)
- [ ] dev_postfilter が val / train と非重複
- [ ] **seed 固定** で再現可能 (`random.Random(seed)` を使用、`random.seed()` グローバル汚染を避ける)
- [ ] CLI 引数のデフォルト値が妥当 (`--val-size=100`, `--val-seed=42`, `--postfilter-seed=43`, `--min-duration=1.0`, `--max-duration=15.0`)
- [ ] エラー処理: `--src-dir` 不存在、subset ディレクトリ不存在、wav 1 個も見つからない、24 kHz 以外を検出、ディスク容量不足 — 各ケースで明示的にエラーメッセージを出す
- [ ] `stats.json` のキー命名が後続 (T-M2.1 / M4) で参照しやすい形 (snake_case、ネスト浅め)
- [ ] CLAUDE.md 既存スタイル準拠 (型ヒント、docstring、命名)
- [ ] `LICENSE-LibriTTS-R.md` がリポジトリ ROOT に存在し、LibriTTS-R 公式 README から最小限引用

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら
- **別案 A: HuggingFace mirror 自動 DL** (`huggingface_hub.snapshot_download` で `mythicinfinity/libritts_r` 等の parquet を取得)
  - メリット: 認証不要 (確認できれば)、再開可能、M6 クラウド環境でも有用、ユーザー操作不要に
  - デメリット: mirror が実際に存在し authentication なしで使えるか未確認
  - **再評価トリガー: 本チケット実装時に存在確認**。認証不要なら採用、要認証なら openslr ユーザー操作のまま
- **別案 B: huggingface datasets を直接利用** (`datasets.load_dataset("openslr/librittsr")`)
  - メリット: filelist 不要、ストリーミング可能、認証不要 (mirror 版があれば)
  - デメリット: 既存 mirror の存在確認が必要、`Dataset` interface が huggingface 依存になり Vocos / WaveFit-PT 流のシンプルさが失われる
  - 採用しなかった理由: 論文再現実装としてオリジナル openslr 版に従う方が他実装との比較がブレない
- **別案 C: lmdb / tar 形式に変換して I/O 高速化**
  - メリット: 大規模訓練 (M6 のフル 410h) で I/O ボトルネック軽減
  - デメリット: 前処理ステップが増え、デバッグ性が下がる
  - **再評価トリガー: M5 完了時に I/O 律速が判明したら**。scaffold 段階で意思決定すると後戻りコストが高い
- **別案 D: WebDataset 形式 (huge dataset shard 化)**
  - メリット: 分散訓練・クラウドストリーミングに最適
  - デメリット: shard 設計が必要、ローカル開発との非対称性
  - **再評価トリガー: M5 完了時に I/O 律速が判明したら** (C と同じ)
- **別案 E: 分散ストレージ (S3 / GCS) からの直接ストリーミング**
  - メリット: ローカルディスク不要、クラウド A100 (M6) と相性が良い
  - デメリット: ネットワーク I/O コスト、認証管理が複雑
  - 採用しなかった理由: M6 で GPU クラスタが確定してから検討する方が無駄がない
- **別案 F: filelist フォーマット = CSV / JSON Lines**
  - 採用しなかった理由: TSV header 付きが pandas / polars / 手 grep いずれにも親和的。タブはパスに混入しないため安全
- **別案 G: speaker-disjoint val** (validation 話者を train から完全除外)
  - 採用しなかった理由: **論文は seen-speaker 合成品質を評価** (`docs/training.md` §5.1 参照)。speaker-disjoint は zero-shot TTS 評価向けで本論文の目的とずれる
  - **再評価トリガー: なし** (採用しない方針確定)
- **別案 H: val の話者内 utterance 選択を「最長」でなく「中央長 (median)」に** (M0 phase review 追加)
  - 現状実装は各話者から **最長 utterance** を選択 (決定論的・metric 安定狙い)。だが最長 utterance は無音/間が混入しやすく、MCD/UTMOS の分散を増やす恐れ
  - 代替: duration が median の utterance を選べば話者の「典型的」発話に近く代表性が高い
  - 採用しなかった理由: 現状は決定論性を優先。実害が出るか未検証
  - **再評価トリガー: M4 で val の MCD/UTMOS 分散が過大なとき** (`speaker_balanced_sample` の選択キーを差し替えるだけで対応可)

### 8.2 思想 / 哲学の見直し
- このサブタスクの粒度は適切 (small)。filelist 生成のみで他要素 (mel 抽出など) を含まないため境界が明確
- **インターフェース定義の昇格**: 単純な「1 行 = wav パス」から **TSV header 付き** へ。後続が後付けで列を増やす負担を回避する初期投資
- **dBFS 統計の M2 連携**: sox `norm` 正規化は peak-based なので「全データの peak が同程度」前提だが、LibriTTS-R は録音条件にばらつきがある。M0.3 で peak/RMS 統計を出すと M2 の正規化方針を validate できる
- **責務の集中**: dev_postfilter.tsv も本チケットで生成する判断 (M3.4 で同じスクリプトを再実行する代替案より責務が綺麗)
- **改善余地**: speaker / chapter 分布を `stats.json` で出力済み → T-M2.1 で speaker balanced batch sampler を実装する余地を残す

### 8.3 LibriTTS-R 以外のデータセット候補 (将来的な ablation 用)
- **VCTK** (英語、109 話者、48 kHz): 多話者性で勝るが論文と sample rate が異なる
- **JVS** (日本語、100 話者): 日本語適用の ablation に
- **LJSpeech** (英語、1 話者、22 kHz): デバッグ用の単一話者
- **Common Voice**: より多言語、品質はやや劣る
- いずれも論文と異なるため M0 では採用しないが、M6 以降で「他データセットでの再現性」を見るときに使えるかもしれない

### 8.4 学んだこと (2026-05-27 実装後に追記)

実装結果:
- `scripts/prepare_libritts.py` 実装 + `tests/test_prepare_libritts.py` (9 件 pass)。`run()` をコア関数化し CLI `main()` から分離 → テストが実データ無しで合成ツリーに対し e2e 実行可能。
- 検証: filter_duration / speaker_balanced_sample / compute_dbfs の unit、合成 LibriTTS-R ツリー (6 spk + short/long + test 4) での e2e (件数・header・重複なし・相対 POSIX path・stats schema・val_hash 再現性)。

想定外と対処:
1. **音声 I/O は soundfile に統一** (torchaudio 2.11 で `load`/`info`/`sox_effects` 廃止、T-M0.1 §9.1)。`sf.info` で sr/frames、`sf.read` で peak/RMS dBFS。当初設計の `torchaudio.load`/`torchaudio.info` は使わない。
2. **"発話数最多の wav" の曖昧さ**: 「各話者から 1 件」の選択規則が曖昧だったため、**最長 utterance (n_samples 最大、tie-break rel_path)** に確定 — 決定論的で metric も安定。seed は話者選択のみに作用。
3. **test-clean-100 表記揺れ解決**: openslr 実 split は `test-clean`。§6.1 参照。
4. **`scripts/` は package でない** ため、テストからは `importlib.util.spec_from_file_location` でパス指定 import。`sys.path` 汚染を最小化。

次の似たタスクで応用できる教訓:
- **コア関数 (`run`) と CLI (`main`) を分離**すると、外部データに依存する処理でも合成 fixture で完全な e2e テストが書ける。
- **val 選択は hash で snapshot 化** (`val_hash`)。実データ投入後に `stats.json.val_hash` を固定値テストに昇格すれば変更検知になる (現状は synthetic での再現性のみ検証)。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **filelist の形式**:
  - フォーマット: **TSV (header 付き, UTF-8, LF 改行)**
  - 列: `rel_path`, `n_samples`, `speaker_id`, `chapter_id`, `duration_sec`, `peak_dbfs`, `rms_dbfs`
  - パス: `<src-dir>` からの相対パス、POSIX 区切り (`/`)
  - 例: `train-clean-100/103/1241/103_1241_000000_000001.wav	72000	103	1241	3.00	-2.15	-21.34`
- **T-M2.1 (Dataset)**:
  - filelist 読み込み時は `Path(args.src_dir) / row["rel_path"]` で wav 絶対パスを構築
  - 音声読み込みは **`soundfile`** (`sf.read`)。torchaudio.load/info は使わない (2.11 で廃止)
  - **BucketSampler 実装に `n_samples` 列を使う** (毎 epoch ファイルを開かない)
  - **dBFS 統計 (`stats.json`)** で **peak 正規化** (= sox `norm` 等価) 方針を validate
  - speaker_id 列で M2.6 / M3.5 smoke の speaker-balance チェックを実装可能
  - **【M0 review 申し送り】min-duration=1.0s (24000 sample) は GAN segment_length 16384 (≈0.68s) より長いが、Diff segment_length 25600 (≈1.067s) より短い**。よって 1.0〜1.067s の train wav は Diff の 1 segment に満たず **反射 pad が必須**。dataset.py は反射 pad → random crop を前提に実装すること (M3 も同様)。
- **T-M3.4 (Post-filter)**:
  - `data/filelists/dev_postfilter.tsv` (200 utterances) を使う (本チケットで生成済み)
  - 同じスクリプトを M3.4 で再実行しない
- **T-M2.6, T-M3.5 (Smoke)**:
  - validation の 100 utterances を speaker-balanced で sample 済み (seed=42)
  - `data/filelists/val.tsv` の各 speaker_id がユニーク (1 話者 1 utterance)
- **T-M5.1, T-M5.2 (1 epoch smoke)**:
  - `train.tsv` の全量で 1 epoch 訓練。filter 後の正確な行数は `stats.json.train.n_utterances` を参照
- **T-M0.2 (Scaffold)** との連携:
  - `data/filelists/.gitkeep` を T-M0.2 で予約、本チケットで実 TSV をファイル commit
  - T-M0.2 の `.gitattributes` で `*.tsv eol=lf` を強制 (CRLF 化け回避)
- **LICENSE 連絡**:
  - `LICENSE-LibriTTS-R.md` をリポジトリ ROOT に配置 (attribution)
  - 内容: LibriTTS-R が CC BY 4.0 で配布、論文引用、データセット作成者 (Koizumi+ NTT) のクレジット

### 9.2 ユーザー操作の引き継ぎ (M0.3 のみ必須)
完了時にユーザーに以下を依頼:
1. LibriTTS-R のライセンス (CC BY 4.0) に同意し、https://www.openslr.org/141/ から `train_clean_100.tar.gz`, `train_clean_360.tar.gz`, `test_clean.tar.gz` をダウンロード
2. 任意の場所に展開 (例 `D:\datasets\LibriTTS_R\`, `~/data/LibriTTS_R/`)
3. ダウンロード先パスを Claude Code に伝達 → Claude Code が `uv run python scripts/prepare_libritts.py --src-dir <path>` を実行
4. (代替案) huggingface mirror が利用可能と判明した場合は `--download` flag による自動 DL に切替

### 9.3 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M0.3 の Acceptance チェックボックス
  - [ ] `docs/tickets/index.md` の T-M0.3 ステータス (`📝 pending` → `✅ completed`)
  - [ ] `docs/tickets/T-M0.3-libritts-r.md` 自身の `status:` と `updated:` フィールド
  - [ ] (必要時) `docs/implementation-plan.md` §4 Phase 0 のチェックボックス
  - [ ] (必要時) `docs/training.md` §5.1 / §5.3 の「test-clean-100」表記揺れ修正 (実装時に論文 §4.1 / Table 1 再確認の上)

### 9.4 Open question として残ったもの
- **実データ実行はユーザー DL 待ち**: スクリプト + テストは完成し synthetic で検証済だが、実際の `train.tsv` (~145k 行) / `test.tsv` (~4,824 行) / `val_hash` 固定値は **ユーザーが LibriTTS-R を DL し `uv run python scripts/prepare_libritts.py --src-dir <path>` を実行**するまで生成されない。本チケットを `in_review` とし、実データ実行後に `completed` へ。
- **test split 表記** ("test-clean-100" vs "test-clean"): ✅ 実装は `test-clean` で確定 (§6.1)。論文 PDF が手元にないため Table 1 の厳密確認のみ未了 — 現解釈で実害なし、PDF 入手時に再確認。
- **huggingface mirror の存在確認**: `mythicinfinity/libritts_r` 等が認証なし DL 可能かは未確認 (本チケットは openslr 手動 DL 前提を維持)。確認できれば `--download` flag を別チケットで追加する余地あり。
