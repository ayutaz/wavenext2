---
id: T-M0.3
title: LibriTTS-R 取得と filelist 生成
milestone: M0
phase: M0
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M2.1, T-M5.1, T-M5.2]
related_docs:
  - docs/milestones.md#m03-libritts-r-取得
  - docs/implementation-plan.md
  - docs/training.md
---

# T-M0.3: LibriTTS-R 取得と filelist 生成

> **マイルストーン**: [M0](../milestones.md#m0-環境整備とデータ準備-作業量-small) / **サブタスク**: [M0.3](../milestones.md#m03-libritts-r-取得)
> **依存**: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M2.1](T-M2.1-dataset.md), [T-M5.1](T-M5.1-gan-1epoch.md), [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
LibriTTS-R (CC BY 4.0) の "train-clean-100" + "train-clean-360" + "test-clean" を取得・展開し、後続の Dataset 実装 (T-M2.1) が読み込める wav の filelist (`data/filelists/{train,val,test}.txt`) を生成する。同時に sample rate が 24 kHz であることを検証し、必要なら resample を案内する (LibriTTS-R は元から 24 kHz であるため通常 resample は不要)。

ユーザー操作が必須なマイルストーン (`docs/milestones.md` §M0.3) であり、ライセンス同意とダウンロード認証はユーザー側で実施する。Claude Code は展開・filelist 生成・整合性検証を担当する。

### ゴール
- [ ] `scripts/prepare_libritts.py` が完成し、`--src-dir` 引数で LibriTTS-R ルートを受け取って動作する
- [ ] `data/filelists/train.txt` が約 145,000 行で生成される (train-clean-100 + 360 から val を除いた残り)
- [ ] `data/filelists/val.txt` が 100 行 (train から seed 固定で hold-out した utterance)
- [ ] `data/filelists/test.txt` が 4,824 行 (test-clean 全件)
- [ ] 任意の wav を `torchaudio.load` で読めて `sample_rate == 24000` であることをサンプル検証 (10 件)
- [ ] train と val に重複がないこと (set 比較) をスクリプト内で assert

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `scripts/prepare_libritts.py` (CLI スクリプト本体)
- 新規 (出力): `data/filelists/train.txt`, `data/filelists/val.txt`, `data/filelists/test.txt`
- 編集: -

### 2.2 主要構造
```python
# scripts/prepare_libritts.py
import argparse
import random
import sys
from pathlib import Path

import torchaudio


def collect_wavs(root: Path, subsets: list[str]) -> list[Path]:
    """subsets 配下の wav を再帰収集して相対パスで返す"""
    ...


def verify_sample_rate(wav_paths: list[Path], n_check: int = 10) -> None:
    """サンプル n_check 件を torchaudio.info で 24 kHz チェック"""
    ...


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="LibriTTS-R 展開ルート (LibriTTS_R/train-clean-100/... を含む)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/filelists"),
                        help="filelist 出力先")
    parser.add_argument("--val-size", type=int, default=100,
                        help="train から hold-out する validation 数")
    parser.add_argument("--seed", type=int, default=42,
                        help="validation hold-out の再現性確保")
    parser.add_argument("--n-verify", type=int, default=10,
                        help="sample_rate=24000 をサンプル検証する件数")
    args = parser.parse_args()
    ...
```

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| sample_rate | 24000 Hz | `docs/training.md` §1.1, §1.2 |
| train subsets | `["train-clean-100", "train-clean-360"]` | `docs/training.md` §1.1 |
| test subset | `["test-clean"]` | `docs/milestones.md` §M0.3 |
| validation 数 | 100 | `docs/training.md` §6 (validation utterances 数: 100) |
| 想定 train 行数 | 約 145,000 | `docs/milestones.md` §M0.3 Acceptance |
| 想定 test 行数 | 4,824 | `docs/milestones.md` §M0.3 Acceptance |
| ランダム seed | 42 (固定) | 後続スモークでの再現性確保 |

### 2.4 アルゴリズム / 処理フロー
1. CLI 引数 (`--src-dir`, `--out-dir`, `--val-size`, `--seed`, `--n-verify`) をパース
2. `--src-dir` 配下に `train-clean-100/`, `train-clean-360/`, `test-clean/` が存在するか検証 (なければ error)
3. 各 subset を `pathlib.Path.rglob("*.wav")` で再帰収集し、`--src-dir` からの **相対パス** に変換 (リポジトリポータビリティ)
4. パス文字列を **POSIX 形式** (`/` 区切り) に統一 (Windows での `\` 混入を避ける)
5. train pool = train-clean-100 + train-clean-360 の全パス。`sorted()` で順序を固定
6. `random.Random(seed)` で train pool から `--val-size` 件を抽出 → val.txt
7. train.txt = train pool - val (set 差集合、`assert len(train) + len(val) == len(train_pool)`)
8. test.txt = test-clean 全件 (sorted)
9. `--n-verify` 件を train pool から無作為 (別 seed) に抽出し、`torchaudio.info(path)` で `sample_rate == 24000` を assert。異なるものがあれば error log + raise
10. `--out-dir` を作成し 3 ファイルを書き出し (UTF-8, LF 改行)
11. 完了サマリ (各行数、検証通過件数) を stdout に出力

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

filelist は 1 行 1 wav 相対パスで以下の形式:
```
train-clean-100/103/1241/103_1241_000000_000001.wav
train-clean-100/103/1241/103_1241_000000_000002.wav
...
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `scripts/prepare_libritts.py` 実装 + Acceptance 自動検証 | general-purpose |
| Tester | 1 | ダウンロード成否確認、整合性 (重複なし / 24 kHz) の e2e 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M0.1 環境構築と T-M0.2 scaffold が前提)
- 並列実行する場合の最大並列数: 1
- 注意: **ユーザー操作 (LibriTTS-R のライセンス同意 + ダウンロード) が必須** のため、Claude Code はユーザーがダウンロード先パスを伝えるまで待機する。
  - 代替案: huggingface mirror (例 `https://huggingface.co/datasets/openslr/librittsr` 等) があれば認証不要で `uv run` 経由の自動 DL も検討可能。要存在確認 (§6 参照)。

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/prepare_libritts.py` (CLI スクリプト)
- `data/filelists/train.txt`, `val.txt`, `test.txt` の生成
- 24 kHz 検証 (サンプル 10 件)
- 再現性確保のための seed 固定 (default 42)
- Windows / POSIX どちらでも動くパス処理 (出力は POSIX 形式)

### Out of Scope
- LibriTTS-R 本体のダウンロード自動化 (ユーザーが手動 DL する前提。huggingface mirror による自動 DL 対応は §8 検討事項に残す)
- 24 kHz 以外からの resample (LibriTTS-R は元から 24 kHz なので不要)
- mel 抽出 (`scripts/extract_mel.py` は別タスク。本タスクでは filelist のみ生成)
- Dataset class 実装 (T-M2.1 で別途実装)
- ファイル単位の音質チェック (clipping 検出など。スコープ外)

### Deliverable
- ファイル: `scripts/prepare_libritts.py`, `data/filelists/{train,val,test}.txt`
- 関数 / クラス: `collect_wavs`, `verify_sample_rate`, `main`
- ドキュメント差分:
  - `docs/milestones.md` §M0.3 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M0.3 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト
不要 (CLI スクリプトのため最小限の動作確認は e2e で十分)。ただし内部関数 `collect_wavs` / `verify_sample_rate` の挙動が想定外だった場合は、後追いで `tests/test_prepare_libritts.py` を追加することを検討。

### 5.2 e2e / 結合テスト
- [ ] **サンプル wav 1 個での smoke**: `scripts/prepare_libritts.py --src-dir <tiny>` がエラーなく完走 (tiny は test-clean から数 wav だけ取り出した tmpdir でも可)
- [ ] **filelist 行数**: 生成された `train.txt` が約 145,000 行、`val.txt` が 100 行、`test.txt` が 4,824 行 (Acceptance criteria 参照)
- [ ] **24 kHz 検証**: `torchaudio.load(wav).sample_rate == 24000` をサンプル 10 件で確認
- [ ] **重複なし**: `set(train) & set(val) == empty` を assert
- [ ] **相対パス**: 出力 filelist の各行が `--src-dir` からの相対パス (絶対パス混入なし)、`/` 区切り

### 5.3 Acceptance criteria (`docs/milestones.md` §M0.3 より転記)
- [ ] `train.txt` に約 145k 行 (train-clean-100 + 360 で約 460h)
- [ ] `test.txt` に 4,824 行
- [ ] 任意の wav ファイルを `torchaudio.load` で読めて `sample_rate=24000` を確認

## 6. 懸念事項

### 6.1 技術的リスク
- **ディスク容量**: LibriTTS-R 解凍後合計 ~50 GB。ユーザー側に空き容量がないとダウンロード途中で失敗する。事前に `df -h` / `Get-PSDrive` で確認するよう案内すること。
- **ネットワーク帯域**: openslr.org からの DL は数 GB × 3 ファイルで、低速回線だと数時間かかる。タイムアウト時は再開可能な `wget -c` / `curl -C -` の使用を推奨。
- **Windows パスの `\` vs `/` 問題**: Windows の `pathlib.Path` は `\` で出力するが、filelist は POSIX 形式 `/` で統一する (`Path.as_posix()` を使用)。後続 Dataset 実装 (T-M2.1) が `Path` で受ければ問題ないが、文字列比較するコードを書く場合の罠。
- **subset ディレクトリ名のゆれ**: openslr 版は `train-clean-100` (ハイフン)、huggingface mirror 版は `train_clean_100` (アンダースコア) など命名が違うことがある。`--src-dir` 配下を `rglob` で検出するロジックは subset 名指定ではなく、`*.wav` の親ディレクトリパターンで検出する方が頑健かもしれない (要設計判断)。
- **読めない wav (corruption)**: 一部の wav が壊れていた場合、`torchaudio.info` で例外が出る。`--n-verify` でサンプル検証時にこれが出たら, error log で該当パスを示してユーザーに再 DL を促す。

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` で LibriTTS-R に関する未解決事項はなし (データセット自体は標準的)
- ただし以下はこのチケット内で確定させる:
  - **validation 100 utterances の選択方法**: 全 train pool から `random.Random(42).sample(pool, 100)` で確定。speaker balance などは取らない (シンプルさ優先)
  - **filelist のパス形式**: 相対パス (POSIX 形式) を採用。絶対パスは環境依存になるため避ける
  - **train/val/test 分割**: openslr の標準 split に従う (test-clean は train から完全分離)

### 6.3 他チケットとの整合性
- **T-M2.1 (Dataset)**: filelist の読み込み側。本チケットで決めた「相対パス + POSIX 区切り」の規約を共有する必要あり。Dataset 側で `Path(root) / line.strip()` のように結合する設計と整合させる
- **T-M2.6 (GAN smoke), T-M3.5 (Diff smoke)**: smoke で validation 100 utterances を seed 固定で参照するため、本チケットで生成した `val.txt` を使用
- **T-M5.1, T-M5.2 (1 epoch smoke)**: train.txt の全量で 1 epoch 訓練するため、行数 (約 145k) が大幅にズレるとステップ数見積もりに影響

## 7. レビュー観点

実装完了後、Reviewer (もしくは Self-review) が以下を確認:

- [ ] filelist のパスが **相対パス** で統一されている (絶対パス混入なし)
- [ ] パス区切り文字が `/` (POSIX 形式) で統一されている (Windows の `\` が残っていない)
- [ ] **24 kHz 確認** が抜けていない (`torchaudio.info` でサンプル検証)
- [ ] val が train から正しく除外されている (`set(train) & set(val) == set()`)
- [ ] **seed 固定** で再現可能 (`random.Random(seed)` を使用、`random.seed()` グローバル汚染を避ける)
- [ ] CLI 引数のデフォルト値が妥当 (`--val-size=100`, `--seed=42`, `--n-verify=10`)
- [ ] エラー処理: `--src-dir` 不存在、subset ディレクトリ不存在、wav 1 個も見つからない、24 kHz 以外を検出 — 各ケースで明示的にエラーメッセージを出す
- [ ] CLAUDE.md 既存スタイル準拠 (型ヒント、docstring、命名)
- [ ] `data/filelists/` ディレクトリが `.gitignore` 対象か明示 (大きいファイルではないので commit してよいか方針確認)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら
- **別案 A: huggingface datasets を直接利用** (`datasets.load_dataset("openslr/librittsr")`)
  - メリット: filelist 不要、ストリーミング可能、認証不要 (mirror 版があれば)
  - デメリット: 既存 mirror の存在確認が必要、`Dataset` interface が huggingface 依存になり Vocos / WaveFit-PT 流のシンプルさが失われる
  - 採用しなかった理由: 論文再現実装としてオリジナル openslr 版に従う方が他実装との比較がブレない
- **別案 B: lmdb / tar 形式に変換して I/O 高速化**
  - メリット: 大規模訓練 (M6 のフル 410h) で I/O ボトルネック軽減
  - デメリット: 前処理ステップが増え、デバッグ性が下がる
  - 採用しなかった理由: M0.3 は最小限のスモーク用。本格訓練段階 (M6 直前) で必要なら別タスクとして実装
- **別案 C: WebDataset 形式 (huge dataset shard 化)**
  - メリット: 分散訓練・クラウドストリーミングに最適
  - デメリット: shard 設計が必要、ローカル開発との非対称性
  - 採用しなかった理由: B と同じ理由
- **別案 D: 分散ストレージ (S3 / GCS) からの直接ストリーミング**
  - メリット: ローカルディスク不要、クラウド A100 (M6) と相性が良い
  - デメリット: ネットワーク I/O コスト、認証管理が複雑
  - 採用しなかった理由: M6 で GPU クラスタが確定してから検討する方が無駄がない

### 8.2 思想 / 哲学の見直し
- このサブタスクの粒度は適切 (small)。filelist 生成のみで他要素 (mel 抽出など) を含まないため境界が明確
- インターフェース定義: 「相対パス POSIX 形式 1 行 1 wav」というシンプルな規約は他チケットからも参照しやすい
- 改善余地: speaker / chapter 分布を JSON で別途出力して T-M2.1 で speaker balanced batch sampler を実装する余地を残す (本チケットでは scope 外)

### 8.3 LibriTTS-R 以外のデータセット候補 (将来的な ablation 用)
- **VCTK** (英語、109 話者、48 kHz): 多話者性で勝るが論文と sample rate が異なる
- **JVS** (日本語、100 話者): 日本語適用の ablation に
- **LJSpeech** (英語、1 話者、22 kHz): デバッグ用の単一話者
- **Common Voice**: より多言語、品質はやや劣る
- いずれも論文と異なるため M0 では採用しないが、M6 以降で「他データセットでの再現性」を見るときに使えるかもしれない

### 8.4 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未記入)
- 次の似たタスクで応用できる教訓: (未記入)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **filelist の形式**:
  - パス: `<src-dir>` からの相対パス、POSIX 区切り (`/`)、UTF-8、LF 改行
  - 例: `train-clean-100/103/1241/103_1241_000000_000001.wav`
- **T-M2.1 (Dataset)**: filelist 読み込み時は `Path(args.src_dir) / line.strip()` で wav 絶対パスを構築。空行・コメント行は無し前提だが念のため `if line.strip()` でフィルタ推奨
- **T-M2.6, T-M3.5 (Smoke)**: `data/filelists/val.txt` の 100 行を validation セットとして固定参照。再生成する場合は同じ seed (42) を使うこと
- **設定値**: 訓練設定 YAML (`configs/gan_wavenext2.yaml`, `configs/diff_wavenext2.yaml`) の `data.root` に LibriTTS-R 展開ルートを設定する

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

### 9.4 Open question として残ったもの
- **huggingface mirror の存在確認**: `openslr/librittsr` 等の mirror が実際に存在し authentication なしで DL 可能か未確認。確認できれば `docs/open-questions.md` に追記し、`--download` flag を別チケットで実装する余地あり
- **`data/filelists/` を git commit するか**: filelist は数 MB 程度だが環境依存性があるため、`.gitignore` で除外して各環境で再生成する方針が望ましい。最終判断は実装時に
