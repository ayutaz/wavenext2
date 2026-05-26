---
id: T-M0.1
title: Python 環境セットアップ (uv + Python 3.13)
milestone: M0
phase: M0
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: []
blocks: [T-M0.2, T-M0.3]
related_docs:
  - docs/milestones.md#m01-python-環境
  - docs/implementation-plan.md
  - docs/open-questions.md
---

# T-M0.1: Python 環境セットアップ (uv + Python 3.13)

> **マイルストーン**: [M0](../milestones.md#m0-環境整備とデータ準備-作業量-small) / **サブタスク**: [M0.1](../milestones.md#m01-python-環境)
> **依存**: なし / **後続**: [T-M0.2](T-M0.2-scaffold.md), [T-M0.3](T-M0.3-libritts-r.md)

## 1. タスク目的とゴール

### 目的
WaveNeXt 2 再現実装プロジェクトの Python 実行環境を `uv` ベースで構築し、後続マイルストーン全体で再現性のある依存関係管理 (`pyproject.toml` + `uv.lock`) を確立する。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `pyproject.toml` (`requires-python = ">=3.13,<3.14"`) と `uv.lock` がリポジトリにコミットされている
- [ ] `.venv/` (Python 3.13.x) がローカルに生成され `.gitignore` で除外されている
- [ ] `uv sync` が clean clone 状態から成功する (再現性確認)
- [ ] `uv run python -c "import torch; print(torch.cuda.is_available())"` が `True` を返し、`torch.__version__ >= 2.10.0`
- [ ] `uv run python -c "import torchaudio; print(torchaudio.list_audio_backends())"` で `sox_io` を含むリストが返る
- [ ] `uv run python -c "import pyworld, librosa; print(pyworld.__version__, librosa.__version__)"` がエラーなく完走する
- [ ] T-M0.2 / T-M0.3 が `uv run` 経由でスクリプトを実行できる状態が引き継がれる

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `pyproject.toml` (PEP 621 形式、uv 管理)
  - `uv.lock` (uv 自動生成、要 commit)
  - `.python-version` (任意、`3.13` 一行のみ)
- 編集:
  - `.gitignore` (`.venv/`, `__pycache__/`, `*.pyc`, `paper.pdf`, `paper.txt`, `page_*.png` 等の除外を追加)
- 生成 (commit しない):
  - `.venv/` (uv が作成)

### 2.2 主要構造

`pyproject.toml` の構造 (雛形):

```toml
[project]
name = "wavenext2"
version = "0.0.1"
description = "PyTorch reimplementation of WaveNeXt 2 (arXiv:2605.25506)"
readme = "README.md"
requires-python = ">=3.13,<3.14"
license = { text = "MIT" }
dependencies = [
    "torch>=2.10",
    "torchaudio>=2.10",
    "numpy",
    "scipy",
    "librosa>=0.11.0",
    "soundfile",
    "pyyaml",
    "tensorboard",
    "matplotlib",
    "tqdm",
    "pymcd",
    "pyworld",
    "einops",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
]

[tool.uv]
# CUDA wheel index (Linux/Windows 用)。Apple Silicon では index 切り替え不要。
# torch >= 2.10 で cp313 wheel が提供される版を選択。
# 暫定: cu126 を採用。CUDA toolkit が cu128 / cu130 なら別 index に切り替える。
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform != 'darwin'" }]
torchaudio = [{ index = "pytorch-cu126", marker = "sys_platform != 'darwin'" }]
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| Python | `3.13` (`>=3.13,<3.14`) | `docs/milestones.md` §M0.1, `docs/implementation-plan.md` §1 |
| torch | `>= 2.10` (cp313 wheel 提供開始版) | `docs/milestones.md` §M0.1 acceptance |
| CUDA wheel | `cu126` (暫定。`nvidia-smi` の driver version に応じて `cu128`/`cu130` に切り替え) | PyTorch 公式 wheel index |
| librosa | `>= 0.11.0` (PyPI classifier が 3.13 まで明示) | `docs/milestones.md` §M0.1 |
| pyworld | 本家 (cp313 wheel あり、cp314 wheel 未提供) | `docs/milestones.md` §M0.1 |

### 2.4 アルゴリズム / 処理フロー

1. **uv の存在確認**: `uv --version`
   - 未導入の場合、Windows: `winget install astral-sh.uv -e` / Unix: `curl -LsSf https://astral.sh/uv/install.sh | sh` をユーザーに案内
2. **NVIDIA driver 確認**: `nvidia-smi` で CUDA driver version を取得し、対応する PyTorch wheel index (`cu126` / `cu128` / `cu130`) を決定
3. **Python 3.13 仮想環境作成**: `uv venv --python 3.13`
4. **`pyproject.toml` を作成** (§2.2 雛形)
5. **依存追加** (uv が `pyproject.toml` を更新):
   ```bash
   uv add "torch>=2.10" torchaudio numpy scipy "librosa>=0.11.0" \
          soundfile pyyaml tensorboard matplotlib tqdm pymcd pyworld einops
   uv add --group dev pytest pytest-cov ruff
   ```
6. **`uv sync`** で `uv.lock` を生成・固定
7. **Acceptance 検証** (§5.2 の 5 項目を順次実行)
8. **`.gitignore` 更新** (`.venv/` 等を除外)
9. **commit**: `pyproject.toml` + `uv.lock` + `.python-version` + `.gitignore` をコミット

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `pyproject.toml` / `.gitignore` 作成、`uv add` / `uv sync` 実行、Acceptance 検証 | general-purpose |
| Reviewer | 1 | `uv.lock` の commit 確認、`requires-python` 制限・torch version 検証、CUDA index 選択妥当性レビュー | general-purpose |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M0.2 / T-M0.3 は T-M0.1 完了後に着手)
- 並列実行する場合の最大並列数: 1 (本チケット自体の作業内で並列化する余地は乏しい)
- size=S かつシンプルな環境セットアップのため Tester 役は Reviewer に統合し、最小編成 (Implementer 1 + Reviewer 1) で OK

## 4. 提供範囲 (Scope)

### In Scope
- `uv` をパッケージマネージャとした Python 3.13 仮想環境構築
- `pyproject.toml` (PEP 621) + `uv.lock` の生成・コミット
- PyTorch CUDA wheel index の指定 (`cu126` 暫定)
- `docs/milestones.md` §M0.1 の Acceptance 4 項目 (Python バージョン / CUDA available / torchaudio sox_io / pyworld+librosa import) の自動検証
- `.gitignore` への `.venv/` 等の追加

### Out of Scope
- ディレクトリ scaffold (`src/`, `tests/`, `configs/`, ...) → **T-M0.2** で実施
- LibriTTS-R のダウンロード・展開・filelist 生成 → **T-M0.3** で実施
- CUDA toolkit / NVIDIA driver のインストール (OS レイヤの作業はユーザー側)
- GPU クラスタの環境 (M6 で別途構築)
- UTMOS / NISQA 用の別 venv 構築 (`docs/milestones.md` §リスク表で言及、必要時に別チケット化)
- CI ワークフロー (.github/workflows) のセットアップ (将来チケット化)

### Deliverable
- ファイル:
  - `pyproject.toml` (新規)
  - `uv.lock` (新規)
  - `.python-version` (新規、任意)
  - `.gitignore` (更新)
- 関数 / クラス: なし (環境構築のみ)
- ドキュメント差分:
  - `docs/milestones.md` §M0.1 の Acceptance チェックボックスを更新
  - `docs/tickets/index.md` の T-M0.1 ステータスを `completed` に更新

## 5. テスト項目

### 5.1 Unit テスト
**不要** (環境セットアップのため pytest で書くテストコードはなし)。代わりに §5.2 の e2e/結合検証を Bash で実行する。

### 5.2 e2e / 結合テスト
`docs/milestones.md` §M0.1 Acceptance に対応する Bash 検証スクリプト:

- [ ] **E1**: `uv sync` が `.venv/` と `uv.lock` を生成 (exit code 0)
- [ ] **E2**: `uv run python --version` が `Python 3.13.x` を表示 (`x >= 0`)
- [ ] **E3**: `uv run python -c "import torch; assert torch.__version__ >= '2.10', torch.__version__; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.is_available())"` が成功
- [ ] **E4**: `uv run python -c "import torchaudio; backends = torchaudio.list_audio_backends(); assert 'sox_io' in backends, backends; print(backends)"` が成功
- [ ] **E5**: `uv run python -c "import pyworld, librosa; print(pyworld.__version__, librosa.__version__)"` がエラーなく完走

GPU が無い環境 (CI 等) で E3 を実行する場合は `torch.cuda.is_available()` のチェックを `os.environ.get("WAVENEXT2_SKIP_CUDA_CHECK")` でスキップ可能にする (将来の CI 用)。本チケットでは検証する開発機に CUDA GPU がある前提。

### 5.3 Acceptance criteria (`docs/milestones.md` §M0.1 より転記)
- [ ] `uv sync` が成功 → `.venv/` と `uv.lock` が生成
- [ ] `uv run python --version` が `3.13.x` を表示
- [ ] `uv run python -c "import torch; print(torch.cuda.is_available())"` で True (torch >= 2.10.0)
- [ ] `uv run python -c "import torchaudio; print(torchaudio.list_audio_backends())"` で sox_io が含まれる (audio 正規化に必須)
- [ ] `uv run python -c "import pyworld, librosa; print(pyworld.__version__, librosa.__version__)"` でエラーなし

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **Windows 環境での CUDA + cuDNN セットアップ問題** | E3 で `torch.cuda.is_available() == False` | `nvidia-smi` の出力をユーザーに提示し、NVIDIA driver を最新版に更新するよう依頼。`torch.cuda` の初期化エラーは `python -c "import torch; torch.zeros(1).cuda()"` で詳細 stack trace を取得 |
| **uv 未インストール** | `uv venv` が失敗 | `uv --version` で事前チェック、未導入なら winget / curl でのインストール手順を案内 |
| **PyTorch wheel index 選択ミス (cu126 vs cu128 vs cu130)** | torch import 時に `RuntimeError: CUDA error` | `nvidia-smi` の driver version とテーブル ([PyTorch 公式 install matrix](https://pytorch.org/get-started/locally/)) を突き合わせて選択。`[tool.uv.sources]` を編集すれば差し替え可能 |
| **PyTorch free-threaded ビルド (`python3.13t`) を誤って選択** | 多くの依存ライブラリ (numpy, librosa) が free-threaded で未テスト | `.python-version` は `3.13` のみ書き、`uv venv --python 3.13` のように明示。`uv venv --python 3.13t` は使わない。`uv run python -c "import sys; print(sys.flags.no_gil)"` が `0` であることを確認 |
| **pyworld の wheel が cp313 で見つからない** | `uv add pyworld` が失敗 | pyworld 本家 (PyPI `pyworld>=0.3.4`) は cp313 wheel を提供している前提。万一無ければ §8.1 の代替案 (`pyworld-prebuilt` fork or Python 3.12 ダウングレード) を検討 |
| **librosa の依存 (numba) が numpy 2.x 非互換** | `import librosa` で ImportError | librosa 0.11.0 系は numpy 2.x 対応済。`numpy>=2,<3` を明示的にピンする必要があれば `pyproject.toml` に追記 |
| **`uv.lock` の OS 依存** | clone 環境で `uv sync` が再現しない | `uv` は cross-platform lock をサポートしている (v0.4+)。`uv.lock` をそのまま使う。差異が出たら `uv lock --upgrade` で再生成 |

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` §C8 はトレーニングハイパーパラメータの確定情報を持つが、Python バージョン・依存ライブラリ version は記載されていない。**本チケットの決定 (Python 3.13, torch>=2.10) を `docs/implementation-plan.md` §1 と `docs/milestones.md` §M0.1 で既に明文化済みなので、ここでは追加の決定なし**。
- CUDA toolkit version (cu126/128/130) は開発機に依存するため、本チケットでは `cu126` を暫定 default とし、必要時に `[tool.uv.sources]` の `index` URL を切り替える運用とする。

### 6.3 他チケットとの整合性
- **T-M0.2**: `src/` 以下に `__init__.py` を配置するが `pyproject.toml` の `[tool.setuptools.packages]` 等は不要 (本プロジェクトは library 化しないため `src` レイアウトをそのまま使用)。整合済み。
- **T-M0.3**: `scripts/prepare_libritts.py` を `uv run python` で実行する前提。本チケットで `uv run` のエントリポイントが動けば問題なし。
- **T-M1.x**: PyTorch / torchaudio の version に依存するテストを書く。**`torch>=2.10` を `pyproject.toml` に明記**しておけば、テスト側で `torch>=2.10` の API を使える。

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `pyproject.toml` の `requires-python = ">=3.13,<3.14"` が正しく設定されているか
- [ ] `pyproject.toml` の dependencies に `torch>=2.10` (バージョン下限) が明記されているか
- [ ] `uv.lock` がリポジトリに commit されているか (`.gitignore` に誤って入れていないか)
- [ ] `.gitignore` で `.venv/`, `__pycache__/`, `*.pyc`, 論文関連の一時ファイル (`paper.pdf` 等) を除外しているか
- [ ] CUDA wheel index (`[[tool.uv.index]]` / `[tool.uv.sources]`) が `nvidia-smi` の driver version に対応しているか
- [ ] §5.2 の E1〜E5 全ての e2e 検証が Bash で成功しているか (出力ログを残す)
- [ ] `docs/milestones.md` §M0.1 と本チケットの Acceptance に差異がないか (転記漏れ)
- [ ] 参考実装 (Vocos, WaveFit-PT, FastDiff, wavenext-impl) のコードを **コピーしていない** か (本チケットは pyproject 雛形のみで該当箇所はないが、CLAUDE.md ポリシー再確認)
- [ ] Python 3.13 を選んだ理由 (cp314 で pyworld wheel が無いこと) がチケット §6.1 や `docs/milestones.md` §M0.1 で記述されているか

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 |
|---|---|---|---|
| **conda + `environment.yml`** | CUDA toolkit ごと conda env に同梱できる | 解決速度が遅い、`pyproject.toml` の標準形式から逸脱、`uv.lock` 相当の lockfile 機構が弱い、`conda-forge` と `pip` のミックスでビルド再現性が落ちる | `docs/implementation-plan.md` §1 で uv 採用を既に決定済み |
| **`pyproject.toml` + Poetry** | エコシステムが成熟、企業利用例が多い | uv (Rust 実装) より 10〜100 倍遅い、Python 3.13 サポートが遅い、PyTorch CUDA wheel index 切り替えが複雑 | Poetry より uv の方が高速かつ PEP 621 ネイティブ。`docs/milestones.md` で uv 採用済み |
| **`requirements.txt` + venv (pip)** | 最もシンプル、Python 標準のみ | lockfile (`pip freeze`) が cross-platform でない、依存 resolver が弱い、開発依存と本番依存の分離が手動 | 再現性が要件のため不採用 |
| **Docker + uv** | 全環境を Docker image に同梱 → ホスト OS 非依存。M6 (GPU クラスタ) で特に有用 | 開発機での hot-reload 開発体験が悪化、GPU passthrough (`--gpus all`) の設定が必要 | **将来 (M6 移行時) に検討する余地あり**。本チケットではローカル開発を最優先するため見送り |
| **Python 3.12 にダウングレード** | numpy 2.x との互換性が枯れている、pyworld・librosa の wheel が確実に揃う、PyTorch 2.10 以前 (2.4/2.5) も使える | 寿命が短い (3.12 は 3.13 より早く EOL)、3.13 で改善した GIL 周りの将来性を捨てる | Python 3.13 は GA 済み・主要ライブラリの wheel も出揃った段階。3.13 を採用しつつ、wheel 問題で詰まった場合のみ 3.12 にフォールバック |
| **Python 3.14 を採用 (`pyworld-prebuilt` 等 fork で wheel 補完)** | 最新版、free-threaded ビルドも視野 | サードパーティ fork はメンテナンス不安定、論文再現性の文脈で「外部 fork に依存」する説明コストが高い、librosa 0.11.0 の classifier が 3.13 までしか明示していない | `docs/milestones.md` §M0.1 の方針通り 3.13 を採用。`pyworld` 本家が cp314 wheel を出した時点で改めて 3.14 への移行を検討 |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M0.1 は size=S と妥当。`pyproject.toml` 作成 + uv 操作 + 5 項目の Acceptance 検証で 1〜2 時間程度の作業量。**分割する必要なし**。
- **scaffold 分離の正当性**: T-M0.2 (ディレクトリ作成) を本チケットに統合する案もあったが、`pyproject.toml` のレビュー観点と `src/` ツリーのレビュー観点が独立しているため、別チケットとした方が PR が小さくなり健全。
- **インターフェース定義の見直し余地**: `uv` のエントリポイント (`uv run python`, `uv run pytest`) を全チケットで統一する方針は維持。`uv tool run` (グローバル tool) は本プロジェクトでは使わない。

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

- **インターフェース**: 全 Python スクリプト実行は `uv run python <script>.py` 経由とする。pytest 実行は `uv run pytest`。**`python` を直接呼ばない**。
- **設定値**:
  - Python: 3.13.x
  - torch: >= 2.10 (cp313 wheel)
  - CUDA: cu126 暫定 (環境次第で `[tool.uv.sources]` を差し替え)
- **依存追加方法**: 後続チケットで追加ライブラリが必要な場合は `uv add <package>` を実行 → `pyproject.toml` + `uv.lock` が更新されるので diff を commit に含める。`pip install` 直叩きは禁止。
- **注意事項**:
  - `.venv/` は OS 依存のためマシン間で共有しない。必ず `uv sync` で再生成する
  - PyTorch CUDA wheel index は `[tool.uv.sources]` で切り替える。開発機交代時に CUDA driver version が変わったら本チケットを再開して `[[tool.uv.index]]` を更新

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M0.1 の Acceptance チェックボックス 4 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M0.1 ステータスを `📝 pending` → `✅ completed` に更新
  - [ ] `docs/tickets/index.md` の M0 進捗サマリ (pending -1, completed +1) を更新
  - [ ] (該当時) `docs/implementation-plan.md` Phase 0 のチェックリストを更新
  - [ ] (該当時) `README.md` の「セットアップ手順」セクションを追加 (`uv sync` の案内)

### 9.3 Open question として残ったもの

- CUDA driver version によって `cu126` 以外の wheel index に切り替える必要が生じる場合がある。本チケットでは `cu126` を暫定採用したが、M6 (本格訓練) で GPU クラスタを確保した時点で再評価が必要。
- Apple Silicon (`sys_platform == 'darwin'`) では `[tool.uv.sources]` のマーカーで CUDA wheel index を回避する設計だが、実際に Mac 開発機で `uv sync` を試していない。Mac で開発する人が出てきた時点で検証。
- 将来 pyworld 本家が cp314 wheel を提供したら Python 3.14 への移行を検討する旨を `docs/open-questions.md` の補遺に追記するか要検討 (現状は本チケット §8.1 にのみ記載)。
