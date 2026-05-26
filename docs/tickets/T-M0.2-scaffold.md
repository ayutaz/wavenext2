---
id: T-M0.2
title: src/wavenext2/ パッケージ scaffold 作成
milestone: M0
phase: M0
status: completed
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.1]
blocks: [T-M1.1, T-M1.2, T-M1.3, T-M1.4, T-M1.5, T-M1.6, T-M2.1, T-M2.2, T-M2.3, T-M4.1, T-M4.2]
related_docs:
  - docs/milestones.md#m02-ディレクトリ-scaffold
  - docs/implementation-plan.md
  - docs/architecture.md
---

# T-M0.2: src/wavenext2/ パッケージ scaffold 作成

> **マイルストーン**: [M0](../milestones.md#m0-環境整備とデータ準備-作業量-small) / **サブタスク**: [M0.2](../milestones.md#m02-ディレクトリ-scaffold)
> **依存**: [T-M0.1](T-M0.1-python-env.md) / **後続**: T-M1.1〜T-M1.6, T-M2.1〜T-M2.3, T-M4.1, T-M4.2 ほか

## 1. タスク目的とゴール

### 目的
後続マイルストーン (M1〜M4) が各モジュールを実装する **置き場所** を確定させる。`docs/implementation-plan.md` §3 と `docs/milestones.md` §M0.2 で示されたディレクトリツリーを `src/wavenext2/` パッケージ化方針に基づいて忠実に再現し、空の `__init__.py` と TODO スタブのみで構成された .py / .yaml ファイル群を一括 commit する。

加えて、M0 フェーズレビュー (3 エージェント合議) で確定した以下を本チケットに含める:
- `src/wavenext2/` パッケージ化 (旧 `src/` flat 構造からの格上げ、後述 §8.1)
- Docker / devcontainer の placeholder stub
- CI (GitHub Actions) / pre-commit の skeleton
- 改行コード強制用 `.gitattributes`
- `.env.example` placeholder
- `tests/conftest.py` + `test_placeholder` で pytest exit 5 回避
- `data/{filelists,raw,cache,processed}/.gitkeep` (T-M0.3 と整合)

### ゴール
- [ ] `src/wavenext2/{data,models,losses,train,inference,eval,utils}/` の 7 サブパッケージが作成され、それぞれに `__init__.py` (空、`__all__` 予約コメントのみ) が存在
- [ ] 各 .py モジュール (e.g., `convnext.py`, `stft.py`, `generator.py` ...) が TODO スタブ付きで存在 (本実装は後続チケットで埋める)
- [ ] `pyproject.toml` の `[tool.hatch.build.targets.wheel] packages = ["src/wavenext2"]` (または uv workspace 相当) が設定され、`uv pip install -e .` で `wavenext2` が import 可能
- [ ] `configs/{gan_wavenext2.yaml, diff_wavenext2.yaml}` の雛形が `docs/implementation-plan.md` §5 の内容を反映して commit 済み (secrets は含まない)
- [ ] `tests/conftest.py` (空) と `tests/test_*.py` 雛形 (`pytest.skip("M1.x で実装")` placeholder) が存在
- [ ] `scripts/{prepare_libritts.py, extract_mel.py, fit_post_filter.py}` が TODO スタブ付きで存在
- [ ] `checkpoints/.gitkeep`, `logs/.gitkeep`, `data/{filelists,raw,cache,processed}/.gitkeep` が存在し、空ディレクトリを git に commit できる
- [ ] `Dockerfile`, `.dockerignore`, `.devcontainer/devcontainer.json` の skeleton (TODO placeholder) が存在
- [ ] `.github/workflows/{test.yml, lint.yml}` の CI skeleton が存在
- [ ] `.pre-commit-config.yaml` の skeleton が存在
- [ ] `.gitattributes` (`* text=auto eol=lf`) が存在
- [ ] `.env.example` (placeholder のみ、secrets 値なし) が存在
- [ ] `uv run python -c "import wavenext2.models, wavenext2.data, wavenext2.losses, wavenext2.train, wavenext2.inference, wavenext2.eval, wavenext2.utils"` がエラーなく成功
- [ ] `uv run pytest tests/ --collect-only` で test_placeholder が collect される (exit code 0)
- [ ] `docs/tickets/index.md` の本チケットステータスを更新済み

## 2. 実装内容の詳細

### 2.1 対象ファイル (新規作成)

#### src/wavenext2/ パッケージ (パッケージ化方針: §8.1 参照)
- `src/wavenext2/__init__.py` (空、`__all__ = []` を予約コメントで明示)
- `src/wavenext2/data/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/data/dataset.py` (TODO stub)
- `src/wavenext2/data/mel.py` (TODO stub。**mel 抽出パラメータ (n_fft, hop_length, win_length, n_mels) は config から受け取る関数として実装する旨を docstring に明記** — §6.1 参照)
- `src/wavenext2/models/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/models/convnext.py` (TODO stub)
- `src/wavenext2/models/stft.py` (TODO stub)
- `src/wavenext2/models/generator.py` (TODO stub)
- `src/wavenext2/models/sub_model.py` (TODO stub)
- `src/wavenext2/models/noise_embedding.py` (TODO stub)
- `src/wavenext2/models/discriminator.py` (TODO stub)
- `src/wavenext2/models/gan_wavenext2.py` (TODO stub)
- `src/wavenext2/models/diff_wavenext2.py` (TODO stub)
- `src/wavenext2/losses/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/losses/adversarial.py` (TODO stub)
- `src/wavenext2/losses/feature_matching.py` (TODO stub)
- `src/wavenext2/losses/stft_loss.py` (TODO stub)
- `src/wavenext2/train/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/train/train_gan.py` (TODO stub)
- `src/wavenext2/train/train_diff.py` (TODO stub)
- `src/wavenext2/inference/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/inference/infer_gan.py` (TODO stub)
- `src/wavenext2/inference/infer_diff.py` (TODO stub)
- `src/wavenext2/inference/post_filter.py` (TODO stub)
- `src/wavenext2/eval/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/eval/compute_metrics.py` (TODO stub)
- `src/wavenext2/eval/run_utmos.py` (TODO stub)
- `src/wavenext2/eval/run_nisqa.py` (TODO stub)
- `src/wavenext2/eval/measure_rtf.py` (TODO stub)
- `src/wavenext2/utils/__init__.py` (空、`__all__` 予約)
- `src/wavenext2/utils/config.py` (TODO stub)
- `src/wavenext2/utils/logging.py` (TODO stub)
- `src/wavenext2/utils/scheduler.py` (TODO stub)
- `src/wavenext2/utils/seed.py` (TODO stub、PyTorch / numpy / random / CUDA seed 一括設定の予告 docstring。T-M0.1 連絡事項より)

#### pyproject.toml (T-M0.1 で既に存在する想定、本チケットで追記)
- `[tool.hatch.build.targets.wheel] packages = ["src/wavenext2"]` を追加
- (または uv workspace 機能を使う場合は対応セクションを追加)
- `[project]` の `name = "wavenext2"` を確認

#### configs/
- `configs/gan_wavenext2.yaml` (雛形を `docs/implementation-plan.md` §5 からコピー。**secrets (wandb_api_key 等) は書かない** — §6.1 参照)
- `configs/diff_wavenext2.yaml` (同上)

#### tests/
- `tests/conftest.py` (空、M1 で fixture 追加時の diff を小さくする)
- `tests/test_convnext.py` (`def test_placeholder(): pytest.skip("M1.1 で実装")`)
- `tests/test_stft_module.py` (`def test_placeholder(): pytest.skip("M1.2 で実装")`)
- `tests/test_generator.py` (`def test_placeholder(): pytest.skip("M1.4 で実装")`)
- `tests/test_sub_model.py` (`def test_placeholder(): pytest.skip("M1.6 で実装")`)
- `tests/test_dataset.py` (`def test_placeholder(): pytest.skip("M2.1 で実装")`)
- `tests/test_post_filter.py` (`def test_placeholder(): pytest.skip("M3.4 で実装")`)
- `tests/__init__.py` は **作成しない** (pytest は test_ prefix で自動収集、namespace package 化を避ける)

#### scripts/
- `scripts/prepare_libritts.py` (TODO stub、後続 T-M0.3 で実装)
- `scripts/extract_mel.py` (TODO stub、後続 T-M2.1 で実装)
- `scripts/fit_post_filter.py` (TODO stub、後続 T-M3.4 で実装)
- `scripts/scaffold.py` (本チケットの一括生成用、冪等性を担保。`if __name__ == "__main__":` guard 必須。再実行時に既存ファイルを上書きしない `Path.exists()` チェックを入れる)

#### Docker / devcontainer (空 stub commit、後続 M5/M6 で完成)
- `Dockerfile` (内容: `# TODO: M5/M6 で完成 — uv + CUDA 12.x base image` のヘッダコメントのみ)
- `.dockerignore` (内容: 既存 `.gitignore` 相当の最小セット, `.git/`, `__pycache__/`, `*.pyc`, `data/`, `checkpoints/`, `logs/` など)
- `.devcontainer/devcontainer.json` (内容: uv + CUDA 12.x base image を想定した最小 skeleton、`image` / `features` / `postCreateCommand` 等の field を TODO で埋める)

#### CI / pre-commit (空 stub commit)
- `.github/workflows/test.yml` (`setup-uv@v3` + `actions/cache@v4` + `uv sync` + `uv run pytest` のテンプレ。`pytest` step 自体は TODO コメントで本実装を予告)
- `.github/workflows/lint.yml` (ruff + mypy 想定の skeleton。`uv run ruff check` / `uv run mypy src/wavenext2` の TODO step)
- `.pre-commit-config.yaml` (ruff format + ruff check + `pre-commit-hooks` (trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict) の major hook を列挙)

#### 改行コード / secrets / その他
- `.gitattributes` (`* text=auto eol=lf`、`*.ps1 text eol=crlf` など Windows-only ファイルは個別指定)
- `.env.example` (内容: `WANDB_API_KEY=`, `HF_TOKEN=`, `OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=0` 等の placeholder のみ、**値は書かない**。`.env` は既存 `.gitignore` で除外済み)

#### 空ディレクトリ commit
- `checkpoints/.gitkeep`
- `logs/.gitkeep`
- `data/.gitkeep` (T-M0.3 と整合)
- `data/filelists/.gitkeep` (LibriTTS-R train/val/test の filelist 配置先)
- `data/raw/.gitkeep` (LibriTTS-R wav 配置先、wav 自体は `.gitignore` で除外済み想定)
- `data/cache/.gitkeep` (precomputed mel cache、M2.1 で利用)
- `data/processed/.gitkeep` (将来 mel/特徴量 cache、M2.1 で利用検討)

### 2.2 主要構造 (各 .py スタブの共通テンプレート)

```python
"""<モジュール名>: <一行説明>

TODO: T-MX.Y で実装。
詳細は docs/architecture.md / docs/milestones.md を参照。
"""

# このファイルは scaffold 段階のスタブです。
# 実装は後続チケットで行います。
```

`__init__.py` は **基本空** で commit する。ただし `__all__ = []` を予約コメントとして 1 行入れる:

```python
"""wavenext2.<sub> public API.

TODO: 後続チケットで `__all__` に re-export を追加。
"""

# __all__ = []  # 後続チケットで明示
```

これにより public API 境界を最初から意識でき、後続が無自覚に internal helper を import する事故を防ぐ (§8.2 参照)。

### 2.3 ディレクトリツリー (完成形)

```
wavenext2/
├── .devcontainer/
│   └── devcontainer.json
├── .dockerignore
├── .env.example
├── .gitattributes
├── .github/
│   └── workflows/
│       ├── test.yml
│       └── lint.yml
├── .pre-commit-config.yaml
├── Dockerfile
├── checkpoints/
│   └── .gitkeep
├── configs/
│   ├── gan_wavenext2.yaml
│   └── diff_wavenext2.yaml
├── data/
│   ├── .gitkeep
│   ├── cache/
│   │   └── .gitkeep
│   ├── filelists/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docs/                       # 既存
├── logs/
│   └── .gitkeep
├── pyproject.toml              # 既存 (T-M0.1)、packages 設定を追記
├── scripts/
│   ├── prepare_libritts.py
│   ├── extract_mel.py
│   ├── fit_post_filter.py
│   └── scaffold.py
├── src/
│   └── wavenext2/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── mel.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── convnext.py
│       │   ├── stft.py
│       │   ├── generator.py
│       │   ├── sub_model.py
│       │   ├── noise_embedding.py
│       │   ├── discriminator.py
│       │   ├── gan_wavenext2.py
│       │   └── diff_wavenext2.py
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── adversarial.py
│       │   ├── feature_matching.py
│       │   └── stft_loss.py
│       ├── train/
│       │   ├── __init__.py
│       │   ├── train_gan.py
│       │   └── train_diff.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── infer_gan.py
│       │   ├── infer_diff.py
│       │   └── post_filter.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── compute_metrics.py
│       │   ├── run_utmos.py
│       │   ├── run_nisqa.py
│       │   └── measure_rtf.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logging.py
│           ├── scheduler.py
│           └── seed.py
└── tests/
    ├── conftest.py
    ├── test_convnext.py
    ├── test_stft_module.py
    ├── test_generator.py
    ├── test_sub_model.py
    ├── test_dataset.py
    └── test_post_filter.py
```

### 2.4 作成方法
- **推奨**: Python スクリプト (`scripts/scaffold.py`) で一括生成。**冪等性必須** — `if __name__ == "__main__":` guard を入れ、各ファイルは `Path.exists()` チェックで既存を上書きしない (§6.1)。
- 代替: PowerShell `New-Item` または Bash `touch + mkdir -p` で逐次作成。Windows 環境でもパスは forward slash で統一する (Python `pathlib.Path` を使う場合は OS 非依存)。
- **注意**: `scripts/scaffold.py` は **commit する** (再現性 / drift 検知のため、§8.1 manifest 案も将来検討)。

### 2.5 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| パッケージ root | `src/wavenext2/` (旧 `src/` flat から昇格) | docs/implementation-plan.md §3 + M0 レビュー |
| import 形式 | `from wavenext2.models.convnext import ConvNeXtBlock` | M0 レビュー §8.1 |
| サブパッケージ数 | 7 (data, models, losses, train, inference, eval, utils) | docs/milestones.md §M0.2 |
| .py スタブ総数 | 約 28 ファイル (utils/seed.py を含む) | docs/milestones.md §M0.2 + M0 レビュー |
| YAML 雛形数 | 2 (GAN / Diff) | docs/implementation-plan.md §5 |
| テストファイル数 | 6 + conftest.py | docs/milestones.md §M0.2 + M0 レビュー |
| スクリプト数 | 4 (scaffold.py を含む) | M0 レビュー |
| CI workflow 数 | 2 (test.yml, lint.yml) | M0 レビュー |
| `.gitkeep` 配置数 | 6 (checkpoints, logs, data, data/filelists, data/raw, data/cache, data/processed) | T-M0.3 整合 |

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | scaffold 一括生成 (Python or shell)、各ファイル作成 | general-purpose |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M0.1 完了後でないと `uv run` 検証ができない)
- 並列実行する場合の最大並列数: 1 (機械的タスクなので分割の意味がない)

## 4. 提供範囲 (Scope)

### In Scope
- ディレクトリツリーの作成 (`src/wavenext2/` パッケージ構造)
- 全 `__init__.py` を `__all__` 予約コメント付きの空相当で配置
- 各 .py モジュールを TODO スタブで配置 (本実装は後続)
- `pyproject.toml` の `packages = ["src/wavenext2"]` 設定追加 (`uv pip install -e .` 動作確認まで)
- `configs/*.yaml` の雛形コピー (secrets 含まない)
- Docker / devcontainer の skeleton commit
- CI / pre-commit の skeleton commit
- `.gitattributes` / `.env.example` commit
- `tests/conftest.py` + `pytest.skip` placeholder で pytest collect 通過
- 空ディレクトリ commit 用 `.gitkeep` 配置 (`data/{filelists,raw,cache,processed}` 含む)
- `docs/tickets/index.md` のステータス更新

### Out of Scope
- 各 .py モジュールの本実装 (M1〜M4 各チケット)
- テストロジックの本実装 (M1〜M2 各チケット、本チケットでは `pytest.skip` placeholder のみ)
- LibriTTS-R ダウンロードロジック (T-M0.3)
- post-filter 実装 (T-M3.4)
- `__init__.py` の re-export (必要なら後続チケットで `__all__` に追加)
- Docker / devcontainer の本実装 (M5/M6 で完成、本チケットでは placeholder のみ)
- CI workflow の本実装 (test step / lint step の中身は本チケットでは TODO コメント)
- pre-commit hook の有効化 (`pre-commit install` は本チケットでは行わない、設定ファイル commit のみ)

### Deliverable
- ファイル: 上記 §2.1 の全ファイル
- パッケージ構造: `wavenext2.models`, `wavenext2.data`, `wavenext2.losses`, `wavenext2.train`, `wavenext2.inference`, `wavenext2.eval`, `wavenext2.utils` の 7 パッケージが `uv pip install -e .` 後に import 可能
- ドキュメント差分: `docs/tickets/index.md` の T-M0.2 ステータス更新のみ

## 5. テスト項目

### 5.1 Unit テスト
- 不要 (構造作成のみで実装ロジックがない)

### 5.2 e2e / 結合テスト
- [ ] `uv pip install -e .` がエラーなく成功 (パッケージ化検証)
- [ ] `uv run python -c "import wavenext2.models, wavenext2.data, wavenext2.losses, wavenext2.train, wavenext2.inference, wavenext2.eval, wavenext2.utils"` がエラーなく成功 (旧 `src.models` ではなく `wavenext2.models` であることを確認)
- [ ] PowerShell: `(Get-ChildItem -Path src,tests,scripts,configs -Recurse -File -Include *.py,*.yaml | Measure-Object).Count` で期待ファイル数が一致
- [ ] Bash 環境では `find src tests scripts configs -type f \( -name "*.py" -o -name "*.yaml" \) | wc -l` で同等の確認
- [ ] `uv run pytest tests/ --collect-only` で `test_placeholder` が 6 個 collect される (exit code 0、exit 5 でない)
- [ ] `uv run pytest tests/` がエラーなく完了 (全 placeholder が skip、failure 0)
- [ ] `git status` で全新規ファイルが untracked として表示される
- [ ] `git check-attr eol -- README.md` で `eol: lf` が返り `.gitattributes` 効いていることを確認
- [ ] `python scripts/scaffold.py` を再実行して既存ファイルが上書きされないこと (冪等性確認)

### 5.3 Acceptance criteria (`docs/milestones.md` §M0.2 より転記)
- [ ] 上記ツリーが作成され、各 `__init__.py` が `__all__` 予約コメント付きで存在

### 5.4 追加 acceptance (本チケット独自)
- [ ] `configs/gan_wavenext2.yaml` と `configs/diff_wavenext2.yaml` の内容が `docs/implementation-plan.md` §5 と整合 (key の typo がない、secrets 値を含まない)
- [ ] `.gitkeep` 配置により `git add checkpoints/ logs/ data/` で空ディレクトリが commit 可能
- [ ] Windows 環境でも import path にバックスラッシュ起因のエラーがない
- [ ] `pyproject.toml` に `packages = ["src/wavenext2"]` (または等価設定) が含まれる
- [ ] `.env.example` に secrets 値が含まれないこと (placeholder のみ)
- [ ] `Dockerfile`, `.devcontainer/devcontainer.json`, `.github/workflows/*.yml`, `.pre-commit-config.yaml` がそれぞれ TODO コメント付きで存在 (本実装は不要)

## 6. 懸念事項

### 6.1 技術的リスク
- **`__init__.py` の中身**: 基本空とするが `__all__ = []` 予約コメントを入れる方針 (§2.2)。後続チケットで re-export を追加する際は `__all__` に明示することで public API 境界を保つ (§8.2 参照)。
- **Windows パス区切り**: PowerShell / Bash 混在環境で動作する必要がある。Python `pathlib.Path` を使うか、forward slash で統一する。
- **CRLF / LF 混在 (Windows 開発者起因)**: `git config core.autocrlf=true` が有効な Windows 環境で CRLF に化けるリスク。本チケットで `.gitattributes` に `* text=auto eol=lf` を commit して LF を強制。`*.ps1` 等の Windows-only ファイルは個別に `eol=crlf` 指定。
- **`__pycache__/` 等のキャッシュ**: `.gitignore` で既に対応済みのはず (T-M0.1)。本チケットでは触らない。
- **YAML 雛形の鮮度**: `docs/implementation-plan.md` §5 の雛形を反映するが、M2.5/M3.2 で完成形に書き換える前提。本チケット時点では「ドラフト」コメントを冒頭に入れる。
- **`configs/*.yaml` の secrets 漏洩リスク**: 雛形時点で `wandb_api_key:` や `hf_token:` 等を **絶対に書かない** 方針を明記。secrets は `.env` / 環境変数経由で読み込む設計とし、本チケットでは `.env.example` を placeholder のみで commit。
- **`tests/test_*.py` 空ファイル問題**: pytest が 0 tests で **exit code 5** を返し CI が壊れる。本チケットでは各 test_*.py に `def test_placeholder(): pytest.skip("MX.Y で実装")` を入れて exit 0 を担保する。
- **mel 抽出パラメータの単一情報源 (SoT) 確保**: `src/wavenext2/data/mel.py` 内の関数で n_fft / hop_length / win_length / n_mels をハードコードすると、`configs/*.yaml` と二重定義になり、GAN(hop=300) と Diff(hop=256) を取り違える事故が起きやすい。**M0.2 では `mel.py` の docstring に「これらのパラメータは config 経由のみで受け取る関数 API として T-M1.3 で実装する」旨を明記**。本チケットではスタブのみで実装はしない。
- **`scripts/scaffold.py` 冪等性**: パスのハードコードで再実行すると上書き or エラー発生のリスク。`if __name__ == "__main__":` guard 必須、各ファイル作成前に `Path.exists()` チェックで既存をスキップ、ディレクトリは `Path.mkdir(parents=True, exist_ok=True)` を使う。

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目: なし (全て確定済み)
- 旧来の判断ポイント (M0 レビューで解決済み):
  - **`configs/*.yaml` を空で commit するか雛形コミットするか** → **雛形 commit** を採用 (`docs/implementation-plan.md` §5 の内容を反映)。後続が「設定 key 名」を変更したくなった時に追跡しやすい。
  - **`src/` flat vs `src/wavenext2/` パッケージ化** → **`src/wavenext2/` パッケージ化採用** (§8.1 参照、3 エージェント合議)。

### 6.3 他チケットとの整合性
- T-M0.1 (Python 環境) から本チケットへ渡される: CI 雛形のフォーマット (`setup-uv@v3`), pre-commit のメジャー hook 構成, `.gitattributes` の方針, `.env.example` の placeholder の置き場所
- T-M0.3 (`scripts/prepare_libritts.py` 配置先 + `data/` 配下) と整合: OK、scaffold で stub と `.gitkeep` を置く
- T-M1.1〜T-M1.6 (`src/wavenext2/models/`, `src/wavenext2/data/` のファイル名) と整合: OK、ファイル名は milestones.md §M0.2 から確定。**import 形式は `from wavenext2.models.convnext import ConvNeXtBlock`**
- T-M1.3 (mel 抽出) へ: mel 抽出パラメータは config 経由のみで関数化する設計を docstring で予告 (§6.1)
- T-M2.1〜T-M2.3, T-M3.x, T-M4.x も同様 (`wavenext2.X` 名前空間)
- `src/wavenext2/inference/post_filter.py` の置き場が `docs/milestones.md` §M3.4 と整合 (旧 `src/inference/post_filter.py` から rename)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] ディレクトリ構造が `docs/implementation-plan.md` §3 および `docs/milestones.md` §M0.2 と整合 (`src/wavenext2/` パッケージ構造であること)
- [ ] 各 `__init__.py` が存在し、`__all__` 予約コメントが入っている (完全空ではないが docstring + コメントのみ)
- [ ] 各 .py スタブが docstring + TODO コメントを持つ (空ファイルではない、import エラーを避けるため)
- [ ] `tests/__init__.py` が **存在しない** (pytest の autodetection を妨げないため)
- [ ] `tests/conftest.py` が存在し空 (今後の fixture 追加用)
- [ ] 各 `tests/test_*.py` に `def test_placeholder(): pytest.skip(...)` が含まれ、`uv run pytest` が exit 0 で通る
- [ ] `configs/*.yaml` が `docs/implementation-plan.md` §5 の内容と一致 (key の typo / 値の typo がない、secrets 値を含まない)
- [ ] `checkpoints/.gitkeep`, `logs/.gitkeep`, `data/*/.gitkeep` が空ファイルで存在し git に commit される
- [ ] `Dockerfile`, `.dockerignore`, `.devcontainer/devcontainer.json` が TODO コメント付きで存在
- [ ] `.github/workflows/{test,lint}.yml` が TODO コメント付きで存在
- [ ] `.pre-commit-config.yaml` がメジャー hook 構成で存在
- [ ] `.gitattributes` に `* text=auto eol=lf` が含まれる
- [ ] `.env.example` に secrets 値が **含まれない** (placeholder のみ)
- [ ] `pyproject.toml` に `[tool.hatch.build.targets.wheel] packages = ["src/wavenext2"]` (または等価) が含まれる
- [ ] Acceptance criteria 全項目クリア
- [ ] `uv run python -c "import wavenext2.<sub>; ..."` がエラーなく通る (全 7 サブパッケージ、`wavenext2.X` 名前空間)
- [ ] CLAUDE.md / 既存ドキュメントのスタイル準拠 (日本語 + 技術用語英語)
- [ ] 参考実装をコピーしていない (一から書いたか)
- [ ] `scripts/scaffold.py` が冪等で再実行可能 (上書きしない、`Path.exists()` チェック有り)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **`src/wavenext2/` パッケージ化を採用** (M0 レビュー / 3 エージェント合議で確定)
  - ディレクトリ: `src/wavenext2/{data, models, losses, train, inference, eval, utils}/`
  - `pyproject.toml`:
    ```toml
    [tool.hatch.build.targets.wheel]
    packages = ["src/wavenext2"]
    ```
    または uv 標準の workspace 機能で等価設定
  - import 形式: `from wavenext2.models.convnext import ConvNeXtBlock`
  - 採用理由 (3 エージェント合議):
    - (a) `import src.models` は jupyter で別プロジェクトと衝突する一般名
    - (b) `sys.path` 操作に依存して `uv run python -c "import src..."` 以外の経路で壊れやすい (pytest discovery がハック前提になる)
    - (c) M6 でクラスタに `uv pip install -e .` 一発展開できる
    - (d) リネームコストは M1 着手前の今が最安

#### Deprecated (旧案、却下根拠)
1. **`src/` flat 構造 (旧採用案)** — `src/{models,data,...}` を root パッケージなしに置く
   - 旧メリット: import path が短い (`from src.convnext import ...`)
   - 却下理由: 上記 (a)(b)(c) により M6 まで保留すると後戻りコストが大きい。M0 レビューで `src/wavenext2/` へ昇格。

2. **flat な構造 (sub-package を分けない)**
   - 例: `src/wavenext2/{convnext.py, stft.py, dataset.py, train_gan.py, ...}` を一階層に置く
   - 却下: ファイル数が 20+ になり機能境界が曖昧

3. **`src/` を廃止して `wavenext2/` 直下に置く**
   - 例: `wavenext2/models/`, `wavenext2/data/` ...
   - メリット: Python パッケージとして自然 (root が import 名と一致)
   - 却下: `src/` layout は uv / hatch / pytest の test discovery と相性が良く、ライブラリと test の混在事故を防げる。ベストプラクティス準拠

4. **`models/` を `gan_models/` + `diff_models/` + `common/` に分割**
   - 例: `src/wavenext2/common/{convnext.py, stft.py, generator.py, sub_model.py}`, `wavenext2/gan/{gan_wavenext2.py, discriminator.py}`, `wavenext2/diff/{diff_wavenext2.py, noise_embedding.py}`
   - 却下: convnext.py が両方で使われる (conditioning の有無で分岐) ため共有部分の所属が曖昧。本実装では **`models/` フラット** + ファイル内で GAN/Diff の対応をコメント明示する方針

#### 追加検討した設計案
- **manifest YAML + idempotent scaffold**: 期待ファイル一覧を `scaffold.manifest.yaml` で持ち、`scripts/scaffold.py` が差分作成型に。CI で drift 検知できる
  - 採否: **将来検討**。本チケットでは `scripts/scaffold.py` のハードコード版で十分 (約 28 ファイル)。manifest 化はファイル数が 50+ になった時 (M2.5/M3 完了後) に検討
- **Docker / devcontainer / CI 雛形を本チケットで commit する判断**: M5/M6 で初めて作るより、scaffold 段階で stub を置く方が後続コストが低い (3 エージェント共通) → **採用**

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| `src/wavenext2/` パッケージ化 | (経験上発生しない) | 採用後の re-evaluation 不要 |
| Docker placeholder → 完成 | M5 直前 | Dockerfile を完成形に書き換え |
| CI placeholder → 完成 | M0.2 完了直後 (`uv sync` の動作確認まで) | test.yml の `uv run pytest` step を有効化 |
| pre-commit 設定 | M1 完了時 | 各 hook の対象 path 調整 |
| `scaffold.py` → manifest 化 | M3 完了時 (ファイル数 50+ の場合) | `scaffold.manifest.yaml` に分離 |
| `models/` のサブ分割 (gan/diff/common) | M3 完了時 | 共有モジュールが明確に分離可能か再評価 |
| `__init__.py` の re-export 追加 | 各サブパッケージ実装完了時 | `__all__` に public symbol 追加 |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。M0.2 を単独チケットにすることで、M1 以降のチケットが「ファイル作成」を含まずに済み、論理的境界が綺麗。
- **別マイルストーンに移すべき部分はないか**: なし。scaffold は環境準備フェーズに含めるのが自然。Docker / CI / pre-commit の skeleton も同時に置くことで M5/M6 の作業コストを減らせる。
- **インターフェース定義の見直し余地**:
  - **`__init__.py` で `__all__` を予約**: public API 境界を最初から明示することで、後続が無自覚に internal helper を import する事故を防ぐ。本チケットで `__all__ = []` のコメントを入れる
  - **`configs/*.yaml` 雛形 commit の意義**: GAN(hop=300) vs Diff(hop=256) のような **モデル別ハイパーパラメータ衝突の検出器** としての価値。設定の単一情報源化 (SoT) を最初から実現
  - **YAML 雛形を本チケットで commit する方針**: M2.5/M3.2 で書き換えるとしても、初期 commit があった方が key 構造の認識が早い
- **再評価トリガー条件**: §8.1 末尾の表を参照

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- 生成: `src/wavenext2/` 7 サブパッケージ + 23 モジュールスタブ + utils/seed.py、scripts 3 stub + scaffold.py、tests conftest + 6 placeholder、configs 2、Docker/CI/pre-commit/.gitattributes/.env.example、.gitkeep 7。
- 検証: `wavenext2.{models,data,losses,train,inference,eval,utils}` 全 import OK / pytest collect 6 (exit 0) / 全 skip / `ruff check` + `ruff format --check` 通過 / scaffold 再実行で 51 件全 skip (冪等) / `git check-attr eol README.md` = lf。
- `pyproject.toml` を installable へ切替: `[build-system] hatchling` + `[tool.hatch.build.targets.wheel] packages=["src/wavenext2"]`、`[tool.uv] package=false` を削除。`uv run`/`uv sync` が自動で editable build する。

想定外と対処:
1. **scaffold.py のパスバグ**: `".data".replace(".", "/")` が `"/data"` (先頭スラッシュ) になり `pkg_root / "/data"` が `C:\data` に化けた。→ `sub.lstrip(".")` で先頭ドット除去。教訓: **dotted module 名から path を作るときは先頭の区切りを必ず strip**。
2. **Windows console (cp932) で日本語 print が文字化け**: 生成ファイルは `encoding="utf-8"` で正常だが stdout が化けた。→ **scaffold の stdout メッセージは ASCII に統一** (ファイル内容の日本語は維持)。
3. **`.env.example` が Write ツールで作成不可** (`.env*` がツール権限でブロック)。→ **Bash heredoc で作成**。教訓: secrets 系ファイル名はツール側ガードに当たるので shell 経由が要る場合がある。
4. **ruff format**: SCRIPT_STUB テンプレの `if __name__` 前の空行 2→1。→ テンプレ修正して再生成も clean に。
5. **CRLF 警告の解消**: T-M0.1 commit 時に出ていた LF→CRLF 警告は本チケットの `.gitattributes` (`* text=auto eol=lf`) で抑止される。

教訓 (次の似たタスク):
- **`uv run python <script>` は実行前に project を editable build する**ので、build-system 導入後はスタブ生成スクリプトも自動でパッケージ install を伴う (副作用として既知)。
- **冪等 scaffold は `Path.exists()` skip が必須**。後続チケット (T-M0.3 等) がスタブを実装で上書きしても scaffold 再実行で壊さない。

## 9. 後続タスクへの連絡事項

### 9.1 T-M0.1 から受け取った情報 (M0 レビュー連絡事項経由)
- CI 雛形フォーマット: `setup-uv@v3` + `actions/cache@v4` を `.github/workflows/test.yml` で採用
- pre-commit メジャー hook 構成: ruff format / ruff check / pre-commit-hooks (trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict)
- `.gitattributes` の方針: `* text=auto eol=lf` を default、Windows-only ファイルは個別指定
- `.env.example` の placeholder: `WANDB_API_KEY=`, `HF_TOKEN=`, `OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=0` 等
- 乱数 seed 集約: `src/wavenext2/utils/seed.py` を予告 stub として置く (PyTorch / numpy / random / CUDA seed 一括設定の docstring)

### 9.2 後続チケットに渡す情報
- **import 形式 (全チケット共通)**: `from wavenext2.<sub>.<mod> import <Symbol>` (旧 `from src.<sub>.<mod>` から変更)。例: `from wavenext2.models.convnext import ConvNeXtBlock`
- **T-M0.3** (`prepare_libritts.py`):
  - `scripts/prepare_libritts.py` がスタブで置いてあるので、内容を埋めるだけで良い
  - `data/filelists/`, `data/raw/`, `data/cache/`, `data/processed/` の場所が確定済み (`.gitkeep` 配置済み)
  - LibriTTS-R wav は `data/raw/LibriTTS_R/...` に展開する想定 (`.gitignore` で wav 自体は除外)
- **T-M1.1〜T-M1.6**:
  - `src/wavenext2/models/{convnext,stft,generator,sub_model,noise_embedding}.py` および `src/wavenext2/data/mel.py` の配置確定
  - テストは `tests/test_*.py` の `pytest.skip` placeholder を実テストに置き換える
- **T-M1.3** (mel) へ:
  - `src/wavenext2/data/mel.py` の docstring に「mel 抽出パラメータ (n_fft, hop_length, win_length, n_mels) は config 経由のみで受け取る関数 API として実装」と予告済み
  - GAN(hop=300) と Diff(hop=256) の取り違え防止のため、ハードコード禁止
- **T-M2.1〜T-M2.6**: `src/wavenext2/data/dataset.py`, `src/wavenext2/models/discriminator.py`, `src/wavenext2/losses/*.py`, `src/wavenext2/models/gan_wavenext2.py`, `src/wavenext2/train/train_gan.py` が配置済み
- **T-M3.1〜T-M3.5**: `src/wavenext2/models/diff_wavenext2.py`, `src/wavenext2/train/train_diff.py`, `src/wavenext2/inference/{infer_diff,post_filter}.py`, `scripts/fit_post_filter.py` が配置済み
- **T-M4.1〜T-M4.3**: `src/wavenext2/eval/{compute_metrics,run_utmos,run_nisqa,measure_rtf}.py` が配置済み
- **設定値**: `configs/gan_wavenext2.yaml` と `configs/diff_wavenext2.yaml` の雛形 key は `docs/implementation-plan.md` §5 と一致。後続が変更する場合は本ファイルの key を up-to-date に保つこと。secrets 値は **絶対に書かない** (`.env` 経由で読み込む設計)
- **T-M5.x (Docker)** へ: `Dockerfile` / `.dockerignore` / `.devcontainer/devcontainer.json` の placeholder が配置済み。M5 で uv + CUDA 12.x base image を完成させる
- **CI**: `.github/workflows/{test,lint}.yml` の skeleton が配置済み。M0.2 完了直後に test step を有効化 (`uv sync` + `uv run pytest`)

### 9.3 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M0.2 の Acceptance チェックボックスをチェック (パッケージ化方針 / 追加 stub 反映)
  - [ ] `docs/tickets/index.md` の T-M0.2 ステータスを `📝 pending` → `✅ completed`
  - [ ] `docs/implementation-plan.md` §3 のディレクトリ構成を `src/wavenext2/` パッケージ化に合わせて更新 (差分がある場合)

### 9.4 Open question として残ったもの
- 解決できなかった疑問: なし (機械的タスクのため)
- `docs/open-questions.md` への追記要否: 不要
- 将来検討事項:
  - `scaffold.manifest.yaml` への移行 (ファイル数 50+ になった時、M3 完了後検討)
  - `models/` サブ分割 (gan/diff/common) の再評価 (M3 完了後)
  - 上記は §8.1 末尾の再評価トリガー表に集約済み
