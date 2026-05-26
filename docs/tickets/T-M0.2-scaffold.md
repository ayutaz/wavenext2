---
id: T-M0.2
title: src/ ディレクトリ scaffold 作成
milestone: M0
phase: M0
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.1]
blocks: [T-M1.1, T-M1.2, T-M1.3, T-M1.4, T-M1.5, T-M1.6, T-M2.1, T-M2.2, T-M2.3, T-M4.1, T-M4.2]
related_docs:
  - docs/milestones.md#m02-ディレクトリ-scaffold
  - docs/implementation-plan.md
  - docs/architecture.md
---

# T-M0.2: src/ ディレクトリ scaffold 作成

> **マイルストーン**: [M0](../milestones.md#m0-環境整備とデータ準備-作業量-small) / **サブタスク**: [M0.2](../milestones.md#m02-ディレクトリ-scaffold)
> **依存**: [T-M0.1](T-M0.1-python-env.md) / **後続**: T-M1.1〜T-M1.6, T-M2.1〜T-M2.3, T-M4.1, T-M4.2 ほか

## 1. タスク目的とゴール

### 目的
後続マイルストーン (M1〜M4) が各モジュールを実装する **置き場所** を確定させる。`docs/implementation-plan.md` §3 と `docs/milestones.md` §M0.2 で示されたディレクトリツリーを忠実に再現し、空の `__init__.py` と TODO スタブのみで構成された .py / .yaml ファイル群を一括 commit する。

### ゴール
- [ ] `src/{data,models,losses,train,inference,eval,utils}/` の 7 サブパッケージが作成され、それぞれに `__init__.py` (空) が存在
- [ ] 各 .py モジュール (e.g., `convnext.py`, `stft.py`, `generator.py` ...) が TODO スタブ付きで存在 (本実装は後続チケットで埋める)
- [ ] `configs/{gan_wavenext2.yaml, diff_wavenext2.yaml}` の雛形が `docs/implementation-plan.md` §5 の内容を反映して commit 済み
- [ ] `tests/{test_convnext.py, test_stft_module.py, ...}` が pytest 認識可能な空ファイルで存在
- [ ] `scripts/{prepare_libritts.py, extract_mel.py, fit_post_filter.py}` が TODO スタブ付きで存在
- [ ] `checkpoints/.gitkeep`, `logs/.gitkeep` が存在し、空ディレクトリを git に commit できる
- [ ] `uv run python -c "import src.models, src.data, src.losses, src.train, src.inference, src.eval, src.utils"` がエラーなく成功
- [ ] `docs/tickets/index.md` の本チケットステータスを更新済み

## 2. 実装内容の詳細

### 2.1 対象ファイル (新規作成)

#### src/ パッケージ
- `src/__init__.py` (空)
- `src/data/__init__.py` (空)
- `src/data/dataset.py` (TODO stub)
- `src/data/mel.py` (TODO stub)
- `src/models/__init__.py` (空)
- `src/models/convnext.py` (TODO stub)
- `src/models/stft.py` (TODO stub)
- `src/models/generator.py` (TODO stub)
- `src/models/sub_model.py` (TODO stub)
- `src/models/noise_embedding.py` (TODO stub)
- `src/models/discriminator.py` (TODO stub)
- `src/models/gan_wavenext2.py` (TODO stub)
- `src/models/diff_wavenext2.py` (TODO stub)
- `src/losses/__init__.py` (空)
- `src/losses/adversarial.py` (TODO stub)
- `src/losses/feature_matching.py` (TODO stub)
- `src/losses/stft_loss.py` (TODO stub)
- `src/train/__init__.py` (空)
- `src/train/train_gan.py` (TODO stub)
- `src/train/train_diff.py` (TODO stub)
- `src/inference/__init__.py` (空)
- `src/inference/infer_gan.py` (TODO stub)
- `src/inference/infer_diff.py` (TODO stub)
- `src/inference/post_filter.py` (TODO stub)
- `src/eval/__init__.py` (空)
- `src/eval/compute_metrics.py` (TODO stub)
- `src/eval/run_utmos.py` (TODO stub)
- `src/eval/run_nisqa.py` (TODO stub)
- `src/eval/measure_rtf.py` (TODO stub)
- `src/utils/__init__.py` (空)
- `src/utils/config.py` (TODO stub)
- `src/utils/logging.py` (TODO stub)
- `src/utils/scheduler.py` (TODO stub)

#### configs/
- `configs/gan_wavenext2.yaml` (雛形を `docs/implementation-plan.md` §5 からコピー)
- `configs/diff_wavenext2.yaml` (同上)

#### tests/
- `tests/test_convnext.py` (空、後続 T-M1.1 で実装)
- `tests/test_stft_module.py` (空、後続 T-M1.2 で実装)
- `tests/test_generator.py` (空、後続 T-M1.4 で実装)
- `tests/test_sub_model.py` (空、後続 T-M1.6 で実装)
- `tests/test_dataset.py` (空、後続 T-M2.1 で実装)
- `tests/test_post_filter.py` (空、後続 T-M3.4 で実装)
- `tests/__init__.py` は **作成しない** (pytest は test_ prefix で自動収集、namespace package 化を避ける)

#### scripts/
- `scripts/prepare_libritts.py` (TODO stub、後続 T-M0.3 で実装)
- `scripts/extract_mel.py` (TODO stub、後続 T-M2.1 で実装)
- `scripts/fit_post_filter.py` (TODO stub、後続 T-M3.4 で実装)

#### 空ディレクトリ commit
- `checkpoints/.gitkeep` (空ファイル)
- `logs/.gitkeep` (空ファイル)

### 2.2 主要構造 (各 .py スタブの共通テンプレート)

```python
"""<モジュール名>: <一行説明>

TODO: T-MX.Y で実装。
詳細は docs/architecture.md / docs/milestones.md を参照。
"""

# このファイルは scaffold 段階のスタブです。
# 実装は後続チケットで行います。
```

`__init__.py` は **完全に空** (再エクスポートを含めない) で commit する。後続チケットで必要に応じて追加。

### 2.3 ディレクトリツリー (完成形)

```
wavenext2/
├── checkpoints/
│   └── .gitkeep
├── configs/
│   ├── gan_wavenext2.yaml
│   └── diff_wavenext2.yaml
├── docs/                       # 既存
├── logs/
│   └── .gitkeep
├── scripts/
│   ├── prepare_libritts.py
│   ├── extract_mel.py
│   └── fit_post_filter.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── mel.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── convnext.py
│   │   ├── stft.py
│   │   ├── generator.py
│   │   ├── sub_model.py
│   │   ├── noise_embedding.py
│   │   ├── discriminator.py
│   │   ├── gan_wavenext2.py
│   │   └── diff_wavenext2.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── adversarial.py
│   │   ├── feature_matching.py
│   │   └── stft_loss.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_gan.py
│   │   └── train_diff.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── infer_gan.py
│   │   ├── infer_diff.py
│   │   └── post_filter.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── compute_metrics.py
│   │   ├── run_utmos.py
│   │   ├── run_nisqa.py
│   │   └── measure_rtf.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── scheduler.py
└── tests/
    ├── test_convnext.py
    ├── test_stft_module.py
    ├── test_generator.py
    ├── test_sub_model.py
    ├── test_dataset.py
    └── test_post_filter.py
```

### 2.4 作成方法
- **推奨**: Python スクリプト (`scripts/scaffold.py` を一時的に作成) で一括生成。冪等性を確保 (既存ファイルは上書きしない)。
- 代替: PowerShell `New-Item` または Bash `touch + mkdir -p` で逐次作成。Windows 環境でもパスは forward slash で統一する (Python `pathlib.Path` を使う場合は OS 非依存)。
- **注意**: 作業終了後、`scripts/scaffold.py` は commit しても良いし削除しても良い (再実行できるよう commit を推奨)。

### 2.5 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| パッケージ root | `src/` | docs/implementation-plan.md §3 |
| サブパッケージ数 | 7 (data, models, losses, train, inference, eval, utils) | docs/milestones.md §M0.2 |
| .py スタブ総数 | 約 26 ファイル | docs/milestones.md §M0.2 |
| YAML 雛形数 | 2 (GAN / Diff) | docs/implementation-plan.md §5 |
| テストファイル数 | 6 | docs/milestones.md §M0.2 |
| スクリプト数 | 3 | docs/milestones.md §M0.2 |

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | scaffold 一括生成 (Python or shell)、各ファイル作成 | general-purpose |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M0.1 完了後でないと `uv run` 検証ができない)
- 並列実行する場合の最大並列数: 1 (機械的タスクなので分割の意味がない)

## 4. 提供範囲 (Scope)

### In Scope
- ディレクトリツリーの作成
- 全 `__init__.py` を空ファイルで配置
- 各 .py モジュールを TODO スタブで配置 (本実装は後続)
- `configs/*.yaml` の雛形コピー
- 空ディレクトリ commit 用 `.gitkeep` 配置
- `docs/tickets/index.md` のステータス更新

### Out of Scope
- 各 .py モジュールの本実装 (M1〜M4 各チケット)
- テストロジックの記述 (M1〜M2 各チケット)
- LibriTTS-R ダウンロードロジック (T-M0.3)
- post-filter 実装 (T-M3.4)
- `__init__.py` の re-export (必要なら後続チケットで追加)

### Deliverable
- ファイル: 上記 §2.1 の全ファイル
- パッケージ構造: `src.models`, `src.data`, `src.losses`, `src.train`, `src.inference`, `src.eval`, `src.utils` の 7 パッケージが import 可能
- ドキュメント差分: `docs/tickets/index.md` の T-M0.2 ステータス更新のみ

## 5. テスト項目

### 5.1 Unit テスト
- 不要 (構造作成のみで実装ロジックがない)

### 5.2 e2e / 結合テスト
- [ ] `uv run python -c "import src.models, src.data, src.losses, src.train, src.inference, src.eval, src.utils"` がエラーなく成功
- [ ] PowerShell: `(Get-ChildItem -Path src,tests,scripts,configs -Recurse -File -Include *.py,*.yaml | Measure-Object).Count` で期待ファイル数 (約 36) が一致
- [ ] Bash 環境では `find src tests scripts configs -type f \( -name "*.py" -o -name "*.yaml" \) | wc -l` で同等の確認
- [ ] `uv run pytest tests/ --collect-only` で 6 個のテストファイルが collect される (空でも 0 tests で OK)
- [ ] `git status` で全新規ファイルが untracked として表示される

### 5.3 Acceptance criteria (`docs/milestones.md` §M0.2 より転記)
- [ ] 上記ツリーが作成され、各 `__init__.py` が空ファイルで存在

### 5.4 追加 acceptance (本チケット独自)
- [ ] `configs/gan_wavenext2.yaml` と `configs/diff_wavenext2.yaml` の内容が `docs/implementation-plan.md` §5 と整合 (key の typo がない)
- [ ] `.gitkeep` 配置により `git add checkpoints/ logs/` で空ディレクトリが commit 可能
- [ ] Windows 環境でも import path にバックスラッシュ起因のエラーがない

## 6. 懸念事項

### 6.1 技術的リスク
- **`__init__.py` の中身**: 現時点では空とするが、後続チケットで `from .convnext import ConvNeXtBlock` のような re-export を追加するかは未定。**初稿は空、必要に応じて後続で追加** という方針を明文化しておく。
- **Windows パス区切り**: PowerShell / Bash 混在環境で動作する必要がある。Python `pathlib.Path` を使うか、forward slash で統一する。
- **`__pycache__/` 等のキャッシュ**: `.gitignore` で既に対応済みのはず。本チケットでは触らない。
- **YAML 雛形の鮮度**: `docs/implementation-plan.md` §5 の雛形を反映するが、M2.5/M3.2 で完成形に書き換える前提。本チケット時点では「ドラフト」コメントを冒頭に入れる。

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目: なし (全て確定済み)
- 唯一の判断ポイント: **`configs/*.yaml` を空で commit するか雛形コミットするか**
  - 推奨: **雛形 commit** (`docs/implementation-plan.md` §5 の内容を反映)。後続が「設定 key 名」を変更したくなった時に追跡しやすい。
  - 別案: 空 commit して M2.5/M3.2 で完成形を書く。
  - 本チケットでは **雛形 commit** を採用する。

### 6.3 他チケットとの整合性
- T-M0.3 (`scripts/prepare_libritts.py` 配置先) と整合: OK、scaffold で stub を置く
- T-M1.1〜T-M1.6 (`src/models/`, `src/data/` のファイル名) と整合: OK、ファイル名は milestones.md §M0.2 から確定
- T-M2.1〜T-M2.3, T-M3.x, T-M4.x も同様
- `src/inference/post_filter.py` の置き場が `docs/milestones.md` §M3.4 では `src/inference/post_filter.py` と書かれており、本チケットでもそれに従う

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] ディレクトリ構造が `docs/implementation-plan.md` §3 および `docs/milestones.md` §M0.2 と完全一致
- [ ] 各 `__init__.py` が存在し、内容が空 (0 byte)
- [ ] 各 .py スタブが docstring + TODO コメントを持つ (空ファイルではない、import エラーを避けるため)
- [ ] `tests/__init__.py` が **存在しない** (pytest の autodetection を妨げないため)
- [ ] `configs/*.yaml` が `docs/implementation-plan.md` §5 の内容と一致 (key の typo / 値の typo がない)
- [ ] `checkpoints/.gitkeep`, `logs/.gitkeep` が空ファイルで存在し git に commit される
- [ ] Acceptance criteria 全項目クリア
- [ ] `uv run python -c "import src.<sub>; ..."` がエラーなく通る (全 7 サブパッケージ)
- [ ] CLAUDE.md / 既存ドキュメントのスタイル準拠 (日本語 + 技術用語英語)
- [ ] 参考実装をコピーしていない (一から書いたか)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用可能だった代替設計
1. **flat な構造 (sub-package を分けない)**
   - 例: `src/{convnext.py, stft.py, dataset.py, train_gan.py, ...}` を一階層に置く
   - メリット: import path が短い (`from src.convnext import ...`)
   - デメリット: ファイル数が 20+ になり機能境界が曖昧、却下

2. **`src/wavenext2/` パッケージ化**
   - `pip install -e .` (uv では `uv pip install -e .`) を意識し、`src/wavenext2/` を root パッケージとする
   - メリット: 外部から `import wavenext2.models` のように使える、PyPI 公開も視野に
   - デメリット: 本リポジトリは内部実装メインで配布予定なし、`src/` でも十分
   - **保留判断**: 本格訓練完了後 (M6 後) に PyPI 公開検討する場合に変更を検討

3. **`src/` を廃止して `wavenext2/` 直下に置く**
   - 例: `wavenext2/models/`, `wavenext2/data/` ...
   - メリット: Python パッケージとして自然 (root が import 名と一致)
   - デメリット: ドキュメント (`docs/implementation-plan.md` §3) で既に `src/` 採用済み、変更は混乱を招く
   - 却下

4. **`models/` を `gan_models/` + `diff_models/` + `common/` に分割**
   - 例: `src/common/{convnext.py, stft.py, generator.py, sub_model.py}`, `src/gan/{gan_wavenext2.py, discriminator.py}`, `src/diff/{diff_wavenext2.py, noise_embedding.py}`
   - メリット: GAN/Diff の境界が明確
   - デメリット: convnext.py が両方で使われる (conditioning の有無で分岐) ため共有部分の所属が曖昧、却下
   - 本実装では **`models/` フラット** + ファイル内で GAN/Diff の対応をコメント明示する方針

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。M0.2 を単独チケットにすることで、M1 以降のチケットが「ファイル作成」を含まずに済み、論理的境界が綺麗。
- **別マイルストーンに移すべき部分はないか**: なし。scaffold は環境準備フェーズに含めるのが自然。
- **インターフェース定義の見直し余地**:
  - `__init__.py` を空にする方針: re-export が必要になった時に決める (YAGNI)
  - YAML 雛形を本チケットで commit する方針: M2.5/M3.2 で書き換えるとしても、初期 commit があった方が key 構造の認識が早い

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **T-M0.3** (`prepare_libritts.py`): `scripts/prepare_libritts.py` がスタブで置いてあるので、内容を埋めるだけで良い
- **T-M1.1〜T-M1.6**: `src/models/{convnext,stft,generator,sub_model,noise_embedding}.py` および `src/data/mel.py` の配置確定。テストは `tests/test_*.py` 空ファイル内に追記する
- **T-M2.1〜T-M2.6**: `src/data/dataset.py`, `src/models/discriminator.py`, `src/losses/*.py`, `src/models/gan_wavenext2.py`, `src/train/train_gan.py` が配置済み
- **T-M3.1〜T-M3.5**: `src/models/diff_wavenext2.py`, `src/train/train_diff.py`, `src/inference/{infer_diff,post_filter}.py`, `scripts/fit_post_filter.py` が配置済み
- **T-M4.1〜T-M4.3**: `src/eval/{compute_metrics,run_utmos,run_nisqa,measure_rtf}.py` が配置済み
- **設定値**: `configs/gan_wavenext2.yaml` と `configs/diff_wavenext2.yaml` の雛形 key は `docs/implementation-plan.md` §5 と一致。後続が変更する場合は本ファイルの key を up-to-date に保つこと。

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M0.2 の Acceptance チェックボックスをチェック
  - [ ] `docs/tickets/index.md` の T-M0.2 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/implementation-plan.md` §3 のディレクトリ構成と差分があれば反映

### 9.3 Open question として残ったもの
- 解決できなかった疑問: なし (機械的タスクのため)
- `docs/open-questions.md` への追記要否: 不要
- 将来検討事項: `src/wavenext2/` パッケージ化 (PyPI 公開検討時) は §8.1 に記載済み
