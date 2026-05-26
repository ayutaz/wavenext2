---
id: T-M4.2
title: UTMOS / NISQA 連携 (自動 MOS 推定)
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
  - docs/milestones.md#m42-utmos--nisqa-連携
  - docs/training.md
---

# T-M4.2: UTMOS / NISQA 連携 (自動 MOS 推定)

> **マイルストーン**: [M4](../milestones.md#m4-評価インフラ-作業量-medium3-サブタスク) / **サブタスク**: [M4.2](../milestones.md#m42-utmos--nisqa-連携)
> **依存**: [T-M0.1](T-M0.1-python-env.md) / **後続**: [T-M5.1](T-M5.1-gan-1epoch.md), [T-M5.2](T-M5.2-diff-1epoch.md)

## 1. タスク目的とゴール

### 目的
論文 Section 4.2 / `docs/training.md` §5.2 で採用された 2 つの **自動 MOS 推定システム** (UTMOS, NISQA) を、生成音声の品質評価パイプラインに統合する。これにより主観評価 (M7、人手必須) を待たずに、GAN / Diff いずれの出力品質も継続的に数値化でき、M5 smoke (1 epoch 後) の sanity check と M6 本格訓練の論文 Table 1〜3 対比に使える。

最大の技術課題は **Python バージョン非互換**: UTMOS22 は古い fairseq / PyTorch 想定、NISQA も特定バージョン依存のため、本体プロジェクトの Python 3.13 + torch>=2.10 (T-M0.1 確定) と同一 venv に同居させると依存衝突が高確率で発生する。本チケットは **subprocess 隔離 + 専用 venv** を第一級の設計方針として据え、本体 import path を汚さずに MOS 推定を呼べる薄い wrapper (`run_utmos.py` / `run_nisqa.py`) を提供する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M4.2 Acceptance を内包):
- [ ] `scripts/setup_eval_models.py` が UTMOS22 (git submodule) と NISQA (submodule or pip) を clone し、checkpoint を自動 download → `eval_models/` 配下に配置
- [ ] `src/wavenext2/eval/run_utmos.py` に `score_utmos(wav_paths, *, device, backend) -> dict[str, float]` が実装され、`from wavenext2.eval.run_utmos import score_utmos` で import 可能 (本体 import path は **UTMOS の重い依存を引き込まない**)
- [ ] `src/wavenext2/eval/run_nisqa.py` に `score_nisqa(wav_paths, *, device) -> dict[str, float]` が実装され、同様に import 可能
- [ ] **subprocess 隔離経路**が動作: 本体 (Python 3.13) から `uv run --project eval_models/utmos_env ...` (or `eval_models/.venv-utmos/bin/python`) を `subprocess.run` で起動し、JSON で score を受け取る
- [ ] GT 音声 (LibriTTS-R test-clean サブセット) で **UTMOS ≈ 4.0 ± 0.2**、**NISQA ≈ 4.5 ± 0.3** (高品質前提) を再現
- [ ] 既知の劣化音声 (white noise 付加 SNR 10 dB) で MOS が **有意に低下** (GT より UTMOS / NISQA とも低い)
- [ ] test-clean サブセット 100 utt をバッチ処理して mean / std を返せる
- [ ] checkpoint が見つからない / 未 setup の場合に **明示的な error** (`EvalModelNotFoundError`) を raise (silent NaN を出さない)
- [ ] `tests/test_eval_mos.py` の全テスト pass (CI では実モデル不在のため大半 `@pytest.mark.skipif` / `@pytest.mark.slow`)
- [ ] `docs/milestones.md` §M4.2 Acceptance クリア、`docs/tickets/index.md` の T-M4.2 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル

- 新規:
  - `scripts/setup_eval_models.py` (UTMOS / NISQA の clone + checkpoint download + 専用 venv 構築 CLI。`uv` は git submodule を直接管理しないため、本スクリプトが external dep のセットアップを一手に引き受ける)
  - `src/wavenext2/eval/run_utmos.py` (`score_utmos` + subprocess driver + in-process backend)
  - `src/wavenext2/eval/run_nisqa.py` (`score_nisqa` + subprocess driver)
  - `src/wavenext2/eval/_mos_common.py` (共通ユーティリティ: wav リスト I/O, JSON プロトコル, `EvalModelNotFoundError`, batch chunking)
  - `tests/test_eval_mos.py` (Unit + 隔離経路の e2e + 劣化音声テスト)
  - `eval_models/.gitkeep` (空 dir 維持。clone 物 / checkpoint / 専用 venv はすべて `.gitignore` 除外)
- 編集:
  - `src/wavenext2/eval/__init__.py` (`score_utmos`, `score_nisqa`, `EvalModelNotFoundError` を `__all__` に追加。**ただし top-level import で UTMOS/NISQA 本体を import しない** — lazy import に徹する)
  - `.gitignore` (`eval_models/UTMOS22/`, `eval_models/NISQA/`, `eval_models/*.ckpt`, `eval_models/*.tar`, `eval_models/.venv-*/`, `eval_models/*_env/` を除外。`.gitkeep` のみ keep)
  - `.gitmodules` (UTMOS22 を submodule 登録する場合。NISQA は pip 経路なら登録不要)
  - `docs/milestones.md` §M4.2 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M4.2 ステータス更新

### 2.2 主要構造

#### `scripts/setup_eval_models.py`

```python
"""setup_eval_models.py — UTMOS22 / NISQA の clone + checkpoint download + 専用 venv 構築.

uv は git submodule を直接管理しないため、external eval モデルのセットアップを
本スクリプトに集約する。本体 (Python 3.13 + torch>=2.10) とは別 venv を作り、
バージョン非互換 (T-M0.1 §6) を subprocess 境界で隔離する。

参考: docs/milestones.md §M4.2 / docs/training.md §5.2 / docs/open-questions.md §D
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EVAL_ROOT = Path("eval_models")
UTMOS_REPO = "https://github.com/sarulab-speech/UTMOS22"
NISQA_REPO = "https://github.com/gabrielmittag/NISQA"

# UTMOS22 は古い fairseq / PyTorch 想定 (T-M0.1 §6)。本体と分離した専用 Python を使う。
# 暫定: Python 3.9 (UTMOS22 README の動作確認バージョン)。fairseq の wheel 提供状況で 3.10 にも可。
UTMOS_PYTHON = "3.9"
NISQA_PYTHON = "3.9"   # NISQA も古い torch/torchaudio 想定 (要 setup 時検証)


def clone_or_update(repo: str, dest: Path, *, as_submodule: bool) -> None:
    """repo を dest に clone (submodule or 通常 clone)。既存なら fetch/pull。"""
    ...


def setup_utmos(force: bool = False) -> None:
    """1. UTMOS22 を clone (submodule)
       2. uv で専用 venv (Python 3.9) を作り fairseq/torch 等を install
       3. checkpoint を download (UTMOS は Google Drive / HuggingFace ミラー)
       4. eval_models/utmos_manifest.json に paths を記録
    """
    ...


def setup_nisqa(force: bool = False) -> None:
    """NISQA を pip install (専用 venv) or submodule + weights download。
    NISQA の weights は GitHub release (`weights/nisqa.tar` 等) から取得。
    """
    ...


def download_checkpoint(url: str, dest: Path, sha256: str | None = None) -> None:
    """gdown (Google Drive) / requests (GitHub release) で download + sha256 検証.

    UTMOS checkpoint は Google Drive 配布のため `gdown` を専用 venv 側に install して使う。
    fail 時は手動 download 手順 (URL + 配置先) を stderr に出して非ゼロ終了。
    """
    ...


def verify() -> dict[str, bool]:
    """setup 完了確認: clone 済 / venv 済 / checkpoint 存在 / smoke 1 wav を score できるか."""
    ...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["utmos", "nisqa", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="既存を消して再 setup")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=EVAL_ROOT / "eval_manifest.json")
    args = parser.parse_args()
    ...
```

#### `src/wavenext2/eval/run_utmos.py`

```python
"""run_utmos.py — UTMOS22 自動 MOS 推定の wrapper (本体 import path を汚さない).

設計: UTMOS は古い fairseq/torch に依存するため、本体 (Python 3.13) からは
subprocess (専用 venv) 経由で呼ぶのが既定。torch.hub の SpeechMOS backend なら
in-process も可 (§8.1 代替案)。

参考: docs/training.md §5.2 / docs/milestones.md §M4.2
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path

from wavenext2.eval._mos_common import (
    EvalModelNotFoundError,
    load_eval_manifest,
    write_wav_list,
)

# UTMOS22 default system (sarulab-speech)。docs/open-questions.md §D で確定。
DEFAULT_BACKEND = "subprocess"   # "subprocess" | "speechmos" (torch.hub 代替, §8.1)


def score_utmos(
    wav_paths: list[str | Path],
    *,
    device: str = "cuda",
    backend: str = DEFAULT_BACKEND,
    batch_size: int = 16,
    manifest_path: Path = Path("eval_models/eval_manifest.json"),
) -> dict[str, float]:
    """wav ファイル群の UTMOS スコアを返す.

    Args:
        wav_paths: 評価対象 wav の path リスト (24 kHz, mono, [-1,1])
        device: "cuda" / "cpu"。subprocess backend では子プロセスに渡す
        backend: "subprocess" (専用 venv 隔離, 既定) | "speechmos" (torch.hub, in-process)
        batch_size: 子プロセス側の推論 batch
    Returns:
        {"utmos_mean": float, "utmos_std": float, "n": int}
        (per-utt が必要なら別 API `score_utmos_per_utt` を用意)
    Raises:
        EvalModelNotFoundError: setup 未完 / checkpoint 不在
    """
    if backend == "speechmos":
        return _score_speechmos(wav_paths, device=device, batch_size=batch_size)

    manifest = load_eval_manifest(manifest_path)          # 不在なら EvalModelNotFoundError
    py = manifest["utmos"]["python"]                       # 専用 venv の python 実行パス
    score_script = manifest["utmos"]["score_script"]       # UTMOS22 を呼ぶ薄い entry
    with write_wav_list(wav_paths) as list_file:           # tempfile に wav path を列挙
        proc = subprocess.run(
            [py, score_script, "--wav-list", str(list_file),
             "--device", device, "--batch-size", str(batch_size)],
            capture_output=True, text=True, timeout=3600,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"UTMOS subprocess failed:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)                          # 子プロセスが JSON を stdout に出す


def _score_speechmos(wav_paths, *, device, batch_size) -> dict[str, float]:
    """torch.hub の `tarepan/SpeechMOS:v1.2.0` を in-process で使う代替経路 (§8.1).

    SpeechMOS は UTMOS22 を packaging し直したもので Python 3.13 でも動く可能性が高い。
    本体 venv で動けば subprocess 不要 → setup コスト激減。M5 で精度を A/B して採否決定。
    """
    ...
```

#### `src/wavenext2/eval/run_nisqa.py`

```python
"""run_nisqa.py — NISQA (Mittag et al., 2021) MOS 予測 wrapper.

NISQA は ONNX or torch checkpoint。バージョン依存のため subprocess (専用 venv) 既定。
参考: docs/training.md §5.2 / docs/milestones.md §M4.2
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path

from wavenext2.eval._mos_common import (
    EvalModelNotFoundError,
    load_eval_manifest,
    write_wav_list,
)


def score_nisqa(
    wav_paths: list[str | Path],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    manifest_path: Path = Path("eval_models/eval_manifest.json"),
) -> dict[str, float]:
    """wav 群の NISQA mos_pred を返す.

    Returns:
        {"nisqa_mos_mean": float, "nisqa_mos_std": float, "n": int}
        (NISQA は noisiness/coloration/discontinuity/loudness の 4 sub-dim も出すが、
         本チケットでは overall mos_pred のみ。sub-dim は §8.1 で optional)
    Raises:
        EvalModelNotFoundError
    """
    manifest = load_eval_manifest(manifest_path)
    py = manifest["nisqa"]["python"]
    nisqa_root = manifest["nisqa"]["root"]
    weights = manifest["nisqa"]["weights"]
    with write_wav_list(wav_paths) as list_file:
        proc = subprocess.run(
            [py, f"{nisqa_root}/run_predict.py",       # NISQA 公式 entry (mode=predict_dir 相当)
             "--mode", "predict_file", "--pretrained_model", weights,
             "--wav-list", str(list_file), "--device", device],
            capture_output=True, text=True, timeout=3600,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"NISQA subprocess failed:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)
```

#### `src/wavenext2/eval/_mos_common.py`

```python
"""_mos_common.py — UTMOS/NISQA wrapper 共通ユーティリティ."""

from __future__ import annotations
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Iterator


class EvalModelNotFoundError(RuntimeError):
    """UTMOS/NISQA の setup 未完 / checkpoint 不在時に raise.

    message に `scripts/setup_eval_models.py --target {utmos|nisqa}` を案内する。
    """


def load_eval_manifest(path: Path) -> dict:
    """eval_models/eval_manifest.json を読み込み。不在なら EvalModelNotFoundError."""
    if not Path(path).exists():
        raise EvalModelNotFoundError(
            f"Eval manifest not found at {path}. "
            f"Run `uv run python scripts/setup_eval_models.py --target all` first."
        )
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def write_wav_list(wav_paths) -> Iterator[Path]:
    """wav path を 1 行ずつ書いた tempfile を yield (子プロセスへ渡す)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"{Path(p).resolve().as_posix()}\n")
        tmp = Path(f.name)
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| UTMOS system | sarulab-speech/UTMOS22 default | docs/open-questions.md §D, docs/training.md §5.2 |
| NISQA version | gabrielmittag/NISQA **v2** (`weights/nisqa.tar`) | docs/open-questions.md §D |
| UTMOS 期待値 (GT) | **4.0 ± 0.2** | docs/milestones.md §M4.2 Acceptance (LibriTTS-R 高品質) |
| NISQA 期待値 (GT) | **4.5 ± 0.3** | docs/milestones.md §M4.2 Acceptance |
| sample_rate | 24000 Hz (UTMOS/NISQA 入力は内部で 16 kHz 想定 → wrapper 側で resample) | docs/training.md §1.1 |
| UTMOS/NISQA 入力 sr | **16000 Hz** (両モデルとも 16 kHz 学習) | UTMOS22 / NISQA README |
| 専用 venv Python (UTMOS) | **3.9** (fairseq 動作確認版、暫定) | UTMOS22 README, T-M0.1 §6 隔離方針 |
| 専用 venv Python (NISQA) | **3.9** (暫定、setup 時に検証) | NISQA README |
| batch_size (子プロセス推論) | 16 (GPU メモリと相談、§6.1 OOM 項目) | 慣例 |
| 評価 subset (sanity) | test-clean **100 utt** | docs/milestones.md §M4.2 §5 |
| 劣化音声テスト | white noise 付加 **SNR 10 dB** | 本チケット決定 (§5.1) |
| backend 既定 | **subprocess** (隔離) / speechmos (torch.hub 代替) | T-M0.1 §6, §8.1 |
| subprocess timeout | 3600 s (100 utt + モデルロード) | 慣例 |
| checkpoint sha256 検証 | 有効 (download 破損検知) | §6.1 |
| 学習中の計算 | **しない** (重いので最終評価 / smoke のみ) | docs/training.md §6 |

> **resample 注意**: UTMOS / NISQA はいずれも 16 kHz で学習されている。本プロジェクトの生成音声は 24 kHz なので、wrapper (もしくは子プロセス側 entry) で **16 kHz へ resample してから** score する。resample の責務をどちらに置くか (子プロセス側に寄せると本体に torchaudio resample 依存が増えず綺麗) は §6.1 で決定。

### 2.4 アルゴリズム / 処理フロー

**(A) setup (`scripts/setup_eval_models.py`、1 度だけ / CI 不要)**
1. `eval_models/` を作成
2. `--target utmos`: UTMOS22 を `git submodule add` (or 通常 clone) → `eval_models/UTMOS22/`
3. UTMOS 専用 venv を `uv venv --python 3.9 eval_models/.venv-utmos` で作成 → fairseq / torch / torchaudio / hydra / gdown 等を `uv pip install` (UTMOS22 の `requirements.txt` / `environment.yaml` 準拠)
4. checkpoint を `gdown` で Google Drive から download → sha256 検証 → `eval_models/UTMOS22/...`
5. `--target nisqa`: NISQA を clone (or `uv pip install` in 専用 venv) → weights を GitHub release から download
6. UTMOS22 / NISQA 各々に薄い **score entry script** (`--wav-list` を受け取り JSON を stdout に吐く) を配置 (リポジトリ側で `eval_models/_entries/` に保持し setup 時に配置、もしくは submodule 内に追加 patch)
7. `eval_models/eval_manifest.json` に `{"utmos": {"python": ".../bin/python", "score_script": "...", "ckpt": "...", "sha256": "..."}, "nisqa": {...}}` を書き出し
8. `--verify-only`: 各モデルで 1 wav を score して smoke 確認

**(B) scoring (本体側, `score_utmos` / `score_nisqa`)**
1. `load_eval_manifest()` → 不在なら `EvalModelNotFoundError` (setup 案内付き)
2. `write_wav_list(wav_paths)` で tempfile に path 列挙
3. `subprocess.run([専用 venv python, score_entry, "--wav-list", ...], capture_output=True, timeout=3600)`
4. 子プロセス: wav を 16 kHz resample → モデルで推論 → per-utt score を集約 → `{"utmos_mean", "utmos_std", "n"}` を **JSON で stdout** に出力
5. 本体: `returncode != 0` なら stderr 末尾を含めて `RuntimeError`、成功なら `json.loads(stdout)`
6. (speechmos backend 時) torch.hub で in-process 実行、subprocess を介さない

**(C) 統合利用 (M5 / M6 から)**
- 評価ドライバ (T-M5.1 / T-M5.2 / 将来の `compute_metrics` 統合 caller) が GT wav と生成 wav を別々に `score_utmos` / `score_nisqa` に渡し、両者の差分を log

### 2.5 設計上の重要決定

- **subprocess 隔離を既定にする** (T-M0.1 §6 / §リスク表「venv で分離」): UTMOS22 は fairseq、NISQA は古い torch/torchaudio に依存する蓋然性が高く、本体 (Python 3.13 + torch>=2.10) と同居させると `uv sync` の依存解決が破綻する。**専用 venv + subprocess + JSON プロトコル**で本体 import path を一切汚さない。本体側 (`run_utmos.py`) は標準ライブラリ (`subprocess`, `json`, `tempfile`) だけで完結し、UTMOS/NISQA を import しない。
- **`scripts/setup_eval_models.py` に external dep を集約** (uv は submodule 非管理): `uv` は git submodule / checkpoint download を管理しないため、clone・専用 venv 構築・checkpoint download をすべて本スクリプトに寄せる。本体 `pyproject.toml` には UTMOS/NISQA の依存を **一切追加しない** (T-M0.1 の clean な lock を守る)。
- **lazy import 徹底**: `src/wavenext2/eval/__init__.py` で `score_utmos` / `score_nisqa` を export するが、これらは内部で重い依存を import しない。speechmos backend のみ `torch.hub` を関数内 lazy import。これにより本体テスト (`pytest`) が UTMOS/NISQA 不在でも import error にならない。
- **`EvalModelNotFoundError` で fail-fast**: checkpoint / venv 不在時に silent に NaN や 0.0 を返すと、M5/M6 で「品質が出ていない」のか「評価系が動いていない」のか切り分け不能になる。明示 raise + setup コマンド案内。
- **16 kHz resample は子プロセス側**: 本体に余計な resample 経路を増やさないため、wav path を渡して子プロセス側 entry で resample する。本体は 24 kHz wav を書き出すだけ。
- **per-utt と aggregate を分離**: 既定 API は mean/std/n を返す軽量形。per-utt 詳細 (CSV dump) が必要な M6 / ablation 向けに `score_*_per_utt` を別途用意 (本チケットでは aggregate を primary、per-utt は薄く追加)。
- **speechmos を代替 backend として同梱** (§8.1): `torch.hub.load("tarepan/SpeechMOS", "utmos22_strong")` は UTMOS22 を再パッケージしたもので Python 3.13 + torch>=2.10 でも動く可能性がある。動けば setup (clone + 専用 venv + gdown) が丸ごと不要になるため、M5 で精度を A/B して採否を決める。本チケットでは **subprocess を default**、speechmos を opt-in に留める。
- **checkpoint は git 管理しない**: UTMOS / NISQA の weights は数十〜数百 MB かつライセンスが本リポジトリ (MIT, 公開予定) と別。`.gitignore` で除外し、`setup_eval_models.py` で download。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `scripts/setup_eval_models.py` + `run_utmos.py` / `run_nisqa.py` / `_mos_common.py` + テスト本実装、専用 venv 構築の検証 | general-purpose |
| Reviewer | 1 | 隔離設計 (本体 import path 非汚染) の検証、JSON プロトコル妥当性、ライセンス/`.gitignore` 確認、論文 §5.2 / open-questions §D 整合 | general-purpose |
| Tester | 1 | `pytest tests/test_eval_mos.py` 実行 + 実 setup (clone + venv + checkpoint) + GT/劣化音声での MOS 検証 + OOM 確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M4.1 客観評価 / T-M4.3 RTF と機能境界が独立。いずれも T-M0.1 のみ依存)
- 並列実行する場合の最大並列数: **3** (M4 の 3 チケットを並列起動可能。ただし同一 GPU を使う実検証フェーズは GPU メモリ競合に注意)
- size=M。setup スクリプトの外部依存 (clone / download / venv) が重いため、Tester 役を分離した 3 名編成を推奨。

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/setup_eval_models.py` (UTMOS submodule clone + NISQA clone/pip + 専用 venv 構築 + checkpoint download + manifest 生成 + verify)
- `src/wavenext2/eval/run_utmos.py` (`score_utmos`, subprocess backend, speechmos backend stub)
- `src/wavenext2/eval/run_nisqa.py` (`score_nisqa`, subprocess backend)
- `src/wavenext2/eval/_mos_common.py` (`EvalModelNotFoundError`, manifest I/O, wav-list tempfile)
- subprocess + JSON プロトコルによる **本体 (Python 3.13) ↔ 専用 venv 隔離**
- GT 音声での期待値 (UTMOS 4.0±0.2 / NISQA 4.5±0.3) 再現検証
- 劣化音声 (white noise SNR 10 dB) での MOS 低下検証
- test-clean 100 utt バッチ処理
- checkpoint / venv 不在時の `EvalModelNotFoundError`
- 16 kHz resample (子プロセス側) と 24 kHz 入力の橋渡し
- `tests/test_eval_mos.py` (mock subprocess を使った Unit + 実モデル slow test)

### Out of Scope
- MCD / log F0 RMSE (→ **T-M4.1** `compute_metrics.py`)
- RTF 測定 (→ **T-M4.3** `measure_rtf.py`)
- 評価メトリクスの統合集計ドライバ / CSV レポート生成 (M5 / M6 着手時、`compute_metrics` 側で UTMOS/NISQA を呼ぶ統合は後続)
- PESQ / STOI / ViSQOL 等の追加客観指標 (論文非採用、§8.1 で DNSMOS のみ optional 言及)
- NISQA の 4 sub-dimension (noisiness / coloration / discontinuity / loudness) の本格活用 (overall mos_pred のみ。§8.1 で optional)
- 主観 MOS テスト (→ M7, 人手必須)
- 本体 `pyproject.toml` への UTMOS/NISQA 依存追加 (**禁止**。T-M0.1 の lock を汚さない)
- Docker による eval 分離 (§8.1 代替案、M6 クラウド移行時に再検討)
- 学習ループ内での UTMOS/NISQA 計算 (docs/training.md §6「学習中は計算しない」)

### Deliverable
- ファイル:
  - `scripts/setup_eval_models.py` (新規)
  - `src/wavenext2/eval/run_utmos.py` (新規)
  - `src/wavenext2/eval/run_nisqa.py` (新規)
  - `src/wavenext2/eval/_mos_common.py` (新規)
  - `tests/test_eval_mos.py` (新規)
  - `eval_models/.gitkeep` (新規)
- 関数 / クラス:
  - `score_utmos(wav_paths, *, device, backend, batch_size, manifest_path) -> dict[str, float]`
  - `score_nisqa(wav_paths, *, device, batch_size, manifest_path) -> dict[str, float]`
  - `EvalModelNotFoundError(RuntimeError)`
  - `load_eval_manifest(path) -> dict`, `write_wav_list(wav_paths)` (contextmanager)
  - `setup_utmos()`, `setup_nisqa()`, `download_checkpoint()`, `verify()`, `main()` (setup CLI)
- ドキュメント差分:
  - `docs/milestones.md` §M4.2 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M4.2 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_eval_mos.py`)

CI (実モデル不在) でも回せるよう、subprocess を mock した経路と純ロジックを優先する。

- [ ] `test_eval_model_not_found_raises` — manifest 不在で `score_utmos` / `score_nisqa` が `EvalModelNotFoundError` を raise、message に `setup_eval_models.py` を含む
- [ ] `test_write_wav_list_roundtrip` — `write_wav_list` が path を 1 行ずつ POSIX 区切りで書き、抜けた後に tempfile が削除される
- [ ] `test_score_utmos_parses_json` (mock) — `subprocess.run` を monkeypatch して固定 JSON (`{"utmos_mean": 4.05, "utmos_std": 0.12, "n": 100}`) を返させ、`score_utmos` が dict をそのまま返す
- [ ] `test_score_nisqa_parses_json` (mock) — 同上 NISQA 版
- [ ] `test_subprocess_failure_raises` (mock) — `returncode=1` + stderr を返させ、`RuntimeError` に stderr 末尾が含まれる
- [ ] `test_no_heavy_import_at_module_level` — `import wavenext2.eval.run_utmos` が fairseq / NISQA を import しない (`sys.modules` に `fairseq` が現れない / import 時間が閾値以下)
- [ ] `test_manifest_schema` — `load_eval_manifest` がテスト用 manifest を読み、`utmos.python` / `nisqa.python` キーを持つ
- [ ] `test_speechmos_backend_lazy` — `backend="speechmos"` 経路が torch.hub を関数内で lazy import する (module top では import しない)

### 5.2 e2e / 結合テスト (実モデル必要、`@pytest.mark.slow` / `skipif(manifest 不在)`)

- [ ] `scripts/setup_eval_models.py --target all` が clone + 専用 venv + checkpoint download + manifest 生成まで完走 (手動実行 / nightly)
- [ ] `--verify-only` が UTMOS / NISQA それぞれ 1 wav の score を返す
- [ ] `score_utmos([gt_wav])` / `score_nisqa([gt_wav])` が subprocess 経由で finite な float を返す
- [ ] 100 wav バッチが 1 GPU で完走 (timeout 内)、`n == 100`
- [ ] speechmos backend (`backend="speechmos"`) が in-process で動く環境では subprocess 版と平均 0.3 以内で整合 (採否判断材料)

### 5.3 Acceptance criteria (`docs/milestones.md` §M4.2 より転記)
- [ ] GT 音声で UTMOS ≈ **4.0 ± 0.2** (LibriTTS-R 高品質前提)
- [ ] GT 音声で NISQA ≈ **4.5 ± 0.3**

### 5.4 追加 acceptance (本チケット独自、タスク指定 §5 より)
- [ ] 既知の劣化音声 (GT に white noise を SNR 10 dB で付加) で UTMOS / NISQA が GT より **有意に低下** (各 mean が GT mean を下回る)
- [ ] バッチ処理で test-clean サブセット (100 utt) を 1 回の呼び出しで処理し mean / std / n を返す
- [ ] UTMOS / NISQA の checkpoint が見つからない場合に `EvalModelNotFoundError` を raise (silent NaN なし、setup 手順を案内)
- [ ] `from wavenext2.eval.run_utmos import score_utmos` / `from wavenext2.eval.run_nisqa import score_nisqa` が UTMOS/NISQA 不在環境でも import 成功 (lazy import)
- [ ] 本体 `pyproject.toml` / `uv.lock` に UTMOS/NISQA 由来の依存が追加されていない

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目

- **CRITICAL: UTMOS22 の古い fairseq / PyTorch / Python 依存 (Python 3.13 で動かない)** [T-M0.1 §6 由来]:
  - UTMOS22 は fairseq (wav2vec2 ベース) に依存し、fairseq は新しい PyTorch / Python で頻繁に build 失敗する。本体 (Python 3.13 + torch>=2.10) の `.venv` に同居させると `uv sync` が破綻する蓋然性が高い。
  - **対策 (本チケット採用)**: 専用 venv (`eval_models/.venv-utmos`, Python 3.9 暫定) + `subprocess` 隔離 + JSON プロトコル。本体は UTMOS を一切 import しない。
  - **再評価トリガー**: setup で fairseq が Python 3.9 でも build 失敗する場合、(a) Python 3.8 へ降格、(b) speechmos backend (§8.1) に全面移行、(c) Docker 隔離 (§8.1) のいずれかへ。
- **CRITICAL: checkpoint download の自動化 (UTMOS = Google Drive, NISQA = GitHub release)**:
  - UTMOS の checkpoint は Google Drive 配布で、`gdown` が Google の "confirm token" 仕様変更で fail することがある。NISQA weights は GitHub release で比較的安定。
  - **対策**: `download_checkpoint` で `gdown` 失敗時に **手動 download 手順 (URL + 配置先 path + 期待 sha256) を stderr に出して非ゼロ終了**。CI ではこの経路を踏まない (実モデルテストは slow/手動)。HuggingFace ミラーが存在すれば fallback URL に追加。
  - **検知**: sha256 検証で download 破損を検出。manifest に記録した sha256 と照合。
- **CRITICAL: GPU メモリ OOM (UTMOS + NISQA + 本体モデルの同時ロード)** [タスク §6 由来]:
  - M5 / M6 で本体 generator が GPU に載った状態で UTMOS (wav2vec2 large) + NISQA を同時ロードすると OOM。
  - **対策**: subprocess 隔離なら **本体プロセスと別プロセス**で動くため、評価フェーズでは本体モデルを `.cpu()` に退避 or 解放してから score を呼ぶ運用を §9.1 で後続に申し送り。さらに `batch_size` を 16 → 8/4 に絞れる引数を用意。UTMOS と NISQA を**逐次** (同時でなく順番に) 呼ぶ。
  - **検知**: 子プロセスの CUDA OOM は `returncode != 0` + stderr に "out of memory" → wrapper が batch_size 半減で 1 回 retry する optional ロジック (§8.1)。

#### 通常項目

- **subprocess JSON プロトコルの脆さ**: 子プロセスが stdout に print デバッグを混ぜると `json.loads` が壊れる。**対策**: 子プロセス entry は score 以外の出力を stderr に流し、stdout は JSON 1 行のみに統一。`json.loads(proc.stdout)` 失敗時は stdout/stderr 両方を例外に含める。
- **16 kHz resample の責務配置**: UTMOS/NISQA は 16 kHz 学習。24 kHz → 16 kHz resample を子プロセス側 (torchaudio/librosa) で行う。**本体に resample 経路を増やさない**ことで T-M0.1 lock を保つ。resample 品質 (sinc vs kaiser) は MOS にほぼ影響しないため default で可。
- **`uv venv --python 3.9` で 3.9 が無い環境**: `uv python install 3.9` を setup スクリプトが先に実行。`only-managed` (T-M0.1) と整合。
- **submodule vs 通常 clone**: タスク仕様は「UTMOS = submodule、NISQA = pip install (or submodule)」。submodule にすると `.gitmodules` がリポジトリに入り clone 時に `--recursive` が要る。**本チケットは submodule を default**、ただし checkpoint / venv は別途 setup スクリプト依存のため、submodule にしてもそのままでは動かない点を README に明記。pip 経路 (NISQA) の方が setup は単純。
- **UTMOS22 / NISQA に score entry script を追加する方法**: 両 repo の公式 CLI が「ディレクトリ内 wav を一括 score → CSV」形式の場合、本チケットの `--wav-list` + JSON stdout 形式に合わせる薄い entry を `eval_models/_entries/score_utmos_entry.py` 等にリポジトリ側で持ち、setup 時に専用 venv から実行できる場所へ配置 (submodule 本体は改変しない)。
- **NISQA が ONNX か torch checkpoint か** [タスク §6 由来]: NISQA v2 は PyTorch checkpoint (`.tar`) が標準。ONNX 版もあるが本チケットは torch を default。バージョン依存で `run_predict.py` の引数名が変わるため setup 時に NISQA の README を確認 (`--pretrained_model` / `--mode`)。
- **decode 安定性**: UTMOS は同一 wav に対して deterministic (推論のみ)。seed 不要。NISQA も同様。再現性テストは「同一 wav → 同一 score」で確認可能 (slow test)。
- **test-clean 表記揺れ**: `docs/training.md` / `docs/milestones.md` で "test-clean-100" / "test-clean" の表記が混在 (M0 phase review で T-M0.3 に解決申し送り済)。本チケットは **T-M0.3 が生成する test filelist** をそのまま使い、subset 100 utt は先頭 100 行で固定 (seed 不要、決定的)。
- **CPU-only 環境での score**: `device="cpu"` で UTMOS/NISQA とも動く (遅いが可)。CI の slow test は GPU 不在でも `device="cpu"` で 1〜数 wav なら回せる余地あり (時間次第で skip)。

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目: **解決済み** (§D「UTMOS / NISQA バージョン: sarulab-speech/UTMOS22 default、gabrielmittag/NISQA v2」標準で確定)。
- 本チケットで決定する曖昧さ:
  - 隔離方式 (subprocess + 専用 venv) — §2.5 / §6.1 で **subprocess を default 採用**、speechmos / Docker は §8.1 代替。
  - 専用 venv の Python バージョン (UTMOS/NISQA とも 3.9 暫定) — setup 時に fairseq / NISQA の動作可否で確定。
  - resample 責務 (子プロセス側) — §6.1 で確定。
  - 期待値の許容幅 (UTMOS 4.0±0.2 / NISQA 4.5±0.3) — milestones.md の値をそのまま acceptance に採用。実測が外れたら subset / resample 設定を見直す (§8.2)。

### 6.3 他チケットとの整合性
- **T-M0.1 (Python 環境)** から受領:
  - 本体は Python 3.13 + torch>=2.10、`uv` 管理、`python-preference = "only-managed"`。本チケットは **この lock を一切汚さない** (UTMOS/NISQA 依存を `pyproject.toml` に足さない)。
  - T-M0.1 §6 / Out of Scope で「UTMOS / NISQA 用の別 venv 構築は必要時に別チケット化」と申し送られた → **本チケットがそれを実行**。`uv python install` / `uv venv --python 3.9` を setup スクリプトで使う。
- **T-M4.1 (客観評価 MCD / logF0RMSE)** と並列:
  - 同じ `src/wavenext2/eval/` 配下。`__init__.py` の `__all__` に互いの export を追加するため、編集競合に注意 (どちらが先でも追記マージ)。
  - 将来の統合集計ドライバ (`compute_metrics.py` 側 or 新規 caller) が MCD/logF0 と UTMOS/NISQA を**両方呼ぶ**。本チケットは MOS API を提供するのみ。
- **T-M4.3 (RTF)** と並列: 機能独立。GPU 実検証フェーズで同一 GPU を使うなら逐次実行。
- **T-M5.1 / T-M5.2 (smoke)** へ渡す:
  - 1 epoch 後の生成 wav を `score_utmos` / `score_nisqa` で評価する caller を smoke 側が組む。本チケットは API + setup を提供。
  - M5 が「UTMOS/NISQA が動かない / 期待値から大きく外れる」場合の **再評価トリガー** (§8.1) を共有。

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 論文 §5.2 / `docs/training.md` §5.2 / `docs/open-questions.md` §D との整合 (UTMOS22 default, NISQA v2)
- [ ] **本体 import path が汚れていない**: `import wavenext2.eval.run_utmos` で fairseq / NISQA が import されない (lazy import 徹底)
- [ ] 本体 `pyproject.toml` / `uv.lock` に UTMOS/NISQA 由来の依存が **追加されていない**
- [ ] subprocess + JSON プロトコルが堅牢 (stdout は JSON のみ、デバッグ出力は stderr)
- [ ] `EvalModelNotFoundError` が setup 未完時に raise され、message が `setup_eval_models.py` を案内
- [ ] checkpoint download に sha256 検証があり、download 破損 / Google Drive fail 時に手動手順を案内
- [ ] Acceptance: GT で UTMOS 4.0±0.2 / NISQA 4.5±0.3、劣化音声で低下、100 utt バッチ完走
- [ ] GPU OOM 対策 (subprocess 別プロセス + batch_size 可変 + UTMOS/NISQA 逐次) が考慮されている
- [ ] `.gitignore` で clone 物 / checkpoint / 専用 venv を除外、`eval_models/.gitkeep` のみ keep
- [ ] ライセンス: UTMOS / NISQA の weights を **commit していない** (本リポジトリ MIT / 公開予定と別ライセンス)
- [ ] CLAUDE.md ポリシー: 参考実装 (UTMOS22 / NISQA) のコードを**コピーしていない** (clone して subprocess で呼ぶのは可、ソース流用は不可)
- [ ] CI で回る Unit テスト (mock subprocess) と実モデル slow テストが明確に分離

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **UTMOS のみ採用 (NISQA は optional)** | setup が半減、NISQA のバージョン依存リスクを回避 | 論文は UTMOS + NISQA 両方を報告 → 対比が片肺 | 論文再現性のため両方実装。ただし NISQA を `--target utmos` で skip 可能にして optional 化の余地は残す | **NISQA の setup が解決不能なとき** (weights download / 古い torch build が詰む) → UTMOS 単独で M5/M6 を進める |
| **SpeechMOS (torch.hub) を UTMOS22 の代替に** | `torch.hub.load("tarepan/SpeechMOS", "utmos22_strong")` の 1 行で in-process、専用 venv / clone / gdown が全部不要、Python 3.13 + torch>=2.10 で動く可能性大 | UTMOS22 公式とスコアが微妙にずれる可能性 (再パッケージ版)、論文の "UTMOS" が公式実装を指す厳密性が薄れる | 公式 UTMOS22 を default にしつつ、speechmos を **opt-in backend** として同梱。M5 で精度 A/B | **M5 で公式 UTMOS22 が動かない or speechmos と平均 0.1 以内で一致**したとき → speechmos を default 昇格、setup を撤去 |
| **DNSMOS (Microsoft) を追加** | ONNX で軽量・依存が薄い、noise/reverb に強い | 論文非採用指標、評価軸が増えるだけ | 論文は UTMOS/NISQA のみ。DNSMOS は補助としてのみ価値 | **UTMOS/NISQA の setup が両方詰んだとき**の最終 fallback、または M6 ablation で追加軸が欲しいとき |
| **Docker で eval 分離 (UTMOS/NISQA イメージ)** | OS / Python / CUDA をイメージに封じ込め、fairseq build 地獄を 1 度だけ解決 | Docker 依存、GPU passthrough 設定、本体との wav 受け渡しに volume mount | ローカル開発では subprocess + 専用 venv が軽量。Docker は M6 クラウド移行で価値 | **M6 で GPU クラスタへ移行**し eval を再現可能イメージ化したいとき (T-M0.1 §8.1 Docker 項と連動) |
| **別 venv ではなく conda env で隔離** | conda は古い CUDA toolkit / Python を env に閉じ込めやすく fairseq の動作実績が多い | プロジェクトは uv 採用 (T-M0.1)、conda 混在は方針逸脱 | uv + 専用 venv で隔離意図は達成できる | **uv venv で fairseq が Python 3.8/3.9 でも build 失敗**し、conda-forge の fairseq build なら通るとき |
| **REST API 化 (eval サーバを別プロセス常駐)** | モデルロードが 1 回で済み、100 utt × 複数回呼んでも高速 | 常駐プロセス管理が複雑、ローカル単発評価には過剰 | 本チケットは単発 subprocess で十分。常駐は M6 大量評価で検討 | **M6 で test-clean 全 4824 utt を何度も評価**しモデルロード overhead が支配的になったとき |

### 8.2 思想 / 哲学の見直し
- **粒度**: T-M4.2 は size=M。setup スクリプト (重い外部依存) + 2 つの wrapper + テストで妥当。setup と wrapper を別チケットに割る案もあるが、両者は manifest スキーマで密結合のため 1 チケットが健全。
- **subprocess JSON プロトコルの一般化**: UTMOS / NISQA で同じ「wav-list 入力 → JSON stdout」プロトコルを使うので、将来 DNSMOS 等を足すときも同じ枠に乗る。`_mos_common.py` にプロトコルを集約した判断は再利用性が高い。
- **期待値 acceptance の扱い**: UTMOS 4.0±0.2 / NISQA 4.5±0.3 は milestones.md の見積り。実 LibriTTS-R GT で外れたら、(a) 16 kHz resample 設定、(b) subset の選び方、(c) UTMOS strong/wide の別、を見直す。acceptance を「GT > 劣化音声」の**相対比較**に軸足を移す案も M4 phase review で検討。
- **本体プロジェクトとの境界**: UTMOS/NISQA を本体 `pyproject.toml` に絶対入れないという制約は、再現性 (T-M0.1 lock のクリーンさ) を守る上で重要。この境界を破ると `uv sync` が壊れて全フェーズに波及するため、§7 レビューで厳格にチェック。

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **インターフェース**:
  - `score_utmos(wav_paths: list[str|Path], *, device="cuda", backend="subprocess", batch_size=16, manifest_path=...) -> {"utmos_mean", "utmos_std", "n"}`
  - `score_nisqa(wav_paths: list[str|Path], *, device="cuda", batch_size=16, manifest_path=...) -> {"nisqa_mos_mean", "nisqa_mos_std", "n"}`
  - 両者とも setup 未完なら `EvalModelNotFoundError`、subprocess 失敗なら `RuntimeError` (stderr 付き)
  - GT wav と生成 wav を**別々に**呼んで差分を取る (caller 責務)
- **設定値**: UTMOS/NISQA 入力は 16 kHz (子プロセスで resample)。本体は 24 kHz wav を書き出すだけ。専用 venv は `eval_models/eval_manifest.json` の `python` パスで参照。
- **セットアップ**: 初回のみ `uv run python scripts/setup_eval_models.py --target all` (clone + 専用 venv + checkpoint download)。CI では実行しない。
- **GPU OOM 運用** (T-M5.1 / T-M5.2 / T-M6.1 / T-M6.2 へ): 評価フェーズでは本体 generator を解放 / `.cpu()` 退避してから `score_*` を呼ぶ。UTMOS と NISQA は逐次呼ぶ。OOM 時は `batch_size` を 8/4 に下げる。
- **注意事項**:
  - 本体テスト (`pytest`) は UTMOS/NISQA 不在でも import 成功する (lazy import)。実モデルテストは `@pytest.mark.slow` + `skipif(manifest 不在)`。
  - `score_*` は学習ループ内で呼ばない (重い、docs/training.md §6)。最終評価 / smoke の検証点のみ。

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M4.2 の Acceptance チェックボックス (UTMOS 4.0±0.2 / NISQA 4.5±0.3) を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M4.2 ステータスを `📝 pending` → `✅ completed`、M4 進捗サマリ更新
  - [ ] (該当時) `README.md` に eval モデル setup 手順 (`scripts/setup_eval_models.py`) を追記
  - [ ] (該当時) `docs/implementation-plan.md` §評価インフラのチェックリスト更新

### 9.3 Open question として残ったもの
- 公式 UTMOS22 (fairseq) を Python 3.9 で動かせるか、それとも speechmos (§8.1) に倒すかは **setup 実行時に確定**。詰んだ場合は `docs/open-questions.md` 補遺に「UTMOS は speechmos backend で代替」と追記要否を検討。
- 専用 venv の Python バージョン (3.9 暫定) が fairseq / NISQA の双方で通らない場合、3.8 降格 or Docker (§8.1) への移行判断を M4 phase review で記録。
- UTMOS checkpoint の Google Drive download が CI / クラウド (M6) で安定しない場合、HuggingFace ミラー URL の確定と manifest への fallback 追加が必要。
