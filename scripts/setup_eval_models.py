"""setup_eval_models.py — UTMOS / NISQA の clone + checkpoint download + 専用 venv 構築 (T-M4.2).

本体 (Python 3.13 + torch>=2.10) と UTMOS22 (旧 fairseq) / NISQA (旧 torch) のバージョン非互換を
**別 venv + subprocess 境界**で隔離する。`uv` は git submodule を直接管理しないため、external eval
モデルのセットアップを本スクリプトに集約する。

**ネットワーク + 大容量 download + Python 3.9 venv 構築を伴うユーザー実行スクリプト**。CI では走らせず、
M5/M6 で MOS 評価が必要になった環境で 1 度だけ実行する。default の UTMOS は speechmos (本体 venv 内
`uv add speechmos` で十分、本 setup は不要) を使うため、本スクリプトは主に **NISQA subprocess 経路**と
**fairseq UTMOS22 (opt-in)** のためにある。

使い方:
    uv run python scripts/setup_eval_models.py --utmos --nisqa
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_ROOT = Path("eval_models")
UTMOS_REPO = "https://github.com/sarulab-speech/UTMOS22"
NISQA_REPO = "https://github.com/gabrielmittag/NISQA"
UTMOS_PYTHON = "3.9"  # UTMOS22 README 動作確認バージョン (fairseq wheel 提供状況で要調整)
NISQA_PYTHON = "3.9"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


def clone_or_update(repo: str, dest: Path) -> None:
    """repo を dest に clone (既存なら fetch/pull)。"""
    if dest.exists():
        _run(["git", "-C", str(dest), "pull", "--ff-only"])  # noqa: S607
    else:
        _run(["git", "clone", "--depth", "1", repo, str(dest)])  # noqa: S607


def setup_nisqa() -> None:
    """NISQA を clone し、Python 3.9 専用 venv に旧 torch/torchaudio を install。

    weights は GitHub release から download (NISQA repo に同梱の場合もある)。venv 側に
    `run_nisqa_json.py` (wav path 群を受け JSON で MOS を返す runner) を配置する想定。
    実際の download/install はネットワーク + 環境依存のため、本関数は手順を整えるのみ。
    """
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    clone_or_update(NISQA_REPO, EVAL_ROOT / "NISQA")
    venv = EVAL_ROOT / ".venv-nisqa"
    if not venv.exists():
        _run(["uv", "venv", "--python", NISQA_PYTHON, str(venv)])  # noqa: S607
    print(
        "NISQA venv に旧 torch/torchaudio + NISQA 依存を install し、"
        "eval_models/NISQA/run_nisqa_json.py (JSON runner) を配置してください。"
        "詳細は NISQA README を参照。"
    )


def setup_utmos() -> None:
    """UTMOS。default は本体 venv の speechmos (`uv add speechmos`) で済むためそれを案内。

    fairseq UTMOS22 (opt-in) を使う場合のみ clone + Python 3.9 venv を構築する。
    """
    print(
        "UTMOS default は speechmos 経路を推奨します: 本体 venv で `uv add speechmos`。\n"
        "fairseq UTMOS22 (opt-in) を使う場合のみ以下を実行:"
    )
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    clone_or_update(UTMOS_REPO, EVAL_ROOT / "UTMOS22")
    venv = EVAL_ROOT / ".venv-utmos"
    if not venv.exists():
        _run(["uv", "venv", "--python", UTMOS_PYTHON, str(venv)])  # noqa: S607
    print("UTMOS22 venv に fairseq/torch を install し checkpoint を download してください。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Setup external eval models (UTMOS / NISQA)")
    parser.add_argument("--utmos", action="store_true", help="UTMOS22 (fairseq opt-in) を setup")
    parser.add_argument("--nisqa", action="store_true", help="NISQA subprocess venv を setup")
    args = parser.parse_args(argv)
    if not (args.utmos or args.nisqa):
        parser.error("--utmos / --nisqa のいずれかを指定してください")
    if args.utmos:
        setup_utmos()
    if args.nisqa:
        setup_nisqa()
    return 0


if __name__ == "__main__":
    sys.exit(main())
