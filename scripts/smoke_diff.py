"""smoke_diff.py — Diff smoke の薄い wrapper (T-M3.5, DRY)。

ロジックは `tests/test_train_diff_overfit.py` (pytest single-source) に集約し、本 script は
pytest を subprocess 起動するだけ (T-M2.6 GAN smoke と同じパターン)。primary = sub-model 4。

使い方:
    uv run python scripts/smoke_diff.py             # 実データ GPU smoke (slow+gpu、要 GPU+データ)
    uv run python scripts/smoke_diff.py --synthetic # LibriTTS-R 不要の synthetic CPU gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Diff smoke via pytest (thin wrapper)")
    parser.add_argument(
        "--synthetic", action="store_true", help="synthetic CPU gate のみ実行 (データ不要)"
    )
    args = parser.parse_args(argv)

    target = (
        "tests/test_train_diff_overfit.py::test_smoke_synthetic_cpu"
        if args.synthetic
        else "tests/test_train_diff_overfit.py::test_smoke_completes_sub_model_4"
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-v", "-m", "slow"], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
