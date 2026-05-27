"""smoke_gan.py — GAN smoke の薄い wrapper (T-M2.6, DRY)。

ロジックは `tests/test_train_gan_overfit.py` (pytest single-source) に集約し、本 script は
pytest を subprocess 起動するだけ。2 経路維持を避ける。

使い方:
    uv run python scripts/smoke_gan.py             # 実データ smoke (slow+gpu、データ要)
    uv run python scripts/smoke_gan.py --synthetic # LibriTTS-R 不要の synthetic gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GAN smoke via pytest (thin wrapper)")
    parser.add_argument(
        "--synthetic", action="store_true", help="synthetic CPU gate のみ実行 (データ不要)"
    )
    args = parser.parse_args(argv)

    target = (
        "tests/test_train_gan_overfit.py::test_smoke_synthetic"
        if args.synthetic
        else "tests/test_train_gan_overfit.py::test_smoke_completes"
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-v", "-m", "slow"], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
