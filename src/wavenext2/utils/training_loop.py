"""training_loop.py — GAN/Diff 訓練ループの共通プリミティブ (T-M3.2 で T-M2.5 から抽出).

GAN (`train_gan.py`) と Diff (`train_diff.py`) は **2 並列実装** で確定 (M2 review)。両者の
`*_step` 関数と checkpoint payload は signature が本質的に異なる (GAN: G/D/opt×2/sch×2、Diff:
単一 model/opt/sub_model_k) ため統一しない。一方、以下の **真に共通な低レベルプリミティブ** だけは
両者でコピペになるため本モジュールに集約する:

- `iter_forever(loader)`         : DataLoader を epoch 境界をまたいで無限に回す
- `atomic_save(obj, path)`       : `.tmp` → os.replace の原子的書き込み (preemption 耐性)
- `capture_rng_state()`          : torch/cuda/numpy/python の RNG state を dict で取得
- `restore_rng_state(rng)`       : 上記を復元 (resume 完全性)
- `register_sigterm_handler(fn)` : SIGTERM/SIGINT で emergency save → SystemExit

checkpoint の save/load 自体は payload が異なるため各 trainer に残し、本モジュールの
`atomic_save` / `capture_rng_state` / `restore_rng_state` を内部で使う。
"""

from __future__ import annotations

import os
import random
import signal
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

__all__ = [
    "atomic_save",
    "capture_rng_state",
    "iter_forever",
    "register_sigterm_handler",
    "restore_rng_state",
]


def iter_forever(loader: DataLoader) -> Iterator[Any]:
    """DataLoader を無限に回す (epoch 境界をまたぐ)。"""
    while True:
        yield from loader


def atomic_save(obj: dict, path: Path) -> None:
    """`path.tmp` に保存後 `os.replace` で原子的に rename する.

    途中で kill されても `path` 自体が中途半端な状態にならないことを保証 (POSIX/Windows 共通)。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def capture_rng_state() -> dict:
    """torch / cuda / numpy / python の RNG state を dict で取得 (resume 完全性)。"""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(rng: dict | None) -> None:
    """`capture_rng_state` で取得した RNG state を復元する (None / 欠損キーは skip)。"""
    if not rng:
        return
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])


def register_sigterm_handler(save_fn: Callable[[], None]) -> None:
    """SIGTERM / SIGINT 受信時に `save_fn()` を呼んでから SystemExit(0)。

    M6 cluster preemption (Slurm / Kubernetes) 対策。save_fn は引数なしの closure
    (各 trainer が checkpoint 保存処理を束ねる)。
    """

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        save_fn()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
