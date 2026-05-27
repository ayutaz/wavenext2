"""乱数 seed の一括設定ユーティリティ (T-M2.5)。

torch / numpy / random / CUDA を一括設定。`deterministic=True` のとき cudnn を決定論モードに。
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """torch / numpy / random / CUDA の seed を一括設定する。

    Args:
        seed: 乱数 seed。
        deterministic: True で cudnn.deterministic=True / benchmark=False (再現性優先、低速)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
