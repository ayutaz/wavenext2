"""YAML config の読み込み (T-M2.5)。

`load_config(path)` で YAML を dict に読み込む。secrets (API key 等) は config に書かず
.env / 環境変数経由とする方針 (configs/*.yaml ヘッダ参照)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config"]


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML config を dict として読み込む。

    Args:
        path: YAML ファイルパス。

    Returns:
        パース済み dict。

    Raises:
        FileNotFoundError: path が存在しない場合。
        ValueError: top-level が mapping でない場合。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config top-level must be a mapping, got {type(cfg).__name__}")
    return cfg
