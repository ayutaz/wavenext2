"""pytest 共有 fixture.

TODO: synthetic audio / mel / config fixture は利用側チケットで順次追加。
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seed():
    """全テストで torch / numpy 系の seed を固定し決定論性を担保."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# CPU/GPU 両方でテストするための device パラメータ (CI は CPU のみ)
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param
