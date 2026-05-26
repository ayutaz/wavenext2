"""wavenext2.losses — GAN-WaveNeXt 2 用 loss 群 + 集約ヘルパ。"""

from __future__ import annotations

import torch

from wavenext2.losses.adversarial import HingeGANLoss
from wavenext2.losses.feature_matching import FeatureMatchingLoss
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss

__all__ = [
    "DEFAULT_WEIGHTS",
    "FeatureMatchingLoss",
    "HingeGANLoss",
    "MultiResolutionSTFTLoss",
    "compute_total_loss",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "d_gan": 1.0,
    "d_fm": 10.0,
    "mrstft_sc": 2.5,
    "mrstft_mag": 2.5,
}


def compute_total_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """名前付き loss の重み付き和。Returns (total, per-component unweighted scalar dict)。

    `sorted(losses)` で加算順を固定 (bf16 の float 加算非可換による再現性破壊を防ぐ)。
    unweighted dict は `.item()` 済み Python float (TensorBoard ログ用、graph 非保持)。
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    keys = sorted(losses.keys())
    if not keys:
        raise ValueError("losses dict is empty")
    total: torch.Tensor | None = None
    unweighted: dict[str, float] = {}
    for k in keys:
        if k not in weights:
            raise KeyError(f"loss key {k!r} has no weight in {sorted(weights)}")
        v = losses[k]
        unweighted[k] = v.detach().item()
        contrib = weights[k] * v
        total = contrib if total is None else total + contrib
    assert total is not None
    return total, unweighted
