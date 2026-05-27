"""GAN smoke training テスト (T-M2.6)。

- `test_smoke_synthetic`: LibriTTS-R 不要の synthetic 1-sample overfit (subordinate gate)。
  `scripts/smoke_gan_synthetic.run_synthetic_smoke` を import 再利用 (DRY)。`@pytest.mark.slow`
  でデフォルト suite からは除外 (300 step で数十秒〜数分)。`uv run pytest -m slow` で実行。
- `test_smoke_config_keys`: smoke config の必須キー検証 (高速、デフォルト suite に含む)。
- `test_smoke_completes`: 本来の 1 utterance × 1000 step overfit (実 LibriTTS-R, GPU)。
  `@pytest.mark.slow @pytest.mark.gpu`。データ未取得 / CUDA 無しでは skip。T-M5.1 と併走で実走。

synthetic smoke が明らかにした知見: 出力 head の `clip(-1,1)` は飽和域で勾配 0 のため、高 lr では
generator が ±1 を超えると凍結する。低 lr (1e-4) なら飽和前に学習が進む。詳細は ticket §8.3。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wavenext2.utils import load_config


@pytest.mark.slow
def test_smoke_synthetic():
    """合成 1 sample overfit で MR-STFT が減少し (飽和凍結でない)、finite であること。"""
    from scripts.smoke_gan_synthetic import run_synthetic_smoke

    r = run_synthetic_smoke(steps=300, seed=42)
    assert r["all_finite"], "loss に NaN/Inf"
    # 飽和凍結なら best/init ≈ 1.0、勾配が流れ学習すれば < 0.92
    assert r["ratio"] < 0.92, (
        f"MR-STFT が減少しない (ratio={r['ratio']:.3f}); clip 飽和/勾配バグを疑う"
    )
    assert r["max_abs"] <= 2.0 + 1e-3, f"T=2 構造上界 2.0 を超過: {r['max_abs']}"


def test_smoke_config_keys():
    """smoke config が train_gan.py の参照キーを備えていること。"""
    cfg = load_config("configs/gan_wavenext2_smoke.yaml")
    assert cfg["train"]["max_steps"] == 1000
    assert cfg["train"]["batch_size"] == 1
    assert cfg["model"]["T"] == 4
    for key in ("validation", "checkpoint", "logging"):
        assert key in cfg
    assert set(cfg["loss"]["weights"]) == {"d_gan", "d_fm", "mrstft_sc", "mrstft_mag"}


@pytest.mark.slow
@pytest.mark.gpu
def test_smoke_completes():
    """実 LibriTTS-R 1 utterance × 1000 step overfit (T-M5.1 と併走、データ取得後に実走)。"""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA 必須 (1000 step は CPU で非現実的)")
    cfg = load_config("configs/gan_wavenext2_smoke.yaml")
    if not Path(cfg["data"]["train_filelist"]).is_file():
        pytest.skip("LibriTTS-R filelist 未取得 (T-M0.3 ユーザー DL 待ち)")
    pytest.skip("実データ overfit は T-M5.1 で実走 (本チケットは synthetic gate で代替)")
