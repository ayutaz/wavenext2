"""smoke_gan_synthetic.py — synthetic (sine 波) 1 sample overfit smoke (T-M2.6).

LibriTTS-R を必要としない CPU 完結の subordinate gate。固定の合成波形 (2 正弦波) を 1 sample
として、その mel を条件に GANWaveNext2 を数百 step 訓練し、**MR-STFT loss が減少する** ことで
訓練 stack (Generator / Discriminator / Loss / train_gan_step) の勾配が正しく流れることを検証する。

本来の 1 sample × 1000 step overfit (実 LibriTTS-R、MR-STFT < 初期値 10%) は T-M5.1 と併せて
実データ取得後に GPU で実走する (`tests/test_train_gan_overfit.py::test_smoke_completes`)。

設計:
- warmup の遅い InverseLR ではなく **定数 LR** (LambdaLR ×1.0) を使い、数百 step で可視な減少を得る
  (T-M2.6 §6.1 「scheduler bypass」修正案 2)。
- overfit ロジックの単一情報源。`tests/test_train_gan_overfit.py` はこの関数を import して検証する (DRY)。
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from wavenext2.data.mel import LogMelSpectrogram
from wavenext2.models.discriminator import MultiScaleDiscriminator
from wavenext2.models.gan_wavenext2 import GANWaveNext2
from wavenext2.train.train_gan import _Criteria, train_gan_step
from wavenext2.utils.seed import set_seed

# 合成 smoke 用の最小 GAN 設定 (本番 configs/gan_wavenext2.yaml と分離)
_SMOKE_CFG = {
    "train": {"grad_clip_norm": 1.0},
    "loss": {"weights": {"d_gan": 1.0, "d_fm": 10.0, "mrstft_sc": 2.5, "mrstft_mag": 2.5}},
}
_MEL_CFG = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 300,
    "win_length": 1200,
    "n_mels": 128,
    "f_min": 20.0,
    "f_max": 12000.0,
}


def _synthetic_target(segment_length: int, device: torch.device) -> torch.Tensor:
    """2 正弦波 (220Hz + 440Hz) の固定波形 (1, segment_length)、peak≈0.7。"""
    t = torch.arange(segment_length, dtype=torch.float32, device=device) / _MEL_CFG["sample_rate"]
    wav = 0.4 * torch.sin(2 * math.pi * 220.0 * t) + 0.3 * torch.sin(2 * math.pi * 440.0 * t)
    return wav.unsqueeze(0)


def run_synthetic_smoke(
    steps: int = 300,
    T: int = 2,
    segment_length: int = 4800,
    lr: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, float | bool]:
    """合成 1 sample を overfit し、MR-STFT 減少 / finite / 出力範囲を計測して返す。

    Returns:
        dict: init_mrstft / final_mrstft / ratio / max_abs / all_finite / steps。
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)

    target = _synthetic_target(segment_length, dev)
    mel = LogMelSpectrogram.from_config(_MEL_CFG).to(dev)(target)  # (1, 128, T_mel)

    G = GANWaveNext2(T=T).to(dev)
    D = MultiScaleDiscriminator().to(dev)
    opt_G = AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-3)
    opt_D = AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-3)
    sch_G = LambdaLR(opt_G, lambda _: 1.0)  # 定数 LR (warmup bypass)
    sch_D = LambdaLR(opt_D, lambda _: 1.0)
    crit = _Criteria()
    crit.mrstft.to(dev)  # MR-STFT の window buffer を device へ

    G.train()
    D.train()
    init_vals: list[float] = []
    history: list[float] = []
    all_finite = True
    for step in range(steps):
        logs = train_gan_step(G, D, opt_G, opt_D, sch_G, sch_D, mel, target, _SMOKE_CFG, crit)
        if not all(math.isfinite(v) for v in logs.values()):
            all_finite = False
            break
        mrstft = logs["loss_g_mrstft_sc"] + logs["loss_g_mrstft_mag"]
        history.append(mrstft)
        if step < 5:  # 定数 LR (warmup bypass) なので step 0-4 が真の初期 loss
            init_vals.append(mrstft)
        if verbose and step % 50 == 0:
            print(f"[smoke] step {step:4d}  MR-STFT={mrstft:.4f}  loss_G={logs['loss_G']:.4f}")

    init_mrstft = max(sum(init_vals) / max(len(init_vals), 1), 1e-4)  # floor で ratio 安定化
    # full-GAN は非単調 (adversarial/FM が摂動) のため、到達した最小値で「学習したか」を判定
    best_mrstft = min(history) if history else float("nan")
    with torch.no_grad():
        G.eval()
        y0 = G(mel)[..., :segment_length]
    return {
        "init_mrstft": init_mrstft,
        "final_mrstft": history[-1] if history else float("nan"),
        "best_mrstft": best_mrstft,
        "ratio": best_mrstft / init_mrstft,  # best/init: 飽和凍結 (≈1.0) と学習 (<1) を判別
        "max_abs": float(y0.abs().max()),
        "all_finite": all_finite,
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic CPU smoke for GAN-WaveNeXt 2")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ratio-threshold", type=float, default=0.9)
    args = parser.parse_args(argv)

    r = run_synthetic_smoke(steps=args.steps, verbose=True)
    print(
        f"[smoke] init={r['init_mrstft']:.4f} best={r['best_mrstft']:.4f} "
        f"final={r['final_mrstft']:.4f} ratio(best/init)={r['ratio']:.4f} "
        f"max_abs={r['max_abs']:.4f} finite={r['all_finite']}"
    )
    # 飽和凍結なら best≈init で ratio≈1.0、学習が進めば <0.9。max_abs は T=2 の構造上界 2.0。
    ok = r["all_finite"] and r["ratio"] < args.ratio_threshold and r["max_abs"] <= 2.0 + 1e-3
    print("[smoke] PASS" if ok else "[smoke] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
