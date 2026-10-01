"""eval_gan_checkpoint.py — GAN checkpoint を evaluate() facade で評価する薄い driver (T-M5.1).

新規ロジックは書かず、T-M4.1 `evaluate()` を 1 entry point で呼ぶだけ (DRY)。実 LibriTTS-R val +
学習済 checkpoint を要するユーザー操作。Windows spawn 対策で `if __name__` ガード下から呼ぶ。

    uv run python scripts/eval_gan_checkpoint.py --config configs/gan_wavenext2_1epoch.yaml \
        --ckpt checkpoints/gan_1epoch/best.pt --out eval_results/gan_1epoch.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import torch

    from wavenext2.data.dataset import LibriTTSRDataset
    from wavenext2.eval.runner import evaluate
    from wavenext2.models.gan_wavenext2 import GANWaveNext2
    from wavenext2.utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate a GAN checkpoint via evaluate() facade")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="eval_results/gan_1epoch.json")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mcd", "log_f0_rmse"],
        help="評価指標 (utmos は要 speechmos)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    g = GANWaveNext2.from_config(cfg["model"])
    g.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["G_state_dict"])
    g.eval()
    data = cfg["data"]
    hop = cfg["model"]["sub_model"]["hop"]
    val_ds = LibriTTSRDataset(
        filelist_path=data["val_filelist"],
        root_dir=data["root_dir"],
        segment_length=cfg["train"]["segment_length"],
        hop_length=hop,
        mel_cfg={**data["mel"], "hop_length": hop},
        mode="val",
    )
    result = evaluate(g, val_ds, metrics=tuple(args.metrics), post_filter=None, save_to=args.out)
    print(result.summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
