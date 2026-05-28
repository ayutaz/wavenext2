"""eval_diff_checkpoint.py — Diff sub-model checkpoint を evaluate() facade で評価する薄い driver (T-M5.2).

4 sub-model 揃った reverse sampling 評価は M6.2 後 (1 sub-model では reverse 不可)。本 driver は
4 つの sub_{1..4}.pt を 1 つの DiffWaveNext2 に load し、`evaluate()` (内部で reverse_sample) で
MCD/log F0 RMSE を取る。1 sub-model のみの段階では smoke (T-M3.5) の loss curve で代替する。

    uv run python scripts/eval_diff_checkpoint.py --config configs/diff_wavenext2_1epoch.yaml \
        --ckpt-dir checkpoints/diff_1epoch --out eval_results/diff_1epoch.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import torch

    from wavenext2.data.dataset import LibriTTSRDataset
    from wavenext2.eval.runner import evaluate
    from wavenext2.models.diff_wavenext2 import DiffWaveNext2
    from wavenext2.train.train_diff import build_sub_model_cfg
    from wavenext2.utils.config import load_config

    parser = argparse.ArgumentParser(description="Evaluate Diff sub-model checkpoints via evaluate()")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt-dir", required=True, help="sub_{1..4}.pt を含む dir")
    parser.add_argument("--out", default="eval_results/diff_1epoch.json")
    parser.add_argument("--metrics", nargs="+", default=["mcd", "log_f0_rmse"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    flat = build_sub_model_cfg(cfg["model"])
    model = DiffWaveNext2.from_config({"sub_model_cfg": flat})  # 全 4 sub-model
    ckpt_dir = Path(args.ckpt_dir)
    for k in range(1, model.K + 1):
        p = ckpt_dir / f"sub_{k}.pt"
        if p.exists():
            state = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
            model.load_state_dict(state, strict=False)  # 当該 sub-model のキーのみ
    model.eval()
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
    result = evaluate(model, val_ds, metrics=tuple(args.metrics), save_to=args.out)
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
