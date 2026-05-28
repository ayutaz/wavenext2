"""run_smoke.py — M5 統合スモークの共通 orchestrator (T-M5.1 / T-M5.2、DRY).

GAN (--mode gan) / Diff (--mode diff) の 1 epoch スモークで共通の
「loss/出力 metrics → divergence gate 判定 → GO/NO-GO レポート → JSON 永続化」を 1 箇所に集約する
(M5 review §8.1 採用昇格)。GAN/Diff 差分は `--mode` と参照する loss キーのみ。

**実 1 epoch 訓練の起動は GPU + 実 LibriTTS-R を要するユーザー操作** (T-M0.3 データ + GPU 確保、
M6 課金前の GO/NO-GO 判定者は user、ticket §8.2/§9.1)。本スクリプトは訓練を直接起動せず、
訓練 + evaluate が出力した metrics JSON を読んで **機械判定可能な divergence gate** を評価し、
user の GO/NO-GO 判断材料 (gate pass/fail + 理由) を提示する。

訓練起動 (ユーザーが実行):
    uv run python -m wavenext2.train.train_gan  --config configs/gan_wavenext2_1epoch.yaml
    uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2_1epoch.yaml --sub-model 4
gate 判定:
    uv run python scripts/run_smoke.py --mode gan --metrics-json <訓練が出力した metrics>.json

metrics JSON schema (訓練監視 + eval driver が生成):
  gan : {"loss_g": [...], "loss_adv": [...], "loss_d": [...],
         "output": {"max_abs": float, "std": float, "all_finite": bool}}
  diff: {"loss_mse": [...], "output": {"max_abs": float, "std": float, "all_finite": bool}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wavenext2.eval.gate import (
    GateResult,
    OutputStats,
    diff_divergence_gate,
    gan_divergence_gate,
)


def _output_stats(d: dict) -> OutputStats:
    o = d["output"]
    return OutputStats(max_abs=o["max_abs"], std=o["std"], all_finite=o["all_finite"])


def evaluate_gate(mode: str, metrics: dict) -> GateResult:
    """metrics dict から mode 別 divergence gate を評価する (純粋関数、テスト可能)."""
    if mode == "gan":
        return gan_divergence_gate(
            metrics["loss_g"], metrics["loss_adv"], metrics["loss_d"], _output_stats(metrics)
        )
    if mode == "diff":
        return diff_divergence_gate(metrics["loss_mse"], _output_stats(metrics))
    raise ValueError(f"unknown mode: {mode} (gan / diff のみ)")


def render_report(mode: str, result: GateResult) -> str:
    """GO/NO-GO 推奨レポートを文字列で生成 (user 提示用)."""
    verdict = "GO (M6 起動推奨)" if result.passed else "NO-GO (要調査、M6 を起動しない)"
    lines = [
        f"=== M5 divergence gate ({mode}) ===",
        f"判定: {'PASS' if result.passed else 'FAIL'}  →  推奨: {verdict}",
        "チェック項目:",
    ]
    for name, ok in result.checks.items():
        lines.append(f"  [{'x' if ok else ' '}] {name}")
    if result.reasons:
        lines.append("FAIL 理由: " + ", ".join(result.reasons))
    lines.append("※ 聴感 (sample audio 4 本) は別途人間が確認。最終 GO/NO-GO は user 承認 (課金前)。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M5 統合スモーク divergence gate orchestrator")
    parser.add_argument("--mode", choices=["gan", "diff"], required=True)
    parser.add_argument("--metrics-json", type=Path, required=True, help="訓練/eval が出力した metrics")
    parser.add_argument("--out", type=Path, default=None, help="gate 結果 JSON 出力先")
    args = parser.parse_args(argv)

    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    result = evaluate_gate(args.mode, metrics)
    print(render_report(args.mode, result))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"mode": args.mode, "passed": result.passed, "checks": result.checks,
                 "reasons": result.reasons},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    # gate FAIL は非ゼロ終了 (CI / orchestration から検知可能)。
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
