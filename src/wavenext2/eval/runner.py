"""runner.py — M4 全体の統一 eval facade (T-M4.1).

M4.1 (MCD / log F0 RMSE / MR-STFT), M4.2 (UTMOS / NISQA), M4.3 (RTF) を 1 entry point
`evaluate(model, dataset, metrics, post_filter) -> EvalResult` で束ね、M5/M6 caller が
「合成 → ペア化 → 各指標」のグルーコードを毎回書くのを排除する。

- model 種別を検出して合成: DiffWaveNext2 → reverse_sample(post_filter=...)、
  GANWaveNext2 → forward を GT 長に crop。
- summary は prefix 規約 `{metric}_mean` / `{metric}_std` / `n` / `n_skipped`。
- UTMOS / NISQA / RTF は T-M4.2 / T-M4.3 が `register_metric_backend` で backend を登録する
  (本 facade は dispatch 枠のみ提供)。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wavenext2.eval.compute_metrics import evaluate_dataset

__all__ = ["EvalResult", "evaluate", "register_metric_backend"]

# T-M4.2 / T-M4.3 が backend を登録する registry。
# backend(pairs: list[(gt_wave, synth_wave)], sr: int) -> (summary: dict, per_utt: list[dict])
_METRIC_BACKENDS: dict[str, Callable[[list[tuple[Any, Any]], int], tuple[dict, list[dict]]]] = {}

# 本 facade (M4.1) が直接処理する波形ペア指標。
_M41_METRICS = ("mcd", "log_f0_rmse")


def register_metric_backend(
    name: str, backend: Callable[[list[tuple[Any, Any]], int], tuple[dict, list[dict]]]
) -> None:
    """T-M4.2 / T-M4.3 が UTMOS/NISQA/RTF backend を登録する (facade の拡張点)。"""
    _METRIC_BACKENDS[name] = backend


@dataclass
class EvalResult:
    """統一 eval 戻り値。summary は集約値、per_utterance は生値."""

    summary: dict[str, float]
    per_utterance: list[dict]

    def to_json(self, path: str | Path) -> None:
        """eval_results/*.json へ per-utterance + summary を dump (再計算回避)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary, "per_utterance": self.per_utterance}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> EvalResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(summary=payload["summary"], per_utterance=payload["per_utterance"])


def _synthesize(
    model: Any, mel: torch.Tensor, audio_len: int, seed: int, post_filter
) -> np.ndarray:
    """model 種別を検出して 1 utterance を合成し mono numpy を返す."""
    from wavenext2.models.diff_wavenext2 import DiffWaveNext2

    if isinstance(model, DiffWaveNext2):
        from wavenext2.inference.infer_diff import reverse_sample

        y = reverse_sample(model, mel, seed=seed, post_filter=post_filter)
    else:  # GANWaveNext2 等: forward を GT 長に crop
        y = model(mel)[..., :audio_len]
    return y.squeeze(0).detach().cpu().numpy()


def evaluate(
    model: Any,
    dataset: Any,
    metrics: tuple[str, ...] = ("mcd", "log_f0_rmse"),
    post_filter: np.ndarray | None = None,
    *,
    sr: int = 24000,
    seed: int = 43,
    save_to: str | Path | None = None,
) -> EvalResult:
    """統一 eval entry point.

    dataset は `{"mel": (1,128,T_mel) or (128,T_mel), "audio": (T,) or (1,T)}` を yield する
    iterable。各 utterance を model で合成 → GT とペア化 → metrics を dispatch。
    """
    model.eval()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for item in dataset:
            mel = item["mel"]
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            gt = item["audio"]
            gt = gt.squeeze(0) if gt.dim() > 1 else gt
            gt_np = gt.detach().cpu().numpy()
            synth = _synthesize(model, mel, gt_np.shape[-1], seed, post_filter)
            pairs.append((gt_np, synth))

    # M4.1 指標 (波形ペア) を evaluate_dataset で。
    m41 = tuple(m for m in metrics if m in _M41_METRICS)
    summary: dict[str, float] = {"n": float(len(pairs)), "sr": float(sr)}
    per_utt: list[dict] = [{"index": i} for i in range(len(pairs))]
    if m41:
        s, pu = evaluate_dataset(pairs, sr=sr, metrics=m41)
        summary.update({k: v for k, v in s.items() if k not in ("n", "sr")})
        for row, extra in zip(per_utt, pu, strict=True):
            row.update({k: v for k, v in extra.items() if k != "index"})

    # UTMOS / NISQA / RTF 等は登録済 backend へ dispatch (T-M4.2 / T-M4.3)。
    for metric in metrics:
        if metric in _M41_METRICS:
            continue
        if metric not in _METRIC_BACKENDS:
            raise NotImplementedError(
                f"metric '{metric}' の backend が未登録です (T-M4.2 / T-M4.3 で "
                f"register_metric_backend('{metric}', ...) が実装されます)。"
            )
        s, pu = _METRIC_BACKENDS[metric](pairs, sr)
        summary.update(s)
        for row, extra in zip(per_utt, pu, strict=True):
            row.update(extra)

    result = EvalResult(summary=summary, per_utterance=per_utt)
    if save_to is not None:
        result.to_json(save_to)
    return result
