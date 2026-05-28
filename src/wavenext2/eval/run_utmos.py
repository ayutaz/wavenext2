"""run_utmos.py — UTMOS (自動 MOS 推定) 連携 (T-M4.2).

default backend は **speechmos** (`torch.hub` の `utmos22_strong` を lazy import で in-process 実行、
setup 不要だが初回 model download にネットワーク要)。speechmos 未インストール / model 取得失敗時は
`EvalModelNotFoundError` を raise (silent NaN を出さない)。subprocess 隔離経路 (古い fairseq UTMOS22)
は scripts/setup_eval_models.py で別 venv を構築する opt-in。

本体 import path に fairseq/torch.hub を引き込まないため、すべて関数内 lazy import。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wavenext2.eval._mos_common import EvalModelNotFoundError, to_wav_list
from wavenext2.eval.runner import register_metric_backend

__all__ = ["score_utmos", "utmos_scores"]

SR_DEFAULT = 24000


def utmos_scores(
    wavs: Any,
    *,
    sr: int = SR_DEFAULT,
    device: str = "cpu",
    backend: str = "speechmos",
) -> list[float]:
    """各 utterance の UTMOS スコア (list[float]) を返す (no-reference, GT 不要).

    Raises:
        EvalModelNotFoundError: backend パッケージ / model が利用不可のとき。
    """
    wav_list = to_wav_list(wavs, sr)
    if backend != "speechmos":
        raise ValueError(f"unsupported UTMOS backend: {backend} (speechmos のみ対応)")
    try:
        from speechmos import utmos22_strong  # lazy: 本体 import path を汚さない
    except ImportError:
        raise EvalModelNotFoundError(
            "speechmos が見つかりません。`uv add speechmos` でインストールするか、"
            "scripts/setup_eval_models.py で subprocess 隔離経路を構築してください。"
        ) from None
    scores: list[float] = []
    for y in wav_list:
        try:
            r = utmos22_strong.run(y, sr)  # {"utmos22_strong": float} 想定
        except Exception as e:  # noqa: BLE001
            raise EvalModelNotFoundError(f"UTMOS model 実行失敗 (checkpoint 未取得?): {e}") from e
        scores.append(float(r["utmos22_strong"] if isinstance(r, dict) else r))
    return scores


def score_utmos(
    wavs: Any,
    *,
    sr: int = SR_DEFAULT,
    device: str = "cpu",
    backend: str = "speechmos",
) -> dict[str, float]:
    """UTMOS の mean / std / n を返す (T-M4.1 prefix 規約)."""
    scores = utmos_scores(wavs, sr=sr, device=device, backend=backend)
    arr = np.array(scores, dtype=np.float64)
    return {
        "utmos_mean": float(arr.mean()) if len(arr) else float("nan"),
        "utmos_std": float(arr.std()) if len(arr) else float("nan"),
        "n": float(len(arr)),
    }


def _utmos_backend(pairs: list[tuple[Any, Any]], sr: int) -> tuple[dict, list[dict]]:
    """evaluate() facade 用 backend。pair=(gt, synth) のうち synth を no-reference 採点."""
    synth = [p[1] for p in pairs]
    scores = utmos_scores(synth, sr=sr)
    arr = np.array(scores, dtype=np.float64)
    summary = {
        "utmos_mean": float(arr.mean()) if len(arr) else float("nan"),
        "utmos_std": float(arr.std()) if len(arr) else float("nan"),
    }
    return summary, [{"utmos": s} for s in scores]


# facade への backend 登録 (import 副作用、heavy dep は load しない)。
register_metric_backend("utmos", _utmos_backend)
