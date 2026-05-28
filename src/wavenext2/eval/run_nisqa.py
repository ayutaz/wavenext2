"""run_nisqa.py — NISQA (自動 MOS 推定) 連携 (T-M4.2).

NISQA は旧 torch/torchaudio 想定のため、scripts/setup_eval_models.py が別 venv (Python 3.9) を
構築し、本関数は **subprocess 隔離経路** で `eval_models/.venv-nisqa` の python を起動して JSON で
score を受け取る。venv / weights 未 setup 時は `EvalModelNotFoundError` を raise (silent NaN なし)。

本体 import path に NISQA 本体を import しないため、すべて遅延・subprocess 経由。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wavenext2.eval._mos_common import EVAL_ROOT, EvalModelNotFoundError, to_wav_list
from wavenext2.eval.runner import register_metric_backend

__all__ = ["nisqa_scores", "score_nisqa"]

SR_DEFAULT = 24000
_NISQA_VENV = EVAL_ROOT / ".venv-nisqa"
_NISQA_DIR = EVAL_ROOT / "NISQA"


def _nisqa_python() -> Path:
    """NISQA 専用 venv の python 実行体パス (OS 差吸収)。"""
    if sys.platform == "win32":
        return _NISQA_VENV / "Scripts" / "python.exe"
    return _NISQA_VENV / "bin" / "python"


def nisqa_scores(
    wavs: Any,
    *,
    sr: int = SR_DEFAULT,
    device: str = "cpu",
) -> list[float]:
    """各 utterance の NISQA MOS スコア (list[float]) を返す (no-reference).

    subprocess 隔離 venv を起動し、wav を tmp に書いて NISQA を実行、JSON で結果を受領。
    Raises:
        EvalModelNotFoundError: 専用 venv / NISQA repo が未 setup のとき。
    """
    py = _nisqa_python()
    if not py.exists() or not _NISQA_DIR.exists():
        raise EvalModelNotFoundError(
            f"NISQA が未 setup です ({py} / {_NISQA_DIR} 不在)。"
            "scripts/setup_eval_models.py を実行してください。"
        )
    import tempfile

    import soundfile as sf

    wav_list = to_wav_list(wavs, sr)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, y in enumerate(wav_list):
            p = Path(td) / f"{i:05d}.wav"
            sf.write(str(p), y, sr)
            paths.append(str(p))
        # 専用 venv 側の runner (setup_eval_models.py が配置) を subprocess 起動。
        cmd = [str(py), str(_NISQA_DIR / "run_nisqa_json.py"), "--sr", str(sr), *paths]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            raise EvalModelNotFoundError(f"NISQA subprocess 失敗: {proc.stderr[-500:]}")
        return [float(v) for v in json.loads(proc.stdout)["scores"]]


def score_nisqa(wavs: Any, *, sr: int = SR_DEFAULT, device: str = "cpu") -> dict[str, float]:
    """NISQA の mean / std / n を返す (T-M4.1 prefix 規約)."""
    scores = nisqa_scores(wavs, sr=sr, device=device)
    arr = np.array(scores, dtype=np.float64)
    return {
        "nisqa_mean": float(arr.mean()) if len(arr) else float("nan"),
        "nisqa_std": float(arr.std()) if len(arr) else float("nan"),
        "n": float(len(arr)),
    }


def _nisqa_backend(pairs: list[tuple[Any, Any]], sr: int) -> tuple[dict, list[dict]]:
    """evaluate() facade 用 backend。pair=(gt, synth) のうち synth を no-reference 採点."""
    synth = [p[1] for p in pairs]
    scores = nisqa_scores(synth, sr=sr)
    arr = np.array(scores, dtype=np.float64)
    summary = {
        "nisqa_mean": float(arr.mean()) if len(arr) else float("nan"),
        "nisqa_std": float(arr.std()) if len(arr) else float("nan"),
    }
    return summary, [{"nisqa": s} for s in scores]


register_metric_backend("nisqa", _nisqa_backend)
