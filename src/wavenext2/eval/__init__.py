"""wavenext2.eval public API."""

from wavenext2.eval._mos_common import EvalModelNotFoundError
from wavenext2.eval.compute_metrics import (
    compute_log_f0_rmse,
    compute_mcd,
    evaluate_dataset,
)
from wavenext2.eval.gate import (
    GateResult,
    OutputStats,
    diff_divergence_gate,
    gan_divergence_gate,
)
from wavenext2.eval.measure_rtf import measure_rtf
from wavenext2.eval.run_nisqa import score_nisqa  # backend 登録の副作用込み
from wavenext2.eval.run_utmos import score_utmos  # backend 登録の副作用込み
from wavenext2.eval.runner import EvalResult, evaluate, register_metric_backend

__all__ = [
    "EvalModelNotFoundError",
    "EvalResult",
    "GateResult",
    "OutputStats",
    "compute_log_f0_rmse",
    "compute_mcd",
    "diff_divergence_gate",
    "evaluate",
    "evaluate_dataset",
    "gan_divergence_gate",
    "measure_rtf",
    "register_metric_backend",
    "score_nisqa",
    "score_utmos",
]
