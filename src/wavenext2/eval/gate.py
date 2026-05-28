"""gate.py — M5 統合スモークの divergence gate (T-M5.1 / T-M5.2).

M5 review §8.1 で「品質 gate でなく divergence gate」に昇格。1 epoch (本番 2M step の ~1.6%) では
品質 (MCD/UTMOS/MR-STFT 絶対値) は判断できないため、gate は **発散の不在** のみを機械判定する:

GAN (T-M5.1):
- loss_G が単調減少傾向 (1 epoch スパンで右肩下がり)
- loss_adv が学習中域 (0.5〜2.0) に留まる
- loss_D が 0.01 を割らない (D 強すぎ検出)
- 生成波形が finite かつ [-1,1] かつ 非無音・非全 0
- NaN/Inf が出現しない

Diff (T-M5.2):
- loss (MSE) が単調減少傾向
- 生成波形 (reverse sample) が finite・[-1,1]・非無音
- NaN/Inf なし

聴感のみ人間に委ね、自動項目はここで pass/fail を返す (再現可能・自動化可能)。最終的な M6 起動の
GO/NO-GO は user 承認 (課金前、§8.2)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["GateResult", "OutputStats", "diff_divergence_gate", "gan_divergence_gate"]

# divergence gate の閾値域 (M5 review §8.1)。
LOSS_ADV_RANGE = (0.5, 2.0)  # loss_adv が学習中に留まるべき域
LOSS_D_FLOOR = 0.01  # loss_D がこれを割ると D 強すぎ
SILENCE_FLOOR = 1e-4  # |y|.max がこれ以下なら無音/全0 とみなす


@dataclass
class OutputStats:
    """生成波形の sanity 統計 (1 つ以上の sample から集約)."""

    max_abs: float  # |y|.max() の最大
    std: float  # y.std() の最小 (非無音判定)
    all_finite: bool  # NaN/Inf を含まないか


@dataclass
class GateResult:
    """divergence gate の機械判定結果 (聴感は別途人間)."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _is_decreasing(history: list[float], *, tol: float = 0.0) -> bool:
    """前半平均 > 後半平均 で「右肩下がり傾向」とみなす (step 揺らぎに頑健)."""
    n = len(history)
    if n < 4:
        return True  # 判定に足る履歴がなければ pass 寄り (短い smoke を弾かない)
    half = n // 2
    first = sum(history[:half]) / half
    second = sum(history[half:]) / (n - half)
    return second < first - tol


def _all_finite(history: list[float]) -> bool:
    return all(math.isfinite(v) for v in history)


def gan_divergence_gate(
    loss_g: list[float],
    loss_adv: list[float],
    loss_d: list[float],
    output: OutputStats,
) -> GateResult:
    """GAN 1 epoch の divergence gate (T-M5.1)."""
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    checks["loss_g_finite"] = _all_finite(loss_g)
    checks["loss_g_decreasing"] = _is_decreasing(loss_g)
    # loss_adv は終盤 (後半) が学習中域に留まるか。
    tail = loss_adv[len(loss_adv) // 2 :] if loss_adv else []
    checks["loss_adv_in_range"] = bool(tail) and all(
        LOSS_ADV_RANGE[0] <= v <= LOSS_ADV_RANGE[1] for v in tail
    )
    checks["loss_d_above_floor"] = bool(loss_d) and all(v >= LOSS_D_FLOOR for v in loss_d)
    checks["output_finite"] = output.all_finite
    checks["output_not_silent"] = output.max_abs > SILENCE_FLOOR and output.std > SILENCE_FLOOR
    checks["output_in_range"] = output.max_abs <= 1.0 + 1e-4

    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name} failed")
    return GateResult(passed=all(checks.values()), checks=checks, reasons=reasons)


def diff_divergence_gate(loss_mse: list[float], output: OutputStats) -> GateResult:
    """Diff 1 sub-model 1 epoch の divergence gate (T-M5.2)."""
    checks: dict[str, bool] = {
        "loss_finite": _all_finite(loss_mse),
        "loss_decreasing": _is_decreasing(loss_mse),
        "output_finite": output.all_finite,
        "output_not_silent": output.max_abs > SILENCE_FLOOR and output.std > SILENCE_FLOOR,
        "output_in_range": output.max_abs <= 1.0 + 1e-4,
    }
    reasons = [f"{n} failed" for n, ok in checks.items() if not ok]
    return GateResult(passed=all(checks.values()), checks=checks, reasons=reasons)
