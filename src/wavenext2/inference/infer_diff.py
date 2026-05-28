"""infer_diff.py — Diff-WaveNeXt 2 の DDIM/DDPM 一般形 4-step reverse sampling.

論文 §3.3 / docs/training.md §4.2 / docs/architecture.md §5 を実装する。

- x_0 予測経由の DDIM/DDPM 一般形 (β-free):
    x0_hat = (x_t - √(1-ᾱ_t) · ε_θ(x_t, c)) / √(ᾱ_t)
    x_next = √(ᾱ_next) · x0_hat + √(1-ᾱ_next-σ²) · ε_θ + σ · z   (z ~ N(0, I))
    σ²     = η² · (1-ᾱ_next) / (1-ᾱ_t) · (1 - ᾱ_t/ᾱ_next)
  最終 step は ᾱ_next = 1 で x = x0_hat。
- β は一切計算しない。論文 schedule は denoising 順で ᾱ が増加列のため
  β_t = 1 - ᾱ_t/ᾱ_{t-1} が負値 (β[1] = -279) になり発散する。x_0 予測経由なら
  denoising 方向で ᾱ_t < ᾱ_next ⇒ σ² ≥ 0 で構造的に安全 (docs/training.md §4.2 CRITICAL)。
- noise schedule: ᾱ = [1e-4, 2.8e-2, 5.6e-1, 9.1e-1] (`DiffWaveNext2.NOISE_SCHEDULE_ABAR`)
- eta: 1.0 → DDPM (確率的) / 0.0 → DDIM (決定論的)。§8.1 で v1 採用 (連続値 η∈[0,1] 可)。
- 1-to-1 dispatch: reverse step t (denoising 順 1..4) で sub-model k=t を直接呼ぶ
  (point-specialized partition、band 判定不要、docs/architecture.md §5)。
- post-filter は本関数では適用しない (T-M3.4 で別途実装、本関数は plain reverse sample のみ)。

公開 API:
- reverse_sample(model, mel, *, seed=None, eta=1.0) -> Tensor:
    mel から波形 (B, T_mel * hop_length) を合成する唯一のエントリポイント。
- eval_mode(model): model.training を一時 False にして復元する context manager。
- import 副作用で `DiffWaveNext2.synthesize = reverse_sample` を注入 (T-M4.3 が GAN/Diff 横断で
  `model.synthesize(mel)` を呼べるようにする、§8.2 / §9.1)。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from wavenext2.inference.post_filter import apply_post_filter
from wavenext2.models.diff_wavenext2 import DiffWaveNext2

__all__ = ["eval_mode", "reverse_sample"]


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[nn.Module]:
    """`model.training` を一時的に False にし、`__exit__` で元の状態へ復元する context manager.

    推論 (reverse_sample / RTF 測定 / smoke) で BatchNorm・Dropout を eval 状態にしつつ、
    呼び出し側の train/eval 状態を破壊しない。`reverse_sample` 内部でも使用する (§8.1 h)。
    """
    prev = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(prev)


def _compute_ddpm_coefficients(abar: torch.Tensor) -> dict[str, torch.Tensor]:
    """ᾱ_t (cumulative noise level) から x_0 予測経由式に必要な値を返す pure function.

    **β を一切計算しない** (docs/training.md §4.2)。論文 schedule は denoising 順で ᾱ が増加列の
    ため β_t = 1 - ᾱ_t/ᾱ_{t-1} が負値になり発散する。x_0 予測経由の射影式は β を必要とせず、
    各 step の ᾱ_t / ᾱ_next と √ 値だけで成立する。

    Args:
        abar: shape (K,)、要素は denoising 順 t=1..K で `ᾱ_1 < ᾱ_2 < ... < ᾱ_K`
              (t=1 が最高ノイズ、t=K が最低ノイズ)。論文の 4-step では K=4。

    Returns:
        dict (**beta / alpha は含まない**):
          - "abar":                (K,) ᾱ_t (そのまま保持)
          - "abar_next":           (K,) ᾱ_next = denoising 順で 1 つクリーン側
                                        (t<K は ᾱ_{t+1}、t=K は 1.0)
          - "sqrt_abar":           (K,) √ᾱ_t (x0_hat の分母)
          - "sqrt_one_minus_abar": (K,) √(1-ᾱ_t) (= conditioning c_t、x0_hat の係数)

    Notes:
        - σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next) は eta 依存のため reverse_sample 側で都度計算。
        - denoising 方向で ᾱ_t < ᾱ_next ⇒ 1 - ᾱ_t/ᾱ_next ≥ 0 ⇒ σ² ≥ 0 (負値問題なし)。
        - device/dtype は abar から継承、in-place 演算なし (pure)。
    """
    k = abar.shape[0]
    # denoising 順で ᾱ が狭義単調増加でないと σ²≥0 が崩れる (docs/training.md §4.2)。
    # 将来 schedule 差し替え時に silent な clamp 吸収でバグが隠れるのを防ぐ明示 assert。
    if k > 1 and not bool(torch.all(abar[:-1] < abar[1:])):
        raise ValueError(f"abar must be strictly increasing (denoising order), got {abar.tolist()}")
    sqrt_abar = torch.sqrt(abar)
    sqrt_one_minus_abar = torch.sqrt(1.0 - abar)

    # ᾱ_next: t<K は次のクリーン側 ᾱ_{t+1}、最終 step t=K は完全クリーン 1.0。
    abar_next = torch.empty_like(abar)
    abar_next[: k - 1] = abar[1:]
    abar_next[k - 1] = 1.0
    return {
        "abar": abar,
        "abar_next": abar_next,
        "sqrt_abar": sqrt_abar,
        "sqrt_one_minus_abar": sqrt_one_minus_abar,
    }


@torch.no_grad()
def reverse_sample(
    model: DiffWaveNext2,
    mel: torch.Tensor,
    *,
    seed: int | torch.Generator | None = None,
    eta: float = 1.0,
    post_filter: np.ndarray | None = None,
) -> torch.Tensor:
    """DDIM/DDPM 一般形 4-step reverse sampling (β-free) で mel から波形を合成する.

    Args:
        model: DiffWaveNext2 instance (T-M3.1)。
               `NOISE_SCHEDULE_ABAR` (Tensor (4,))、`sub_models[k-1](mel, x_t, c_t) -> ε_pred`、
               `hop_length` (= 256) を期待。
        mel:   (B, 128, T_mel) log-mel-spectrogram (Diff 設定 hop=256, n_fft=1024)。
        seed:  None → stochastic (`torch.randn` 直接呼び出し)、
               int → `torch.Generator(device).manual_seed(seed)` で deterministic、
               torch.Generator → 外部注入 (global RNG 非汚染、T-M3.4 / T-M4.3 の反復測定用)。
        eta:   stochasticity フラグ。1.0 → DDPM (確率的、σ·z を加算) / 0.0 → DDIM (決定論的)。
               連続値 η∈[0,1] も可。
        post_filter: (512,) linear-phase FIR (T-M3.4)。与えると各 batch 要素に
               `apply_post_filter` を適用して再 clamp。None で post-filter 無効 (plain reverse sample)。

    Returns:
        (B, T_audio) 合成波形 ∈ [-1, 1]、`T_audio = T_mel * model.hop_length`。

    Notes:
        - **β を一切計算しない** (§6.1 CRITICAL): denoising 方向で ᾱ_t < ᾱ_next ⇒ σ² ≥ 0。
        - `@torch.no_grad()` で autograd graph を構築しない。
        - `eval_mode(model)` で model.training を保護 (内部で eval に切替・復元、§8.1 h)。
        - post-filter は `post_filter` 引数を与えた場合のみ適用 (T-M3.4、apply/no-apply switch)。
    """
    if mel.dim() != 3:
        raise ValueError(f"mel must be (B, 128, T_mel), got {tuple(mel.shape)}")

    device, dtype = mel.device, mel.dtype
    b, _, t_mel = mel.shape
    t_audio = t_mel * model.hop_length

    # 係数は **常に fp32** で計算する (ML review 重要-1): bf16 では ᾱ_1=1-1e-4 が 1.0 に丸まり
    # √(1-ᾱ)=0 → conditioning c_t=0 / x0_hat=x/√ᾱ の情報破壊が silent に起きる。x は mel.dtype の
    # まま回し、fp32 スカラ係数は使用箇所で `.to(dtype)` する (x の dtype を保つ)。
    abar = model.NOISE_SCHEDULE_ABAR.to(device=device, dtype=torch.float32)
    coef = _compute_ddpm_coefficients(abar)
    k_steps = abar.shape[0]  # = 4

    # seed の 3 形式 (None / int / Generator) を local Generator に正規化 (global RNG 非汚染)。
    if isinstance(seed, torch.Generator):
        gen: torch.Generator | None = seed
    elif seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
    else:
        gen = None

    def _randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=gen, device=device, dtype=dtype)

    with eval_mode(model):
        x = _randn((b, t_audio))  # 初期 Gaussian noise (denoising 順 t=1 入力)

        for t in range(1, k_steps + 1):  # t = 1..K (1-indexed denoising 順)
            idx = t - 1  # 0-indexed tensor access
            abar_t = coef["abar"][idx]  # fp32 スカラ
            abar_next = coef["abar_next"][idx]  # t<K: ᾱ_{t+1}、t=K: 1.0
            sqrt_abar_t = coef["sqrt_abar"][idx]
            sqrt_one_minus_abar_t = coef["sqrt_one_minus_abar"][idx]
            c_t = sqrt_one_minus_abar_t.expand(b).to(dtype)  # conditioning は x の dtype へ

            # === point-specialized 1-to-1 dispatch (k=t、band 判定不要、architecture.md §5) ===
            eps_pred = model.sub_models[idx](mel, x, c_t)

            # 1. x_0 を推定 (β 不使用)。fp32 スカラ係数を x.dtype に cast して x の dtype を保つ。
            x0_hat = (
                (x - sqrt_one_minus_abar_t.to(dtype) * eps_pred) / sqrt_abar_t.to(dtype)
            ).clamp(-1.0, 1.0)

            if t < k_steps:
                # 2. 次のクリーン側 ᾱ_next へ射影 (DDIM/DDPM 一般形、β-free)
                #    σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next) ≥ 0 (denoising 方向、fp32 計算)
                sigma2 = eta**2 * (1.0 - abar_next) / (1.0 - abar_t) * (1.0 - abar_t / abar_next)
                coef_eps = torch.sqrt(torch.clamp(1.0 - abar_next - sigma2, min=0.0))
                x = torch.sqrt(abar_next).to(dtype) * x0_hat + coef_eps.to(dtype) * eps_pred
                if eta > 0:
                    x = x + torch.sqrt(sigma2).to(dtype) * _randn(tuple(x.shape))
            else:
                # 3. 最終 step (t=K): ᾱ_next = 1 で x_0 推定値がそのまま出力
                x = x0_hat

    x = x.clamp(-1.0, 1.0)
    if post_filter is not None:
        # 各 batch 要素に linear-phase FIR を畳み込み (T-M3.4)、再 clamp。
        rows = [apply_post_filter(x[i], post_filter) for i in range(x.shape[0])]
        x = torch.stack(rows, dim=0).clamp(-1.0, 1.0)
    return x


# import 副作用: DiffWaveNext2.synthesize を reverse_sample alias として注入 (§8.2 / §9.1)。
# reverse_sample の第 1 引数が model なので、class attribute 化で `model.synthesize(mel)` →
# `reverse_sample(model, mel)` の method binding が成立する。T-M4.3 (RTF) が GAN/Diff 横断で
# `model.synthesize(mel)` を呼べるようにする。
DiffWaveNext2.synthesize = reverse_sample  # type: ignore[attr-defined]
