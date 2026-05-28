---
id: T-M3.3
title: Reverse sampler (DDPM 4-step + 1-to-1 dispatch + post-filter プレースホルダ)
milestone: M3
phase: M3
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-28
depends_on: [T-M3.1]
blocks: [T-M3.4, T-M3.5, T-M4.3]
related_docs:
  - docs/milestones.md#m33-reverse-sampler-srcwavenext2inferenceinfer_diffpy
  - docs/training.md
  - docs/open-questions.md
---

# T-M3.3: Reverse sampler (DDPM 4-step + 1-to-1 dispatch + post-filter プレースホルダ)

> **マイルストーン**: [M3](../milestones.md#m3-diff-wavenext-2-作業量-large5-サブタスク) / **サブタスク**: [M3.3](../milestones.md#m33-reverse-sampler-srcwavenext2inferenceinfer_diffpy)
> **依存**: [T-M3.1](T-M3.1-diff-model.md) / **後続**: [T-M3.4](T-M3.4-post-filter.md), [T-M3.5](T-M3.5-diff-smoke.md), [T-M4.3](T-M4.3-rtf.md)

## 1. タスク目的とゴール

### 目的
論文 Section 3.3 / Figure 3 / `docs/training.md` §4.2 に示される **DDIM/DDPM 一般形 4-step reverse sampling (x_0 予測経由、β-free)** を 1 関数 (`reverse_sample`) に閉じ込め、`DiffWaveNext2` (T-M3.1) を入力として **mel-spectrogram から波形 `(B, T_audio)` を合成する唯一のエントリポイント** とする。

- **point-specialized 1-to-1 dispatch**: 各 reverse step `t ∈ {1, 2, 3, 4}` で sub-model k=t を直接呼ぶ (band 判定不要、`docs/architecture.md` §5)
- **β-free な x_0 予測経由式** (`docs/training.md` §4.2): 各 step で `x0_hat = (x - √(1-ᾱ_t)·ε_θ) / √(ᾱ_t)` を推定し、次のクリーン側 `ᾱ_next` (最終は 1.0) へ射影 `x = √(ᾱ_next)·x0_hat + √(1-ᾱ_next-σ²)·ε_θ + σ·z`、`σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`
- **β を一切計算しない**: 論文 schedule は denoising 順で ᾱ が増加列のため `β_t = 1 - ᾱ_t/ᾱ_{t-1}` が負値 (`β[1]=-279`) になり発散する。x_0 予測経由なら denoising 方向で `ᾱ_t < ᾱ_next ⇒ σ² ≥ 0` で構造的に安全 (§6.1 参照)
- **noise schedule**: `ᾱ = [1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` を `DiffWaveNext2.NOISE_SCHEDULE_ABAR` から取得 (β 逆算なし)
- **eta フラグ**: `eta=1.0` で DDPM (確率的)、`eta=0.0` で DDIM (決定論的)。§8.1 で v1 採用
- **post-filter は本チケットでは適用しない**: T-M3.4 で別途実装、本チケットは plain reverse sample のみ

これにより:
- T-M3.4 (post-filter) は `reverse_sample(model, mel)` の戻り値に FIR convolution を後段で適用するだけ
- T-M3.5 (smoke) は 1 utterance の mel から `reverse_sample` で波形を再構成して GT と比較
- T-M4.3 (RTF) は `reverse_sample` の wall time を直接測定

### ゴール
- [ ] `src/wavenext2/inference/infer_diff.py` に `reverse_sample(model, mel, *, seed=None, eta=1.0) -> Tensor` 関数が実装され、`from wavenext2.inference.infer_diff import reverse_sample` で import 可能
- [ ] `DiffWaveNext2.NOISE_SCHEDULE_ABAR` (shape `(4,)`、device-aware) から **β を計算せず** 各 step の `(abar_t, abar_next, sqrt_abar_t, sqrt_one_minus_abar_t)` と σ² 計算用の値を返す純粋関数 `_compute_ddpm_coefficients(abar) -> dict[str, Tensor]` を helper として切り出し、unit test 可能にする
- [ ] 4-step ループで sub-model k (k=1..4, 1-indexed) を 1 回ずつ呼び、x_0 予測経由の射影式 (β-free) を順に適用、最終 `x ∈ (B, T_audio)` を `clamp(-1, 1)` して返す
- [ ] `eta` で stochasticity 制御 (`eta=1.0` DDPM / `eta=0.0` DDIM)、`seed=None` で stochastic (`torch.randn` 直接呼び出し)、`seed=int` で `torch.Generator(device=mel.device).manual_seed(seed)` 経由で deterministic
- [ ] `T_audio = T_mel * model.hop_length`(= 256) を自動算出、`model.eval()` モードを内部で保証 (revert は呼び出し側責務 OR with-context で実装)
- [ ] forward は `torch.no_grad()` で実行、autograd graph を構築しない
- [ ] `tests/test_reverse_sampler.py` が `uv run pytest tests/test_reverse_sampler.py` で全 pass
- [ ] `docs/milestones.md` §M3.3 Acceptance criteria 2 項目クリア (4 step で [-1,1] 波形 / 同 seed deterministic)

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/inference/infer_diff.py` (本実装)
  - `tests/test_reverse_sampler.py` (5.1 / 5.3)
- 編集:
  - `src/wavenext2/inference/__init__.py` (`__all__` に `reverse_sample` を追加、`from .infer_diff import reverse_sample`)
  - `docs/milestones.md` §M3.3 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M3.3 ステータス更新

### 2.2 主要構造

#### import 形式
```python
from wavenext2.inference.infer_diff import reverse_sample
# 任意: helper を直接 test 用に exposed
from wavenext2.inference.infer_diff import _compute_ddpm_coefficients
```

#### 関数シグネチャ

```python
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
- eta: 1.0 → DDPM (確率的) / 0.0 → DDIM (決定論的)。§8.1 で v1 採用 (stochastic フラグ)
- 1-to-1 dispatch: reverse step t (denoising 順 1..4) で sub-model k=t を直接呼ぶ
  (point-specialized partition、band 判定不要、docs/architecture.md §5)
- post-filter は本関数では適用しない (T-M3.4 で別途実装、本関数は plain reverse sample のみ)

公開 API:
- reverse_sample(model, mel, *, seed=None, eta=1.0) -> Tensor:
    mel から波形 (B, T_mel * hop_length) を合成する唯一のエントリポイント。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wavenext2.models.diff_wavenext2 import DiffWaveNext2


def _compute_ddpm_coefficients(
    abar: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """ᾱ_t (cumulative noise level) から x_0 予測経由式に必要な値を返す pure function.

    **β を一切計算しない** (docs/training.md §4.2)。論文 schedule は denoising 順で
    ᾱ が増加列のため β_t = 1 - ᾱ_t/ᾱ_{t-1} が負値になり発散する。x_0 予測経由の
    射影式は β を必要とせず、各 step の ᾱ_t / ᾱ_next と √ 値だけで成立する。

    Args:
        abar: shape (K,)、要素は denoising 順 t=1..K で `ᾱ_1 < ᾱ_2 < ... < ᾱ_K`
              (t=1 が最高ノイズ、t=K が最低ノイズ)。論文の 4-step では K=4。

    Returns:
        dict with keys (**beta / alpha は含まない**):
          - "abar":                shape (K,),   ᾱ_t (そのまま保持、射影で √ᾱ_next を引くため)
          - "abar_next":           shape (K,),   ᾱ_{next} = denoising 順で 1 つクリーン側
                                                  (t<K は ᾱ_{t+1}、t=K は 1.0)
          - "sqrt_abar":           shape (K,),   √ᾱ_t (x0_hat の分母)
          - "sqrt_one_minus_abar": shape (K,),   √(1-ᾱ_t) (= conditioning c_t、x0_hat の係数)

    Notes:
        - σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next) は eta に依存するため
          reverse_sample 側で都度計算する (本 helper は eta 非依存の値のみ返す)
        - denoising 方向で ᾱ_t < ᾱ_next ⇒ 1 - ᾱ_t/ᾱ_next ≥ 0 ⇒ σ² ≥ 0 (負値問題なし)
        - device/dtype は abar から継承、in-place 演算なし (pure)
        - K=1 (degenerate) は本実装範囲外、論文の 4-step のみ想定
    """
    K = abar.shape[0]
    sqrt_abar = torch.sqrt(abar)
    sqrt_one_minus_abar = torch.sqrt(1.0 - abar)

    # ᾱ_next: t<K は次のクリーン側 ᾱ_{t+1}、最終 step t=K は完全クリーン 1.0
    abar_next = torch.empty_like(abar)
    abar_next[: K - 1] = abar[1:]                 # ᾱ_{t+1} (t=1..K-1)
    abar_next[K - 1] = 1.0                         # 最終 step は ᾱ_next = 1 (完全クリーン)
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
    seed: int | None = None,
    eta: float = 1.0,
) -> torch.Tensor:
    """DDIM/DDPM 一般形 4-step reverse sampling (β-free) で mel から波形を合成する.

    Args:
        model: DiffWaveNext2 instance (T-M3.1)。
               以下の属性を期待:
               - `NOISE_SCHEDULE_ABAR`: Tensor shape (4,) (class attribute or buffer)
               - `sub_models[k-1]`: (mel, x_t, c_t) -> ε_pred (k=1..4)
               - `hop_length`: int (= 256)
        mel:   (B, 128, T_mel) log-mel-spectrogram (Diff 設定 hop=256, n_fft=1024)
        seed:  None → stochastic (`torch.randn` 直接呼び出し)、
               int → `torch.Generator(device=mel.device).manual_seed(seed)` で deterministic
        eta:   stochasticity フラグ。1.0 → DDPM (確率的、σ·z を加算) /
               0.0 → DDIM (決定論的、σ=0)。§8.1 で v1 採用。

    Returns:
        (B, T_audio) 合成波形 ∈ [-1, 1]、`T_audio = T_mel * model.hop_length`

    Algorithm (docs/training.md §4.2 を完全準拠、β 不使用):
        abar = model.NOISE_SCHEDULE_ABAR              # (4,)、denoising 順 t=1..4 (ᾱ 増加列)
        coef = _compute_ddpm_coefficients(abar)       # ᾱ_t, ᾱ_next, √ᾱ_t, √(1-ᾱ_t) (β なし)
        x = torch.randn(B, T_audio)                    # 初期 Gaussian noise (t=1 入力)
        for t in 1..4 (denoising 順、1-indexed):
            c_t = coef["sqrt_one_minus_abar"][t-1]    # conditioning value √(1-ᾱ_t)
            k = t                                      # 1-to-1 dispatch (point-specialized)
            eps = model.sub_models[k-1](mel, x, c_t.expand(B))
            # 1. x_0 を推定 (β 不使用):
            x0_hat = (x - √(1-ᾱ_t) · eps) / √(ᾱ_t)
            x0_hat = x0_hat.clamp(-1, 1)               # 波形は [-1, 1] (任意)
            if t < 4:
                # 2. 次のクリーン側 ᾱ_next へ射影 (DDIM/DDPM 一般形):
                #    σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1 - ᾱ_t/ᾱ_next)  ≥ 0 (denoising 方向)
                sigma2   = eta**2 * (1-ᾱ_next)/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_next)
                coef_eps = √(clamp(1 - ᾱ_next - σ², min=0))
                x = √(ᾱ_next) · x0_hat + coef_eps · eps
                if eta > 0:
                    x = x + √(σ²) · z                  # z ~ N(0, I)、seed があれば deterministic
            else:
                # 3. 最終 step (t=4): ᾱ_next = 1 で x_0 推定値がそのまま出力
                x = x0_hat
        return x.clamp(-1.0, 1.0)

    Notes:
        - **β を一切計算しない** (§6.1 CRITICAL): denoising 方向で ᾱ_t < ᾱ_next ⇒ σ² ≥ 0
        - `@torch.no_grad()` で autograd graph を構築しない
        - `model.eval()` 切替は本関数内で保証しない (呼び出し側責務、§6.1 参照)
        - post-filter は本関数では適用しない (T-M3.4)
    """
    # device / dtype 整合
    device, dtype = mel.device, mel.dtype
    B, _, T_mel = mel.shape
    T_audio = T_mel * model.hop_length

    # x_0 予測経由式に必要な係数を取得 (β は計算しない、device 同期)
    abar = model.NOISE_SCHEDULE_ABAR.to(device=device, dtype=dtype)
    coef = _compute_ddpm_coefficients(abar)

    # Generator for deterministic sampling (seed が指定された場合)
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

        def randn_like_x(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shape, generator=gen, device=device, dtype=dtype)
    else:
        def randn_like_x(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shape, device=device, dtype=dtype)

    K = abar.shape[0]  # = 4
    x = randn_like_x((B, T_audio))  # 初期 Gaussian noise (denoising 順 t=1 入力)

    for t in range(1, K + 1):       # t = 1..4 (1-indexed)
        idx = t - 1                  # 0-indexed for tensor access
        abar_t = coef["abar"][idx]
        abar_next = coef["abar_next"][idx]            # t<K: ᾱ_{t+1}、t=K: 1.0
        sqrt_abar_t = coef["sqrt_abar"][idx]
        sqrt_one_minus_abar_t = coef["sqrt_one_minus_abar"][idx]
        c_t = sqrt_one_minus_abar_t.expand(B)
        # === point-specialized 1-to-1 dispatch ===
        # docs/architecture.md §5: reverse step t で sub-model k=t を呼ぶ (band 判定不要)
        eps_pred = model.sub_models[idx](mel, x, c_t)

        # 1. x_0 を推定 (β 不使用)
        x0_hat = (x - sqrt_one_minus_abar_t * eps_pred) / sqrt_abar_t
        x0_hat = x0_hat.clamp(-1.0, 1.0)              # 波形は [-1, 1] (任意)

        if t < K:
            # 2. 次のクリーン側 ᾱ_next へ射影 (DDIM/DDPM 一般形、β-free)
            #    σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1 - ᾱ_t/ᾱ_next)
            #    denoising 方向で ᾱ_t < ᾱ_next ⇒ 1 - ᾱ_t/ᾱ_next ≥ 0 ⇒ σ² ≥ 0 (負値問題なし)
            sigma2 = eta ** 2 * (1.0 - abar_next) / (1.0 - abar_t) * (1.0 - abar_t / abar_next)
            coef_eps = torch.sqrt(torch.clamp(1.0 - abar_next - sigma2, min=0.0))
            x = torch.sqrt(abar_next) * x0_hat + coef_eps * eps_pred
            if eta > 0:
                x = x + torch.sqrt(sigma2) * randn_like_x(x.shape)
        else:
            # 3. 最終 step (t=K=4): ᾱ_next = 1 で x_0 推定値がそのまま出力
            x = x0_hat

    return x.clamp(-1.0, 1.0)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `abar` (cumulative noise level ᾱ) | `[1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` | docs/architecture.md §5, docs/training.md §3.3 |
| `K` (step 数) | 4 | docs/training.md §3.3 / Fig 3 |
| `hop_length` | 256 | docs/architecture.md §3, docs/training.md §1.2 |
| 初期 `x` 分布 | `N(0, I)` | docs/training.md §4.2, DDPM 標準 |
| `eta` (stochasticity) | **1.0** (DDPM) default、`0.0` で DDIM | docs/training.md §4.2, §8.1 v1 採用 |
| `σ²` (η>0 時) | `η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)` ≥ 0 | docs/training.md §4.2 (β-free) |
| 最終 step (`ᾱ_next=1`) | `x = x0_hat` (射影なし) | docs/training.md §4.2 「最終ステップは x_0 推定値がそのまま出力」 |
| `β` 逆算 | **しない** (構造的に負値・発散) | docs/training.md §4.2 CRITICAL |
| sub-model dispatch | **1-to-1 (k=t)** | docs/architecture.md §5, docs/open-questions.md §B1 |
| 出力 clip | `[-1, 1]` (関数末尾で適用) | docs/architecture.md §2 (clip(-1,1)) |
| autograd | `@torch.no_grad()` | 推論専用 (T-M4.3 RTF 測定にも必要) |
| post-filter | **適用しない** (T-M3.4) | docs/milestones.md §M3.3 / §M3.4 |

### 2.4 アルゴリズム / 処理フロー

#### `_compute_ddpm_coefficients(abar)` (pure helper、β-free)
1. `sqrt_abar = √abar`、`sqrt_one_minus_abar = √(1-abar)`
2. `abar_next[:K-1] = abar[1:]` (denoising 順で 1 つクリーン側 ᾱ_{t+1})、`abar_next[K-1] = 1.0` (最終 step は完全クリーン)
3. **β / α / σ は計算しない** (denoising 順で ᾱ 増加列のため β が負値になり発散、§6.1 CRITICAL)
4. σ² は eta 依存なので reverse_sample 側で都度計算 (本 helper は eta 非依存の値のみ返す)
5. 全テンソルを `abar` の device/dtype と同じに保つ (in-place 演算なし)
6. dict (`abar`, `abar_next`, `sqrt_abar`, `sqrt_one_minus_abar`) として返す

#### `reverse_sample(model, mel, *, seed=None, eta=1.0)`
1. `B, _, T_mel = mel.shape` を取得、`T_audio = T_mel * model.hop_length`
2. `abar = model.NOISE_SCHEDULE_ABAR.to(device=mel.device, dtype=mel.dtype)`
3. `coef = _compute_ddpm_coefficients(abar)` (β なし)
4. seed が指定されたら `gen = torch.Generator(device=mel.device).manual_seed(seed)`、否なら `None`
5. `x = randn_like_x((B, T_audio))` で初期化 (denoising 順 t=1 入力)
6. `for t in range(1, K+1):` (1-indexed denoising 順):
   - `idx = t - 1`
   - `abar_t, abar_next, sqrt_abar_t, sqrt_one_minus_abar_t` を coef から取得
   - `c_t = sqrt_one_minus_abar_t.expand(B)`
   - `eps_pred = model.sub_models[idx](mel, x, c_t)` (1-to-1 dispatch)
   - `x0_hat = (x - √(1-ᾱ_t)·eps_pred) / √(ᾱ_t)`、`x0_hat.clamp(-1, 1)`
   - `t < K`: `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`、`x = √(ᾱ_next)·x0_hat + √(clamp(1-ᾱ_next-σ²,0))·eps_pred`、η>0 なら `+ √(σ²)·z`
   - `t == K`: 最終 step (ᾱ_next=1、`x = x0_hat`)
7. `return x.clamp(-1, 1)`

#### 不変条件
- **β を一切計算しない** (helper 戻り値に beta key なし、forward path でも参照しない)
- denoising 方向で `ᾱ_t < ᾱ_next` ⇒ `σ² ≥ 0` (全 step で sqrt の中身が非負)
- 各 sub-model は **1 回ずつ** だけ呼ばれる (`mock.call_count == 1`)
- `x` の shape は forward 中で変化しない (各 reverse step は同 shape 入出力)
- `x.device == mel.device`, `x.dtype == mel.dtype` を全 step で維持
- `torch.no_grad()` 内で実行、勾配グラフを構築しない (`x.requires_grad == False`)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `infer_diff.py` 実装 + `tests/test_reverse_sampler.py` 記述 | general-purpose |
| Reviewer | 1 | docs/training.md §4.2 (x_0 予測経由 β-free 式) / docs/architecture.md §5 整合性確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_reverse_sampler.py -v` + mock による sub-model call_count 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **partial** (T-M3.2 train_diff とは並列可能、T-M3.4 / T-M3.5 は本チケット完了後)
- 並列実行する場合の最大並列数: 2 (T-M3.2, T-M3.3 を 2 並列)

## 4. 提供範囲 (Scope)

### In Scope
- `reverse_sample(model, mel, *, seed=None, eta=1.0) -> Tensor` の実装
- `_compute_ddpm_coefficients(abar) -> dict` の pure helper 実装 (**β-free**、x_0 予測経由式用の値のみ)
- `@torch.no_grad()` で勾配グラフ非構築
- seed=None → stochastic、seed=int → deterministic の切替
- `eta` フラグ (1.0 DDPM / 0.0 DDIM) による stochasticity 切替
- point-specialized 1-to-1 dispatch (band 判定なし)
- 出力末尾の `clamp(-1, 1)` + 各 step の `x0_hat.clamp(-1, 1)`
- 4-step ループの x_0 予測経由射影式 (DDIM/DDPM 一般形、β 不使用)
- shape 検証 (`mel.dim() == 3`、`abar.shape == (4,)`)
- `tests/test_reverse_sampler.py` (5.1 / 5.3)
- `__init__.py` への `__all__` 追加 + docstring (英文 + 日本語)

### Out of Scope
- **Post-filter の fit / apply** (T-M3.4 で実装、本関数は plain reverse sample のみ)
- **訓練時の noise sampling** (T-M3.2 train_diff の責務)
- **`DiffWaveNext2` クラス本体** (T-M3.1 の責務、本チケットは consumer)
- **`SubModelDiff` クラス本体** (T-M1.6 の責務)
- **CLI entry point** (T-M3.5 smoke で必要なら別途、本チケットは関数のみ)
- **mixed precision (`--amp`)** (T-M4.3 / M6.2 で評価)
- **Batch-level seed 分離 (B 個の utterance に異なる seed)** (現実装は batch 全体で 1 seed、将来拡張)
- **β を逆算する旧式の DDPM reverse step** (構造的に発散するため不採用、§6.1 参照)
- **post-filter の wrapper API** (T-M3.4 が `reverse_sample` 後段で適用)

### Deliverable
- ファイル:
  - `src/wavenext2/inference/infer_diff.py` (新規)
  - `tests/test_reverse_sampler.py` (新規)
  - `src/wavenext2/inference/__init__.py` (re-export 追加)
- 関数:
  - `reverse_sample(model, mel, *, seed=None, eta=1.0) -> Tensor`
  - `_compute_ddpm_coefficients(abar) -> dict[str, Tensor]` (β-free)
- ドキュメント差分:
  - `docs/milestones.md` §M3.3 Acceptance チェック
  - `docs/tickets/index.md` の T-M3.3 ステータス

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_reverse_sampler.py`)

#### Shape / 範囲
- [ ] `test_output_shape`: `B=2, T_mel=80, hop=256` で `reverse_sample(model, mel).shape == (2, 80*256) == (2, 20480)` (milestones.md §M3.3 Acceptance #1)
- [ ] `test_output_range`: 出力が `[-1, 1]` に収まる (関数末尾 clamp、milestones.md §M3.3 Acceptance #1)
- [ ] `test_t_audio_equals_t_mel_times_hop`: 様々な `T_mel` (32, 64, 94, 100) で `T_audio = T_mel * model.hop_length` 整合

#### Deterministic / Stochastic
- [ ] `test_same_seed_deterministic`: 同じ mel + 同じ seed=42 で 2 回呼んで `torch.allclose(out1, out2)` (milestones.md §M3.3 Acceptance #2)
- [ ] `test_different_seed_different_output`: seed=42 と seed=43 で `not torch.allclose(out1, out2)`、`(out1 - out2).abs().mean() > 1e-3` (stochastic 性)
- [ ] `test_seed_none_stochastic`: `seed=None` で 2 回呼んで `not torch.allclose(out1, out2)` (毎回 `torch.randn` を直接呼ぶ)
- [ ] `test_no_global_seed_pollution`: 関数呼び出し前後で `torch.initial_seed()` / RNG state が変化しないこと (Generator local 化の確認)

#### Numerical sanity
- [ ] `test_no_nan_inf`: 出力に `torch.isnan().any() == False` かつ `torch.isinf().any() == False`
- [ ] `test_no_grad`: `out.requires_grad == False` (`@torch.no_grad()` 内で実行された証拠)
- [ ] `test_dtype_preserved`: `mel.dtype == torch.float32` で `out.dtype == torch.float32` (将来 fp16 対応の足場)
- [ ] `test_device_preserved`: CPU mel で CPU 出力、GPU mel で GPU 出力 (CUDA 環境のみ marker)

#### Dispatch / sub-model 呼び出し回数
- [ ] `test_each_submodel_called_once`: `unittest.mock.patch` で 4 つの sub-model を mock 化、`reverse_sample` 実行後 `for k in 0..3: mock[k].call_count == 1` (1-to-1 dispatch の核心)
- [ ] `test_submodel_call_order`: mock に call order tracker を仕込み、k=0, 1, 2, 3 の順 (denoising 順 t=1→4) で呼ばれることを確認
- [ ] `test_submodel_args`: mock の各 call で `args == (mel, x, c_t)` の形であることを assert (`c_t.shape == (B,)`)

#### coefficients helper (β-free)
- [ ] `test_compute_coefficients_shapes`: `abar = torch.tensor([1e-4, 2.8e-2, 5.6e-1, 9.1e-1])` で `coef["abar_next"].shape == (4,)`、`coef["sqrt_abar"].shape == (4,)` 等
- [ ] `test_compute_coefficients_no_beta_key`: `"beta" not in coef and "alpha" not in coef` (β/α を一切計算しない、§6.1 CRITICAL)
- [ ] `test_compute_coefficients_abar_next`: `coef["abar_next"][:3] == abar[1:]` (ᾱ_{t+1})、`coef["abar_next"][3] == 1.0` (最終 step は完全クリーン)
- [ ] `test_compute_coefficients_sqrt_one_minus_abar`: `coef["sqrt_one_minus_abar"]` が `[0.99995, 0.9858, 0.6633, 0.3]` 程度 (`docs/architecture.md` §5 表と整合)
- [ ] `test_compute_coefficients_abar_next_increasing`: 全 step で `coef["abar"][i] < coef["abar_next"][i]` (denoising 方向で ᾱ 増加 ⇒ σ² ≥ 0 の前提)
- [ ] `test_compute_coefficients_device_inherit`: `abar` を CUDA に移したら coef の全テンソルも CUDA (`@pytest.mark.gpu`)

#### Edge cases
- [ ] `test_mel_dim_validation`: `mel.dim() != 3` で `ValueError` (silent failure 回避)
- [ ] `test_batch_size_1`: `B=1` で正常動作
- [ ] `test_batch_size_8`: `B=8` で正常動作 (バッチ次元の broadcast 整合)
- [ ] `test_t_mel_short`: `T_mel=8` (短発話相当) でも shape 整合
- [ ] `test_model_eval_not_forced`: 本関数は `model.eval()` を呼ばない (呼び出し側責務、§6.1 参照)。`model.train()` 状態で呼んでも例外を上げず動く (sub-model に dropout がない前提で deterministic、§6.1 で再評価)

#### CRITICAL safety nets (§6.1 critical 由来)
- [ ] **`test_no_negative_beta_used`** (Critical): **β を計算しない** ことを assert。`_compute_ddpm_coefficients([1e-4, 2.8e-2, 5.6e-1, 9.1e-1])` の戻り値に `"beta"` / `"alpha"` key が存在しないこと、かつ `reverse_sample` の実装が β を **forward path で参照しない** こと。同 schedule で `reverse_sample` を実行して NaN/Inf が出ないこと (β を計算しないため `β[1]=-279` 問題が構造的に発生しないことの証明)
- [ ] **`test_sigma_non_negative`** (Critical、新規): 論文 schedule `[1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` の全 step (t=1..K-1) で `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next) ≥ 0` を `eta=1.0` で assert。`1 - ᾱ_next - σ² ≥ 0` (coef_eps の sqrt 中身) も全 step で非負。`ᾱ_t < ᾱ_next` (denoising 方向 ᾱ 増加) を前提に成立
- [ ] **`test_sigma_indexing_consistency`**: σ² が `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)` 式で計算されることを 1 例 pin。例: `eta=1.0`, t=1 (`ᾱ_t=1e-4`, `ᾱ_next=2.8e-2`) で `σ² = (1-2.8e-2)/(1-1e-4)·(1-1e-4/2.8e-2)` を手計算値と `torch.allclose` で照合 (`ᾱ_t`/`ᾱ_next` の index 整合確認)
- [ ] **`test_stochastic_flag`**: `eta=1.0` (DDPM) と `eta=0.0` (DDIM) で **異なる出力**。同 seed でも eta が切り替われば出力が変わる (`eta=0` は σ=0 で z 加算なし ⇒ deterministic projection)
- [ ] **`test_eta_zero_deterministic_given_seed`**: `eta=0.0` (DDIM) は初期ノイズ以外に乱数を消費しない。同 `seed=7`, `eta=0.0` で 2 回呼んで `torch.allclose(out1, out2)` (初期 `x=randn` が seed で固定 ⇒ 射影部に z 加算がないため完全 deterministic)。`eta=1.0` 版と比較して出力が異なること (z 加算の有無)
- [ ] **`test_seed_with_external_generator`**: `gen = torch.Generator(device).manual_seed(2025)` を外部から注入で deterministic、global RNG (`torch.initial_seed()`) は変化しない
- [ ] **`test_eval_mode_context`**: `eval_mode(model)` context manager が `model.training` を一時的に `False` にし、`__exit__` で元に戻すこと (BatchNorm/Dropout の eval 状態確認)

### 5.2 e2e / 結合テスト
- [ ] `test_with_real_diff_model`: `DiffWaveNext2.from_config(cfg)` 経由で実 model を生成 (T-M3.1 完了後)、`mel = torch.randn(1, 128, 32)` で `reverse_sample(model, mel)` が `(1, 8192)` を返し NaN/Inf なし
- [ ] `test_reverse_sample_integration` (`@pytest.mark.slow`): `T-M3.5` 想定の 1 utterance smoke (`mel` は実際の `extract_mel.py` 出力) で reverse_sample が動作。T-M3.5 で実音声テストするため本チケットでは smoke ではなく実 mel + 未訓練 model で「動く」だけ確認

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.3 より転記)
- [ ] mel → 4 step で `[-1, 1]` 範囲の波形が出力される
- [ ] 同じ mel + 同じ seed で deterministic

### 5.4 追加 acceptance (本チケット独自)
- [ ] `pytest tests/test_reverse_sampler.py` が exit code 0 (slow / gpu マーカー除く)
- [ ] `from wavenext2.inference.infer_diff import reverse_sample` で import 可能
- [ ] **`reverse_sample` 内で各 sub-model が exactly 1 回呼ばれる** (mock 検証で point-specialized 1-to-1 確認)
- [ ] 出力 tensor が NaN/Inf 非含、`[-1, 1]` 内
- [ ] `_compute_ddpm_coefficients` が pure function (副作用なし、in-place 演算なし、global state を変更しない)

### 5.5 テスト戦略
- **mock sub-model**: 軽量 `nn.Identity` ラッパや `MagicMock(return_value=torch.zeros_like(x))` で 4 sub-model を差し替え、call_count / call order を検証 (実 model は重いため fixture で 1 回だけ instantiate)
- **`scope="module"` fixture**: `DiffWaveNext2` の init は ~57.68M params で重いため、fixture で再利用
- **`@pytest.mark.slow`**: 実 model + 実 mel 統合テストは除外可能 (default off)
- **`@pytest.mark.gpu`**: CUDA 限定テスト (CPU 環境では skip)
- **CI 時間目標**: T-M3.3 全テスト合計 **~30 秒以内** (mock 中心、4 step × 軽量 forward で sub-second/test)
- **coverage 目標**: 90% (実音声 e2e は T-M3.5 でカバー)

## 6. 懸念事項

### 6.1 技術的リスク

#### CRITICAL [✅ 解決 (2026-05-27)]: `β_t` 負値問題 → x_0 予測経由 β-free 式で構造的に解消
- **解決ステータス (2026-05-27)**: `docs/training.md` §4.2 で reverse step が **β を使わない x_0 予測経由の DDIM/DDPM 一般形** に確定した。本チケット §2 のコード (`_compute_ddpm_coefficients` / `reverse_sample`) も同式に更新済み。**β を一切計算しないため、`β[1]=-279` 問題は構造的に発生しない**。denoising 方向で `ᾱ_t < ᾱ_next ⇒ σ² ≥ 0` が保証され、sqrt の中身が常に非負。
- **以下は「なぜ β を使わないか」の根拠** (旧 CRITICAL の原因分析を保持):
- **問題 (旧)**: `NOISE_SCHEDULE_ABAR = [1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]` に対して標準 DDPM の `β_t = 1 - ᾱ_t / ᾱ_{t-1}` を **adjacent indices に直接適用すると β が負値になる**
  - 計算例: `β[1] = 1 - ᾱ[1] / ᾱ[0] = 1 - 2.8e-2 / 1.0e-4 = 1 - 280 = -279`
  - 同様に `β[2] = 1 - 5.6e-1 / 2.8e-2 = 1 - 20 = -19`、`β[3] = 1 - 9.1e-1 / 5.6e-1 ≈ -0.625`
  - **根本原因**: 論文の 4 点 schedule は **連続 DDPM schedule (通常 T=1000) のサブサンプリング** であり、`ᾱ_{t-1}` は「直前 step」ではなく「数百 step 前」の値。**adjacent indices 間の β は意味を持たない (実質 3-step skip)**。β を逆算した `α_t = ᾱ_t/ᾱ_{t-1}` も `α_2 = 280`, `α_3 = 20`, `α_4 ≈ 1.625` と `α > 1` になり DDPM の前提 (`α ≤ 1`) を破る
- **採用した解決策** (`docs/training.md` §4.2):
  - `_compute_ddpm_coefficients` は β / α / σ を **一切計算しない** (戻り値 key から削除)。`ᾱ_t`, `ᾱ_next`, `√ᾱ_t`, `√(1-ᾱ_t)` のみ返す
  - reverse step は各 step で `x0_hat = (x - √(1-ᾱ_t)·ε) / √(ᾱ_t)` を推定し、次のクリーン側 `ᾱ_next` (最終 1.0) へ射影 `x = √(ᾱ_next)·x0_hat + √(1-ᾱ_next-σ²)·ε + σ·z`
  - `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`。denoising 方向で `ᾱ_t < ᾱ_next ⇒ 1-ᾱ_t/ᾱ_next ≥ 0 ⇒ σ² ≥ 0` (負値問題なし)
  - これは「論文 schedule は連続 DDPM の sub-sampling 結果であり、adjacent transition を β で表現できない」という事実に対し、**β を経由せず x_0 推定値を skip-aware に射影する** ことで対処する (DDIM の skip-step 定式化に一致)
- **検証 (本チケット §5)**:
  - `test_no_negative_beta_used`: coefficients に `"beta"`/`"alpha"` key がないこと + 論文 schedule で NaN/Inf が出ないこと
  - `test_sigma_non_negative`: 全 step で `σ² ≥ 0` かつ `1-ᾱ_next-σ² ≥ 0` を assert
  - `test_sigma_indexing_consistency`: σ² 式の手計算値と照合
  - **T-M3.5 smoke**: NaN/Inf の出現を即検知 (β を計算しないため発生しない想定だが、保険として残す)

#### CRITICAL: `ᾱ_t` / `ᾱ_next` indexing 整合 (β-free 式での off-by-one リスク)
- **問題**: x_0 予測経由式では各 step で「現在 `ᾱ_t`」と「次のクリーン側 `ᾱ_next`」の 2 値を使う。0-indexed `idx = t-1` で `ᾱ_t = coef["abar"][idx]`、`ᾱ_next = coef["abar_next"][idx]` が **正しいペア** (= `ᾱ_{t+1}`、最終 step は 1.0) になっているか確認が必要
- **混乱の根源**:
  - σ² は `ᾱ_t` (現在) と `ᾱ_next` (1 つクリーン側) の **両方** を参照する。`abar_next[idx]` が `abar[idx+1]` (t<K) / `1.0` (t=K) に正しくマップされているか
  - x0_hat の係数 (`√(1-ᾱ_t)`, `√ᾱ_t`) は **現在 step** の値、射影の `√(ᾱ_next)` は **次 step** の値で、混在しやすい
- **本チケット採用**: helper が `abar_next` を明示的に構築 (`abar_next[:K-1]=abar[1:]`, `abar_next[K-1]=1.0`) し、`reverse_sample` は `coef["abar"][idx]` と `coef["abar_next"][idx]` を同じ idx で引く。**手計算で 1 例 pin** (`test_sigma_indexing_consistency`)
- **検知タイミング**:
  - **T-M3.5 smoke 前**: 1 utterance × 既知 seed で 1 step だけ手計算実行、`σ²[idx=0]` が `(ᾱ_t=1e-4, ᾱ_next=2.8e-2)` の組で期待値と一致するか pin
  - **T-M3.3 unit test (5.1)**: `test_sigma_indexing_consistency` / `test_compute_coefficients_abar_next` で整合
- **未解決のまま実装すると**: 偶然動くが品質低下 (off-by-one が音質を間接的に劣化)、M5.2 で MCD が想定より悪い形で気付く

#### CRITICAL: `reverse_step` 式の正確性 (x_0 予測経由 DDIM/DDPM 一般形と整合確認)
- **問題**: `docs/training.md` §4.2 の x_0 予測経由式と実装の **添字対応** が混乱しやすい
  - denoising 順 (t=1→K で純ノイズ→クリーン) で `ᾱ` は **増加列**、射影先は常に「より大きい `ᾱ_next`」
  - 最終 step は `ᾱ_next = 1` で射影項が消え `x = x0_hat` になる特殊形
- **本チケット採用解釈** (`docs/training.md` §4.2 完全準拠):
  - `t in 1..K` (1-indexed denoising 順)
  - sub-model k=t を呼んで `eps_pred` を得る
  - `x0_hat = (x - √(1-ᾱ_t)·eps) / √(ᾱ_t)`、`clamp(-1,1)`
  - `t < K`: `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`、`x = √(ᾱ_next)·x0_hat + √(clamp(1-ᾱ_next-σ²,0))·eps + (η>0 なら σ·z)`
  - `t == K`: 最終 step、`x = x0_hat` (ᾱ_next=1)
- **検証手段**:
  1. `test_compute_coefficients_*` で `ᾱ_next` / `√(1-ᾱ)` の値が論文表と整合
  2. `test_each_submodel_called_once` で dispatch order が正しい
  3. `test_sigma_non_negative` で全 step `σ² ≥ 0` (β-free 式の正当性)
  4. M5.2 / M3.5 smoke で「生成波形が GT に近づく」ことで間接的に検証
- **未解決のまま実装すると**: 生成波形がノイズに近い (聴感破綻)、M3.5 / M5.2 で検知

#### CRITICAL: `eta` の選択 (DDPM η=1 vs DDIM η=0)
- **問題**: x_0 予測経由式の `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)` で η をいくつにするか
- **本チケット採用**: **`eta=1.0` (DDPM、確率的) を default**、`eta=0.0` (DDIM、決定論的) を引数で切替可能 (`docs/training.md` §4.2 / §8.1 v1 採用)
  - 中間 step (t=1..K-1) で η>0 のとき `σ·z` を加算、η=0 なら射影のみ (deterministic)
  - 最終 step (t=K) は `ᾱ_next=1` で `σ²=0` (η に依らず deterministic、`x=x0_hat`)
- **検知**: M3.5 smoke で reverse sample 品質を `eta=1.0` / `eta=0.0` で比較、4-step では DDIM の方が安定する場合あり

#### CRITICAL: `c_t = sqrt(1 - abar[t-1])` の index 計算 (1-indexed vs 0-indexed)
- **問題**: 仕様セクションの擬似コード `c_t = sqrt(1 - abar[t-1])` と本実装の `c_t = coef["sqrt_one_minus_abar"][idx]` (idx = t-1) は同じ値だが、**実装時に 1-indexed/0-indexed の混乱** で off-by-one リスク
- **本チケット採用**: 内部は 0-indexed `idx = t - 1` で統一、コメントで 1-indexed t との対応を必ず明示
- **検証手段**: `test_compute_coefficients_sqrt_one_minus_abar` で `[0.99995, 0.9858, 0.6633, 0.3]` 表と整合確認

#### CRITICAL: 中間 step stochastic vs 最終 step deterministic (eta との両立)
- **問題**: 「最終 step は `ᾱ_next=1` で `σ²=0` (deterministic、`x=x0_hat`)」(`docs/training.md` §4.2) という仕様と、「同じ seed で deterministic」(Acceptance) が **両立するか**
- **本チケット採用**: `eta=1.0` のとき **中間 step (t<K) のみ `σ·z` で stochastic**、最終 step は構造的に deterministic。`eta=0.0` なら全 step deterministic (DDIM)。seed=int の場合、Generator local 化により stochastic step も再現可能 → Acceptance 「同 seed deterministic」と矛盾しない (seed が乱数 sequence を fix するため)
- **検証**: `test_same_seed_deterministic` で 2 回呼び同じ出力、`test_different_seed_different_output` で異なる seed で異なる出力、`test_eta_zero_deterministic_given_seed` で `eta=0` の決定性

#### CRITICAL: `model.eval()` モード切替の責務
- **問題**: 本関数が `model.eval()` を内部で呼ぶか、呼び出し側に委ねるか
- **本チケット採用** (現状): **呼び出し側責務** (T-M3.5 / T-M4.3 等が `model.eval()` を呼ぶ)
- **理由**:
  - 関数が `model.train()` 状態を `eval()` に変更すると **呼び出し側の意図に反する副作用**
  - `with model.eval():` context manager 化も検討したが PyTorch 標準は `model.eval()` メソッドのみで context 形式は非標準
  - 推論専用関数なので呼び出し側は通常 `eval()` 呼んでから渡す
- **代替案** (§8.1 案 5): `force_eval: bool = True` 引数を追加し、`True` のとき `prev_training = model.training; model.eval()` → 関数末尾で `model.train(prev_training)` で restore する with-context 実装
- **検知**: `test_model_eval_not_forced` で `model.train()` 状態のまま呼んでも例外を上げず動くことを確認 (sub-model に dropout なしの前提、§6.3 で再評価)

#### 通常項目

- **`post-filter` 適用しないが index/seed 互換性を保つ必要性**:
  - T-M3.4 が `reverse_sample(model, mel, seed=42)` の出力 (B, T_audio) を受け取り FIR convolution を適用する
  - 本関数の `clamp(-1, 1)` を **post-filter 前** に適用するか、**post-filter 後** に適用するかは T-M3.4 の責務
  - 本チケット v1 では **clamp(-1, 1) を関数末尾で適用**、T-M3.4 が必要なら別 clamp する設計
  - 代替: `clip_output: bool = True` 引数を追加して T-M3.4 で `False` で呼ぶ選択肢、§8 で検討

- **`NOISE_SCHEDULE_ABAR` の device 同期**:
  - T-M3.1 で `NOISE_SCHEDULE_ABAR` を class attribute (CPU Tensor) か buffer (auto device sync) として定義するか未確定
  - 本関数は `abar = model.NOISE_SCHEDULE_ABAR.to(device=mel.device, dtype=mel.dtype)` で明示同期
  - T-M3.1 で buffer 化されたらこの行は no-op になるが安全のため残す

- **`seed` の Generator scope 問題**:
  - `torch.Generator(device=mel.device).manual_seed(seed)` で生成した Generator は **batch 全体で 1 つ** (B 個の utterance に同じ noise sequence を共有しない)
  - B=2 で同じ mel × 2 を渡すと、各サンプルが異なる noise sequence を持つ (Generator の 1 シーケンスから順に消費するため)
  - **これは現実的な振る舞い** (1 utterance ずつ生成するのが通常、batch 推論は速度向上目的)
  - 将来 batch ごとに別 seed を渡したい場合は `seed: int | list[int]` に拡張、§8 で検討

- **`SubModelDiff.forward` の戻り値 shape** (T-M1.6 / T-M3.1 から伝搬):
  - 期待: `(B, T_audio)` の noise prediction `eps_pred`
  - T-M1.6 で `eps_pred.shape == x.shape` を保証していることを前提
  - 不整合があれば T-M1.6 / T-M3.1 の修正依頼

- **`torch.no_grad()` と `@torch.inference_mode()` の選択**:
  - `@torch.no_grad()` は autograd graph 非構築、`@torch.inference_mode()` はさらに version counter も不要で速い
  - 本チケット v1 は `@torch.no_grad()` (互換性高)、`inference_mode()` は M6.2 / T-M4.3 で RTF 高速化必要時に評価

- **`torch.compile` の互換性**:
  - 本関数を将来 `torch.compile` で wrap するときに `if t < K:` の dynamic control flow が compile 不可能な可能性
  - 4-step 固定なので `torch.compile` ではなく manual unroll で対処可、§8 で検討

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目:
  - **DDPM 標準形**: ✅ `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (PDF Fig 1b 画像確認、§A5)
  - **4-step noise schedule**: ✅ `ᾱ = [1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` (PDF §3.3、§A5)
  - **point-specialized 1-to-1 dispatch**: ✅ Table 1 のパラメータ数 57.68M=14.42M×4 から確定 (§B1)
  - **Diff reverse step**: ✅ **x_0 予測経由 DDIM/DDPM 一般形 (β-free)** に確定 (`docs/training.md` §4.2、2026-05-27)。β を逆算する旧式は denoising 順 ᾱ 増加列で発散するため不採用 (§6.1)
  - **post-filter は本チケットで適用しない**: ✅ docs/milestones.md §M3.3 / §M3.4 で分離
- 本チケット内での決定:
  - **`seed: int | None = None` (keyword-only)**: stochastic と deterministic を seed 引数 1 つで切替 (`Optional[int]` で揃える)
  - **`eta: float = 1.0`**: stochasticity フラグ (1.0 DDPM / 0.0 DDIM)、`docs/training.md` §4.2 / §8.1 v1 採用
  - **β を一切計算しない**: x_0 予測経由式で構造的に σ² ≥ 0 (§6.1 解決済み)
  - **最終 step は `ᾱ_next=1` で `x=x0_hat` (deterministic)**: `docs/training.md` §4.2 完全準拠
  - **`@torch.no_grad()` で wrap**: 推論専用、勾配グラフ非構築
  - **`model.eval()` 呼び出しは呼び出し側責務**: 副作用回避

### 6.3 他チケットとの整合性

- **T-M3.1 (DiffWaveNext2)** との整合 (上流):
  - 期待 attribute: `NOISE_SCHEDULE_ABAR: Tensor shape (4,)`、`sub_models: nn.ModuleList[SubModelDiff]` (length 4)、`hop_length: int = 256`
  - 期待 signature: `model.sub_models[k](mel, x_t, c_t) -> eps_pred (B, T_audio)`
  - 不整合があれば T-M3.1 を修正 (本チケットは consumer)

- **T-M3.4 (post-filter)** との整合 (下流):
  - 本関数の出力 `(B, T_audio)` を T-M3.4 が FIR convolution で後処理
  - `clamp(-1, 1)` の適用タイミング: 本関数では関数末尾、T-M3.4 が必要なら別途 clamp
  - 代替: `clip_output: bool` 引数を追加 (§8 検討)

- **T-M3.5 (Diff smoke)** との整合 (下流):
  - 1 utterance で `reverse_sample` を呼んで GT に近づくことを確認
  - `seed=42` で deterministic、`seed=None` で stochastic 性確認

- **T-M4.3 (RTF measurement)** との整合 (下流):
  - 期待呼び出し: `model.eval(); rtf = measure_rtf(reverse_sample, model, mel, ...)`
  - `@torch.no_grad()` で勾配計算なしで速度測定可能 (`@torch.inference_mode()` への変更は §8 で評価)
  - CPU 測定 (`torch.set_num_threads(1)`)、GPU 測定 (`torch.cuda.synchronize()`) 対応

- **T-M3.2 (train_diff)** との整合 (姉妹):
  - 訓練時の noise sampling (band 内 uniform) と推論時の reverse step は **別実装**
  - 訓練と推論で sub-model forward signature を統一 (`(mel, x_t, c_t) -> eps_pred`)、T-M1.6 / T-M3.1 で確定

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §4.2 の擬似コード (1-indexed denoising 順 t=1..K, x0_hat → 射影, ᾱ_next, σ², eta, 最終 step) と整合
- [ ] **β / α を一切計算していない** (helper 戻り値・forward path とも)、x_0 予測経由式のみ (§6.1 CRITICAL)
- [ ] `docs/architecture.md` §5 の point-specialized 1-to-1 dispatch (k=t) と整合
- [ ] `docs/open-questions.md` §A5 / §B1 / §B3 (DDPM 標準形、4-step schedule、reverse step) と整合
- [ ] σ² 式が `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)` と一致、denoising 方向で `σ² ≥ 0`
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)
- [ ] Unit テスト全 pass、特に mock による `call_count == 1` 検証
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`、`*` で keyword-only)
- [ ] エラー処理: `mel.dim() != 3` で `ValueError` (silent failure なし)
- [ ] **参考実装 (FastDiff / BDDM `bddm/sampler/sampler.py`) をコピーしていない** (CLAUDE.md 末尾の方針、`docs/training.md` §4.2 の論述から再構成)
- [ ] `__init__.py` の `__all__` に `reverse_sample` が追加されている
- [ ] `@torch.no_grad()` が適用されている (autograd graph 非構築)
- [ ] `_compute_ddpm_coefficients` が pure function (副作用なし、in-place 演算なし)
- [ ] `seed=None` で `torch.randn` 直接呼び出し、`seed=int` で local Generator 経由 (global RNG 非汚染)
- [ ] 出力末尾の `clamp(-1, 1)` が適用されている
- [ ] CPU でテストが pass する (CUDA 不要、`@pytest.mark.gpu` で分離)
- [ ] mock による sub-model `call_count == 1` 検証が pass

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M3) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **`reverse_sample(model, mel, *, seed=None, eta=1.0)` の 1 関数 + `_compute_ddpm_coefficients` pure helper (β-free) + `eval_mode` context manager helper、x_0 予測経由 DDIM/DDPM 一般形 4-step + 1-to-1 dispatch + 最終 step `x=x0_hat`**
  - 理由:
    - (a) 1 関数で `from wavenext2.inference.infer_diff import reverse_sample` の唯一のエントリポイント、T-M3.4 / T-M3.5 / T-M4.3 が薄い wrapper で済む
    - (b) `_compute_ddpm_coefficients` を pure helper として切り出すことで unit test が容易 (式の正確性を mock 不要で検証可能)
    - (c) 1-to-1 dispatch は `docs/architecture.md` §5 / Table 1 で確定 (point-specialized partition、band 判定不要)
    - (d) `@torch.no_grad()` で勾配グラフ非構築、推論専用
    - (e) `seed: int | torch.Generator | None = None` で stochastic / deterministic / external generator 注入の 3 形式を 1 引数切替、global RNG 非汚染
    - (f) 最終 step σ=0 は `docs/training.md` §4.2 完全準拠 (DDPM 原典は通常 σ_t を全 step 適用するが、4-step BDDM 風では最終 step decoder-like が慣例)
    - (g) **`eta: float = 1.0` を v1 化** (旧 `stochastic: bool` を float 一般化): 4-step (K=4) の reverse process では中間 σ·z の stochasticity が品質を主導するため、DDPM (`eta=1.0`) と DDIM (`eta=0.0`) の両方を 1 引数で default 実装し、CLI で切替可能化 (M3 phase review 結果、§8.1 採用昇格)。x_0 予測経由式では σ² が eta² に比例するため float 連続値で η∈[0,1] の中間も表現可能
    - (h) **`@contextmanager def eval_mode(model)` helper** を切り出し、`reverse_sample` 内で常時使用 (副作用なし、`model.train(prev_training)` で復元、`prev_training = model.training; model.eval(); yield; model.train(prev_training)`)。T-M3.4 / T-M3.5 / T-M4.3 が全部 `model.eval()` を書く重複を避ける (M3 phase review 結果)

#### 検討追加 (M3 phase review 結果)

- **`variance_type` (将来検討)**: 本チケット v1 は x_0 予測経由式の `σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)` (DDIM 流の skip-aware small variance) を採用
  - 論文 §3.3 では σ² の具体形式は明記なし。β-free 式は β を逆算しないため Ho2020 の `σ²=β_t` (large) 形式は **そもそも計算不能** (β が負値)。large variant が必要なら別途 σ² 式を差し替える
  - **再評価タイミング**: M3.5 smoke 後の品質確認時、必要なら η 以外の σ² 形式を実験

- **`seed: int | torch.Generator | None = None` に拡張**: int 1 つだけでなく外部から `torch.Generator` を注入可能化
  - **理由**: M3.4 post-filter fit (200 utterance) と T-M4.3 RTF (warmup 含む反復測定) で **global RNG を汚染しない** ため、外部 Generator を注入したい
  - **実装**: `if isinstance(seed, torch.Generator): gen = seed; elif isinstance(seed, int): gen = torch.Generator(device).manual_seed(seed); else: gen = None`
  - **v1 で採用**: M3 phase review 結果、複雑度が低く下流タスクに有用

#### Deprecated (却下案)

1. **DDIM (η=0、全 step deterministic) — ✅ v1 で `eta=0.0` として in-scope 化**
   - メリット: 推論が (初期ノイズを除き) deterministic、4-step では DDPM より安定する場合あり
   - 当初は「論文外」として却下していたが、`docs/training.md` §4.2 の x_0 予測経由式は η パラメータで DDPM (η=1) / DDIM (η=0) を **同一式で表現** できるため、`eta: float = 1.0` の一部として v1 採用 (default は DDPM の η=1.0、論文準拠)

2. **4 step 以外の step 数 (8 / 16 / 32)**
   - メリット: 多 step で品質向上の可能性 (FastDiff / DDPM-1000 等)
   - 却下: 論文は **4-step 固定**、step 数変更は ablation スコープ (M6.3)
   - **採用代替**: `n_steps: int = 4` 引数で将来拡張可能化、本チケット v1 では固定

3. **DPM-Solver-2 / DPM-Solver-3 (高速 ODE solver)**
   - メリット: 数 step で高品質、論文外の SOTA solver
   - 却下: 論文外、再現実装スコープ外

4. **conditional guidance (CFG, classifier-free guidance)**
   - メリット: mel に対する忠実度向上の可能性
   - 却下: 本論文では使わない (mel→audio mapping は強い条件付きなので CFG 不要)

5. **`force_eval: bool = True` で `model.eval()` を内部で呼ぶ**
   - メリット: 呼び出し側が `eval()` 忘れる事故を防止
   - 却下 (現状): 関数内で `model.train()` 状態を変えるのは副作用 (with-context で restore する設計は複雑)
   - **採用代替**: `force_eval: bool = False` 引数を追加 (default False)、`True` のとき with-context で eval mode 切替 + restore、§9 検討

6. **`@torch.inference_mode()` で wrap (`@torch.no_grad()` より高速)**
   - メリット: version counter 不要で更に速い、T-M4.3 RTF で有利
   - 却下 (現状): 互換性高い `@torch.no_grad()` を v1 採用、`inference_mode()` は M6.2 / T-M4.3 で評価
   - **採用代替**: `use_inference_mode: bool = False` 引数を追加、`True` のとき `@torch.inference_mode()` decorator を動的に切替 (実装複雑、関数ごと別実装の方が単純)

7. **batch ごとに別 seed (`seed: int | list[int]`)**
   - メリット: B=4 で 4 つの異なる seed を渡せる (utterance ごとに reproducible)
   - 却下 (v1): default は batch 全体で 1 seed、現実的 (1 utterance ずつ生成する用途が多い)
   - **採用代替**: `seed: int | list[int] | None` に拡張、§9 検討

8. **`clip_output: bool = True` 引数で末尾 clamp を制御**
   - メリット: T-M3.4 post-filter が `clip_output=False` で生波形を受け取り、自分で clip 制御
   - 却下 (v1): 関数末尾で常に `clamp(-1, 1)` を適用、T-M3.4 は必要なら別 clamp
   - **採用代替**: `clip_output: bool = True` を引数化、§9 検討

9. **`reverse_sample_with_intermediates` で中間 x_t を list で返す**
   - メリット: debugging / visualization に便利
   - 却下 (v1): 最終 x のみ返す、debugging は `hook` 経由で実装可能
   - **採用代替**: `return_intermediates: bool = False` 引数を追加 (T-M2.4 GAN モデルと整合)、§9 検討

10. **CLI entry point として `python -m wavenext2.inference.infer_diff --mel ... --output ...` で起動**
    - メリット: shell から直接使える
    - 却下 (v1): 関数のみ、CLI は T-M3.5 / T-M6.2 で必要なら別途
    - **採用代替**: `if __name__ == "__main__":` ブロックを追加、§9 検討

11. **`torch.compile` で関数全体を pre-compile**
    - メリット: 推論速度向上 (T-M4.3 RTF 改善)
    - 却下 (v1): `if t < K:` の dynamic control flow が compile 不可能な可能性、4-step 固定なので manual unroll で対処可
    - **採用代替**: 4-step を手動 unroll した版 `reverse_sample_unrolled` を別途実装、§9 検討

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| DDPM vs DDIM | M5.2 smoke 発散時 | 品質低下なら `eta=0.0` (DDIM) に切替 (v1 で引数化済み) |
| 4-step 固定 | M6.3 ablation | 8/16/32 step で品質比較 |
| `@torch.no_grad()` vs `@torch.inference_mode()` | T-M4.3 RTF 測定時 | inference_mode で 5%+ 速度向上が見られたら変更 |
| `force_eval` 引数 | M6.2 訓練後の推論時 | 事故が起きたら追加 |
| `clip_output` 引数 | T-M3.4 post-filter 実装時 | post-filter が clip を必要としたら追加 |
| batch ごとに別 seed | M6.2 batch 推論時 | 必要性が出たら拡張 |
| `return_intermediates` | T-M3.5 smoke / 可視化要件発生時 | 可視化が必要なら追加 (T-M2.4 GAN と整合) |
| CLI entry point | T-M6.2 推論パイプライン構築時 | shell からの直接起動要件があれば追加 |
| `torch.compile` 適用 | T-M4.3 RTF 測定後 | 推論速度ボトルネックが reverse_sample なら対応 |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。
  - Diff-WaveNeXt 2 の推論本質は「4-step reverse sampling + 1-to-1 dispatch」であり、これを 1 関数に閉じ込めることで T-M3.4 / T-M3.5 / T-M4.3 が「`reverse_sample(model, mel, seed=42)` を呼ぶだけ」で済む
  - post-filter は **別チケット (T-M3.4)** に分離することで、reverse sampling 本体の正確性と post-filter の正確性を **独立に検証** できる
- **別マイルストーンに移すべき部分はないか**: なし。M3 (Diff) の中核として正しい位置
- **GAN との API 不整合**:
  - GAN: `model.forward(mel, audio_length=None, return_intermediates=False)` (T-M2.4)
  - Diff: `reverse_sample(model, mel, *, seed=None)` (本チケット、関数形式)
  - 整合性: GAN は `nn.Module` の `forward`、Diff は `model` を受け取る関数。API 形式が異なるが、**それぞれのドメインで自然な形** (GAN は iteration ごとに sub-model 呼び出し、Diff は reverse process が外部関数で表現される慣例 DDPM/BDDM 流)
  - 将来 `BaseVocoder.synthesize(mel)` のような統一 API を抽出するかは **M3 phase review** で再評価 (T-M2.4 §8.2 で「M3.1 着手前に判断」と申し送り)
- **`reverse_sample` を `DiffWaveNext2.synthesize(mel)` method に統合する案**:
  - メリット: GAN の `model(mel)` と API 統一
  - 却下 (v1 関数本体は分離): 関数として独立させた方が test しやすい (mock 化容易)、`DiffWaveNext2` 本体がシンプル
  - **採用代替 (M3 phase review で昇格): `DiffWaveNext2.synthesize = reverse_sample` thin alias を **今すぐ T-M3.3 内に実装する** (T-M3.1 で既に採用済み、本チケットから re-export して整合保持)
    - **理由**: T-M4.3 (RTF) は GAN/Diff 横断で `model.synthesize(mel)` を測りたいため、両モデルで `synthesize` API が揃っていることが望ましい
    - **実装場所**: `infer_diff.py` 末尾で `DiffWaveNext2.synthesize = reverse_sample` を 1 行追加 (monkey-patch、import side effect)
    - **代替実装**: `DiffWaveNext2.synthesize` を T-M3.1 内で `def synthesize(self, mel, **kw): from wavenext2.inference.infer_diff import reverse_sample; return reverse_sample(self, mel, **kw)` 形式で定義 (circular import 回避のため遅延 import)
    - **T-M3.1 との整合**: T-M3.1 §9 で本件を申し送り、両チケットで実装合意
- **`_compute_ddpm_coefficients` を `DiffWaveNext2` の class method に統合する案**:
  - メリット: model と coefficients の責務統一
  - 却下 (v1): pure helper として切り出した方が unit test 容易、再利用性高 (将来 DDIM 等で別 step 数を使う際にも流用可)

### 8.3 学んだこと (2026-05-28 実装完了)

- **チケット §2 の擬似コードがほぼそのまま実装に使えた**: M3 phase review で β-free 式・eta・eval_mode・seed 3 形式が事前確定していたため、設計判断の出戻りなし。CRITICAL (β 負値) が解決済みで、`_compute_ddpm_coefficients` が β/α を持たない構造が NaN 発生を構造的に排除。実 model でも `test_real_model_no_negative_beta_nan` で確認。
- **`DiffWaveNext2.synthesize = reverse_sample` の method binding が成立**: `reverse_sample` の第 1 引数が `model` なので、class attribute として代入すると `model.synthesize(mel, seed=...)` が `reverse_sample(model, mel, seed=...)` に束縛される。`@torch.no_grad()` で wrap 済みでも `functools.wraps` 経由で plain function のままなので binding に影響しない。T-M3.1 で deferred としていた alias を本チケットで注入完了 (`test_synthesize_alias` で同等性検証)。
- **`seed: int | torch.Generator | None` の 3 形式正規化**: `isinstance(seed, torch.Generator)` を最初に判定 (Generator は int ではないため順序は任意だが明示)。int の場合のみ `torch.Generator(device).manual_seed(int(seed))` で local 生成。`test_no_global_seed_pollution` で `torch.random.get_rng_state()` が seed 指定時に不変なことを確認。
- **`eval_mode` を reverse_sample 内部で常時使用 (§8.1 h の phase review 判断を採用)**: §6.1 の「呼び出し側責務」は phase review で「内部で eval_mode を使い train 状態を復元」に更新された。`test_reverse_sample_restores_training_state` で `model.train()` 状態が呼び出し後も保持されることを確認。`test_model_eval_not_forced` の旧意図 (例外を上げず動く) も満たす。
- **mock sub-model で dispatch 検証**: 実 DiffWaveNext2 (~57.42M) は重いため、`_MockSub` (eps=x*scale、call 記録) + `_MockModel` (noise_schedule_abar buffer + NOISE_SCHEDULE_ABAR property) で軽量化。`call_count==1` / denoising 順 (idx 0→3) / c_t=√(1-ᾱ_t) を 1.65 秒で検証。実 model 結合テストは別 fixture (module scope) で最小限。
- **37 tests pass、全体 312 passed / 1 skipped / 2 deselected、ruff clean**。
- 教訓: phase review で CRITICAL を事前解決し §2 コードを確定形にしておくと、実装は写経に近くなり高速・低リスク。mock interface は実 class の必要属性 (`sub_models` / `hop_length` / `NOISE_SCHEDULE_ABAR`) だけを最小実装すれば dispatch ロジックを実 model 非依存に検証できる。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M3.1 (DiffWaveNext2) から受領 (M3 phase review)
- **`synthesize` alias を T-M3.3 内に実装**: `DiffWaveNext2.synthesize = reverse_sample` (`infer_diff.py` 末尾の import side effect、または T-M3.1 内で遅延 import 形式の method として定義)
- **理由**: T-M4.3 (RTF 測定) は GAN / Diff 横断で `model.synthesize(mel)` を測りたい (両モデルで API 統一)
- **実装位置**: `infer_diff.py` 末尾の 1 行 `DiffWaveNext2.synthesize = reverse_sample` (T-M3.1 側で重複実装を避ける)、または T-M3.1 内で thin wrapper method を定義 (circular import 回避のため遅延 import)
- **T-M3.1 との同期点**: T-M3.1 §9 で本件を申し送り、実装の主管を確定

#### T-M3.4 (post-filter) へ
- **使用方法**:
  ```python
  from wavenext2.inference.infer_diff import reverse_sample
  from wavenext2.inference.post_filter import apply_post_filter, load_fir

  fir = load_fir("checkpoints/diff/fir.npy")
  # reverse sampling (deterministic for fit reproducibility)
  y_synth = reverse_sample(model, mel, seed=43)         # (B, T_audio)
  # post-filter (T-M3.4)
  y_post = apply_post_filter(y_synth, fir)              # (B, T_audio)
  ```
- **重要事項**:
  - 本関数の出力は `clamp(-1, 1)` 済み、`shape == (B, T_mel * 256)`、`device == mel.device`
  - **clamp(-1, 1) を post-filter 前に適用**: 本関数末尾で適用済み、T-M3.4 で convolution 後に再度 clamp が必要なら別途実装
  - **`clip_output: bool = True` 引数追加要否**: T-M3.4 が `clip_output=False` で「raw signal を受け取って自分で clip」を要求するなら本チケットを再オープン (§8.1 案 8)
  - **post-filter 適用は T-M3.4 の責務**、本関数は plain reverse sample のみ
  - **M3 phase review 追加**: T-M3.4 (post-filter fit) で `seed=43` を `reverse_sample` 呼び出しに渡して deterministic 化、200 utterance の fit が再現可能に
  - **`apply_post_filter` を `reverse_sample` の引数化検討**: `post_filter: np.ndarray | None = None` 引数を `reverse_sample` に追加し、与えられたら関数内で post-filter を適用する API も検討 (T-M3.4 実装時に判断、§8.1 案として追加可能)

#### T-M3.5 (Diff smoke) へ
- **使用方法**:
  ```python
  # eval_mode helper を使う (M3 phase review)
  from wavenext2.inference.infer_diff import reverse_sample, eval_mode
  with eval_mode(model):  # model.training 状態を保護
      y_pred = reverse_sample(model, mel, seed=42)
  loss = mr_stft_loss(y_pred, x_gt)  # GT との比較
  ```
- **重要事項**:
  - **β 負値問題が起きないことを 1 例手計算で先に pin** (§6.1 critical 最優先): smoke 実行前に `_compute_ddpm_coefficients([1e-4, 2.8e-2, 5.6e-1, 9.1e-1])` の戻り値を pytest で 1 例検証し、reverse step が NaN なく完走することを確認
  - **smoke で NaN/Inf 確認**: 1 epoch 訓練後の reverse sample に `assert not torch.isnan(y_pred).any() and not torch.isinf(y_pred).any()`
  - `model.eval()` を **必ず呼ぶ** か `eval_mode(model)` context manager を使う (本関数は内部で呼ばない、§6.1 critical)
  - `seed=42` で deterministic、`seed=None` で stochastic 性確認
  - 1 utterance × 1000 step 訓練後の reverse sample が GT に近づくことを確認 (milestones.md §M3.5)
  - `eta=1.0` / `eta=0.0` で品質比較 (DDPM vs DDIM)、smoke 内で両方走らせて MR-STFT loss を pin (M3 phase review)

#### T-M4.3 (RTF measurement) へ
- **使用方法 (M3 phase review: `model.synthesize` で GAN/Diff 横断統一)**:
  ```python
  from wavenext2.inference.infer_diff import eval_mode

  with eval_mode(model):  # eval_mode helper を使う (副作用なし)
      # GAN/Diff 横断で `model.synthesize(mel)` を測定 (両モデルで API 統一)
      # warmup 用に外部 Generator を注入し global RNG 非汚染
      warmup_gen = torch.Generator(device).manual_seed(0)
      for _ in range(warmup_steps):
          _ = model.synthesize(mel, seed=warmup_gen)
      torch.cuda.synchronize()  # GPU 測定時
      start = time.perf_counter()
      for _ in range(measure_steps):
          _ = model.synthesize(mel)  # 本番測定は stochastic で OK
      torch.cuda.synchronize()
      rtf = (time.perf_counter() - start) / measure_steps / (audio_length / sr)
  ```
- **重要事項**:
  - `model.eval()` を必ず呼ぶ、または `eval_mode(model)` context manager を使う (本チケット §6.1)
  - **`model.synthesize(mel)` で GAN/Diff 横断計測** (M3 phase review): T-M2.4 GAN と T-M3.3 Diff の API を `synthesize` で統一、T-M4.3 が両モデルで同じコードで RTF 測定可能
  - `@torch.no_grad()` は内部で適用済み (二重 wrap でも問題なし)
  - CPU 測定は `torch.set_num_threads(1)`、GPU 測定は `torch.cuda.synchronize()` で正確に
  - 論文 RTF (Diff 4-step): A100 / 1-core CPU の値は論文 Table 2 と比較
  - **`@torch.inference_mode()` 化**: 5%+ 速度向上が見られたら本チケットを再オープン (§8.1 案 6)
  - **外部 Generator 注入で global RNG 非汚染** (M3 phase review): warmup phase で `torch.Generator` を注入し、`torch.initial_seed()` を変更しないこと

#### T-M3.1 (DiffWaveNext2) へ (双方向同期)
- **期待 attribute**:
  - `NOISE_SCHEDULE_ABAR: Tensor shape (4,)` (buffer 推奨で device auto sync)
  - `sub_models: nn.ModuleList[SubModelDiff]` (length 4、0-indexed)
  - `hop_length: int = 256`
- **期待 signature**:
  - `model.sub_models[k](mel, x_t, c_t) -> eps_pred (B, T_audio)`、`c_t.shape == (B,)` (1-D)
- **本チケットが consumer**: 不整合があれば T-M3.1 を修正

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M3.3 の Acceptance チェックボックス 2 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M3.3 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/training.md` §4.2 の擬似コードを本実装に揃えて update (1-indexed / 0-indexed 表記の混乱解消)
  - [ ] (該当時) `docs/architecture.md` §5 の dispatch 表に「sub-model index = 0-indexed `[t-1]`」の注記追加

### 9.3 Open question として残ったもの
- **解決できなかった疑問**:
  - **post-filter 適用時の clip タイミング** (T-M3.4 で決定): 本関数末尾の clamp(-1, 1) を post-filter 前に適用するか、post-filter 後に再度 clamp するかは T-M3.4 で決定 (§8.1 案 8 `clip_output` 引数の要否)
  - **`model.eval()` 自動切替の妥当性** (M6.2 訓練後の推論時に判断): 事故が起きたら `force_eval: bool` 引数追加 (§8.1 案 5)
  - **`@torch.inference_mode()` への変更** (T-M4.3 RTF 測定時に判断): 5%+ 速度向上があれば変更
  - **batch ごとに別 seed** (M6.2 batch 推論時に判断): 必要性が出たら `seed: int | list[int]` に拡張
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - DDPM vs DDIM (M5.2 smoke 発散時) — `eta: float` を v1 採用済みなので CLI 切替可能 (eta=1.0/0.0)
  - 4-step 以外の step 数 (M6.3 ablation)
  - `force_eval` 引数追加 (M6.2 訓練後の推論時) — `eval_mode` context manager を v1 採用済み
  - `clip_output` 引数追加 (T-M3.4 実装時)
  - `return_intermediates` 引数追加 (T-M3.5 smoke / 可視化要件発生時)
  - CLI entry point 追加 (T-M6.2 推論パイプライン構築時)
  - `torch.compile` 適用 (T-M4.3 RTF 測定後)
  - `DiffWaveNext2.synthesize = reverse_sample` thin alias — **M3 phase review で v1 採用** (T-M3.1 連携)
  - `variance_type: Literal["small", "large"]` (M3.5 smoke 後の品質確認時、§8.1 検討追加)
  - `post_filter: np.ndarray | None` 引数化 (T-M3.4 実装時、§9.1 T-M3.4 連絡)
- **`docs/open-questions.md` への追記要否**: 不要 (DDPM 形式 / 4-step schedule / 1-to-1 dispatch / post-filter スコープ分離は既に確定済み、`docs/open-questions.md` は論文事実確認の場であり実装判断は ticket 内で解決)
