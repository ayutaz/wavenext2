---
id: T-M2.4
title: GAN-WaveNeXt 2 モデル (T sub-model 直列 + fixed-point iteration)
milestone: M2
phase: M2
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M1.6]
blocks: [T-M2.5, T-M4.3]
related_docs:
  - docs/milestones.md#m24-gan-モデル-srcwavenext2modelsgan_wavenext2py
  - docs/training.md
  - docs/architecture.md
  - docs/open-questions.md
---

# T-M2.4: GAN-WaveNeXt 2 モデル (T sub-model 直列 + fixed-point iteration)

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.4](../milestones.md#m24-gan-モデル-srcwavenext2modelsgan_wavenext2py)
> **依存**: [T-M1.6](T-M1.6-sub-model.md) / **後続**: [T-M2.5](T-M2.5-train-gan.md), [T-M4.3](T-M4.3-rtf.md)

## 1. タスク目的とゴール

### 目的
論文 Section 3.2 / Figure 1a に示される **fixed-point iteration による residual denoising** を、`T` 個の `SubModelGAN` を `nn.ModuleList` に直列に並べた 1 つの `nn.Module` (`GANWaveNext2`) として実装する。

これにより:
- T-M2.5 (`train_gan.py`) は `GANWaveNext2(T=4)` を instantiate して `model(mel, audio_length)` を呼ぶだけで、論文の fixed-point iteration が回り `y_0` が得られる
- T-M4.3 (RTF measurement) は同じ `GANWaveNext2(T=4)` で推論時間を測れる
- ablation (T=2, 3, 5) は `T` 引数を変えるだけで実行可能

論文「**iteration ごとに独立した sub-model を並べる (shared ではない)**」(Table 1 の T 線形パラメータ数から判明) という設計を **コードの依存グラフ上でも明示** する。

### ゴール
- [ ] `src/wavenext2/models/gan_wavenext2.py` に `GANWaveNext2(nn.Module)` クラスが実装され、`T=4` で `~59.94M` params (Table 1 整合) を持つ
- [ ] `forward(mel, audio_length=None, return_intermediates=False)` が `(B, T_audio)` (default) または `list[Tensor]` 長さ T+1 (return_intermediates=True) を返し、内部で `y_T = torch.zeros(...)` 初期化 + 逆順 `for t in range(T, 0, -1)` の fixed-point iteration を実行
- [ ] **`return_intermediates: bool = False` を v1 から導入**: `True` のとき `[y_T, y_{T-1}, ..., y_0]` の (T+1) 要素 list を返す (T-M2.5 で中間 y_t も loss 対象にする論文 §3.2 / Fig 1a の WaveFit ベース設計の核心、§6.1 critical 参照)
- [ ] **`audio_length=None` で auto-infer** (`mel.shape[2] * self.hop_length`) をサポートし、`x_gt.shape[-1]` を渡す T-M2.5 と shape mismatch を起こさない
- [ ] T-M1.6 で確定する **SubModelGAN の戻り値仕様 (n_t vs y_{t-1})** に従い `y` 更新式を pin (実装時点で `docs/architecture.md` §2 + `docs/training.md` §2 を再読して確定)
- [ ] `enable_grad_ckpt=True` を `__init__` 引数で受け取り、T 個の各 sub-model へ propagation
- [ ] forward + backward が動作し、すべての T 個 sub-model の全パラメータに grad が流れる
- [ ] `T=1` で sub-model 1 個分が直接波形 (または `y_T - n_T`) を出力することを確認 (degenerate case)
- [ ] `docs/milestones.md` §M2.4 Acceptance criteria 全 3 項目をクリア
- [ ] `tests/test_gan_wavenext2.py` が `uv run pytest tests/test_gan_wavenext2.py` で全 pass

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/models/gan_wavenext2.py`
  - `tests/test_gan_wavenext2.py`
- 編集:
  - `src/wavenext2/models/__init__.py` (`__all__` に `GANWaveNext2` を追加)
  - `docs/milestones.md` §M2.4 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` のステータス更新
  - `docs/tickets/T-M1.6-sub-model.md` (戻り値仕様確定後に docstring と §6.1 critical を update)

### 2.2 主要構造

#### import 形式
```python
from wavenext2.models.gan_wavenext2 import GANWaveNext2
```

#### クラスシグネチャ

```python
"""gan_wavenext2.py — T 個の SubModelGAN を直列化した GAN-WaveNeXt 2 generator.

論文 Fig 1a / Section 3.2 の fixed-point iteration を実装する。
- T 個の SubModelGAN を独立に保持 (重み共有なし、Table 1 のパラメータ数 T 線形比例から確定)
- 初期 y_T = torch.zeros(...) (論文「initial input noise isn't required」)
- 逆順イテレーション t = T, T-1, ..., 1 で residual denoising

戻り値仕様 (n_t vs y_{t-1}) は T-M1.6 SubModelGAN の確定に従う (§6.1 critical 参照)。
詳細は docs/architecture.md §2, docs/training.md §2 を参照。
"""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

from wavenext2.models.sub_model import SubModelGAN


class GANWaveNext2(nn.Module):
    """GAN-WaveNeXt 2 generator: T 個の SubModelGAN を直列化した fixed-point iterator."""

    def __init__(
        self,
        T: int = 4,
        sub_model_cfg: dict | None = None,
        enable_grad_ckpt: bool = False,
    ) -> None:
        """
        Args:
            T: iteration 回数 / sub-model 数。論文 Table 1 では T=2, 3, 4, 5 を比較、推奨 T=4。
            sub_model_cfg: SubModelGAN の __init__ に渡す dict。
                例: {"mel_channels": 128, "n_fft": 2048, "hop_length": 300, "win_length": 1200, ...}
            enable_grad_ckpt: 各 sub-model に gradient checkpointing を有効化するか
                (T 個直列で OOM 防止のため、24GB GPU では True 推奨。§6.1 参照)。
        """
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        self.T = T
        self.enable_grad_ckpt = enable_grad_ckpt
        sub_model_cfg = dict(sub_model_cfg or {})
        # gradient checkpointing flag を sub-model に propagation
        sub_model_cfg["enable_grad_ckpt"] = enable_grad_ckpt
        # factory 経由 (T-M1.6 §8.2 整合); 直接 __init__ も許容 (T-M1.6 完了時に決定)
        self.sub_models = nn.ModuleList([
            SubModelGAN.from_config(sub_model_cfg, mode="gan") for _ in range(T)
        ])
        # SubModelGAN の hop_length をキャッシュ (audio_length 検算用)
        self.hop_length: int = self.sub_models[0].hop_length

    def forward(
        self,
        mel: torch.Tensor,
        audio_length: int | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Fixed-point iteration による波形合成.

        Args:
            mel:                  (B, 128, T_mel) log-mel-spectrogram.
            audio_length:         出力波形長 (= T_mel * hop_length)。
                                   `None` のとき `mel.shape[2] * self.hop_length` で auto-infer
                                   (T-M1.3 で center=True 整合性確定後に式を再評価)。
            return_intermediates: `False` (default) → 最終 `y_0` のみ。
                                   `True` → `[y_T, y_{T-1}, ..., y_0]` の (T+1) 要素 list。
                                   T-M2.5 の fixed-point iteration loss (中間 y_t も loss 対象)
                                   に必須 (論文 §3.2 / Fig 1a の WaveFit ベース設計の核心、§6.1 critical 参照)。

        Returns:
            (B, audio_length) 合成波形 y_0 ∈ [-1, 1] (return_intermediates=False).
            または `list[Tensor]` 長さ T+1 (return_intermediates=True), 全要素 (B, audio_length).

        Algorithm:
            y_T = torch.zeros(B, audio_length)                # 論文: initial input noise isn't required
            (intermediates = [y_T])                            # return_intermediates=True 時のみ
            for t in range(T, 0, -1):
                out = self.sub_models[t-1](mel, y)
                # T-M1.6 で確定する戻り値仕様に従い、以下のいずれかを採用:
                #   (A) out = n_t (ノイズ residual)  → y = y - out
                #   (B) out = y_{t-1} (denoised wave) → y = out
                # ※ 実装時に §6.1 critical を解決して pin、却下案は §8 に残す。
                (intermediates.append(y))                      # return_intermediates=True 時のみ
            return y                                           # または intermediates
        """
        # Shape validation
        if mel.dim() != 3:
            raise ValueError(f"mel must be (B, 128, T_mel), got {tuple(mel.shape)}")
        B, _, T_mel = mel.shape
        expected = T_mel * self.hop_length
        if audio_length is None:
            audio_length = expected  # auto-infer (T-M1.3 で center=True 整合性確定後に再評価)
        elif audio_length != expected:
            raise ValueError(
                f"audio_length ({audio_length}) must equal T_mel * hop_length "
                f"({T_mel} * {self.hop_length} = {expected})"
            )

        y = torch.zeros(B, audio_length, device=mel.device, dtype=mel.dtype)
        intermediates: list[torch.Tensor] = [y] if return_intermediates else []
        # 逆順イテレーション (t = T → 1)。sub_models は 0-indexed なので t-1 で参照。
        for t in range(self.T, 0, -1):
            # gradient checkpointing は T 毎に挟む設計 (§8.1 案 5、use_reentrant=False)
            out = self.sub_models[t - 1](mel, y)
            # === T-M1.6 確定後に置換 ===
            # (A) out = n_t →  y = y - out  (※ 実装時に sub-model 側 clip を外す or ここで clip)
            # (B) out = y_{t-1} → y = out
            y = self._residual_update(y, out)
            if return_intermediates:
                intermediates.append(y)
        return intermediates if return_intermediates else y

    def _residual_update(self, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """sub-model 出力で y を更新する単一情報源 (SoT).

        T-M1.6 critical 解決後、ここに (A) `return y - out` または (B) `return out` を pin する。
        テスト (test_residual_update_semantics) でこの実装を検証。

        Note (§8.1 案追加): T-M1.6 の戻り値仕様が実装時にまだ未確定なら、本 method を
        `Callable[[Tensor, Tensor], Tensor]` 引数として `__init__` で受け取り、A/B 両方を
        unit test fixture で差し替え可能にする選択肢もあり (critical の解決を後送りでき、
        CI が grey にならない)。本チケット v1 では method 直接実装、後送りは §8 検討。
        """
        # === 実装時に T-M1.6 SubModelGAN docstring + docs/architecture.md §2 を再読して確定 ===
        raise NotImplementedError(
            "Pending T-M1.6 SubModelGAN return-value spec resolution. "
            "See §6.1 critical item."
        )
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `T` (sub-model 数 / iteration 回数) | 4 (推奨) / 2, 3, 5 (ablation) | docs/architecture.md §4, Table 1, docs/open-questions.md §A4 |
| 初期 `y_T` | `torch.zeros(B, audio_length)` | docs/open-questions.md §D, docs/training.md §2 「論文: initial input noise isn't required」 |
| sub-model 重み共有 | **しない** (T 個独立) | docs/architecture.md §4, Table 1 (29.97M=T2 / 44.96M=T3 / 59.94M=T4 / 74.93M=T5 が T 線形) |
| パラメータ数目標 (T=4) | ~59.94M | docs/architecture.md §4 (Table 1) |
| パラメータ数目標 (T=1, smoke) | ~14.99M | docs/architecture.md §4 (Table 1 / sub-model) |
| iteration 順序 | 逆順 `for t in range(T, 0, -1)` | docs/architecture.md §2, docs/training.md §2.1 |
| `audio_length` | `T_mel * hop_length` | docs/training.md §1.3, docs/architecture.md §3 |
| `enable_grad_ckpt` (default) | `False` (M2 smoke) / `True` (M5 / M6) | docs/milestones.md §リスクと緩和策、§6.1 (24GB GPU で必要) |

### 2.4 アルゴリズム / 処理フロー

#### `__init__`
1. `T >= 1` を assert
2. `sub_model_cfg` を copy して `enable_grad_ckpt` を埋め込み (immutable な default 取り扱い)
3. `SubModelGAN.from_config(sub_model_cfg, mode="gan")` を T 回呼び `nn.ModuleList` に格納
4. `self.hop_length = self.sub_models[0].hop_length` を cache (audio_length 検算用)

#### `forward(mel, audio_length)`
1. shape 検証 (`mel.dim() == 3`, `audio_length == T_mel * hop_length`)
2. `y = torch.zeros(B, audio_length, device=mel.device, dtype=mel.dtype)` で初期化
3. 逆順 for ループ `for t in range(T, 0, -1)`:
   - `out = self.sub_models[t-1](mel, y)`
   - `y = self._residual_update(y, out)`
4. `return y` (= y_0)

#### `_residual_update(y, out)` (T-M1.6 確定後に pin)
- パターン (A): `return y - out` (sub-model 戻り値が **n_t = ノイズ residual** の場合、論文 Fig 1a `y_{t-1} = y_t - n_t` 表記の素直な実装)
- パターン (B): `return out` (sub-model 戻り値が **y_{t-1} = denoised waveform** の場合、generator 出力が直接波形)

#### 不変条件
- `audio_length` は forward 中で変化しない (各 sub-model は同じ長さの y を入出力)
- `y.device == mel.device`, `y.dtype == mel.dtype` を全 iteration で維持
- 各 sub-model は state を持たない (T-M1.6 §6.3) ため、forward を 2 回呼んでも干渉しない

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `gan_wavenext2.py` 実装 + `tests/test_gan_wavenext2.py` 記述 | general-purpose |
| Reviewer | 1 | 論文 Fig 1a / `docs/architecture.md` §2 / `docs/training.md` §2 整合性確認 + T-M1.6 SubModelGAN docstring との同期確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_gan_wavenext2.py -v` + memray で `enable_grad_ckpt` 動作確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **partial** (T-M2.2 Discriminator / T-M2.3 Losses とは並列可能、T-M2.1 Dataset とも並列可能)
- 並列実行する場合の最大並列数: 3 (T-M2.2, T-M2.3, T-M2.4 を 3 並列)
- T-M2.5 (train_gan) は本チケット + T-M2.1 + T-M2.2 + T-M2.3 すべて完了後に着手

## 4. 提供範囲 (Scope)

### In Scope
- `class GANWaveNext2(nn.Module)` の実装
- T 個の `SubModelGAN` を `nn.ModuleList` で保持 (重み非共有)
- `forward(mel, audio_length=None, return_intermediates=False)` の fixed-point iteration (逆順 T → 1)
  - **`return_intermediates: bool = False` 引数を v1 から導入** (§8.1 採用昇格、§6.1 critical 参照)
  - **`audio_length=None` で auto-infer** (`mel.shape[2] * self.hop_length`)
- 初期 `y_T = torch.zeros(...)` の生成
- `enable_grad_ckpt` 引数 + 各 sub-model への propagation
- `_residual_update(y, out)` メソッドに T-M1.6 確定の戻り値仕様を pin
- shape 検証 (`mel.dim() == 3`, `audio_length == T_mel * hop_length` or `None`)
- `tests/test_gan_wavenext2.py` (5.1 / 5.3 すべて)
- パラメータ数検証テスト (T=1, 2, 3, 4, 5 すべて)
- `__init__.py` への `__all__` 追加
- docstring (英文 + 日本語)

### Out of Scope
- 訓練ループ / optimizer / scheduler (T-M2.5)
- Discriminator / Loss 関数 (T-M2.2 / T-M2.3)
- TensorBoard / checkpoint 保存 (T-M2.5)
- 推論専用 API (推論時も `forward` を流用、別 API は不要)
- post-filter (Diff のみ、T-M3.4)
- mixed precision / torch.compile (M5 / M6 で必要時)
- 中間 `y_t` の loss 計算 / stop_gradient 制御 (T-M2.5 の責務; 本チケットは `return_intermediates=True` で list を返すまでに留め、loss 設計と stop_gradient 挿入は T-M2.5)
- ablation 実験スクリプト (T-M6.3)
- LibriTTS-R real audio との e2e (T-M5.1 で 1 epoch smoke)
- `BaseVocoder` / `IterativeVocoder` 親クラス抽出 (T-M3.1 着手前に判断、§8.2 参照)

### Deliverable
- ファイル:
  - `src/wavenext2/models/gan_wavenext2.py` (新規)
  - `tests/test_gan_wavenext2.py` (新規)
  - `src/wavenext2/models/__init__.py` (re-export 追加)
- 関数 / クラス:
  - `class GANWaveNext2(nn.Module)`
  - `_residual_update(y, out)` method
- ドキュメント差分:
  - `docs/milestones.md` §M2.4 の Acceptance チェック
  - `docs/tickets/index.md` の T-M2.4 ステータス
  - `docs/tickets/T-M1.6-sub-model.md` (戻り値仕様確定箇所の update)

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_gan_wavenext2.py`)

#### パラメータ数
- [ ] `test_param_count_T4`: `T=4` で `sum(p.numel() for p in model.parameters()) ≈ 59.94M ± 0.2M` (Table 1)
- [ ] `test_param_count_T2`: `T=2` で `≈ 29.97M ± 0.1M`
- [ ] `test_param_count_T3`: `T=3` で `≈ 44.96M ± 0.15M`
- [ ] `test_param_count_T5`: `T=5` で `≈ 74.93M ± 0.25M`
- [ ] `test_param_count_T1`: `T=1` で `≈ 14.99M ± 0.05M` (= sub-model 1 個分)
- [ ] `test_param_count_linear_in_T`: T を変えてパラメータ数が **線形** であること (回帰直線の R^2 > 0.999)

#### Forward / Backward
- [ ] `test_forward_shape`: `B=2`, `T_mel=80`, `audio_length=24000` (= 80×300) で `out.shape == (2, 24000)` (milestones.md §M2.4 Acceptance #1〜2)
- [ ] `test_forward_output_range`: 出力が `[-1, 1]` に収まる (sub-model 内 clip)
- [ ] `test_backward_all_params_have_grad`: `model(mel, audio_length).sum().backward()` で **T 個 sub-model すべて** の全 parameter に `grad is not None` (milestones.md §M2.4 Acceptance #2)
- [ ] `test_backward_T4_all_4_submodels`: T=4 で `for sm in model.sub_models: sm.parameters() に全 grad あり` (各 sub-model の独立性確認)

#### T=1 degenerate case
- [ ] `test_T1_equals_single_sub_model`: `T=1` で forward が `y_0 = _residual_update(zeros, sub_model(mel, zeros))` と等価。具体的には:
  - パターン (A) なら `y_0 == -sub_model(mel, zeros)` (`zeros - n_T = -n_T`)
  - パターン (B) なら `y_0 == sub_model(mel, zeros)` (generator が直接波形を出力)
  - 確定したパターンに従う 1 種類を pin (milestones.md §M2.4 Acceptance #3)

#### Determinism
- [ ] `test_deterministic`: 同じ mel + 同じ seed (`torch.manual_seed(42)`) で 2 回 forward して `torch.allclose(out1, out2)` (sub-model 内に dropout なし、`y_T = zeros` で固定)

#### `enable_grad_ckpt` propagation
- [ ] `test_enable_grad_ckpt_propagation`: `GANWaveNext2(T=4, enable_grad_ckpt=True)` で各 sub-model の `enable_grad_ckpt` 属性 (T-M1.4 経由) が `True`
- [ ] `test_enable_grad_ckpt_memory_reduction` (`@pytest.mark.slow`): `memray` で `enable_grad_ckpt=False` vs `True` の peak memory を計測、True の方が少ない (例: 20% 以上削減)。CPU で実施可能 (relative 比較なら GPU 不要)
- [ ] `test_enable_grad_ckpt_output_equivalence`: `enable_grad_ckpt=True` と `False` で同じ入力 + 同じ seed の forward 出力が一致 (gradient checkpointing は数値的 noop)

#### Shape validation
- [ ] `test_audio_length_mismatch_raises`: `audio_length = T_mel * hop_length + 1` で `ValueError`
- [ ] `test_invalid_mel_dim_raises`: `mel.dim() != 3` で `ValueError`
- [ ] `test_invalid_T_raises`: `T=0` や `T=-1` で `ValueError` (`__init__` 内)

#### Residual update semantics (T-M1.6 確定後)
- [ ] `test_residual_update_semantics`: `_residual_update(y, out)` の振る舞いが T-M1.6 確定パターンと一致 (e.g., パターン (A) なら `_residual_update(y, out) == y - out`)

#### `return_intermediates` (v1 必須)
- [ ] `test_return_intermediates`: `return_intermediates=True` で `[y_T, y_{T-1}, ..., y_0]` 長さ (T+1) の list が返り、各 shape が `(B, audio_length)` で一致
- [ ] `test_return_intermediates_T4`: T=4 で list 長さ == 5、`out[0]` が `torch.zeros` (= y_T)、`out[-1]` が最終 y_0 (= `forward(..., return_intermediates=False)` の出力と allclose)
- [ ] `test_return_intermediates_grad_flow`: `return_intermediates=True` で取った各中間 y_t に対し `y_t.sum().backward(retain_graph=True)` で対応する sub-model の全 param に grad が流れる (中間 y_t loss 対象設計の動作確認)

#### `audio_length=None` auto-infer
- [ ] `test_audio_length_auto_infer`: `audio_length=None` で呼んだ結果が `audio_length=T_mel*hop_length` 明示渡しと allclose
- [ ] `test_audio_length_x_gt_compat`: T-M2.5 想定の `x_gt.shape[-1]` を渡しても、auto-infer 結果と一致する (`x_gt.shape[-1] == T_mel * hop_length` の前提)

#### T=1 clip range (パターン A 採用時の懸念解消)
- [ ] `test_T1_clip_range`: T=1 で `y_0` が `[-1, 1]` を維持する設計を pin。
  - パターン (A) 採用なら sub-model 戻り値 `n_T` は clip(-1,1) しない設計 (residual はそのスケール外を取りうる) で、本チケットの `_residual_update` または return 直前で `y.clamp(-1, 1)` を行う
  - パターン (B) 採用なら sub-model が clip(-1,1) で `y_0` も `[-1, 1]`、追加 clip 不要
  - assertion: `assert y_0.min() >= -1.0 and y_0.max() <= 1.0` (パターンに関わらず満たすこと)

### 5.2 e2e / 結合テスト
- [ ] `test_gan_wavenext2_with_real_sub_model_cfg`: `configs/gan_wavenext2.yaml` の sub_model_cfg を読んで `GANWaveNext2.from_config(...)` 経由 (factory 追加した場合) で生成、forward が動作 (T-M2.5 の YAML 読み込み経路を模す)
- [ ] `test_module_list_independence`: `model.sub_models[0]` と `model.sub_models[1]` が **異なるオブジェクト** であり、片方の weight 変更がもう片方に伝播しないこと
- [ ] `test_real_audio` (`@pytest.mark.slow`): LibriTTS-R sample 1 件で実 mel 抽出 → `GANWaveNext2(T=4).forward(...)` → backward が通る。T-M0.3 完了後、T-M2.1 dataset 未完成期間は `pytest.skip`

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.4 より転記)
- [ ] パラメータ数が Table 1 と一致 (T=4 で ~59.94M)
- [ ] forward → backward が動作
- [ ] T=1 で `y_0 = ±sub_model(mel, zeros)`、つまり generator が直接波形を出力する形になっていることを確認

### 5.4 追加 acceptance (本チケット独自)
- [ ] `pytest tests/test_gan_wavenext2.py` が exit code 0 (slow / gpu マーカー除く)
- [ ] `from wavenext2.models import GANWaveNext2` で import 可能
- [ ] `_residual_update` が確定パターン (A) or (B) で実装され、もう一方は §8 に却下根拠付きで残る
- [ ] T-M1.6 SubModelGAN docstring と本チケットの戻り値仕様記述が **同期** している (両方を update 済み)

### 5.5 テスト戦略
- **scope="module" fixture**: `GANWaveNext2(T=4)` の init は ~60M params で重いため、fixture で再利用 (各テストで init し直さない)
- **`@pytest.mark.slow`**: memray メモリ測定 / 実音声テストはデフォルト除外
- **CI 時間目標**: T-M2.4 全テスト合計 **~30 秒以内** (T=4 forward+backward が CPU で ~10 秒 × 3 ケース程度)
- **coverage 目標**: 85% (実音声 e2e は T-M5.1 でカバー)

## 6. 懸念事項

### 6.1 技術的リスク

#### CRITICAL: SubModelGAN の戻り値仕様 (n_t vs y_{t-1}) 確定 (T-M1.6 から伝搬)

- **問題**: T-M1.6 §6.1 で指摘されている通り、`SubModelGAN.forward(mel, y_prev)` の戻り値が:
  - パターン (A): **n_t (ノイズ residual)** で、外側 (= 本チケット) で `y = y - n_t` を実行
  - パターン (B): **y_{t-1} (denoised waveform)** で、外側で `y = out` 代入のみ
  - のどちらかが未確定
- **論文記述の整合性**:
  - `docs/architecture.md` §2「**出力が波形そのものではなくノイズ成分** になっている点が WaveNeXt 2 の改変点」→ パターン (A) を示唆
  - `docs/training.md` §2.1 pseudo-code: `n_t = sub_model[t](mel, y_t); y_t = y_t - n_t` → パターン (A) を示唆
  - 一方 T-M1.6 `test_gan_output_range` は出力が `[-1, 1]` clip を期待 → n_t ∈ [-1, 1] だと residual のスケールに不自然 (パターン (B) 寄り)
  - WaveFit (`yukara-ikemiya/wavefit-pytorch`) の sub-model 戻り値仕様を **本チケット実装着手時に必ず再確認** (参考実装はコピーしないが、戻り値仕様の判断材料としては利用可)
- **本チケット実装着手時の MUST DO**:
  1. `docs/architecture.md` §2 / §4 を再読
  2. `docs/training.md` §2.1 pseudo-code を再読
  3. T-M1.6 SubModelGAN の docstring (戻り値仕様) を確認、T-M1.6 がまだ未確定なら **T-M1.6 を先に解決**
  4. `_residual_update` の実装を (A) or (B) で **pin**
  5. T-M1.6 SubModelGAN docstring と本チケットの両方を **update** (片方だけ update すると不整合)
  6. もう一方のパターンを §8 に **却下根拠付きで残す**
- **未解決のまま実装すると**: 訓練が divergent (loss が下がらない、または NaN) で M2.6 smoke が通らない

#### CRITICAL (追加): T=1 degenerate test の clip 範囲問題 (パターン A 採用時)

- **問題**: §6.1 critical で確定する戻り値仕様が **パターン (A) (residual n_t)** の場合、T=1 で `y_0 = zeros - n_T = -n_T` となる。sub-model 内の clip(-1,1) は **denoised output** に対するものなので、residual `n_t` はそのスケール外を取りうる
  - 例: 真の波形が `[-0.5, 0.5]` 範囲なら `n_T = zeros - y_0 = [0.5, -0.5]` で範囲内だが、`n_T` が `[-1.5, 1.5]` に達することも構造上ありうる
  - その場合 `test_T1_equals_single_sub_model` / `test_forward_output_range` の `assert y_0 ∈ [-1, 1]` が **成立しない懸念**
- **解決策 (パターン (A) 採用時)**:
  - (a) sub-model 側 clip(-1,1) を外し、本チケットの `_residual_update` 内 (= `return y - out`) または return 直前で `y.clamp(-1, 1)` を行う設計に変更
  - (b) T-M1.6 の sub-model docstring を「`n_t` は無制約 (clip なし)」と pin、本チケットで `y.clamp(-1, 1)` を return 前に挿入
- **解決策 (パターン (B) 採用時)**: sub-model が `y_{t-1}` を直接出力し clip(-1,1) を内部で行うため、本チケットで追加 clip 不要
- **本チケット実装着手時の MUST DO**: §6.1 critical 解決後、パターンに応じて clip 位置を決定し `test_T1_clip_range` で動作確認

#### CRITICAL (追加, Critical 寄り): `best.pt` race condition (T-M2.5 への要請)

- **問題**: val improvement 頻発期 (M5.1 1 epoch では起きやすい) に複数プロセス / 複数 step が `best.pt` を同時書き込みすると **破損 / 部分書き込み** リスク
- **責務分離**: 本チケットは model 実装のみ、checkpoint 保存は T-M2.5 だが **本チケット §9 から明示要請**:
  - **atomic rename pattern**: `torch.save(state, "best.pt.tmp")` → `os.replace("best.pt.tmp", "best.pt")` (`os.replace` は POSIX/Windows 共通でアトミック)
  - DDP / multi-process では rank0 のみが書き込む `if rank == 0:` ガード必須
  - 48000 ファイル checkpoint 削除のレース (2M step で 10k step ごと save = 200 ckpt、`keep_last_n=5` で削除する際の filesystem race) も同様に rank0 only で対応

#### 通常項目

- **T 個直列の OOM リスク** (M2 phase):
  - segment_length=16384, B=16, fp32 で activation memory ≈ T × 14.99M × forward 中間値
  - A100 40GB では T=4 で問題なく動作 (HiFi-GAN / WaveFit 慣例)
  - 24GB GPU (RTX 3090 / 4090) では **borderline**。`enable_grad_ckpt=True` で T-M1.4 の checkpoint を経由してメモリ削減 (一般に 20〜40% 削減期待)
  - **対応**: `enable_grad_ckpt=True` を smoke 失敗時に first action として有効化、`memray` でメモリ profiling
  - **代替**: `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` を **T 毎の sub-model 呼び出しで挟む** (§8.1 採用昇格)。PyTorch 2.10+ で安定し、`checkpoint_sequential` より柔軟 (M6 OOM 対策の現実解)
  - **検知**: M2.6 smoke / M5.1 1 epoch で OOM ログを `torch.cuda.OutOfMemoryError` で捕捉

- **T 個 sub-model の重み独立 vs shared** (論文 Table 1 で確定済):
  - Table 1: 29.97M (T=2) / 44.96M (T=3) / 59.94M (T=4) / 74.93M (T=5) は T 線形 → **重み非共有**
  - 仮に shared なら全 T で同じ 14.99M params になるはず
  - 本チケットは **重み非共有** で実装 (`nn.ModuleList` で T 個独立 instance)
  - shared 案は §8 で却下根拠を明示

- **fixed-point iteration の loss 設計 (中間 y_t の扱い)** (T-M2.5 で実装):
  - **本チケット v1 で `return_intermediates: bool = False` を導入** (§8.1 採用昇格)。論文 §3.2 / Fig 1a の WaveFit ベース fixed-point iteration では中間 y_t に loss をかけないと sub-model 間の役割分担が崩れる (全 sub-model が「最終 y_0 に向かう」最適化になり residual denoising 構造が失われる)
  - `docs/training.md` §2.1 の式 `loss_G = MR-STFT(y_0) + λ_adv * adv(y_0) + λ_fm * fm(y_0)` は **y_0 のみ** だが、中間 y_t loss を取らないと WaveFit-original の挙動と乖離 (T-M2.5 で再評価)
  - 中間 `y_t (t > 0)` を loss 対象にする場合、stop_gradient を適切に挿入しないと勾配が T 倍に増幅 (連鎖 backward の累積) → **本チケットの forward は intermediates list を返すのみ、stop_gradient 挿入は T-M2.5 の責務**
  - T-M2.5 で必要になってから戻すのは遅い (API change で migration cost) ため、本チケット v1 で対応

- **`audio_length` の center=True 整合性** (T-M1.6 §6.1 critical から伝搬):
  - `torch.stft(center=True)` で `T_mel = floor(T_audio / hop) + 1` の規約により `T_audio = (T_mel - 1) * hop_length` の可能性
  - T-M1.3 / T-M1.6 で確定する `T_audio = T_mel * hop_length` か `(T_mel-1) * hop_length` のどちらかに **本チケットも揃える**
  - 不整合だと shape mismatch で silent failure

- **`torch.zeros_like(x_gt)` vs `torch.zeros(B, audio_length)`**:
  - T-M2.5 で `model(mel, x_gt.shape[1])` と呼べば `x_gt` の長さに自動追従、`torch.zeros_like(x_gt)` 相当の挙動を実現可能
  - ただし `x_gt` を forward に渡すと「ground-truth を generator が見ている」誤解を招くため、本チケットでは **`audio_length: int` を明示引数** にする
  - 別案: `forward(mel)` のみで内部で `audio_length = mel.shape[2] * self.hop_length` 計算 → 採用 (§8 案 6 参照、shape mismatch リスク減)。**本チケット v1 では `audio_length` 引数を採用**、v2 で省略可能化を検討

- **factory パターン (`SubModelGAN.from_config`) の存在前提**:
  - T-M1.6 §8.2 で factory パターン採用が決定済みだが、T-M1.6 完了時に実装されていることを確認
  - 未実装なら直接 `SubModelGAN(**sub_model_cfg)` で代替 (本チケット実装時のフォールバック)

- **`enable_grad_ckpt` の T-M1.4 / T-M1.6 サポート確認**:
  - T-M1.4 (WaveNextGenerator) で `enable_grad_ckpt` 引数を受け取る設計が T-M1.6 §6.1 で要請済み
  - T-M1.6 SubModelGAN.from_config が `enable_grad_ckpt` を受理しない場合は T-M1.4 / T-M1.6 のリビジョンを要求 (本チケットは consumer)

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目:
  - **GAN fixed-point 初期化**: ✅ `torch.zeros_like(x_gt)` で確定 (§D)、本チケット採用
  - **T sub-model の重み非共有**: ✅ Table 1 で確定、本チケット採用
  - **iteration 順序 (逆順 T → 1)**: ✅ `docs/architecture.md` §2 / `docs/training.md` §2.1 で確定
  - **戻り値 (n_t vs y_{t-1})**: ⏸ T-M1.6 と本チケットで確定 (§6.1 critical)
  - **中間 y_t の loss 対象**: ⏸ T-M2.5 で確定 (本チケットでは forward のみ)
- 旧来の判断ポイント (本チケット作成時に確定):
  - **`forward` の引数 (mel, audio_length) vs (mel, x_gt)**: `audio_length: int` を採用 (§6.1 / §8 案 6)
  - **`nn.ModuleList` vs `nn.Sequential`**: `nn.ModuleList` を採用 (`nn.Sequential` は forward が 1 引数固定で、`(mel, y)` の 2 引数を取れない)
  - **forward が中間 y_t を返すか**: 返さない (最終 y_0 のみ)。T-M2.5 で必要なら option 追加

### 6.3 他チケットとの整合性

- **T-M1.6 (SubModelGAN)** との整合:
  - 期待 signature: `sub_model(mel, y_prev) -> (B, T_audio)`
  - 期待 attribute: `hop_length`, `enable_grad_ckpt` (T-M1.4 経由)
  - 期待 classmethod: `SubModelGAN.from_config(cfg, mode="gan")` (T-M1.6 §8.2 採用済)
  - **戻り値仕様**: §6.1 critical 参照、T-M1.6 と本チケットで **両方 update**
  - 不整合があれば T-M1.6 を修正 (T-M2.4 は consumer)

- **T-M2.5 (train_gan)** へ渡す情報 (§9.1 参照):
  - YAML config: `T: 4`, `sub_model_cfg: {...}`, `enable_grad_ckpt: false` (smoke) / `true` (full)
  - 訓練時の呼び出し: `y_0 = model(mel, x_gt.shape[1])`
  - 中間 y_t を loss に使う設計を採用する場合は本チケットに forward option を追加要請

- **T-M4.3 (RTF measurement)** へ渡す情報:
  - `model.eval()` で deterministic 推論 (本チケットは dropout を含まないが念のため `.eval()` 呼び出しを推奨)
  - `audio_length` を test set 平均長に設定して測定
  - `torch.set_num_threads(1)` で CPU 1-core 推論時間を測定

- **T-M2.6 (smoke training)** へ渡す情報:
  - 1 utterance overfitting で `MR-STFT loss < 初期値の 10%` を期待 (`docs/milestones.md` §M2.6)
  - 失敗時の first action: `enable_grad_ckpt=True` で再試行 / `T=1` に減らして sub-model 単体の品質を確認

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §2, §4 (Table 1 パラメータ数) と整合
- [ ] `docs/training.md` §2.1 pseudo-code (逆順 iteration, `y_T = zeros`) と整合
- [ ] `docs/open-questions.md` §D (GAN fixed-point 初期化 = zeros) と整合
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 全クリア
- [ ] パラメータ数が T=4 で ~59.94M (Table 1 ±0.2M)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: `T<1`, `mel.dim()!=3`, `audio_length` 不整合で `ValueError` (silent failure なし)
- [ ] **参考実装 (WaveFit-PT / Vocos) をコピーしていない** (CLAUDE.md 末尾の方針、`docs/training.md` §2 の論述から再構成)
- [ ] `__init__.py` の `__all__` に `GANWaveNext2` が追加されている
- [ ] `_residual_update` のパターン (A) / (B) が **1 つに pin** され、もう一方は §8 に却下根拠付きで残る
- [ ] T-M1.6 SubModelGAN docstring と本チケットの **戻り値仕様記述が同期**
- [ ] `enable_grad_ckpt` が各 sub-model へ正しく propagation
- [ ] `nn.ModuleList` で T 個 sub-model が **独立 instance** (weight 非共有)
- [ ] forward に副作用がない (state を持たない、再現性のため)
- [ ] CPU でテストが pass する (CUDA 不要)
- [ ] 中間 y_t の stop_gradient 制御を **本チケットでは行っていない** (T-M2.5 の責務)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M2) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **T 個の独立 `SubModelGAN` を `nn.ModuleList` で保持、forward で逆順 `for t in range(T, 0, -1)` ループ、初期 `y_T = torch.zeros(B, audio_length)`、`_residual_update` で T-M1.6 確定パターンを pin**
  - 理由:
    - (a) `nn.Sequential` は forward の 1 引数固定で 2 引数 `(mel, y)` を取れない
    - (b) `nn.ModuleList` で各 sub-model のパラメータ数を `parameters()` で取得可能、checkpoint 保存・復元・partial freeze が容易
    - (c) 重み非共有は `docs/architecture.md` §4 / Table 1 でパラメータ数が T 線形であることから確定
    - (d) 逆順 iteration は `docs/training.md` §2.1 / `docs/architecture.md` §2 で確定
    - (e) `_residual_update` を独立 method にすることで、(A) / (B) パターン切り替え時の変更箇所が **1 行に局所化**

- **`return_intermediates: bool = False` 引数を v1 から導入** (採用昇格 / 重要):
  - 理由: 論文 §3.2 / Fig 1a の WaveFit ベース fixed-point iteration では、中間 y_t に loss をかけないと sub-model 間の役割分担が崩れる (全 sub-model が「最終 y_0 に向かう」最適化になり residual denoising 構造が失われる)
  - `forward(mel, audio_length=None, return_intermediates=False)` で `False` のとき y_0 のみ、`True` のとき `[y_T, y_{T-1}, ..., y_0]` (長さ T+1) を返す
  - T-M2.5 で必要になってから戻すのは遅い (API change で migration cost)、本チケット v1 で対応
  - 副作用: API が広がるが、default が False のため既存呼び出しは無変更

- **`audio_length=None` で auto-infer** (採用昇格):
  - `mel.shape[2] * self.hop_length` または `(mel.shape[2] - 1) * hop_length` を default (T-M1.3 center=True 整合性確定後に式を pin)
  - T-M2.5 が `x_gt.shape[-1]` を渡す現状設計で shape mismatch リスク減
  - 明示的に `audio_length` を渡すケース (推論時の任意長合成) も従来通り動く

- **`checkpoint(..., use_reentrant=False)` を T 毎に挟む** (採用昇格):
  - PyTorch 2.10+ で安定、`checkpoint_sequential` より柔軟 (`nn.ModuleList` で直接利用可能)
  - M6 OOM 対策の現実解 (T-M1.4 block 内 checkpoint と組み合わせて memory 削減を多段化可能)
  - `enable_grad_ckpt=True` のとき本チケット forward 内でも `torch.utils.checkpoint.checkpoint(self.sub_models[t-1], mel, y, use_reentrant=False)` を呼ぶ実装に拡張

- **`_residual_update` を `Callable[[Tensor, Tensor], Tensor]` 引数化** (検討追加):
  - T-M1.6 戻り値仕様 (A 説 n_t / B 説 y_{t-1}) が実装時に未確定なら、両方の callable を unit test で fixture 化して critical 解決を後送り可能
  - CI が grey にならない (テストで A/B 両方を差し替え実行可能)
  - 採用条件: T-M1.6 が本チケット着手時に未確定の場合のみ、本チケットの `__init__(..., residual_update_fn: Callable | None = None)` を追加し、`None` のとき method デフォルトを使用

#### Deprecated (却下案)

1. **shared sub-model (T 個で同じパラメータ使い回し)**
   - メリット: パラメータ数を T で割れる (`T=4` で ~15M)
   - 却下: **Table 1 で却下確定**。パラメータ数が T=2/3/4/5 で線形に増加 (29.97M / 44.96M / 59.94M / 74.93M) しているため、論文は **重み非共有**。shared にすると論文の vehicle が破綻
   - 再現実装で shared を試す場合は M6.3 ablation で別個に評価

2. **T=2, 3, 5 への変更 (ablation)**
   - メリット: 速度 / 品質トレードオフを探索
   - 却下: 本チケットでは推奨 T=4 を default にし、`T: int = 4` 引数で他値も対応。**ablation は M6.3 で別チケット化**

3. **初期入力を `zeros` ではなく `ε ~ N(0, σ²)` (WaveFit-original 流)**
   - メリット: WaveFit と完全同一の挙動、初期入力 noise の多様性
   - 却下: 論文「initial input noise isn't required」明記 (`docs/open-questions.md` §D)、`docs/training.md` §2.2 でも「初期入力ノイズの分布は単純化」と明記。zeros の方が**論文準拠**かつ実装シンプル。ablation したい場合は引数化 `initial_noise: Literal["zeros", "gaussian"] = "zeros"` を将来追加可

4. **`nn.Sequential` で T 個並べる**
   - メリット: forward が `self.sub_models(x)` の 1 行で済む
   - 却下: 上記 (a)、forward が 2 引数 `(mel, y)` を取れない (`nn.Sequential` は 1 引数固定)。`nn.ModuleList` の方が iteration 制御が明示的

5. **forward を `torch.utils.checkpoint.checkpoint_sequential` で 4 segment 化**
   - メリット: T-M1.4 の `enable_grad_ckpt` (block 内 checkpoint) と組み合わせて activation memory を更に削減
   - 却下 (本チケット v1): `checkpoint_sequential` は `nn.Sequential` 専用で `nn.ModuleList` で使うには wrapper が必要、複雑度増加
   - **採用代替**: `torch.utils.checkpoint.checkpoint(sub_model, mel, y, use_reentrant=False)` を T 毎に直接挟む方式を採用昇格 (上記「採用設計」参照)。`checkpoint_sequential` より柔軟

6. **`forward(mel)` のみで内部で `audio_length = mel.shape[2] * self.hop_length` 計算**
   - メリット: 呼び出し側の負担減、shape mismatch リスク減
   - **採用昇格**: 本チケット v1 で `audio_length: int | None = None` (None なら自動計算) を採用 (上記「採用設計」参照)。T-M2.5 が `x_gt.shape[-1]` を渡す現状設計で shape mismatch リスク減

7. **forward が中間 `y_t (t=1..T-1)` も `list[torch.Tensor]` で返す**
   - メリット: T-M2.5 で中間 y_t も loss 対象にする場合に必要 (論文 §3.2 / Fig 1a の WaveFit ベース fixed-point iteration の核心)
   - **採用昇格 (重要)**: 本チケット v1 で `return_intermediates: bool = False` を採用 (上記「採用設計」参照)。T-M2.5 で必要になってから戻すのは API migration cost で遅い

8. **`forward(mel, x_gt: torch.Tensor)` で `audio_length` の代わりに `x_gt` を渡す**
   - メリット: `torch.zeros_like(x_gt)` で直接初期化、dtype/device も `x_gt` から取得
   - 却下: 「ground-truth を generator が forward 時に見ている」誤解を招く。`audio_length: int` の方が **inference time にも自然** (`x_gt` 不在の推論時に `audio_length=hop * desired_T_mel` で呼べる)

9. **`from_config(cls, cfg: dict)` classmethod factory**
   - メリット: T-M1.6 と一貫した API、YAML config drift 耐性、T-M2.5 / T-M4.3 の生成経路統一
   - **採用候補** (M1 phase review 横断テーマ): T-M1.6 §8.2 で factory パターン全モジュール一貫化が決定済。本チケットでも `GANWaveNext2.from_config(cfg)` を追加することを推奨。**実装時に T-M1.6 の factory が利用可能か確認**

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| `_residual_update` のパターン (A)/(B) pin | M5 smoke 後 | smoke が通らない場合に逆パターンを試す |
| `T=4` default | M6.3 ablation | T=2,3,5 と品質 / RTF 比較後 |
| `audio_length=None` auto-infer 式 | T-M1.3 確定後 | `T_mel * hop` か `(T_mel-1) * hop` か center=True 整合性で確定 |
| 中間 y_t loss 設計 (stop_gradient 配置) | T-M2.5 実装時 | 本チケットは intermediates list を返すのみ、loss / stop_gradient は T-M2.5 |
| `nn.ModuleList` 構造 | T-M3.1 着手前 | DiffWaveNext2 と共通親クラス `BaseVocoder` / `IterativeVocoder` を抽出するか評価 (§8.2 参照) |
| `checkpoint(..., use_reentrant=False)` 挟み込み | M5 / M6 OOM 発生時 | 24GB GPU で OOM なら enable_grad_ckpt=True 経路で発動 |
| `from_config` factory 追加 | T-M1.6 完了時 | factory が確立してたら本チケットも一貫化 |
| `_residual_update` を Callable 引数化 | 本チケット実装時 | T-M1.6 戻り値仕様がまだ未確定なら採用 (CI grey 回避) |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。
  - GAN-WaveNeXt 2 の本質は「T 個の sub-model を直列に並べた fixed-point iteration」であり、これを 1 ファイル / 1 クラスに閉じ込めることで T-M2.5 (train) / T-M4.3 (RTF) が「`GANWaveNext2(T=4)` を instantiate するだけ」で済む
  - 中間 y_t の **loss 設計** (stop_gradient 挿入箇所等) は T-M2.5 に委譲し、本チケットは forward に intermediates list を返す option を持つだけに留める
- **別マイルストーンに移すべき部分はないか**: なし。M2 (GAN) の中核として正しい位置
- **M3 (Diff) との抽象化レベル不整合リスク (新規追加 / Critical 評価項目)**:
  - T-M3.1 (DiffWaveNext2) は **4 sub-model を band 独立で呼び出す** (point-specialized partition)、本チケットの GAN は **T sub-model を逐次更新** (fixed-point iteration)
  - 両者の forward signature が divergent な可能性:
    - GAN: `forward(mel, audio_length=None, return_intermediates=False) -> Tensor | list[Tensor]`
    - Diff: `forward(mel, noise_level=None, ...) -> Tensor` (推定)
  - **T-M3.1 着手前に共通親クラス `BaseVocoder(nn.Module)` または `IterativeVocoder(BaseVocoder)` を抽出するか評価**:
    - 案 1: `BaseVocoder.forward(mel, **kwargs) -> Tensor` を pin、両者で **kwargs** で柔軟に対応
    - 案 2: 抽象化が無理なら明示的に「2 つの並列実装」と決め、共通化を諦める (T-M3.1 ticket §8 で記録)
  - **本チケット v1 では single-class 設計を維持**、T-M3.1 着手時 (= M3 phase 開始時) に再評価
- **インターフェース定義の見直し余地**:
  - **`forward(mel, audio_length, return_intermediates)` の引数順序**: T-M1.6 SubModelGAN が `forward(mel, y_prev)` で mel が先頭 → 整合
  - **`audio_length: int` を `audio_shape: tuple[int, ...]` (= `(B, T_audio)`) に変更**: 多 channel 対応が将来出たときの拡張性向上。**M1 phase review 風の検討**: 現状 mono のみ前提なので `int` で十分、stereo / multichannel は M7 で再考
  - **`from_config(cls, cfg: dict)` factory 追加**: T-M1.6 §8.2 (factory パターン全モジュール一貫化) と整合させるため、本チケットでも factory を追加することを **推奨**

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M1.6 (SubModelGAN) へ (双方向同期)
- **戻り値仕様 (n_t vs y_{t-1})** を本チケット実装時に確定したら、T-M1.6 SubModelGAN docstring の対応箇所を **同時に update**。両方の docstring が同じパターン (A) または (B) を pin していることを確認
- もし T-M1.6 が未確定のまま本チケットを着手するなら、**先に T-M1.6 §6.1 critical を解決** (T-M1.6 → T-M2.4 → T-M2.5 の順序を厳守)
- **T-M1.6 戻り値仕様**: 本チケット実装時に確定、**両チケットを同時 update** (片方だけ update すると不整合)
- **パターン (A) 採用時のサブモデル clip 仕様**: §6.1 critical 「T=1 degenerate test の clip 範囲問題」参照。sub-model 側 clip(-1,1) を外すか、本チケットの `_residual_update` 後に clip を入れる設計を pin (T-M1.6 docstring にも反映)

#### T-M2.5 (train_gan) へ
- **使用方法**:
  ```python
  from wavenext2.models.gan_wavenext2 import GANWaveNext2

  # YAML config から
  model = GANWaveNext2(T=cfg.T, sub_model_cfg=cfg.sub_model, enable_grad_ckpt=cfg.grad_ckpt)
  # 訓練 step (中間 y_t も loss 対象にする場合)
  intermediates = model(mel, return_intermediates=True)  # [y_T, y_{T-1}, ..., y_0]
  y_0 = intermediates[-1]
  loss_G = sum(mr_stft(y_t, x_gt) for y_t in intermediates[1:])  # 中間 + 最終
  loss_G += λ_adv * adv(D, y_0) + λ_fm * fm(D, y_0, x_gt)
  # または y_0 のみで簡易訓練
  y_0 = model(mel)  # audio_length=None で auto-infer
  ```
- **重要事項**:
  - **`T=4`** が論文推奨。ablation で T=2,3,5 を試す場合は YAML config の `T` を変更
  - **`enable_grad_ckpt`**: A100 40GB なら `False`、24GB GPU では `True` 推奨
  - **`return_intermediates=True` で中間 y_t も loss 対象にする** (fixed-point iteration 本論文の中核): 論文 §3.2 / Fig 1a の WaveFit ベース設計では中間 y_t に loss をかけないと sub-model 間の役割分担が崩れる。**T-M2.5 では default で `return_intermediates=True` を使い、中間 y_t を loss 対象に含める**
  - **`audio_length=None` で auto-infer を default にする**: T-M2.5 では明示的な `audio_length` を渡さず `model(mel)` で呼ぶことを推奨 (shape mismatch リスク減)。`x_gt.shape[-1]` を渡す場合は **T-M1.3 で `T_audio = T_mel * hop` か `(T_mel-1) * hop` が確定したら本チケットも揃える** (§6.1 通常項目)
  - **`best.pt` atomic rename pattern**: val improvement 頻発期 (M5.1 1 epoch では起きやすい) の race condition 回避のため、`torch.save(state, "best.pt.tmp")` → `os.replace("best.pt.tmp", "best.pt")` のアトミック書き込みを採用 (§6.1 critical 寄り参照)。DDP では rank0 のみが書き込む `if rank == 0:` ガード必須。**`keep_last_n` checkpoint 削除** (200 ckpt から 5 残し) も同様の race 注意
  - **`fixed-point loss 設計`**: `docs/training.md` §2.1 では y_0 のみが loss 対象だが、中間 y_t を使う設計が WaveFit-original 流。stop_gradient 挿入箇所は **T-M2.5 で設計** (本チケットの forward は中間に stop_gradient を入れていない、生の autograd 連鎖を返す)
  - **初期化**: `model` 内部で `torch.zeros(B, audio_length)` を生成、T-M2.5 側で zeros を作る必要なし
  - **Discriminator (T-M2.2)** と組み合わせ: `y_0 = intermediates[-1]; D_out = D(y_0)` で hinge loss を計算 (Discriminator は最終 y_0 のみに適用、中間 y_t は MR-STFT loss のみ)
  - **MR-STFT loss (T-M2.3)** と組み合わせ: 中間 y_t 全てに対し `mr_stft(y_t, x_gt)` を計算 (T 個合算)

#### T-M4.3 (RTF measurement) へ
- **使用方法**:
  ```python
  model = GANWaveNext2(T=4).eval()
  with torch.no_grad():
      for _ in range(warmup_steps):
          _ = model(mel, audio_length)
      start = time.perf_counter()
      for _ in range(measure_steps):
          _ = model(mel, audio_length)
      rtf = (time.perf_counter() - start) / measure_steps / (audio_length / sr)
  ```
- **重要事項**:
  - `model.eval()` を必ず呼ぶ (本チケットの forward は eval/train で差がない設計だが安全側)
  - `torch.no_grad()` 内で測定 (推論専用)
  - CPU 測定は `torch.set_num_threads(1)`、GPU 測定は `torch.cuda.synchronize()` で正確に
  - 論文 RTF (T=4): GPU 0.0066, CPU 0.20 (`docs/milestones.md` §M6.1)

#### T-M2.6 (smoke training) へ
- **使用方法**:
  - 1 utterance × 1000 step で `MR-STFT loss < 初期値の 10%` を期待
  - 失敗時の first action: `enable_grad_ckpt=True` で再試行 / `T=1` に減らして sub-model 単体の品質確認

#### T-M3.1 (Diff モデル) へ (新規追加)
- **`BaseVocoder` / `IterativeVocoder` 抽出判断を M3.1 着手時に行う** (§8.2 参照):
  - Diff は band 独立呼び出しなので forward signature が異なる:
    - GAN: `forward(mel, audio_length=None, return_intermediates=False) -> Tensor | list[Tensor]`
    - Diff: `forward(mel, noise_level, ...) -> Tensor` (推定、point-specialized partition で 4 sub-model を band 独立に呼ぶ)
  - 共通親クラス案:
    - 案 1: `BaseVocoder(nn.Module).forward(mel, **kwargs) -> Tensor` を pin、両者で **kwargs** で柔軟に対応
    - 案 2: `IterativeVocoder(BaseVocoder)` で iteration ロジック (`forward` の for ループ部分) を共通化、`_step(t, y, mel) -> y_new` 抽象 method を sub-model 毎に override
  - **共通化が無理なら明示的に「2 つの並列実装」設計と決め、T-M3.1 §8 で記録**
  - T-M3.1 着手前に決めないと両者が divergent な API になり、T-M4.x (RTF, MCD 等) で重複コードが発生する
- **GAN との API 不整合リスク**:
  - GAN は `return_intermediates: bool` を持つが Diff は band 独立呼び出しで「中間 y_t」概念が異なる (Diff は reverse process で各 step が異なる schedule index に対応)
  - 抽象化レベル不整合の典型例、M3.1 着手時に明示的に評価

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.4 の Acceptance チェックボックス 3 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M2.4 ステータスを `📝 pending` → `✅ completed`
  - [ ] `docs/tickets/T-M1.6-sub-model.md` の戻り値仕様 docstring を pin 結果に合わせて update (§6.1 critical 同期)
  - [ ] (該当時) `docs/architecture.md` §2 / §4 の補足追記 (例: `_residual_update` の数式表現)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**:
  - **中間 y_t loss の stop_gradient 配置** (T-M2.5 で決定): 本チケット v1 で `return_intermediates: bool` の option は提供するが、stop_gradient 挿入箇所は T-M2.5 の loss 設計時に確定。`docs/training.md` §2.1 の y_0 のみ式から拡張する場合の数式表現も T-M2.5 で明文化
  - **`BaseVocoder` / `IterativeVocoder` 抽出可否** (T-M3.1 着手前に決定): GAN / Diff の抽象化レベル不整合リスク、共通化無理なら 2 並列実装で確定
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - `T=4` default の妥当性 (M6.3 ablation で T=2,3,5 と比較)
  - `from_config` factory 追加 (T-M1.6 と一貫化)
  - `audio_length=None` の auto-infer 式 (T-M1.3 center=True 整合性確定後)
  - `checkpoint(..., use_reentrant=False)` T 毎挟み込み (M5/M6 OOM 時に発動)
  - `_residual_update` を Callable 引数化 (T-M1.6 戻り値仕様が本チケット着手時に未確定なら採用)
  - DiffWaveNext2 (T-M3.1) と共通親クラス `BaseVocoder` / `IterativeVocoder` 抽出 (T-M3.1 着手前に判断)
  - パターン (A) 採用時の clip 配置設計 (sub-model 側 or 本チケット側 or 両方)
- **`docs/open-questions.md` への追記要否**: 不要 (戻り値仕様は T-M1.6 / 本チケット内で解決、`docs/open-questions.md` は論文事実確認の場であり実装判断は ticket 内で解決)
