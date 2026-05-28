---
id: T-M3.1
title: Diff-WaveNeXt 2 モデル (4 sub-model + point-specialized partition)
milestone: M3
phase: M3
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-28
depends_on: [T-M1.6]
blocks: [T-M3.2, T-M3.3, T-M4.3]
related_docs:
  - docs/milestones.md#m31-diff-モデル-srcwavenext2modelsdiff_wavenext2py
  - docs/architecture.md
  - docs/training.md
  - docs/open-questions.md
---

# T-M3.1: Diff-WaveNeXt 2 モデル (4 sub-model + point-specialized partition)

> **マイルストーン**: [M3](../milestones.md#m3-diff-wavenext-2-作業量-large5-サブタスク) / **サブタスク**: [M3.1](../milestones.md#m31-diff-モデル-srcwavenext2modelsdiff_wavenext2py)
> **依存**: [T-M1.6](T-M1.6-sub-model.md) / **後続**: [T-M3.2](T-M3.2-train-diff.md), [T-M3.3](T-M3.3-reverse-sampler.md), [T-M4.3](T-M4.3-rtf.md)

## 1. タスク目的とゴール

### 目的
論文 Section 3.3 / Figure 1b / Figure 3 に示される **Diff-WaveNeXt 2** を、4 個の独立 `SubModelDiff` を `nn.ModuleList` に並列保持した 1 つの `nn.Module` (`DiffWaveNext2`) として実装する。各 sub-model は **point-specialized partition** で 4-step 固定 noise schedule の各点に 1-to-1 で対応する band を担当する (`docs/architecture.md` §5)。

これにより:
- T-M3.2 (`train_diff.py`) は `DiffWaveNext2(...)` を instantiate して `sample_noise_level(k, B)` で各 sub-model k の band 内 uniform sampling を取得、各 sub-model を独立に訓練できる
- T-M3.3 (`reverse_sampler`) は `model.sub_models[k-1]` を schedule の denoising 方向 t=1..4 順に呼ぶだけで 4-step DDPM 逆プロセスが回る
- T-M4.3 (RTF) は同じ `DiffWaveNext2` で推論時間を測れる

論文「**4 sub-model を異なる noise level range に特化させて独立訓練**」(Table 1 / `docs/architecture.md` §5) という設計を **コードの依存グラフ上でも明示** する。

### ゴール
- [ ] `src/wavenext2/models/diff_wavenext2.py` に `DiffWaveNext2(nn.Module)` クラスが実装され、4 個の独立 `SubModelDiff` (パラメータ非共有) を保持し total ~57.68M params (Table 1 整合)
- [ ] `NOISE_SCHEDULE_ABAR = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])` を `register_buffer` で固定保持 (state_dict に含めて推論時整合を保証)
- [ ] `BAND_BOUNDS: list[tuple[float, float]] = [(0.9929, 1.0), (0.8246, 0.9929), (0.4817, 0.8246), (0.0, 0.4817)]` をクラス属性で公開 (`√(1-ᾱ)` 軸、`docs/architecture.md` §5)
- [ ] `sample_noise_level(k, batch_size) -> torch.Tensor` (shape `(batch_size,)`) が k-th sub-model の band `[L_k, U_k)` 内 uniform sampling を返す
- [ ] `reverse_sample(mel) -> torch.Tensor` のスタブを v1 で定義し、`NotImplementedError` または T-M3.3 への delegation で記述 (詳細実装は T-M3.3)
- [ ] forward + backward が各 sub-model k で独立に動作し、対象 sub-model のみに grad が流れる
- [ ] band 境界の **重複・隙間なし** (k=1 lower == k=2 upper == 0.9929 等) を pytest で検証
- [ ] `docs/milestones.md` §M3.1 Acceptance criteria 全 3 項目をクリア
- [ ] `tests/test_diff_wavenext2.py` が `uv run pytest tests/test_diff_wavenext2.py` で全 pass

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/models/diff_wavenext2.py`
  - `tests/test_diff_wavenext2.py`
- 編集:
  - `src/wavenext2/models/__init__.py` (`__all__` に `DiffWaveNext2` を追加)
  - `docs/milestones.md` §M3.1 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` のステータス更新

### 2.2 主要構造

#### import 形式
```python
from wavenext2.models.diff_wavenext2 import DiffWaveNext2
```

#### クラスシグネチャ

```python
"""diff_wavenext2.py — 4 個の SubModelDiff を point-specialized partition で保持する Diff-WaveNeXt 2.

論文 Fig 1b / Section 3.3 / Fig 3 の 4 sub-model 独立訓練 + 4-step reverse sampling を実装する。
- 4 個の SubModelDiff を独立に保持 (重み共有なし、Table 1 のパラメータ数 57.68M = 14.42M × 4)
- noise schedule ᾱ = [1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1] (denoising 順 t=1..4)
- point-specialized band partition: 隣接 schedule 点 √(1-ᾱ) の中点で band を区切る

詳細は docs/architecture.md §5, docs/training.md §3 / §4.2 を参照。
"""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

from wavenext2.models.sub_model import SubModelDiff


class DiffWaveNext2(nn.Module):
    """Diff-WaveNeXt 2: 4 個の SubModelDiff を point-specialized partition で並列保持."""

    # 4-step 固定 noise schedule (denoising 方向 t=1..4)
    # t=1: 純ノイズ側 (ᾱ=1e-4), t=4: ほぼクリーン (ᾱ=9.1e-1)
    # docs/architecture.md §5 / docs/training.md §3.3
    NOISE_SCHEDULE_ABAR_DEFAULT: tuple[float, ...] = (1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1)

    # Point-specialized band partition (√(1-ᾱ) 軸、隣接点の中点で区切る)
    # k=1 (最高ノイズ) → k=4 (最低ノイズ) の順
    # docs/architecture.md §5 Band 境界表で確定
    BAND_BOUNDS: list[tuple[float, float]] = [
        (0.9929, 1.0),     # k=1: 中心 √(1-ᾱ_1) = 0.99995
        (0.8246, 0.9929),  # k=2: 中心 √(1-ᾱ_2) = 0.9858
        (0.4817, 0.8246),  # k=3: 中心 √(1-ᾱ_3) = 0.6633
        (0.0, 0.4817),     # k=4: 中心 √(1-ᾱ_4) = 0.3
    ]

    # sub-model 数 (4 で固定、論文 §3.3)
    K: int = 4

    def __init__(
        self,
        sub_model_cfg: dict | None = None,
    ) -> None:
        """
        Args:
            sub_model_cfg: SubModelDiff の __init__ に渡す dict。
                例: {"mel_channels": 128, "n_fft": 1024, "hop_length": 256,
                     "win_length": 1024, "sinusoidal_dim": 128, "cond_dim": 512, ...}
        """
        super().__init__()
        sub_model_cfg = dict(sub_model_cfg or {})
        # factory 経由 (T-M1.6 §8.2 整合); 直接 __init__ も許容
        self.sub_models = nn.ModuleList([
            SubModelDiff.from_config(sub_model_cfg, mode="diff") for _ in range(self.K)
        ])

        # noise schedule を state_dict 互換で保持 (推論時の整合性確保、§6.1 参照)
        self.register_buffer(
            "noise_schedule_abar",
            torch.tensor(self.NOISE_SCHEDULE_ABAR_DEFAULT, dtype=torch.float32),
            persistent=True,
        )

        # SubModelDiff の hop_length をキャッシュ (audio_length 検算用)
        self.hop_length: int = self.sub_models[0].hop_length

        # band 境界の不整合チェック (隙間/重複なし)
        self._validate_band_bounds()

    @classmethod
    def _validate_band_bounds(cls) -> None:
        """BAND_BOUNDS が docs/architecture.md §5 の point-specialized partition と整合するか検証.

        - 隣接 band の境界が一致 (k の lower == k-1 の upper)
        - 全体カバレッジ: k=K の lower == 0.0, k=1 の upper == 1.0
        """
        bb = cls.BAND_BOUNDS
        if len(bb) != cls.K:
            raise ValueError(f"BAND_BOUNDS must have {cls.K} entries, got {len(bb)}")
        if bb[0][1] != 1.0:
            raise ValueError(f"BAND_BOUNDS[0] upper must be 1.0, got {bb[0][1]}")
        if bb[-1][0] != 0.0:
            raise ValueError(f"BAND_BOUNDS[-1] lower must be 0.0, got {bb[-1][0]}")
        for k in range(cls.K - 1):
            # k (1-indexed) の lower == k+1 の upper
            if bb[k][0] != bb[k + 1][1]:
                raise ValueError(
                    f"BAND_BOUNDS gap/overlap at k={k + 1}/{k + 2}: "
                    f"bb[{k}][0]={bb[k][0]} != bb[{k + 1}][1]={bb[k + 1][1]}"
                )

    def sample_noise_level(self, k: int, batch_size: int) -> torch.Tensor:
        """sub-model k (1-indexed) の band [L_k, U_k) 内で uniform sampling.

        Args:
            k:          sub-model index (1..K = 4)。
            batch_size: 出力サンプル数。

        Returns:
            (batch_size,) tensor of √(1-ᾱ) ∈ [L_k, U_k).
        """
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        L, U = self.BAND_BOUNDS[k - 1]
        # device/dtype は呼び出し側で .to() する設計 (sub_model の training step で device 一致)
        return torch.empty(batch_size).uniform_(L, U)

    def get_band(self, k: int) -> tuple[float, float]:
        """sub-model k の band [L_k, U_k) を返す (T-M3.2 から参照可)."""
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        return self.BAND_BOUNDS[k - 1]

    def forward(
        self,
        mel: torch.Tensor,
        x_t: torch.Tensor,
        c: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """単一 sub-model k の forward (訓練時に使用、T-M3.2 から呼ばれる).

        Args:
            mel: (B, 128, T_mel) log-mel.
            x_t: (B, T_audio = T_mel * hop_length) noised waveform.
            c:   (B,) noise level √(1-ᾱ) ∈ band [L_k, U_k).
            k:   呼び出す sub-model index (1..K=4)。

        Returns:
            (B, T_audio) 予測ノイズ ε_θ (MSE ターゲット、Fig 1b)。

        Note:
            複数 sub-model を一括で呼ぶ API は提供しない (band 独立訓練の設計上 1 sub-model ずつ呼ぶ)。
            推論 (reverse sampling) は T-M3.3 の `reverse_sample` で別実装。
        """
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        return self.sub_models[k - 1](mel, x_t, c)

    def reverse_sample(self, mel: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """4-step DDPM 逆プロセスで mel から波形を合成 (推論時に使用).

        T-M3.3 の reverse_sampler に詳細実装を委譲。本チケット v1 ではスタブ。
        詳細は docs/training.md §4.2 を参照。
        """
        raise NotImplementedError(
            "reverse_sample is implemented in T-M3.3. "
            "See src/wavenext2/inference/infer_diff.py."
        )

    @classmethod
    def from_config(cls, cfg: dict, only_sub_model: int | None = None) -> "DiffWaveNext2":
        """YAML config から DiffWaveNext2 を生成 (T-M1.6 §8.2 factory パターン一貫化).

        Args:
            cfg: dict with key "sub_model_cfg" (and optional future fields).
            only_sub_model: 指定時は対象 sub-model (1-indexed) のみを instantiate し、
                他は None placeholder で `sub_models` に格納する (state_dict 互換維持)。
                T-M3.2 で 1 sub-model のみ訓練するときの OOM 回避 (4×14.42M=~58M weight +
                optimizer state を削減) 用。default=None で全 4 sub-model を instantiate。
        """
        sub_model_cfg = cfg.get("sub_model_cfg", {})
        if only_sub_model is None:
            return cls(sub_model_cfg=sub_model_cfg)
        # lazy instantiation: 指定 sub-model 1 個のみ Module、他は None
        if not 1 <= only_sub_model <= cls.K:
            raise ValueError(f"only_sub_model must be in [1, {cls.K}], got {only_sub_model}")
        instance = cls(sub_model_cfg=sub_model_cfg)
        for i in range(cls.K):
            if (i + 1) != only_sub_model:
                instance.sub_models[i] = None  # type: ignore[index]
        return instance

    # T-M3.3 から `reverse_sample` を re-export して staticmethod alias 化する想定 (thin alias)
    # T-M4.3 (RTF) が GAN/Diff 横断で `model.synthesize(mel)` を呼べるようにする dispatch 用
    # synthesize = staticmethod(reverse_sample)  # T-M3.3 完成後に有効化
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `K` (sub-model 数) | 4 (固定) | docs/architecture.md §5, docs/training.md §3.1, Table 1 |
| `NOISE_SCHEDULE_ABAR` | `[1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]` | docs/training.md §3.3, docs/architecture.md §5, docs/open-questions.md §解決済み |
| `BAND_BOUNDS` (k=1) | `(0.9929, 1.0)` | docs/architecture.md §5 Band 境界表 |
| `BAND_BOUNDS` (k=2) | `(0.8246, 0.9929)` | docs/architecture.md §5 Band 境界表 |
| `BAND_BOUNDS` (k=3) | `(0.4817, 0.8246)` | docs/architecture.md §5 Band 境界表 |
| `BAND_BOUNDS` (k=4) | `(0.0, 0.4817)` | docs/architecture.md §5 Band 境界表 |
| `sub-model 重み共有` | **しない** (4 個独立) | docs/architecture.md §5.4 末尾、Table 1 (57.68M = 14.42M × 4) |
| パラメータ数目標 (4 sub-model 計) | ~57.68M | docs/architecture.md §5 (Table 1) |
| パラメータ数目標 (1 sub-model) | ~14.42M | docs/architecture.md §5 (Table 1) |
| n_fft / hop / win (Diff) | 1024 / 256 / 1024 | docs/architecture.md §3, §7 |
| Diffusion 式 | `x_t = √ᾱ · x_0 + √(1-ᾱ) · ε` (DDPM 標準形) | docs/open-questions.md §解決済み, PDF Fig 1b 確認 |

### 2.4 アルゴリズム / 処理フロー

#### `__init__`
1. `sub_model_cfg` を copy
2. `SubModelDiff.from_config(sub_model_cfg, mode="diff")` を `K=4` 回呼び `nn.ModuleList` に格納
3. `register_buffer("noise_schedule_abar", torch.tensor(NOISE_SCHEDULE_ABAR_DEFAULT))` で state_dict に固定保持
4. `self.hop_length = self.sub_models[0].hop_length` を cache
5. `_validate_band_bounds()` を classmethod で呼び境界整合性チェック (隙間 / 重複なし)

#### `sample_noise_level(k, batch_size)`
1. `1 <= k <= K` を assert
2. `L, U = self.BAND_BOUNDS[k - 1]`
3. `return torch.empty(batch_size).uniform_(L, U)` で `(batch_size,)` shape の連続値 sample
4. device / dtype 移動は呼び出し側 (T-M3.2 train step) で `.to(mel.device)`

#### `forward(mel, x_t, c, k)`
1. `k` 範囲 assert
2. `return self.sub_models[k - 1](mel, x_t, c)` (SubModelDiff へ delegate)

#### `reverse_sample(mel)` (v1 スタブ、T-M3.3 で実装)
- `NotImplementedError` を raise
- T-M3.3 で `docs/training.md` §4.2 の DDPM 標準形 4-step を実装

#### 不変条件
- `noise_schedule_abar.shape == (K,)` かつ降順ではなく **denoising 順 (t=1..4)** で `ᾱ_1 < ᾱ_2 < ᾱ_3 < ᾱ_4` (1e-4 → 9.1e-1)
- `BAND_BOUNDS[k][0] < BAND_BOUNDS[k][1]` (lower < upper)
- 4 sub-model それぞれが独立 instance (id() が異なる)、forward を 2 回呼んでも干渉しない

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `diff_wavenext2.py` 実装 + `tests/test_diff_wavenext2.py` 記述 | general-purpose |
| Reviewer | 1 | 論文 Fig 1b / `docs/architecture.md` §5 / `docs/training.md` §3 整合性確認 + Table 1 (57.68M) 整合 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_diff_wavenext2.py -v` 実行 + band 境界数値検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **partial** (T-M3.3 reverse_sampler は本チケット完了後に着手、T-M3.4 post-filter は独立)
- 並列実行する場合の最大並列数: 1 (M3 の他チケットは本チケット完了に依存)
- T-M3.2 (train_diff) は本チケット完了後に着手

## 4. 提供範囲 (Scope)

### In Scope
- `class DiffWaveNext2(nn.Module)` の実装
- 4 個の `SubModelDiff` を `nn.ModuleList` で保持 (重み非共有)
- `NOISE_SCHEDULE_ABAR_DEFAULT` クラス属性 + `register_buffer("noise_schedule_abar", ...)` で state_dict 含め
- `BAND_BOUNDS` クラス属性公開 (`docs/architecture.md` §5 で確定値)
- `sample_noise_level(k, batch_size)` メソッド (band 内 uniform sampling)
- `get_band(k)` メソッド (T-M3.2 から band 取得可)
- `forward(mel, x_t, c, k)` で単一 sub-model k を呼ぶ訓練用 API
- `_validate_band_bounds()` classmethod (init 時の境界整合性チェック)
- `reverse_sample(mel)` スタブ (T-M3.3 へ delegate、`NotImplementedError`)
- `from_config(cfg)` factory (T-M1.6 §8.2 と一貫)
- shape 検証 (`k` 範囲、`sub_model_cfg` 型)
- `tests/test_diff_wavenext2.py` (5.1 / 5.3 すべて)
- パラメータ数検証テスト (4 sub-model 合計 ~57.68M、1 sub-model ~14.42M)
- `__init__.py` への `__all__` 追加
- docstring (英文 + 日本語)

### Out of Scope
- 訓練ループ / optimizer / scheduler (T-M3.2)
- Reverse sampling の実装 (T-M3.3、スタブのみ提供)
- Post-filter (T-M3.4)
- TensorBoard / checkpoint 保存 (T-M3.2)
- `return_intermediates` の概念 (Diff は band 独立で並列訓練、中間 y_t の loss 連鎖は存在しない、§6.1 参照)
- Mixed precision / torch.compile (M5 / M6 で必要時)
- noise schedule 自体の学習 (BDDM predictor、再現不要 `docs/open-questions.md` §解決済み)
- equal partition (Okamoto21 strict) の実装 (§8 で却下根拠明示、ablation したい場合は M6.3)
- 4 sub-model 一括 forward API (band 独立呼び出し設計のため不要)
- `BaseVocoder` / `IterativeVocoder` 親クラス抽出 (§6 / §8 で「2 並列実装」判断、共通化は `from_config()` のみ)
- LibriTTS-R real audio との e2e (T-M5.2 で 1 sub-model 1 epoch smoke)

### Deliverable
- ファイル:
  - `src/wavenext2/models/diff_wavenext2.py` (新規)
  - `tests/test_diff_wavenext2.py` (新規)
  - `src/wavenext2/models/__init__.py` (re-export 追加)
- 関数 / クラス:
  - `class DiffWaveNext2(nn.Module)`
  - `sample_noise_level(k, batch_size) -> torch.Tensor`
  - `get_band(k) -> tuple[float, float]`
  - `forward(mel, x_t, c, k) -> torch.Tensor`
  - `reverse_sample(mel) -> torch.Tensor` (スタブ)
  - `_validate_band_bounds()` classmethod
  - `from_config(cfg)` classmethod
  - クラス属性 `NOISE_SCHEDULE_ABAR_DEFAULT`, `BAND_BOUNDS`, `K`
  - buffer `noise_schedule_abar`
- ドキュメント差分:
  - `docs/milestones.md` §M3.1 の Acceptance チェック
  - `docs/tickets/index.md` の T-M3.1 ステータス

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_diff_wavenext2.py`)

#### パラメータ数
- [ ] `test_param_count_total`: 4 sub-model 合計 `sum(p.numel() for p in model.parameters()) ≈ 57.68M ± 0.2M` (Table 1)
- [ ] `test_param_count_per_sub_model`: 各 sub-model `sum(p.numel() for p in model.sub_models[k].parameters()) ≈ 14.42M ± 0.05M`
- [ ] `test_param_count_linear`: 1 sub-model × 4 = 全体パラメータ数 (重み非共有確認、shared なら 14.42M で固定のはず)
- [ ] `test_param_independence`: `model.sub_models[0]` と `model.sub_models[1]` が異なる `id()` を持ち、片方の weight を変更しても他に伝播しない

#### 4 sub-model 独立性 (milestones.md §M3.1 Acceptance #1)
- [ ] `test_four_sub_models_independent`: `model.sub_models` 長さ = 4、各 sub-model が `SubModelDiff` 型
- [ ] `test_independent_grad`: `model.forward(mel, x_t, c, k=2).sum().backward()` で **sub_models[1] のみ** に grad が流れ、他 sub-model (0, 2, 3) の grad は None
- [ ] `test_sub_model_state_dict_separate`: `model.state_dict()` に `sub_models.0.*`, `sub_models.1.*`, `sub_models.2.*`, `sub_models.3.*` が含まれる

#### `sample_noise_level` の band 範囲 (milestones.md §M3.1 Acceptance #2)
- [ ] `test_sample_noise_level_k1_range`: `sample_noise_level(k=1, batch_size=1000)` の全要素が `[0.9929, 1.0)` に収まる
- [ ] `test_sample_noise_level_k2_range`: k=2 で `[0.8246, 0.9929)` に収まる
- [ ] `test_sample_noise_level_k3_range`: k=3 で `[0.4817, 0.8246)` に収まる
- [ ] `test_sample_noise_level_k4_range`: k=4 で `[0.0, 0.4817)` に収まる
- [ ] `test_sample_noise_level_uniform_distribution`: 各 band 内で大数の law で uniform 分布 (KS test の p-value > 0.05 で `uniform([L, U])` と整合)
- [ ] `test_sample_noise_level_invalid_k`: `k=0` / `k=5` / `k=-1` で `ValueError`
- [ ] `test_sample_noise_level_shape`: 戻り値 shape == `(batch_size,)`

#### `BAND_BOUNDS` 数値検証
- [ ] `test_band_bounds_no_gap`: 隣接 band の境界が一致 (`bb[0][0] == bb[1][1] == 0.9929`, etc.)、隙間なし
- [ ] `test_band_bounds_no_overlap`: 同じ境界値 (重複なし) → 上記と同条件で検証可
- [ ] `test_band_bounds_full_coverage`: `bb[-1][0] == 0.0`, `bb[0][1] == 1.0` (全体 `[0, 1]` をカバー)
- [ ] `test_band_bounds_match_architecture_md`: 各 band 値が `docs/architecture.md` §5 Band 境界表と数値一致 (±1e-4 許容、丸め誤差吸収)
- [ ] `test_band_bounds_centers_match_schedule`: band の中心 (近似) と `√(1-ᾱ)` schedule 点の対応:
  - k=1 中心 ≈ 0.99995 (= √(1 - 1e-4))
  - k=2 中心 ≈ 0.9858 (= √(1 - 2.8e-2))
  - k=3 中心 ≈ 0.6633 (= √(1 - 5.6e-1))
  - k=4 中心 ≈ 0.3 (= √(1 - 9.1e-1))

#### `NOISE_SCHEDULE_ABAR` buffer (state_dict 整合性、§6 参照)
- [ ] `test_noise_schedule_buffer_register`: `"noise_schedule_abar"` が `model.state_dict()` に含まれる
- [ ] `test_noise_schedule_values`: `model.noise_schedule_abar` の値が `[1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]` で `torch.allclose(..., atol=1e-7)`
- [ ] `test_noise_schedule_load_state_dict`: `model.load_state_dict(model.state_dict())` で問題なくロード可
- [ ] `test_noise_schedule_dtype`: `model.noise_schedule_abar.dtype == torch.float32`

#### `forward(mel, x_t, c, k)`
- [ ] `test_forward_shape`: `B=2`, `T_mel=94`, `audio=24064` (= 94×256) で `out.shape == (2, 24064)` (1 sub-model k=2 を呼ぶ)
- [ ] `test_forward_all_k`: k=1, 2, 3, 4 すべてで forward 動作確認
- [ ] `test_forward_invalid_k`: k=0 / k=5 で `ValueError`
- [ ] `test_forward_backward_isolates_grad`: `forward(..., k=2).sum().backward()` で sub_models[1] のみに grad

#### `reverse_sample` スタブ
- [ ] `test_reverse_sample_not_implemented`: `model.reverse_sample(mel)` で `NotImplementedError` (T-M3.3 で実装される)

#### Determinism
- [ ] `test_sample_noise_level_deterministic`: `torch.manual_seed(42)` 固定で `sample_noise_level(k=1, B=4)` を 2 回呼んだ結果が一致
- [ ] `test_forward_deterministic`: 同 mel + 同 c + 同 seed で 2 回 forward が一致 (sub_model 内に dropout なし)

#### `_validate_band_bounds` (内部 invariant)
- [ ] `test_validate_band_bounds_pass`: default `BAND_BOUNDS` で `_validate_band_bounds()` がエラーなく完走
- [ ] `test_validate_band_bounds_detects_gap`: 一時的に `BAND_BOUNDS` を改変 (隙間ありに) → `_validate_band_bounds()` で `ValueError` (monkeypatch でテスト)

#### `from_config` factory
- [ ] `test_from_config`: `DiffWaveNext2.from_config({"sub_model_cfg": {...}})` で生成可、`isinstance(model, DiffWaveNext2)`
- [ ] `test_from_config_empty`: `DiffWaveNext2.from_config({})` でも default の `sub_model_cfg={}` で生成可
- [ ] `test_from_config_lazy_instantiation`: `DiffWaveNext2.from_config({"sub_model_cfg": {...}}, only_sub_model=1)` で `sub_models[0]` のみ `SubModelDiff` instance、`sub_models[1]`, `sub_models[2]`, `sub_models[3]` が `None` placeholder。`state_dict()` に `sub_models.0.*` のみ含まれること (None は state_dict 出力されない) も確認
- [ ] `test_from_config_lazy_instantiation_invalid_k`: `only_sub_model=0` / `only_sub_model=5` で `ValueError`
- [ ] `test_from_config_lazy_param_count`: `only_sub_model=k` 指定時に `sum(p.numel() for p in model.parameters()) ≈ 14.42M` (1 sub-model 分のみ)、OOM 回避効果を検証

#### `synthesize` alias (T-M3.3 完成後に有効化)
- [ ] `test_synthesize_alias` (`@pytest.mark.skipif(reverse_sample 未実装)`): `model.synthesize(mel)` の戻り値が `reverse_sample(model, mel)` と同等 (T-M3.3 完成後)。T-M4.3 (RTF) の `getattr(model, "synthesize", model.forward)` dispatch を担保

### 5.2 e2e / 結合テスト
- [ ] `test_diff_wavenext2_with_real_sub_model_cfg`: `configs/diff_wavenext2.yaml` の sub_model_cfg を読んで `DiffWaveNext2.from_config(cfg)` で生成、forward が動作 (T-M3.2 の YAML 読み込み経路を模す)
- [ ] `test_module_list_independence`: 上記 `test_param_independence` と同等を結合レベルでも確認
- [ ] `test_real_audio` (`@pytest.mark.slow`): LibriTTS-R sample 1 件で実 mel 抽出 → `DiffWaveNext2().forward(mel, x_t, c, k=1)` → backward が通る。T-M0.3 完了後、T-M2.1 dataset 未完成期間は `pytest.skip`

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.1 より転記)
- [ ] 4 sub-model それぞれが独立してパラメータを保持
- [ ] `sample_noise_level(k=1)` の値が `[0.9929, 1.0]` に収まる (k=2,3,4 も同様)
- [ ] パラメータ総数 = 4 × 14.42M ≈ 57.68M (Table 1)

### 5.4 追加 acceptance (本チケット独自)
- [ ] `pytest tests/test_diff_wavenext2.py` が exit code 0 (slow / gpu マーカー除く)
- [ ] `from wavenext2.models import DiffWaveNext2` で import 可能
- [ ] `NOISE_SCHEDULE_ABAR_DEFAULT` がクラス属性で公開、`noise_schedule_abar` が `register_buffer` で state_dict 含む
- [ ] `BAND_BOUNDS` の数値が `docs/architecture.md` §5 と完全一致 (±1e-4 許容)
- [ ] T-M2.4 (GAN モデル) と **共通親クラスを抽出しない方針** を §8 に記録 (forward signature 不整合のため、§6.2 / §8.2 参照)

### 5.5 テスト戦略
- **scope="module" fixture**: `DiffWaveNext2()` の init は ~57.68M params で重いため、fixture で再利用 (各テストで init し直さない)
- **`@pytest.mark.slow`**: 実音声テストはデフォルト除外
- **CI 時間目標**: T-M3.1 全テスト合計 **~30 秒以内** (4 sub-model fp32 forward+backward が CPU で ~6 秒 × 5 ケース程度)
- **coverage 目標**: 85% (`reverse_sample` の実装テストは T-M3.3 でカバー)
- **band 境界の数値テスト**: `pytest.approx(rel=1e-4)` で `docs/architecture.md` §5 の値と比較 (丸め誤差を吸収)

## 6. 懸念事項

### 6.1 技術的リスク

#### CRITICAL候補: MSE loss の per-sub-model scale 不均衡

- **問題**: ε-prediction で MSE loss を取ると、sub-model k による c の値域差が loss scale 差として顕在化する
  - sub-model 1 (`c ∈ [0.9929, 1.0)`、`c≈0.999`): `x_t ≈ eps` (純ノイズ側)、`eps_pred` 予測対象が dominant component → loss ~ O(1)
  - sub-model 4 (`c ∈ [0, 0.4817)`、`c≈0.3`): `eps` の `x_t` 寄与が小さく `x_0` 寄与が大きい → eps 予測は容易、loss ~ O(0.1)
- **影響**:
  - TensorBoard で sub-model 別の `loss_mse` が **1〜2 桁ズレる** ことが前提 (見た目の不均衡で「sub-model 4 が学習していない」と誤判定しないよう注意)
  - sub-model 別に lr を変える ablation のトリガー (§8.1 案 11、§10)
- **対応**: 本チケット v1 では同一 lr で全 sub-model 訓練 (`docs/training.md` §3.4 通り)、M5.2 smoke で scale 不均衡を観察してから ablation 検討
- **救命策**: §6.1 通常項目「v-prediction parameterization」(Salimans & Ho 2022) を M3.5 smoke で loss 学習されない場合に切替検討

#### CRITICAL候補: `BAND_BOUNDS` の数値が `docs/architecture.md` §5 と完全一致するか

- **問題**: `BAND_BOUNDS = [(0.9929, 1.0), (0.8246, 0.9929), (0.4817, 0.8246), (0.0, 0.4817)]` は `docs/architecture.md` §5 表で記載された値だが、`(√(1-ᾱ_k) + √(1-ᾱ_{k+1})) / 2` の計算結果と完全に一致するか実装時に検証が必要
- **計算式 (`docs/architecture.md` §5)**:
  - `c = √(1-ᾱ) = [0.99995, 0.9858, 0.6633, 0.3]`
  - `(c[0]+c[1])/2 = (0.99995+0.9858)/2 = 0.99287` → 表記 `0.9929` (丸め)
  - `(c[1]+c[2])/2 = (0.9858+0.6633)/2 = 0.82455` → 表記 `0.8246` (丸め)
  - `(c[2]+c[3])/2 = (0.6633+0.3)/2 = 0.48165` → 表記 `0.4817` (丸め)
- **対応**:
  - 案 A: 表記値 `[0.9929, 0.8246, 0.4817]` を hardcode (シンプル、現状の `BAND_BOUNDS` がこれ)
  - 案 B: `NOISE_SCHEDULE_ABAR` から runtime 計算 (`(c[k-1] + c[k]) / 2`) で `BAND_BOUNDS` 生成 (schedule 変更時に自動追従)
- **本チケット v1 採用**: 案 A (hardcode)、ただし `_validate_band_bounds()` で `NOISE_SCHEDULE_ABAR` から計算した値と一致するかテストで照合
- **再評価トリガー**: M5.2 smoke で sub-model k=2 の training band と実際の schedule 点 √(1-ᾱ_2)=0.9858 がズレる場合、案 B に切り替え

#### CRITICAL候補: 4 sub-model 独立 vs shared (論文 Table 1 で確定済)

- **問題**: 論文 Table 1 で「Diff-WaveNeXt 2 (w/ sub-model)」が 57.68M、「(wo/ sub-model)」が 14.42M。前者は **4 個独立**、後者は **1 sub-model 全 noise level で訓練** の意味
- **確定**: 本チケットは **4 個独立 (重み非共有)** で実装 (`nn.ModuleList` で 4 個独立 instance)
- shared 案は §8 で却下根拠を明示

#### CRITICAL候補: `NOISE_SCHEDULE_ABAR` を register_buffer or constant にするか (state_dict 互換性)

- **問題**:
  - constant (class attribute) にすると state_dict に含まれず、推論時にスキーマ変更 (将来の K=8 拡張等) に弱い
  - register_buffer にすると state_dict に含まれて推論時の整合性チェックが可能だが、保存サイズが微増 (4 float = 16B、無視可能)
- **本チケット v1 採用**: **両方を併用**:
  - クラス属性 `NOISE_SCHEDULE_ABAR_DEFAULT` (immutable な tuple で固定値を明示)
  - インスタンス buffer `noise_schedule_abar` (register_buffer で state_dict 含み)
- **理由**: 「論文の固定 schedule」という constraint を class level で明示しつつ、checkpoint 保存時に推論時整合性 (将来の schedule 変更検知) を確保
- **代替案**: §8 参照

#### CRITICAL候補: STFTModule window 共有メモリ節約効果 (`docs/architecture.md` §3 STFT パラメータ)

- **問題**: Diff は 4 sub-model 全部 n_fft=1024 同じなので `STFTModule.window` (Hann window, n_fft+1 float32) を 4 重複保持 (`4KB × 4 = 16KB`)
- **メモリ影響**: 16KB は微小で optimization 不要レベル
- **本チケット v1 採用**: **共有しない** (sub-model 内に独立保持、設計シンプル)
- **再評価トリガー**: M3.1 実装後の `torch.cuda.memory_allocated()` profiling で 16KB が問題になる可能性は皆無だが、構造的整合性 (sub-model が pure に独立) を優先
- **代替案 (DI)**: `SubModelDiff` 外で `STFTModule` を 1 個 init して DI → T-M1.6 §8.1 案 9 で「M3.1 で要再評価」とされていた。本チケットで **不採用** を確定し、`docs/tickets/T-M1.6-sub-model.md` §8.1 案 9 に「M3.1 で不採用確定」を追記

#### CRITICAL候補: `BaseVocoder` / `IterativeVocoder` 親クラス抽出可否 (T-M2.4 から伝搬)

- **問題**: T-M2.4 §8.2 / §9.1 で「M3.1 着手前に判断」と決定された設計事項
- **GAN と Diff の forward signature 比較**:
  - GAN (`GANWaveNext2`): `forward(mel, audio_length=None, return_intermediates=False) -> Tensor | list[Tensor]`
    - 内部で `for t in range(T, 0, -1)` の逐次 fixed-point iteration、sub-model 間に状態 (前 y_t) が連鎖
  - Diff (`DiffWaveNext2`): `forward(mel, x_t, c, k) -> Tensor`
    - **単一 sub-model k を band 独立に呼ぶ** (4 個一括 forward の API は不要、band 独立訓練のため)
    - 推論時 (`reverse_sample`) は別 API で T-M3.3 が責務
- **共通化判断**:
  - 案 1 (`BaseVocoder.forward(mel, **kwargs)` で **kwargs 柔軟対応): forward signature を `**kwargs` で抽象化、両者で異なる引数を吸収。**却下**: `**kwargs` は型ヒントが効かず IDE 補完も弱い、TorchScript 互換性低
  - 案 2 (`IterativeVocoder(BaseVocoder)` で iteration を抽象化): GAN の逐次 iteration と Diff の band 独立呼び出しは **iteration の意味自体が異なる** (GAN: 状態を引き継ぐ逐次、Diff: 並列独立) ため、共通化すると低レベル化して可読性低下。**却下**
  - 案 3 (`BaseVocoder` に `from_config()` factory のみ実装、forward は各サブクラスで自由): factory パターンだけ統一して forward は別実装。**保留** (本チケット v1 では採用しない、M5 phase review で再評価)
- **本チケット v1 採用**: **「2 並列実装」設計を明示確定**
  - GAN と Diff は **forward signature が本質的に異なる** (逐次 vs 並列、状態あり vs なし)
  - 共通化するなら `from_config()` factory のみ統一可だが、`SubModelGAN.from_config()` / `SubModelDiff.from_config()` の sub-model level の factory が既に T-M1.6 §8.2 で存在するため、上位クラスの factory 統一は **冗長**
  - したがって `BaseVocoder` 親クラス抽出は **不採用**、`DiffWaveNext2` と `GANWaveNext2` は独立した `nn.Module` サブクラスとして並列実装
- **記録**: 本判断を本チケット §8.2 / §9.1 に明示記録、T-M2.4 §8.2 / §9.1 にも参照リンクを追記

#### 通常項目

- **`register_buffer(persistent=True)` で `state_dict` に schedule 16B×4=64B 重複保存**:
  - 各 sub-model の `STFTModule.window` (n_fft+1 float32) と合わせて、buffer の冗長性が積み上がる
  - **影響**: M6 で `keep_last_n=5 × 4 sub` = 20 ckpt × 14.42M weight ≈ 11 GB に対し誤差レベル (無視可能)
  - **将来の BDDM 学習で schedule mismatch が起きる場合**: `strict=False` で load_state_dict 推奨 or `persistent=False` で skip
  - **本チケット v1 採用**: `persistent=True` (整合性チェック優先)、再評価は M6.2 本格訓練時

- **v-prediction parameterization (Salimans & Ho 2022) 選択肢**:
  - **問題**: ε-prediction だと §6.1 critical「per-sub-model scale 不均衡」が顕在化 (sub-model 1 で loss 大、sub-model 4 で消える)
  - **v-prediction**: `v = sqrt(abar) * eps - sqrt(1-abar) * x_0` を予測ターゲット化、SNR-weighted で scale 均衡化
  - **本チケット v1 採用**: ε-prediction (`docs/architecture.md` §5 / `docs/training.md` §3.1 の論文準拠)
  - **救命策トリガー**: M3.5 smoke で loss が学習されない / sub-model 別 loss 差が 2 桁以上の場合、v-prediction に切替検討 (`SubModelDiff.forward` の出力意味変更が必要、影響範囲大)

- **`forward(mel, x_t, c, k)` の引数順序と `nn.Module.forward` 慣例**:
  - PyTorch の `nn.Module.forward` は positional 引数を想定するが、本クラスは `k: int` (Tensor ではない discrete index) を最後に取る
  - `k` を `__call__(mel, x_t, c, k=k)` のように kwarg で渡す呼び出しは OK だが、`model(mel, x_t, c, 2)` の位置引数渡しも動く
  - **対応**: docstring で「`k` は 1-indexed discrete index、kwarg 推奨」を明示

- **`sample_noise_level` の device / dtype**:
  - 戻り値は CPU tensor で、呼び出し側 (T-M3.2) で `.to(mel.device)` する設計
  - 別案: `sample_noise_level(k, batch_size, device=None, dtype=torch.float32)` で device 引数を取る
  - **本チケット v1 採用**: CPU tensor 返却、呼び出し側で移動 (シンプル、`torch.uniform_` は CPU で高速)
  - **再評価トリガー**: T-M3.2 で multi-GPU の DDP 訓練時に non-blocking transfer が問題になれば device 引数を追加

- **4-step schedule の denoising 方向 (t=1 が純ノイズ、t=4 がクリーン)**:
  - `docs/architecture.md` §5 / `docs/training.md` §3.3 で確定: `ᾱ = [1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]` で `t=1` が `ᾱ_1=1e-4` (純ノイズ側)、`t=4` が `ᾱ_4=9.1e-1` (ほぼクリーン)
  - 本チケットは **denoising 順** で配列を保持 (t=1..4 が `[0]..[3]`)
  - 推論時 (T-M3.3) は `range(K)` の自然な順序で sub_models[t] を呼べば良い (1-indexed の `k = t + 1`)

- **`reverse_sample` スタブの責務**:
  - 本チケット v1 では `NotImplementedError` を raise (T-M3.3 で実装)
  - 別案: T-M3.3 を blocking deps とせず、本チケットで実装してしまう → **却下** (`docs/milestones.md` で T-M3.3 を独立チケットとしているため、本チケットは model 構造に集中)

- **`k` を 1-indexed か 0-indexed か**:
  - 論文 / `docs/architecture.md` §5 / `docs/training.md` §3.2 で sub-model index は **1-indexed** (k=1..4)
  - 内部 `self.sub_models[k - 1]` で 0-indexed list 変換
  - **API は 1-indexed で統一** (論文表記に従い、誤用防止)

- **`SubModelDiff.from_config()` の存在前提**:
  - T-M1.6 §8.2 で factory パターン採用が決定済だが、T-M1.6 完了時に実装されているか確認
  - 未実装なら直接 `SubModelDiff(**sub_model_cfg)` で代替 (本チケット実装時のフォールバック)

- **`return_intermediates` の不要性** (T-M2.4 との対比):
  - GAN は逐次 fixed-point iteration で中間 y_t に loss を載せる設計 (T-M2.4 §6.1 通常項目)
  - Diff は band 独立で各 sub-model が **独立に訓練** されるため「中間」概念がない
  - 推論時は T-M3.3 reverse_sampler が 4-step を別 API で扱う
  - **本チケット forward は `return_intermediates` 引数を持たない** (設計上不要)

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目:
  - **Diffusion 式**: ✅ DDPM 標準形 `x_t = √ᾱ · x_0 + √(1-ᾱ) · ε` で確定 (PDF Fig 1b 画像確認)
  - **4 値固定 schedule**: ✅ `[1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1]` を直接使用 (BDDM predictor 再現不要)
  - **Point-specialized partition**: ✅ 1-to-1 mapping、band 境界は隣接 schedule 点の中点 (`docs/architecture.md` §5)
  - **Diff conditioning 注入**: ✅ 射影なし additive bias、共有 NoiseEmbedding を各 block で直接加算 (per-block fc_t は Table 1 +14% のため撤去、2026-05-27 §C7。T-M1.1/T-M1.5 で実装済)
  - **4 sub-model 独立訓練**: ✅ Table 1 (57.68M = 14.42M × 4) で確定
- 旧来の判断ポイント (本チケット作成時に確定):
  - **`forward` の引数 (mel, x_t, c, k) vs (mel, x_t, c) で k は別 API**: `forward(mel, x_t, c, k)` を採用 (`sample_noise_level(k)` と対応)
  - **k は kwarg only にするか positional 許容するか**: positional 許容、docstring で kwarg 推奨
  - **`nn.ModuleList` vs `nn.ModuleDict`**: `nn.ModuleList` を採用 (`nn.ModuleDict` で `{1: ..., 2: ...}` key にすると key conversion (int↔str) が必要、シンプルさで `nn.ModuleList` 優位)

### 6.3 他チケットとの整合性

- **T-M1.6 (SubModelDiff)** との整合:
  - 期待 signature: `sub_model(mel, x_t, c) -> (B, T_audio)`
  - 期待 attribute: `hop_length`, `n_fft`, `mel_channels`
  - 期待 classmethod: `SubModelDiff.from_config(cfg, mode="diff")` (T-M1.6 §8.2 採用済)
  - 不整合があれば T-M1.6 を修正 (T-M3.1 は consumer)
  - **STFTModule window 共有 (DI)** は本チケットで **不採用** を確定 (§6.1 critical 候補参照)、T-M1.6 §8.1 案 9 にフィードバック

- **T-M2.4 (GANWaveNext2)** との整合:
  - **共通親クラス抽出は不採用** を本チケットで確定 (§6.1 critical 候補参照)
  - `BaseVocoder` / `IterativeVocoder` を抽出せず、GAN / Diff は独立 `nn.Module` サブクラスとして並列実装
  - T-M2.4 §8.2 / §9.1 に「M3.1 で不採用確定」を追記

- **T-M3.2 (train_diff)** へ渡す情報 (§9.1 参照):
  - YAML config: `sub_model_cfg: {...}`
  - 訓練時の呼び出し: `c = model.sample_noise_level(k=cli_k, batch_size=B).to(mel.device)`
  - 訓練 step: `eps_pred = model(mel, x_t, c, k=cli_k)`
  - sub-model k は CLI 引数 or YAML config で指定、1 sub-model ずつ独立訓練
  - 各 sub-model の checkpoint は独立保存 (`checkpoints/diff/sub_1.pt`, ..., `sub_4.pt`)

- **T-M3.3 (reverse_sampler)** へ渡す情報:
  - `model.noise_schedule_abar` (register_buffer) を参照して 4-step DDPM 逆プロセス実装
  - `model.sub_models[t-1](mel, x_t, c)` を t=1..4 順に呼ぶ (point-specialized 1-to-1 mapping)
  - `model.reverse_sample(mel)` スタブを T-M3.3 で実装するか、`src/wavenext2/inference/infer_diff.py` の `reverse_sample(model, mel)` 関数として外部実装するかは T-M3.3 で決定

- **T-M4.3 (RTF measurement)** へ渡す情報:
  - `model.eval()` で deterministic 推論
  - `model.reverse_sample(mel)` (T-M3.3 完了後) の推論時間を測定
  - 論文 RTF (Diff w/ sub-model, 4-step): 別途確認 (`docs/training.md` §5.3 で記載されていれば参照)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §5 (point-specialized partition、band 境界表) と数値完全一致 (±1e-4 許容)
- [ ] `docs/training.md` §3.2 (各 sub-model 独立訓練、band 内 uniform sampling) と整合
- [ ] `docs/training.md` §3.3 (4 値固定 noise schedule) と整合
- [ ] `docs/open-questions.md` §解決済み (DDPM 標準形、4 値直接使用、point-specialized) と整合
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 全クリア
- [ ] パラメータ数が 4 sub-model 合計 ~57.68M (Table 1 ±0.2M)、1 sub-model ~14.42M (±0.05M)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: `k` 範囲外、`sub_model_cfg` 型不整合で `ValueError` (silent failure なし)
- [ ] **参考実装 (FastDiff / BDDM) をコピーしていない** (CLAUDE.md 末尾の方針、`docs/architecture.md` §5 / `docs/training.md` §3 の論述から再構成)
- [ ] `__init__.py` の `__all__` に `DiffWaveNext2` が追加されている
- [ ] `NOISE_SCHEDULE_ABAR_DEFAULT` クラス属性 + `noise_schedule_abar` register_buffer の **両方** が実装されている
- [ ] `BAND_BOUNDS` のクラス属性公開、4 値の隣接整合性
- [ ] `_validate_band_bounds()` が `__init__` で呼ばれている
- [ ] `nn.ModuleList` で 4 個 sub-model が **独立 instance** (weight 非共有)
- [ ] `sample_noise_level(k, batch_size)` の戻り値 shape `(batch_size,)` で正しい
- [ ] forward に副作用がない (state を持たない、再現性のため)
- [ ] `reverse_sample` スタブが `NotImplementedError` を raise (T-M3.3 で実装される旨を docstring に明示)
- [ ] CPU でテストが pass する (CUDA 不要)
- [ ] T-M2.4 (GANWaveNext2) との **共通親クラス抽出を不採用** とする判断記録が §8 に明示

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M3) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計

- **4 個の独立 `SubModelDiff` を `nn.ModuleList` で保持、`sample_noise_level(k, B)` で band 内 uniform sampling、`forward(mel, x_t, c, k)` で単一 sub-model k を呼ぶ、`NOISE_SCHEDULE_ABAR` を register_buffer + class attribute の併用で保持**
  - 理由:
    - (a) 4 sub-model 独立は `docs/architecture.md` §5.4 末尾「sub-model 間で共有しない」+ Table 1 (57.68M = 14.42M × 4) で確定
    - (b) `nn.ModuleList` は `parameters()` / `state_dict()` / partial freeze と相性が良く、独立訓練に適する
    - (c) `BAND_BOUNDS` のクラス属性公開で T-M3.2 / T-M3.3 が直接参照可、設定 drift 耐性
    - (d) `register_buffer` + class attribute 併用で、推論時の整合性 (state_dict load) と設計上の constraint 明示 (class level) を両立

- **`reverse_sample` を v1 スタブ (`NotImplementedError`) とし、T-M3.3 で実装**:
  - 理由: 本チケットは model 構造 (4 sub-model + band partition) に集中、reverse sampling は推論経路として T-M3.3 で独立実装するのが `docs/milestones.md` の設計
  - スタブを置くことで、T-M3.3 着手時に「`model.reverse_sample` を実装する」という明確な entry point ができる

- **`forward(mel, x_t, c, k)` で `k` を引数化、4 sub-model 一括 forward の API は提供しない**:
  - 理由: band 独立訓練の設計上、1 sub-model ずつ独立に呼ぶのが訓練時の自然な経路 (T-M3.2)
  - 4 一括 forward は推論時の `reverse_sample` (T-M3.3) で必要だが、これは内部で逐次 sub-model[t] を呼ぶだけ
  - 一括 API を提供すると、`reverse_sample` の DDPM step を 4 サブモデル一括に書き換える際に、step ごとに schedule index ↔ sub-model index の対応を取る複雑度が増す

- **`from_config(cfg, only_sub_model: int | None = None)` lazy instantiation を v1 採用**:
  - 採用昇格 (旧 §8.1 案 10 関連の拡張、独立採用判断)
  - 4 sub-model 全 instantiate (`only_sub_model=None`、default) と、1 sub-model のみ instantiate (`only_sub_model=k`) を切替可能化
  - `only_sub_model=k` 指定時: 対象 sub-model `k` のみ Module、`sub_models[other_idx] = None` placeholder で `state_dict()` 互換維持
  - 理由:
    - **T-M3.2 で 1 sub-model のみ訓練するときの OOM 回避**: 4×14.42M=~58M weight + optimizer state (Adam の m/v で 2×) = 訓練対象でない 3 sub-model 分が memory 占有
    - **M6.2 32h 訓練の wall-clock 短縮**: 4 sub-model 並列訓練 (別ジョブ) で各ジョブが 1 sub-model 分の memory のみ確保、ジョブあたり VRAM 余裕が大きい → batch_size 拡大可能
  - 別案 (置換可能): YAML config の `sub_model_cfg.skip_indices: [2, 3, 4]` を読んで skip → 却下 (config 階層が肥大、CLI 引数経由のほうが train スクリプトとの結合が natural)
  - 再評価トリガー: M5.2 / M6.2 で OOM が発生しない場合は default の 4 全 instantiate のみで十分

- **`DiffWaveNext2.synthesize = staticmethod(reverse_sample)` thin alias を v1 で追加** (T-M3.3 から re-export):
  - 採用昇格 (T-M4.3 RTF measurement で GAN/Diff 横断 dispatch を簡素化)
  - T-M3.3 完成後に `from wavenext2.inference.infer_diff import reverse_sample` を import して `synthesize = staticmethod(reverse_sample)` を class level で追加
  - 理由:
    - **T-M4.3 (RTF) が `getattr(model, "synthesize", model.forward)` で GAN/Diff 統一 dispatch 可能**:
      - GAN: `synthesize` = `forward` の自然 alias (逐次 fixed-point iteration が forward)
      - Diff: `synthesize` = `reverse_sample` の alias (4-step DDPM 逆プロセス)
    - **`model.synthesize(mel)` で API 表記統一**: 訓練 (`forward`) と推論 (`synthesize`) の責務分離を class API として明示
  - 別案: T-M3.3 で `model.reverse_sample` 本実装を入れる → 採用 (synthesize は staticmethod alias、本実装は reverse_sample にあるか class method として注入)
  - 再評価トリガー: T-M4.3 で `getattr` dispatch が必要なくなる (例えば protocol class を抽出する) 場合は alias 不要

#### Deprecated (却下案)

1. **shared sub-model (4 個で同じパラメータ使い回し)**
   - メリット: パラメータ数を 4 で割れる (~14.42M)
   - 却下: **Table 1 で却下確定**。「Diff-WaveNeXt 2 (w/ sub-model)」が 57.68M、「(wo/ sub-model)」が 14.42M で **4 倍**。論文は **重み非共有**

2. **equal partition (Okamoto21 strict、`[k-1)/K, k/K)`)**
   - メリット: 計算が simple、Okamoto21 (K=10 + N=6/N=25 step の strict partition) と一貫
   - 却下: 論文は K=N=4 で point-specialized 1-to-1 partition を採用 (`docs/architecture.md` §5)。strict equal partition だと band 1 ([0, 0.25)) が無使用になり、4 sub-model deploy (Table 1 整合) と矛盾
   - 再評価: M6.3 ablation で品質比較する場合は別 config として実装

3. **各 sub-model の noise range の境界を学習可能パラメータ化**
   - メリット: band 境界も最適化対象とし、データに合わせて auto-tuning
   - 却下: 論文は固定境界 (point-specialized で中点) を採用。学習可能化は論文の意図と乖離、安定性も不明。研究的に面白いが本チケット (再現実装) では却下
   - 将来検討: M6.3 ablation or 別研究として

4. **`nn.ModuleDict` で k=1..4 を key として保持**
   - メリット: `model.sub_models["k1"]` のように name accessible
   - 却下: int↔str key conversion が必要、`nn.ModuleList` の `[k-1]` で十分シンプル。一貫性のために `nn.ModuleList` 採用

5. **`forward(mel, x_t, c, k)` 引数を `forward(mel, x_t, c)` にして、内部で `c` から band を判定し `k` を auto-infer**
   - メリット: 呼び出し側が `k` を明示する必要なし
   - 却下: c が複数 band の境界近傍にあるとき判定が曖昧 (e.g., c=0.4817 が k=3 か k=4 か)。`k` を明示する方が誤用防止に強い

6. **`BAND_BOUNDS` を `NOISE_SCHEDULE_ABAR` から runtime 計算**
   - メリット: schedule 変更時に自動追従、hardcode との drift 防止
   - **保留** (本チケット v1 では hardcode 採用)。論文の 4 値が **固定** (BDDM predictor 再現不要、`docs/open-questions.md` §解決済み) のため drift リスクは低い。M5.2 smoke で band 境界の不整合が顕在化した場合に切り替え

7. **`BaseVocoder` / `IterativeVocoder` 親クラスを抽出**
   - メリット: GAN と Diff で共通インターフェース (`forward(mel, **kwargs)`) で抽象化、T-M4.3 (RTF) / T-M5 (smoke) で重複コード削減
   - 却下: §6.1 critical 候補で詳述。GAN は逐次状態あり、Diff は band 独立で状態なし、forward signature が **本質的に異なる**。`**kwargs` で抽象化すると型ヒントが効かず保守性低下。`from_config()` factory のみ統一する案は、sub-model level (T-M1.6 §8.2) で既に factory パターンが確立しているため上位での factory 統一は冗長。**「2 並列実装」設計で確定**

8. **4 sub-model 一括 forward API (`forward_all(mel, x_t_list, c_list) -> list[Tensor]`)**
   - メリット: 推論時 (`reverse_sample`) の DDPM step を一括処理
   - 却下: 訓練時に 1 sub-model ずつ独立呼び出しが主用法、一括 API は推論時の minor 利便性のみ。`reverse_sample` 内で逐次 `sub_models[t-1]` を呼べば十分

9. **`STFTModule` window を 4 sub-model で共有 (DI 経由)**
   - メリット: window buffer 16KB のメモリ節約
   - 却下: 16KB は微小で optimization 不要。sub-model が独立構造を保つ方が構造的整合性が高い。T-M1.6 §8.1 案 9 で「M3.1 で要再評価」とされていたが、本チケットで **不採用確定**

10. **`from_config(cls, cfg: dict)` で `NOISE_SCHEDULE_ABAR` も config から読む**
    - メリット: schedule を YAML で variant 化可能 (将来の K=8 拡張等)
    - 採用候補: 本チケット v1 では class attribute hardcode、`from_config` は `sub_model_cfg` のみ。schedule 変更は別 ticket (M6.3 ablation 等) で扱う

11. **log-uniform sampling** (`sample_noise_level` の mode 引数化)
    - **設計**: `sample_noise_level(k, batch_size, mode: Literal["uniform", "log_uniform"] = "uniform")` で切替可能化
    - **uniform** (現状 default): `torch.empty(B).uniform_(L, U)` (区間内一様)
    - **log_uniform**: `exp(uniform(log(L), log(U)))` (対数スケールで一様)
    - **メリット**:
      - sub-model 4 (`c ∈ [0.0, 0.4817)`) は線形 uniform だと `c≈0` 付近 (signal-rich 領域) の確率質量が小さい
      - 対数スケール sampling で `c≈0` 付近に確率質量集中、loss scale バランス改善期待 (§6.1 critical「per-sub-model scale 不均衡」の緩和策)
    - **注意**: sub-model 4 の lower bound が 0.0 なので `log(0)=-inf` で数値不安定。実装時は `max(L, 1e-6)` で clipping 必要
    - **本チケット v1 採用**: uniform (`docs/training.md` §3.2 「band 内 uniform sampling」論文準拠)
    - **再評価トリガー**: M5.2 smoke で sub-model 4 が学習しない (loss が初期値から下がらない) 場合に log_uniform 切替検討

12. **`__init_subclass__` で BAND_BOUNDS validation 1 度のみ**
    - **問題**: 現状 `_validate_band_bounds()` を classmethod として `__init__` 内で呼ぶため、4 sub-model instantiate ごとに 4 回呼ばれる (CI で冗長)
    - **改善案**: `__init_subclass__(cls, **kwargs)` で class 定義時に 1 度だけ validation 実行 (BAND_BOUNDS は class attribute なので class level で十分)
    - 別案: `__init__` の最初に `if not hasattr(type(self), "_validated"): self._validate_band_bounds(); type(self)._validated = True` で class level flag
    - **本チケット v1 採用**: classmethod 4 回呼び出し維持 (validation cost が小さく optimization 不要)、CI 冗長性が問題になれば `__init_subclass__` 化
    - **再評価トリガー**: CI 実行時間で validation が顕在化する場合 (PyTorch import overhead より大きいことは現実的に皆無)

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| 4 sub-model 独立 vs shared | M6.3 ablation | shared 案を ablation で比較 (品質低下の確認用) |
| Point-specialized vs equal partition | M6.3 ablation | equal partition を別 config で実装し品質比較 |
| `BAND_BOUNDS` hardcode vs runtime 計算 | M5.2 smoke | band 境界と schedule 点 √(1-ᾱ) のズレが品質に影響するか確認 |
| `NOISE_SCHEDULE_ABAR` register_buffer | M6.2 本格訓練 | state_dict 互換性問題が出たら class attribute のみに簡素化 |
| `reverse_sample` スタブ vs 本実装 | T-M3.3 着手時 | T-M3.3 で本実装、本チケット v1 はスタブで OK |
| `BaseVocoder` 親クラス抽出不採用 | M5 phase review | RTF / smoke で重複コードが目立つ場合に `from_config()` 共通化を再検討 |
| `STFTModule` window 共有 | M3.1 完了後の memory profiling | 16KB 節約効果は皆無なので不採用維持の見込み |
| `forward(mel, x_t, c, k)` の引数順序 | M5 phase review | k を kwarg-only にするか検討 |
| `sample_noise_level` の device 引数 | T-M3.2 実装時 | multi-GPU DDP で non-blocking transfer が問題になれば追加 |
| `sample_noise_level` の mode 引数 (uniform / log_uniform) | M5.2 smoke | sub-model 4 が学習しない場合に log_uniform 切替 (§8.1 案 11) |
| `__init_subclass__` での BAND_BOUNDS validation | CI 実行時間プロファイル時 | classmethod 4 回呼び出しが CI コストになる場合 (§8.1 案 12) |
| MSE loss の per-sub-model scale 不均衡 → sub-model 別 lr | M5.2 smoke | TensorBoard で sub-model 別 loss が 1〜2 桁ズレた場合に ablation (§6.1 critical / §10) |
| v-prediction parameterization 切替 | M3.5 smoke | loss が学習されない / scale 差 2 桁以上 (§6.1 通常項目) |
| `from_config(only_sub_model=k)` lazy instantiation | M5.2 / M6.2 | 4 全 instantiate で OOM が発生しない場合は default のみで十分 |
| `synthesize` staticmethod alias | T-M4.3 設計時 | protocol class 抽出等で `getattr` dispatch 不要になれば削除 |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切。
  - Diff-WaveNeXt 2 の本質は「4 sub-model 独立 + point-specialized partition」であり、これを 1 ファイル / 1 クラスに閉じ込めることで T-M3.2 (train) / T-M3.3 (reverse_sampler) / T-M4.3 (RTF) が分業しやすい
  - `reverse_sample` のスタブを残すことで、T-M3.3 着手時の entry point が明確化
- **別マイルストーンに移すべき部分はないか**: なし。M3 (Diff) の中核として正しい位置
- **T-M2.4 (GANWaveNext2) との抽象化レベル**:
  - **「2 並列実装」設計を確定** (§6.1 critical 候補で詳述)
  - GAN: 逐次 fixed-point iteration、forward 内で sub-model 間に状態 (前 y_t) が連鎖
  - Diff: band 独立 forward、4 sub-model は独立に訓練される (forward は 1 sub-model のみ呼ぶ)
  - 両者の forward signature の差は本質的 (アルゴリズムの違い)、`BaseVocoder` で抽象化すると低レベル化して可読性低下
  - 共通化するなら `from_config()` factory のみだが、sub-model level (T-M1.6 §8.2) で既に factory パターンが確立しているため上位での factory 統一は冗長
  - **記録**: T-M2.4 §8.2 / §9.1 に「M3.1 で不採用確定」を追記、本チケット §6.1 と §8.2 に判断記録
- **`forward` 死蔵 API 問題**:
  - **問題**: T-M3.2 (`train_diff.py`) line 337 (想定) で `sub_models[k-1](mel, x_t, c)` を直接呼ぶ実装になっており、`DiffWaveNext2.forward(mel, x_t, c, k)` が **死蔵 API** 化する
    - 訓練 step: T-M3.2 は `model.sub_models[k-1](mel, x_t, c)` で sub-model 直叩き (model 経由のオーバーヘッド回避、grad isolation 明示)
    - 推論 step: T-M3.3 reverse_sampler も `model.sub_models[t-1](mel, x, c)` で直叩き想定
    - → `DiffWaveNext2.forward` が誰からも呼ばれない可能性
  - **対応案**:
    - 案 A: `forward` を `@deprecated` 扱い (将来削除予告)、dispatch は consumer 側 (T-M3.2 / T-M3.3) で統一
    - 案 B: `forward` を削除して `DiffWaveNext2` は容器 (sub_models container) に徹する設計
    - 案 C: 現状維持 (`forward` 提供、consumer は使うも使わないも自由)
  - **本チケット v1 採用**: **案 A** (`@deprecated` 扱い、消極的維持)
    - 理由: `nn.Module` の慣例として `forward` 定義は推奨される (torch.nn.Module.__call__ が forward を呼ぶ前提)
    - consumer 側 (T-M3.2 / T-M3.3) で sub_models 直叩きする設計を採用するが、`forward` も動作する状態は維持 (テストでも検証)
    - docstring に「consumer は `sub_models[k-1]` 直叩き推奨、`forward` は abstraction layer」と明記
  - **再評価トリガー**: T-M3.2 / T-M3.3 完成時に `forward` の利用箇所が 0 件なら案 B (削除) 検討

- **インターフェース定義の見直し余地**:
  - **`forward(mel, x_t, c, k)` の引数 `k` の位置**: 現状 positional 最後だが、kwarg-only (`*` で強制) にする案も可。M5 phase review で再評価
  - **`sample_noise_level` の戻り値を `(B,)` の Tensor から `(B, 1)` に変更**: SubModelDiff.forward の c 引数 shape が `(B,)` なので一致、不要。`(B,)` で確定
  - **`reverse_sample` を class method ではなく外部関数 (`infer_diff.reverse_sample(model, mel)`) として実装**: T-M3.3 で決定。本チケットはスタブを置くのみ
  - **`from_config` factory 追加**: T-M1.6 §8.2 (factory パターン全モジュール一貫化) と整合させるため、本チケットでも `DiffWaveNext2.from_config(cfg)` を追加 (採用済)

### 8.3 学んだこと (2026-05-28 実装完了)

- **`from_config` の `mode=` 引数は実装に存在しない**: チケット §2.2 / §2.4 の擬似コード `SubModelDiff.from_config(sub_model_cfg, mode="diff")` は誤り (M2 review 申し送り #5 で指摘済)。実 API は `SubModelDiff.from_config(cfg)` のみ (`src/wavenext2/models/sub_model.py:139`)。本実装は `SubModelDiff.from_config(cfg)` を使用。GAN/Diff の区別は `SubModelGAN` / `SubModelDiff` の **クラス自体** で表現されており、`mode=` フラグは不要な設計。
- **`NOISE_SCHEDULE_ABAR` は class attribute ではなく buffer への property alias で実装**: チケット擬似コードは class attribute (`NOISE_SCHEDULE_ABAR = torch.tensor(...)`) と buffer の併用だったが、class attribute の Tensor は device 追従しない (`model.to("cuda")` 後も CPU のまま)。T-M3.3 が `model.NOISE_SCHEDULE_ABAR.to(device)` で参照する際、buffer 経由 property (`@property def NOISE_SCHEDULE_ABAR -> buffer`) なら device-aware かつ load_state_dict 整合性を両立できる。class level の immutable 定数明示は `NOISE_SCHEDULE_ABAR_DEFAULT` (tuple) が担う。
- **パラメータ数は Table 1 と一致しない (既知、fc_t 撤去)**: 実装値は 1 sub-model = **14,354,434** (14.354M)、×4 = **57,417,736** (57.42M)。Table 1 (14.42M / 57.68M) に対し −0.46%。これは M1 で per-block fc_t (2.1M) を撤去した結果 (docs/open-questions.md §C7)。テストは厳密実装値 (`== 14_354_434`) で pin しつつ、Table 1 への近接 (±1%) も別 assert で担保した。チケット §5.1 の `± 0.05M` 許容は fc_t 撤去前の値 (14.42M) 基準だったため、撤去後の実装値に合わせて訂正。
- **`_validate_band_bounds` は lower < upper チェックを追加**: チケット擬似コードの隣接整合 + 全体被覆に加え、各 band の `lower < upper` も検証 (monkeypatch で改変した bad bounds を確実に検出するため)。`__init__` 冒頭で呼ぶ (sub-model instantiate 前に fail-fast)。
- **`synthesize` staticmethod alias は未実装 (T-M3.3 follow-up)**: §8.1 採用昇格の `DiffWaveNext2.synthesize = staticmethod(reverse_sample)` は reverse_sample 本実装 (T-M3.3) 完成後に有効化する設計のため、本チケットでは追加せず。T-M3.3 完了時に注入する。
- **45 tests pass、全体 275 passed / 1 skipped / 2 deselected、ruff clean**。
- 教訓: チケット擬似コードは「設計意図の記録」であり実 API と乖離しうる (特に fc_t 撤去のような後発の確定事項が反映されていない箇所)。実装着手時は必ず依存先の実コード (`sub_model.py`) のシグネチャを Read で確認してから書く。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.4 (GANWaveNext2) へ (双方向同期)
- **共通親クラス `BaseVocoder` / `IterativeVocoder` 抽出は不採用** を本チケットで確定 (§6.1 critical 候補、§8.2 参照)
- T-M2.4 §8.2 / §9.1 の「M3.1 着手前に判断」項目に **「不採用確定」を追記する** (T-M2.4 ticket 修正)
- 理由: GAN / Diff の forward signature が本質的に異なる (逐次状態あり vs band 独立)、`**kwargs` 抽象化は型ヒント・保守性低下、`from_config()` 統一は sub-model level (T-M1.6 §8.2) で既存のため上位での重複

#### T-M1.6 (SubModelDiff) へ (双方向同期)
- **`STFTModule` window 共有 (DI)** は本チケットで **不採用確定** (§6.1 critical 候補参照)
- T-M1.6 §8.1 案 9 (「M3.1 で要再評価」) に **「M3.1 で不採用確定 (16KB 微小、構造的整合性優先)」を追記** (T-M1.6 ticket 修正)
- 4 sub-model それぞれが独立 `STFTModule` を保持する設計を維持

#### T-M3.2 (train_diff) へ
- **使用方法**:
  ```python
  from wavenext2.models.diff_wavenext2 import DiffWaveNext2

  # YAML config から (4 sub-model 全 instantiate, 通常)
  model = DiffWaveNext2.from_config({"sub_model_cfg": cfg.sub_model})

  # 1 sub-model のみ instantiate (lazy、OOM 回避 / wall-clock 短縮、§8.1 採用昇格)
  k = args.sub_model_k  # CLI 引数 (1..4)
  model = DiffWaveNext2.from_config({"sub_model_cfg": cfg.sub_model}, only_sub_model=k)
  # → model.sub_models[k-1] のみ Module、他は None placeholder

  # または直接
  model = DiffWaveNext2(sub_model_cfg=cfg.sub_model)

  # 訓練対象 sub-model index を CLI 引数 or YAML config から取得 (1..4)
  k = args.sub_model_k

  for batch in dataloader:
      mel, x_0 = batch  # (B, 128, T_mel), (B, T_audio)
      B = mel.shape[0]

      # 1. 担当 band 内で noise level sampling
      c = model.sample_noise_level(k=k, batch_size=B).to(mel.device)  # (B,)
      abar = 1.0 - c ** 2                                              # (B,)
      sqrt_abar = torch.sqrt(abar)
      sqrt_one_minus_abar = c

      # 2. 拡散式 (DDPM 標準形)
      eps = torch.randn_like(x_0)
      x_t = sqrt_abar.view(-1, 1) * x_0 + sqrt_one_minus_abar.view(-1, 1) * eps

      # 3. 単一 sub-model k で forward
      eps_pred = model(mel, x_t, c, k=k)

      # 4. MSE loss
      loss = mse_loss(eps_pred, eps)
      loss.backward()
  ```
- **重要事項**:
  - **k は CLI 引数 or YAML config で 1..4 を指定** (1 sub-model ずつ独立訓練)
  - **`from_config(only_sub_model=k)` で 1 sub-model のみインスタンス化、他は None で skip** (OOM 回避、wall-clock 短縮、§8.1 採用昇格)
    - `model.sub_models[k-1]` のみ Module、`model.sub_models[other_idx] is None`
    - optimizer は `model.sub_models[k-1].parameters()` のみを対象に作成 (None placeholder の parameters() は呼ばない)
    - state_dict 互換維持: `model.state_dict()` には `sub_models.{k-1}.*` のみ含まれる (None は serialize されない)
    - resume 時は同じ k で `only_sub_model=k` 指定すれば `load_state_dict(strict=True)` で OK
  - **`sample_noise_level(k, B)` の戻り値は CPU tensor**、`.to(mel.device)` で移動が必要
  - **4 sub-model それぞれ独立に checkpoint 保存**: `checkpoints/diff/sub_1.pt`, ..., `sub_4.pt`
  - **band 内 uniform sampling** が TensorBoard で確認可能 (各 step の c のヒストグラム plot を推奨)
  - **per-sub-model loss scale 不均衡を TensorBoard で観察** (§6.1 critical、§9.1 T-M5.2 連絡):
    - sub-model 1 で loss ~ O(1)、sub-model 4 で loss ~ O(0.1) を前提として view
    - **sub-model 別 lr を変える ablation トリガー判断**: 4 sub-model 間で loss が桁違いに学習速度が異なる場合
  - **`forward(mel, x_t, c, k)` の backward** は sub_models[k-1] のみに grad、他 sub-model の `model.parameters()` イテレーション時に grad=None を許容する必要あり (optimizer は sub-model k のみを対象に作成)
  - **`forward` 死蔵 API 問題** (§8.2): consumer は `model.sub_models[k-1](mel, x_t, c)` 直叩き推奨 (model 経由のオーバーヘッド回避、grad isolation 明示)、`forward` は abstraction layer として残るが consumer 側統一
  - **Optimizer**: Adam(lr=2e-4, betas=[0.9, 0.98], wd=0)、`docs/training.md` §3.4

#### T-M3.3 (reverse_sampler) へ
- **使用方法**:
  ```python
  # 推奨: 外部関数として実装 (src/wavenext2/inference/infer_diff.py)
  def reverse_sample(model: DiffWaveNext2, mel: torch.Tensor) -> torch.Tensor:
      abar = model.noise_schedule_abar  # (4,) register_buffer
      K = len(abar)
      # ... DDPM 標準形 4-step (docs/training.md §4.2)
      x = torch.randn(...)
      for t in range(K):
          c_t = torch.sqrt(1.0 - abar[t]).expand(B)
          # point-specialized 1-to-1 mapping: t (0-indexed) → k (1-indexed) = t + 1
          eps_pred = model.sub_models[t](mel, x, c_t)
          # ... DDPM reverse step (docs/training.md §4.2 数式)
      return x

  # または model.reverse_sample(mel) を本実装 (本チケットではスタブ)
  ```
- **重要事項**:
  - **`model.noise_schedule_abar`** (register_buffer) を参照、`docs/training.md` §4.2 の β / σ 計算式に従う
  - **point-specialized 1-to-1**: 推論順 t=1..4 (0-indexed t=0..3) で sub_models[t] を順に呼ぶ
  - **`model.sub_models[t](mel, x, c)`** で SubModelDiff の forward を直接呼ぶ (DiffWaveNext2.forward 経由でも OK)
  - **post-filter 適用は T-M3.4 で実装**、reverse_sampler は post-filter 適用前の x_0 を返す
  - **`model.reverse_sample(mel)` 本実装 or 外部関数** どちらにするかは T-M3.3 で決定
  - **`synthesize` alias から呼ばれる前提**: T-M3.1 で `DiffWaveNext2.synthesize = staticmethod(reverse_sample)` を再エクスポート (§8.1 採用昇格)。T-M4.3 (RTF) が `model.synthesize(mel)` で GAN/Diff 横断 dispatch する
  - **全 sub-model がロードされている前提**: reverse_sampler は 4 sub-model すべてを順に呼ぶため、`only_sub_model` lazy instantiation は推論時に使用不可。推論用 model は `DiffWaveNext2.from_config(cfg)` (only_sub_model=None) で作成し、各 sub-model checkpoint を `model.sub_models[k-1].load_state_dict(torch.load(f"sub_{k}.pt"))` で個別 load

#### T-M4.3 (RTF measurement) へ
- **使用方法**:
  ```python
  model = DiffWaveNext2().eval()
  with torch.no_grad():
      # warmup
      for _ in range(warmup_steps):
          _ = reverse_sample(model, mel)  # T-M3.3 完了後
      # measure
      start = time.perf_counter()
      for _ in range(measure_steps):
          _ = reverse_sample(model, mel)
      rtf = (time.perf_counter() - start) / measure_steps / (audio_length / sr)
  ```
- **重要事項**:
  - `model.eval()` を必ず呼ぶ
  - `torch.no_grad()` 内で測定
  - CPU 測定は `torch.set_num_threads(1)`、GPU 測定は `torch.cuda.synchronize()`
  - 4-step reverse sampling 全体の推論時間を測定 (sub_model forward 4 回 + DDPM step computation)
  - **`model.synthesize(mel)` で GAN/Diff 横断計測** (§8.1 採用昇格):
    - T-M3.1 の `DiffWaveNext2.synthesize = staticmethod(reverse_sample)` alias 経由
    - `getattr(model, "synthesize", model.forward)` で GAN (forward が synthesize 相当) / Diff (synthesize=reverse_sample alias) を統一 dispatch
    - RTF 測定ループは `synthesize_fn = getattr(model, "synthesize", model.forward); _ = synthesize_fn(mel)` でモデル非依存に書ける

#### T-M3.5 (Diff smoke training) へ
- **使用方法**:
  - 1 utterance × 1000 step で sub-model 1 (k=1) を訓練し `MSE loss < 初期値の 5%` を期待 (`docs/milestones.md` §M3.5)
  - 失敗時の first action: `k=1` の代わりに `k=4` (低ノイズ side、訓練が容易) で試す / batch_size を減らす

#### T-M5.2 (smoke 統合) へ
- **per-sub-model loss scale 不均衡を TensorBoard で観察** (§6.1 critical 候補):
  - sub-model 別に `loss_mse` を log し、TensorBoard で 4 系列を重ね描画
  - sub-model 1 (`c≈0.999`): loss ~ O(1)、sub-model 4 (`c≈0.3`): loss ~ O(0.1) を **前提として view**
  - 「sub-model 4 が学習していない」と誤判定しないよう、loss 値ではなく **「初期値からの相対減少率」** を見る
- **ablation トリガー判断**:
  - 4 sub-model 間で loss の絶対値ではなく **学習速度 (相対減少率)** が桁違いに異なる場合 → sub-model 別 lr ablation 検討 (§8.1 案 11、§10)
  - sub-model 4 が学習しない (1000 step で初期値の 80% 以上) → log_uniform sampling 切替 (§8.1 案 11) or v-prediction parameterization 切替 (§6.1 通常項目)
- **観察項目**:
  - 各 sub-model k の `loss_mse` 推移
  - 各 sub-model k の c (noise level) ヒストグラム (band 内 uniform を確認)
  - 各 sub-model k の `eps_pred` の norm (sub-model 1 / 4 で 1〜2 桁差を確認)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M3.1 の Acceptance チェックボックス 3 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M3.1 ステータスを `📝 pending` → `✅ completed`
  - [ ] `docs/tickets/T-M2.4-gan-model.md` §8.2 / §9.1 に「`BaseVocoder` 抽出は M3.1 で不採用確定」を追記
  - [ ] `docs/tickets/T-M1.6-sub-model.md` §8.1 案 9 (STFTModule window 共有 DI) に「M3.1 で不採用確定」を追記
  - [ ] (該当時) `docs/architecture.md` §5 の補足追記 (例: `BAND_BOUNDS` 計算式の明示、丸め誤差説明)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**:
  - **`BAND_BOUNDS` hardcode vs runtime 計算**: M5.2 smoke で再評価 (§8.1 案 6)
  - **`reverse_sample` を class method or 外部関数**: T-M3.3 で決定 (本チケットはスタブのみ)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - 4 sub-model 独立 vs shared の品質比較 (M6.3 ablation)
  - Point-specialized vs equal partition の品質比較 (M6.3 ablation)
  - `BaseVocoder` 親クラス抽出可否の M5 phase review 再評価
  - `STFTModule` window 共有メモリ profiling (M3.1 完了後、優先度低)
  - `sample_noise_level` の device 引数追加 (T-M3.2 multi-GPU DDP 時)
  - schedule 変更時の `NOISE_SCHEDULE_ABAR` config 化 (M6.3 等の variant)
- **`docs/open-questions.md` への追記要否**: 不要 (BAND_BOUNDS / register_buffer 設計判断は ticket 内で解決、`docs/open-questions.md` は論文事実確認の場であり実装判断は ticket 内で解決)
