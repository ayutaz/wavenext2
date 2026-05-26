---
id: T-M2.2
title: Multi-Scale Discriminator (MSD ×3、MPD なし、WaveFit-PT 準拠)
milestone: M2
phase: M2
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.2]
blocks: [T-M2.3, T-M2.5]
related_docs:
  - docs/milestones.md#m22-discriminator-srcwavenext2modelsdiscriminatorpy
  - docs/architecture.md
  - docs/open-questions.md
---

# T-M2.2: Multi-Scale Discriminator (MSD ×3、MPD なし、WaveFit-PT 準拠)

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.2](../milestones.md#m22-discriminator-srcwavenext2modelsdiscriminatorpy)
> **依存**: [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M2.3](T-M2.3-losses.md), [T-M2.5](T-M2.5-train-gan.md)

## 1. タスク目的とゴール

### 目的
GAN-WaveNeXt 2 の adversarial 訓練に必要な **Multi-Scale Discriminator (MSD)** を `src/wavenext2/models/discriminator.py` に実装する。WaveFit-PyTorch 参考実装 (`src/model/discriminator.py`) に **完全準拠** し、3 つの sub-discriminator (NLayerDiscriminator) を `AvgPool1d(kernel=4, stride=2)` で段階的に downsample した波形に適用する MelGAN 系統の構造を採用する。論文の Discriminator 仕様は WaveFit と同一と明記されており (Section 4.1)、HiFi-GAN 系で使われる **MPD (Multi-Period Discriminator) は不使用** ([docs/open-questions.md §C5](../open-questions.md#c5-discriminator-仕様-wavefit-と同一-wavefit-pt-srcmodeldiscriminatorpy) で確定)。

これにより:
- T-M2.3 (Loss) は `D(real)`, `D(fake)` の **3 sub-D の logits リスト** から hinge GAN loss を計算可能になる
- T-M2.3 (Feature matching loss) は **各 sub-D の 7 層中間特徴のリスト** から L1 distance を計算可能になる
- T-M2.5 (Training script) は `discriminator` を `nn.Module` として 1 個保持し、`opt_D = AdamW(D.parameters(), lr=2e-4, ...)` で標準的に最適化できる

論文の核となる "WaveFit と同一の discriminator" を **コードとして単一クラスに閉じ込め**、後続の Loss / Training から具体的な構造に依存しない設計にする。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/models/discriminator.py` に `MultiScaleDiscriminator` クラスが実装され、`from wavenext2.models.discriminator import MultiScaleDiscriminator` で import 可能
- [ ] 入力 `(B, 1, T)` の波形に対して **3 つの sub-discriminator** から `(logits, features)` の tuple リストを返す
- [ ] 各 sub-discriminator は 7 層の Conv1d + LeakyReLU で構成され、入力 conv (k=15, ch=1→16) + downsampling layer × 4 (k=41, stride=4, groups conv) + 終端 conv 2 つの組み合わせ
- [ ] 隣接 sub-D 間で `AvgPool1d(kernel_size=4, stride=2, padding=1)` により audio を downsample (`MelGAN` 慣例、`WaveFit-PT` 準拠)
- [ ] 全 Conv1d に `weight_norm` を適用 (`HiFi-GAN` / `WaveFit-PT` 準拠、spectral_norm ではない)
- [ ] LeakyReLU の slope は **0.2** (`WaveFit-PT` 準拠)
- [ ] **MPD (MultiPeriodDiscriminator) を含まない** (MSD のみ)
- [ ] `tests/test_discriminator.py` の Unit テストが全 pass (shape / param count sanity / channel 進行 / hinge GAN 互換性 / gradient flow / determinism)
- [ ] 参考実装 (WaveFit-PT, ParallelWaveGAN) を **コピーしていない** (CLAUDE.md ポリシー準拠、構造は参考だが変数名・実装順序・コメントは独自)
- [ ] `docs/milestones.md` §M2.2 Acceptance criteria 全 3 項目クリア
- [ ] `tests/test_discriminator.py` の `pytest.skip` placeholder (T-M0.2 配置) を実テストに置き換え、`uv run pytest tests/test_discriminator.py` が全 pass

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規実装 (T-M0.2 で空 stub として配置済み):
  - `src/wavenext2/models/discriminator.py` (本実装)
  - `tests/test_discriminator.py` (本実装、`pytest.skip` placeholder を置換)
- 編集:
  - `src/wavenext2/models/__init__.py` (`__all__` に `"MultiScaleDiscriminator"` を追加、必要なら re-export)
  - `docs/milestones.md` §M2.2 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` のステータス更新

### 2.2 主要構造

#### import 形式 (T-M0.2 で確定)
```python
from wavenext2.models.discriminator import MultiScaleDiscriminator
```

#### クラスシグネチャ

```python
"""discriminator.py — Multi-Scale Discriminator (MSD ×3、MPD なし、WaveFit-PT 準拠)。

論文 Section 4.1 で「Discriminator は WaveFit と同一」と明記されており、参考実装 WaveFit-PyTorch
の MSD のみ構成 (`src/model/discriminator.py`) を採用する。MelGAN 流の 3 段スケールで、
隣接 sub-D 間で AvgPool1d により audio を downsample する。

設計詳細:
- 3 sub-discriminator (NLayerDiscriminator)、各 7 層
- 隣接 sub-D 間: AvgPool1d(kernel=4, stride=2, padding=1)
- 各 sub-D 内: 入力 conv (k=15, ch=1→16) + downsampling × 4 (k=41, stride=4) + 終端 conv 2 つ
- 全 Conv1d に weight_norm
- LeakyReLU(0.2)
- 出力: 3 sub-D × (final logits, 中間特徴 list)

詳細は docs/architecture.md §6 / docs/open-questions.md §C5 / docs/training.md §2.3 を参照。
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class NLayerDiscriminator(nn.Module):
    """単一 scale 用 discriminator (MelGAN 流 NLayerDiscriminator)。

    7 層の Conv1d + LeakyReLU で audio を classify する。FM loss 用に各層の中間特徴も返す。
    """

    def __init__(
        self,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4,
        max_channels: int = 1024,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # Layer 0: 入力 conv (kernel=15, ch=1→ndf)
        # WaveFit-PT: ReflectionPad1d(7) → Conv1d(1, 16, k=15) → LeakyReLU(0.2)
        layers: list[nn.Module] = []
        layers.append(
            nn.Sequential(
                nn.ReflectionPad1d(7),
                weight_norm(nn.Conv1d(1, ndf, kernel_size=15)),
            )
        )

        # Layers 1〜n_layers: depthwise-separable conv で downsample
        # kernel = downsampling_factor * 10 + 1 (e.g., stride=4 → kernel=41)
        # stride = downsampling_factor
        # groups = nf_prev // 4 (グループ畳み込み)
        # channels は 2 倍ずつ増加、max_channels で頭打ち
        nf = ndf
        for i in range(n_layers):
            nf_prev = nf
            nf = min(nf_prev * downsampling_factor, max_channels)
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        nf_prev,
                        nf,
                        kernel_size=downsampling_factor * 10 + 1,
                        stride=downsampling_factor,
                        padding=downsampling_factor * 5,
                        groups=nf_prev // 4,
                    )
                )
            )

        # Layer 5: channel 倍化 + LeakyReLU (downsample なし)
        nf_prev = nf
        nf = min(nf_prev * 2, max_channels)
        layers.append(
            weight_norm(
                nn.Conv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2)
            )
        )

        # Layer 6: 最終出力 conv → 1ch (no activation 後段、logits)
        layers.append(
            weight_norm(nn.Conv1d(nf, 1, kernel_size=3, stride=1, padding=1))
        )

        # 個別レイヤを保持して FM loss 用に中間特徴を取り出せるようにする
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            x: (B, 1, T) 波形。

        Returns:
            (logits, features)
                logits:   (B, 1, T') 最終出力 (hinge GAN loss 用、activation なし)
                features: list of (B, C_i, T_i) 各層 (入口 conv + 4 downsample + channel 倍化 conv)
                          の中間特徴 (FM loss 用、L1 比較)。最終 logits 層は含めない。
        """
        features: list[torch.Tensor] = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                # 最終 conv (logits) 以外は LeakyReLU 通過、中間特徴として収集
                h = F.leaky_relu(h, self.leaky_relu_slope)
                features.append(h)
        # 最終 logits は activation なし (hinge GAN loss 互換)
        return h, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD ×3、WaveFit-PT 準拠、MPD なし)。

    3 つの NLayerDiscriminator を保持し、各 sub-D の前段で AvgPool1d により
    入力 audio を段階的に downsample する。出力は 3 sub-D × (logits, features) の list。
    """

    NUM_D: int = 3                       # sub-discriminator 数 (WaveFit-PT 準拠)
    POOL_KERNEL: int = 4                 # AvgPool1d kernel_size
    POOL_STRIDE: int = 2                 # AvgPool1d stride
    POOL_PADDING: int = 1                # AvgPool1d padding (MelGAN 慣例)

    def __init__(
        self,
        num_D: int = NUM_D,
        ndf: int = 16,
        n_layers: int = 4,
        downsampling_factor: int = 4,
        max_channels: int = 1024,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_D = num_D
        self.sub_discriminators = nn.ModuleList(
            [
                NLayerDiscriminator(
                    ndf=ndf,
                    n_layers=n_layers,
                    downsampling_factor=downsampling_factor,
                    max_channels=max_channels,
                    leaky_relu_slope=leaky_relu_slope,
                )
                for _ in range(num_D)
            ]
        )
        self.pool = nn.AvgPool1d(
            kernel_size=self.POOL_KERNEL,
            stride=self.POOL_STRIDE,
            padding=self.POOL_PADDING,
            count_include_pad=False,
        )

    def forward(
        self, x: torch.Tensor
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """Args:
            x: (B, 1, T) 波形 (`[-1, 1]` 範囲、Generator 出力 or GT 波形)。

        Returns:
            list of length num_D (=3) of (logits, features) tuples.
                logits:   (B, 1, T_k') 最終 logits (k 番目 sub-D)
                features: list of intermediate features (L1 FM loss 用)
        """
        outputs: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        h = x
        for k, sub_d in enumerate(self.sub_discriminators):
            logits, feats = sub_d(h)
            outputs.append((logits, feats))
            if k < self.num_D - 1:
                # 次の sub-D に渡す前に audio を downsample
                h = self.pool(h)
        return outputs

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiScaleDiscriminator":
        """YAML config dict から生成 (factory パターン、M1 横断テーマと整合)。

        Args:
            cfg: dict with keys among {num_D, ndf, n_layers, downsampling_factor,
                                       max_channels, leaky_relu_slope}.
        """
        allowed = {"num_D", "ndf", "n_layers", "downsampling_factor",
                   "max_channels", "leaky_relu_slope"}
        filtered = {k: v for k, v in cfg.items() if k in allowed}
        return cls(**filtered)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `num_D` (sub-discriminator 数) | **3** | docs/architecture.md §6, docs/open-questions.md §C5 (WaveFit-PT) |
| `ndf` (base channels) | **16** | docs/open-questions.md §C5 |
| `n_layers` (downsampling layers per sub-D) | **4** | docs/open-questions.md §C5 |
| `downsampling_factor` | **4** (kernel=41, stride=4) | docs/architecture.md §6 |
| `max_channels` | **1024** | docs/open-questions.md §C5 |
| `leaky_relu_slope` | **0.2** | docs/architecture.md §6 (LeakyReLU(0.2)) |
| `AvgPool1d kernel_size` | **4** | docs/architecture.md §6 |
| `AvgPool1d stride` | **2** | docs/architecture.md §6, docs/training.md §2.3 |
| `AvgPool1d padding` | **1** | MelGAN 慣例 (中央化 downsample) |
| `weight_norm` | 全 Conv1d に適用 | docs/open-questions.md §C5 |
| 入力 conv kernel | **15** (ch=1→16) | docs/open-questions.md §C5 |
| 入力 padding | `ReflectionPad1d(7)` | docs/open-questions.md §C5 (WaveFit-PT) |
| downsample conv kernel | **41** (`= 4*10+1`) | docs/architecture.md §6 |
| downsample conv groups | `nf_prev // 4` (グループ畳み込み) | docs/architecture.md §6 |
| 最終出力 ch | **1** (logits) | docs/architecture.md §6 |
| 出力 activation | なし (hinge GAN compatible logits) | docs/training.md §2.3 |
| 各 sub-D の Conv1d レイヤ数 | **7** (入口 + 4 downsample + channel 倍化 + 最終) | docs/architecture.md §6 |

### 2.4 アルゴリズム / 処理フロー

#### `MultiScaleDiscriminator.forward`
1. 入力 `x.shape == (B, 1, T)` を期待 (波形、`[-1, 1]` 範囲)
2. `outputs = []`、`h = x` で初期化
3. `for k in range(num_D=3)`:
   1. `logits_k, feats_k = sub_discriminators[k](h)` を呼び出し
   2. `outputs.append((logits_k, feats_k))` で結果を蓄積
   3. `k < num_D - 1` の間、`h = pool(h)` で audio を半分にダウンサンプル
4. `outputs` を返す (`len(outputs) == 3`)

#### `NLayerDiscriminator.forward`
1. 入力 `x.shape == (B, 1, T)` (k=0) or `(B, 1, T/2)` (k=1) or `(B, 1, T/4)` (k=2)
2. `features = []`、`h = x` で初期化
3. `for i, layer in enumerate(self.layers)` (= 7 layers):
   1. `h = layer(h)`
   2. `i < 6` (最終 logits 層以外) なら `h = leaky_relu(h, 0.2)` を適用し `features.append(h)`
4. 最終層 (`i == 6`) の `h` は **activation なしの logits** として返す
5. `return h, features` (`len(features) == 6`)

#### 不変条件
- 入力 channel 数は **1** (mono 波形)。stereo を渡すと shape error (上位で mono 化が責務、`docs/architecture.md` §6.6)
- 出力 logits は **activation 適用しない**。`tanh` / `sigmoid` を入れると hinge GAN loss が期待値範囲外になる
- 中間特徴 list の長さは 6 (= 7 layers - 1 logits 層)。FM loss は L1 で **3 sub-D × 6 layer = 18 個** の特徴を比較する形になる

### 2.5 各 sub-D の channel / kernel の進行 (sanity check 用)

WaveFit-PT 設定 (`ndf=16, n_layers=4, downsampling_factor=4, max_channels=1024`):

| Layer | type | in_ch | out_ch | kernel | stride | groups | LeakyReLU | feature 収集 |
|---|---|---|---|---|---|---|---|---|
| 0 | ReflectionPad + Conv | 1 | 16 | 15 | 1 | 1 | yes | yes |
| 1 | Conv (depthwise-sep) | 16 | 64 | 41 | 4 | 4 (= 16//4) | yes | yes |
| 2 | Conv | 64 | 256 | 41 | 4 | 16 (= 64//4) | yes | yes |
| 3 | Conv | 256 | 1024 | 41 | 4 | 64 (= 256//4) | yes | yes |
| 4 | Conv | 1024 | 1024 (clip) | 41 | 4 | 256 (= 1024//4) | yes | yes |
| 5 | Conv (channel 倍化, no downsample) | 1024 | 1024 (clip) | 5 | 1 | 1 | yes | yes |
| 6 | Conv (logits) | 1024 | 1 | 3 | 1 | 1 | **no** | **no** |

**memo**: Layer 4 で nf_prev=1024 から `min(1024*4, 1024) = 1024` で頭打ち。Layer 5 でも `min(1024*2, 1024) = 1024`。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `discriminator.py` 実装 + `tests/test_discriminator.py` 記述 | general-purpose |
| Reviewer | 1 | WaveFit-PT `src/model/discriminator.py` との **構造整合** + コピペ無し確認 + 論文 §4.1 整合 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_discriminator.py -v` 実行と Acceptance 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M2.1 Dataset, T-M2.3 Loss と独立に並列着手可能、互いに別ファイルを編集)
- 並列実行する場合の最大並列数: 3 (T-M2.1, T-M2.2, T-M2.3 を別 worker で並列)
- 注意: T-M2.3 (Loss) は本チケットの `MultiScaleDiscriminator.forward` の **戻り値仕様** (`list[(logits, features)]`) に依存。T-M2.3 着手前に本チケットの I/O 契約を固定すること

## 4. 提供範囲 (Scope)

### In Scope
- `MultiScaleDiscriminator` クラスの本実装 (single class、`nn.Module` 継承)
- `NLayerDiscriminator` 内部クラス (sub-D 単体) の実装
- `weight_norm` 適用 (`torch.nn.utils.parametrizations.weight_norm` 新 API 使用、`torch 2.10+` で deprecated 警告なし)
- `AvgPool1d` 隣接 sub-D 間 downsample 処理
- 各層 LeakyReLU(0.2) 適用 + 中間特徴蓄積 (FM loss 用)
- 最終 logits は activation なし (hinge GAN 互換)
- `__init__.py` への `__all__` 追加 (`"MultiScaleDiscriminator"`)
- `tests/test_discriminator.py` の本実装 (`pytest.skip` 置換、milestones.md §M2.2 Acceptance を網羅)
- パラメータ数 sanity check テスト (例: 各 sub-D ≈ 16〜17M、合計 ≈ 50M の order が出ることを確認、桁外れは検出)
- forward + backward の動作確認テスト
- `MultiScaleDiscriminator.from_config(cfg)` factory メソッド (M1 横断テーマと整合)
- docstring (関数・クラス) を英文 + 日本語混在で記述

### Out of Scope
- Hinge GAN loss / Feature matching loss の本実装 → **T-M2.3** で実装
- MR-STFT loss → **T-M2.3** で実装
- Discriminator の training step (gradient update) → **T-M2.5** で実装
- Spectral normalization (`spectral_norm`) → 不採用 (WaveFit-PT は `weight_norm`、§6.1 で確定)
- LSGAN loss / non-saturating loss → 不採用 (hinge GAN のみ、`docs/training.md` §2.3)
- MultiPeriodDiscriminator (MPD) → **不採用** (`docs/open-questions.md` §C5 で確定)
- MultiResolutionSTFTDiscriminator (MS-STFT) → **不採用** (MelGAN 流 MSD のみ)
- Generator 側との連携 (`y_0 = generator(...); D(y_0)`) → T-M2.4 / T-M2.5
- Conditional discriminator (mel-conditioned) → 不採用 (WaveFit-PT は unconditional、`docs/architecture.md` §6)
- ONNX / TorchScript export → M6 以降の最適化チケットで検討
- AMP / `torch.compile` 対応 → 本チケットでは触らない (M5/M6 で再評価)

### Deliverable
- ファイル:
  - `src/wavenext2/models/discriminator.py` (新規実装、約 160〜200 行)
  - `tests/test_discriminator.py` (新規実装、約 120〜180 行)
  - `src/wavenext2/models/__init__.py` (re-export 追加)
- 関数 / クラス:
  - `class MultiScaleDiscriminator(nn.Module)`
  - `class NLayerDiscriminator(nn.Module)` (internal)
  - `MultiScaleDiscriminator.from_config(cfg)` classmethod
- ドキュメント差分:
  - `docs/milestones.md` §M2.2 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M2.2 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_discriminator.py`)

#### shape / I/O 契約
- [ ] `test_msd_output_structure`: 入力 `(2, 1, 16384)` → `outputs` が `list` of length 3、各要素が `(logits, features)` の tuple
- [ ] `test_msd_logits_shape`: 各 sub-D の logits が `(2, 1, T_k)` で `T_k` が k 増加とともに小さくなる (downsample 確認)
- [ ] `test_msd_features_count`: 各 sub-D の `features` リスト長が **6** (7 layers - 1 logits 層)
- [ ] `test_msd_input_assertion`: `(B, 2, T)` (stereo) や `(B, T)` (channel dim 欠落) を渡すと PyTorch 標準のエラーが raise される (本クラスは明示的な assert は入れない、`nn.Conv1d` の標準エラーに任せる)
- [ ] `test_nlayer_features_intermediate_only`: `NLayerDiscriminator` の `features` に **最終 logits 層は含まない**、最終 logits は activation なし
- [ ] `test_msd_downsample_factor`: k=0 の logits T と k=1 の logits T の比が **約 2 倍** (AvgPool1d stride=2 で確認)、k=1 と k=2 も同様

#### channel / kernel 進行
- [ ] `test_nlayer_channel_progression`: `NLayerDiscriminator` の各 `nn.Conv1d` の `in_channels` / `out_channels` が §2.5 表と一致 (Layer 0: 1→16, Layer 1: 16→64, ..., Layer 6: 1024→1)
- [ ] `test_nlayer_kernel_size`: Layer 0: kernel=15, Layer 1〜4: kernel=41, Layer 5: kernel=5, Layer 6: kernel=3
- [ ] `test_nlayer_stride`: Layer 1〜4: stride=4, それ以外: stride=1
- [ ] `test_nlayer_groups`: Layer 1〜4 の groups が `nf_prev // 4` (16//4=4, 64//4=16, 256//4=64, 1024//4=256)

#### weight_norm 適用確認
- [ ] `test_weight_norm_applied`: 全 `nn.Conv1d` が `weight_norm` でラップされている。**API 非依存判定** として `torch.nn.utils.parametrize.is_parametrized(conv, "weight")` を使用 (旧 API `hasattr(conv, "weight_v")` と新 API `parametrizations.weight` の両対応で CI flakiness を回避)
- [ ] `test_no_spectral_norm`: `spectral_norm` が **適用されていない** (parametrizations に `spectral_norm` 名が無い)

#### パラメータ数 sanity check
- [ ] `test_param_count_sanity`: 全体 `sum(p.numel() for p in model.parameters())` が **桁レベルで妥当** (e.g., 30M〜80M の range、order check)。WaveFit-PT 設定で実測値を pin (1 回計算してテストに hardcode、±2% 許容)。**PyTorch version 依存性に注意**: hardcode 値は PyTorch のレイヤ実装変更で揺れる可能性があり、本チケットでは `torch >= 2.10` pin (`pyproject.toml`) で対処、実測した pin 値 (合計 ~50M order) は実装着手後に固定
- [ ] `test_per_sub_d_param_count_equal`: 3 sub-D のパラメータ数が完全一致 (`set(sub_d_params) == 1` を確認)

#### gradient flow / 学習互換性
- [ ] `test_gradient_flow_logits`: `loss = sum(logits.mean() for logits, _ in model(x)); loss.backward()` で全パラメータに `grad is not None`
- [ ] `test_gradient_flow_features`: `loss = sum(sum(f.mean() for f in feats) for _, feats in model(x)); loss.backward()` で中間特徴経由の grad も流れる
- [ ] `test_hinge_gan_compatible_range`: real 入力 `torch.randn(2, 1, 16384) * 0.1` (典型的な audio 振幅) で logits が `[-inf, +inf]` の任意値を取りうる (sigmoid / tanh で clamp されていないことを確認、`logits.abs().max() > 0.01` 等)

#### determinism / モード切替
- [ ] `test_deterministic`: 同 seed / 同入力で出力が一致 (`torch.manual_seed` 固定後 2 回 forward)
- [ ] `test_eval_train_equivalent`: `.eval()` と `.train()` で出力が一致 (BatchNorm / Dropout を含まないため、変わったらバグ)

#### factory / config
- [ ] `test_from_config`: `MultiScaleDiscriminator.from_config({"num_D": 3, "ndf": 16, ...})` で生成可能、余分な key は無視される (drift 耐性)

#### device / B=1
- [ ] `test_device_cpu`: CPU で全テスト pass (CUDA 不要環境でも CI が通る)
- [ ] `test_batch_size_one`: B=1 で forward + backward が成功 (BatchNorm を含まないため問題ないはず、edge case 確認)

### 5.2 e2e / 結合テスト
- [ ] T-M2.3 (Loss) で `MultiScaleDiscriminator` 出力から hinge GAN loss + FM loss が計算可能 (smoke import で確認、本チケットでは未実装でも import 成立だけ確認)
- [ ] T-M2.5 (training) で `opt_D = AdamW(D.parameters(), ...)` で標準的に最適化できる (smoke、T-M2.5 で実施)
- [ ] **`tests/test_discriminator.py::test_real_audio`**: LibriTTS-R sample 1 件で実行 (T-M0.3 完了後)、`@pytest.mark.slow` でデフォルト除外。実音声 (例: 16384 samples) を D に通して forward + backward が通る。T-M2.1 dataset 未完成期間は `pytest.skip` で skip

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.2 より転記)
- [ ] 入力 `(B, 1, T)` で 3 つの sub-discriminator から出力リストを返す
- [ ] 各 sub-discriminator の中間特徴も返す (FM loss 用)
- [ ] AvgPool1d で隣接 sub-discriminator 間 downsample

### 5.4 追加 acceptance (本チケット独自)
- [ ] **MPD を含まない**: `model.modules()` 走査で `MultiPeriodDiscriminator` クラス相当が存在しないこと (テスト名 `test_no_mpd`)
- [ ] **MS-STFT discriminator を含まない**: STFT を内部で実行する Module が存在しない (テスト名 `test_no_msstft`)
- [ ] `MultiScaleDiscriminator.NUM_D == 3` クラス属性として公開され、テストから参照可能
- [ ] `pytest tests/test_discriminator.py` が **exit code 0** で完了 (skip 0 件 / GPU/slow マーカー除く、すべて pass)
- [ ] `from wavenext2.models import MultiScaleDiscriminator` で import 可能 (`__init__.py` 経由)

### 5.5 テスト戦略 (M2 phase で確立)

- **CI 時間目標**: CPU で全テスト合計 **< 20 秒** (M2 全体 60s 目標から)。各 sub-D は ~17M params だが forward は B=2 / T=16384 で軽量
  - **必須最適化**: `scope="module"` fixture で `MultiScaleDiscriminator` インスタンスを再利用 (各テストで `init` し直さない)
- **API 非依存 weight_norm テスト**: `torch.nn.utils.parametrize.is_parametrized(conv, "weight")` を使用 (旧 API `weight_v` / 新 API `parametrizations.weight` の両方を意識せず判定可能、CI flakiness の温床を回避)
- **GPU テスト分離**: `@pytest.mark.gpu` マーカーで GPU テストを CI runner 別に分離。CPU only CI では `pytest -m "not gpu"` で除外
- **`@pytest.mark.slow` マーカー**: `test_real_audio` 等 LibriTTS-R 実音声を使うテストは `slow` でデフォルト除外
- **coverage 目標**: 本チケットのカバレッジ目標 **90%** (forward / `_init`、`from_config` の全分岐をカバー)

## 6. 懸念事項

### 6.1 技術的リスク

#### `weight_norm` API の deprecation 警告
- **問題**: PyTorch 2.x で `torch.nn.utils.weight_norm` が deprecated、新 API は `torch.nn.utils.parametrizations.weight_norm`
- **対応**: 新 API (`parametrizations.weight_norm`) を **採用**。state_dict の key 形式が変わる (`weight_v`/`weight_g` → `parametrizations.weight.original0`/`original1`) ため、warm-start から WaveFit 重みをロードする場合は **マッピング層が必要** (本チケットの範囲外、T-M5 / T-M6 で warm-start 試行時に判明したら別チケット)
- **検証**: `test_weight_norm_applied` で **新 API のいずれの判定方法** (parametrizations 経由) で確認

#### `AvgPool1d` の padding 仕様
- **問題**: WaveFit-PT 参考実装で `AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)` だが、論文 §6 では `kernel=4, stride=2` のみ明記、padding は不明
- **対応**: MelGAN 慣例 (`padding=1, count_include_pad=False`) を採用。テストで T_k の進行が **約 2 倍ずつ** になることを確認 (`test_msd_downsample_factor`)
- **不整合検知**: padding を 0 にすると T_1 = (T - 4)/2 + 1 で T=16384 → T_1 = 8191 となり、padding=1 の `T_1 = 8192` から 1 ずれる。**T-M2.3 FM loss の長さ整合性で気付く可能性**

#### `groups = nf_prev // 4` の整合性
- **問題**: ndf=16 で Layer 1: `groups = 16//4 = 4` で `Conv1d(16, 64, k=41, groups=4)` は `16 % 4 == 0 and 64 % 4 == 0` で OK だが、ndf を変えると groupable でない組み合わせが起きうる (例: ndf=10 だと `10//4 = 2`、`Conv1d(10, 40, ..., groups=2)` で `10 % 2 == 0` だが `40 % 2 == 0` で OK)
- **対応**: 本チケットでは `ndf=16` 固定 default、変更時は `assert in_ch % groups == 0 and out_ch % groups == 0` を runtime で検証 (`nn.Conv1d` 内部の assertion に任せる)
- **テスト**: `test_nlayer_groups` で `ndf=16` の値を pin

#### LeakyReLU slope の確定
- **問題**: HiFi-GAN は 0.1、MelGAN / WaveFit-PT は 0.2。論文記述では曖昧
- **対応**: WaveFit-PT 準拠で **0.2** を採用 (`docs/architecture.md` §6 で確定)。`leaky_relu_slope=0.2` を default、引数で上書き可能

#### 中間特徴を返すインターフェース判断
- **問題**: FM loss 用に中間特徴を返す方法として:
  - **案 A (採用)**: `forward(x) -> list[(logits, features)]` で同時に返す
  - 案 B: `forward(x) -> list[logits]` と `extract_features(x) -> list[list[features]]` の 2 method 分離
  - 案 C: Forward hook で外部から取得 (`register_forward_hook`)
- **採用根拠**: 案 A は 1 forward で済むため計算コスト最小、PyTorch コミュニティ標準 (HiFi-GAN, MelGAN, WaveFit-PT 全て案 A)、テストもシンプル
- **却下**: 案 B は同じ計算を 2 回行うため非効率、案 C は API が暗黙的でデバッグ困難

#### 7 層 = Conv1d 7 個の正確な spec
- **問題**: WaveFit-PT `src/model/discriminator.py` を実装時に再確認すべき (本チケット §2.5 表は docs から逆算した値)
- **対応**: 実装時に WaveFit-PT 該当ファイルの **構造のみ目視確認** (コピペは禁止、CLAUDE.md ポリシー)、§2.5 表との差分を Implementer が報告。差分があれば本チケットの §2.5 を **真値で更新** + テストの期待値も更新
- **検知**: パラメータ数 sanity check (`test_param_count_sanity`) で大きく外れたら spec ミス

#### Audio 振幅と weight_norm の安定性
- **問題**: Generator 出力は `[-1, 1]`、GT 波形も sox `norm` で `[-1, 1]` 周辺だが、weight_norm の initial weight 分布によっては初期 logits が極端に大きく / 小さくなり hinge GAN が dead
- **対応**: 本チケットでは初期化は PyTorch デフォルトに任せる (Kaiming uniform)。M2.6 smoke test で `logits.abs().mean() > 100` 等の極端な initial 出力を観測したら、`nn.init.normal_(weight, std=0.02)` 等の追加 init を検討 (本チケット範囲外、smoke でフィードバック)
- **検知**: `test_hinge_gan_compatible_range` で logits の order of magnitude を粗くチェック

#### B=1 動作
- **問題**: WaveFit-PT は B>=2 で実装されている可能性、B=1 で groups conv の挙動を確認
- **対応**: `test_batch_size_one` で明示的に検証

#### D 強すぎ問題と D capacity の関係
- **問題**: hinge GAN で D が強すぎると G が学習しない (D が 0 loss に張り付き G の更新シグナルが消える)。これは D update 頻度調整 (T-M2.5 §8.1) と並んで **D capacity の選択** とも表裏一体の関係にある
- **対応**: 本チケットは default `num_D=3, ndf=16` (WaveFit-PT 準拠) で実装するが、M2.6 / M5.1 smoke で D dominate (`loss_D < 0.01`, `loss_G_adv` 発散 / 停滞) が観測されたら、capacity ablation として `num_D=2` または `ndf=8` を試す余地を残す。`MultiScaleDiscriminator(num_D=2)` で動作することを `from_config` の余分 key 無視と並ぶ柔軟性として担保
- **再評価トリガー**: M5.1 で hinge GAN D が dominate (`loss_D` が極小、`loss_G_adv` が学習しない) → `num_D=2` ablation を実施
- **検知**: M2.6 smoke で `loss_D` / `loss_G_adv` の比率を観測

#### `param_count_sanity` の hardcode 値の PyTorch version 依存
- **問題**: WaveFit-PT 設定で pin した param count 値は、PyTorch のレイヤ実装変更 (例: `weight_norm` parametrization 内部構造の変更) で揺れうる
- **対応**: `pyproject.toml` で `torch >= 2.10` pin することで version flakiness を抑える。WaveFit-PT で実測した pin 値 (合計 ~50M order) は **実装着手後に固定** し、テストで ±2% 許容で照合。PyTorch メジャー version 更新時はテスト失敗で気付ける形を維持
- **検知**: CI で `test_param_count_sanity` 失敗時に、PyTorch version 差分を確認 (`uv pip list | grep torch`)

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み** (§C5, §C6)
  - MSD ×3 with AvgPool1d downsample: `docs/architecture.md` §6 で確定
  - MPD なし: `docs/open-questions.md` §C5 で確定 (WaveFit-PT 準拠)
  - LeakyReLU(0.2): `docs/architecture.md` §6 で確定
  - weight_norm 全 Conv1d 適用: `docs/open-questions.md` §C5 で確定
  - 出力 activation なし (hinge GAN compatible): `docs/training.md` §2.3 で確定
- 本チケットで決定する項目 (本ドキュメント内で確定):
  - `AvgPool1d padding=1` 採用 (MelGAN 慣例)
  - `weight_norm` 新 API (`torch.nn.utils.parametrizations.weight_norm`) 採用
  - 中間特徴の返し方: forward 戻り値の tuple に含める (案 A)
  - `nn.init` カスタム初期化なし (PyTorch デフォルトに任せる、smoke でフィードバック予定)
  - 入力 conv の padding: `ReflectionPad1d(7)` (WaveFit-PT 準拠)

### 6.3 他チケットとの整合性

- **T-M2.3 (Loss)** との整合:
  - 期待入力: `outputs = D(x)` で `outputs: list[(logits, features)]`、`len(outputs) == 3`
  - hinge GAN loss は `logits` リストのみ使用、FM loss は `features` リストのみ使用
  - 不整合があった場合は T-M2.3 側を本チケットの I/O 契約に合わせて修正
- **T-M2.4 (GAN モデル)** との整合:
  - `GANWaveNext2.forward(...)` の出力 `y_0` を `D(y_0.unsqueeze(1))` で評価
  - `y_0.shape == (B, T_audio)` から `(B, 1, T_audio)` への変換は **呼び出し側 (T-M2.5)** の責務 (本クラスは shape チェックを strict にしない)
- **T-M2.5 (Training script)** との整合:
  - `opt_D = AdamW(D.parameters(), lr=2e-4, betas=[0.8, 0.99], weight_decay=1e-3)` で標準的に最適化
  - `D.train() / D.eval()` 切替は副作用なし (BatchNorm / Dropout 不含、§5.1 `test_eval_train_equivalent` で担保)
- **T-M2.6 (smoke)** との整合:
  - 1 utterance での hinge GAN loss が finite (NaN / inf でない) ことを確認 → 本チケットでは「初期 logits が極端でない」ことだけ test、本格的な訓練安定性は T-M2.6 で評価

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §6 (Discriminator 構造) と実装が完全一致
- [ ] `docs/open-questions.md` §C5 (Discriminator 仕様、WaveFit-PT 準拠) と整合
- [ ] `docs/training.md` §2.3 (hinge GAN compatible logits, FM L1 features) と整合
- [ ] MSD ×3 のみ実装、**MPD・MS-STFT discriminator は実装されていない** (§5.4 `test_no_mpd`, `test_no_msstft` で確認)
- [ ] 全 Conv1d に `weight_norm` (`torch.nn.utils.parametrizations.weight_norm`) が適用されている
- [ ] LeakyReLU slope=0.2 (`docs/architecture.md` §6)
- [ ] AvgPool1d kernel=4, stride=2 で隣接 sub-D 間 downsample
- [ ] 各 sub-D の channel/kernel 進行が §2.5 表と一致
- [ ] 最終 logits は activation なし (hinge GAN 互換)
- [ ] 中間特徴 list の長さは **6** (7 layers - 1 logits)
- [ ] Acceptance criteria §5.3 全 3 項目クリア、§5.4 追加 acceptance 全クリア
- [ ] Unit テスト 全 pass (`uv run pytest tests/test_discriminator.py -v`)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語混在、`from __future__ import annotations`)
- [ ] エラー処理: stereo / channel 欠落入力時の挙動は PyTorch 標準エラーに委譲 (silent failure なし)
- [ ] パラメータ数 / メモリ消費が想定内 (sanity check の order が桁レベルで合う)
- [ ] **参考実装 (WaveFit-PT, ParallelWaveGAN, MelGAN) をコピーしていない** (CLAUDE.md 末尾の方針)。変数名・コメント・実装順序が独自で、`docs/architecture.md` の論述から再構成しているか
- [ ] `from_config(cfg)` factory メソッドが M1 横断テーマと整合
- [ ] `__init__.py` の `__all__` に `MultiScaleDiscriminator` が追加され、`NLayerDiscriminator` は internal helper として **export されていない**

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M2) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **MSD ×3 のみ (MPD なし、MS-STFT なし)、WaveFit-PT 完全準拠**
  - 理由 (3 観点):
    - (a) 論文 §4.1 で「Discriminator は WaveFit と同一」と明記、`docs/open-questions.md` §C5 で WaveFit-PT 参考実装の MSD ×3 のみ構成と確定
    - (b) HiFi-GAN 系の MPD ×5 を追加すると **論文と異なる結果** になり、再現実装の意義 (論文の主張を検証する) が損なわれる
    - (c) BigVGAN 等の最近の vocoder では MR-STFT discriminator が使われるが、WaveFit / WaveNeXt 2 は MSD のみで品質を達成しており、本チケットでは **論文準拠を優先**

- **`DiscriminatorOutput(logits, features)` NamedTuple 導入 (M2 phase review で採用昇格)**
  - 理由: T-M2.3 §6.1 critical の `num_features=18 vs 21` 問題 (FM loss で features の個数を取り違える / logits を features 側に含めるかどうか曖昧) の **根本原因が戻り値仕様の不明瞭さ**。`list[tuple[Tensor, list[Tensor]]]` で素の tuple を返すと「features に logits を含めるべきか」が呼び出し側に委ねられて bug の温床になる
  - 仕様: `DiscriminatorOutput` は `typing.NamedTuple` で `logits: torch.Tensor` と `features: list[torch.Tensor]` を持つ。**`features` に logits は含めない** (中間特徴 6 個のみ、3 sub-D × 6 layer = 18 個の features を T-M2.3 で L1 比較する)
  - `MultiScaleDiscriminator.forward(x) -> list[DiscriminatorOutput]` (length=3)
  - 後方互換: `DiscriminatorOutput` は NamedTuple なので `(logits, features)` の tuple unpacking がそのまま動く、既存コード `for logits, features in D(x):` は無修正
  - 副次効果: docstring / type hint が読みやすくなり、T-M2.3 のテストで `output.features` で属性アクセス可能
  - **却下していた根拠 (素の tuple)** : 当初は「Python 標準 tuple で軽量化を優先」だったが、`num_features` 誤認の bug リスクが軽量化メリットを上回ると M2 phase review で判明

- **`unsqueeze(1)` 責務を D 側で defensive に受け入れる**
  - 理由: Generator 出力 `(B, T)` を `(B, 1, T)` に unsqueeze する責務を呼び出し側 (T-M2.5) に投げると、忘れた瞬間 `nn.Conv1d` の cryptic な shape error (`RuntimeError: Expected 3D ... input ...`) しか出ず、デバッグが困難
  - 仕様: `MultiScaleDiscriminator.forward` 冒頭で `if x.dim() == 2: x = x.unsqueeze(1)` を入れて、`(B, T)` と `(B, 1, T)` の両方を defensive に受け入れる。stereo (`(B, 2, T)`) は引き続き `Conv1d in_channels=1` mismatch エラーに任せる
  - 副次効果: T-M2.5 で `D(audio.unsqueeze(1))` が不要、`D(audio)` で OK (`audio.shape == (B, T)` でも `(B, 1, T)` でも)
  - **却下していた根拠** : 当初は「shape チェックを strict にしない、呼び出し側責務」だったが、`(B, T)` を受け取ったときの cryptic error が後続 (T-M2.5 / T-M2.6 smoke) のデバッグ時間を増やすと判断

- **中間特徴を forward 戻り値の NamedTuple に含める (案 A、NamedTuple 化で更に明確化)**
  - 理由: 1 forward で済む、コミュニティ標準、テスト容易性

- **`weight_norm` 新 API (parametrizations) 採用**
  - 理由: torch 2.10+ で旧 API は deprecated、warning なしで動作
  - **テスト判定は API 非依存**: `torch.nn.utils.parametrize.is_parametrized(conv, "weight")` を使用 (旧 API `hasattr(conv, "weight_v")` と新 API `parametrizations.weight` の両対応で CI flakiness の温床を回避)

#### Deprecated (代替案、却下根拠)

1. **MPD ×5 + MSD ×3 (HiFi-GAN/BigVGAN 風)**:
   - メリット: HiFi-GAN で実績、periodic features 補強で品質向上の可能性
   - 却下: **WaveFit-PT 準拠から逸脱**、論文と異なる結果になり再現実装の意義喪失
   - 再評価トリガー: **M6.3 ablation** で MOS 差を測定して論文との差が `> 0.2` なら MPD 追加を検討

2. **CoMoSpeech / EnCodec 風の MS-STFT discriminator**:
   - メリット: 周波数領域の構造を直接学習、最近の SOTA で実績
   - 却下: WaveFit-PT に存在せず、論文準拠から逸脱
   - 再評価トリガー: **M6.3 ablation** で MOS が論文に届かない場合に MS-STFT D を補足検討

3. **Multi-period のみ (MSD なし)**:
   - メリット: 計算量削減
   - 却下: WaveFit-PT の MSD を完全に置換するため、論文との直接比較が困難
   - 再評価トリガー: なし (この方向への変更は本リポジトリの目的と整合しない)

4. **`spectral_norm` を `weight_norm` の代わりに使用**:
   - メリット: GAN の安定性向上で実績
   - 却下: HiFi-GAN / WaveFit-PT 全て `weight_norm`、論文準拠から逸脱
   - 再評価トリガー: M2.6 smoke で訓練不安定 (`loss_D` divergence) なら `spectral_norm` 試行 (`docs/architecture.md` §6.1 リスク表)

5. **Conditional discriminator (mel-conditioned)**:
   - メリット: condition-aware で品質向上の可能性
   - 却下: WaveFit-PT は unconditional、論文準拠
   - 再評価トリガー: なし

6. **中間特徴を `extract_features(x)` separate method**:
   - メリット: API が明確、forward が単純化
   - 却下: 同じ計算を 2 回行うため非効率 (`forward + extract_features`)

7. **中間特徴を `register_forward_hook` で外部取得**:
   - メリット: 内部 API を変更せず取れる
   - 却下: 暗黙的でデバッグ困難、PyTorch コミュニティ慣例から外れる

8. **`weight_norm` 旧 API (`torch.nn.utils.weight_norm`)**:
   - メリット: WaveFit-PT の state_dict と bit-exact 互換
   - 却下: torch 2.10+ で deprecated 警告、将来削除予定
   - 再評価トリガー: M5 で warm-start (WaveFit 重みロード) が必要なら state_dict マッピング層を別チケットで実装

9. **AvgPool1d ではなく Conv1d stride=2 で downsample**
   - メリット: 学習可能で表現力が高い
   - 却下: MelGAN / WaveFit-PT は `AvgPool1d`、論文準拠
   - 再評価トリガー: M6.3 ablation で検討余地

10. **kernel size を 41 ではなく 15 / 9 等で短縮**
    - メリット: メモリ削減、速度向上
    - 却下: WaveFit-PT で `kernel = stride*10 + 1` の慣例、論文準拠

11. **`groups` を `nf_prev // 4` ではなく depthwise (`groups = nf_prev`)**:
    - メリット: パラメータ削減
    - 却下: WaveFit-PT は `// 4` の depthwise-separable 中間設計

12. **LeakyReLU slope を 0.1 (HiFi-GAN 風)**:
    - メリット: HiFi-GAN で実績
    - 却下: WaveFit-PT 準拠で 0.2

13. **D capacity ablation (`num_D=2`)**:
    - メリット: hinge GAN の D 強すぎ問題への対処。Discriminator capacity 選択と D 強度バランスは **表裏一体** であり、D update 頻度調整 (T-M2.5 §8.1) と並んで D capacity を ablation する余地
    - 採用: **再評価候補** として明示残し。default は WaveFit-PT 準拠 `num_D=3`、`MultiScaleDiscriminator(num_D=2)` で動作するよう `from_config` の引数化は維持
    - 再評価トリガー: **M5.1 で hinge GAN D が dominate** (`loss_D < 0.01` 張り付き、`loss_G_adv` 学習しない) → `num_D=2` ablation を実施
    - 関連: §6.1 「D 強すぎ問題と D capacity の関係」、T-M2.5 §8.1 D update 頻度調整

#### 再評価トリガー条件

| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| MSD ×3 のみ (MPD なし) | M6.3 ablation | MOS 差 > 0.2 なら MPD ×5 追加 |
| `weight_norm` | M2.6 smoke | 訓練 divergence なら `spectral_norm` |
| LeakyReLU(0.2) | M2.6 smoke | dead ReLU 多発なら 0.1 / GELU 試行 |
| AvgPool1d downsample | M6.3 ablation | 学習可能 downsample (`Conv1d stride=2`) 試行 |
| 中間特徴 forward 戻り値 | (済) M2 phase review | `DiscriminatorOutput` NamedTuple 採用昇格 |
| `weight_norm` 新 API | M5 warm-start | 旧 API state_dict との互換性が必要なら旧 API も追加サポート |
| D capacity (`num_D=3`) | M5.1 smoke | D dominate (`loss_D` 極小) なら `num_D=2` ablation |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切。
  - discriminator は `nn.Module` 1 個に閉じる thin layer であり、独立チケット化することで T-M2.3 (Loss) と T-M2.5 (training) が「D を使うだけ」に集中できる
  - 逆に、本チケットを T-M2.3 / T-M2.5 に統合すると、両方のチケットで同じ D を再実装するか、片方からのみ参照する変則的依存になる
- **別マイルストーンに移すべき部分はないか**: なし。M2 (GAN-WaveNeXt 2) の Loss と Training の前提として、M2 内に位置するのが自然。
- **インターフェース定義の見直し余地**:
  - **`MultiScaleDiscriminator.NUM_D` クラス属性**: T-M2.3 / T-M2.5 / FM loss の重み計算で `D.NUM_D` を参照可能。docstring だけだと変更検知が弱い
  - **`forward` 戻り値の typing**: (済) M2 phase review で `DiscriminatorOutput(logits, features)` NamedTuple を採用昇格。`num_features=18 vs 21` 取り違え問題の予防
  - **`from_config(cfg)` factory**: M1 横断テーマ (T-M1.2 / T-M1.3 / T-M1.4 / T-M1.5 / T-M1.6 と同じ) と一貫化

#### `unsqueeze(1)` 責務の defensive design (M2 phase review で追記)

- **問題**: Generator 出力 `y_0.shape == (B, T)` を `(B, 1, T)` に unsqueeze する責務が現状 **呼び出し側 (T-M2.5)** に投げられている。`y_0.unsqueeze(1)` を忘れた瞬間、`nn.Conv1d` の cryptic な shape error (`RuntimeError: Expected 3D ... input ...`) しか出ず、原因特定に時間を取られる
- **採用方針**: D 側で受け入れる方が **defensive** であり、本チケットで採用する:
  ```python
  def forward(self, x: torch.Tensor) -> list[DiscriminatorOutput]:
      if x.dim() == 2:
          x = x.unsqueeze(1)   # (B, T) → (B, 1, T), defensive
      # ... 以降は (B, 1, T) を前提に処理
  ```
- **理由**: silent failure を避け、間違えた呼び出し方でも reasonable な動作をする (API contract 緩和、cryptic error の予防)。stereo (`(B, 2, T)`) は引き続き `Conv1d in_channels=1` mismatch error に任せて明示的に失敗させる
- **副次効果**: T-M2.5 で `D(audio.unsqueeze(1))` 不要、`D(audio)` で OK。M2.6 smoke で `unsqueeze` 忘れ bug の余地を消す
- **テスト**: `test_msd_accepts_2d_input` で `(B, T)` 入力でも `(B, 1, T)` と同じ出力になることを確認

#### M2 phase で検討する追加設計原則

- **`MultiScaleDiscriminator` と `NLayerDiscriminator` を **module 1 個** に閉じる**:
  - 採用根拠: `NLayerDiscriminator` は MSD 専用の internal helper で、外部から直接生成する必要なし。export しないことで API surface を最小化
  - `__init__.py` の `__all__` には `MultiScaleDiscriminator` のみ

- **`forward` 戻り値の dataclass / NamedTuple 化 (M2 phase review で採用昇格)**:
  - **採用**: `DiscriminatorOutput(logits: Tensor, features: list[Tensor])` の `typing.NamedTuple` を導入
  - **採用根拠**: T-M2.3 §6.1 critical の `num_features=18 vs 21` 問題 (FM loss で features に logits を含めるか否かが曖昧、3 sub-D × {6 or 7} layer = {18 or 21} の取り違え) の **根本原因が戻り値仕様の不明瞭さ**。型レベルで `logits` と `features` を分離し、`features` には logits を **含めない** 方針を明示
  - **後方互換**: NamedTuple なので `(logits, features)` の tuple unpacking がそのまま動く (既存 docstring のサンプルコード無修正)
  - **保留していた根拠の見直し** : 当初は「軽量化を優先」だったが、`num_features` の取り違え bug リスクが軽量化メリットを上回ると M2 phase review で判明

- **`MultiScaleDiscriminator(num_D=3)` の default を 3 に固定**:
  - 採用根拠: WaveFit-PT 準拠を default で守る。`num_D=5` 等を渡すと M6.3 ablation 用途として動作するが、default では論文準拠
  - `MultiScaleDiscriminator.NUM_D = 3` クラス属性で「論文準拠の正本」を公開

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `MultiScaleDiscriminator` (MSD×3) + `NLayerDiscriminator` 実装、`tests/test_discriminator.py` 12 件 pass。channel 進行 [16,64,256,1024,1024,1024] 確認、weight_norm は `torch.nn.utils.parametrizations.weight_norm`。
- 出力を **`SubDiscOutput(logits, features)` NamedTuple の list** に (M2 review 採用)。`(B,T)→(B,1,T)` defensive unsqueeze。

実装上の判断:
1. **hinge GAN 互換性テストは「logits>1」でなく「Tanh/Sigmoid 不在」で構造検証**: init 時の logits は weight_norm で小さく `|logits|>1` が保証されない (脆い)。`not any(isinstance(m,(nn.Tanh,nn.Sigmoid)))` で「bounded activation 無し」を直接確認する方が頑健。教訓: **「出力が unbounded」は値でなく構造 (activation の不在) で検証する**。
2. **weight_norm テストは API 非依存に**: `torch.nn.utils.parametrize.is_parametrized(conv, "weight")` で parametrization 有無を確認 (新旧 weight_norm API どちらでも通る)。
3. grouped conv の `groups=nf_prev//4` は全層で in/out が groups で割り切れることを確認 (16→64 g4, 64→256 g16, ...)。

次の似たタスクで応用できる教訓:
- 未訓練モデルの「unbounded」性質は値テストでなく構造テスト (禁止 module の不在) で書く。
- NamedTuple 出力は下流 (T-M2.3 loss) が `.logits`/`.features` で読めて tuple index より可読。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.3 (Loss 関数) へ

**最重要**: 本チケットの `MultiScaleDiscriminator.forward` の戻り値仕様を **T-M2.3 で hinge GAN loss と FM loss の入力として直接使う**。

##### `DiscriminatorOutput` NamedTuple 仕様 (M2 phase review で確定)

T-M2.3 §6.1 critical の `num_features=18 vs 21` 取り違え問題を予防するため、本チケットで以下を確定:

- **`DiscriminatorOutput(logits: Tensor, features: list[Tensor])` を `typing.NamedTuple` として定義**
- **`features` に logits は含めない** (中間特徴 6 個のみ、各 sub-D)
- FM loss は **3 sub-D × 6 layer = 18 個** の features を L1 比較 (21 ではない)
- `MultiScaleDiscriminator.forward(x) -> list[DiscriminatorOutput]` (length=3、`NUM_D=3`)
- 後方互換: NamedTuple は tuple unpacking 互換、既存サンプルコード `for logits, features in D(x):` は無修正で動作

##### 使用方法
```python
from wavenext2.models.discriminator import MultiScaleDiscriminator
import torch.nn.functional as F


def hinge_gan_loss_D(D, y_real, y_fake):
    """Discriminator hinge GAN loss.

    Args:
        D:      MultiScaleDiscriminator
        y_real: (B, 1, T) GT 波形
        y_fake: (B, 1, T) generator 出力 (detach 済み)
    """
    out_real = D(y_real)                                  # list[(logits, features)]
    out_fake = D(y_fake.detach())
    loss = 0.0
    for (logits_real, _), (logits_fake, _) in zip(out_real, out_fake):
        loss = loss + F.relu(1.0 - logits_real).mean()    # `(1 - D(real))_+`
        loss = loss + F.relu(1.0 + logits_fake).mean()    # `(1 + D(fake))_+`
    return loss


def feature_matching_loss(D, y_real, y_fake):
    """L1 FM loss across all sub-D × intermediate features.

    Note: 3 sub-D × 6 layer = 18 個の中間特徴を L1 比較。
    """
    out_real = D(y_real)
    out_fake = D(y_fake)
    loss = 0.0
    n = 0
    for (_, feats_real), (_, feats_fake) in zip(out_real, out_fake):
        for f_r, f_f in zip(feats_real, feats_fake):
            loss = loss + F.l1_loss(f_f, f_r.detach())
            n += 1
    return loss / max(n, 1)
```

##### 重要事項
- **`forward` 戻り値**: `list[tuple[logits, features]]`、`len() == 3` (sub-D 数)
  - `logits.shape == (B, 1, T_k)`、`T_k` は k 増加とともに半分ずつ縮小
  - `features.shape == [(B, C_i, T_i) × 6]`、各層の LeakyReLU 後の出力
- **hinge GAN loss**: `relu(1 - logits_real).mean() + relu(1 + logits_fake).mean()` (D-loss、3 sub-D の和)
- **Generator 側 hinge loss**: `relu(1 - logits_fake).mean()` だが、論文 (`docs/training.md` §2.3) では `-logits_fake.mean()` の形 (non-saturating) も使われる。T-M2.3 で確定
- **FM loss**: 全 3 sub-D × 6 layer = **18 個** の特徴を L1 平均
- **`y_real.detach()` / `y_fake.detach()`**: D 側更新時に Generator に勾配を流さないため、呼び出し側 (T-M2.5) で `detach()` する責務

#### T-M2.4 (GAN モデル) へ
- T-M2.4 は本クラスを使わない (Generator 側のチケット)。本クラスとの接続は T-M2.5 (Training script) で行う
- ただし `GANWaveNext2.forward(...)` の出力 shape (`(B, T_audio)`) を、T-M2.5 で `unsqueeze(1)` して `(B, 1, T_audio)` に変換してから D に渡す手順を T-M2.5 で明確化する責務がある

#### T-M2.5 (Training script) へ
- **インスタンス化**:
  ```python
  from wavenext2.models.discriminator import MultiScaleDiscriminator

  D = MultiScaleDiscriminator(num_D=3, ndf=16, n_layers=4,
                              downsampling_factor=4, max_channels=1024,
                              leaky_relu_slope=0.2)
  # または factory 経由 (推奨)
  D = MultiScaleDiscriminator.from_config(cfg["discriminator"])
  ```
- **optimizer**:
  ```python
  opt_D = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-3)
  ```
- **shape 注意 (M2 phase review で defensive 化)**:
  - **`unsqueeze(1)` は D 側で defensive に受け入れる**: Generator 出力 `y_0.shape == (B, T_audio)` のままで `D(y_0)` 呼び出し可能 (D 側で `if x.dim() == 2: x = x.unsqueeze(1)` を実行)
  - `(B, 1, T)` で渡しても同じ結果 (NumPy / PyTorch 慣例の両方を許容)
  - 呼び出し側 (T-M2.5) で明示的に `unsqueeze(1)` は **不要**、忘れても D 側で吸収するため cryptic な `Conv1d` shape error を予防
  - stereo (`(B, 2, T)`) は引き続き `Conv1d in_channels=1` mismatch error で明示的に失敗
  - segment_length = 16,384 (Vocos 流) で T=16384 / 2^3 = 2048 が最小 (3 sub-D 後の最終 T_k)
- **D.train() / D.eval()**: 切替の副作用なし (BatchNorm / Dropout 不含)、`opt_D.step()` 前後で気にしなくて良い

#### T-M2.6 (smoke) へ
- 初期 logits が極端 (`abs().mean() > 100`) なら追加 init を検討
- `loss_D` が NaN / inf なら weight_norm の初期化 or hinge loss の数式を再確認

#### 設定値 (configs/*.yaml に追加予定)
```yaml
discriminator:
  num_D: 3
  ndf: 16
  n_layers: 4
  downsampling_factor: 4
  max_channels: 1024
  leaky_relu_slope: 0.2
```

#### 注意事項 (後続が踏みそうな罠)
1. **入力 shape は `(B, 1, T)` または `(B, T)`** (M2 phase review で defensive 化): `(B, T)` でも D 側で `unsqueeze(1)` を自動付与、stereo `(B, 2, T)` のみ `in_channels=1` mismatch エラー
2. **中間特徴 list の長さは 6** (7 layers - 1 logits 層)、FM loss は 3 sub-D × 6 layer = **18 個** の特徴を比較。`DiscriminatorOutput` NamedTuple の `features` には **logits を含めない** (`num_features=21` は誤り)
3. **最終 logits は activation なし**: 上位で `sigmoid` / `tanh` を**適用しない** (hinge GAN は raw logits を期待)
4. **`weight_norm` 新 API**: state_dict key 形式が旧 API と異なる。WaveFit-PT 重みを warm-start する場合は別途マッピング層が必要。テストは API 非依存に `torch.nn.utils.parametrize.is_parametrized(conv, "weight")` で判定
5. **`y_fake.detach()` 忘れ**: D 側更新時に detach しないと Generator にも勾配が流れる
6. **`AvgPool1d padding`**: padding=1 で T が半減ずつになる。padding=0 だと 1 ずれて FM loss の shape mismatch リスク
7. **`B=1` 動作**: テスト済 (`test_batch_size_one`)、`groups` conv も問題なし
8. **`DiscriminatorOutput` NamedTuple**: tuple unpacking 互換だが、属性アクセス (`output.logits`, `output.features`) も可能。可読性を優先する場合は属性アクセス推奨

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.2 の Acceptance チェックボックス 3 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M2.2 ステータスを `📝 pending` → `✅ completed`
  - [ ] `docs/tickets/index.md` の M2 進捗サマリ (pending -1, completed +1) を更新
  - [ ] (該当時) `docs/architecture.md` §6 に「実装上の補足」追記 (各 sub-D の channel 進行表、AvgPool1d padding 確定値などレビューで判明した知見があれば)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` で全 100% 確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - MPD ×5 追加判断 (M6.3 ablation)
  - MS-STFT discriminator 追加判断 (M6.3 ablation)
  - `weight_norm` 旧 API 互換性 (M5 warm-start 時)
  - 中間特徴の dataclass 化 (M5 phase review)
- **`docs/open-questions.md` への追記要否**: 不要 (本チケットの設計は既存の §C5 / §C6 で完全に確定)
