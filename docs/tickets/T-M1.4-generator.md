---
id: T-M1.4
title: WaveNeXt-based Generator (Conv → LN → ConvNeXt×8 → Linear×2 → clip)
milestone: M1
phase: M1
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M1.1]
blocks: [T-M1.6]
related_docs:
  - docs/milestones.md#m14-generator-srcwavenext2modelsgeneratorpy
  - docs/architecture.md
  - docs/open-questions.md
---

# T-M1.4: WaveNeXt-based Generator (Conv → LN → ConvNeXt×8 → Linear×2 → clip)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.4](../milestones.md#m14-generator-srcwavenext2modelsgeneratorpy)
> **依存**: [T-M1.1](T-M1.1-convnext-block.md) (前提: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md)) / **後続**: [T-M1.6](T-M1.6-sub-model.md)

## 1. タスク目的とゴール

### 目的
WaveNeXt-based generator を `src/wavenext2/models/generator.py` に実装する。これは sub-model の中核であり、`(mel-spec, STFT-spec)` を結合した tensor を入力に取り、`Conv1d → LayerNorm → ConvNeXt × 8 → LayerNorm → Linear(dim, n_fft+2) → Linear(n_fft+2, hop_length, bias=False) → reshape → clip(-1, 1)` の構造でノイズ成分 `n_t` (波形と同じ shape) を返す。GAN モードでは conditioning なし、Diff モードでは noise level embedding (512 次元) を ConvNeXt block 内に additive bias として注入する単一のクラスを提供し、両モードで Table 1 のパラメータ数 (GAN ≈ 14.99M / Diff ≈ 14.42M) と整合させる。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/models/generator.py` に `WaveNextGenerator` クラスが実装され、`from wavenext2.models.generator import WaveNextGenerator` で import 可能
- [ ] GAN 設定 (`input_channels=2176`, `n_fft=2048`, `hop_length=300`, `conditioning_dim=None`): 入力 `(B, 2176, 80)` → 出力 `(B, 24000)` / dtype=float32
- [ ] Diff 設定 (`input_channels=1152`, `n_fft=1024`, `hop_length=256`, `conditioning_dim=512`): 入力 `(B, 1152, 94)` + `cond=(B, 512)` → 出力 `(B, 24064)`
- [ ] 出力範囲が `[-1.0, 1.0]` に確実に収まる (`torch.clip(-1, 1)` が機能)
- [ ] パラメータ数: GAN ≈ 14.99M, Diff ≈ 14.42M (Table 1 と整合、許容 ±5%)
- [ ] 重み初期化: 全 `Conv1d` / `Linear` の `weight` が `trunc_normal_(std=0.02)`、`bias` が全 zero
- [ ] `tests/test_generator.py` の Unit テストが全 pass (shape / param count / 出力範囲 / 重み初期化 / gradient flow / determinism)
- [ ] 参考実装 (Vocos / WaveNeXt 非公式 / wetdog) を **コピーしていない** (CLAUDE.md ポリシー準拠)
- [ ] `docs/milestones.md` §M1.4 Acceptance 4 項目チェック済み

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/models/generator.py` (本実装)
  - `tests/test_generator.py` (Unit テスト、T-M0.2 で配置済みの placeholder を置換)
- 編集:
  - `src/wavenext2/models/__init__.py` (`__all__` に `"WaveNextGenerator"` を追加、必要なら re-export)

### 2.2 主要構造

```python
"""WaveNeXt-based generator (Fig. 2a).

Conv1d → LayerNorm → ConvNeXt × 8 → LayerNorm → Linear(dim, n_fft+2)
                                → Linear(n_fft+2, hop_length, bias=False)
                                → reshape (B, T, hop) → (B, T*hop)
                                → torch.clip(-1, 1)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from wavenext2.models.convnext import ConvNeXtBlock


class WaveNextGenerator(nn.Module):
    """WaveNeXt-based generator for GAN / Diff modes.

    Args:
        input_channels: Conv1d 入力 ch。GAN=128+2048=2176, Diff=128+1024=1152.
        dim: ConvNeXt embed dim (512, fixed).
        intermediate_dim: ConvNeXt MLP hidden (1536, fixed).
        n_blocks: ConvNeXt block 数 (8, fixed per paper Fig. 2).
        kernel_size: depthwise conv kernel (7, fixed).
        n_fft: STFT size。GAN=2048, Diff=1024.
        hop_length: hop。GAN=300, Diff=256.
        conditioning_dim: None なら GAN モード、512 なら Diff モード。
        layer_scale_init: ConvNeXt block の LayerScale 初期値 (1e-6).
    """

    def __init__(
        self,
        input_channels: int,
        n_fft: int,
        hop_length: int,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
        conditioning_dim: int | None = None,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conditioning_dim = conditioning_dim

        # 1. Input embedding: Conv1d(in=C_in, out=dim, k=7, p=3)
        self.embed = nn.Conv1d(
            input_channels, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True,
        )
        # 2. Input LayerNorm (channels_last, applied after transpose)
        self.norm_in = nn.LayerNorm(dim, eps=1e-6)

        # 3. ConvNeXt × n_blocks (T-M1.1 から呼び出し)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                kernel_size=kernel_size,
                layer_scale_init=layer_scale_init,
                conditioning_dim=conditioning_dim,
            )
            for _ in range(n_blocks)
        ])

        # 4. Final LayerNorm (channels_last)
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)

        # 5. Linear head (Vocos `ISTFTHead` 互換、warm-start 用)
        self.linear_1 = nn.Linear(dim, n_fft + 2, bias=True)
        # 6. Linear → hop_length (bias=False、元 WaveNeXt 実装で明示)
        self.linear_2 = nn.Linear(n_fft + 2, hop_length, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """trunc_normal_(std=0.02), bias=0. ConvNeXt block 内部は block 側で初期化済みなのでスキップ。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # nn.LayerNorm はデフォルト初期化 (weight=1, bias=0) のまま

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            x: (B, input_channels, T_mel)
            cond: (B, conditioning_dim) or None
        Returns:
            (B, T_mel * hop_length) waveform in [-1, 1].
        """
        if (self.conditioning_dim is None) != (cond is None):
            raise ValueError(
                "cond must be provided iff conditioning_dim is not None. "
                f"Got conditioning_dim={self.conditioning_dim}, cond is None={cond is None}"
            )

        B, _, T_mel = x.shape
        h = self.embed(x)                       # (B, dim, T_mel)
        h = h.transpose(1, 2)                   # (B, T_mel, dim) channels_last
        h = self.norm_in(h)
        h = h.transpose(1, 2)                   # (B, dim, T_mel)

        for block in self.blocks:
            # T-M1.1 の ConvNeXtBlock は cond を受け取って additive bias を加算する
            h = block(h, cond=cond) if self.conditioning_dim is not None else block(h)

        h = h.transpose(1, 2)                   # (B, T_mel, dim)
        h = self.norm_out(h)
        h = self.linear_1(h)                    # (B, T_mel, n_fft+2)
        h = self.linear_2(h)                    # (B, T_mel, hop_length)
        h = h.reshape(B, T_mel * self.hop_length)
        h = torch.clip(h, min=-1.0, max=1.0)
        return h
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 (GAN / Diff) | 出典 |
|---|---|---|
| `input_channels` | 2176 (=128+2048) / 1152 (=128+1024) | docs/architecture.md §2 表 |
| `dim` | 512 | docs/architecture.md §2.1, Vocos `vocos/modules.py` |
| `intermediate_dim` | 1536 | docs/architecture.md §2.1 ConvNeXt block |
| `n_blocks` | 8 (全モデル共通) | docs/architecture.md §2 (PDF Fig 2 キャプション) |
| `kernel_size` | 7 (Conv1d / ConvNeXt depthwise 共通) | docs/architecture.md §2 |
| `padding` | 3 (= `kernel_size // 2`) | 同上 |
| `n_fft` | 2048 / 1024 | docs/architecture.md §2 表 |
| `hop_length` | 300 / 256 | docs/architecture.md §2 表 |
| `linear_1 out` | `n_fft + 2` = 2050 / 1026 | docs/open-questions.md §C2 (Vocos ISTFTHead 互換) |
| `linear_2 out` | `hop_length` = 300 / 256, **bias=False** | docs/open-questions.md §C2 (wetdog 実装) |
| `conditioning_dim` | None / 512 | docs/architecture.md §5.4 (FastDiff additive bias) |
| LayerNorm eps | 1e-6 | docs/architecture.md §2.1 |
| `layer_scale_init` | 1e-6 | docs/architecture.md §2.1 (Vocos 既定) |
| 重み初期化 | `trunc_normal_(std=0.02)`, bias=zero | docs/open-questions.md §C2 |
| 最終 activation | `torch.clip(-1, 1)` (NOT tanh) | docs/open-questions.md §C2 / wetdog 実装 |

### 2.4 アルゴリズム / 処理フロー

1. **`__init__`**:
   1. `nn.Conv1d(input_channels, 512, kernel_size=7, padding=3, bias=True)` を作る (input embedding)
   2. `nn.LayerNorm(512, eps=1e-6)` (channels_last)
   3. `nn.ModuleList([ConvNeXtBlock(...) for _ in range(8)])` (T-M1.1 から import)
   4. `nn.LayerNorm(512, eps=1e-6)` (final, head 前)
   5. `nn.Linear(512, n_fft+2, bias=True)` (linear_1, Vocos ISTFTHead 互換)
   6. `nn.Linear(n_fft+2, hop_length, bias=False)` (linear_2)
   7. `_init_weights()`: 全 `Conv1d` / `Linear` の weight を `trunc_normal_(std=0.02)`, bias=0 で初期化
2. **`forward(x, cond=None)`**:
   1. `conditioning_dim` と `cond` の整合性 assert (片方だけ None は ValueError)
   2. `embed(x)`: `(B, C_in, T_mel) → (B, 512, T_mel)`
   3. `transpose(1, 2)`: channels_last 化 → `norm_in` → `transpose` で channels_first に戻す
   4. ConvNeXt block × 8 を `cond` 付きで適用 (Diff モード) または cond 無しで適用 (GAN モード)
   5. 再度 `transpose(1, 2)` → `norm_out`
   6. `linear_1`: `(B, T_mel, 512) → (B, T_mel, n_fft+2)`
   7. `linear_2`: `(B, T_mel, n_fft+2) → (B, T_mel, hop_length)`
   8. `reshape`: `(B, T_mel, hop_length) → (B, T_mel * hop_length)`
   9. `torch.clip(-1, 1)` (tanh ではない、`docs/open-questions.md` 確定)
3. **インターフェース契約 (T-M1.6 への申し送り)**:
   - 入力 `x` は **既に mel + STFT-spec が concat 済み** であること (本クラスは結合しない、T-M1.6 が結合)
   - 出力は **ノイズ成分 `n_t`** (波形ではなく、`y_t - n_t = y_{t-1}` の `n_t`)
   - `cond` は **既に noise embedding を通過した (B, 512) tensor** であること (本クラスは sinusoidal を計算しない、T-M1.5 を T-M1.6 が呼ぶ)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `generator.py` + `tests/test_generator.py` 実装、Acceptance まで自己検証 | general-purpose |
| Reviewer | 1 | コードレビュー (論文整合 / `docs/architecture.md` 整合 / コピペ無し)、param 数の計算検証 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_generator.py -v` 実行、Acceptance criteria 5.3 を網羅検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **partial yes** — T-M1.1 (ConvNeXt block) が `completed` になっていれば T-M1.4 を T-M1.2 / T-M1.3 / T-M1.5 と並列着手可能 (互いに `models/` 配下の別ファイルを編集するだけで干渉しない)
- 並列実行する場合の最大並列数: 4 (T-M1.2, T-M1.3, T-M1.4, T-M1.5 を別 worker で並列。T-M1.6 はこれら全完了後)
- 注意: ConvNeXt block の I/O 仕様 (`forward(x, cond=None) -> Tensor`) を T-M1.1 完了時点で固定すること。本チケットの forward はその契約に依存する

## 4. 提供範囲 (Scope)

### In Scope
- `WaveNextGenerator` クラスの本実装 (single class、GAN/Diff 両対応)
- `__init__` での全 sub-module 構築と `trunc_normal_(std=0.02)` / bias=zero 初期化
- `forward(x, cond=None)` の channels_first ↔ channels_last 変換含む処理フロー
- 出力 `torch.clip(-1, 1)` の適用
- Unit テスト 6 項目 (§5.1):
  1. GAN モード shape
  2. Diff モード shape
  3. 出力範囲 [-1, 1]
  4. パラメータ数 (GAN / Diff)
  5. 重み初期化 (`trunc_normal_` std≈0.02, bias=0)
  6. gradient flow (`loss.backward()` で全 param に grad)
- `cond` と `conditioning_dim` の整合性 assert (ValueError)
- `__init__.py` への re-export (任意、`__all__` に追加)

### Out of Scope
- ConvNeXt block の内部実装 → **T-M1.1** で完了済み (本チケットは呼び出すのみ)
- STFT module (波形 → STFT-spec の変換) → **T-M1.2** で実装
- mel-spectrogram 抽出 → **T-M1.3** で実装
- Noise embedding (`c → (B, 512)`) → **T-M1.5** で実装
- mel + STFT-spec の concat → **T-M1.6** (sub-model wrapper) で実装
- Discriminator → **T-M2.2**
- ノイズ schedule、fixed-point iteration、reverse sampling → **T-M2.4 / T-M3.1 / T-M3.3**
- Generator EMA → 不採用 (docs/open-questions.md §C8 確定)
- ONNX / TorchScript export → M6 以降の最適化チケットで検討
- `torch.compile` 適用 → 本チケットでは触らない (cp313 + torch 2.10 の安定性に懸念、M3 smoke で評価)

### Deliverable
- ファイル:
  - `src/wavenext2/models/generator.py` (新規、約 120〜150 行)
  - `tests/test_generator.py` (新規、約 100〜150 行)
- 関数 / クラス:
  - `WaveNextGenerator(nn.Module)`: `__init__`, `_init_weights`, `forward`
- ドキュメント差分:
  - `docs/milestones.md` §M1.4 Acceptance チェックボックスを ☑ に更新
  - `docs/tickets/index.md` の T-M1.4 ステータスを `📝 pending` → `✅ completed`
  - (必要なら) `docs/architecture.md` §2 に「実装上の補足」追記 (channels-first ↔ channels-last の transpose 位置等、レビュー時に判明した場合)

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_generator.py`)

- [ ] `test_gan_shape`: GAN 設定で入力 `(2, 2176, 80)` → 出力 `(2, 24000)` (dtype=float32, device=cpu)
- [ ] `test_diff_shape`: Diff 設定で入力 `(2, 1152, 94)` + `cond=(2, 512)` → 出力 `(2, 24064)`
- [ ] `test_output_range`: GAN/Diff 両方で生成出力が `[-1.0, 1.0]` に収まる (`torch.all(out >= -1.0) and torch.all(out <= 1.0)`)
- [ ] `test_param_count_gan`: GAN 設定の `sum(p.numel() for p in model.parameters())` が `14.99M ± 5%` (約 14.24M〜15.74M) の範囲
- [ ] `test_param_count_diff`: Diff 設定の `sum(p.numel() for p in model.parameters())` が `14.42M ± 5%` (約 13.70M〜15.14M) の範囲
- [ ] `test_weight_init_conv1d`: `model.embed.weight.std()` が `0.02 ± 0.01` に近い (trunc_normal を実測したらおよそ 0.02 になる)
- [ ] `test_weight_init_linear_bias_zero`: `model.linear_1.bias` が全 0、`model.linear_2.bias is None` (= no bias)
- [ ] `test_gradient_flow`: GAN/Diff モードで `loss = model(...).abs().mean(); loss.backward()` 実行後、全 `model.parameters()` の `p.grad` が None でなく nan/inf を含まない
- [ ] `test_deterministic`: `torch.manual_seed(0)` 固定下で同入力に対する出力が 2 回呼び出しで一致 (eval モード必須、dropout なし)
- [ ] `test_cond_mismatch_raises`: GAN モード (`conditioning_dim=None`) で `cond` を渡すと `ValueError`、Diff モード (`conditioning_dim=512`) で `cond=None` 呼び出しも `ValueError`

```python
# tests/test_generator.py (テスト骨格)
import pytest
import torch
from wavenext2.models.generator import WaveNextGenerator


@pytest.fixture
def gan_model():
    return WaveNextGenerator(
        input_channels=2176, n_fft=2048, hop_length=300,
        dim=512, intermediate_dim=1536, n_blocks=8,
        conditioning_dim=None,
    )


@pytest.fixture
def diff_model():
    return WaveNextGenerator(
        input_channels=1152, n_fft=1024, hop_length=256,
        dim=512, intermediate_dim=1536, n_blocks=8,
        conditioning_dim=512,
    )


def test_gan_shape(gan_model):
    x = torch.randn(2, 2176, 80)
    out = gan_model(x)
    assert out.shape == (2, 24000)


def test_diff_shape(diff_model):
    x = torch.randn(2, 1152, 94)
    cond = torch.randn(2, 512)
    out = diff_model(x, cond=cond)
    assert out.shape == (2, 24064)


def test_output_range(gan_model):
    x = torch.randn(2, 2176, 80) * 100  # 大きな入力で clip を確実に発火
    out = gan_model(x)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_param_count_gan(gan_model):
    n_params = sum(p.numel() for p in gan_model.parameters())
    assert 14.24e6 <= n_params <= 15.74e6, f"GAN params = {n_params/1e6:.2f}M, expected ~14.99M"


def test_param_count_diff(diff_model):
    n_params = sum(p.numel() for p in diff_model.parameters())
    assert 13.70e6 <= n_params <= 15.14e6, f"Diff params = {n_params/1e6:.2f}M, expected ~14.42M"


def test_weight_init_linear_bias_zero(gan_model):
    assert torch.allclose(gan_model.linear_1.bias, torch.zeros_like(gan_model.linear_1.bias))
    assert gan_model.linear_2.bias is None


def test_gradient_flow(gan_model):
    x = torch.randn(2, 2176, 80, requires_grad=False)
    out = gan_model(x)
    loss = out.abs().mean()
    loss.backward()
    for name, p in gan_model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has nan/inf"


def test_cond_mismatch_raises(gan_model, diff_model):
    x_gan = torch.randn(2, 2176, 80)
    x_diff = torch.randn(2, 1152, 94)
    with pytest.raises(ValueError):
        gan_model(x_gan, cond=torch.randn(2, 512))
    with pytest.raises(ValueError):
        diff_model(x_diff, cond=None)
```

### 5.2 e2e / 結合テスト
- [ ] T-M1.6 で `WaveNextGenerator` を sub-model wrapper から呼び出す統合テスト (本チケットの範囲外、T-M1.6 で実施)
- [ ] 実音声 1 utterance (LibriTTS-R の任意の wav) を mel 抽出 + ダミー STFT-spec (zeros) と結合して forward + backward が成功することは T-M1.6 で実施

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.4 より転記)
- [ ] GAN 設定 (`input_channels=2176`, `hop=300`, `n_fft=2048`): 入力 `(B, 2176, 80)` → 出力 `(B, 24000)`
- [ ] Diff 設定 (`input_channels=1152`, `hop=256`, `n_fft=1024`): 入力 `(B, 1152, 94)` → 出力 `(B, 24064)`
- [ ] 出力範囲が `[-1, 1]` に収まる (clip が機能)
- [ ] パラメータ数: GAN 版 ≈ 14.99M, Diff 版 ≈ 14.42M (Table 1 と整合)
- [ ] 重み初期化: `Conv1d.weight.std() ≈ 0.02`, `Linear.bias` がゼロ

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **パラメータ数が論文 Table 1 と微妙に異なる** | Acceptance §5.3 で fail。LayerNorm の (weight, bias) 2 要素を含めるか、ConvNeXt block の LayerScale `gamma` を含めるかで差が出る | 内訳を Bash で表示する補助テスト (`test_param_count_breakdown`) を追加。差が ±5% を超えた場合は (a) ConvNeXt block 側でカウントしている param がここでも重複していないか、(b) `bias=True/False` の指定漏れがないか、を確認。最終的には Table 1 の 14.99M / 14.42M を **目安値** として扱い、許容範囲を ±5% にする |
| **`Conv1d` の channels-first ↔ ConvNeXt の channels-last 変換** | forward で間違った transpose を入れると shape 不一致や微妙な channel ミックスが発生 | `forward` 内で `transpose(1, 2)` を入れる箇所と回数を明示的にコメント。`test_gan_shape` / `test_diff_shape` が pass すれば検出される |
| **`linear_1` の Linear が channels-last 入力前提** | `(B, T_mel, 512)` ではなく `(B, 512, T_mel)` のまま `nn.Linear` を通すと最終 dim が C と勘違いされ shape error | `forward` 内で `norm_out` の **後** に必ず `transpose(1, 2)` 済みの状態を維持。コメントで明示。`shape` テストで担保 |
| **`linear_1` (out=n_fft+2=2050) のメモリ消費** | GAN モードで `(B, T_mel, 2050)` の中間 tensor は大きい (B=16, T_mel=80 で約 10MB / sample / fp32)。本チケットでは問題なくても M2.4 (T=4 直列で `T × 16 × 80 × 2050 = 105MB`) で OOM 候補 | 本チケット時点では問題なし。M2 で `gradient checkpointing` を `WaveNextGenerator` 内 ConvNeXt block 集合に適用するオプションを `enable_grad_ckpt=False` パラメータで予約。docs/milestones.md §リスク表でも明記 |
| **`trunc_normal_` の挙動 (truncation 範囲)** | デフォルトは `a=-2, b=2` (std=0.02 だと実質 ±0.04)。Vocos と完全一致するかどうか | PyTorch `nn.init.trunc_normal_` のデフォルト `a=-2, b=2` を採用、std=0.02 で初期化。`test_weight_init_conv1d` は `std() == 0.02 ± 0.01` の許容で確認 (truncation の影響で実測 std がやや小さくなる) |
| **ConvNeXt block の `cond` 引数仕様 (T-M1.1) との不整合** | T-M1.1 の `ConvNeXtBlock.forward(x, cond=None)` の引数名と本チケットの呼び出しが食い違うと runtime error | T-M1.1 完了時点で `forward(x, cond=None) -> Tensor` の signature を **固定** することを §3 並列度の注意に明記。本チケットでは `block(h, cond=cond)` の形で呼ぶ |
| **`hop_length` が mel の time 軸と未整合** | `(B, T_mel) * hop_length` の reshape で破綻 | 本クラスは入力 `T_mel` を信頼するのみ。T_mel = audio_len / hop_length の整合性は **上位 (T-M1.6 / T-M2.1)** で担保。本チケットは reshape の正しさのみ shape テストで確認 |
| **batch_size=1 でのみ動作確認漏れ** | B=1 の broadcast 挙動が異なる可能性 | テストで `B=2` を使い、追加で `B=1` のエッジケースを `test_batch_size_one` として追加 |
| **fp16 / bf16 精度での clip の挙動** | M6 で AMP 訓練する際に clip が float16 で動作するか確認が必要 | 本チケットでは fp32 のみテスト。fp16 検証は M6 (本格訓練) チケットで実施 |
| **`__init__.py` の `__all__` 更新漏れ** | `from wavenext2.models import WaveNextGenerator` が動かない | テストで `from wavenext2.models.generator import WaveNextGenerator` を使う (フルパス指定)。`__all__` への追加は任意 (T-M0.2 の方針に従う) |

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` §C2 で Generator output head と重み初期化は **全て確定済み** (`trunc_normal_(std=0.02)`, bias=zero, `clip(-1, 1)`, `linear_2.bias=False`)。本チケットで追加の決定は不要。
- `linear_1` の bias は **True** (元 WaveNeXt poster + Vocos `ISTFTHead` で確定)
- LayerNorm の `eps=1e-6` (Vocos `vocos/modules.py`)。標準値 1e-5 ではない点に注意
- `padding_mode='zeros'` (Conv1d デフォルト) で OK。reflection padding は使わない (Vocos も zeros)

### 6.3 他チケットとの整合性
- **T-M1.1 (ConvNeXt block)**: `ConvNeXtBlock.__init__(dim, intermediate_dim, kernel_size, layer_scale_init, conditioning_dim)` と `forward(x, cond=None)` のシグネチャに依存。**T-M1.1 で本シグネチャを固定する必要あり** (M1 フェーズ内整合)
- **T-M1.2 (STFT module)**: 本クラスの `input_channels = 128 + (2F-2)` の `2F-2` 計算は T-M1.2 の出力 ch 数に依存。GAN: 2F-2=2046 で `128+2048=2176` の "2048" は実質 `2F-2=2046+2` (DC/Nyquist 復元) ではなく、`2F` (実部 F + 虚部 F-2 + ??) → docs/architecture.md §2 表が `2F-2` を採用するため、**正確には GAN: `128 + 2046 = 2174` になる可能性あり**。**docs/architecture.md §2 では `128 + 2048 = 2176`** と書かれているのでこちらを採用 (実部 F=1025 + 虚部 F-2=1023 = 2048 → これが `2F-2 = 2*1025 - 2 = 2048`、整合)。docs と一致するので問題なし
- **T-M1.5 (Noise embedding)**: `cond` の `(B, 512)` 仕様は本チケットで `conditioning_dim=512` 固定値として受け入れる。T-M1.5 の出力次元 512 と整合
- **T-M1.6 (Sub-model wrapper)**: 本クラスを `nn.Module` として `self.generator = WaveNextGenerator(...)` で保持し、`forward(x, cond=cond)` で呼ぶ前提。本チケットの forward signature を変えると T-M1.6 にも影響
- **T-M2.4 (GAN モデル)**: T sub-model を直列に並べる際、本クラスを T 個生成して `ModuleList` に入れる。本クラスが state を持たない (各 forward は純粋関数) ことを保証 → OK
- **T-M3.1 (Diff モデル)**: 4 sub-model を独立に保持。本クラスの `conditioning_dim=512` 経由で noise embedding を受け取る。OK

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §2 (Generator 構造) と実装が完全一致 (Conv1d → LN → ConvNeXt×8 → LN → Linear → Linear → reshape → clip の順序)
- [ ] `docs/open-questions.md` §C2 (output head, init, clip vs tanh) と整合
- [ ] `linear_1` は `bias=True`, `linear_2` は `bias=False` (元 WaveNeXt 実装で確定)
- [ ] LayerNorm の `eps=1e-6` (Vocos と一致、標準 1e-5 ではない)
- [ ] 全 Conv1d / Linear の重みが `trunc_normal_(std=0.02)`、bias が zero
- [ ] 最終 activation は `torch.clip(-1, 1)` (tanh ではない)
- [ ] Acceptance criteria §5.3 全 5 項目クリア (shape × 2, range, param count × 2, init)
- [ ] Unit テスト 10 項目全 pass (`uv run pytest tests/test_generator.py -v`)
- [ ] CLAUDE.md / 既存コードのスタイル準拠: 型ヒント (`from __future__ import annotations`)、docstring (各メソッド)、命名 (snake_case)
- [ ] エラー処理: `cond` と `conditioning_dim` の整合性で `ValueError` を raise
- [ ] パラメータ数 / メモリ消費が想定内 (GAN ≈ 14.99M, Diff ≈ 14.42M)
- [ ] 参考実装 (Vocos `vocos/models.py`, wetdog `wavenext_pytorch/heads.py`, FastDiff) を **コピーしていない** (構造は参考だが、変数名・コメント・実装順序が独自であること)
- [ ] `from wavenext2.models.convnext import ConvNeXtBlock` で T-M1.1 を再利用、独自に ConvNeXt を再実装していない
- [ ] T-M1.6 で sub-model wrapper から呼べる forward signature (`forward(x, cond=None)`) が維持されている
- [ ] §6.3 の `2F-2` 計算式が docs/architecture.md §2 表と一致 (`128 + 2048 = 2176` for GAN)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **出力 head を `ConvTranspose1d` で置換 (HiFi-GAN 風)** | アップサンプリングを Linear ではなく学習可能な転置畳み込みで行えば、より局所的な構造を学べる可能性 | パラメータ数が増える、論文との乖離、Vocos warm-start 不可 | 論文 Fig 2a + 元 WaveNeXt poster で **2 段 Linear** が確定。warm-start のため `linear_1` の `n_fft+2` 次元も Vocos `ISTFTHead` と一致させる設計上の意図あり | **M5 で品質が論文と大きく乖離した場合に ablation 対象として検討** |
| **`torch.clip(-1, 1)` を `tanh` に置換 (BigVGAN 風)** | tanh は微分可能で gradient flow が滑らか、BigVGAN で実績あり | docs/open-questions.md §C2 で `clip(-1, 1)` が確定 (wetdog 実装で明示)。論文と異なる | 論文準拠を優先。`clip` は端で gradient=0 になるが、出力が大半 `[-1, 1]` 内に収まれば訓練中はほぼ tanh と同等 | **M5 / M6 で訓練が divergent な場合 (clip の dead gradient 起因と特定された場合)** に tanh / soft_clip で ablation |
| **LayerNorm を `RMSNorm` に置換** | RMSNorm は計算が軽く、LLM 系で実績 | Vocos / WaveNeXt 系は LayerNorm 慣例、論文準拠から逸脱 | 論文準拠 + Vocos warm-start 互換性のため LayerNorm | **M6 で `nn.LayerNorm` の bottleneck が profiling で判明した場合のみ** |
| **ConvNeXt block 数を 6 / 10 / 12 に変更** | 6: 速度向上、12: 品質向上の可能性 | 論文 Fig 2 で `n=8` 固定が明示、Table 1 のパラメータ数とも一致 | 論文準拠。本クラスは `n_blocks` 引数を受け取るので将来 ablation 可能 | **M6.3 ablation で実施 (チケット既存)** |
| **`linear_1 (dim=512, n_fft+2=2050)` を中間 dim 経由で 2 段化** (例: `Linear(512, 1024) → GELU → Linear(1024, n_fft+2)`) | 表現力が増える可能性 | パラメータ数が増える、warm-start 不可、論文と異なる | 論文準拠 | **M5 で `linear_1` 周辺が表現力 bottleneck と判明した場合** |
| **`Conv1d` 入力 embedding を 2 段 (Conv1d×2 + LN) に拡張** | より柔軟な特徴抽出 | Vocos / WaveNeXt はいずれも 1 段、論文準拠から逸脱 | Vocos 慣例 + 論文準拠 | **M5/M6 で input ch 数 2176 の Conv1d 単段が bottleneck と profile で判明した場合** |
| **`conditioning_dim` を生クラス引数ではなく `forward` で動的に受け取る** | 同一インスタンスで GAN/Diff を切り替え可能 | 設計が複雑化、GAN/Diff で別インスタンスを作る方が明快 | 別インスタンス方針で十分 (M2.4 と M3.1 で別途生成) | **M6 で 1 model で GAN + Diff の hybrid 訓練を試す場合のみ** |
| **`enable_grad_ckpt` を `__init__` 引数に追加して gradient checkpointing 対応** | M2.4 で T=4 直列 OOM 回避 | 本チケットでは未検証、複雑性が増す | 本チケットでは Out of Scope。M2.4 で OOM が確認された時点でフラグ追加 | **M2.4 smoke で OOM 発生時、ConvNeXt block 集合に `torch.utils.checkpoint.checkpoint_sequential` を適用** |
| **`forward` 内 `transpose` 回数を最小化する設計** (channels_last を維持する) | 計算効率向上の可能性 | `nn.Conv1d` が channels_first 必須、`nn.LayerNorm` が channels_last (last_dim) を期待、`nn.Linear` も channels_last なので transpose は最低限必要 | 現実装の transpose 2 回 (`embed 後` と `last block 後`) が最小構成 | **M6 profiling で transpose が hot path に出た場合のみ** |
| **`torch.compile` を `__init__` で適用** | 速度向上 | cp313 + torch 2.10 の安定性に懸念、graph break リスク | 本チケットでは触らない。M3 smoke で評価 | **M3 smoke 後、graph break が起きなければ M6 で導入** |

### 8.2 思想 / 哲学の見直し
- **粒度**: size=M (約 120〜150 行の本体 + 150 行のテスト) は妥当。1 セッションで完了可能。
- **`ConvNeXtBlock` と分離した設計**: T-M1.1 で別ファイルにした判断は正解。Generator が ConvNeXt 内部実装を知らない契約を維持すれば、後続 ablation (RMSNorm / 別 block) も差し替え可能
- **`__init__.py` re-export**: `from wavenext2.models import WaveNextGenerator` 形式を許可する場合は `__init__.py` の `__all__` に追加。本チケットではフルパス (`from wavenext2.models.generator import WaveNextGenerator`) でテストするため、re-export は **任意** とする
- **`forward` の `cond` 引数のオプショナル性**: `Optional[Tensor]` で受け取り、`conditioning_dim` との整合性を `__init__` 時点ではなく `forward` 時点で検証する設計を採用 (lazy validation)。これにより同一クラスで GAN/Diff を区別できる
- **インターフェース定義**:
  - `forward(x, cond=None) -> Tensor` の signature を T-M1.6 と T-M2.4 / T-M3.1 で共有
  - 戻り値は **常に `(B, T_mel * hop_length)`** (波形と同 shape のノイズ成分)
  - `clip(-1, 1)` を `forward` 内に閉じる (上位で再 clip しない)
- **再評価トリガー条件** (§8.1 表末尾参照):
  - パラメータ数が論文と ±10% を超えて乖離した場合 → ConvNeXt block の内部実装 (T-M1.1) と LayerScale / LayerNorm のパラメータ計上を再確認
  - 出力が clip で頻繁に飽和して訓練が unstable な場合 → tanh / soft_clip への切替検討 (M5/M6)
  - メモリ消費が想定の 2 倍を超えた場合 → `linear_1` 周辺で `n_fft+2` の中間 tensor が問題、gradient checkpointing 導入

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M1.6 (Sub-model wrapper) へ
- **import 経路**: `from wavenext2.models.generator import WaveNextGenerator`
- **インスタンス化**:
  ```python
  # GAN モード
  generator = WaveNextGenerator(
      input_channels=128 + (2 * 1025 - 2),  # =2176
      n_fft=2048, hop_length=300,
      conditioning_dim=None,
  )
  # Diff モード
  generator = WaveNextGenerator(
      input_channels=128 + (2 * 513 - 2),   # =1152
      n_fft=1024, hop_length=256,
      conditioning_dim=512,
  )
  ```
- **forward 呼び出し**:
  - GAN: `out = generator(x)` (`x` は mel + STFT-spec を concat 済み)
  - Diff: `out = generator(x, cond=noise_emb)` (`noise_emb` は T-M1.5 の `(B, 512)`)
- **入力 `x` の前処理は T-M1.6 の責務**: T-M1.6 が STFT module (T-M1.2) と mel-spec を concat してから本クラスに渡す
- **出力は `[-1, 1]` 範囲のノイズ成分** (波形と同 shape)。GAN では `y_t - n_t = y_{t-1}`、Diff では `epsilon` 予測として使う
- **state を持たない**: `forward` は純粋関数 (parameters 以外の state なし)。T-M2.4 で `T` 個直列に並べても各インスタンスが独立

#### T-M2.4 (GAN モデル) へ
- T 個の `WaveNextGenerator` を `nn.ModuleList([WaveNextGenerator(...) for _ in range(T)])` で保持
- 各 generator は独立したパラメータを持つ (重みは共有しない)
- `conditioning_dim=None` で初期化、`forward` は `(x,)` のみ受け取る

#### T-M3.1 (Diff モデル) へ
- 4 個の `WaveNextGenerator` を `conditioning_dim=512` で初期化
- 各 sub-model は独立した `WaveNextGenerator` インスタンスを保持 (sub-model 間で重み共有しない)
- `cond` 引数経由で noise embedding (T-M1.5 経由) を渡す

#### 設定値 (configs/*.yaml に追加予定)
```yaml
generator:
  input_channels: 2176   # GAN: 128+2048, Diff: 1152 (=128+1024)
  dim: 512
  intermediate_dim: 1536
  n_blocks: 8
  kernel_size: 7
  n_fft: 2048            # GAN: 2048, Diff: 1024
  hop_length: 300        # GAN: 300, Diff: 256
  conditioning_dim: null # GAN: null, Diff: 512
  layer_scale_init: 1.0e-6
```

#### 注意事項 (後続が踏みそうな罠)
1. **`2F-2 = 2*1025-2 = 2048`** で `input_channels = 128 + 2048 = 2176` (GAN)。**`2F = 2050` ではない** ことに注意
2. **`linear_2.bias = False`** であり、`bias is None` で確認する (`torch.zeros` ではない)
3. **`torch.clip(-1, 1)` を上位で再適用しない** (forward 内で適用済み)
4. **`cond` は GAN モードでは絶対に渡さない** (`ValueError` raise)
5. **ConvNeXt block の `cond` 受け取り順序**: 本実装では `for block in self.blocks: h = block(h, cond=cond)`。block 入口で additive bias 加算するのは T-M1.1 の責務 (本クラスは bias 計算しない)
6. **fp16 訓練時の clip 挙動**: M6 で AMP を有効化する場合、clip が float16 で正常動作することを確認 (本チケットでは fp32 のみテスト)
7. **重みの再初期化**: `model.apply(init_weights_fn)` を上位で適用すると、`_init_weights` の効果が上書きされる可能性。上位はカスタム初期化を呼ばないこと

### 9.2 ドキュメント更新

- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M1.4 の Acceptance チェックボックス 4 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M1.4 ステータスを `📝 pending` → `✅ completed`
  - [ ] `docs/tickets/index.md` の M1 進捗サマリ (pending -1, completed +1) を更新
  - [ ] (該当時) `docs/architecture.md` §2 に「実装上の補足」を追記 (channels-first ↔ channels-last の transpose 位置、メモリ消費の実測値など、レビュー時に判明した知見があれば)
  - [ ] (該当時) `docs/open-questions.md` の §C2 が実装で追加確認できた事項があれば補記

### 9.3 Open question として残ったもの
- パラメータ数の **正確な内訳** (LayerScale / LayerNorm の (weight, bias) を含めて何個になるか) は実装してから測定。±5% 許容で実装し、論文記載 14.99M / 14.42M と乖離した場合のみ詳細 break-down を行う
- `n_fft+2` の中間 tensor `(B, T_mel, n_fft+2)` のメモリ消費が M2.4 (T=4 直列) で問題になるかは、smoke test まで未確認 → §6.1 リスクとして M2 で再評価
- fp16 / bf16 での `clip` と `trunc_normal_` 初期化の挙動は M6 で AMP 有効時に再評価。本チケットでは fp32 のみ
- `forward` 内 transpose 回数 (現状 2 回) が profiling 上のホットスポットかは M6 で再評価。channels_last の `LayerNorm` のみ呼ぶ別実装で性能向上の余地あり
