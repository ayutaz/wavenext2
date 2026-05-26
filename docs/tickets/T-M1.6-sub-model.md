---
id: T-M1.6
title: Sub-model wrapper (STFT module + Generator、GAN/Diff 両対応)
milestone: M1
phase: M1
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M1.2, T-M1.3, T-M1.4, T-M1.5]
blocks: [T-M2.4, T-M3.1]
related_docs:
  - docs/milestones.md#m16-sub-model-wrapper-srcwavenext2modelssub_modelpy
  - docs/architecture.md
  - docs/open-questions.md
  - docs/implementation-plan.md
---

# T-M1.6: Sub-model wrapper (STFT module + Generator、GAN/Diff 両対応)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.6](../milestones.md#m16-sub-model-wrapper-srcwavenext2modelssub_modelpy)
> **依存**: [T-M1.2](T-M1.2-stft-module.md), [T-M1.3](T-M1.3-mel-spectrogram.md), [T-M1.4](T-M1.4-generator.md), [T-M1.5](T-M1.5-noise-embedding.md) / **後続**: [T-M2.4](T-M2.4-gan-model.md), [T-M3.1](T-M3.1-diff-model.md)

## 1. タスク目的とゴール

### 目的
論文 Fig 2b の **sub-model** (= STFT module + WaveNeXt-based generator) を、GAN-WaveNeXt 2 / Diff-WaveNeXt 2 の **両方で同一構造の sub-model を直列に並べる/独立に並べる** ことを可能にする抽象レイヤとして 1 ファイル (`src/wavenext2/models/sub_model.py`) に閉じ込める。

これにより:
- T-M2.4 (GAN モデル) は `nn.ModuleList([SubModelGAN(...) for _ in range(T)])` でループするだけで fixed-point iteration が組める
- T-M3.1 (Diff モデル) は `nn.ModuleList([SubModelDiff(...) for _ in range(4)])` でループするだけで 4-step reverse sampling が組める

論文の核アイデア「**同じ sub-model 構造を GAN/Diff 両用にする**」を **コードの依存グラフ上でも明示** する。

### ゴール
- [ ] `src/wavenext2/models/sub_model.py` に `SubModelGAN` と `SubModelDiff` の 2 クラスが実装され、両者とも `nn.Module` を継承
- [ ] `SubModelGAN(mel, y_prev) -> (B, T_audio)` が動作 (conditioning なし)
- [ ] `SubModelDiff(mel, x_t, c) -> (B, T_audio)` が動作 (noise level conditioning あり)
- [ ] 内部で `STFTModule` (T-M1.2) と `WaveNextGenerator` (T-M1.4)、Diff 版では加えて `NoiseEmbedding` (T-M1.5) を **構成 (composition)** する
- [ ] mel と STFT-spec の channel concat の順序が `[mel (128 ch), stft_spec (2F-2 ch)]` で確定し docstring + 定数で明示
- [ ] GAN/Diff それぞれの設定で `SubModelGAN` ≈ 14.99M params, `SubModelDiff` ≈ 14.42M params が `tests/test_sub_model.py` で検証される
- [ ] forward + backward が動作し全 sub-module に grad が流れる
- [ ] `docs/milestones.md` §M1.6 Acceptance criteria 全 4 項目をクリア
- [ ] `tests/test_sub_model.py` の `pytest.skip` placeholder (T-M0.2 配置) を実テストに置き換え、`uv run pytest tests/test_sub_model.py` が全 pass

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規実装 (T-M0.2 で空 stub として配置済み):
  - `src/wavenext2/models/sub_model.py` (本実装)
  - `tests/test_sub_model.py` (本実装、`pytest.skip` placeholder を置換)
- 編集:
  - `src/wavenext2/models/__init__.py` (`__all__` に `SubModelGAN`, `SubModelDiff` を追加)
  - `docs/milestones.md` §M1.6 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` のステータス更新

### 2.2 主要構造

#### import 形式 (T-M0.2 で確定)
```python
from wavenext2.models.sub_model import SubModelGAN, SubModelDiff
```

#### クラスシグネチャ

```python
"""sub_model.py — STFT module + WaveNeXt-based generator のラッパ。

論文 Fig 2b に対応する sub-model 単位を、GAN/Diff の差異を吸収して定義する。
- SubModelGAN  : STFTModule + WaveNextGenerator (conditioning なし)
- SubModelDiff : STFTModule + WaveNextGenerator (conditioning_dim=512) + NoiseEmbedding

mel と STFT-spec の channel concat 順序は [mel (128 ch), stft_spec (2F-2 ch)] で固定する。
詳細は docs/architecture.md §1〜§4 を参照。
"""

from __future__ import annotations
import torch
import torch.nn as nn

from wavenext2.models.stft import STFTModule
from wavenext2.models.generator import WaveNextGenerator
from wavenext2.models.noise_embedding import NoiseEmbedding


# channel concat 順序の単一情報源 (SoT)。T-M2.4 / T-M3.1 がこの定数を import 可能。
CONCAT_ORDER: tuple[str, str] = ("mel", "stft_spec")


class SubModelGAN(nn.Module):
    """GAN-WaveNeXt 2 用 sub-model (conditioning なし)."""

    def __init__(
        self,
        mel_channels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 300,
        win_length: int = 1200,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.mel_channels = mel_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_spec_channels = 2 * (n_fft // 2 + 1) - 2  # = 2F - 2
        input_channels = mel_channels + stft_spec_channels
        self.generator = WaveNextGenerator(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            n_fft=n_fft,
            hop_length=hop_length,
            conditioning_dim=None,
        )

    def forward(self, mel: torch.Tensor, y_prev: torch.Tensor) -> torch.Tensor:
        """Args:
            mel:    (B, 128, T_mel) log-mel.
            y_prev: (B, T_audio = T_mel * hop_length) 前ステップ波形。
        Returns:
            (B, T_audio) ノイズ成分 n_{t-1} (波形そのものではない、§docs/architecture.md §2)。
        """
        T_mel = mel.shape[2]
        stft_spec = self.stft_module(y_prev, T_mel=T_mel)         # (B, 2F-2, T_mel)
        x = torch.cat([mel, stft_spec], dim=1)                    # (B, C_in, T_mel)
        return self.generator(x)                                  # (B, T_mel * hop)


class SubModelDiff(nn.Module):
    """Diff-WaveNeXt 2 用 sub-model (noise level conditioning あり)."""

    def __init__(
        self,
        mel_channels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        dim: int = 512,
        intermediate_dim: int = 1536,
        n_blocks: int = 8,
        kernel_size: int = 7,
        sinusoidal_dim: int = 128,
        cond_dim: int = 512,
    ) -> None:
        super().__init__()
        self.mel_channels = mel_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cond_dim = cond_dim
        self.stft_module = STFTModule(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        self.noise_embedding = NoiseEmbedding(
            sinusoidal_dim=sinusoidal_dim, mid_dim=cond_dim, out_dim=cond_dim
        )
        stft_spec_channels = 2 * (n_fft // 2 + 1) - 2
        input_channels = mel_channels + stft_spec_channels
        self.generator = WaveNextGenerator(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            n_fft=n_fft,
            hop_length=hop_length,
            conditioning_dim=cond_dim,
        )

    def forward(
        self, mel: torch.Tensor, x_t: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            mel: (B, 128, T_mel).
            x_t: (B, T_audio = T_mel * hop_length) 現ステップノイズ付き波形。
            c:   (B,) noise level √(1 - ᾱ_t) ∈ [0, 1].
        Returns:
            (B, T_audio) 予測ノイズ ε_θ (Fig 1b の MSE ターゲット)。
        """
        cond = self.noise_embedding(c)                             # (B, cond_dim)
        T_mel = mel.shape[2]
        stft_spec = self.stft_module(x_t, T_mel=T_mel)             # (B, 2F-2, T_mel)
        x = torch.cat([mel, stft_spec], dim=1)                     # (B, C_in, T_mel)
        return self.generator(x, cond=cond)                        # (B, T_mel * hop)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | GAN 値 | Diff 値 | 出典 |
|---|---|---|---|
| `mel_channels` | 128 | 128 | docs/architecture.md §6.5, milestones.md §M1.3 |
| `n_fft` | 2048 | 1024 | docs/architecture.md §3 STFT パラメータ表 |
| `hop_length` | 300 | 256 | docs/architecture.md §3 / §7 |
| `win_length` | 1200 | 1024 | docs/architecture.md §3 |
| `dim` (ConvNeXt embed) | 512 | 512 | docs/architecture.md §2 / §7 |
| `intermediate_dim` | 1536 | 1536 | docs/architecture.md §2 |
| `n_blocks` | 8 | 8 | docs/architecture.md §2 (Fig 2 キャプション) |
| `kernel_size` | 7 | 7 | docs/architecture.md §2 |
| `stft_spec_channels` (= 2F-2) | 2046 | 1022 | docs/architecture.md §2 入力次元表 |
| `input_channels` (= 128 + 2F-2) | 2176 | 1152 | docs/architecture.md §2 入力次元表 |
| `sinusoidal_dim` (Diff のみ) | n/a | 128 | docs/architecture.md §5.4, milestones.md §M1.5 |
| `cond_dim` (Diff のみ) | n/a | 512 | docs/architecture.md §5.4 |
| `CONCAT_ORDER` | `("mel", "stft_spec")` | `("mel", "stft_spec")` | docs/architecture.md §2 Conv1d in_ch 計算式 |
| パラメータ数目標 | ~14.99M | ~14.42M | docs/architecture.md §4 / §5、Table 1 |

### 2.4 アルゴリズム / 処理フロー

#### SubModelGAN.forward
1. `mel.shape == (B, 128, T_mel)` を assert (動的形状チェックは §6.1 参照)
2. `y_prev.shape == (B, T_mel * hop_length)` の前提で `stft_module(y_prev, T_mel)` を呼び `(B, 2F-2, T_mel)` を得る
3. `torch.cat([mel, stft_spec], dim=1)` で `(B, 128 + 2F-2, T_mel)` を作る
4. `generator(x)` で `(B, T_mel * hop_length)` のノイズ成分を返す (clip(-1, 1) は generator 内部で実施)

#### SubModelDiff.forward
1. `c.shape == (B,)` を assert (連続 noise level、§docs/architecture.md §5.4)
2. `cond = noise_embedding(c)` で `(B, 512)` を得る
3. `stft_module(x_t, T_mel=mel.shape[2])` で `(B, 2F-2, T_mel)`
4. concat → `generator(x, cond=cond)` → `(B, T_mel * hop_length)`

#### 不変条件
- mel と stft_spec の **時間次元 T_mel が必ず一致** している必要がある。STFTModule (T-M1.2) が truncate でこれを担保する責務を持つため、本ラッパでは shape assertion のみ追加。
- channel concat の順序は `[mel, stft_spec]` で固定。T-M2.4 / T-M3.1 はこの順序を前提に何もしない (順序の選択肢を露出しない設計、§6.1 参照)。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `sub_model.py` 実装 + `tests/test_sub_model.py` 記述 | general-purpose |
| Reviewer | 1 | 論文 Fig 2b / `docs/architecture.md` §1〜§4 整合性確認 + 参考実装非コピー確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_sub_model.py -v` 実行と Acceptance 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M1.2〜T-M1.5 完了が前提)
- 並列実行する場合の最大並列数: 1
- 後続 T-M2.4 / T-M3.1 とは順次。本チケット完了後にそれぞれ別タスクとして並列起動可能。

## 4. 提供範囲 (Scope)

### In Scope
- `SubModelGAN`, `SubModelDiff` 2 クラスの本実装
- `CONCAT_ORDER` 定数の公開 (T-M2.4 / T-M3.1 が import 可能)
- `__init__.py` への `__all__` 追加
- `tests/test_sub_model.py` の本実装 (`pytest.skip` 置換、milestones.md §M1.6 Acceptance を網羅)
- パラメータ数検証テスト (±0.05M 許容、§6.1)
- forward + backward の動作確認テスト
- docstring (関数・クラス) を英文 + 日本語混在で記述

### Out of Scope
- T 個 / 4 個の sub-model を並列保持する wrapper (T-M2.4 / T-M3.1 で実装)
- Fixed-point iteration の実装 (T-M2.4)
- Reverse sampling / DDPM step の実装 (T-M3.3)
- Noise schedule / band sampler (T-M3.1)
- Discriminator や Loss (T-M2.2 / T-M2.3)
- 訓練ループ (T-M2.5 / T-M3.2)
- post-filter (T-M3.4)
- パラメータ数ベースの **理論計算式** ドキュメント化 (本チケットでは数値整合だけ確認)
- Generator EMA (Vocos / WaveFit-PT 共に未使用、`docs/architecture.md` §8 で確定)
- Mixed precision / torch.compile 対応 (M5/M6 で必要に応じて検討)

### Deliverable
- ファイル:
  - `src/wavenext2/models/sub_model.py` (新規実装)
  - `tests/test_sub_model.py` (新規実装)
  - `src/wavenext2/models/__init__.py` (re-export 追加)
- 関数 / クラス:
  - `class SubModelGAN(nn.Module)`
  - `class SubModelDiff(nn.Module)`
  - `CONCAT_ORDER: tuple[str, str]` 定数
- ドキュメント差分:
  - `docs/milestones.md` §M1.6 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M1.6 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_sub_model.py`)

#### GAN 系
- [ ] `test_gan_forward_shape`: B=2 で `mel(2, 128, 80) + y_prev(2, 24000)` → `out.shape == (2, 24000)` (milestones.md §M1.6 Acceptance #1)
- [ ] `test_gan_forward_no_cond_arg`: forward が `(mel, y_prev)` の 2 引数のみで呼べる (positional のみ / kwarg どちらでも可)
- [ ] `test_gan_param_count`: `sum(p.numel() for p in model.parameters()) ≈ 14.99M ± 0.05M`
- [ ] `test_gan_backward`: `model(mel, y_prev).sum().backward()` で全 parameter に `grad is not None`
- [ ] `test_gan_concat_order`: `CONCAT_ORDER == ("mel", "stft_spec")` を直接確認 + `model.generator.input_channels == 128 + 2046` を確認
- [ ] `test_gan_output_range`: 出力が `[-1, 1]` に収まる (generator 内 clip)

#### Diff 系
- [ ] `test_diff_forward_shape`: `mel(2, 128, 94) + x_t(2, 24064) + c(2,)` → `out.shape == (2, 24064)` (milestones.md §M1.6 Acceptance #2、24064 = 94 × 256)
- [ ] `test_diff_param_count`: `sum(p.numel() for p in model.parameters()) ≈ 14.42M ± 0.05M`
- [ ] `test_diff_backward`: 全 parameter に grad
- [ ] `test_diff_cond_propagation`: `c = torch.zeros(2)` と `c = torch.ones(2)` で出力が異なる (cosine similarity < 0.999、noise embedding が effective に動作)
- [ ] `test_diff_concat_order`: `model.generator.input_channels == 128 + 1022`
- [ ] `test_diff_c_shape_check`: `c.shape == (B,)` 以外 (例: `(B, 1)`) を渡したときに ValueError or 適切なエラー (`docs/architecture.md` §5.4 の sinusoidal_embedding 仕様準拠)

#### 共通
- [ ] `test_deterministic`: 同 seed / 同入力で出力が一致 (`torch.manual_seed` 固定後 2 回 forward)
- [ ] `test_device_cpu`: CPU で全テスト pass (CUDA 不要環境でも CI が通る、T-M0.2 CI 整合)
- [ ] `test_no_train_eval_diff_for_static_layers`: `.train()` / `.eval()` 切替で出力が変わらない (dropout / batchnorm を含まないため、変わったらバグ)

### 5.2 e2e / 結合テスト
- [ ] T-M2.4 (GANWaveNext2) で `nn.ModuleList([SubModelGAN(**cfg) for _ in range(T)])` が作成可能 (smoke import で確認、本チケットでは未実装でも import 成立だけ確認)
- [ ] T-M3.1 (DiffWaveNext2) で `nn.ModuleList([SubModelDiff(**cfg) for _ in range(4)])` が作成可能
- [ ] 実音声 1 utterance (LibriTTS-R 1 sample、T-M0.3 取得済み想定) を mel 化 → SubModelGAN.forward → backward が通る (T-M2.1 dataset 未完成のため、本チケットでは合成 tensor で代替し、e2e 実音声テストは T-M2.4 / T-M3.1 へ送り)

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.6 より転記)
- [ ] GAN: `mel(B, 128, 80) + y_prev(B, 24000) → out(B, 24000)`
- [ ] Diff: `mel(B, 128, 94) + x_t(B, 24064) + c(B,) → out(B, 24064)`
- [ ] パラメータ数が Table 1 と一致 (GAN: ~14.99M, Diff: ~14.42M)
- [ ] forward + backward が動作

### 5.4 追加 acceptance (本チケット独自)
- [ ] `CONCAT_ORDER` 定数が `("mel", "stft_spec")` で公開され、T-M2.4 / T-M3.1 が import 可能
- [ ] `pytest tests/test_sub_model.py` が **exit code 0** で完了 (skip 0 件、すべて pass)
- [ ] `SubModelGAN` と `SubModelDiff` を `from wavenext2.models import SubModelGAN, SubModelDiff` で import 可能 (`__init__.py` 経由)

## 6. 懸念事項

### 6.1 技術的リスク

- **channel concat 順序の固定化**: 順序を変えると T-M2.4 / T-M3.1 / 学習済み重みの互換性が壊れる。本チケットで `CONCAT_ORDER` 定数を公開し、generator の `input_channels` を計算する基準とすることで **単一情報源 (SoT) 化**。順序を変更する選択肢を **API に露出しない** (引数化しない) ことで誤用を防ぐ。
- **パラメータ数誤差**: ConvNeXt block の LayerScale γ や Linear の bias 有無のわずかな違いで ±数万 params ずれる可能性。本チケットでは **±0.05M (5万 params)** の許容幅で検証。実数値が論文と乖離した場合は T-M1.1 / T-M1.4 にフィードバック (depends_on の修正要求)。
- **STFT module の n_fft/hop 不一致**: GAN (n_fft=2048, hop=300) と Diff (n_fft=1024, hop=256) で別インスタンスを保持する必要がある。共有はしない (sub-model ごとに独立 module、Diff は 4 個独立)。`SubModelGAN`/`SubModelDiff` のデフォルト引数で `docs/architecture.md` §3 の値を埋め込み、誤った組み合わせを物理的に防止。
- **win_length と n_fft が一致しないケース** (GAN: win=1200 < n_fft=2048): `torch.stft` 内部で win を n_fft までゼロパッドする。STFTModule (T-M1.2) が `win_length != n_fft` を正しく扱うことを前提とし、本チケットでは引数を transparent に渡すのみ。
- **T_mel と y_prev の長さ不整合**: `y_prev.shape[1] != T_mel * hop_length` のときに silent failure する。本チケットでは forward 内で `assert y_prev.shape[1] == T_mel * hop_length` を入れる (TorchScript 互換のため `assert` ではなく `if ...: raise ValueError(...)` を採用)。
- **mel の dtype**: `LogMelSpectrogram` 出力は `float32`、generator は `float32` 前提。本チケットでは dtype 変換は行わない (`mel.dtype == y_prev.dtype` を期待し、混在時は明示エラー)。
- **B=1 と B=2 の grad accumulation**: BatchNorm を含まないため B=1 でも grad が問題なく流れるはずだが、`test_gan_backward` は B=2 で実行する (B=1 fallback テストは Out of Scope)。

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み**。
  - channel concat 順序: `docs/architecture.md` §2 入力次元表で `C_in = 128 + (2F-2)` (mel が先) と確定
  - Diff conditioning 注入 (additive bias, per-block): `docs/architecture.md` §5.4 で確定、本チケットでは generator (T-M1.4) に責務を委譲
  - Noise embedding の sinusoidal + FC×2 SiLU: T-M1.5 で確定
  - GAN sub-model の forward が `(mel, y_prev)` を **位置引数** で受けるか **キーワード** で受けるか → **位置引数 (positional)** を採用 (`docs/architecture.md` §1 図中の信号流に対応)
- 旧来の判断ポイント (本チケット作成時に確定):
  - **共通親クラス導入 (`SubModelBase`)**: 採用しない (§8.1 参照、composition 優先)
  - **`forward()` で `T_mel = mel.shape[2]` を渡すか shape 推論するか**: shape 推論 (STFTModule の引数として渡す)。理由: STFTModule (T-M1.2) は `T_mel` を明示引数として既に取る仕様 (`milestones.md §M1.2`)。
  - **mel と stft_spec の concat 軸**: `dim=1` (channel 軸)、`docs/architecture.md` §2 で確定。

### 6.3 他チケットとの整合性

- **T-M1.2 (STFTModule)** との整合:
  - 期待 signature: `stft_module(y, T_mel) -> (B, 2F-2, T_mel)`
  - 期待 attribute: `n_fft`, `hop_length`, `win_length` を `__init__` 引数として受ける
  - 不整合があった場合は T-M1.2 側を修正 (T-M1.6 は consumer)
- **T-M1.3 (LogMelSpectrogram)** との整合:
  - mel は外部から渡される前提 (本 sub-model は mel 抽出を内部で行わない)
  - 整合確認: 1 秒入力で GAN は (B, 128, 80)、Diff は (B, 128, 94) になることを `tests/test_sub_model.py` で fixture として用意し T-M1.3 と同値性を担保
- **T-M1.4 (WaveNextGenerator)** との整合:
  - 期待 signature: `generator(x, cond=None) -> (B, T_mel * hop)`
  - 期待 attribute: `input_channels`, `hop_length` (テストで参照)
  - `conditioning_dim=None` で GAN モード、`conditioning_dim=512` で Diff モード
  - 不整合があった場合は T-M1.4 側を修正
- **T-M1.5 (NoiseEmbedding)** との整合:
  - 期待 signature: `noise_embedding(c) -> (B, 512)` where `c: (B,)`
  - 不整合があった場合は T-M1.5 側を修正
- **T-M2.4 (GANWaveNext2)** へ渡す情報:
  - `SubModelGAN` を `nn.ModuleList` に T 個並べるだけで fixed-point iteration が組めるよう、状態 (state) を保持しない設計 (forward は pure function)
  - 初期 y_T = `torch.zeros_like(...)` は T-M2.4 側で生成する責務 (本チケットでは生成しない)
- **T-M3.1 (DiffWaveNext2)** へ渡す情報:
  - `SubModelDiff` を `nn.ModuleList` に 4 個並べる
  - 4 sub-model 間で **重みを共有しない** (各々独立、`docs/architecture.md` §5.4 末尾「sub-model 間で共有しない」)
  - noise level `c = √(1-ᾱ_t)` を呼び出し側で計算 (本チケットでは受け取るだけ)
  - band sampler は T-M3.1 / T-M3.2 が責務

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §1〜§4 (特に §2 入力次元表) と整合
- [ ] `docs/architecture.md` §5.4 (Diff conditioning) と整合 (本クラスは generator/noise_embedding に委譲しているのみで、本クラス内で FiLM や別経路で cond を扱っていないこと)
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 全クリア
- [ ] パラメータ数が GAN ~14.99M / Diff ~14.42M (±0.05M)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: `mel.shape`, `y_prev.shape`, `c.shape` 不整合時に `ValueError` (silent failure なし)
- [ ] **参考実装 (Vocos / WaveFit / FastDiff) をコピーしていない** (CLAUDE.md 末尾の方針、`docs/architecture.md` の論述から再構成しているか)
- [ ] `__init__.py` の `__all__` に `SubModelGAN`, `SubModelDiff` が追加され、それ以外の internal helper は export されていない
- [ ] `CONCAT_ORDER` 定数が module top-level で公開され、docstring に意図 (T-M2.4 / T-M3.1 が import 可能) が明示されている
- [ ] STFT module / Generator / NoiseEmbedding の利用が **構成 (composition)** であり継承ではない
- [ ] forward に副作用がない (state を持たない、再現性のため)
- [ ] テストが `tests/test_sub_model.py::test_*` 命名規約に従う
- [ ] CPU でテストが pass する (CUDA を要求しない)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M1) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計
- **2 クラス (`SubModelGAN`, `SubModelDiff`) を構成 (composition) で書き、共通親クラスは導入しない**
  - 理由 (3 観点):
    - (a) 共通インターフェース (`forward(mel, *args)`) の引数数が GAN=2 / Diff=3 で異なり、`nn.Module` 標準の `forward` の variadic 設計と相性が悪い (TorchScript / TypedDict 化が苦しい)
    - (b) GAN/Diff の差分は「conditioning の有無」だけで、抽象基底クラスを置くと差分が下位クラスの `__init__` に集中するだけで、保守コストに見合わない
    - (c) `docs/architecture.md` §1 図中で「sub-model 構造は同じ」と書かれているのは **層構造の共有 (= 部品の流用)** であり、Python のクラス階層を共有することではない。実装上は **STFTModule / WaveNextGenerator / NoiseEmbedding の流用** で意図を満たす

- **mel と stft_spec の concat 順序を `[mel, stft_spec]` (mel が先) に固定**
  - 理由:
    - `docs/architecture.md` §2 入力次元表で `C_in = 128 + (2F-2)` (mel が先と読める並び)
    - mel を先頭にすると後続の T-M2.4 / T-M3.1 が `x[:, :128, :]` で mel を取り出して可視化 / 検証する際に slice が直感的
    - 反対順 `[stft_spec, mel]` を採用するメリットがない

#### Deprecated (旧案、却下根拠)

1. **`SubModelBase` 抽象クラス + `SubModelGAN`/`SubModelDiff` が継承**
   - メリット: forward の差分を `_compute_cond(mel, *args)` 等の hook method に切り出せる
   - 却下: 上記 (a)(b) に加え、Python の class hierarchy が論文の「sub-model 構造は同じ」意図と乖離しやすい (継承だと「同じ」を強調しすぎる)。**composition + 部品共有** で意図を表現する方が論文に忠実。

2. **`SubModel(mode: Literal["gan", "diff"])` 1 クラスで分岐**
   - メリット: ファイル数が減る
   - 却下: forward 内 `if mode == "gan"` 分岐で TorchScript / `torch.compile` 最適化を阻害、テスト時に両モードを同時 instantiate するコストが高い、`docs/architecture.md` の Fig 2b と 1 対 1 にならない

3. **STFT module の代わりに learnable conv (1D conv with stride=hop)**
   - メリット: 学習可能で表現力が高い
   - 却下: `docs/architecture.md` §3 で STFT module は **解析的な前処理** と明示。learnable にすると論文の「統一フレームワーク」の意義 (前ステップ波形を解析して条件化) が崩れる。

4. **mel と STFT-spec を sum / avg で混ぜる**
   - メリット: パラメータ削減 (Conv1d input_channels が 128 のみで済む)
   - 却下: mel と STFT-spec で意味が異なる (mel = 知覚特徴, STFT = 物理特徴) ため sum/avg は破壊的。論文は `docs/architecture.md` §2 で **concat** と明記。

5. **mel と STFT-spec を FiLM で混ぜる (mel が backbone, STFT が cond)**
   - メリット: 物理特徴で意味特徴を変調する解釈ができる
   - 却下: 論文 Fig 2b は明確に concat。FiLM は §5.4 で **noise embedding** のために予約されている設計位置。混在させると 2 種の FiLM が共存して可読性低下

6. **`forward(mel, y_prev)` ではなく `forward(features: dict)` で受ける**
   - メリット: GAN/Diff で signature を統一できる
   - 却下: dict-based API は TorchScript / `torch.compile` と相性が悪い、IDE 補完が効かない、型チェックが弱い

7. **`__init__` で `STFTModule` を引数として受け取る (DI)**
   - メリット: テスト時に mock を注入できる
   - 却下: M1 段階では mock が不要 (STFTModule 自体が pure function で副作用なし)。M5/M6 で実装ニーズが出たらリファクタを検討 (§8.1 再評価トリガー参照)

#### 追加検討した設計案

- **`CONCAT_ORDER` を tuple ではなく `Enum` で定義**: 型安全性は向上するが M1 段階では over-engineering。tuple で十分 (3 エージェント合議で確認、本チケット独自判断)。
- **`SubModelGAN` の forward を `(mel, y_prev=None)` にして `y_prev=None` のとき内部で `torch.zeros_like(...)` 生成**: T-M2.4 (GAN モデル) で初期 y_T を作る責務を吸収できる
  - 採否: **却下**。責務を sub-model に持たせると iteration 内部状態が漏れて、T-M2.4 の fixed-point iteration の流れが見えにくくなる。初期化は T-M2.4 側で行う。

#### 再評価トリガー条件
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| `SubModelGAN` / `SubModelDiff` 2 クラス分割 | M3 完了時 | 共通テスト fixture が肥大化したら `SubModelBase` 抽象化を再検討 |
| `CONCAT_ORDER` 定数公開 | M2.4 / M3.1 実装時 | T-M2.4 / T-M3.1 で実際に import するかを確認、しなければ削除可 |
| forward の `T_mel = mel.shape[2]` 推論 | M1 phase review | mel の shape を `[B, T_mel, 128]` (channels last) に変更した方が PyTorch 慣例に合う場合がある |
| STFTModule の DI 化 | M5 phase review | torch.compile / TorchScript で問題が出たら再考 |
| `forward(features: dict)` への移行 | M6 phase review | データセット側で features dict を返す設計に統一する場合 |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切。
  - sub_model.py は **STFT + Generator + (NoiseEmbedding)** を束ねるだけの thin layer であり、独立チケット化することで T-M2.4 / T-M3.1 が「sub_model を並べる」だけに集中できる
  - 逆に、本チケットを M2.4 / M3.1 に統合すると、両方のチケットで同じ wrapper を再実装するか、片方からのみ参照する変則的依存になる
- **別マイルストーンに移すべき部分はないか**: なし。M1 (コア部品) の最後のサブタスクとして、M1.1〜M1.5 の集約点に位置するのが自然。
- **インターフェース定義の見直し余地**:
  - **`CONCAT_ORDER` 定数の公開**: T-M2.4 / T-M3.1 / 学習済み重みのロード時に「mel が先か stft が先か」を **コード内で参照** できるようにする。docstring だけだと変更検知が弱い。
  - **`SubModelGAN.forward` / `SubModelDiff.forward` の signature を Protocol で型定義する案**: M5/M6 で議論。M1 段階では `nn.Module` のままで十分。
  - **State (前回の y_prev) を sub-model 内部に持つ案**: 却下済 (§8.1 案 7 と同じ理由)。iteration 状態は呼び出し側 (T-M2.4) で管理する。

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.4 (GAN モデル `gan_wavenext2.py`)
- **使用方法**:
  ```python
  from wavenext2.models.sub_model import SubModelGAN

  class GANWaveNext2(nn.Module):
      def __init__(self, T: int = 4, sub_model_cfg: dict | None = None):
          super().__init__()
          sub_model_cfg = sub_model_cfg or {}
          self.T = T
          self.sub_models = nn.ModuleList([SubModelGAN(**sub_model_cfg) for _ in range(T)])

      def forward(self, mel: torch.Tensor, audio_length: int) -> torch.Tensor:
          B = mel.shape[0]
          y = torch.zeros(B, audio_length, device=mel.device, dtype=mel.dtype)
          for t in range(self.T):
              n_t = self.sub_models[t](mel, y)  # ノイズ予測
              y = y - n_t                        # 残差で denoising
          return y  # y_0
  ```
- **重要事項**:
  - `SubModelGAN` は state を持たない (各 forward は独立)
  - 初期 y_T = zeros は T-M2.4 側で生成する (sub_model 内部で生成しない、`docs/open-questions.md` §D の `torch.zeros_like(x_gt)` を T-M2.4 で実装)
  - T=4 で `sub_models` のパラメータ総数 ≈ 14.99M × 4 = 59.94M (Table 1 整合)
- **`CONCAT_ORDER`**: 必要なら `from wavenext2.models.sub_model import CONCAT_ORDER` で取得して可視化/ログに利用可

#### T-M3.1 (Diff モデル `diff_wavenext2.py`)
- **使用方法**:
  ```python
  from wavenext2.models.sub_model import SubModelDiff

  class DiffWaveNext2(nn.Module):
      NOISE_SCHEDULE_ABAR = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])
      BAND_BOUNDS = [(0.9929, 1.0), (0.8246, 0.9929), (0.4817, 0.8246), (0.0, 0.4817)]

      def __init__(self, sub_model_cfg: dict | None = None):
          super().__init__()
          sub_model_cfg = sub_model_cfg or {}
          # 4 sub-model 独立 (重み共有なし、docs/architecture.md §5.4 末尾)
          self.sub_models = nn.ModuleList([SubModelDiff(**sub_model_cfg) for _ in range(4)])
  ```
- **重要事項**:
  - 4 sub-model それぞれが独立パラメータを保持 (重み共有なし)
  - noise level `c = √(1-ᾱ_t)` は呼び出し側で計算 (本チケットでは受け取るだけ)
  - パラメータ総数 ≈ 14.42M × 4 = 57.68M (Table 1 整合)
  - point-specialized 1-to-1 マッピングは T-M3.1 / T-M3.3 で実装 (本チケットでは関知しない)

#### 共通の注意事項
- **channel concat 順序**: `[mel (128 ch), stft_spec (2F-2 ch)]` で **確定**。順序変更は破壊的変更となり、学習済み重みのロード時に互換性が壊れる。
- **forward の dtype/device**: `mel.device == y_prev.device` (Diff は `x_t.device` も) を呼び出し側で保証。本チケットでは混在チェックは行わない (PyTorch 標準のエラーに任せる)。
- **B=1 動作**: テスト済 (`test_gan_backward` は B=2 だが、ConvNeXt / Linear は B=1 でも正常動作)。

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M1.6 の Acceptance チェックボックス 4 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M1.6 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/architecture.md` §2 入力次元表の更新 (channel concat 順序を明示的に追記する余地あり、レビュー時に判断)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` で全 100% 確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - `SubModelGAN` / `SubModelDiff` 2 クラス分割を維持するか抽象化するか (M3 完了後)
  - `CONCAT_ORDER` 定数を維持するか削除するか (M2.4 / M3.1 実装後)
  - mel の channels axis (channels first vs last) 慣例 (M1 phase review)
  - STFTModule の DI 化 (M5 phase review)
- **`docs/open-questions.md` への追記要否**: 不要
