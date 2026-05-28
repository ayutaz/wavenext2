---
id: T-M1.6
title: Sub-model wrapper (STFT module + Generator、GAN/Diff 両対応)
milestone: M1
phase: M1
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-27
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
- [ ] **`tests/test_sub_model.py::test_real_audio`** (M1 phase review 追加): LibriTTS-R sample 1 件で実行 (T-M0.3 完了後)、`@pytest.mark.slow` でデフォルト除外。実音声を mel 化 → SubModelGAN.forward → backward が通る。T-M2.1 dataset 未完成期間は `pytest.skip` で skip
- [ ] **`test_concat_order_pin`** (M1 phase review 追加): `CONCAT_ORDER` の SHA256 / 文字列を pin することで、order 変更を **破壊的変更として CI で検知**。例:
  ```python
  EXPECTED_CONCAT_ORDER_SHA = "..."  # ("mel", "stft_spec") の SHA256
  assert hashlib.sha256(repr(SubModelGAN.CONCAT_ORDER).encode()).hexdigest() == EXPECTED_CONCAT_ORDER_SHA
  ```

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.6 より転記)
- [ ] GAN: `mel(B, 128, 80) + y_prev(B, 24000) → out(B, 24000)`
- [ ] Diff: `mel(B, 128, 94) + x_t(B, 24064) + c(B,) → out(B, 24064)`
- [ ] パラメータ数が Table 1 と一致 (GAN: ~14.99M, Diff: ~14.42M)
- [ ] forward + backward が動作

### 5.4 追加 acceptance (本チケット独自)
- [ ] `CONCAT_ORDER` 定数が `("mel", "stft_spec")` で公開され、T-M2.4 / T-M3.1 が import 可能 (および `SubModelGAN.CONCAT_ORDER` クラス属性経由でも参照可能、§8.2 参照)
- [ ] `pytest tests/test_sub_model.py` が **exit code 0** で完了 (skip 0 件 / GPU マーカー除く、すべて pass)
- [ ] `SubModelGAN` と `SubModelDiff` を `from wavenext2.models import SubModelGAN, SubModelDiff` で import 可能 (`__init__.py` 経由)
- [ ] `SubModelGAN.from_config(cfg, mode="gan")` / `SubModelDiff.from_config(cfg, mode="diff")` で生成可能 (factory パターン、§8.1 案 8 / §8.2 参照)

### 5.5 テスト戦略 (M1 phase review 追加)

- **GPU テスト分離**: `@pytest.mark.gpu` マーカーで GPU テストを CI runner 別に分離 (T-M0.2 の `.github/workflows/test.yml` skeleton と整合)。CPU only CI では `pytest -m "not gpu"` で除外
- **`@pytest.mark.slow` マーカー**: `test_real_audio` 等 LibriTTS-R 実音声を使うテストは `slow` でデフォルト除外、`pytest -m slow` で明示実行
- **CI 時間目標**: SubModelGAN ≈ 14.99M params の forward+backward は CPU で 1 ケース **~3 秒**、5 ケースで **~15 秒**。M1 フェーズ全体のテスト合計目標 **30 秒以内**
  - **必須最適化**: `scope="module"` fixture で `SubModelGAN` / `SubModelDiff` インスタンスを再利用 (各テストで `init` し直さない)
- **coverage 目標**: 本チケットのカバレッジ目標 **85%** (e2e 部分は T-M2.4 / T-M3.1 でカバー)。残り 15% は backward path 一部 + factory error path

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (✅ 2026-05-27 実装時に解決)

- **✅ 解決: `SubModelGAN.forward(mel, y_prev)` の戻り値 = n_t (clip[-1,1])**:
  - architecture.md §4 (`n_t = sub_model(mel, y_t); y_{t-1} = y_t - n_t`) より、**generator 出力 = n_t (残差、clip[-1,1])**。減算 `y_{t-1}=y_t-n_t` は **呼び出し側 (T-M2.4)** が行う。sub-model 自身は減算しない。docstring に明記済み。
  - **✅ 関連解決: Diff の ε は clip しない**。training.md §4.2 reverse は `ε_pred` を生で使い波形 x0_hat のみ clamp。ε~N(0,1) は |ε|>1 が ~32% で clip すると破壊的 → **SubModelDiff の generator は `final_activation="none"`** (WaveNextGenerator に "none" を追加)。SubModelGAN は "clip"。

- ~~(旧) 戻り値の意味の曖昧さ~~ (上記で解決):
  - 現状 docstring に「ノイズ成分 n_t」と書きながら `test_gan_output_range` では `[-1, 1]` clip を期待しており **論理矛盾**
  - GAN の `y_t - n_t = y_{t-1}` で `y_{t-1} ∈ [-1, 1]` を担保するには `n_t = y_t - y_{t-1} ∈ [-2, 2]` であり、clip 範囲 `[-1, 1]` だと `t→0` で `y_{t-1}` が clip されて情報損失
  - **仮説**: 実は generator 出力は `y_{t-1}` そのもの (denoised waveform) で `n_t = y_t - y_{t-1}` を後段 (T-M2.4 GANWaveNext2) で計算する設計の方が WaveFit 準拠の可能性
  - **本チケット実装着手時の MUST DO**:
    1. `docs/architecture.md` §2 を再読
    2. `docs/training.md` §2 GAN training の式 (`y_t - n_t = y_{t-1}` か `y_t = y_{t-1}` か) を再読
    3. WaveFit (参考実装 `yukara-ikemiya/wavefit-pytorch`) の sub-model forward 戻り値仕様を確認
    4. 戻り値仕様を **docstring + テスト名で確定** (例: `test_gan_output_is_y_prev_clipped` または `test_gan_output_is_noise_residual`)
    5. 確定した仕様を T-M2.4 (GANWaveNext2) に §9.1 経由で伝達
  - **このタスクが未解決のままだと T-M2.4 で fixed-point iteration の `y = y - n_t` か `y = sub_model(mel, y)` かの選択を誤る**

#### 通常項目

- **CRITICAL候補: mel と STFT-spec の dynamic range mismatch** (M1 phase review):
  - T-M1.2 §6 と同件。`torch.cat([mel, stft_spec], dim=1)` 直後の `Conv1d(2176, 512, k=7)` が STFT-spec の大きい振幅に dominated されて mel ch [0:128] が無視される懸念
  - **検証手段**: `test_input_balance` で `mel` 側と `stft_spec` 側の grad norm を比較。`weight[:, :128, :].grad.norm() / weight[:, 128:, :].grad.norm() < 0.01` (mel side grad が 1% 未満) なら **mel が無視されている** と判定
  - 検証結果に応じて T-M1.2 (STFT 側 LayerNorm 追加) または T-M1.4 (入力 LayerNorm 追加) にフィードバック
- **CRITICAL候補: `y_prev.shape` assertion の center=True 整合性** (M1 phase review):
  - 現状 `assert y_prev.shape[1] == T_mel * hop_length` を想定するが、`torch.stft(center=True)` で `T_mel = floor(T_audio/hop) + 1` の場合、逆算すると `T_audio = (T_mel - 1) * hop` で **1 hop ぶん不足**
  - **T-M1.3 で T_mel が +1 されるか確定したら**、本コードで `T_audio == T_mel * hop_length` または `T_audio == (T_mel - 1) * hop_length` のどちらが正しいかを **コードで pin** (定数化 + テスト化)
  - 不整合のまま実装すると forward 時 silent shape mismatch
- **gradient checkpointing propagation** (M1 phase review):
  - T-M2.4 で T=4 直列 (sub-model 4 段) を 24GB GPU で fp32 forward+backward する場合、activation memory が OOM の危険
  - **対応**: `SubModelGAN(enable_grad_ckpt: bool = False)` 引数を **本チケットで予約**。内部実装は generator (T-M1.4) に propagate する責務
  - T-M1.4 側で `WaveNextGenerator(enable_grad_ckpt=...)` の引数を受け取る設計を要請 (T-M1.4 がまだ受け取らない場合は T-M1.4 のリビジョンを要求)
  - T-M2.4 では YAML config 経由で T 個 sub-model すべてに `enable_grad_ckpt=True` を propagation
- **channel concat 順序の固定化**: 順序を変えると T-M2.4 / T-M3.1 / 学習済み重みの互換性が壊れる。本チケットで `CONCAT_ORDER` 定数を公開し、generator の `input_channels` を計算する基準とすることで **単一情報源 (SoT) 化**。順序を変更する選択肢を **API に露出しない** (引数化しない) ことで誤用を防ぐ。`CONCAT_ORDER` は `SubModelGAN.CONCAT_ORDER` クラス属性化で「sub_model に閉じた SoT」とする (§8.2 参照)。
- **パラメータ数誤差**: ConvNeXt block の LayerScale γ や Linear の bias 有無のわずかな違いで ±数万 params ずれる可能性。本チケットでは **±0.05M (5万 params)** の許容幅で検証。実数値が論文と乖離した場合は T-M1.1 / T-M1.4 にフィードバック (depends_on の修正要求)。
- **STFT module の n_fft/hop 不一致**: GAN (n_fft=2048, hop=300) と Diff (n_fft=1024, hop=256) で別インスタンスを保持する必要がある。共有はしない (sub-model ごとに独立 module、Diff は 4 個独立)。`SubModelGAN`/`SubModelDiff` のデフォルト引数で `docs/architecture.md` §3 の値を埋め込み、誤った組み合わせを物理的に防止。
- **win_length と n_fft が一致しないケース** (GAN: win=1200 < n_fft=2048): `torch.stft` 内部で win を n_fft までゼロパッドする。STFTModule (T-M1.2) が `win_length != n_fft` を正しく扱うことを前提とし、本チケットでは引数を transparent に渡すのみ。
- **T_mel と y_prev の長さ不整合**: `y_prev.shape[1] != T_mel * hop_length` のときに silent failure する。本チケットでは forward 内で `assert y_prev.shape[1] == T_mel * hop_length` を入れる (TorchScript 互換のため `assert` ではなく `if ...: raise ValueError(...)` を採用)。**ただし上記 center=True 整合性 critical 項目の解決を待って expected 長を確定**。
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

#### 新規追加案 (M1 phase review で採用 / 検討)

8. **`from_config(cls, cfg: dict, mode: Literal["gan", "diff"])` classmethod factory** (採用):
   - **問題**: T-M2.4 / T-M3.1 が `SubModelGAN(**sub_model_cfg)` と書く際、`sub_model_cfg` 内に余分な key が紛れ込むと `TypeError`
   - **採用根拠**:
     - Factory で受理 key を絞ると **config drift 耐性が高い** (YAML config に新 key が追加されてもクラスは無視可能)
     - T-M1.2, T-M1.3, T-M1.4, T-M1.5 の factory パターンと **統一** (M1 phase review 横断テーマ、§8.2 参照)
     - mode 引数で GAN/Diff の取り違えを runtime check 可能 (今は `SubModelGAN()` / `SubModelDiff()` の class 名で区別するが、factory 経由なら 1 関数で両対応)
   - **API 案**:
     ```python
     @classmethod
     def from_config(cls, cfg: dict, mode: Literal["gan", "diff"]) -> "SubModelGAN | SubModelDiff":
         allowed_keys = cls._allowed_keys(mode)
         filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
         if mode == "gan":
             return SubModelGAN(**filtered)
         elif mode == "diff":
             return SubModelDiff(**filtered)
         else:
             raise ValueError(f"unknown mode: {mode}")
     ```

9. **`STFTModule` window buffer を 4 sub-model 間で共有** (Diff のみ):
   - **問題**: Diff で 4 sub-model 全部 n_fft=1024 が同じなので window buffer の 4 重複は無駄 (`n_fft + 1` float32 = ~4KB × 4 = 16KB、微小だがアーキ整合性の問題)
   - **案**: `SubModelDiff` 外で `STFTModule` を 1 個 init して DI (`SubModelDiff(stft_module=...)`)
   - **再評価トリガー**: M3.1 実装時のメモリ profiling で 16KB が問題にならなければ却下、問題なら採用
   - **現状**: 本チケットでは未採用 (M1 段階では sub-model ごとに独立保持、§8.1 案 7 と整合)。M3.1 で要再評価。
   - **✅ M3.1 で不採用確定 (2026-05-28)**: 16KB は微小で optimization 不要、sub-model が独立構造を保つ方が構造的整合性が高い (各 sub-model が pure に独立)。`DiffWaveNext2` は 4 個の `SubModelDiff` を `nn.ModuleList` で保持し、各々が独立 `STFTModule` を持つ。詳細は T-M3.1 §6.1 / §9.1。

10. **`runtime_checkable Protocol` で `forward` signature を duck-type 統一**:
    - **メリット**: 継承不要、TorchScript 互換、Python 3.13 の `typing.Protocol` で型チェック可能
    - **案**:
      ```python
      from typing import Protocol, runtime_checkable
      @runtime_checkable
      class SubModelLike(Protocol):
          def forward(self, mel: torch.Tensor, *args: torch.Tensor) -> torch.Tensor: ...
      ```
    - **採否**: **保留** (M5 で再評価)。M1 では 2 クラス分割で十分。

11. **STFTModule / Generator / NoiseEmbedding を DI で受け取る案** (テスト性向上):
    - **メリット**: `SubModelGAN(stft_module=..., generator=...)` で pytest-mock 注入し sub-model wrapper **単独テスト < 50ms** 化 (現状は STFT/Generator/NoiseEmbedding 全部 init で重い)
    - **採否**: **保留** (M3 完了後に再評価)。M1 段階では DI なしで実装、テスト速度が問題になったら採用。
    - 再評価トリガー: `tests/test_sub_model.py` 全体が 30 秒を超えた場合

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

#### M1 phase review で追加された設計原則

- **`CONCAT_ORDER` をクラス属性に変更**:
  - **現状**: module-level 定数 `CONCAT_ORDER = ("mel", "stft_spec")`
  - **変更案**: `SubModelGAN.CONCAT_ORDER` / `SubModelDiff.CONCAT_ORDER` のクラス属性化
  - **採用根拠**: 「sub_model に閉じた SoT」とすることで、T-M2.4 / T-M3.1 が `SubModelGAN.CONCAT_ORDER` で参照可能。module-level 定数だと「どのクラスの concat 順序か」が不明瞭
  - **実装メモ**: module-level 定数は互換性のため残し、クラス属性を **正本** とする (重複定義リスクは class 内 `assert SubModelGAN.CONCAT_ORDER == CONCAT_ORDER` で検出可能)

- **`SubModelGAN(mel_channels=128, n_fft=2048, ...)` の default 値多過ぎ問題** (M1 phase review):
  - **現状**: §2.2 のシグネチャで全パラメータに default 値あり (例: `mel_channels: int = 128`)
  - **問題**: T-M1.3 で確立した「mel パラメータは config 経由のみ、ハードコード禁止」原則と矛盾。T-M2.4 / T-M3.1 で `SubModelGAN()` (引数なし) と呼ばれた時に GAN/Diff の取り違えが **silent に発生** (e.g., n_fft=2048 のままで Diff config を流し込む)
  - **対応**:
    - 案 A: **default 値を全削除**して引数必須化 (`mel_channels: int` のみ)
    - 案 B: `from_config(cfg)` factory を **唯一の生成経路** にして直接 `__init__` 呼び出しを禁止 (§8.1 新規追加案 8 と整合)
  - **採用**: **案 B (factory パターン)** を本チケット実装時に採用。`__init__` の default 値は維持しつつ、docstring に「**production code では `from_config` を使うこと**」を明記
  - **連絡先**: T-M2.4 / T-M3.1 §9.1 で `from_config` 経由生成を要請

- **factory パターン全モジュール一貫化** (M1 phase review 横断テーマ):
  - M1 のすべての主要モジュール (`STFTModule`, `LogMelSpectrogram`, `WaveNextGenerator`, `NoiseEmbedding`, `SubModelGAN/Diff`) が `from_config(cfg)` classmethod を持つようにする
  - 効果:
    - YAML config 経由生成の **一貫した API** (T-M2.4 / T-M3.1 / T-M2.5 / T-M3.2 の訓練 entry point が単純化)
    - config drift 耐性 (新 key が増えても古いクラスは無視)
    - config validation の集約点 (`from_config` 内で `allowed_keys` を絞る)
  - 本チケットは M1 の最終チケットなので、factory パターンの **整合性を最終確認する責務** を持つ

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `SubModelGAN` / `SubModelDiff` (composition: STFTModule + WaveNextGenerator + Diff は NoiseEmbedding) 実装、`tests/test_sub_model.py` 18 件 pass。`CONCAT_ORDER=("mel","stft_spec")` module 定数 + class 属性、`from_config` factory (余分 key 無視)。
- param: SubModelGAN = 15.43M (= generator)、SubModelDiff = 16.46M (= generator 16.13M + NoiseEmbedding 0.33M)。

統合で解決した想定外:
1. **§6.1 CRITICAL 戻り値意味の確定**: architecture.md §4 より **SubModelGAN は n_t (clip[-1,1] 残差) を返し、減算 y_{t-1}=y_t-n_t は T-M2.4 が行う**。sub-model は減算しない。
2. **Diff の ε を clip してはいけない (重要)**: training.md §4.2 reverse は ε_pred を生で使い波形のみ clamp。ε~N(0,1) は \|ε\|>1 が ~32% で clip すると破壊的。→ **WaveNextGenerator に `final_activation="none"` を追加** (T-M1.4 を改訂)、SubModelDiff はこれを使う。GAN は "clip"。**教訓: 同一 generator を GAN(波形/残差) と Diff(ε) で再利用する場合、出力 activation はモードで変える必要がある (統合して初めて顕在化)**。
3. **cond propagation テストは未訓練だと cosine 0.9998**: LayerScale γ=1e-6 で block が near-identity のため init 時の conditioning effect は小さい。閾値 cosine<0.999 は訓練前提で過剰 → 「出力が変わる (`(o0-o1).abs().max()>1e-3`)」の直接検証に変更。
4. **Diff param +14% 超過 (per-block fc_t)**: SubModelDiff 16.46M vs Table 1 14.42M → ✅ **解決 (2026-05-27)**: per-block fc_t 撤去 (射影なし additive 注入)、SubModelDiff = **14.354M** (Table 1 −0.46%)。詳細 `docs/open-questions.md` §C7 (T-M1.4 §8.3 と同件)。

次の似たタスクで応用できる教訓:
- 共有部品 (generator) を複数モードで使い回すと、出力 activation や clip など「末端の差」が統合時に露見する。早めに最小統合テストを書くと検出が早い。
- 未訓練モデルのテストは「学習後に期待する強い性質」でなく「機構が伝播する弱い性質」で書く (LayerScale init を考慮)。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M0.2 (conftest.py / fixture) へ
- **`gan_sub_model_cfg` / `diff_sub_model_cfg` YAML fixture** を `conftest.py` に追加することを推奨
  - これにより T-M2.4 / T-M3.1 でも同じ fixture を継承可能
  - 例:
    ```python
    # tests/conftest.py
    @pytest.fixture(scope="session")
    def gan_sub_model_cfg() -> dict:
        return {"mel_channels": 128, "n_fft": 2048, "hop_length": 300, ...}
    @pytest.fixture(scope="session")
    def diff_sub_model_cfg() -> dict:
        return {"mel_channels": 128, "n_fft": 1024, "hop_length": 256, ...}
    ```

#### T-M2.4 (GAN モデル `gan_wavenext2.py`)
- **使用方法** (戻り値仕様確定後に確定する設計、暫定):
  ```python
  from wavenext2.models.sub_model import SubModelGAN

  class GANWaveNext2(nn.Module):
      def __init__(self, T: int = 4, sub_model_cfg: dict | None = None):
          super().__init__()
          sub_model_cfg = sub_model_cfg or {}
          self.T = T
          # factory 経由で生成 (§8.1 案 8、§8.2 整合)
          self.sub_models = nn.ModuleList([
              SubModelGAN.from_config(sub_model_cfg, mode="gan") for _ in range(T)
          ])

      def forward(self, mel: torch.Tensor, audio_length: int) -> torch.Tensor:
          B = mel.shape[0]
          y = torch.zeros(B, audio_length, device=mel.device, dtype=mel.dtype)
          # 逆順イテレーション (t=T → t=1)
          for t in range(self.T, 0, -1):
              out = self.sub_models[self.T - t](mel, y)
              # 以下の 2 パターンを §6.1 critical 解決後に確定:
              # (A) n_t = out, y = y - n_t
              # (B) y_{t-1} = out, y = out
              y = ...
          return y  # y_0
  ```
- **重要事項**:
  - **戻り値仕様 (n_t vs y_{t-1})** は §6.1 critical 項目で確定。T-M2.4 で `n_t = y_t - y_{t-1}` 後段計算 or `y_{t-1}` 直接出力のどちらを使うか伝達
  - **`enable_grad_ckpt=True`** を YAML config で T 個 sub-model **すべてに propagation** (§6.1 通常項目)
  - **`nn.ModuleList`** で 4 個 (T=4) sub-model を保持
  - **forward は逆順** (`for t in range(T, 0, -1)`)。`docs/architecture.md` §2 の iteration 順序と整合
  - `SubModelGAN` は state を持たない (各 forward は独立)
  - 初期 y_T = zeros は T-M2.4 側で生成する (sub_model 内部で生成しない、`docs/open-questions.md` §D の `torch.zeros_like(x_gt)` を T-M2.4 で実装)
  - T=4 で `sub_models` のパラメータ総数 ≈ 14.99M × 4 = 59.94M (Table 1 整合)
- **`CONCAT_ORDER`**:
  - `from wavenext2.models.sub_model import CONCAT_ORDER` または `SubModelGAN.CONCAT_ORDER` で取得可
  - **確定値**: `CONCAT_ORDER = ("mel", "stft_spec")` で `[mel (128 ch), stft_spec (2F-2 ch)]` の順 (本チケット + T-M1.2 で一貫)

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
          # factory 経由 (§8.1 案 8)
          self.sub_models = nn.ModuleList([
              SubModelDiff.from_config(sub_model_cfg, mode="diff") for _ in range(4)
          ])
  ```
- **重要事項**:
  - **`STFTModule` window 共有 (DI)** の検討: §8.1 新規追加案 9 参照。M3.1 実装時にメモリ profiling を実施
  - **4 個 SubModelDiff を `nn.ModuleList`** で保持、point-specialized partition は T-M3.1 で実装
  - **`NoiseEmbedding` 出力 (B, 512)** を Generator の `cond` 引数で per-block `fc_t` に渡す経路 (T-M1.4 generator が責務)
  - 4 sub-model それぞれが独立パラメータを保持 (重み共有なし)
  - noise level `c = √(1-ᾱ_t)` は呼び出し側で計算 (本チケットでは受け取るだけ)
  - パラメータ総数 ≈ 14.42M × 4 = 57.68M (Table 1 整合)
  - point-specialized 1-to-1 マッピングは T-M3.1 / T-M3.3 で実装 (本チケットでは関知しない)

#### 共通の注意事項
- **channel concat 順序**: `CONCAT_ORDER = [mel (128 ch), stft_spec (2F-2 ch)]` で **確定** (本チケット + T-M1.2 で一貫)。順序変更は破壊的変更となり、学習済み重みのロード時に互換性が壊れる。
- **forward の dtype/device**: `mel.device == y_prev.device` (Diff は `x_t.device` も) を呼び出し側で保証。本チケットでは混在チェックは行わない (PyTorch 標準のエラーに任せる)。
- **B=1 動作**: テスト済 (`test_gan_backward` は B=2 だが、ConvNeXt / Linear は B=1 でも正常動作)。
- **factory パターン**: `SubModelGAN.from_config(cfg, mode="gan")` / `SubModelDiff.from_config(cfg, mode="diff")` を **唯一の生成経路** とする (§8.2 参照)。直接 `__init__` を呼ぶと config drift 耐性が失われる。

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
