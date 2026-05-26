---
id: T-M2.3
title: Loss 関数 (Hinge GAN + FM L1 + MR-STFT)
milestone: M2
phase: M2
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.2]
blocks: [T-M2.5]
related_docs:
  - docs/milestones.md#m23-loss-関数-srcwavenext2losses
  - docs/training.md
  - docs/open-questions.md
---

# T-M2.3: Loss 関数 (Hinge GAN + FM L1 + MR-STFT)

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.3](../milestones.md#m23-loss-関数-srcwavenext2losses)
> **依存**: [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M2.5](T-M2.5-train-gan.md)

## 1. タスク目的とゴール

### 目的
GAN-WaveNeXt 2 の訓練ループで Generator / Discriminator の最適化に用いる 3 種類の loss 関数を `src/wavenext2/losses/` 配下の 3 モジュールに実装する。論文 §3.2 / `docs/training.md` §2.4 / `docs/open-questions.md` §C6 で確定した WaveFit-PT 仕様に厳密に従い、以下を独立にテスト可能な単一クラス群として提供する:

- `HingeGANLoss` (`adversarial.py`): D / G 両方の hinge 形式 GAN loss
- `FeatureMatchingLoss` (`feature_matching.py`): MSD ×3 の中間特徴 L1 距離平均
- `MultiResolutionSTFTLoss` (`stft_loss.py`): 3 解像度 STFT の Spectral Convergence + Magnitude L1

これにより T-M2.5 (train_gan) は 4 つの loss を `compute_total_loss(losses, weights) -> tuple[Tensor, dict[str, float]]` で集約し、total + per-component unweighted scalar を取得して optimizer に渡すだけで済む (weighted/unweighted 両方を TensorBoard ログするために tuple 戻り値とする — §8.1 採用昇格)。Diff 側 (T-M3.2) は MSELoss のみを使うため本チケットを参照しない。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M2.3 Acceptance を内包):
- [ ] `src/wavenext2/losses/adversarial.py` に `HingeGANLoss` クラスが実装され、`from wavenext2.losses.adversarial import HingeGANLoss` で import 可能
- [ ] `src/wavenext2/losses/feature_matching.py` に `FeatureMatchingLoss` クラスが実装され、`from wavenext2.losses.feature_matching import FeatureMatchingLoss` で import 可能
- [ ] `src/wavenext2/losses/stft_loss.py` に `MultiResolutionSTFTLoss` クラスが実装され、`from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import 可能
- [ ] HingeGAN: `D_loss = relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()` / `G_loss = -D(fake).mean()` (`docs/milestones.md` §M2.3 Acceptance #1)
- [ ] FM: 3 sub-D × 7 layer の全中間特徴の L1 距離を `num_features` で平均 (Acceptance #2)
- [ ] MR-STFT: `n_ffts=[512,1024,2048]`, `win_lengths=[360,900,1800]`, `hop_sizes=[80,150,300]` の 3 解像度で SC + Mag L1 を平均、`eps=1e-5` (Acceptance #3)
- [ ] 重み定数を `compute_total_loss(losses: dict, weights: dict) -> tuple[torch.Tensor, dict[str, float]]` ヘルパで集約可能。戻り値は `(total, unweighted_dict)` で、unweighted_dict は per-component scalar (`.item()` 済み) — TensorBoard で weighted/unweighted 両方ログするのが GAN デバッグで必須 (重み: d_gan=1.0, d_fm=10.0, mrstft_sc=2.5, mrstft_mag=2.5 — Acceptance #4)
- [ ] `compute_total_loss` 内で `sorted(losses.keys())` で float 加算順序を固定 (bf16 学習で float 加算順序が再現性を破壊するため M6 で必須)
- [ ] `requires_grad=True` 入力で各 loss が backward 可能、勾配方向の sanity check (`D(real)` を増やすと `D_loss` が減る) が pass
- [ ] 数値範囲: 同入力で 0、異入力で正、NaN / Inf を含まない
- [ ] `tests/test_adversarial.py`, `tests/test_feature_matching.py`, `tests/test_stft_loss.py` の全テスト pass
- [ ] `docs/tickets/index.md` の本チケットステータスが更新済み

## 2. 実装内容の詳細

### 2.1 対象ファイル

- 新規実装 (T-M0.2 で stub commit 済み、本チケットで本実装に置換):
  - `src/wavenext2/losses/adversarial.py` (本実装)
  - `src/wavenext2/losses/feature_matching.py` (本実装)
  - `src/wavenext2/losses/stft_loss.py` (本実装)
  - `tests/test_adversarial.py` (新規実装、または既存 `tests/test_losses.py` を 3 ファイルに分割)
  - `tests/test_feature_matching.py` (新規実装)
  - `tests/test_stft_loss.py` (新規実装)
- 編集:
  - `src/wavenext2/losses/__init__.py` (`__all__` に 3 クラス + `compute_total_loss` ヘルパを追加)
  - `docs/milestones.md` §M2.3 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M2.3 ステータス更新

### 2.2 主要構造

#### import 形式 (T-M0.2 / T-M2.5 で確定)

```python
from wavenext2.losses.adversarial import HingeGANLoss
from wavenext2.losses.feature_matching import FeatureMatchingLoss
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss
from wavenext2.losses import compute_total_loss
```

#### `adversarial.py` — Hinge GAN loss (WaveFit-PT 準拠)

```python
"""adversarial.py — Hinge GAN loss for GAN-WaveNeXt 2.

D loss: relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()
G loss: -D(fake).mean()

参考: docs/training.md §2.4, docs/open-questions.md §C6
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeGANLoss(nn.Module):
    """Hinge formulation of GAN loss (Lim & Ye 2017, Miyato et al. 2018)。

    Discriminator 出力は **scalar logit** (最終 Linear/Conv で活性化なし) を前提とする。
    Discriminator の最終 activation が tanh 等の場合、relu(1 ± D) の `1` が
    意味を持たなくなる (§6.1 critical 項目参照)。
    """

    def __init__(self) -> None:
        super().__init__()

    def d_loss(
        self,
        d_real_list: list[torch.Tensor],
        d_fake_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Discriminator 側 loss.

        Args:
            d_real_list: real 入力に対する各 sub-D の最終 logit のリスト (len=3)
            d_fake_list: fake 入力に対する各 sub-D の最終 logit のリスト (len=3)
        Returns:
            scalar tensor (全 sub-D 合計 or 平均、§6.1 で確定)
        """
        loss = 0.0
        for d_real, d_fake in zip(d_real_list, d_fake_list):
            loss_real = F.relu(1.0 - d_real).mean()
            loss_fake = F.relu(1.0 + d_fake).mean()
            loss = loss + loss_real + loss_fake
        return loss / len(d_real_list)  # sub-D 個数で平均 (§6.1)

    def g_loss(self, d_fake_list: list[torch.Tensor]) -> torch.Tensor:
        """Generator 側 loss: -D(fake).mean() (3 sub-D 平均)."""
        loss = 0.0
        for d_fake in d_fake_list:
            loss = loss - d_fake.mean()
        return loss / len(d_fake_list)
```

#### `feature_matching.py` — Feature matching L1

```python
"""feature_matching.py — Feature matching loss for MSD ×3.

3 sub-D × 7 layer の中間特徴 (T-M2.2 が返すリスト) の L1 距離を取り、
全 feature 数で平均する。

参考: docs/training.md §2.4 (中間特徴 L1)、docs/open-questions.md §C6
"""

from __future__ import annotations
import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    """3 sub-D × N layer の中間特徴 L1 平均."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        feats_real: list[list[torch.Tensor]],
        feats_fake: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            feats_real: 3 sub-D × N layer の特徴リスト (real 入力)
                shape: [[(B, C_l, T_l) for l in layers] for sub_d in sub_ds]
            feats_fake: 同上 (fake 入力)
        Returns:
            scalar tensor: sum(|f_real - f_fake|).mean() / num_features
        """
        num_features = 0
        total = torch.tensor(0.0, device=feats_real[0][0].device)
        for fr_d, ff_d in zip(feats_real, feats_fake):
            for fr_l, ff_l in zip(fr_d, ff_d):
                total = total + (fr_l.detach() - ff_l).abs().mean()
                num_features += 1
        return total / max(num_features, 1)
```

> **注**: 上記の `fr_l.detach()` で real 側勾配を遮断するか否か (`G` 更新時には G の勾配のみ伝播させるため通常 detach する) は §6.1 通常項目で WaveFit-PT 仕様を確認して確定する。

#### `stft_loss.py` — Multi-resolution STFT loss

```python
"""stft_loss.py — Multi-resolution STFT loss (SC + Mag L1).

3 resolution:
    n_ffts      = [512, 1024, 2048]
    win_lengths = [360,  900, 1800]
    hop_sizes   = [ 80,  150,  300]

各 resolution で:
    SC      = ||S_real - S_fake||_F / (||S_real||_F + eps)
    Mag L1  = |log(|S_real| + eps) - log(|S_fake| + eps)|.mean()

最終 loss は 3 resolution の平均で、sc / mag の 2 成分を分けて返す。

参考: docs/training.md §2.4, docs/open-questions.md §C6
"""

from __future__ import annotations
import torch
import torch.nn as nn

DEFAULT_FFTS = (512, 1024, 2048)
DEFAULT_WIN_LENGTHS = (360, 900, 1800)
DEFAULT_HOP_SIZES = (80, 150, 300)
DEFAULT_EPS = 1.0e-5


class _STFTLossSingleResolution(nn.Module):
    """SC + Mag L1 を 1 resolution で計算する基底。"""

    def __init__(self, n_fft: int, win_length: int, hop_size: int, eps: float = DEFAULT_EPS):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_size = hop_size
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → |STFT|: (B, F, T_frames)
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.win_length,
            window=self.window, center=True, return_complex=True,
        )
        return spec.abs()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mag_pred = self._stft_mag(y_pred)
        mag_true = self._stft_mag(y_true)
        # Spectral Convergence (eps で 0 除算防御)
        sc = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + self.eps)
        # Log-magnitude L1
        mag = (torch.log(mag_true + self.eps) - torch.log(mag_pred + self.eps)).abs().mean()
        return sc, mag


class MultiResolutionSTFTLoss(nn.Module):
    """3 resolution の SC + Mag L1 を平均する."""

    def __init__(
        self,
        n_ffts: tuple[int, ...] = DEFAULT_FFTS,
        win_lengths: tuple[int, ...] = DEFAULT_WIN_LENGTHS,
        hop_sizes: tuple[int, ...] = DEFAULT_HOP_SIZES,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()
        if not (len(n_ffts) == len(win_lengths) == len(hop_sizes)):
            raise ValueError("n_ffts / win_lengths / hop_sizes must have same length")
        self.resolutions = nn.ModuleList([
            _STFTLossSingleResolution(n, w, h, eps=eps)
            for n, w, h in zip(n_ffts, win_lengths, hop_sizes)
        ])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: (B, T) waveform pair. Returns: (sc_mean, mag_mean) scalar tensors."""
        sc_sum = 0.0
        mag_sum = 0.0
        for res in self.resolutions:
            sc, mag = res(y_pred, y_true)
            sc_sum = sc_sum + sc
            mag_sum = mag_sum + mag
        n = len(self.resolutions)
        return sc_sum / n, mag_sum / n
```

#### `__init__.py` — `compute_total_loss` ヘルパ

```python
"""wavenext2.losses — GAN-WaveNeXt 2 用 loss モジュール群."""
from __future__ import annotations
import torch

from .adversarial import HingeGANLoss
from .feature_matching import FeatureMatchingLoss
from .stft_loss import MultiResolutionSTFTLoss

__all__ = [
    "HingeGANLoss",
    "FeatureMatchingLoss",
    "MultiResolutionSTFTLoss",
    "compute_total_loss",
    "DEFAULT_WEIGHTS",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "d_gan": 1.0,
    "d_fm": 10.0,
    "mrstft_sc": 2.5,
    "mrstft_mag": 2.5,
}


def compute_total_loss(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of named loss tensors.

    Args:
        losses: {"d_gan": ..., "d_fm": ..., "mrstft_sc": ..., "mrstft_mag": ...}
        weights: None で DEFAULT_WEIGHTS を使用
    Returns:
        (total, losses_unweighted):
            total: scalar tensor = sum(weights[k] * losses[k] for k in sorted(losses))
            losses_unweighted: {k: losses[k].detach().item() for k in sorted(losses)}
                — TensorBoard で weighted (total) と unweighted (per-component) の
                両方をログするための per-component scalar (§8.1 採用昇格)。

    Notes:
        - `sorted(losses.keys())` で float 加算順序を固定する (§8.1 採用昇格)。
          dict iteration 順序は Python 3.7+ で挿入順保証されるが、
          float の加算は非可換のため bf16 学習で再現性を破壊するリスクがあり、
          M6 本格訓練では順序固定が必須。
        - `losses_unweighted` 側は呼び出し側で TensorBoard logger に渡すだけで済むよう
          `.item()` 済み Python float の dict として返す (Tensor のままだと graph 保持
          で memory leak の懸念)。
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    keys = sorted(losses.keys())  # 順序固定 (§8.1 採用昇格)
    if not keys:
        raise ValueError("losses dict is empty")
    total = None
    losses_unweighted: dict[str, float] = {}
    for k in keys:
        v = losses[k]
        if k not in weights:
            raise KeyError(f"loss key {k!r} has no corresponding weight in {list(weights)}")
        losses_unweighted[k] = v.detach().item()
        contrib = weights[k] * v
        total = contrib if total is None else total + contrib
    return total, losses_unweighted
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `n_ffts` (MR-STFT) | `[512, 1024, 2048]` | docs/training.md §2.4 / open-questions.md §C6 |
| `win_lengths` (MR-STFT) | `[360, 900, 1800]` | docs/training.md §2.4 |
| `hop_sizes` (MR-STFT) | `[80, 150, 300]` | docs/training.md §2.4 |
| `eps` (SC 0 除算防御 / log floor) | `1.0e-5` | docs/open-questions.md §C6 |
| Hinge GAN: D loss | `relu(1-D(real)).mean() + relu(1+D(fake)).mean()` | docs/training.md §2.4 / docs/architecture.md §6.2 |
| Hinge GAN: G loss | `-D(fake).mean()` | WaveFit-PT 慣例 |
| FM 計算: 距離 | L1 (`abs().mean()`) | docs/training.md §2.4 |
| FM 計算: 集約 | `sum / num_features` (3 sub-D × N layer) | docs/milestones.md §M2.3 Acceptance #2 |
| 重み `d_gan` | 1.0 | docs/implementation-plan.md §5 / docs/training.md §2.4 |
| 重み `d_fm` | 10.0 | 同上 |
| 重み `mrstft_sc` | 2.5 | 同上 |
| 重み `mrstft_mag` | 2.5 | 同上 |
| 重み `mel_mae` | 0.0 (本チケットでは未使用、key 予約のみ) | docs/implementation-plan.md §5 |

### 2.4 アルゴリズム / 処理フロー

#### `HingeGANLoss.d_loss(d_real_list, d_fake_list)`
1. 各 sub-D `(d_real, d_fake)` ペアに対して:
   - `loss_real = relu(1 - d_real).mean()`
   - `loss_fake = relu(1 + d_fake).mean()`
   - 累積に加算
2. 3 sub-D の合計を `len(d_real_list)` で割って平均

#### `HingeGANLoss.g_loss(d_fake_list)`
1. 各 sub-D `d_fake` について `-d_fake.mean()` を累積
2. `len(d_fake_list)` で平均

#### `FeatureMatchingLoss.forward(feats_real, feats_fake)`
1. `num_features = 0`, `total = 0`
2. 3 sub-D × N layer をループ:
   - `total += (fr.detach() - ff).abs().mean()`
   - `num_features += 1`
3. `return total / num_features`

#### `MultiResolutionSTFTLoss.forward(y_pred, y_true)`
1. 3 resolution それぞれで:
   - `mag_pred = |STFT(y_pred)|`, `mag_true = |STFT(y_true)|` (Hann window, center=True)
   - `sc = ||mag_true - mag_pred||_F / (||mag_true||_F + eps)`
   - `mag = mean(|log(mag_true + eps) - log(mag_pred + eps)|)`
2. 3 resolution 平均で `(sc_mean, mag_mean)` を返す
3. **2 成分を分けて返す**理由: 集約時の重み (`mrstft_sc=2.5`, `mrstft_mag=2.5`) を train_gan で個別に乗算できるため

#### `compute_total_loss(losses, weights) -> tuple[Tensor, dict[str, float]]`
1. `weights` が `None` なら `DEFAULT_WEIGHTS` を使用
2. `keys = sorted(losses.keys())` で **float 加算順序を固定** (§8.1 採用昇格、bf16 再現性)
3. 各 `k in keys` に対して:
   - `losses_unweighted[k] = losses[k].detach().item()` (TensorBoard 用 per-component scalar)
   - `total += weights[k] * losses[k]`
4. `losses` 内に `weights` にない key があれば `KeyError`、空 dict なら `ValueError`
5. `return (total, losses_unweighted)`

### 2.5 設計上の重要決定

- **3 クラス分離 (1 ファイル統合しない)**: `adversarial.py` / `feature_matching.py` / `stft_loss.py` を分けることで `T-M3.2 (train_diff)` が `MSELoss` のみ使う際に他 2 つを import しなくて済む (Diff 側は本チケット不使用)
- **`compute_total_loss` を `__init__.py` に集約**: train_gan で 4 loss を `dict` で渡すだけで weighted sum できる。重みは YAML config から `DEFAULT_WEIGHTS` を上書きする想定 (T-M2.5 へ渡す情報、§9.1 参照)
- **MR-STFT の `(sc, mag)` 2 戻り値**: 1 つの scalar に合算しない理由は §2.4 #3 と同じ (個別重み付け、TensorBoard でも個別ログ)
- **`FeatureMatchingLoss` で `fr.detach()`**: 通常 G 更新時に real 側勾配は不要のため detach。仕様確認後に確定 (§6.1)
- **`HingeGANLoss` は state を持たない pure Module**: `nn.Module` 継承だが parameters は 0 個 (`buffer` も持たない)
- **`MultiResolutionSTFTLoss` の `window` は `register_buffer(persistent=False)`**: device 自動移動するが state_dict に保存しない (再構築可能、checkpoint サイズ削減)
- **`MultiResolutionSTFTLoss` を M2 専用と決めつけない**: M3 でも Diff の post-filter fit (`docs/architecture.md` §6.5) や M4 評価 (`eval_metrics.py`) で再利用する可能性が高い。`losses/stft_loss.py` は GAN 専用ではなく汎用 MR-STFT モジュールとして設計し、M3 / M4 から `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import 可能にする (§9.1 で T-M3.2 / T-M3.4 / T-M4 へ申し送り)
- **MR-STFT eps=1e-5 と T-M1.3 log eps=1e-7 の使い分け根拠**: 両方とも `log(mag + eps)` 形式だが、T-M1.3 は **mel スペクトログラム** (slaney scale / norm で値が小さい、~1e-5 オーダ) に対する log floor、本チケットの MR-STFT は **linear magnitude STFT** (Hann window で値が大きめ、~1e-2〜10 オーダ) に対する log floor。数値範囲に対する eps の感度が違うため意図的に分離。eps を統一すると mel 側で過剰な log floor がかかり dynamic range が崩れる (mel 値 1e-5 オーダで eps=1e-5 だと log(2e-5) ≒ log(1e-5) に潰れる)。逆に MR-STFT 側を eps=1e-7 にすると STFT magnitude が 0 に近い周波数帯で `log(1e-7)` まで log が落ち込み勾配が爆発する。**T-M1.3 と本チケットで eps を統一しない**ことを §8.1 で明示

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | 3 loss モジュール本実装 + 3 個別テスト記述 + `__init__.py` ヘルパ | general-purpose |
| Reviewer | 1 | WaveFit-PT 仕様準拠確認 + 数値範囲・勾配方向 sanity + コピー流用していないか | general-purpose |
| Tester | 1 | `uv run pytest tests/test_adversarial.py tests/test_feature_matching.py tests/test_stft_loss.py -v` 実行 + Acceptance 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M0.2 のみに依存、T-M2.1 / T-M2.2 / T-M2.4 とは独立)
- 並列実行する場合の最大並列数: 3 (loss 3 種を 3 エージェントに分担可能、ただし `__init__.py` / `compute_total_loss` の統合は最後にまとめて 1 人が担当)
- 後続 T-M2.5 とは順次。本チケット完了後に T-M2.5 を起動。

## 4. 提供範囲 (Scope)

### In Scope
- `HingeGANLoss` クラス本実装 (D loss / G loss メソッド)
- `FeatureMatchingLoss` クラス本実装 (3 sub-D × N layer の L1 平均)
- `MultiResolutionSTFTLoss` クラス本実装 (3 resolution の SC + Mag L1 平均)
- `compute_total_loss(losses, weights)` ヘルパ + `DEFAULT_WEIGHTS` 定数
- 3 個別 pytest ファイル (`test_adversarial.py`, `test_feature_matching.py`, `test_stft_loss.py`)
- `__init__.py` の `__all__` 設定
- docstring (英文 + 日本語混在)

### Out of Scope
- Discriminator 本体実装 (T-M2.2)
- Mel-MAE loss (`weights.mel_mae=0.0` で未使用、key 予約のみ。BigVGAN ablation 用に保留)
- LSGAN / NS-GAN / R1 regularizer (§8.1 代替案参照、本チケット未実装)
- LeCAM regularization (§8.1 代替案、本チケット未実装)
- 訓練ループ統合 (T-M2.5)
- Diff 側 MSE loss (T-M3.2 で `torch.nn.functional.mse_loss` 直使用、本チケット範囲外)
- Mixed precision / `torch.compile` 対応 (M5/M6 で必要に応じて検討)
- Vectorized STFT (3 resolution 並列計算) — 単純 for-loop で実装、後で profile が必要なら最適化

### Deliverable
- ファイル:
  - `src/wavenext2/losses/adversarial.py` (本実装)
  - `src/wavenext2/losses/feature_matching.py` (本実装)
  - `src/wavenext2/losses/stft_loss.py` (本実装)
  - `src/wavenext2/losses/__init__.py` (re-export + `compute_total_loss`)
  - `tests/test_adversarial.py`, `tests/test_feature_matching.py`, `tests/test_stft_loss.py` (新規実装)
- 関数 / クラス:
  - `class HingeGANLoss(nn.Module)` with `d_loss` / `g_loss` メソッド
  - `class FeatureMatchingLoss(nn.Module)` with `forward`
  - `class MultiResolutionSTFTLoss(nn.Module)` with `forward → (sc, mag)`
  - `compute_total_loss(losses: dict, weights: dict | None) -> torch.Tensor`
  - `DEFAULT_WEIGHTS: dict[str, float]` 定数
- ドキュメント差分:
  - `docs/milestones.md` §M2.3 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M2.3 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト

#### `tests/test_adversarial.py`
- [ ] `test_d_loss_formula`: 手計算と一致 (`D(real)=[2.0]`, `D(fake)=[-2.0]` → `relu(1-2)+relu(1-2) = 0`、`D(real)=[0.5]`, `D(fake)=[0.5]` → `relu(0.5)+relu(1.5) = 2.0`)
- [ ] `test_g_loss_formula`: `D(fake)=[3.0]` → `g_loss = -3.0`
- [ ] `test_d_loss_gradient_direction`: `D(real)` を `requires_grad=True` で増やすと `d_loss.backward()` 後の grad が **負方向** に向き、勾配降下で `d_loss` が減少することを 1 step optimizer で確認
- [ ] `test_g_loss_gradient_direction`: 同様に `D(fake)` を上げると `g_loss` が下がる方向
- [ ] `test_multi_sub_d`: 3 sub-D 入力 `[t1, t2, t3]` で出力が `(loss(t1)+loss(t2)+loss(t3))/3` に一致
- [ ] `test_numeric_range`: ランダム入力で NaN / Inf を含まない、loss が `[0, +inf)` (D-loss)
- [ ] `test_no_parameters`: `HingeGANLoss` に `parameters()` がゼロ
- [ ] `test_deterministic`: 同入力で 2 回呼び bit-exact 一致

#### `tests/test_feature_matching.py`
- [ ] `test_same_features_zero`: `feats_real == feats_fake` で loss = 0
- [ ] `test_diff_features_positive`: ランダム異入力で loss > 0
- [ ] `test_num_features_count`: 3 sub-D × 7 layer = **18 個** の場合の `num_features` 確認 (T-M2.2 から受領: `DiscriminatorOutput.features` は logits を含まない、§6.3 参照)
- [ ] `test_l1_formula`: 単純 case (1 sub-D, 1 layer, scalar tensor) で `abs(a-b).mean()` と一致
- [ ] `test_backward`: `requires_grad=True` 入力で backward 動作、`fake` 側に grad が流れる (`fr.detach()` で real 側に grad が **流れない**)
- [ ] `test_shape_mismatch`: `feats_real` と `feats_fake` の長さ不一致で適切なエラー (silent failure なし)

#### `tests/test_stft_loss.py`
- [ ] `test_same_audio_zero`: `y_pred == y_true` で `(sc, mag) ≈ (0, 0)` (eps による微小残差は許容)
- [ ] `test_diff_audio_positive`: ランダム異入力で `(sc, mag) > 0`
- [ ] `test_three_resolutions_average`: 1 resolution のみ手計算した結果と `MultiResolutionSTFTLoss` の平均が `(r1+r2+r3)/3` と一致
- [ ] `test_sc_formula`: 単純 sine 波で `sc = ||mag_true - mag_pred||_F / (||mag_true||_F + eps)` の値が計算式通り
- [ ] `test_mag_formula`: `log(mag+eps)` の L1 が計算式通り
- [ ] `test_eps_no_div_zero`: `y_true = zeros_like(...)` でも NaN / Inf を出さない (eps が effective)
- [ ] `test_backward`: backward 動作、`y_pred.grad` が non-None
- [ ] `test_window_buffer_device`: `model.to("cuda")` で `window` も CUDA に移動 (`register_buffer` 確認)
- [ ] `test_persistent_false`: `model.state_dict()` に `window` が含まれない (`persistent=False` 確認)
- [ ] `test_window_dtype_aware`: `model(x.bfloat16(), y.bfloat16())` で window が bf16 に再生成され runtime error が出ないことを確認 (§6.1 「resume 時 dtype 不整合」対策、`if self.window.dtype != x.dtype` 分岐の検証)

#### 共通 / `compute_total_loss`
- [ ] `tests/test_compute_total_loss::test_weighted_sum`: 既知の `losses` / `weights` で手計算と一致
- [ ] `tests/test_compute_total_loss::test_default_weights`: `weights=None` で `DEFAULT_WEIGHTS` 適用
- [ ] `tests/test_compute_total_loss::test_unknown_key_error`: `losses` に未登録 key で `KeyError`
- [ ] `tests/test_compute_total_loss::test_empty_dict_error`: 空 dict で `ValueError`
- [ ] `tests/test_compute_total_loss::test_return_tuple`: 戻り値が tuple `(total: Tensor, unweighted_dict: dict[str, float])` の 2 要素であることを `isinstance` で確認 (§8.1 採用昇格)
- [ ] `tests/test_compute_total_loss::test_unweighted_dict_contents`: `unweighted_dict[k] == losses[k].item()` で **un-weighted scalar** が返ることを確認、`unweighted_dict["d_fm"]` が `losses["d_fm"].item()` と一致 (重み 10.0 を掛けていない)
- [ ] `tests/test_compute_total_loss::test_unweighted_dict_is_python_float`: `isinstance(unweighted_dict[k], float)` で Tensor ではなく Python float が返ることを確認 (graph 保持による memory leak 防止)
- [ ] `tests/test_compute_total_loss::test_order_independence`: 同じ `losses` dict を異なる insertion 順序 (`{"a":..., "b":...}` vs `{"b":..., "a":...}`) で渡しても `total` が **bit-exact 一致**することを確認 (`sorted(losses.keys())` による順序固定検証、§8.1 採用昇格)

### 5.2 e2e / 結合テスト

- [ ] T-M2.2 (Discriminator) が完成後に統合できる前提:
  - **本チケット完了時点では Discriminator は未完成のため**、`feats_real` / `feats_fake` / `d_real_list` / `d_fake_list` は **ダミー tensor (Mock)** で構成し API 互換性のみ担保
  - 統合テスト本番は T-M2.5 (train_gan) で `D(y_0)` の戻り値を直接渡して動作確認
- [ ] T-M2.5 (train_gan) の `compute_total_loss({"d_gan":..., "d_fm":..., "mrstft_sc":..., "mrstft_mag":...}, DEFAULT_WEIGHTS)` が動作することの smoke test (本チケットでは未実装でも import + 関数呼び出しだけ確認)

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.3 より転記)
- [ ] hinge GAN: `D_loss = relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()`
- [ ] FM: 3 sub-D × 7 layer の全中間特徴の L1 平均
- [ ] MR-STFT: 3 resolution `[512,1024,2048]` × `[360,900,1800]` × `[80,150,300]`、各 SC + Mag L1
- [ ] 重み: D-GAN=1.0, D-FM=10.0, MRSTFT-SC=2.5, MRSTFT-Mag=2.5

### 5.4 追加 acceptance (本チケット独自)
- [ ] `from wavenext2.losses import HingeGANLoss, FeatureMatchingLoss, MultiResolutionSTFTLoss, compute_total_loss, DEFAULT_WEIGHTS` が動作
- [ ] `pytest tests/test_adversarial.py tests/test_feature_matching.py tests/test_stft_loss.py` が **exit code 0** で完了
- [ ] 3 ファイル全体のテスト合計実行時間が CPU で **30 秒以内**
- [ ] 3 loss クラスを `model.to("cuda")` / `.to("cpu")` で device 移動して数値結果が device 非依存 (相対誤差 1e-5 以内)

### 5.5 テスト戦略

- **CPU 中心**: 全テスト CPU で pass、CUDA は `pytest.mark.gpu` で別 marker (M1 phase review に整合)
- **Fixture 共有**: `tests/conftest.py` に `random_audio_pair` (B=2, T=24000) fixture を追加し 3 ファイルで共有
- **数値安定性テスト**: `torch.set_default_dtype(torch.float64)` に切り替えて 1 ケース実行し、float32 / float64 の差が 1e-4 以内であることを確認 (eps による安定性検証)
- **coverage 目標**: 本チケットのカバレッジ **90%** (3 loss は単純構造で error path も少ない)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (実装着手前に必ず確認)

- **CRITICAL: Hinge loss の `relu(1 ± D)` の `1` は Discriminator 出力 scale に依存**:
  - Discriminator の最終層が `tanh` の場合、`D ∈ [-1, 1]` となり `relu(1 - D)` が常に `≥ 0` で勾配が消失する範囲が広がる
  - `Linear` (活性化なし) であれば `D ∈ ℝ` で hinge の margin `1` が意味を持つ
  - **本チケット実装着手前の MUST DO**:
    1. T-M2.2 (Discriminator) の最終層が `Linear` / `Conv1d` (活性化なし) であることを確認
    2. WaveFit-PT (`yukara-ikemiya/wavefit-pytorch/src/model/discriminator.py`) の最終層が weight_norm 付き Conv1d で活性化なしであることを確認
    3. 確認結果を T-M2.2 §9.1 に逆連絡 (本チケット → T-M2.2 への要求事項として明示)
  - **本確認が未解決のままだと hinge loss が無効化される (silent failure)**

- **RESOLVED (T-M2.2 phase review): FM の `num_features` = 18 個 (3 sub-D × 7 layer、logits 除外)**:
  - T-M2.2 から受領 (`DiscriminatorOutput(logits, features)` NamedTuple): `features` には最終 logit を **含まない**
  - 確定値: `num_features = 3 sub-D × 7 layer = 18` で `test_num_features_count` を pin
  - WaveFit-PT / HiFi-GAN 慣例と一致 (最終 logit 層は FM 対象外、`feature_loss` 関数で除外)

#### 通常項目

- **MR-STFT の SC で 0 除算リスク**: `||mag_true||_F = 0` (全 0 入力時) で発生。`+ eps` で防御済みだが、`test_eps_no_div_zero` で明示検証
- **MR-STFT の `torch.stft` window device**: `register_buffer("window", ..., persistent=False)` で `model.to(device)` で自動移動する設計。`register_buffer` ではなく `.to()` で個別移動する誤実装に注意
- **MR-STFT の n_fft=2048 + signal 短い時の挙動**: signal 長 < n_fft の場合 `torch.stft` がデフォルトで center pad するが念のため `test_short_signal` で確認 (segment_length=16384 なら問題なし)
- **`FeatureMatchingLoss` の `.detach()` 妥当性**: G 更新時に D の中間特徴に対する勾配を遮断するために `feats_real` 側は通常 detach する。WaveFit-PT 仕様確認後に確定 (§6.1 critical 項目 #2 と関連)
- **重み (`d_fm=10.0`) と hinge G (`d_gan=1.0`) のスケール差 — FM gradient dominant リスク (強制要請)**: D-FM だけが他より 1 桁大きい (WaveFit / HiFi-GAN 慣例)。`compute_total_loss` で合算した時に **FM gradient が dominant** になり hinge G が学習しないリスクがある。これを早期検知するため、本チケットでは `compute_total_loss` の戻り値を `(total, losses_unweighted)` tuple とし、T-M2.5 が `losses_unweighted` を TensorBoard に **別途 log することを強制要請** とする (推奨ではなく、§9.1 で T-M2.5 への **インターフェース要請**)。これにより各 component の unweighted scalar と weighted contribution (`weights[k] * unweighted[k]`) を train ループ側で個別にプロットでき、`g_fm` が `g_gan` を一桁以上上回る現象を即座に検知可能
- **`compute_total_loss` の集約順序 (採用昇格)**: dict iteration 順序は Python 3.7+ で挿入順保証されるが、float の加算は非可換のため bf16 学習で再現性を破壊するリスクがあり、M6 本格訓練 (A100, bf16) では必須化。本チケットで `sorted(losses.keys())` による順序固定を **採用昇格** (§8.1 採用設計)、テストで bit-exact 確認 (§5.1 `test_order_independence`)
- **`MultiResolutionSTFTLoss.window` の `persistent=False` resume 時 dtype 不整合**: fp32 で学習した checkpoint を bf16 で resume する場合、`persistent=False` のため window が state_dict に保存されず、`__init__` 時の dtype (デフォルト fp32) で再生成される。`y_pred` が bf16 で `window` が fp32 だと `torch.stft` が **silent に型 promote**するか、**runtime error** になる (PyTorch version 依存)。**対策**: `_STFTLossSingleResolution.forward` の冒頭で dtype-aware に window を再生成する:
  ```python
  if self.window.dtype != x.dtype:
      self.window = torch.hann_window(self.win_length, dtype=x.dtype, device=x.device)
  ```
  (`register_buffer` は dtype 固定なので、再代入は `self.window = ...` で OK。ただし register_buffer 経由のため `nn.Module.__setattr__` が適切に処理する。テストで `model(x.bfloat16())` → window が bf16 化することを確認)
- **`HingeGANLoss` の sub-D 平均 vs 合計**: WaveFit-PT は **平均** (`/ len(d_real_list)`)、HiFi-GAN は **合計**。本チケットでは **平均** を採用 (WaveFit-PT 準拠)。代替候補として `reduce: Literal["mean", "sum"]` 引数化を §8.1 で議論
- **`MultiResolutionSTFTLoss` の SC 分母**: `||mag_true||_F + eps` (本チケット採用) vs `(||mag_true||_F + eps)^2` などの実装バリアントあり。WaveFit-PT の `src/loss/mrstft.py` の **正確な式** を確認して pin
- **`MR-STFT` の Mag 計算: log vs linear**: `docs/training.md` §2.4 で「log-amplitude」と明記されているため `log(mag + eps)` を採用。ParallelWaveGAN は linear、Vocos は log と異なるので注意
- **B (batch dim) と T (time dim) の autodetect**: `torch.stft` の入力は `(B, T)` または `(T,)`。本チケットでは `(B, T)` 前提、`(T,)` は呼び出し側で unsqueeze する責務
- **dtype**: 全 loss は float32 前提。AMP (fp16) は T-M2.5 で `GradScaler` を介して扱う (本チケットでは float32 のみ)

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み** (§C6 で完全確定)
  - Hinge GAN loss 採用、LSGAN 不採用 → §C6
  - MR-STFT 3 resolution の値、SC + Mag L1 形式、eps=1e-5 → §C6
  - 重み D-GAN=1.0, D-FM=10.0, MRSTFT-SC=2.5, MRSTFT-Mag=2.5 → §C6 / implementation-plan.md §5
- 本チケットで確定する曖昧さ:
  - **G loss の sign convention**: `-D(fake).mean()` (本チケット採用、§2.2 docstring に明記) vs `relu(1 - D(fake)).mean()` (Geometric GAN 形式)。WaveFit-PT 準拠で前者
  - **FM の `.detach()` 位置**: real 側 detach (本チケット採用) で確定 (§6.1 critical で WaveFit-PT 仕様確認後)

### 6.3 他チケットとの整合性

- **T-M2.2 (Discriminator)** との整合:
  - 期待 signature: `discriminator(audio) -> DiscriminatorOutput(logits, features)` (T-M2.2 で確定済 NamedTuple、`docs/tickets/T-M2.2-discriminator.md` 参照)
  - `logits` は 3 sub-D の最終 logit リスト、`features` は 3 sub-D × N layer の中間特徴リスト
  - **T-M2.2 から受領 (確定事項)**: `num_features` は **logits を含まない** (= 各 sub-D 7 layer × 3 sub-D = **18 個**)。本チケットの `test_num_features_count` を **18** で pin (§6.1 critical #2 は T-M2.2 側で解決済み)
  - **本チケットからの逆要求 (残)**: 最終層は活性化なし (`Linear` / `Conv1d`) であること (§6.1 critical #1)
  - 不整合があった場合は T-M2.2 側を修正
- **T-M2.4 (GANWaveNext2)** との整合:
  - 本チケットは fake 側 generator 出力 `y_0` を取り、Discriminator に通した結果を loss に渡す経路は T-M2.4 / T-M2.5 が責務
  - `y_0.shape == (B, T_audio)` で `T_audio = T_mel * hop_length` (= 16384 for segment training)
- **T-M2.5 (train_gan)** へ渡す情報:
  - `from wavenext2.losses import HingeGANLoss, FeatureMatchingLoss, MultiResolutionSTFTLoss, compute_total_loss, DEFAULT_WEIGHTS` でまとめて import 可能
  - 訓練ループ内での使い分け:
    ```python
    d_outputs_real, feats_real = D(x_gt)
    d_outputs_fake, feats_fake = D(y_0.detach())
    d_loss = hinge.d_loss(d_outputs_real, d_outputs_fake)

    # G 更新
    d_outputs_fake_for_g, feats_fake_for_g = D(y_0)
    g_gan  = hinge.g_loss(d_outputs_fake_for_g)
    g_fm   = fm(feats_real, feats_fake_for_g)
    sc, mag = mrstft(y_0, x_gt)
    g_total, losses_unweighted = compute_total_loss(
        {"d_gan": g_gan, "d_fm": g_fm, "mrstft_sc": sc, "mrstft_mag": mag},
        weights=cfg.loss.weights,
    )
    # losses_unweighted は T-M2.5 で TensorBoard に別途 log すること (強制要請、§9.1)
    ```
- **T-M3.2 (train_diff)** との整合:
  - **本チケットは Diff 側で不使用**。T-M3.2 は `torch.nn.functional.mse_loss(eps_pred, eps_gt)` のみ
  - 共有 loss は **なし**、`compute_total_loss` も Diff 側は使わない (単一 loss なら直接最適化)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §2.4 / `docs/open-questions.md` §C6 の仕様と完全整合
- [ ] 5.1 Unit テスト全 pass、5.3 Acceptance 全クリア
- [ ] Hinge loss の `1` margin が D 出力 scale (Linear / Conv 出力、活性化なし) と整合
- [ ] FM の `num_features` が WaveFit-PT 仕様 (最終層含む/除く) と一致 (§6.1 critical で確定)
- [ ] MR-STFT の SC 分母が `||S_real||_F + eps` で 0 除算防御
- [ ] MR-STFT の Mag が `log(mag + eps)` の L1 (linear ではない)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 不正入力 (空 list, shape mismatch, dict 内未登録 key) で `ValueError` / `KeyError`
- [ ] `register_buffer("window", ..., persistent=False)` で device 自動移動 + state_dict 除外
- [ ] **参考実装 (WaveFit-PT / HiFi-GAN / ParallelWaveGAN) をコピーしていない**: docstring / 構造を独自に書き起こしているか、変数名や順序が明らかに参考実装をなぞっていないかをスポットチェック
- [ ] `__init__.py` の `__all__` に 3 クラス + `compute_total_loss` + `DEFAULT_WEIGHTS` が含まれる
- [ ] テストが `tests/test_*.py` 命名規約に従う
- [ ] CPU でテストが pass する (CUDA を要求しない)
- [ ] 各 loss が `nn.Module` を継承して `model.to(device)` で正しく移動

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M2) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計

- **3 ファイル分割 (`adversarial.py` / `feature_matching.py` / `stft_loss.py`)**:
  - 理由 (3 観点):
    - (a) Diff 側 (T-M3.2) が MSE のみ使うため、loss 間の独立 import を可能にする
    - (b) WaveFit-PT のディレクトリ構造 (`src/loss/{mrstft,adversarial,feature_matching}.py`) と整合
    - (c) 各 loss が独立してテスト可能 (テストファイルも分割)

- **`compute_total_loss` を `__init__.py` に集約**:
  - 理由: 重み付き合算は train ループ全体に関わるため、loss モジュール群の **entry point** として `__init__.py` に置くのが直感的。`DEFAULT_WEIGHTS` も併置することで YAML config との対応関係を 1 箇所で管理

- **MR-STFT が `(sc, mag)` の 2 戻り値**:
  - 理由: train 側で重みを別々に乗じる (`mrstft_sc=2.5`, `mrstft_mag=2.5`)、TensorBoard に別系列でログする

- **`compute_total_loss(losses, weights) -> tuple[Tensor, dict[str, float]]` (採用昇格)**:
  - 旧案: `compute_total_loss(losses, weights) -> Tensor` 単一戻り値
  - **採用昇格根拠**: GAN デバッグでは weighted total と unweighted per-component の **両方** を TensorBoard ログするのが必須 (D-FM が dominant になっていないか・hinge G が学習しているか・MR-STFT が train loss を引っ張っていないかを別系列で確認するため)。M2 phase review で、T-M2.5 が `losses_unweighted` を別途集めて TensorBoard log するインターフェースを強制要請とする決定に伴い、`compute_total_loss` 側で per-component scalar dict を返す責務を負う設計に昇格
  - per-component は `.item()` 済み Python float で返す: Tensor のまま返すと train ループ側で参照保持されて backward graph が解放されない memory leak の懸念

- **`compute_total_loss` 内で `sorted(losses.keys())` で float 加算順序を固定 (採用昇格)**:
  - 旧案: dict iteration 順序 (Python 3.7+ 挿入順保証) に任せる
  - **採用昇格根拠**: M6 本格訓練は A100 bf16 mixed precision で行うが、bf16 は仮数部 7-bit のため float の加算順序が異なると `total` の bit 値が変わり、checkpoint resume / multi-seed 比較で再現性が破壊される。`sorted(losses.keys())` で alphabetical 順 (= `d_fm, d_gan, mrstft_mag, mrstft_sc`) に固定すれば dict insertion 順に依存せず bit-exact 一致が保証される。テストで bit-exact 確認 (§5.1 `test_order_independence`)

#### Deprecated (旧案、却下根拠)

1. **`AdversarialLoss(loss_type="hinge"|"lsgan"|"non_saturating")` の汎用クラス**:
   - メリット: 後で LSGAN ablation する時に切り替え容易
   - 却下: M2 段階では hinge 1 種で十分、汎用化は YAGNI。LSGAN ablation は M6.3 で必要になったら `LSGANLoss` を別クラス追加する方が責任分離

2. **`feature_matching` を `adversarial.py` に統合**:
   - メリット: FM は GAN loss の一部とも言える
   - 却下: FM は D 出力 (`d_outputs`) ではなく中間特徴 (`features`) を扱うため、入力構造が異なる。分離した方が test も import も簡潔

3. **`MultiResolutionSTFTLoss` の戻り値を `total = sc + mag` scalar 化**:
   - メリット: train 側で `loss = mrstft(y_pred, y_true)` 1 行で済む
   - 却下: 重み (`mrstft_sc=2.5`, `mrstft_mag=2.5`) を train 側で個別調整する設計 (`docs/implementation-plan.md` §5) と齟齬。本チケットは tuple 戻り値で柔軟性を確保

4. **STFT 計算を `torchaudio.transforms.Spectrogram` で実装**:
   - メリット: window / FFT は最適化済み
   - 却下: `torchaudio.transforms.Spectrogram` は `power=2.0` がデフォルトで magnitude 取得に追加 `.pow(0.5)` が必要。`torch.stft` 直呼びの方が透明性が高い

5. **`HingeGANLoss` を関数 (関数型) として実装**:
   - メリット: state を持たないので `nn.Module` 不要
   - 却下: 他 loss (`FeatureMatchingLoss`, `MultiResolutionSTFTLoss`) は `nn.Module` (buffer / Module 内包) のため、一貫性のため `HingeGANLoss` も `nn.Module` で統一

#### 追加検討した代替設計 (本チケット採用 / 検討)

6. **LSGAN (最小二乗) への切替**:
   - **メリット**: 学習安定性が向上する報告あり、HiFi-GAN は LSGAN 採用
   - **却下根拠**: `docs/open-questions.md` §C6 で **hinge と明記**、WaveFit-PT 準拠が論文整合性の最優先
   - **再評価トリガー**: M6.3 ablation で hinge が divergent → LSGAN 切替を検討

7. **NS-GAN (非飽和) への切替**:
   - **メリット**: G loss の勾配が D output 全域で安定
   - **却下根拠**: hinge は WaveFit-PT 仕様で確定、NS は元来 DCGAN 系の慣例
   - **再評価トリガー**: M6.3 ablation

8. **R1 / R2 regularizer (D の安定化)**:
   - **メリット**: D の Lipschitz 制御で学習安定
   - **却下根拠**: WaveFit-PT は不採用、本論文も言及なし
   - **再評価トリガー**: M6.3 ablation で D loss が 0 に張り付く現象が出たら導入検討

9. **LeCAM regularization 追加**:
   - **メリット**: D の prediction smoothing で mode collapse 防止
   - **却下根拠**: WaveFit-PT 不採用、論文記述なし
   - **再評価トリガー**: M6.3 ablation

10. **FM を cosine similarity ベース化**:
    - **メリット**: scale-invariant、L1 より robust
    - **却下根拠**: WaveFit-PT は L1、論文も `docs/training.md` §2.4 で「L1 distance」明記
    - **再評価トリガー**: M6.3 ablation

11. **MR-STFT を Mel L1 (BigVGAN 流) や STFT L1 のみに変更**:
    - **メリット**: 計算コスト削減 (1 resolution のみ)
    - **却下根拠**: WaveFit-PT は 3 resolution の SC + Mag L1 で確定 (§C6)
    - **再評価トリガー**: M6.3 ablation で MR-STFT が遅すぎる場合

12. **`compute_total_loss` を `WeightedSumModule(nn.Module)` 化**:
    - **メリット**: 重みを `nn.Parameter` 化して学習可能にできる
    - **却下根拠**: M2 段階では固定重みで十分、過剰な抽象化
    - **再評価トリガー**: M6.3 で重み調整 ablation が必要なら検討

13. **MR-STFT eps を T-M1.3 (mel log floor) と統一する**:
    - **メリット**: eps が 1 箇所で管理できて見通しが良い、`from wavenext2.constants import LOG_EPS` で一元化可能
    - **却下根拠**: 本チケットの MR-STFT eps=1e-5 と T-M1.3 mel log eps=1e-7 は **数値範囲に対する eps の感度が違うため意図的に分離** している。
      - **T-M1.3 (mel)**: slaney scale + norm により mel 値は ~1e-5 オーダ。eps=1e-7 は値の 1% 以下で dynamic range を潰さない。
      - **本チケット (MR-STFT)**: linear magnitude STFT (Hann window) は ~1e-2〜10 オーダ。eps=1e-5 は値の 0.1% 以下で同様に dynamic range を保持。
      - eps を 1e-7 に統一すると MR-STFT 側で magnitude が 0 に近い周波数帯 (高周波や無音区間) で `log(1e-7) ≈ -16.1` まで log が落ち込み、勾配が爆発する。
      - 逆に eps を 1e-5 に統一すると mel 側で `log(2e-5) ≈ log(1e-5)` に潰れ dynamic range が崩れる (mel 値 ~1e-5 オーダで eps=1e-5 だと log floor が信号と同オーダになる)。
    - **再評価トリガー**: M1.3 phase review で mel 値のオーダが変わった (slaney → htk 等の scale 変更) 場合、本チケット eps も再検討

#### 再評価トリガー条件

| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| 3 ファイル分割 (`adversarial.py` / `fm.py` / `stft_loss.py`) | M2 phase review | T-M2.5 統合時に Loss クラス間で共通 helper が出てきたら `_common.py` 追加 |
| `compute_total_loss` の `__init__.py` 配置 | M2.5 / M3.2 実装時 | Diff 側で別 helper を作る場合は `losses/total.py` に切り出し |
| Hinge GAN vs LSGAN | M6.3 ablation | hinge が学習発散 / mode collapse → LSGAN 切替 |
| FM の L1 vs cosine | M6.3 ablation | L1 が dominant でなくなる場合 cosine 化 |
| MR-STFT の `(sc, mag)` 戻り値 | M2.5 実装時 | TensorBoard ログ要件が変わったら struct 化 |
| `register_buffer("window", persistent=False)` | M5 phase review | checkpoint resume で window 不整合が出たら `persistent=True` 化 |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: **適切**
  - 3 loss は構造的に独立で、それぞれが論文の異なるパラグラフ (`docs/training.md` §2.4 の 4 種 loss 重み) に対応する
  - 1 チケットに統合すると test ファイルも 1 つに肥大化し、並列実装 (3 エージェント分担) が困難
  - 逆に M2.3a / M2.3b / M2.3c のような細分化は子チケット数が増えすぎ、依存グラフが複雑になる
- **別マイルストーンに移すべき部分はないか**: **なし**
  - 3 loss はすべて GAN 訓練 (T-M2.5) に必要で、M2 phase 内に閉じている
  - Diff 側 MSE は 1 行で済むので別チケット化不要 (T-M3.2 で直書き)
- **インターフェース定義の見直し余地**:
  - **`HingeGANLoss` の `d_loss` / `g_loss` を `forward` 統一する案**: `forward(mode="d", ...)` で分岐 → **却下** (TorchScript 互換性低下、明示的メソッド名の方が train 側で読みやすい)
  - **`compute_total_loss` を class 化 (`TotalLoss(nn.Module)`)**: §8.1 #12 で検討、現状は関数で十分
  - **`DEFAULT_WEIGHTS` を pydantic / dataclass 化**: type safety 向上、ただし dict も十分。YAML から `**weights_dict` で展開可能なため現状維持

#### M2 phase review で追加されうる設計原則 (M2 phase 完了時に再評価)

- **factory パターン全モジュール一貫化 (T-M1.6 §8.2 で提唱) に loss も合流するか**:
  - 現状: 3 loss クラスは `__init__` で直接生成
  - 案: `HingeGANLoss.from_config(cfg)` / `MultiResolutionSTFTLoss.from_config(cfg)` を追加
  - 採否: **保留** (M2.5 train_gan 実装時に YAML config 経由生成の必要性を再評価)
- **`Protocol` 経由の loss interface 統一**:
  - 案: `class LossLike(Protocol): def forward(self, *args) -> torch.Tensor: ...`
  - 採否: **保留** (M2.5 / M3.2 で実際に共通インターフェースが必要になったら導入)

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `HingeGANLoss` (d_loss/g_loss、sub-D 平均)、`FeatureMatchingLoss` (real 側 detach)、`MultiResolutionSTFTLoss` (SC + log-mag L1、3 解像度)、`compute_total_loss` ((total, unweighted dict)、sorted 加算順、DEFAULT_WEIGHTS) を実装。`tests/test_losses.py` 15 件 pass。
- チケット §2.2 のドラフトがほぼ正確だったため忠実に実装 (d_loss at 0 → 2.0、FM identical → 0、MR-STFT identical → ~0 を検証)。

実装上の判断:
1. **FM loss は real 側を detach** (HiFi-GAN/WaveFit 標準: real は target)。`test_fm_detaches_real_side` で real.grad is None / fake.grad not None を検証。
2. **loss の初期化に `tensor.new_zeros(())`** を使い、device/dtype を入力に追従させた (float `0.0` 開始だと CPU scalar 混入の懸念)。
3. **`compute_total_loss` は `sorted(losses)` で加算順固定** (bf16 の float 加算非可換による M6 再現性破壊を防止)。unweighted は `.item()` 済み float (graph 非保持)。
4. discriminator の `SubDiscOutput` から loss へは呼び出し側 (T-M2.5) が `[o.logits ...]` / `[o.features ...]` を抽出して渡す (loss API は list 受けで D 構造非依存)。

次の似たタスクで応用できる教訓:
- 勾配方向の sanity (D(real)↑ → d_loss↓ つまり ∂loss/∂d_real<0) は loss の正しさを値非依存で検証でき強力。
- 集約ヘルパは加算順を明示固定しておくと後の bf16/分散学習で再現性問題を未然に防げる。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.2 (Discriminator) へ (逆方向の要求)
- **最終層は活性化なし (`Linear` or `Conv1d`)** が必須 (本チケット §6.1 critical #1)
- **`features` リストの最終層含む / 除く** 仕様を WaveFit-PT 準拠で確定 (本チケット §6.1 critical #2)。本チケットの `test_num_features_count` が確定値で pin される
- Discriminator の戻り値 signature: `(d_outputs: list[Tensor], features: list[list[Tensor]])` (3 sub-D × N layer)
- AvgPool1d で隣接 sub-D 間 downsample (`kernel=4, stride=2`)、WaveFit-PT 完全準拠

#### T-M2.5 (train_gan) へ
- **import 経路** (§2.2 確定):
  ```python
  from wavenext2.losses import (
      HingeGANLoss, FeatureMatchingLoss, MultiResolutionSTFTLoss,
      compute_total_loss, DEFAULT_WEIGHTS,
  )
  ```
- **訓練ループ統合例** (§6.3 参照):
  ```python
  hinge = HingeGANLoss()
  fm = FeatureMatchingLoss()
  mrstft = MultiResolutionSTFTLoss(eps=1e-5).to(device)

  # D 更新
  d_real, feats_real = D(x_gt)
  d_fake, feats_fake = D(y_0.detach())
  loss_d = hinge.d_loss(d_real, d_fake)
  loss_d.backward(); opt_D.step()

  # G 更新
  d_fake_g, feats_fake_g = D(y_0)
  g_gan = hinge.g_loss(d_fake_g)
  g_fm  = fm(feats_real, feats_fake_g)
  sc, mag = mrstft(y_0, x_gt)
  losses = {"d_gan": g_gan, "d_fm": g_fm, "mrstft_sc": sc, "mrstft_mag": mag}
  loss_g = compute_total_loss(losses, weights=cfg.loss.weights or DEFAULT_WEIGHTS)
  loss_g.backward(); opt_G.step()
  ```
- **重要事項 (T-M2.5 への強制要請 / インターフェース要件)**:
  - **`compute_total_loss` 戻り値は tuple `(total, losses_unweighted)`** (§8.1 採用昇格): 旧 `Tensor` 単一戻り値ではないことに注意。受け取り側は必ず 2 値で unpack すること:
    ```python
    g_total, losses_unweighted = compute_total_loss(losses, weights)
    ```
    1 値で受けると `tuple` が `g_total` に代入されて `g_total.backward()` が失敗する。
  - **`losses_unweighted` を TensorBoard で別途 log する (強制要請、推奨ではない)**: D-FM (weight=10.0) と hinge G (weight=1.0) の **スケール差** により FM gradient が dominant になり hinge G が学習しないリスクがある (§6.1 「FM weight=10.0 と hinge G weight=1.0 のスケール差」)。T-M2.5 は以下を必ず log:
    ```python
    # weighted total
    writer.add_scalar("loss/g_total", g_total.item(), step)
    # unweighted per-component (FM dominant 検知のため)
    for k, v in losses_unweighted.items():
        writer.add_scalar(f"loss/unweighted/{k}", v, step)
    # weighted contribution (バランス確認のため)
    for k, v in losses_unweighted.items():
        writer.add_scalar(f"loss/weighted/{k}", weights[k] * v, step)
    ```
    本要請は M2.5 ticket の Acceptance に明記すること。
  - **`DEFAULT_WEIGHTS`** で `mel_mae` key は **未登録**。M6.3 ablation で Mel-MAE 追加するなら本チケットの `DEFAULT_WEIGHTS` を拡張
  - **`losses` dict iteration 順序は `compute_total_loss` 内で `sorted` 固定済** (§8.1 採用昇格): T-M2.5 側で順序を意識する必要なし。M6 bf16 学習でも bit-exact 再現性が保証される
  - **`mrstft.to(device)` 必須**: `register_buffer("window")` は `to()` で移動するため、optimizer 起動前に device 移動を済ませる。さらに **fp32 checkpoint → bf16 resume 時の dtype 不整合** に注意 (§6.1 通常項目)、本チケット側で dtype-aware window 再生成を実装済

#### T-M3.2 (train_diff) へ
- **訓練の主 loss としては使わない**。Diff 側の主 loss は `torch.nn.functional.mse_loss(eps_pred, eps_gt)` のみ
- **ただし `MultiResolutionSTFTLoss` は post-filter fit / validation metric として再利用可能**:
  - `docs/architecture.md` §6.5 の post-filter fitting で MR-STFT を eval metric として使う場合は本チケットの `MultiResolutionSTFTLoss` を再利用
  - `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import (`losses/__init__.py` 経由でも可)
  - 本チケットでは `losses/stft_loss.py` を **GAN 専用と決めつけない汎用 MR-STFT モジュール** として設計済 (§2.5)
- `HingeGANLoss` / `FeatureMatchingLoss` は Diff 側で使わないため誤って import しないこと

#### T-M3.4 (Diff post-filter) へ
- **`MultiResolutionSTFTLoss` 再利用パス**: Diff の post-filter は GT vs Diff 出力の MR-STFT を closed-form で fit する (`docs/architecture.md` §6.5)。本チケットの `MultiResolutionSTFTLoss(n_ffts=[512,1024,2048], win_lengths=[360,900,1800], hop_sizes=[80,150,300], eps=1e-5)` をそのまま `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import して使用可能
- post-filter は Diff の hop_size=256 で動作するが、MR-STFT 自体の解像度設定は GAN と共通 (post-filter 出力の評価用のため、`hop_sizes` を Diff hop に揃える必要はない)

#### T-M4 (evaluation infra) へ
- **`MultiResolutionSTFTLoss` 再利用パス**: M4 の評価メトリクス (MCD, log F0 RMSE, UTMOS, NISQA) に加えて MR-STFT 値を auxiliary metric として log するなら、本チケットの `MultiResolutionSTFTLoss` をそのまま `eval_metrics.py` で再利用可能
- `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import
- 本チケットでは `losses/stft_loss.py` を M2 専用と決めつけない汎用モジュールとして設計済 (§2.5)、M4 で evaluator が独自に再実装する必要はない

#### 共通の注意事項
- **数値再現性**: `register_buffer("window", persistent=False)` のため checkpoint resume 時に window が自動再生成。state_dict mismatch が出たら `persistent=True` 化を検討 (§8.1 再評価トリガー)
- **dtype 統一**: 全 loss は `y_pred.dtype == y_true.dtype` を前提。AMP (fp16) で feats / d_outputs が mixed dtype になる場合は呼び出し側で `.float()` cast
- **device 統一**: `mrstft = MultiResolutionSTFTLoss().to(device)` で window を device 移動。`y_pred.device != mrstft.window.device` だと runtime error

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.3 の Acceptance チェックボックス 4 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M2.3 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/training.md` §2.4 の loss 式に本実装の式を引用追記 (例: `HingeGANLoss.d_loss` の擬似コード) は不要 (既に明記済み)

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` §C6 で 100% 確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - Hinge vs LSGAN ablation (M6.3)
  - FM `num_features` の 18 / 21 (§6.1 critical 解決後に pin)
  - LeCAM / R1 / R2 regularization (M6.3 ablation)
  - MR-STFT 解像度数 ablation (M6.3)
- **`docs/open-questions.md` への追記要否**: 不要
