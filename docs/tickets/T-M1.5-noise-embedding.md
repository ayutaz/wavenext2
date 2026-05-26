---
id: T-M1.5
title: Noise level embedding (sinusoidal + FC×2 SiLU, Diff 用)
milestone: M1
phase: M1
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M1.6]
related_docs:
  - docs/milestones.md#m15-noise-embedding-srcwavenext2modelsnoise_embeddingpy-diff-のみ
  - docs/architecture.md
---

# T-M1.5: Noise level embedding (sinusoidal + FC×2 SiLU, Diff 用)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.5](../milestones.md#m15-noise-embedding-srcwavenext2modelsnoise_embeddingpy-diff-のみ)
> **依存**: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M1.6](T-M1.6-sub-model.md)

## 1. タスク目的とゴール

### 目的
Diff-WaveNeXt 2 で連続的な noise level `c = √(1-ᾱ)` を ConvNeXt block 群へ条件として注入するための **shared embedding head** を実装する。論文 §3.3 で FastDiff 準拠と示された **sinusoidal embedding (128-dim, log-spaced) → FC×2 SiLU (128→512→512)** をモジュール化し、`SubModelDiff` (T-M1.6) から呼び出される共有 head として確立する。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/models/noise_embedding.py` に `sinusoidal_embedding` 関数と `NoiseEmbedding` クラスが実装され、`from wavenext2.models.noise_embedding import NoiseEmbedding, sinusoidal_embedding` でエラーなく import できる
- [ ] `c: (B,)` を入力に取り `(B, 512)` を出力する forward が動作する
- [ ] sinusoidal frequencies が `log(10000)/63` に基づく log-spaced (= `freq = exp(-arange(64) * log(10000)/63)`) であることをテストで確認
- [ ] 同一 `c` に対して deterministic、異なる `c` に対して cosine similarity < 0.99 (出力が区別可能)
- [ ] `tests/test_noise_embedding.py` の Unit テストが全 pass (`uv run pytest tests/test_noise_embedding.py` exit 0)
- [ ] 参考実装 (FastDiff `util.py::calc_diffusion_step_embedding` / `module/FastDiff_model.py`) の **コードコピーをしていない** (一から再構成、CLAUDE.md ポリシー)
- [ ] T-M1.6 (SubModelDiff) が `self.noise_embedding(c) → (B, 512)` を `cond` として ConvNeXtBlock (T-M1.1) の per-block `fc_t` に渡す経路がドキュメント化されている

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/models/noise_embedding.py` (本実装)
  - `tests/test_noise_embedding.py` (Unit テスト)
- 編集:
  - (なし。`src/wavenext2/models/__init__.py` の `__all__` re-export は **任意**。後続 T-M1.6 で SubModel と一緒に追加する方が diff が小さい)

### 2.2 主要構造

```python
"""Noise level embedding for Diff-WaveNeXt 2.

入力: 連続値 noise level c = √(1-ᾱ) (shape (B,))
出力: (B, 512) — ConvNeXtBlock (T-M1.1) の per-block `fc_t(512, 512)` が
      additive bias projection の入力として受け取る埋め込みベクトル。
"""

import math

import torch
from torch import nn


def sinusoidal_embedding(c: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Compute sinusoidal positional embedding for a continuous noise level.

    log-spaced frequencies: ``freq[i] = exp(-i * log(10000) / (half - 1))``
    where ``half = dim // 2`` (= 64 for dim=128).

    Args:
        c: shape ``(B,)`` (or broadcastable to (B,)). Continuous noise level.
        dim: total embedding dimension (must be even). Default 128.

    Returns:
        ``(B, dim)`` embedding tensor, ``[sin(c*freq); cos(c*freq)]``.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    # log-spaced freq: exp(-arange(half) * log(10000) / (half - 1))
    log_scale = math.log(10000.0) / (half - 1)
    freq = torch.exp(
        -log_scale * torch.arange(half, dtype=c.dtype, device=c.device)
    )                                              # (half,)
    args = c.unsqueeze(-1) * freq.unsqueeze(0)     # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class NoiseEmbedding(nn.Module):
    """Shared embedding head for Diff sub-model.

    sinusoidal (128) → Linear(128, 512) → SiLU → Linear(512, 512) → SiLU
    """

    def __init__(
        self,
        sinusoidal_dim: int = 128,
        mid_dim: int = 512,
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.fc1 = nn.Linear(sinusoidal_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B,) noise level → (B, out_dim)"""
        e = sinusoidal_embedding(c, dim=self.sinusoidal_dim)
        e = self.act(self.fc1(e))
        e = self.act(self.fc2(e))
        return e
```

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| sinusoidal embedding dim | **128** | `docs/architecture.md` §5.4 / `docs/open-questions.md` §C7 (FastDiff 仕様) |
| `half = dim // 2` | 64 | 同上 |
| 周波数 scale | `log(10000) / (half - 1) = log(10000) / 63` | 同上 (`calc_diffusion_step_embedding`) |
| FC1 in/out | 128 → 512 | `docs/architecture.md` §5.4 shared head |
| FC2 in/out | 512 → 512 | 同上 |
| 活性化関数 | **SiLU** (= Swish) | `docs/architecture.md` §5.4 (FastDiff 採用) |
| 入力 c の意味 | `√(1 - ᾱ)` (連続値) | `docs/architecture.md` §5.3 / `docs/open-questions.md` §B1 |
| 入力 c のレンジ | `[0, 1]` 連続 (band は `[L_k, U_k] ⊂ [0, 1]`) | 同上 |
| 出力 dim | **512** (ConvNeXt embed_dim と一致) | `docs/architecture.md` §5.4 |

### 2.4 アルゴリズム / 処理フロー

```
c (B,) ──► sinusoidal_embedding(c, dim=128) ──► e (B, 128)
                                                     │
                                                     ▼
                                              Linear(128, 512)
                                                     │
                                                     ▼
                                                  SiLU
                                                     │
                                                     ▼
                                              Linear(512, 512)
                                                     │
                                                     ▼
                                                  SiLU
                                                     │
                                                     ▼
                                                e (B, 512)
                                                     │
                                                     ▼
   T-M1.6 (SubModelDiff) で:
   各 ConvNeXt block の per-block `fc_t: Linear(512, 512)` (T-M1.1) に渡す
   → block 入口で `x = x + fc_t(e).unsqueeze(-1)` (additive bias、broadcast on T)
```

**重要な設計境界**:
- **本モジュール (`NoiseEmbedding`)** は forward あたり 1 回だけ実行される **shared head**。8 個全ての ConvNeXt block でこの同一出力を共有する
- **per-block projection** (`Linear(512, 512)`) は **T-M1.1 の `ConvNeXtBlock`** 内で保持される (`self.fc_t`)。本モジュールには **含めない**
- 各 sub-model k (k=1..4) は **独立した** `NoiseEmbedding` + 8 個の per-block `fc_t` を持つ (sub-model 間で重み共有しない)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `noise_embedding.py` + `test_noise_embedding.py` 実装 | general-purpose |
| Reviewer | 1 | FastDiff 仕様 (sinusoidal log-space 式、SiLU、shape) との整合性レビュー、参考実装コピー有無確認 | general-purpose |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M1.1 ConvNeXtBlock / T-M1.2 STFT / T-M1.3 Mel / T-M1.4 Generator と独立に作業可能。本モジュール単独で Unit テストが完結する)
- 並列実行する場合の最大並列数: 5 (M1.1〜M1.5 を並列、M1.6 のみ全完了後)
- size=S かつ依存が薄いので Tester 役は Reviewer に統合し最小編成 (Implementer 1 + Reviewer 1) で OK

## 4. 提供範囲 (Scope)

### In Scope
- `sinusoidal_embedding(c, dim=128)` 関数の実装 (log-spaced freq、sin || cos concat)
- `NoiseEmbedding` nn.Module の実装 (FC1 → SiLU → FC2 → SiLU)
- Unit テスト (§5.1 全項目)
- §5.3 Acceptance criteria の全項目検証
- docstring (Args / Returns / Shape / 出典 references)

### Out of Scope
- **per-block additive bias 注入** → **T-M1.1 (ConvNeXtBlock)** の `fc_t: Linear(512, 512)` で実装
- **`SubModelDiff` への組み込み** → **T-M1.6** で `self.noise_embedding = NoiseEmbedding(...)` を保持し、forward 内で `cond = self.noise_embedding(c)` → ConvNeXtBlock へ渡す
- GAN-WaveNeXt 2 用の embedding (GAN は conditioning なし、本モジュールは Diff 専用)
- 学習可能 Fourier feature / GaussianFourierProjection 等の代替実装 (§8.1 で代替案として列挙)
- training script への組み込み (M3.2 で `train_diff.py` 実装時)
- ONNX export 動作確認 (M6 以降、推論最適化フェーズ)

### Deliverable
- ファイル:
  - `src/wavenext2/models/noise_embedding.py` (新規)
  - `tests/test_noise_embedding.py` (新規)
- 関数 / クラス:
  - `sinusoidal_embedding(c: torch.Tensor, dim: int = 128) -> torch.Tensor`
  - `class NoiseEmbedding(nn.Module)` (`__init__(sinusoidal_dim=128, mid_dim=512, out_dim=512)`, `forward(c: torch.Tensor) -> torch.Tensor`)
- ドキュメント差分:
  - `docs/milestones.md` §M1.5 の Acceptance チェックボックス 4 項目を更新
  - `docs/tickets/index.md` の T-M1.5 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_noise_embedding.py`)

- [ ] `test_sinusoidal_shape`: `sinusoidal_embedding(torch.tensor([0.5]), dim=128).shape == (1, 128)`
- [ ] `test_sinusoidal_even_dim_required`: `sinusoidal_embedding(c, dim=127)` で `ValueError`
- [ ] `test_sinusoidal_freq_log_spaced`: **厳密一致テスト**: `assert freq[0] / freq[-1] == 10000.0` (許容なし、float32 で完全一致を確認)。実装で `freq = exp(-arange(64) * log(10000) / 63)` なので `freq[0] = exp(0) = 1.0`, `freq[-1] = exp(-log(10000)) = 1e-4`、比は厳密に 10000。log_scale off-by-one バグ (`/half` vs `/(half-1)`) の早期検知に必須
- [ ] `test_sinusoidal_concat_order`: 前半 64 次元が sin、後半 64 次元が cos であることを検証 (`embed[:, 64:]` が cos(0) = 1 から始まる成分を持つ etc.)。**concat 順注記**: FastDiff は `[sin; cos]` 順、Transformer 慣例 (Diffusion-LM 等) は `[cos; sin]` や `[sin_even; cos_odd]` interleave の場合もある。後段 `Linear(128, 512)` で permutation invariant なので学習可能性に影響しないが、SHA256 pin (下記) との一貫性のため `[sin; cos]` 固定とする
- [ ] `test_sinusoidal_sha256_pin`: `sinusoidal_embedding(torch.tensor([0.5, 0.9]), dim=128)` の出力テンソルの SHA256 を `tests/snapshots/noise_emb.json` に **pin**。log_scale off-by-one バグ / concat 順変更 / dtype 不一致を CI で即検知する regression test。値が変わる変更は意図的に snapshot を更新するレビューを要する
- [ ] `test_sinusoidal_property` (`hypothesis`): `sinusoidal_embedding` の `c` を `st.floats(min_value=0.0, max_value=1.0)` で property test。**不変条件**: (a) `freq[0]/freq[-1] ≈ 10000` を 100 ケースで保持、(b) 出力 dim が常に `dim` 引数と一致、(c) `(out**2).sum(-1)` が `dim/2 = 64` (= sin² + cos² = 1 × half)。`hypothesis` を `pyproject.toml` 開発依存に追加
- [ ] `test_noise_embedding_forward_shape`: `NoiseEmbedding()(torch.tensor([0.5]))` の出力 shape が `(1, 512)`
- [ ] `test_noise_embedding_batch`: `NoiseEmbedding()(torch.rand(8))` の出力 shape が `(8, 512)`
- [ ] `test_noise_embedding_deterministic`: 同じ c を 2 回 forward した結果が `torch.allclose` (eval mode、no Dropout)
- [ ] `test_noise_embedding_distinguishes_inputs`: `c1 = 0.1`, `c2 = 0.9` に対する出力の cosine similarity < 0.99 (異なる noise level で区別可能)
- [ ] `test_noise_embedding_param_count`: パラメータ数 = `128*512 + 512 + 512*512 + 512 = 328,704` (FC1: 65,536 + 512 = 66,048; FC2: 262,144 + 512 = 262,656; 合計 328,704)
- [ ] `test_noise_embedding_gradient_flow`: `loss = NoiseEmbedding()(c).sum(); loss.backward()` で全パラメータの `.grad` が `None` でない (学習可能性確認)
- [ ] `test_noise_embedding_silu_activation`: forward 内で SiLU が使われていることを (a) `isinstance(module.act, nn.SiLU)` で確認、(b) ReLU と異なる挙動を負入力で確認
- [ ] `test_noise_embedding_devices_dtype`: `c.dtype` と `c.device` に出力が追従する。**fp16/bf16 追加**: CPU float32 / CUDA float32 / float64 / **bf16 (`(freq > 0).all()` を確認)** / fp16 (`(freq > 0).all()` を確認、ただし CPU fp16 の arange+log が非決定なら `pytest.mark.skip`)。CUDA テストは `pytest.mark.skipif(not torch.cuda.is_available())`

### 5.2 テスト実装上の取り決め
- **`@pytest.fixture(scope="module") def noise_emb_model`**: `NoiseEmbedding()` instance を 12 テスト (forward 系) で再利用し、毎テストでの初期化コストを削減。`conftest.py` 側 (T-M0.2) に置くか、`test_noise_embedding.py` 内 module scope どちらでも可だが、本ファイル内に置けば責務が明確
- **CI 時間目標**: < 2 秒 (size=S かつ純粋関数中心、SHA256 pin 比較も高速)
- **coverage 目標**: 100% (純粋関数中心、branch も含めて全カバー可能。`pytest --cov=wavenext2.models.noise_embedding --cov-fail-under=100` を CI 設定の指針)
- **fp16 deterministic**: CPU の fp16 で `arange + log + exp` の合成は PyTorch 公式で deterministic 保証なし。**fp16 テストは値比較ではなく `(freq > 0).all()` などの不変条件のみ**を確認し、SHA256 pin は float32 限定で行う

### 5.3 e2e / 結合テスト
- (本チケット時点では実施しない) T-M1.6 (SubModelDiff) 完成後に統合テストで確認:
  - `SubModelDiff(...)` 内の `self.noise_embedding(c)` → ConvNeXtBlock の `fc_t(e)` → block forward まで通る
  - 異なる noise level c を入れて出力波形が変わる (T-M3.5 smoke test の前段確認)

### 5.4 Acceptance criteria (`docs/milestones.md` §M1.5 より転記)
- [ ] `c = torch.tensor([0.5])` で出力 shape `(1, 512)`
- [ ] 同じ c に対する出力が deterministic
- [ ] 異なる c に対する出力が異なる (cosine similarity < 0.99)
- [ ] freq の log-spaced 確認: `freq[0] / freq[-1] ≈ 10000` (= `log(10000)` スケール)

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **sinusoidal vs Fourier feature の選択誤り** | Diff サンプリング品質劣化 / 学習不安定 | FastDiff は **sinusoidal (DDPM/Transformer 標準)** を採用 (`docs/open-questions.md` §C7)。本実装も sinusoidal を採用。Gaussian Fourier projection (= NeRF/Score-based) は §8.1 代替案で言及。再評価条件: M3.5 smoke で loss が収束しない場合 |
| **SiLU vs Swish の命名揺れ** | コードレビューでの混乱 | PyTorch では `nn.SiLU = Swish` (同一関数 `x * sigmoid(x)`、PyTorch 1.7+ で `SiLU` 公式名)。本実装は `nn.SiLU` を採用。docstring に「= Swish」を明記 |
| **per-block projection の境界混同** | T-M1.1 ConvNeXtBlock との責務重複・抜け | 本モジュールには `fc_t` を **含めない**。`NoiseEmbedding` は **shared head** のみ。per-block `Linear(512, 512)` は ConvNeXtBlock 内の `self.fc_t` (T-M1.1) に置く。本チケット §2.4 のフロー図と §9.1 で明示 |
| **`log_scale = log(10000) / (half - 1)`** vs **`log_scale = log(10000) / half`** の off-by-one | freq[0]/freq[-1] 比が 10000 にならず Acceptance §5.3 4 項目目が落ちる | FastDiff 実装は `(half - 1)` (= 63 for half=64) を採用。本実装も同じ。`freq[0] = exp(0) = 1.0`, `freq[-1] = exp(-log(10000)) = 1e-4`、比は 10000。テスト `test_sinusoidal_freq_log_spaced` で確認 |
| **入力 c の dtype 不一致** | `freq = torch.exp(... * arange(...))` で float64/float32 mismatch エラー、特に AMP fp16 で発生 | `torch.arange(half, dtype=c.dtype, device=c.device)` で c に追従させる。テスト `test_noise_embedding_devices_dtype` で float32/float64 ラウンドトリップを確認。AMP fp16 / bf16 は M3.2 (`train_diff.py`) で対応 |
| **入力 c の shape の柔軟性** | `(B,)`/`(B, 1)`/scalar の混在で broadcast 事故 | docstring で `(B,)` を仕様とし、`(B, 1)` で来た場合は呼び出し側で `squeeze` する責務にする。本実装は `c.unsqueeze(-1)` だけ行うので `(B,)` 専用。Out-of-shape の場合は assertion を入れるか型ヒントで `(B,)` を明示。**判断**: `(B,)` 専用とし、assertion は入れず docstring で十分とする (Python の duck typing) |
| **CUDA seed 設定が必要なケース** | Acceptance §5.3「同じ c で deterministic」は学習可能パラメータが固定であれば自動的に成立 | テスト側で `torch.manual_seed(42)` を fixture で固定し、新規 `NoiseEmbedding()` のパラメータも reproducible にする (`tests/conftest.py` で global seed 設定) |
| **既存の `torch.nn.Embedding` 機能との誤解** | 「embedding」という名前から `nn.Embedding(num_embeddings, dim)` (= lookup table) と混同されがち | docstring 冒頭に「これは **連続値** noise level を **continuous な** 埋め込み空間にマップする shared head であり、discrete index lookup の `nn.Embedding` ではない」と明記 |
| **FastDiff 参考実装の license 確認** | コードコピーすると license 違反リスク (FastDiff は MIT) | CLAUDE.md ポリシー通り **コピー流用しない**。式 (sinusoidal + FC×2 SiLU) の同一性は本論文 / FastDiff 仕様で確定済みのため、独自に PyTorch で書き起こす。Reviewer で git blame と FastDiff 原コードを照合確認。**FastDiff `nn.Linear(bias=True)` の根拠**: 現状「FastDiff `Linear` は default `bias=True`」記述だが、FastDiff の `module/FastDiff_model.py` の具体的な行番号や class 名 (`DiffusionEmbedding` / `Conv` ラッパ) を docstring/comment に残す。Reviewer で原コード行番号を付記して根拠を確実にトレースできる状態にする |
| **`c=0.0` で `Linear.bias` 支配的になる問題** (重要) | `sin(0)=0, cos(0)=1` で sinusoidal 出力が決定的パターン (前半 64 dim = 0、後半 64 dim = 1) → `Linear(128, 512).bias` がそのまま `cond` の支配項。学習初期に c≈0 の sub-model (k=1 など、低 noise band) で `bias` がそのまま信号として漏れる懸念。**緩和案**: (a) `fc1.bias` を **zero init** (デフォルト Kaiming uniform から変更)、(b) `fc1.bias` を **trainable のまま** (FastDiff 原コード準拠) のどちらを採用するか議論。本チケットでは **(b) FastDiff 準拠 (default init)** を採用し、Reviewer 確認項目に追加。M3.5 smoke で k=1 sub-model の学習不安定が観測された場合 (a) zero init を ablation で試す |
| **bf16/fp16 で `freq` underflow** | `freq[-1] = exp(-9.21) ≈ 1e-4`。bf16 (mantissa 8bit, range fp32 と同等) では保持可能だが、fp16 (range 6.1e-5〜65504) では underflow ギリギリ。AMP fp16 訓練 (M3.2) で `freq[-1]` が 0 になり sin/cos がすべて 0 化するリスク | `test_noise_embedding_devices_dtype` に **fp16 ケースを追加** し `(freq > 0).all()` を確認。bf16 は安全なので preferred dtype として推奨。**CPU の fp16 で arange + log の合成が非決定 risk あり** (PyTorch CPU の fp16 算術は完全 deterministic 保証なし)、fp16 deterministic テストは skip 候補とし、bf16 で確認する |

### 6.2 仕様の曖昧さ
- **`docs/open-questions.md` §C7** で sinusoidal + FC×2 SiLU が確定済み。残る曖昧さは **per-block projection の bias の有無** (FastDiff `Linear` は default `bias=True`)。本チケットの shared head は `nn.Linear` の default (`bias=True`) を採用。T-M1.1 ConvNeXtBlock の `fc_t` も同様に `bias=True` を採用する想定だが、T-M1.1 の責務として明示する (本チケットの §9.1 で連絡)
- **入力 c の事前 scaling**: FastDiff では c をそのまま (= `√(1-ᾱ)` ∈ [0, 1]) sinusoidal に渡す。本実装も同様。DDPM の `t` (integer index 0..T) を渡す慣例とは異なる
- **`mid_dim` = `out_dim` = 512** か `mid_dim > out_dim` (bottleneck) か: FastDiff は両方 512 (= `Linear(128, 512) → Linear(512, 512)`)。本実装も同じ

### 6.3 他チケットとの整合性
- **T-M1.1 (ConvNeXtBlock)**: Diff モード時の `conditioning_dim=512` と本モジュールの `out_dim=512` が一致する必要あり。本チケットで `out_dim=512` を **default** とし、T-M1.1 と T-M1.5 のレビューで両方の値が一致していることを確認 (§9.1 連絡事項に明記)
- **T-M1.4 (WaveNextGenerator) パラメータ数 cross-reference**: `NoiseEmbedding` は **0.33M** (= 328,704 params)。Diff sub-model では `conditioning_dim=512` を `WaveNextGenerator` の各 ConvNeXtBlock `fc_t(512, 512)` (per-block 約 0.26M × 8 block = **約 2.1M**) に渡す。論文 Table 1 の Diff **14.42M** に NoiseEmbedding 自身のパラメータ (0.33M) が含まれているか **T-M1.6 で再確認**。仮に含まれていない場合は per-sub-model で 0.33M × 4 sub-model = 1.32M の追加カウントが発生 (§9.1 に明記)
- **T-M1.6 (SubModelDiff)**: `self.noise_embedding = NoiseEmbedding()` を保持、forward で `cond = self.noise_embedding(c)` → `self.generator(x, cond=cond)` を呼ぶ。**本モジュールの forward 返り値 shape `(B, 512)` を T-M1.6 と T-M1.1 が共通の `conditioning_dim` 値 (= 512) で扱う**
- **T-M3.1 (DiffWaveNext2)**: 4 sub-model それぞれが **独立した** `NoiseEmbedding` を持つ。本チケットでは「sub-model 間で重み共有しない」方針を docstring に明記
- **T-M3.2 (train_diff.py)**: 各 sub-model の訓練時に `c ~ U(L_k, U_k)` を sample してこの module に渡す。本チケットでは sampling は実装しない (T-M3.1 `sample_noise_level(k, B)` の責務)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] **論文記述 / docs 整合**: `docs/architecture.md` §5.4 の式・shape・活性化関数 (SiLU) と一致
- [ ] **FastDiff 仕様一致**: `freq = exp(-arange(64) * log(10000) / 63)`、`[sin(c*freq); cos(c*freq)]` の concat 順、`Linear(128, 512) → SiLU → Linear(512, 512) → SiLU` の構造
- [ ] **参考実装をコピーしていない** (CLAUDE.md ポリシー): FastDiff `util.py::calc_diffusion_step_embedding` の本体 Python コードを `git blame` で確認し、独自に書き起こされていること (式は同じだが実装の表現が異なる)
- [ ] **Acceptance criteria 全項目クリア** (§5.3 4 項目)
- [ ] **Unit テスト全 pass** (`uv run pytest tests/test_noise_embedding.py` exit 0、§5.1 全項目)
- [ ] **スタイル準拠** (CLAUDE.md): 型ヒント、docstring (Args/Returns/Shape)、命名、`from torch import nn` の import 方式
- [ ] **エラー処理**: `dim` が奇数の時 `ValueError`、c の shape mismatch は assertion or duck typing どちらかで一貫
- [ ] **パラメータ数 / メモリ消費が想定内** (= 328,704、約 0.33M。sub-model 14.42M の 2% 程度。`docs/architecture.md` §5 のパラメータ予算と整合)
- [ ] **`nn.SiLU` を採用** (Swish と等価、PyTorch 公式名)
- [ ] **`out_dim=512` default**: T-M1.1 ConvNeXtBlock の `conditioning_dim` 想定値 (512) と一致
- [ ] **per-block projection が含まれない**: 本モジュールは shared head のみで `fc_t: Linear(512, 512)` を持たない (それは T-M1.1 ConvNeXtBlock の責務)
- [ ] **import path**: `from wavenext2.models.noise_embedding import NoiseEmbedding, sinusoidal_embedding` でエラーなし

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **`c * 1000` rescale (DDPM ステップ相当)** (重要): sinusoidal 前に `c_scaled = c * 1000` を適用 | `c ∈ [0, 1]` という極めて小さな入力に対し `freq[0] = 1.0` でも `sin(1.0 * 1.0) = 0.84` で **sin の 1 周期分も使えない** 問題を解消。`log(10000)/63` 周波数 scale は DDPM の `t ∈ [0, 1000]` 整数想定で設計されており、`c * 1000` で本来の動作域に揃う。低周波数 bin (freq < 1) が無駄にならず表現力が回復する | FastDiff 原実装は `c` (∈ [0,1]) をそのまま渡す方針 (FastDiff `util.py::calc_diffusion_step_embedding` の引数 `diffusion_steps` が連続値 c)。論文再現性最優先 | **M3.5 smoke で発散 / 学習不安定 / loss 停滞時の最優先 ablation**。`NoiseEmbedding(input_rescale=1000.0)` フラグ追加で切替可能にしておく。低周波数 bin が無駄になっている点は本質的な懸念で、本チケット §6.1 に重要懸念として追記済み |
| **`log_scale = log(10000)` を直接調整** (例: `log(100)` に変更し freq[-1] = 0.01 にする) | `c * 1000` rescale 案の代替。freq の range を縮めて c の小ささに適応させる | 全 freq bin の特性が同時に変わり、表現空間の解析性が下がる。`c * 1000` の方が "DDPM 標準に揃える" という解釈で見通しが良い | FastDiff の `log(10000)` を維持し、`c * 1000` の方を ablation 候補とする | 同上 (M3.5 smoke ablation) |
| **`sin || cos` vs `cos || sin` concat 順序** | Diffusion-LM の論文では `[cos; sin]` 順を採用、Transformer 慣例は `[sin_even; cos_odd]` interleave、FastDiff は `[sin; cos]` | 後段 `Linear(128, 512)` で **permutation invariant** なので学習可能性 / 最終性能に影響しないが、warm-start や ablation 比較時に snapshot SHA256 が変わって混乱する | FastDiff 準拠の `[sin; cos]` 順を採用 (本チケット §5.1 SHA256 pin の一貫性) | **学習済み weight の互換性が問題になった場合** (例えば FastDiff pretrained を warm-start に使う ablation を行うとき)。基本的には再評価不要 |
| **Learnable Fourier feature (NeRF 風)** `exp(2π i · W · c)` with learnable W ~ N(0, σ²) | 連続値の細部の表現力が高い、frequency が train data 統計に適応。**`c * 1000` rescale が不要になる可能性**もあり、適応的に解決 | パラメータ追加 (W: `(64, 1)`)、初期化感度が高い、論文 / FastDiff の標準から外れる | FastDiff 標準の sinusoidal を採用。論文再現性最優先 | **M3.5 smoke で sinusoidal で品質が出ない場合** に学習可能化を試す。`NoiseEmbedding(use_learnable_fourier=True)` のフラグ追加で互換性を保つ。**`c * 1000` rescale 試行後の二次 ablation 候補** |
| **Gaussian Fourier Projection (DDPM 系)** `[sin(2π·B·c); cos(2π·B·c)]` with fixed B ~ N(0, σ²·I) | DDPM/score-based 系で実績、高周波数の表現に強い | σ² の hyperparameter チューニングが追加、初期化乱数依存で再現性に乱数 seed 管理が要る | FastDiff (sinusoidal) と論文整合性を優先 | **score-based 系の vocoder と比較する ablation 時** に試す。**`c * 1000` rescale 試行後の二次 ablation 候補** |
| **Discrete embedding lookup (`nn.Embedding(N_steps, 512)`)** | 学習可能、最も柔軟 | **連続値 c の sampling と非互換** (band 内 uniform sampling を使う本論文では使えない)、推論で schedule の正確な値以外を渡せない | 連続値 conditioning が論文 §3.3 の前提 (band 内 uniform sampling のため) なので採用不可 | **連続性が不要になった場合** (= schedule 点を 4 固定値だけで使う設計に戻す場合)。現状は採用しない |
| **時刻 t を直接 sin/cos (`sin(c)`, `cos(c)`) のみ、log-spaced 不使用** | 実装が極小 | dim=2 では表現力不足、`half=1` で degeneracy | FastDiff / DDPM の確立された方式 (log-spaced 64 freq + sin/cos = 128 dim) を採用 | **再評価しない** (表現力不足が明らか) |
| **FC×2 SiLU を FC×3 SiLU に拡張** | 表現力増大 | パラメータ増加 (Linear(512, 512) を 1 段追加で +262,656)、過学習リスク | FastDiff の `Linear(128, 512) + Linear(512, 512)` の **2 段** が標準 | **M3.5 smoke で表現力不足が確認された場合** |
| **SiLU を GELU に置換** (本実装の generator 内の ConvNeXt block は GELU) | activation を generator 内と統一 | FastDiff 標準と乖離、論文の "Swish" 記述と非整合 | FastDiff / 論文の SiLU 採用 | **再評価しない** (FastDiff 仕様準拠が論文整合性の基準) |
| **shared head を sub-model 間で共有** (= 1 つの head を 4 sub-model で再利用) | パラメータ約 1MB 削減、汎化向上の可能性 | sub-model ごとの特化を妨げる、論文の「sub-model k が band k に特化」の方針に反する、各 sub-model 独立訓練のスケジュールに不適 | 論文 §3.3 「each sub-model is trained independently」と整合させ、**sub-model 間で重み共有しない** (本チケットの §6.3 / §9.1) | **共有版でも品質が落ちないと示せた場合** (M6 ablation) |
| **sinusoidal_embedding を nn.Module 化 (キャッシュ付き)** | freq buffer を `register_buffer` で持てる、`.to(device)` で自動移動 | 状態を持つ Module が増える、関数のシンプルさが失われる | 入力 c のサイズが (B,) と小さく、freq 計算コストが軽微なので関数のままで良い | **freq 計算が hot loop で profile に出た場合** (M6 推論最適化フェーズで再評価) |

### 8.2 思想 / 哲学の見直し

- **粒度**: T-M1.5 は size=S と妥当。`noise_embedding.py` 単独で完結し、テストも約 30 分で書ける。**分割不要**
- **本チケットを T-M1.1 (ConvNeXtBlock) に統合する案**: 本モジュールは「shared head」、ConvNeXtBlock 内の `fc_t` は「per-block projection」。**責務が異なる** ため分離した方が境界が綺麗。統合すると ConvNeXtBlock が Diff 固有の sinusoidal を抱え込み、GAN モードでの再利用性が下がる
- **本チケットを T-M1.6 (SubModelDiff) に統合する案**: SubModelDiff は (STFT + Generator + NoiseEmbedding) の集約モジュール。NoiseEmbedding を分離することで T-M1.5 と T-M1.6 の Unit テストが独立化し、debug 効率が上がる
- **`sinusoidal_embedding` を関数 vs Module**: 状態を持たない (= freq が入力 dtype/device に追従するだけ) ため **関数** が自然。`NoiseEmbedding` Module から呼び出される単純な helper として位置づける
- **インターフェース定義の見直し余地**:
  - 出力 dim を **default 512** にしておくことで、ConvNeXtBlock の `conditioning_dim` 想定値 (512) と一致しやすい
  - sub-model k = 1..4 のどれであるかを **本モジュールに渡さない** 設計 (sub-model ごとに別 instance を作る方が独立性が高い)
- **`embed_dim` Single Source of Truth (SoT) の明示**: 現状、以下の **3 箇所** が同期必要:
  - (1) `NoiseEmbedding.out_dim = 512`
  - (2) `ConvNeXtBlock.dim = 512`
  - (3) `ConvNeXtBlock.conditioning_dim = 512`

  将来 `embed_dim` を変更する場合 (例: 256 / 1024 への変更 ablation) の SoT は **`configs/diff_wavenext2.yaml` の `model.sub_model.convnext.embed_dim`** とする。本チケットでは default 512 をハードコードせず、`SubModelDiff` (T-M1.6) の `__init__` で config から両方に伝播する想定で **`NoiseEmbedding(out_dim=embed_dim)` を呼び出す**。本ファイル §9.1 に T-M1.6 / T-M3.1 への申し送りとして明記
- **`factory` パターンの予約**: T-M1.2 (STFTModule) / T-M1.3 (MelTransform) / T-M1.4 (WaveNextGenerator) と統一して **`NoiseEmbedding.from_config(cfg)` クラスメソッド** を予約 (本チケットでは実装しない、T-M1.6 SubModelDiff 統合時に config 駆動の初期化を導入する際に追加)。これにより `cfg.model.sub_model.noise_embedding` ブロックから一発で `NoiseEmbedding` を構築できる
- **再評価トリガー**: §8.1 の表を参照

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

- **インターフェース** (T-M1.6 SubModelDiff / T-M3.1 DiffWaveNext2 が利用):
  - `from wavenext2.models.noise_embedding import NoiseEmbedding, sinusoidal_embedding`
  - `NoiseEmbedding(sinusoidal_dim=128, mid_dim=512, out_dim=512)`
  - `forward(c: (B,)) -> (B, 512)`
  - **入力 c の意味**: `√(1 - ᾱ)` (連続値、`[0, 1]` 区間)。**`ᾱ` そのものや DDPM の integer step t を渡さない**
- **T-M0.2 conftest.py へ追加要請**:
  - `set_seed` autouse fixture (毎テスト前に `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` を設定) を `tests/conftest.py` に追加
  - `noise_emb_model` fixture (module scope) は本チケットのテストファイル内に置くか、後続で他テストが再利用するなら `conftest.py` へ昇格
- **T-M1.6 (SubModelDiff)** が踏まえる経路:
  ```python
  class SubModelDiff(nn.Module):
      def __init__(self, ...):
          self.noise_embedding = NoiseEmbedding()                  # ← 本チケット
          self.stft_module = STFTModule(...)                        # T-M1.2
          self.generator = WaveNextGenerator(..., conditioning_dim=512)  # T-M1.4
      def forward(self, mel, x_t, c):                                # c: (B,)
          cond = self.noise_embedding(c)                            # (B, 512)
          stft_spec = self.stft_module(x_t, T_mel=mel.shape[2])
          x = torch.cat([mel, stft_spec], dim=1)
          return self.generator(x, cond=cond)                       # cond → 各 ConvNeXt block の fc_t
  ```
  - `cond` は **8 個の ConvNeXt block すべてに共通で渡る**。各 block の `fc_t` (T-M1.1 内、per-block 独立) が `Linear(512, 512)` で射影して additive bias 化する
  - **`fc1.bias = 0` zero init の判断**: §6.1「`c=0.0` で `Linear.bias` 支配的になる問題」を踏まえ、T-M1.6 統合時に zero init を採用するか否かを Reviewer が判断 (現状は **FastDiff 準拠 default init** を採用、k=1 sub-model の学習不安定が観測された場合 ablation)
  - **Linear `bias=True` 採用根拠**: FastDiff `module/FastDiff_model.py` の `DiffusionEmbedding` クラスで `nn.Linear` の default (`bias=True`) を使用している。具体的な行番号と class 名は本実装の docstring / コメントに記載することで根拠をトレース可能にする
  - **NoiseEmbedding パラメータ数の Table 1 cross-reference**: `NoiseEmbedding` 自身は 0.33M (= 328,704)。Diff sub-model 全体は per-sub-model で 14.42M (論文 Table 1 / `docs/architecture.md` §5)。**Table 1 の 14.42M に NoiseEmbedding 0.33M が含まれているか / 別計上か** を T-M1.6 統合時の Reviewer で再確認 (per-sub-model か all-sub-model 合計かも併せて確認)
- **T-M1.1 (ConvNeXtBlock) との取り決め**:
  - `ConvNeXtBlock(conditioning_dim=512)` (Diff モード) の `cond` 引数は本チケットの `NoiseEmbedding` の出力をそのまま受け取る
  - `conditioning_dim=512` と本モジュールの `out_dim=512` が **同一値であることをコードレビューで保証**
- **T-M3.1 (DiffWaveNext2)** との取り決め:
  - 4 sub-model は **独立した** `NoiseEmbedding` を持つ (sub-model 間で重み共有しない)
  - sub-model k の forward 時に `c ~ U(L_k, U_k)` を渡す (`sample_noise_level(k, batch_size)` は T-M3.1 の責務)
  - **パラメータ数勘定**: 4 sub-model 独立で `NoiseEmbedding` × 4 = **0.33M × 4 = 1.32M**。論文 Table 1 が per-sub-model 表記 (14.42M) なら NoiseEmbedding 0.33M がその内訳に入っているか、全モデル合計 4 × 14.42M = 57.68M に NoiseEmbedding 1.32M が入っているか、**Table 1 cross-check が必要** (T-M1.6 / T-M3.1 で確定)
- **M3 smoke 発散時の最優先 ablation**: `c * 1000` rescale (DDPM ステップ相当へ rescale、§8.1 参照) を最初に試す。`NoiseEmbedding(input_rescale=1000.0)` フラグで切替できるように設計しておくと ablation が容易
- **注意事項**:
  - dtype/device は c に追従する (`torch.arange(..., dtype=c.dtype, device=c.device)`)
  - `(B,)` 専用、`(B, 1)` は呼び出し側で `squeeze(-1)` する
  - eval mode / train mode で挙動は変わらない (Dropout/BatchNorm を含まないため)
  - bf16 推奨、fp16 は `freq[-1] ≈ 1e-4` の underflow リスクをテストで確認した上で使用すること (§6.1)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M1.5 の Acceptance チェックボックス 4 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M1.5 ステータスを `📝 pending` → `✅ completed`
  - [ ] (該当時) `docs/architecture.md` §5.4 に実装ファイル参照リンクを追加 (`src/wavenext2/models/noise_embedding.py`)
  - [ ] (該当時) `src/wavenext2/models/__init__.py` の `__all__` に `NoiseEmbedding`, `sinusoidal_embedding` を re-export (T-M1.6 と合わせて実施でも可)

### 9.3 Open question として残ったもの
- 解決できなかった疑問: なし (FastDiff 仕様で完全に確定)
- `docs/open-questions.md` への追記要否: 不要
- 将来検討事項:
  - Gaussian Fourier projection や learnable Fourier feature への切り替え判断は M3.5 smoke 後 (§8.1 再評価トリガー参照)
  - sub-model 間の `NoiseEmbedding` 共有 ablation は M6 で検討
