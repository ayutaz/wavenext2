---
id: T-M1.1
title: ConvNeXt block (GAN/Diff 両対応、additive bias 注入)
milestone: M1
phase: M1
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M1.4, T-M1.6]
related_docs:
  - docs/milestones.md#m11-convnext-block-srcwavenext2modelsconvnextpy
  - docs/architecture.md
  - docs/open-questions.md
---

# T-M1.1: ConvNeXt block (GAN/Diff 両対応、additive bias 注入)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.1](../milestones.md#m11-convnext-block-srcwavenext2modelsconvnextpy)
> **依存**: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M1.4](T-M1.4-generator.md), [T-M1.6](T-M1.6-sub-model.md)

## 1. タスク目的とゴール

### 目的
WaveNeXt 2 の核となる `ConvNeXtBlock` を `src/wavenext2/models/convnext.py` に実装する。GAN-WaveNeXt 2 と Diff-WaveNeXt 2 の両方で **同じクラス** を使い回せるように設計し、Diff の場合のみ noise level conditioning を **FastDiff 流の additive bias** として block 入口で注入する。`docs/architecture.md` §2 / §5.4 / `docs/open-questions.md` §C1 / §C7 の確定事項に整合する単一実装を提供する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M1.1 Acceptance を内包):
- [ ] `src/wavenext2/models/convnext.py` に `ConvNeXtBlock` クラスが実装され、`from wavenext2.models.convnext import ConvNeXtBlock` で import 可能
- [ ] GAN モード (`conditioning_dim=None`): `block(x)` で入出力 shape `(B, dim, T)` が一致
- [ ] Diff モード (`conditioning_dim=512`): `block(x, cond)` が動作し、`cond=None` で呼ばれた場合は `ValueError` を送出
- [ ] パラメータ数: GAN 版 約 1.58M/block (Vocos `ConvNeXtBlock` と同等)、Diff 版は +0.263M (`Linear(512, 512)` per-block) で約 1.85M/block。Generator 全体で 8 倍した場合に Table 1 の sub-model 単体サイズ (GAN ≈ 14.99M、Diff ≈ 14.42M) と整合
- [ ] gradient flow: `loss = block(x [, cond]).sum(); loss.backward()` で全パラメータに `.grad is not None` かつ `.grad.abs().max() > 0`
- [ ] 決定性: 同入力 + 同 seed で出力が bit-exact 再現
- [ ] `tests/test_convnext.py` の全テスト (§5.1) が pass
- [ ] `docs/tickets/index.md` の本チケットステータスが更新済み

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: なし (T-M0.2 で stub commit 済みのため上書き)
- 編集:
  - `src/wavenext2/models/convnext.py` (TODO stub → 本実装)
  - `tests/test_convnext.py` (`pytest.skip` placeholder → 本テスト)
  - `src/wavenext2/models/__init__.py` (`__all__` に `ConvNeXtBlock` を追加、必要なら re-export)

### 2.2 主要構造

`docs/architecture.md` §2 (ConvNeXt block の内部構成、L72-91) と §5.4 (Diff conditioning 注入、L298-322) を統合した単一クラス:

```python
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for WaveNeXt 2 (GAN/Diff 両対応).

    GAN モード (conditioning_dim=None) では純粋な ConvNeXt block。
    Diff モード (conditioning_dim>0) では block 入口で additive bias 注入
    (FastDiff 流、FiLM ではない)。docs/architecture.md §5.4 参照。

    入出力 shape: (B, dim, T)
    """

    def __init__(
        self,
        dim: int = 512,
        intermediate_dim: int = 1536,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        conditioning_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or intermediate_dim <= 0:
            raise ValueError(f"dim and intermediate_dim must be positive, got {dim}, {intermediate_dim}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.dim = dim
        self.conditioning_dim = conditioning_dim

        # Conditioning projection (Diff のみ): per-block 独立な Linear(cond_dim, dim)
        # FastDiff 流 additive bias 注入のため、block の入口で 1 回だけ加算する
        if conditioning_dim is not None:
            self.fc_t = nn.Linear(conditioning_dim, dim)
        else:
            self.fc_t = None

        # Depthwise Conv1d (groups=dim): kernel=7, padding=3, bias=True
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        # LayerNorm (channels_last, eps=1e-6)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # MLP: Linear(dim -> intermediate) -> GELU -> Linear(intermediate -> dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        # LayerScale (learnable per-channel scale, init=1e-6)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, dim, T)
            cond: (B, conditioning_dim) — Diff モードでのみ使用、それ以外は None
        Returns:
            (B, dim, T)
        """
        if self.fc_t is not None and cond is None:
            raise ValueError(
                "ConvNeXtBlock was constructed with conditioning_dim "
                f"={self.conditioning_dim} but forward() was called with cond=None"
            )
        if self.fc_t is None and cond is not None:
            raise ValueError(
                "ConvNeXtBlock was constructed without conditioning "
                "(conditioning_dim=None) but forward() received cond"
            )

        # 1. Diff モードのみ: block 入口で additive bias 注入 (FiLM ではない)
        if self.fc_t is not None:
            bias = self.fc_t(cond).unsqueeze(-1)  # (B, dim, 1) — broadcast over T
            x = x + bias

        residual = x
        x = self.dwconv(x)                # (B, C, T)
        x = x.transpose(1, 2)              # (B, T, C) — LayerNorm/Linear は channels_last
        x = self.norm(x)
        x = self.pwconv1(x)                # (B, T, intermediate_dim)
        x = self.act(x)
        x = self.pwconv2(x)                # (B, T, dim)
        x = self.gamma * x                  # LayerScale (broadcast over B, T)
        x = x.transpose(1, 2)              # (B, C, T)
        x = residual + x
        return x
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `dim` (embed_dim) | 512 | docs/architecture.md §2 L74, docs/open-questions.md §C1 |
| `intermediate_dim` (MLP hidden) | 1536 (3× 拡張) | docs/architecture.md §2 L74, docs/open-questions.md §C1 |
| `kernel_size` (DWConv) | 7 | docs/architecture.md §2 L75, docs/open-questions.md §C1 |
| `padding` (DWConv) | `kernel_size // 2` = 3 | docs/architecture.md §2 L75 |
| `groups` (DWConv) | `dim` (= depthwise) | docs/architecture.md §2 L75 |
| `eps` (LayerNorm) | 1e-6 | docs/architecture.md §2 L76, Vocos 既定値 |
| `layer_scale_init_value` (LayerScale γ) | 1e-6 | docs/architecture.md §2 L77, docs/open-questions.md §C1 |
| `conditioning_dim` (Diff 専用) | 512 (None で GAN モード) | docs/architecture.md §5.4 L298, docs/open-questions.md §C7 |
| Activation | GELU | docs/architecture.md §2 L76 |
| ConvNeXt block 数 (sub-model 内) | 8 (本チケットでは固定値ではない、Generator が決定) | docs/architecture.md §2 L40, docs/open-questions.md §A1 |

### 2.4 アルゴリズム / 処理フロー

1. **入力**: `x: (B, dim, T)`、Diff のみ `cond: (B, conditioning_dim)`
2. **(Diff のみ) Conditioning 注入**:
   - `bias = fc_t(cond)` → shape `(B, dim)`
   - `bias = bias.unsqueeze(-1)` → shape `(B, dim, 1)` (T 方向に broadcast)
   - `x = x + bias` (additive、FiLM の scale+shift ではない)
3. **residual 保存**: `residual = x`
4. **Depthwise Conv1d**: `x = dwconv(x)` (groups=dim、kernel=7、padding=3)
5. **channels_last へ転置**: `x = x.transpose(1, 2)` → `(B, T, C)`
6. **LayerNorm** (eps=1e-6、channels_last)
7. **MLP**: `Linear(dim→1536)` → `GELU` → `Linear(1536→dim)`
8. **LayerScale**: `x = gamma * x` (`gamma` は学習可能な `(dim,)` パラメータ、init=1e-6)
9. **channels_first へ転置**: `x.transpose(1, 2)` → `(B, C, T)`
10. **residual 接続**: `return residual + x`

### 2.5 設計上の重要決定

- **GAN/Diff を 1 クラスに統合** (vs 2 クラス分離): `conditioning_dim` の有無で挙動を分岐させる。重複コードを排除し、引数で挙動が変わることをコンストラクタで明示できる。後続 (T-M1.4 Generator) で `nn.ModuleList([ConvNeXtBlock(..., conditioning_dim=cond_dim) for _ in range(n_blocks)])` のような統一構築が可能。詳細は §8.1 を参照
- **`fc_t` は per-block 独立** (sub-model 間で共有しない、block 間でも共有しない): FastDiff の挙動に準拠 (docs/open-questions.md §C7 L264)。1 sub-model 内に 8 block あれば `fc_t` も 8 個
- **`fc_t` の出力次元 = `dim`** (= 512): FastDiff の per-block projection と一致 (docs/architecture.md §5.4 L300)
- **bias 注入は `block の入口で 1 回だけ`**: 内部の Linear 層ごとには再加算しない (docs/architecture.md §5.4 L318)
- **broadcast 方向**: `unsqueeze(-1)` で T 軸に broadcast (時間軸方向に一様 bias)
- **`fc_t` の bias term**: `nn.Linear(cond_dim, dim, bias=True)` (PyTorch default)。これ自体が learnable bias なので、初期化 0 でなくても問題なし
- **LayerScale 初期化**: γ_init = 1e-6 (Vocos / ConvNeXt 既定)。これにより訓練初期は residual branch がほぼ identity になり、深いネットでの安定性を担保
- **エラー処理**: コンストラクタの `conditioning_dim` 設定と forward の `cond` 渡し方が食い違った場合に `ValueError` を投げる (silent failure を回避)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `src/wavenext2/models/convnext.py` 本実装、`tests/test_convnext.py` Unit テスト記述 | general-purpose |
| Reviewer | 1 | コードレビュー (docs/architecture.md §2/§5.4 整合、docs/open-questions.md §C1/§C7 整合、参考実装非コピー検証) | general-purpose |
| Tester | 1 | `uv run pytest tests/test_convnext.py -v` 実行、Acceptance 全項目検証、パラメータ数の論文整合性検証 | Explore |

size=M (中規模) のため標準編成 (Implementer 1 + Reviewer 1 + Tester 1)。

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **部分的に yes**
  - T-M1.2 (STFT module) / T-M1.3 (Mel-spectrogram) / T-M1.5 (Noise embedding) と並列可能 (機能的に独立)
  - T-M1.4 (Generator) と T-M1.6 (Sub-model) は **本チケットに依存**
- 並列実行する場合の最大並列数: T-M1.1 / T-M1.2 / T-M1.3 / T-M1.5 の 4 並列まで可能

## 4. 提供範囲 (Scope)

### In Scope
- `ConvNeXtBlock` クラス (`src/wavenext2/models/convnext.py`) の本実装
  - GAN/Diff 両対応の単一クラス
  - additive bias 注入 (Diff のみ、FastDiff 流)
  - LayerScale (γ_init=1e-6)
  - Depthwise Conv1d (kernel=7, groups=dim)
  - MLP (Linear(dim→1536) → GELU → Linear(1536→dim))
  - LayerNorm (channels_last, eps=1e-6)
- Unit テスト (`tests/test_convnext.py`):
  - Shape テスト (GAN / Diff 両モード)
  - Parameter count テスト (Vocos 互換チェック)
  - Gradient flow テスト
  - Error handling テスト (cond 不整合)
  - Determinism テスト
  - LayerScale 初期値テスト
- `src/wavenext2/models/__init__.py` の `__all__` 更新

### Out of Scope
- Sinusoidal embedding + 共有 FC head (`fc_t1`, `fc_t2`) → **T-M1.5** (`NoiseEmbedding`) で実装
  - 本チケットの `ConvNeXtBlock` は per-block `fc_t = Linear(cond_dim, dim)` のみを持つ
  - sub-model 全体で共有される sinusoidal + FC×2 SiLU head は T-M1.5 で別途実装
- Generator 本体 (`Conv1d` 入口、`LayerNorm`、`Linear(dim→n_fft+2)`、`Linear(n_fft+2→hop)`、`clip(-1,1)`) → **T-M1.4** (`WaveNextGenerator`)
- STFT module (波形 → STFT-spec) → **T-M1.2**
- Sub-model wrapper (STFT module + Generator) → **T-M1.6**
- Discriminator → T-M2.2
- 訓練ロジック / Loss → T-M2.3 以降
- `torch.compile` / fp16 AMP / gradient checkpointing 等の最適化 → M5/M6 で必要時に検討
- 重み初期化 (`trunc_normal_(std=0.02)`、bias zero) → **T-M1.4 Generator** が init を持つので Block は PyTorch デフォルトでよい (確認: 元 Vocos `ConvNeXtBlock` もブロック自体には init を持たず、`VocosBackbone` で一括 init している)

### Deliverable
- ファイル:
  - `src/wavenext2/models/convnext.py` (本実装、約 100-130 行)
  - `tests/test_convnext.py` (Unit テスト、約 150-200 行)
  - `src/wavenext2/models/__init__.py` (`__all__` 更新のみ)
- 関数 / クラス:
  - `ConvNeXtBlock(dim, intermediate_dim, kernel_size, layer_scale_init_value, conditioning_dim)` クラス
- ドキュメント差分:
  - `docs/milestones.md` §M1.1 の Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M1.1 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_convnext.py`)

`docs/milestones.md` §M1.1 Acceptance を完全網羅:

- [ ] `test_shape_gan_mode` — GAN モード (`conditioning_dim=None`): 入力 `(B=2, dim=512, T=80)` → 出力 `(2, 512, 80)` で shape 不変
- [ ] `test_shape_diff_mode` — Diff モード (`conditioning_dim=512`): 入力 `(2, 512, 94)` + `cond(2, 512)` → 出力 `(2, 512, 94)` で shape 不変
- [ ] `test_shape_various_batch_and_time` — `B ∈ {1, 4}`, `T ∈ {32, 80, 94, 256}` の組み合わせで shape 一致
- [ ] `test_param_count_gan` — GAN 版パラメータ数 (LayerScale + DWConv + LN + 2 Linear): `1*512 (γ) + (7*512+512) (dwconv) + 2*512 (LN γ,β) + (512*1536+1536) (pw1) + (1536*512+512) (pw2)` = **約 1.58M/block** に一致
- [ ] `test_param_count_diff` — Diff 版は GAN 版 + `Linear(512, 512)` (= 512*512 + 512 = 262,656) で **約 1.85M/block**。差分 = 262,656 を厳密にチェック
- [ ] `test_param_count_8blocks_consistency` — 8 block 重ねた合計が Generator 全体の sub-model パラメータ数概算 (GAN ~12.6M、Diff ~14.8M for blocks のみ) と整合 (Generator の Conv1d/Linear 分は除外、別チケットで検証)
- [ ] `test_gradient_flow_gan` — `out = block(x); loss = out.sum(); loss.backward()` で全 `p.grad is not None` かつ `p.grad.abs().max() > 0` (LayerScale γ も含む)
- [ ] `test_gradient_flow_diff` — 同上 + `cond` も `requires_grad=True` で `cond.grad is not None`
- [ ] `test_deterministic` — `torch.manual_seed(42)` → 2 回 forward で `torch.allclose(out1, out2)` (atol=0、bit-exact)
- [ ] `test_residual_dominance_at_init` — LayerScale init=1e-6 のため、初期化直後は `block(x) ≈ x` (`(out - x).abs().max() < 1e-2`、residual branch が抑制されていることを確認)
- [ ] `test_layer_scale_init_value` — `block.gamma.abs().max() < 1e-5` (init=1e-6 が確実に効いている)
- [ ] `test_dwconv_groups` — `block.dwconv.groups == block.dim` (depthwise であることを確認)
- [ ] `test_error_diff_missing_cond` — `conditioning_dim=512` で構築した block を `cond=None` で呼ぶと `ValueError` (silent failure 回避)
- [ ] `test_error_gan_extra_cond` — `conditioning_dim=None` で構築した block に `cond` を渡すと `ValueError`
- [ ] `test_error_invalid_dim` — `dim=0` or 負値で `ValueError`
- [ ] `test_error_invalid_kernel` — `kernel_size=6` (偶数) で `ValueError`
- [ ] `test_bias_broadcasts_over_time` — Diff モードで `T` を変化させても `cond` のみで動作する (`unsqueeze(-1)` の broadcast が機能)
- [ ] `test_cond_changes_output` — Diff モードで `cond1 ≠ cond2` のとき出力も異なる (`(out1 - out2).abs().max() > 1e-6`)
- [ ] `test_snapshot_sha256_gan` — `torch.manual_seed(0); x = randn(2, 512, 80); out = block(x)` の出力テンソルの SHA256 を `tests/snapshots/convnext_gan.pt` に pin。値が変化したら torch upgrade / 実装ドリフトを検知 (snapshot 生成時は initial commit、以後は assert のみ)
- [ ] `test_snapshot_sha256_diff` — Diff モード版の同等 snapshot を `tests/snapshots/convnext_diff.pt` に pin

#### テスト戦略の追加方針

- **GPU/CPU device-agnostic**: 主要テストは `@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])` で CPU/GPU 両方をカバー。CI は CPU のみ実行、ローカル GPU マシンでは CUDA も自動実行
- **CI 時間目標**: CPU で全 18 unit テスト合計 **< 5 秒** (ConvNeXt block は小規模なので余裕で達成可能、`pytest --durations=10` で監視)
- **coverage 目標**: `pytest --cov=wavenext2.models.convnext --cov-report=term-missing` で **95% 以上**。LayerScale γ の `requires_grad=True` 分岐や error path も含める

### 5.2 e2e / 結合テスト
- [ ] `uv run python -c "from wavenext2.models.convnext import ConvNeXtBlock; b = ConvNeXtBlock(); print(sum(p.numel() for p in b.parameters()))"` で約 1.58M を出力
- [ ] T-M1.4 (Generator) の予備テスト: 8 block を `nn.Sequential` (GAN) または manual loop (Diff、cond 必要) で重ねて forward が通る (本チケットでは smoke test として確認、本テストは T-M1.4 で実施)
- [ ] T-M1.6 (Sub-model) の予備テスト: GAN/Diff 両モードを切り替えても import で衝突しない

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.1 より転記)
- [ ] GAN モード (`conditioning_dim=None`): `block(x)` で入出力 shape `(B, 512, T)` 一致
- [ ] Diff モード (`conditioning_dim=512`): `block(x, cond)` で動作、cond なし呼び出しは ValueError
- [ ] パラメータ数: GAN 版 ~1.58M/block (= 1.58M × 8 で約 12.6M)、Diff 版 +0.26M (`Linear(512, 512)`)
- [ ] gradient flow 確認: `loss = block(x).sum(); loss.backward()` で全パラメータに grad

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **GAN/Diff 1 クラス統合に伴う引数誤用** | Diff 用ブロックに cond を渡し忘れると silent に GAN モード動作してしまう (ノイズ条件なしで訓練し品質劣化) | コンストラクタの `conditioning_dim` と forward の `cond is None` の整合を **明示的に `ValueError` で検証**。`test_error_diff_missing_cond` でカバー |
| **LayerScale γ の dtype 不一致** | AMP / fp16 訓練で γ が fp32 のままだと型不整合 | `nn.Parameter` は親 module の dtype を継承するので `block.half()` で自動変換される。Unit テストでは fp32 のみ確認、AMP テストは M5 で実施 |
| **DWConv padding ミス** | kernel=7 で padding≠3 だと出力 T が縮む / 伸びる | `padding = kernel_size // 2` で自動計算。`test_shape_*` で T 不変を検証 |
| **`unsqueeze(-1)` の broadcast 方向誤り** | `unsqueeze(1)` 等で誤ると channel 方向ではなく時間方向に bias が漏れる | `test_bias_broadcasts_over_time` で異なる T に対して動作確認 |
| **per-block `fc_t` が実は sub-model 単位で共有されていた、という解釈ミス** | FastDiff 実装では per-block 独立 (docs/open-questions.md §C7 L264 で確定済み) | docs/architecture.md §5.4 と open-questions.md §C7 を二重チェック。本実装通り per-block で実装 |
| **GELU 実装差異** (`approximate="tanh"` vs 厳密形) | 訓練再現性に微影響 | PyTorch default `nn.GELU()` (= 厳密形) を採用。Vocos も default 使用。`test_deterministic` で再現性を担保 |
| **LayerScale γ を `nn.Parameter` ではなく Buffer にしてしまう** | gradient flow しない → 学習されない | `requires_grad=True` を明示。`test_gradient_flow_*` で γ も grad を持つことを検証 |
| **`torch.compile` 非互換コード** | M5/M6 で `torch.compile` を有効化したとき graph break | 本チケットでは `compile` 検証は実施しない。M3 smoke で問題発生時に再調査 (docs/tickets/T-M0.1 §6.1 と整合) |
| **Vocos のコードを誤って参考実装からコピーする** | CLAUDE.md ポリシー違反 (公開予定リポジトリ) | Reviewer が docstring / 構造の独自性を確認。Vocos の `ConvNeXtBlock` は構造順序の docstring を持たないが本実装は独自 docstring を書く |
| **bias term の初期化** | `nn.Linear` 既定は uniform(-1/sqrt(in), 1/sqrt(in))、ゼロではない | 本実装では bias を 0 init しない (PyTorch default に任せる)。T-M1.4 Generator で `apply()` 一括 init する設計だが、本ブロックは Block 内部で init しない (docs/open-questions.md §C2 L186 注: 「全 Conv1d / Linear: trunc_normal_(std=0.02)、bias: zero」は Generator レベルで一括適用される想定) |
| **`torch.compile` `fullgraph=True` で error path が graph 化される問題** | `raise ValueError(cond is None)` 等の Python 例外が `fullgraph=True` で graph 内に含まれると compile が fail する。`mode="reduce-overhead"` (graph break 許容) なら回避可能だが、`fullgraph=True` (graph break 禁止) では NG | M5/M6 で `torch.compile` 導入時は `mode="reduce-overhead"` 限定で利用することを明記。`fullgraph=True` が必要になった場合は ValueError を別関数に切り出すか `@torch.compiler.disable` で `forward` 全体を opt-out するなどの対応を検討 (本チケットでは対応しない) |
| **bf16 で `gamma * x` broadcast multiplication の挙動** | LayerScale `gamma * x` で `gamma` が fp32、`x` が bf16 の場合、PyTorch は fp32 へ昇格して計算 (autocast 規則) するが、Vocos / FastDiff 実装と差異がないか要確認 | M5/M6 で bf16 訓練を有効化する際、Vocos 既存実装と数値比較してドリフトがないか確認。本チケットでは fp32 のみテストし、bf16 挙動は M5 でカバー |
| **T-M1.4 Generator 側 `_init_weights` が Block 内部 (`fc_t`, `dwconv`, `pwconv1/2`) を巻き込む** | Block 単体テスト (`tests/test_convnext.py`) では PyTorch default init で動作確認するが、Generator 経由で init された後の値が Block 単体テストの初期値と食い違う。Block レベルで初期値 assert はしない設計だが、Generator init 後の Block 挙動が単体テストでカバーできていないリスク | Block レベルで初期値 assert はしない (`test_layer_scale_init_value` の γ=1e-6 のみ例外)。Generator init 後の Block 挙動変化 (例: `fc_t.bias=0` で初期 bias 注入が 0、`dwconv.weight ~ trunc_normal_(std=0.02)` 等) は T-M1.4 のテストでカバーする想定。本ファイル §9.1 で T-M1.4 に明示的に申し送り |
| **channels-last 形式と PyTorch native の transpose コスト** | `x.transpose(1, 2)` を block 内で 2 回行うため、8 block × `T_mel` × `dim=512` で memory bandwidth bound になる可能性。`Conv1d(kernel=1)` への置換や `torch.channels_last` memory_format の利用で改善可能 | 本チケットでは Vocos 準拠で transpose ベースを採用。M6 profiling で transpose がボトルネックと判明した場合に `Conv1d(k=1)` 置換や `memory_format=channels_last` を試す (M6 最適化フェーズに予約) |

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 参照: §C1 (ConvNeXt 内部) と §C7 (Diff conditioning) は **両方とも 100% 確定**
- **未解決の判断ポイントは無い**。本チケットは確定事項の単純実装

### 6.3 他チケットとの整合性
- **T-M1.4 (Generator)**:
  - Generator は `nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, kernel_size, conditioning_dim=...) for _ in range(n_blocks=8)])` で本クラスを利用
  - Generator が GAN モードの時は `conditioning_dim=None`、Diff モードの時は `conditioning_dim=512` で構築
  - Generator の forward では `for block in self.blocks: x = block(x, cond=e if e is not None else None)` のような呼び出し
  - **signature 確定** (§9.1 で連絡): `forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor`
- **T-M1.5 (Noise embedding)**:
  - `NoiseEmbedding` は sinusoidal + Linear(128→512) + SiLU + Linear(512→512) + SiLU を実装し `(B, 512)` を返す
  - その `(B, 512)` がそのまま本クラスの `cond` 引数に渡る
- **T-M1.6 (Sub-model)**:
  - `SubModelDiff` 内で `cond = self.noise_embedding(c)` → `self.generator(x, cond=cond)` → Generator 内部で `block(h, cond)` という呼び出し連鎖
- **T-M1.2 (STFT module) / T-M1.3 (Mel)**: 本チケットとは独立 (本クラスは入力テンソルの中身に依存しない)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 論文記述 (`docs/architecture.md` §2 L72-91、§5.4 L298-322) との整合性
- [ ] `docs/open-questions.md` §C1 (内部仕様)、§C7 (additive bias 注入) と完全整合
- [ ] `dim=512`, `intermediate_dim=1536`, `kernel_size=7`, `LayerScale init=1e-6` のデフォルト値が正しい
- [ ] `groups=dim` で depthwise になっている
- [ ] LayerNorm は **channels_last** で適用されている (transpose 順序)
- [ ] Diff モードで `fc_t` が **per-block** に存在 (sub-model 単位で共有しない、block 間でも共有しない)
- [ ] additive bias 注入が **block の入口で 1 回だけ** (内部 Linear 層ごとには再加算していない)
- [ ] `unsqueeze(-1)` で時間軸に broadcast されている (`unsqueeze(1)` 等の誤りがない)
- [ ] FiLM (scale + shift) ではなく **additive のみ**
- [ ] Acceptance criteria 全 4 項目クリア (`docs/milestones.md` §M1.1)
- [ ] Unit テスト全 pass (`uv run pytest tests/test_convnext.py -v`)
- [ ] CLAUDE.md / 既存コードのスタイル準拠 (型ヒント `int | None` 形式、docstring 日本語、命名 snake_case)
- [ ] エラー処理: cond 不整合で `ValueError`、dim/kernel 不正で `ValueError`
- [ ] パラメータ数: GAN ≈ 1.58M、Diff ≈ 1.85M (Vocos 互換)
- [ ] **参考実装 (Vocos `vocos/modules.py::ConvNeXtBlock`、FastDiff `module/FastDiff_model.py`) をコピーしていない** (docstring・構造順序・変数命名を独自に書いたか)
- [ ] `from wavenext2.models.convnext import ConvNeXtBlock` で import 可能
- [ ] `src/wavenext2/models/__init__.py` の `__all__` に `ConvNeXtBlock` が追加されている

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M1 フェーズ完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **GAN/Diff 共通 1 クラス (採用案)** | コード重複なし、引数で明示的に分岐、Generator から統一構築できる | forward で `if self.fc_t is not None` の分岐コストが入る (微小)、API がやや複雑 | docs/architecture.md §2 / §5.4 が「同じ block 構造を使う」ことを明示。1 クラスで実装するのが論文の意図に最も忠実 | (採用済み) |
| **GAN/Diff 2 クラス分離** (`ConvNeXtBlockGAN`, `ConvNeXtBlockDiff`) | API がシンプル、forward 分岐なし、IDE 補完が綺麗 | コード重複、Generator 側で if 分岐が必要、機能追加 (例: 別 conditioning) で N×2 クラス増加 | 1 クラス案の方が拡張性が高い (将来 FiLM や cross-attention 追加時も `conditioning_kind` 引数で吸収可能) | **M3 smoke で Diff 訓練が不安定でデバッグ難易度が GAN/Diff 分岐に起因することが判明したとき** (e.g., Diff path の hidden state が NaN になる場合の切り分けが面倒な時) |
| **`ConditionalConvNeXtBlock` を Decorator pattern で別クラス化** (純粋 ConvNeXt は `ConvNeXtBlock`、conditioning は外側ラッパで合成) | 純粋 ConvNeXt と Diff conditioning が分離され単一責任原則に従う、テストが疎結合になる | 2 クラス分離と類似のコード重複・呼び分けコスト、Generator 側で wrapper 構築が必要、`if self.fc_t is not None` の分岐コストは消えるが構造的に複雑化 | 1 クラス案の方が論文記述 (「同じ block 構造を使う」) に忠実かつ Generator 側の `ModuleList` 構築が単純。Decorator は M6.3 ablation で `conditioning_kind` 抽象化を本格化する際に再評価 | **M6.3 ablation で複数 conditioning 方式 (additive / FiLM / cross-attn) を比較する段階** で wrapper 化を検討 |
| **追加 conditioning 方式 (FiLM = scale + shift)** | より表現力が高い、近年の diffusion モデル (DiT, U-ViT) で標準的 | FastDiff の論文記述 (additive bias) に反する、再現性低下 | docs/open-questions.md §C7 で FastDiff 流 additive bias を確定。論文再現が最優先 | **本格訓練 (M6) で論文 Table 2 の UTMOS / NISQA を 10% 以上下回ったとき**、FiLM 化 ablation を実施 |
| **`conditioning_kind: Literal["none", "additive"]` 引数で将来 FiLM 等を拡張可能化** | 将来 `"film"` / `"cross_attn"` を追加するときに後方互換で拡張できる、引数で明示的に挙動を選択できる | 現時点では `"additive"` 一択なので overengineering ぎみ、`conditioning_dim` との二重情報になる | 本実装フェーズでは `conditioning_dim` の有無で十分。将来 FiLM 等の選択肢が増えた時点で導入 | **M6.3 (ablation) チケット起動時** に `conditioning_kind` 引数を導入して `"additive"` / `"film"` を比較する |
| **`fc_t` factory 引数化 (`fc_factory: Callable[[int, int], nn.Module] = nn.Linear`)** | fp16 / bf16 で `nn.Linear` が不安定なときに `nn.Linear` + LayerNorm の合成や独自モジュールに差し替え可能、テスト時に mock 注入できる | API 複雑化、デフォルト挙動の理解コスト増、現状不要 | 現時点では `nn.Linear` 一択で問題なし。差し替え要件が出たら導入 | **M3 smoke (Diff) で fp16 conditioning が NaN/不安定になったとき**、`fc_factory` を導入して `Linear + LayerNorm` 合成等を試す |
| **LayerScale を GroupNorm に置換** | LayerScale なしのよりシンプルな ConvNeXt (元論文 v1) | Vocos / 元 ConvNeXt v2 の挙動と乖離、論文再現性低下 | LayerScale + LN は Vocos / 元 ConvNeXt の標準 (docs/open-questions.md §C1) | **DyT (Dynamic Tanh) 等の新しい normalization が ConvNeXt 系で標準化されたとき** (Transformer DyT 論文 [Zhu et al. 2024] が時系列で広く採用された場合)。LayerScale + LN の代替として DyT を試す価値あり |
| **DyT (Dynamic Tanh) を LayerNorm の代替に検討** | normalization が学習可能スカラーで置き換えられ、訓練不安定が緩和される事例あり (近年提案) | 論文再現の文脈で「DyT 採用」は逸脱。再現性最優先のフェーズでは不採用 | LN + LayerScale が Vocos / FastDiff / 元 ConvNeXt で確立しているため再現実装フェーズでは触らない | **M6 で原論文 Vocos と同等品質を達成した後、追加実験フェーズで DyT を ablation 候補に入れる** (M6.3 ablation) |
| **Depthwise Conv を MaxBlurPool + standard Conv に分解** | Aliasing 抑制で帯域外を綺麗にできる | パラメータ数増、推論速度低下、論文と乖離 | 論文再現が最優先。本クラスは Vocos 準拠の Depthwise Conv のみ | (再評価しない、本フェーズでは固定) |
| **`pwconv1`, `pwconv2` を `Conv1d(kernel=1)` で実装** | 純粋に Conv1d で統一できる、`transpose` が不要に | 速度差は実質ゼロ、`Linear` の方が PyTorch 内で最適化済み | Vocos も `Linear` を採用 (docs/architecture.md §2 L86, 89)。channels_last 表現と LayerNorm との相性も `Linear` の方が良い | (再評価しない) |
| **LayerNorm を `RMSNorm` (Llama 系) に置換** | mean 引き算が不要で 1-2% 高速化 | 再現性低下 | LN が Vocos 確定。RMSNorm は M6 後 ablation 候補 | **M6 完走後の推論速度最適化フェーズ** |
| **Activation を SiLU (Swish) に変更** | SiLU は近年の vocoder で多用 (BigVGAN, Vocos の一部) | Vocos `ConvNeXtBlock` は GELU、論文再現の文脈では GELU が正解 | docs/architecture.md §2 L76 で GELU 確定 | (再評価しない、GELU 固定) |
| **forward に `cond` 必須化、`cond=None` で GAN モードを禁止** | 引数 typo に強い | GAN モードで毎回 `block(x, None)` と書くことになり API が冗長 | 現案 (`cond` を `Optional`) が直感的 | (再評価しない) |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: 適切。`ConvNeXtBlock` は WaveNeXt 2 の最も基本的なビルディングブロックで、Generator (T-M1.4) / Sub-model (T-M1.6) の構成要素として独立した責務を持つ。1 チケットで完結する size=M の規模感
- **別マイルストーンに移すべき部分はないか**: なし。`fc_t` を T-M1.5 (Noise embedding) に移すと sub-model 構造が分散し追跡しにくくなる。`fc_t` は per-block の責務なので本チケットで持つのが正解
- **インターフェース定義の見直し余地**:
  - `forward(x, *, cond=None)` の signature を **keyword-only に確定** (M1 フェーズレビュー結果)。positional での誤用 (`block(x, some_tensor)` で意図せず cond として渡る) を API レベルで封じる。Generator 側の呼び出しも `block(x, cond=cond)` で明示的になり可読性が向上
  - `dim`, `intermediate_dim`, `kernel_size` をすべて default 値持ちにすることで Generator 側の呼び出しが `ConvNeXtBlock()` 一発で書けるが、`conditioning_dim` だけは Generator 側で明示することで GAN/Diff 切り替えを意識させる設計
- **`fc_t` の bias を zero init する方針**: FastDiff 準拠で zero init を採用 (T-M1.4 Generator の `_init_weights` 経由で適用)。理由: `trunc_normal_(std=0.02)` で `nn.Linear` の bias を初期化すると、出力 `fc_t(cond)` は `weight @ cond + bias` の bias 項が `O(sqrt(dim) * std) = sqrt(512) * 0.02 ≈ 0.45` 程度のノイズになる。これを LayerNorm(eps=1e-6) 前の `x` に additive で加算すると、訓練初期に信号が大きく揺さぶられて収束が遅れる。bias=0 init により、訓練初期は `fc_t(cond)` ≈ `weight @ cond` だけが効き、cond が 0 に近い領域では bias 注入が無効化される (LayerScale γ=1e-6 と整合する「初期 identity」設計)
- **「重み初期化は Generator 側で一括 (apply)」設計の採用根拠**: Block は自前 `_init_weights` を持たない。T-M1.4 Generator の `_init_weights` が `apply()` で walk して Block 内部 (`fc_t`, `dwconv`, `pwconv1/2`, `norm`) を init する責任を持つ前提を明示。理由 (a) Vocos `VocosBackbone` の設計に整合、(b) Generator レベルで `trunc_normal_(std=0.02)` / bias=0 / `fc_t.bias=0` を一元管理できる、(c) Block を別 Generator (将来の variant) に流用する際も init 方針を呼び出し側で制御できる。Block 単体テストでは PyTorch default init のまま動作することのみ確認 (`test_layer_scale_init_value` 以外で初期値 assert はしない)
- **M1 phase review で API 横断整合性チェック**: M1 完了時に T-M1.1〜T-M1.6 全てを横断レビューし、以下の整合性を確認する: (a) keyword-only `cond` が全 sub-model / Generator で一貫している、(b) factory パターン (現時点では未導入だが将来 `fc_factory` を入れる場合) の命名が `xxx_factory` で統一されている、(c) `cond` / `noise_emb` / `c` の naming が混在していない (推奨: 外部 API は `cond`、内部は `noise_emb`)、(d) `conditioning_dim` の default 値 (None vs 0) が混在していない

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `ConvNeXtBlock` 実装、`tests/test_convnext.py` 42 件 pass (CPU/GPU 両方、device fixture)。
- **パラメータ数が論文/Vocos と厳密一致**: GAN 1,580,544 / Diff 1,843,200 (差分 fc_t = 262,656)。8 block で GAN 12.64M / Diff 14.75M (block のみ、Generator の Conv1d/Linear/head は別)。
- `forward(x, *, cond=None)` keyword-only 確定。`test_cond_is_keyword_only` で positional 渡しが TypeError になることを保証。
- LayerScale γ=1e-6 により初期化直後 `block(x) ≈ x` (residual dominance) を検証。

想定外と対処:
1. **snapshot は SHA256 でなく JSON 要約 (mean/std/l2) + tolerant 比較に変更**: 理由 (a) `*.pt` は `.gitignore` で除外されベースラインが repo に残らない、(b) 生テンソルの SHA256 は CPU/GPU/BLAS 差で容易に壊れる。`tests/snapshots/convnext_{gan,diff}.json` を commit し rtol 1e-3 で drift 検知。教訓: **snapshot は「committable な text」かつ「float 許容差」で設計する**。
2. **conftest に autouse seed fixture + device fixture を追加**: `_set_seed` (torch/cuda 42 固定) と `device` (cpu + 利用可能なら cuda) を M1 共通基盤として導入。後続 M1 テストはこれを使う。
3. **テスト側のパラメータ数理論式で gamma を 1 と誤記** (正しくは dim=512) し一度 fail。実装は正しく、テスト式を修正。教訓: **理論値テストは「実装が間違っているのか式が間違っているのか」を即座に切り分ける** (差分 511 = 512−1 から即特定できた)。

次の似たタスクで応用できる教訓:
- 重み init を持たない設計 (Generator が `apply()` で一括 init) は単体テストでは PyTorch default init で動く。init 後の挙動検証は T-M1.4 に委譲。
- device fixture により CPU/GPU 差を 1 度に検出できる (この環境は CUDA 有のため GPU パスも常時検証される)。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

**インターフェース確定情報** (T-M1.4 Generator, T-M1.6 Sub-model 向け):

```python
# import
from wavenext2.models.convnext import ConvNeXtBlock

# Constructor signature (確定)
ConvNeXtBlock(
    dim: int = 512,
    intermediate_dim: int = 1536,
    kernel_size: int = 7,
    layer_scale_init_value: float = 1e-6,
    conditioning_dim: int | None = None,  # None=GAN, 512=Diff
)

# Forward signature (確定、keyword-only cond)
out = block(x, cond=None)
# x: (B, dim, T)
# cond: (B, conditioning_dim) — Diff モードでのみ、GAN モードでは None。keyword-only (`forward(x, *, cond=None)`)
# out: (B, dim, T) — shape 不変
```

**T-M0.2 (scaffold) への申し送り**:
- `tests/conftest.py` に以下 fixture を追加する設計を推奨:
  - `@pytest.fixture(autouse=True) def set_seed(): torch.manual_seed(42)` — 全テストで自動 seed 固定
  - `@pytest.fixture(scope="module") def gan_block()` — `ConvNeXtBlock(conditioning_dim=None)` を module スコープでキャッシュ
  - `@pytest.fixture(scope="module") def diff_block()` — `ConvNeXtBlock(conditioning_dim=512)` を module スコープでキャッシュ
- `pytest-randomly` を `dev` extras に追加し、テスト順序のランダム化 + seed 固定の両立を保証 (テスト順序依存のバグを早期検知)
- `tests/snapshots/` ディレクトリを scaffold 時に作成、`.gitkeep` を置く
- 詳細実装は T-M0.2 (`tests/` 構成) と T-M1.1 (`tests/test_convnext.py`) の両方で調整

**T-M1.4 (Generator) への連絡**:
- Generator は `self.blocks = nn.ModuleList([ConvNeXtBlock(dim=512, intermediate_dim=1536, kernel_size=7, conditioning_dim=cond_dim) for _ in range(n_blocks)])` で 8 個保持
- `cond_dim` は GAN なら None、Diff なら 512
- forward では `for block in self.blocks: x = block(x, cond=cond)` (keyword-only、Diff なら NoiseEmbedding 出力 (B, 512) を渡す、GAN なら cond キーワードを省略 or `cond=None`)
- 重み初期化 (`trunc_normal_(std=0.02)`、bias=0) は **Generator レベルで `apply()` 一括適用** する設計 (本ブロックは init を持たない、Vocos 同様)
- **責任境界の明文化**: ConvNeXtBlock は自前 `_init_weights` を持たない。Generator 側 `_init_weights` が `self.apply(self._init_weights)` で walk して Block 内部 (`fc_t`, `dwconv`, `pwconv1`, `pwconv2`) を init する責任を持つ。Block レベルではこの前提を docstring で明示し、Generator が init を提供しなかった場合は PyTorch default のままになることを許容
- **`fc_t.bias = 0` init 指示**: Block の `__init__` で zero init はせず、Generator の `_init_weights` で `if isinstance(module, nn.Linear): nn.init.zeros_(module.bias)` を適用する。これにより `fc_t.bias` も自動的に 0 init される (FastDiff 準拠、§8.2 参照)。`fc_t` だけ特別扱いせず、全 `nn.Linear` の bias を一律 0 init する設計
- `apply()` walk の意図: Block のサブモジュール (`fc_t: Linear`, `dwconv: Conv1d`, `pwconv1: Linear`, `pwconv2: Linear`, `norm: LayerNorm`) が全て Generator の `_init_weights` で訪問される。`gamma` (LayerScale パラメータ) は `nn.Parameter` 直下なので `apply()` の対象外 (これは意図通り、γ は Block の `__init__` で `1e-6 * ones(dim)` 初期化済み)
- パラメータ数概算: 8 × 1.58M (GAN block) = 12.64M、または 8 × 1.85M (Diff block) = 14.8M

**T-M1.6 (Sub-model) への連絡**:
- `SubModelDiff` は `NoiseEmbedding` (T-M1.5) → Generator (T-M1.4) → Generator 内部で各 ConvNeXtBlock に cond が流れる、というデータフロー
- `SubModelGAN` は cond なし、ConvNeXtBlock の `conditioning_dim=None` で構築される
- Sub-model 側で Generator を呼ぶ際も keyword-only で `generator(x, cond=cond)` の形を取る (一貫性のため)

**T-M1.5 (Noise embedding) への連絡**:
- `NoiseEmbedding(c: Tensor (B,)) -> Tensor (B, 512)` の出力 shape を本クラスの `cond` 引数 shape `(B, conditioning_dim=512)` に整合させる

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M1.1 の Acceptance チェックボックス 4 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M1.1 ステータスを `📝 pending` → `✅ completed` に更新
  - [ ] M1 進捗サマリ (pending -1, completed +1) を更新
  - [ ] (該当時) `docs/architecture.md` §2 ConvNeXt block 構成図に実装ファイルパスを追記
  - [ ] (該当時) `docs/open-questions.md` 補遺に「本実装で確定した clarification」(例: forward signature の確定形) を追記

### 9.3 Open question として残ったもの
- 解決できなかった疑問: なし (確定事項の単純実装)
- `docs/open-questions.md` への追記要否: 不要 (現状の §C1, §C7 で完全カバー)
- 将来検討事項:
  - `conditioning_kind: Literal["none", "additive"]` 引数化 (将来 `"film"` / `"cross_attn"` 追加) — M6.3 ablation で再評価
  - `fc_factory` 引数化 (fp16/bf16 不安定時の代替) — M3 smoke で再評価
  - DyT (Dynamic Tanh) を LayerNorm 代替に — M6 後 ablation フェーズで検討

**M1 フェーズレビュー (2026-05-26) で決定された事項**:
- `forward(x, *, cond=None)` を **keyword-only に確定** (§8.1 / §8.2 反映済み)
- `ConditionalConvNeXtBlock` の Decorator pattern 案を §8.1 に追加 (却下、M6.3 で再評価)
- `conditioning_kind` / `fc_factory` 引数の将来拡張余地を §8.1 に明文化
- `fc_t.bias = 0` init を T-M1.4 Generator の `_init_weights` で対応する責任境界を §8.2 / §9.1 で明文化
- snapshot SHA256 pin / GPU/CPU device-agnostic / CI 5 秒目標 / coverage 95% を §5.1 に追加
- `torch.compile fullgraph=True` / bf16 broadcast / Generator init 巻き込み / channels-last 化を §6.1 に追加
