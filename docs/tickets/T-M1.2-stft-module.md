---
id: T-M1.2
title: STFT module (波形 → STFT-spec 2F-2 ch 変換)
milestone: M1
phase: M1
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M1.6]
related_docs:
  - docs/milestones.md#m12-stft-module-srcwavenext2modelsstftpy
  - docs/architecture.md
---

# T-M1.2: STFT module (波形 → STFT-spec 2F-2 ch 変換)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.2](../milestones.md#m12-stft-module-srcwavenext2modelsstftpy)
> **依存**: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M1.6](T-M1.6-sub-model.md)

## 1. タスク目的とゴール

### 目的
WaveNeXt 2 統一フレームワークの **要** である **STFT module** を実装する。前ステップ波形 (GAN では `y_{t-1}`、Diff では `x_t`) を STFT して **`2F-2` チャネルの STFT-spec** に変換し、mel-spectrogram (128 ch) と concat して generator に入力できるテンソル表現を生成する。論文 §3.1 の処理手順 (Hann window、center=True、time-axis truncation、DC/Nyquist 虚部削除) を忠実に再現する。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/models/stft.py` に `STFTModule(nn.Module)` クラスが実装され、`from wavenext2.models.stft import STFTModule` で import 可能
- [ ] GAN 設定 (`n_fft=2048, hop=300, win=1200`) で入力 `(B, T_mel*hop)` から `(B, 2046, T_mel)` を返す
- [ ] Diff 設定 (`n_fft=1024, hop=256, win=1024`) で入力 `(B, T_mel*hop)` から `(B, 1022, T_mel)` を返す
- [ ] DC/Nyquist 虚部 (常に 0) が削除されている (虚部の最初/最後 bin が落ちている)
- [ ] 時間軸 truncation により出力の `T` 次元が引数 `T_mel` と **完全一致**
- [ ] `tests/test_stft_module.py` の全テストが pass (`uv run pytest tests/test_stft_module.py -v`)
- [ ] `torch.stft` の `return_complex=True` モードを使用し、Vocos / FastDiff のコードを **コピーしていない**
- [ ] `docs/milestones.md` §M1.2 の Acceptance 5 項目が全て ☑
- [ ] `docs/tickets/index.md` の T-M1.2 ステータスが `completed` に更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - 実装本体は T-M0.2 で作成済みのスタブを実装で置き換え: `src/wavenext2/models/stft.py`
  - テストは T-M0.2 で作成済みの placeholder を実装で置き換え: `tests/test_stft_module.py`
- 編集:
  - (該当時) `docs/milestones.md` §M1.2 Acceptance チェックボックス
  - `docs/tickets/index.md` のステータス

### 2.2 主要構造

```python
# src/wavenext2/models/stft.py
import torch
import torch.nn as nn


class STFTModule(nn.Module):
    """波形を STFT して 2F-2 ch の STFT-spec を返すモジュール (論文 §3.1)。

    入力波形 y に Hann window で STFT (center=True, normalized=False, onesided=True)
    を適用し、複素 spectrogram を mel-spec の時間長 T_mel に truncate した上で、
    実部 F bin と虚部 F-2 bin (DC/Nyquist 削除) を concat した tensor を返す。

    Args:
        n_fft: STFT 点数。F = n_fft//2 + 1。
        hop_length: STFT hop。mel-spec の hop と同一であること。
        win_length: STFT 窓長。

    Forward I/O:
        y       : (B, T_audio)  ← T_audio = T_mel * hop_length が想定値
        T_mel   : int           ← mel-spec 側の時間長
        returns : (B, 2*F - 2, T_mel) = (B, n_fft, T_mel)
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        if win_length > n_fft:
            raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # window は学習しない buffer として登録 (device 追従用)
        self.register_buffer(
            "window", torch.hann_window(win_length), persistent=False
        )

    def forward(self, y: torch.Tensor, T_mel: int) -> torch.Tensor:
        if y.dim() != 2:
            raise ValueError(f"expected y: (B, T_audio), got {tuple(y.shape)}")
        if T_mel <= 0:
            raise ValueError(f"T_mel must be positive, got {T_mel}")

        # (B, F, T_stft) complex
        Y = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        # mel-spec の時間長に truncate (center=True の余剰フレームを落とす)
        if Y.shape[-1] < T_mel:
            raise RuntimeError(
                f"STFT produced {Y.shape[-1]} frames, but T_mel={T_mel} required. "
                f"check that T_audio >= T_mel * hop_length"
            )
        Y = Y[..., :T_mel]

        real = Y.real                          # (B, F, T_mel)
        imag = Y.imag[:, 1:-1, :]              # (B, F-2, T_mel) -- DC/Nyquist 虚部削除
        return torch.cat([real, imag], dim=1)  # (B, 2F-2, T_mel)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | GAN 値 | Diff 値 | 出典 |
|---|---|---|---|
| `n_fft` | 2048 | 1024 | docs/architecture.md §3 表 |
| `hop_length` | 300 | 256 | docs/architecture.md §3 表 |
| `win_length` | 1200 | 1024 | docs/architecture.md §3 表 |
| `window` | Hann | Hann | docs/architecture.md §3 (論文 §3.1) |
| `center` | True | True | docs/architecture.md §3 (論文 §3.1) |
| `normalized` | False | False | docs/architecture.md §3 |
| `onesided` | True | True | docs/architecture.md §3 |
| `return_complex` | True | True | PyTorch 推奨 API (実数 stack は deprecated) |
| 出力 ch (2F-2) | 2046 | 1022 | docs/architecture.md §3 |

### 2.4 アルゴリズム / 処理フロー

1. **コンストラクタ**:
   - `n_fft / hop_length / win_length` を保存
   - `torch.hann_window(win_length)` を **persistent=False の buffer** として登録 (`.to(device)` で window も自動で device 追従、ただし `state_dict` には保存しない)
   - `win_length > n_fft` の不正入力で `ValueError`
2. **Forward**:
   1. 入力 `y` の shape を `(B, T_audio)` で検証
   2. `T_mel > 0` を検証
   3. `torch.stft(y, ..., return_complex=True)` で複素 spectrogram `(B, F, T_stft)` を取得
   4. `T_stft < T_mel` ならば `RuntimeError` (入力波形の長さ不足、呼び出し側の bug を早期検知)
   5. `Y = Y[..., :T_mel]` で時間軸 truncation
   6. `real = Y.real` (shape `(B, F, T_mel)`)
   7. `imag = Y.imag[:, 1:-1, :]` (shape `(B, F-2, T_mel)`、DC bin 0 と Nyquist bin F-1 を削除)
   8. `torch.cat([real, imag], dim=1)` で `(B, 2F-2, T_mel)` を返す
3. **数値安全性**: `STFTModule` は学習パラメータを持たない (window は buffer)。dtype は入力 `y` を踏襲。fp16/bf16 入力時の精度低下は呼び出し側で fp32 にキャストする方針 (本モジュール内では強制キャストしない)。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `STFTModule` 実装 + `tests/test_stft_module.py` ユニットテスト記述 | general-purpose |
| Reviewer | 1 | 論文 §3.1 / docs/architecture.md §3 との整合性、参考実装 (Vocos / FastDiff) のコピー有無、`return_complex=True` のリスク確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_stft_module.py -v` 実行、round-trip 周波数集中検証、GAN/Diff 両設定で shape 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** — T-M1.1 (ConvNeXt block), T-M1.3 (mel), T-M1.5 (noise embedding) と並列実行可。これら 4 つはどれも T-M0.2 のみに依存し、互いに依存関係を持たない
- 並列実行する場合の最大並列数: 4 (T-M1.1, T-M1.2, T-M1.3, T-M1.5)
- T-M1.6 (sub-model) は本チケット完了後に着手

## 4. 提供範囲 (Scope)

### In Scope
- `STFTModule(nn.Module)` クラスの実装 (forward / `__init__`)
- Hann window を `register_buffer` で device 追従
- 時間軸 truncation の実装と境界チェック
- DC/Nyquist 虚部削除の実装
- GAN/Diff 両 config (n_fft=2048/1024) で動作するパラメータ化
- ユニットテスト 7 種 (§5.1 参照)
- docs/milestones.md §M1.2 / docs/tickets/index.md のステータス更新

### Out of Scope
- mel-spectrogram 抽出 (`src/wavenext2/data/mel.py`) → **T-M1.3** で実施
- STFT-spec と mel-spec の concat → **T-M1.6** (sub-model wrapper) で実施
- WaveFit のような **生波形をそのまま入力する** 経路 (本論文では使わない、生波形 ↔ STFT 比較 ablation も Out of Scope)
- 逆 STFT (iSTFT) — Vocos `ISTFTHead` 相当の出力側変換は **本論文では使わない** (Generator は時系列 hop_length を直接出すため)
- learnable STFT (1D conv で代替) → §8.1 代替案として記載のみ
- `torch.compile` 対応 (`return_complex=True` と compile の互換性は M3 smoke で検証)

### Deliverable
- ファイル:
  - `src/wavenext2/models/stft.py` (新規実装、T-M0.2 スタブを置換)
  - `tests/test_stft_module.py` (新規実装、T-M0.2 placeholder を置換)
- 関数 / クラス:
  - `class STFTModule(nn.Module)`
  - `STFTModule.__init__(self, n_fft, hop_length, win_length)`
  - `STFTModule.forward(self, y: Tensor, T_mel: int) -> Tensor`
- ドキュメント差分:
  - `docs/milestones.md` §M1.2 Acceptance チェックボックス 5 項目を ☑
  - `docs/tickets/index.md` の T-M1.2 ステータスを `📝 pending` → `✅ completed`

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_stft_module.py`)

- [ ] `test_shape_gan` — GAN 設定 `(n_fft=2048, hop=300, win=1200)`、`B=2, T_mel=80, T_audio=24000` で出力 shape `(2, 2046, 80)` を確認
- [ ] `test_shape_diff` — Diff 設定 `(n_fft=1024, hop=256, win=1024)`、`B=2, T_mel=94, T_audio=24064` で出力 shape `(2, 1022, 94)` を確認
- [ ] `test_truncation_exact` — `T_mel=80` 指定時の出力 T 次元が **完全に 80** (center=True で生まれる余剰フレームが落ちている)
- [ ] `test_dc_nyquist_imag_dropped` — 入力に対し `torch.stft` を内部と同条件で計算し、出力の **後半 F-2 = (n_fft-2) ch が `Y.imag[:, 1:-1, :]` と数値一致** することを `torch.allclose` で確認 (DC / Nyquist 虚部の削除位置の検証)
- [ ] `test_roundtrip_sine_440hz` — 24kHz の 440 Hz 正弦波 1 秒を入力し、GAN 設定での STFT-spec の **440 Hz 付近の実部 bin** にエネルギーが集中することを確認。具体的には:
  - 入力: `y = torch.sin(2π * 440 * t)`, `t = arange(24000) / 24000`
  - 期待 bin: `bin_440 = round(440 / 24000 * 2048) = 38`
  - 実部 magnitude `|real[:, 38, :]|.mean()` が `|real[:, [10, 100, 500, 1000], :]|.mean()` より少なくとも **20×** 大きい
- [ ] `test_deterministic` — 同入力 + 同 seed (window 固定) で 2 回 forward して `torch.equal` で同一性を確認
- [ ] `test_gradient_flow` — 入力 `y` を `requires_grad_()` 化、`stft_module(y, T_mel).sum().backward()` で `y.grad` が None ではなく有限値であることを確認
- [ ] `test_device_buffer` — `module.to("cuda")` (CUDA 利用可能時のみ skip しない) で `module.window.device.type == "cuda"` を確認 (`register_buffer` 動作検証)
- [ ] `test_dtype_preserve` — fp32 入力で出力 dtype が fp32、fp64 入力で fp64 を確認 (キャスト勝手にしないこと)
- [ ] `test_invalid_inputs` — 以下が想定通りエラーを出す:
  - `y` が 1D の場合 `ValueError`
  - `T_mel <= 0` で `ValueError`
  - `T_audio < T_mel * hop_length` (STFT frame 数不足) で `RuntimeError`
  - `win_length > n_fft` のコンストラクタ呼び出しで `ValueError`

### 5.2 e2e / 結合テスト
- [ ] **本チケット内 e2e**: 後続チケット T-M1.6 で sub-model に組み込むため、本チケットでは結合テストなし。代わりに `tests/test_stft_module.py::test_concat_compatibility` を 1 つ追加し、mel-spec ダミー (B, 128, T_mel) と STFT-spec を `torch.cat([mel, stft_spec], dim=1)` した結果が `(B, 128 + 2F-2, T_mel)` になることを確認 (T-M1.6 への前哨)

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.2 より転記)
- [ ] GAN 設定 (n_fft=2048, win=1200, hop=300): 入力 `(B, T_mel*300)` → 出力 `(B, 2046, T_mel)`
- [ ] Diff 設定 (n_fft=1024, win=1024, hop=256): 入力 `(B, T_mel*256)` → 出力 `(B, 1022, T_mel)`
- [ ] 時間長 truncation 後の T 次元が指定 T_mel と完全一致
- [ ] 実部・虚部の DC/Nyquist 扱いが正しい (虚部の最初/最後 bin が削除されている)
- [ ] 単純な正弦波で round-trip テスト: STFT → 期待される周波数 bin にエネルギー集中

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **`torch.stft` の `return_complex=True` 廃止/変更** | torch >= 2.10 の API 変更で挙動が変わる可能性 | torch 2.10 では complex tensor がデフォルト挙動。`return_complex=False` モードは **deprecated 警告**。本チケットでは `return_complex=True` に固定し、torch バージョン依存性を最小化。`return_complex=False` は使わない |
| **`center=True` で生まれる reflection padding 起因の不確定性** | 入力の最初・最後フレームに padding 由来のエネルギー漏れが入る | `torch.stft` のデフォルト pad_mode は `"reflect"`。本論文 §3.1 で `center=True` 指定があるため、この挙動を許容。`test_roundtrip_sine_440hz` で中心 frame のエネルギーが期待 bin に集中することを検証して品質を担保 |
| **時間軸 truncation の境界** | `T_stft >= T_mel + 1` ぐらいの余剰が出る場合があり、`Y[..., :T_mel]` で安全に切れるが、`T_stft < T_mel` の場合は呼び出し側のバグ | コンストラクタではなく forward で `RuntimeError` を投げる。エラーメッセージに「T_audio >= T_mel * hop_length が必要」と明示。T-M1.6 sub-model wrapper 側で `T_audio = T_mel * hop_length` を担保する責任 |
| **DC/Nyquist 虚部の本当に 0 か?** | 数値計算上の round-off で完全な 0 ではない可能性、削除しないと無意味な channel が入る | `test_dc_nyquist_imag_dropped` で `Y.imag[:, 0, :]` と `Y.imag[:, -1, :]` が `< 1e-5` を確認。削除位置 `[:, 1:-1, :]` が正しいことを検証 |
| **`register_buffer(persistent=False)` 選択の意図** | `state_dict` に window が乗らず、resume 時に再生成される。意図通り | window は決定的に再生成可能なので persistent=False で OK。`save/load` 時のサイズ削減効果。`module.to(device)` での device 追従は維持される |
| **fp16/bf16 入力時の STFT 精度** | 訓練で AMP を使うと `torch.stft` が fp16 でエラー、または精度低下 | torch の `torch.stft` は内部で fp32 cast する版もあるが、安定性のため本モジュール内では `y.float()` を強制しない方針。AMP 利用時は呼び出し側 (sub-model wrapper) で `with autocast(enabled=False):` ガードを推奨。これを §9.1 で T-M1.6 に申し送る |
| **Vocos `STFT` クラス / FastDiff `STFT` のコピー誘惑** | CLAUDE.md ポリシー違反 | 本実装は `torch.stft` を直叩きする最小ラッパとし、参考実装の class 構造は流用しない。レビュー観点 §7 で確認 |
| **CUDA で `torch.stft` が deterministic でない** | `test_deterministic` が CUDA で flaky | PyTorch docs では CUDA `torch.stft` は deterministic と謳われているが、algo によって異なる。CPU で deterministic を確認すれば足りるので、`test_deterministic` は CPU 強制で実行 (`device="cpu"`) |

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の関連項目はすべて **確定済み**:
  - STFT module パラメータ: `n_fft / win_length / hop_length` は mel-spec と完全一致 (§A2 + §C3)
  - DC/Nyquist 虚部削除: 論文 §3.1 明記 (§A2)
  - center=True、Hann window、normalized=False、onesided=True: 論文 §3.1 (§A2)
  - 時間軸 truncation: 論文 §3.1「mel-spec の時間長に truncate」(§A2)
- 残る曖昧さなし。本チケットでは追加の決定不要

### 6.3 他チケットとの整合性
- **T-M0.2 (scaffold)**: `src/wavenext2/models/stft.py` の TODO スタブを本チケットで実装に置換。`tests/test_stft_module.py` の `pytest.skip` placeholder を本チケットの実テストに置換
- **T-M1.3 (mel-spec)**: `hop_length` を mel-spec と同一値にする必要あり (GAN: 300, Diff: 256)。両チケットで **`configs/{gan,diff}_wavenext2.yaml` の同一 key (`hop_length`) を参照する** ことで整合性を担保。本チケット時点では `STFTModule` のコンストラクタ引数として受ける形にし、config 読み込みは T-M1.6 / T-M2.5 / T-M3.2 で実施
- **T-M1.6 (sub-model)**: 本モジュールの forward I/O 仕様 (`y: (B, T_audio), T_mel: int → (B, 2F-2, T_mel)`) を sub-model wrapper が呼び出す。channel 順序 (mel が先、STFT-spec が後の `torch.cat([mel, stft_spec], dim=1)`) は §9.1 で明示
- **T-M1.4 (Generator)**: Generator の `input_channels = 128 + (2F-2)` 計算は本モジュールの出力 ch 数 `2F-2` と整合 (docs/architecture.md §2 の表)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 論文 §3.1 と docs/architecture.md §3 の処理手順 (1〜6) と実装が **完全一致**:
  - (1) Hann window: `torch.hann_window(win_length)` を `register_buffer` で登録
  - (2) STFT は center=True、normalized=False、onesided=True、return_complex=True
  - (3) mel-spec 時間長に truncate: `Y[..., :T_mel]`
  - (4) 実部 (全帯域 F) と虚部 (F-2、DC/Nyquist 除外) を分離: `Y.real` / `Y.imag[:, 1:-1, :]`
  - (5) channel 次元で concat: `torch.cat([real, imag], dim=1)`
  - (6) 出力 ch 数 `2F-2`
- [ ] Acceptance criteria 全 5 項目クリア (§5.3)
- [ ] Unit テスト全 pass (`uv run pytest tests/test_stft_module.py -v` で 11 件 (10 + concat 1) 通過)
- [ ] CLAUDE.md / 既存コードのスタイル準拠:
  - 型ヒント (`y: torch.Tensor, T_mel: int -> torch.Tensor`)
  - docstring (クラス + forward)
  - 命名: `STFTModule`, `n_fft`, `hop_length`, `win_length`
- [ ] エラー処理:
  - `win_length > n_fft` → コンストラクタで `ValueError`
  - `y.dim() != 2` → forward で `ValueError`
  - `T_mel <= 0` → forward で `ValueError`
  - `T_audio < T_mel * hop_length` (STFT frame 数不足) → `RuntimeError`
- [ ] パラメータ数: STFTModule は **学習パラメータ 0** (window は buffer)。`sum(p.numel() for p in module.parameters()) == 0` を確認
- [ ] メモリ消費: GAN 設定で batch=16, T_mel=85 で複素 spectrogram メモリ < 100 MB (fp32)
- [ ] 参考実装 (Vocos `vocos/spectral_ops.py` 等、FastDiff `module/stft.py` 等) を **コピーしていない** (`torch.stft` 直叩きの最小ラッパ)
- [ ] `register_buffer(persistent=False)` で state_dict に window が乗っていないこと

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン M1) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **`torchaudio.transforms.Spectrogram` に置き換え** | 高水準 API、`power` / `pad_mode` の設定が宣言的、torchaudio の trace/scripting 対応 | (a) `Spectrogram` は magnitude を返すため、本チケットで必要な **実部/虚部分離** が一手間 (`return_complex=True` 指定で複素を取れるが、Vocos と同じ手間)。(b) `n_fft / win_length / hop_length` をそのまま渡せるが、**center=True / normalized=False / onesided=True を default にできる保証** は torchaudio バージョンによる。(c) `forward()` シグネチャが `(audio) -> spectrogram` 固定で、`T_mel` を渡せず外側で truncate する必要あり | 本論文 §3.1 の処理手順を **一行ずつ追跡** したいため `torch.stft` 直叩きが最も透明。`Spectrogram` を挟むと将来のデバッグで torchaudio 内部実装の確認が増える | **`torch.stft` の API が breaking change を起こす場合** (例: torch 3.0 で `return_complex` が deprecated になるなど)、または **AMP/`torch.compile` 対応で torchaudio の方が安定と判明したとき** |
| **real/imag concat ではなく magnitude のみ提供** | チャネル数が半減 (`F` ch のみ) → Generator 入力次元が `128 + F` に削減、メモリ・計算量がやや軽い | 位相情報が失われるため、generator が直接位相復元する必要が生じる。これは **本論文の `2F-2` ch 構成と論理的に異なる別アーキ** (Vocos の `magnitude + phase` 分解とも違う) | 論文 §3.1 で「実部 F + 虚部 F-2 を concat」と明記されており、magnitude のみは仕様違反 | **再評価しない** (仕様逸脱) |
| **learnable STFT (1D conv で代替)** | 学習可能 filter bank として generator に最適化される可能性、Fourier 基底に固定されない柔軟性 | (a) パラメータ数増 (n_fft * win_length 程度の重み)。(b) 論文では STFT module を **fixed** な前処理として扱っている (§3.1)。(c) 学習可能化すると Vocos 系の warm-start 互換性が崩れる | 論文に "STFT module" と明記され、`torch.stft` 等価の固定変換として記述されている。学習可能化は別 ablation 項目 | **本実装の品質が論文と乖離した場合** (例: M5 smoke で STFT-spec を入れても入れなくても品質が変わらないとき)、または **M6 ablation で `learnable STFT` を試すとき** |
| **`torch.fft.rfft` + `unfold` で完全自前実装** | torch.stft 内部実装ブラックボックスを排除、`pad_mode` を完全制御 | コード量が増え、center=True の reflection padding を自分で書く必要、Hann window の broadcast を自分で組む必要 | `torch.stft` で過不足なく要件を満たせるため自前実装は YAGNI | **`torch.stft` で deterministic / `torch.compile` 互換が崩れた場合** |
| **`STFTModule` の forward を `(y, T_mel)` ではなく `(y)` のみにし、T_mel は外側で truncate** | API が単純化 (T_mel 引数不要) | 呼び出し側で必ず truncate を書く必要があり、漏れの risk が増える。論文 §3.1 で truncate は STFT module の一部として記述されているので意味的にも本モジュール内で完結すべき | API の一貫性のため `(y, T_mel)` を維持 | **T-M1.6 sub-model wrapper の実装中に `T_mel` を渡しにくいケースが判明した場合** |
| **`STFTModule` を関数化 (nn.Module ではなく `def stft_module(...)`)** | 学習パラメータが無いので関数で十分 (`forward` の overhead がない) | `register_buffer` で window を device 追従させたいので Module 化が便利。state_dict 統合・`.to(device)` 一括移動の利点も大きい | Module 化を維持 (window 管理の利便性) | **本実装が `torch.compile` で Module overhead がボトルネックと判明した場合** |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。size=M で 1 ファイル + テスト 1 ファイル。STFT module は論文 §3.1 が独立して定義されているので、単独チケット化が自然
- **別マイルストーンに移すべき部分はないか**: なし。STFT module は M1 (コア部品) の中核
- **インターフェース定義の見直し余地**:
  - `forward(y, T_mel)` の `T_mel` を引数で受けるか、`hop_length` から逆算するか: **引数で受ける** を維持 (mel-spec 側の正解値を信頼)
  - 出力 channel ordering (`real → imag` 順): 本実装で `torch.cat([real, imag], dim=1)` とする。Generator 側もこの順序を仮定 (real F bins → imag F-2 bins)。**§9.1 で T-M1.6 に明示**
  - 出力次元の名前: 本実装では `(B, 2F-2, T_mel)`、Vocos 系では `(B, T_mel, 2F-2)` のような時間軸位置の選択肢があるが、ConvNeXt block が `(B, C, T)` を要求するので channel-first を維持
- **`pyproject.toml` への依存追加**: なし (torch のみ使用)
- **再評価トリガー条件**: §8.1 末尾の表を参照

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未着手)
- 次の似たタスクで応用できる教訓: (未着手)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M1.6 (Sub-model wrapper) への申し送り (最重要)

- **インターフェース**: `STFTModule(n_fft, hop_length, win_length).forward(y, T_mel) -> (B, 2F-2, T_mel)`
- **mel-spec との concat の channel 順序 (確定)**:
  ```python
  # T-M1.6 sub-model wrapper 内
  stft_spec = self.stft_module(y_prev, T_mel=mel.shape[2])  # (B, 2F-2, T_mel)
  x = torch.cat([mel, stft_spec], dim=1)                     # (B, 128 + 2F-2, T_mel)
  # ↑ 順序: [mel (128 ch)] [real (F ch)] [imag (F-2 ch)]
  # 結果のチャネル境界:
  #   - [0 : 128)         = mel (log-mel-spec)
  #   - [128 : 128+F)     = STFT 実部 (F = n_fft//2 + 1)
  #   - [128+F : 128+2F-2)= STFT 虚部 (DC/Nyquist 除外、F-2 個)
  ```
  この順序を Generator 側 (T-M1.4) の `Conv1d(input_channels, dim, kernel=7)` も前提とする
- **GAN 設定**: `input_channels = 128 + 2*(2048//2+1) - 2 = 128 + 2048 = 2176`
- **Diff 設定**: `input_channels = 128 + 2*(1024//2+1) - 2 = 128 + 1024 = 1152`
- **入力波形長の制約**: `T_audio >= T_mel * hop_length` を sub-model wrapper 側で保証すること (本モジュールは不足時に `RuntimeError`)
- **AMP/autocast の取り扱い**: `torch.stft` の AMP 安定性は未確認。sub-model wrapper 側で `with torch.cuda.amp.autocast(enabled=False):` ガード推奨。**本チケット時点では実装しない** が、M5 smoke で NaN/Inf が出た場合はここに戻って対処
- **device 移動**: `STFTModule.to("cuda")` で window buffer も追従。`register_buffer(persistent=False)` のため state_dict には乗らない (save/load 時に注意不要)

#### T-M1.4 (Generator) への申し送り
- Generator の `input_channels` を `128 + (2 * (n_fft // 2 + 1) - 2)` = `128 + n_fft` で算出 (= `128 + 2F-2`)
- GAN: 2176, Diff: 1152

#### T-M1.3 (Mel-spec) との整合
- `hop_length` を `STFTModule` と完全一致させること。具体的な値は config (`configs/{gan,diff}_wavenext2.yaml`) に集約

### 9.2 ドキュメント更新
完了時に更新するドキュメント:
- [ ] `docs/milestones.md` §M1.2 の Acceptance チェックボックス 5 項目を ☑ に
- [ ] `docs/tickets/index.md` の T-M1.2 ステータス `📝 pending` → `✅ completed`
- [ ] `docs/tickets/index.md` の M1 進捗サマリ (pending -1, completed +1)
- [ ] (該当時) `docs/architecture.md` §3 の擬似コードと実装が一致していることを確認し、差分があれば修正
- [ ] (該当時) `docs/open-questions.md` に補遺 (本チケットで気づいた事項)

### 9.3 Open question として残ったもの

- **AMP/`torch.compile` 互換**: `torch.stft` の fp16 / `torch.compile` 動作は M3 smoke (T-M3.5) で初めて検証される。落ちた場合は本チケットに戻って `with autocast(enabled=False):` ラッパを追加するか、`torchaudio.transforms.Spectrogram` 経路への移行 (§8.1) を再評価
- **deterministic CUDA**: `test_deterministic` を CPU 強制で実行するが、訓練本番で CUDA `torch.stft` が非決定的かどうかは未検証。問題があれば `torch.backends.cudnn.deterministic = True` で確認 (T-M0.2 で予約済み key)
- これらは `docs/open-questions.md` への新規追記は不要 (現状の `__確定済み__` 表に影響しない)
