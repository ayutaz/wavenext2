---
id: T-M1.3
title: Log-Mel spectrogram 抽出 (slaney scale/norm, power=1, log+clamp)
milestone: M1
phase: M1
status: completed
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-27
depends_on: [T-M0.1, T-M0.2]
blocks: [T-M1.6, T-M2.1]
related_docs:
  - docs/milestones.md#m13-mel-spectrogram-抽出
  - docs/architecture.md
  - docs/training.md
---

# T-M1.3: Log-Mel spectrogram 抽出 (slaney scale/norm, power=1, log+clamp)

> **マイルストーン**: [M1](../milestones.md#m1-コア部品-sub-model-の構成要素-作業量-large6-サブタスク) / **サブタスク**: [M1.3](../milestones.md#m13-mel-spectrogram-抽出)
> **依存**: [T-M0.1](T-M0.1-python-env.md), [T-M0.2](T-M0.2-scaffold.md) / **後続**: [T-M1.6](T-M1.6-sub-model.md), [T-M2.1](T-M2.1-dataset.md)

## 1. タスク目的とゴール

### 目的
LibriTTS-R の生波形から **128-dim log-mel-spectrogram** を抽出する `LogMelSpectrogram` モジュールを実装する。GAN-WaveNeXt 2 (hop=300, n_fft=2048, win=1200) と Diff-WaveNeXt 2 (hop=256, n_fft=1024, win=1024) の両 config を **同一クラス** でカバーし、後続の Dataset (T-M2.1) や Sub-model (T-M1.6) から `config 経由のみ` で安全にインスタンス化できる SoT 原則を確立する。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/data/mel.py` に `LogMelSpectrogram(nn.Module)` が実装され、ハードコードを排して全パラメータを `__init__` 引数として受け取る
- [ ] `torchaudio.transforms.MelSpectrogram(power=1.0, mel_scale="slaney", norm="slaney", center=True)` を内部で利用し、forward 末尾で `torch.log(torch.clamp(mel, min=eps))` を適用する (Vocos `safe_log` + HiFi-GAN 慣例)
- [ ] GAN 設定 (sample_rate=24000, n_fft=2048, hop=300, win=1200, n_mels=128, f_min=20, f_max=12000) で 1 秒入力 → 出力 shape `(B, 128, 80)`
- [ ] Diff 設定 (n_fft=1024, hop=256, win=1024) で 1 秒入力 → 出力 shape `(B, 128, 94)`
- [ ] 出力範囲が `[log(eps), log(max)]` (M1 暫定 eps=1e-7 → `log(1e-7) ≈ -16.1181`) の下限内に収まる (`assert (out >= log(eps) - 1e-6).all()`)
- [ ] `tests/test_mel.py` が新規作成され、Acceptance 全項目を pytest で網羅 (`uv run pytest tests/test_mel.py` が pass)
- [ ] `from wavenext2.data.mel import LogMelSpectrogram` が import 可能
- [ ] T-M0.2 で予告した「mel パラメータは config 経由のみで関数化」の SoT 原則を docstring / 型ヒントに明示

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/data/mel.py` (本体実装、T-M0.2 で TODO stub 配置済み → 本実装で書き換え)
  - `tests/test_mel.py` (新規。milestones.md §M0.2 のテストリストには明示されていないが、本チケットで追加する。T-M0.2 の `pytest.skip` placeholder は本ファイル名では作成されていないため、completely new file)
- 編集:
  - `configs/gan_wavenext2.yaml` (mel セクションを確定値で commit する。雛形には既に SoT 候補 key が並んでいる想定。差異があれば更新)
  - `configs/diff_wavenext2.yaml` (同上)
- 編集 (ステータス更新):
  - `docs/milestones.md` §M1.3 の Acceptance チェックボックス
  - `docs/tickets/index.md` の T-M1.3 ステータス行

### 2.2 主要構造

```python
# src/wavenext2/data/mel.py
"""Log-Mel-spectrogram 抽出モジュール.

SoT (Single source of truth): mel 抽出パラメータは config (configs/*.yaml) 経由のみで
受け取る。本モジュール内に hop/n_fft/win/n_mels/f_min/f_max のデフォルト「マジック値」を
書かないこと。GAN(hop=300) と Diff(hop=256) の取り違えを防ぐため、呼び出し側は必ず
config 経由で明示的に値を渡す。

確定根拠:
- docs/architecture.md §6.5 (Mel-spectrogram 抽出と log 正規化)
- docs/training.md §1.2 (Mel-spectrogram 抽出パラメータ)
- docs/open-questions.md §C3 (参考実装からの確定)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    """Log-mel-spectrogram extractor with slaney scale/norm + log+clamp.

    Vocos `safe_log` (`torch.log(torch.clamp(mel, min=eps))`) + HiFi-GAN/BigVGAN 慣例
    (slaney scale / slaney norm / center=True / power=1.0) の組み合わせ。

    Args:
        sample_rate: 24000 (LibriTTS-R)
        n_fft: GAN=2048, Diff=1024
        hop_length: GAN=300, Diff=256
        win_length: GAN=1200, Diff=1024
        n_mels: 128 (両モデル共通)
        f_min: 20.0
        f_max: 12000.0
        eps: **1e-7 (M1 段階の暫定値、Vocos warm-start 互換性のため採用)**.
            HiFi-GAN 慣例の 1e-5 は学習安定性で有利だが、M1 では warm-start 互換性を
            優先。M2 smoke で発散時に 1e-5 へ上げる方針 (§8.1 再評価トリガー参照)
        mel_transform: 引数注入可能な mel 変換モジュール (DI / mock テスト用、`None`
            の場合は `torchaudio.transforms.MelSpectrogram` を本クラス内で構築)

    Forward:
        Input  audio: (B, T_audio) float32 in [-1, 1]
        Output log_mel: (B, n_mels, T_mel) float32
            T_mel = ceil(T_audio / hop_length) + 1 (center=True padding により +1)
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        eps: float = 1e-7,  # M1: Vocos warm-start 互換性のため 1e-7、M2 smoke で発散時 1e-5 へ
        mel_transform: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # ハードコードのデフォルト値を持たない (SoT 原則)
        # 全引数を必須化することで、呼び出し側が config から明示的に渡すことを強制
        # SoT enforcement: docstring 明記に加えて runtime assert で違反検出
        assert eps > 0, f"eps must be positive (got {eps}); see docstring SoT note"
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps

        # mel_transform を DI 可能にすることで test で mock 注入 (<10ms 完走)
        if mel_transform is not None:
            self.mel = mel_transform
        else:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=1.0,             # magnitude (Vocos と同じ、power=2.0 ではない)
                mel_scale="slaney",   # HiFi-GAN/BigVGAN 慣例 (htk ではない)
                norm="slaney",        # per-band normalization
                center=True,           # center=True padding
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T_audio) or (T_audio,)
        mel = self.mel(audio)
        # natural log + clamp (`torch.log10` ではなく `torch.log`)
        log_mel = torch.log(torch.clamp(mel, min=self.eps))
        return log_mel

    @classmethod
    def from_config(cls, cfg: dict) -> "LogMelSpectrogram":
        """Config dict (例: cfg['mel']) からインスタンス化する factory.

        T-M1.2 (STFT module) との一貫性のため、M1 phase review で全モジュールに
        `from_config()` classmethod factory を持たせる方針で統一済み。

        Args:
            cfg: {"sample_rate": ..., "n_fft": ..., ...} を含む dict
                (`configs/{gan_wavenext2,diff_wavenext2}.yaml` の `mel:` セクション)

        Returns:
            LogMelSpectrogram instance
        """
        required_keys = {
            "sample_rate", "n_fft", "hop_length", "win_length",
            "n_mels", "f_min", "f_max",
        }
        missing = required_keys - cfg.keys()
        if missing:
            raise ValueError(
                f"Missing required mel config keys: {missing}. "
                f"All mel parameters must be specified via config (SoT principle)."
            )
        return cls(**{k: cfg[k] for k in required_keys | ({"eps"} & cfg.keys())})


def log_mel_spectrogram(audio: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Stateless functional API (Vocos `feature_extractors.py` 風 hybrid 案).

    内部で `functools.lru_cache(maxsize=2)` により GAN/Diff の 2 種類の
    `LogMelSpectrogram` インスタンスをキャッシュ。Notebook / 評価スクリプトで
    モジュールインスタンスを保持しない簡易呼び出し向け。

    本格利用 (Dataset / 訓練ループ) ではキャッシュヒット保証のため、必ず
    `LogMelSpectrogram.from_config()` で明示インスタンス化する経路を使うこと。
    """
    # 実装は dict が hashable でないため、frozen tuple 変換 + lru_cache を内側 helper で実施。
    # 詳細は実装時に確定 (本チケット内で判断)
    raise NotImplementedError("see implementation note in mel.py")
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 (GAN) | 値 (Diff) | 出典 |
|---|---|---|---|
| sample_rate | 24000 | 24000 | LibriTTS-R (`docs/architecture.md` §7) |
| n_fft | 2048 | 1024 | `docs/architecture.md` §6.5 / `docs/training.md` §1.2 |
| hop_length | 300 | 256 | `docs/architecture.md` §6.5 |
| win_length | 1200 | 1024 | `docs/architecture.md` §6.5 |
| n_mels | 128 | 128 | `docs/architecture.md` §6.5 |
| f_min | 20.0 | 20.0 | `docs/architecture.md` §6.5 (WaveFit-PT) |
| f_max | 12000.0 | 12000.0 | `docs/architecture.md` §6.5 (WaveFit-PT) |
| power | 1.0 | 1.0 | Vocos (magnitude、`docs/open-questions.md` §C3) |
| mel_scale | "slaney" | "slaney" | HiFi-GAN/BigVGAN 慣例 |
| norm | "slaney" | "slaney" | per-band normalization |
| center | True | True | Vocos |
| log type | natural log (`torch.log`) | 同左 | Vocos `safe_log` (NOT `torch.log10`) |
| log eps | **1e-7 (M1 暫定)** | **1e-7 (M1 暫定)** | Vocos warm-start 互換性。M2 smoke で発散時に 1e-5 (HiFi-GAN 慣例) に上げる方針 (§8.1 再評価トリガー) |

T_mel の算出 (center=True の場合):
```
T_mel = floor(T_audio / hop_length) + 1
GAN: 24000 / 300 = 80, +1 = 81 (torchaudio の挙動次第。実機検証で確定)
Diff: 24000 / 256 = 93.75, floor +1 = 94
```

milestones.md §M1.3 では GAN を「24000/300 = 80」、Diff を「24000/256 ≈ 94」と記述している。torchaudio の center=True 挙動 (片側 pad 後の長さに依存) で +1 ずれる可能性があるため、**実機テストで確定値を `tests/test_mel.py` に固定**する (§6.1)。

### 2.4 アルゴリズム / 処理フロー

1. `__init__`:
   - 全 mel パラメータをハードコードのデフォルト値なしで必須引数として受け取る (eps のみデフォルトあり)
   - `torchaudio.transforms.MelSpectrogram` を `power=1.0, mel_scale="slaney", norm="slaney", center=True` で構築
2. `forward(audio)`:
   - `audio` (`(B, T_audio)` または `(T_audio,)`) を `self.mel` に渡し、`(B, n_mels, T_mel)` を取得
   - `torch.log(torch.clamp(mel, min=self.eps))` で natural log + 下限 clip
3. (任意) `build_log_mel_from_config(cfg)` factory:
   - `cfg` dict から必須キー存在チェック → `LogMelSpectrogram(**cfg)` を返す
   - SoT 原則: ハードコード defalut で fallback しない (`ValueError` で fail-fast)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `src/wavenext2/data/mel.py` 実装 + `tests/test_mel.py` 記述 | general-purpose |
| Reviewer | 1 | docs/architecture.md §6.5 / docs/training.md §1.2 との完全整合確認、librosa 数値比較レビュー | general-purpose |
| Tester | 1 | pytest 実行 + Acceptance 検証 + GAN/Diff config の取り違え試験 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes**
  - T-M1.1 (ConvNeXt block), T-M1.2 (STFT module), T-M1.5 (Noise embedding) と並列可能 (依存なし、ファイル独立)
  - T-M1.4 (Generator) は T-M1.1 依存のため、T-M1.3 とは独立に並列可
- 並列実行する場合の最大並列数: 同フェーズ M1 内で 4 (M1.1, M1.2, M1.3, M1.5)
- size=S かつ機能境界がクリアなため、Implementer 1 + Reviewer 1 で実施可。Tester は Acceptance 量が多くなければ Reviewer に統合可

## 4. 提供範囲 (Scope)

### In Scope
- `LogMelSpectrogram(nn.Module)` クラスの実装 (`__init__` + `forward`)
- `torchaudio.transforms.MelSpectrogram` の wrapper として `power=1.0, mel_scale="slaney", norm="slaney", center=True` を内部固定
- forward 末尾の `torch.log(torch.clamp(mel, min=eps))` 適用
- `build_log_mel_from_config(cfg)` factory (任意、推奨)
- `tests/test_mel.py` 新規作成: GAN/Diff 各 config での shape, range, deterministic, config 経由必須化テスト
- docstring に「mel パラメータは config 経由のみで関数化」の SoT 原則を明記
- `configs/{gan_wavenext2,diff_wavenext2}.yaml` の `mel:` セクション値検証 (差異あれば更新)

### Out of Scope
- LibriTTS-R Dataset (T-M2.1) - 本チケットでは toy 音声 (`torch.randn` / sine wave) でテストのみ
- mel-cache の precompute スクリプト (`scripts/extract_mel.py`) - T-M2.1 で実装
- STFT module (T-M1.2) - sub-model 入力側の STFT は別ファイル `src/wavenext2/models/stft.py` で実装済み (本チケットの mel とは別物)
- Generator (T-M1.4) / Sub-model (T-M1.6) との結合 - T-M1.6 で実施
- librosa との完全数値一致の検証 - §6.1 に書く通り「整合度の許容差」を実装後に再評価
- Mel inversion (mel → audio) - 本プロジェクトでは不要 (generator 側で逆変換)

### Deliverable
- ファイル:
  - `src/wavenext2/data/mel.py` (本実装)
  - `tests/test_mel.py` (新規、§5 のテスト全て)
- 関数 / クラス:
  - `class LogMelSpectrogram(nn.Module)`
  - `def build_log_mel_from_config(cfg: dict) -> LogMelSpectrogram` (任意、推奨)
- ドキュメント差分:
  - `docs/milestones.md` §M1.3 の Acceptance チェック更新
  - `docs/tickets/index.md` の T-M1.3 ステータス更新
  - `configs/*.yaml` の `mel:` セクションが §2.3 表と一致 (差異あれば更新コミットを別途)

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_mel.py`)

#### Shape 系
- [ ] `test_gan_shape_1sec`: GAN 設定で 1 秒入力 (24000 samples) → 出力 shape `(B, 128, T_mel_gan)` (実機で `T_mel_gan` の確定値を pin、milestones は 80 を想定)
- [ ] `test_diff_shape_1sec`: Diff 設定で 1 秒入力 → 出力 shape `(B, 128, T_mel_diff)` (milestones は 94 を想定)
- [ ] `test_batch_dim_preserved`: `(4, 24000)` 入力 → `(4, 128, T_mel)` 出力
- [ ] `test_arbitrary_length`: 0.5 秒, 2 秒, 5 秒入力で shape 整合性 (T_mel が hop_length に比例)
- [ ] `test_unsqueezed_input`: `(T_audio,)` 入力 (バッチ次元なし) でも動作するか確認 (torchaudio.MelSpectrogram の標準挙動に従う)

#### Range 系
- [ ] `test_output_lower_bound`: 出力 >= `log(eps) - 1e-6` (clamp が機能、M1 暫定 `log(1e-7) ≈ -16.1181`)
- [ ] `test_output_finite`: 全要素 finite (`torch.isfinite(out).all()`)
- [ ] `test_silence_input`: `torch.zeros(B, 24000)` 入力で出力が全要素 `log(eps)` (M1 暫定 ≈ -16.1181) (silence は mel=0 → clamp で eps)
- [ ] `test_sine_wave`: 単一周波数の sine wave (例: 1000 Hz) 入力で、対応する mel-band の値が他 band より大きい (高 SNR 検証)

#### Deterministic / config 経由必須
- [ ] `test_deterministic`: 同入力で同出力 (`torch.equal` で 2 回 forward 結果一致)
- [ ] `test_requires_all_keys`: `build_log_mel_from_config({"sample_rate": 24000})` 等の不完全 cfg で `ValueError`
- [ ] `test_gan_diff_distinct_shape`: 同じ audio に対して GAN config と Diff config で異なる T_mel が出力される (取り違え検出器)

#### 数値整合
- [ ] `test_no_negative_eps_failure`: `eps=-1.0` 等の不正値で `AssertionError` を期待 (`__init__` で `assert eps > 0` を実装)
- [ ] `test_power_is_1`: `m.mel.spectrogram.power == 1.0` を assert (HiFi-GAN/Vocos=1.0、Tacotron2=2.0 の混同リスク検出)
- [ ] `test_dtype_robustness`: `m.half()` した後で `m(audio.half()).dtype` を確認、mel filter bank (`MelScale.fb`) が誤って fp16 化されると計算崩壊するため、fp16 化挙動を pin する
- [ ] `test_di_mock`: `mel_transform=MagicMock()` を `__init__` に注入し、forward が mock を呼び出すこと (DI 動作確認、テスト <10ms)

### 5.2 e2e / 結合テスト
- [ ] `uv run python -c "from wavenext2.data.mel import LogMelSpectrogram; m = LogMelSpectrogram(sample_rate=24000, n_fft=2048, hop_length=300, win_length=1200, n_mels=128, f_min=20.0, f_max=12000.0); import torch; print(m(torch.randn(2, 24000)).shape)"` がエラーなく shape 出力
- [ ] T-M2.1 (Dataset) と T-M1.6 (Sub-model) からの import 経路が `from wavenext2.data.mel import LogMelSpectrogram` で通る (本チケット完了時点では下流未実装のため import 確認のみ)
- [ ] `uv run pytest tests/test_mel.py -v` が exit 0 で完走

### 5.4 テスト戦略 (M1 phase review 反映)

- **librosa との数値整合**: M5 smoke で `librosa.feature.melspectrogram` と本実装の RMS 差を CI に **metric として記録 (報告のみ、fail させない)**。閾値超過時に主観評価を実施
- **deterministic fixture seed**: `tests/conftest.py` で `pytest_randomly` の seed 制御方針を T-M0.2 に申し送り (§9.1 参照)。本チケットのテストは fixture `sample_audio_24khz_1sec` (seed=42) を使用
- **coverage 目標 90%**: `tests/test_mel.py` 単体で `src/wavenext2/data/mel.py` の line coverage 90% 以上を目標 (factory の `ValueError` 経路、`assert eps > 0` 経路、DI 経路を含む)

### 5.3 Acceptance criteria (`docs/milestones.md` §M1.3 より転記)
- [ ] GAN 設定で 1 秒 (24000 samples) 入力 → mel shape `(B, 128, 80)` (24000/300 = 80) — torchaudio の center=True 挙動次第で +1 ずれる場合は test 側で実機値を pin して milestones を更新
- [ ] Diff 設定で 1 秒入力 → mel shape `(B, 128, 94)` (24000/256 ≈ 94)
- [ ] 出力範囲が `[log(eps), log(max)]` (M1 暫定 eps=1e-7 → `log(1e-7) ≈ -16.12`) に収まる

## 6. 懸念事項

### 6.1 技術的リスク

| リスク | 影響範囲 | 検知方法 / 緩和策 |
|---|---|---|
| **torchaudio.MelSpectrogram の T_mel が milestones 想定値とずれる (center=True padding)** | Acceptance 不合格、後続 (Dataset / Sub-model) の shape 仮定がずれる | 実機テストで確定値を `tests/test_mel.py` に pin する。milestones.md §M1.3 の数値が +1 ずれる場合は milestones を更新する (本チケットの責任範囲)。HiFi-GAN/BigVGAN 慣例は `center=False` + 手動 pad だが、Vocos は center=True なので Vocos 準拠で進める |
| **librosa との数値乖離 (実装後に確認)** | 主観評価で品質劣化、論文との再現性低下 | M5 smoke 後に librosa-mel vs torchaudio-mel の RMS 差を計測し、許容差 (例: <1e-3) を満たすか確認。乖離が大きい場合は §8.1 「librosa mel に統一」の代替案へ移行。本チケットでは数値整合の正式検証は実施せず、torchaudio 路線を選択した根拠 (Vocos / WaveFit-PT の慣例) を docstring に明記 |
| **slaney vs htk スケールの取り違え** | f_min/f_max が低周波寄り or 高周波寄りにずれて品質劣化 | テストで `mel_scale="slaney"` が明示的に渡されることを assert (constructor 引数の値を attribute で保持し、test_init で確認)。type hint で `Literal["slaney", "htk"]` を使う案も検討 (Out of Scope だが推奨) |
| **eps が小さすぎて log(0) → -inf 発生 (clamp が機能しない)** | NaN/-inf が後続に伝播 | `torch.clamp(min=eps)` を必ず適用。eps=0 や負値を渡された場合の挙動は `eps > 0` の前提とし、ガード assertion を `__init__` に追加するか検討 (実装は本チケット内で判断) |
| **`win_length > n_fft` のエッジケース** | torchaudio が `RuntimeError` を raise | GAN: win=1200 < n_fft=2048 OK / Diff: win=1024 == n_fft=1024 OK。`win_length <= n_fft` が前提だが、config の typo で win > n_fft になった場合のエラーメッセージが分かりやすいか確認 |
| **forward 中の prepare_window がデバイス間で再アロケート** | GPU 上で繰り返し呼び出すと遅い | `torchaudio.transforms.MelSpectrogram` は `nn.Module` なので `.to(device)` で window buffer が一緒に移動する (内部実装に依存)。M2 smoke で確認、遅延が大きければ register_buffer を明示する |
| **forward 入力 audio が `[-1, 1]` 範囲外の場合** | mel 値が常識を逸脱、後続学習で不安定化 | sox `norm` (T-M2.1 で適用) が `[-1, 1]` を保証する前提。本チケットでは入力 range の事前 assertion は **入れない** (重複チェックでコスト増のため)。docstring で「入力は `[-1, 1]` 想定」と明記する |
| **テスト時の T_mel 計算の表記揺れ (milestones は `24000/300 = 80` だが center=True で `+1`)** | テスト失敗時に milestones.md の修正が必要 | 実装着手前に PoC で `m = torchaudio.transforms.MelSpectrogram(...); m(torch.randn(1, 24000)).shape[-1]` を実機実行し、確定値を本チケット §2.3 と `tests/test_mel.py` に反映する。milestones.md も同時に更新 |
| **`T_mel = 80 vs 81` の確定値が `configs/*.yaml` に SoT 化されていない** | T-M2.1 が segment_length 計算で `T_mel_per_sec` を参照、本チケット完了時点で配信されないと downstream で再計算が発生 | 実機 PoC 確定後、`configs/{gan_wavenext2,diff_wavenext2}.yaml` に `mel.T_mel_per_sec: <確定値>` を追加。本チケット完了報告で T-M2.1 へ申し送り (§9.1 参照) |
| **`MelScale.fb` が fp16 化されるリスク** | `model.half()` で mel filter bank が誤って fp16 化されると、低周波 band の計算が崩壊して NaN | `test_dtype_robustness` で `model.half(); m(audio.half()).dtype` の挙動を pin。filter bank は `register_buffer` で float32 保持する設計を内部実装で確認 (torchaudio バージョン依存) |
| **`mel_scale="slaney"` filter bank が torchaudio version 依存** | バージョン跨ぎで 1e-4 オーダの数値差異事例あり (`docs/open-questions.md` で報告) | `pyproject.toml` で `torchaudio >= 2.10` を pin (T-M0.1 で既に対応済みのはず、本チケット完了時に再確認)。CI で torchaudio version を固定 |
| **`power=1.0` vs `power=2.0` の混同リスク** | HiFi-GAN/Vocos は magnitude (1.0)、Tacotron2 は power spectrum (2.0)。混同すると mel 値スケールが全 band で 2 倍ずれる | `test_power_is_1` で `m.mel.spectrogram.power == 1.0` を assert (§5.1 「数値整合」参照) |

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` ではすべて確定済み (§C3 mel 抽出残り)。本チケットで追加の決定なし
- `eps` のデフォルト値は **M1 段階では 1e-7 (Vocos warm-start 互換性のため暫定採用)**。M2 smoke で発散時に HiFi-GAN 慣例の 1e-5 に上げる方針 (`docs/training.md` §1.2 と一時的に乖離するため M5 phase review で最終決定)
- `LogMelSpectrogram` の log を `torch.log` (natural log) で取ることは確定 (Vocos `safe_log` 準拠、`torch.log10` ではない)

### 6.3 他チケットとの整合性
- **T-M0.2**: `src/wavenext2/data/mel.py` の TODO stub を本チケットで書き換える。stub の docstring に「config 経由のみで関数化」の予告があるため、本チケットで完全実装。`__init__` の必須引数化で SoT を保証
- **T-M1.6 (Sub-model wrapper)**: `LogMelSpectrogram` を import してインスタンス化することは **しない**。Sub-model の入力 mel は T-M2.1 Dataset 側で precompute して渡す設計 (本リポジトリは on-the-fly mel 計算をしない方針が Vocos 慣例)。ただし `extract_mel.py` (T-M2.1 の precompute スクリプト) では本クラスを使う
- **T-M2.1 (Dataset)**: Dataset 側で `LogMelSpectrogram` を呼んで mel-spec を計算する設計と整合 (LibriTTS-R wav → audio normalize → mel)。GAN/Diff の hop 違いを config で切り替えるパターンも本チケットで確立済み
- **`configs/{gan_wavenext2,diff_wavenext2}.yaml`** の `mel:` セクション値と本クラスの `__init__` 引数名が完全一致していること (型ヒントと YAML key の typo がないか確認)
- **T-M0.1**: torchaudio >= 2.10 が installed。`MelSpectrogram` の `mel_scale` / `norm` パラメータは 0.10+ で API 安定

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/architecture.md` §6.5 のコード例と本実装が完全一致 (`power=1.0, mel_scale="slaney", norm="slaney", center=True`)
- [ ] `docs/training.md` §1.2 表のパラメータと `__init__` 引数の値が一致
- [ ] `__init__` にハードコードの mel パラメータ default 値がない (eps を除き全引数必須化)
- [ ] forward 末尾の `torch.log(torch.clamp(mel, min=self.eps))` が `torch.log10` でないこと
- [ ] `tests/test_mel.py` が GAN/Diff の取り違えを検出するテストを含む (`test_gan_diff_distinct_shape`)
- [ ] Acceptance criteria 全項目クリア (実機 T_mel 値が pin され、milestones.md §M1.3 と整合)
- [ ] Unit テスト全 pass (`uv run pytest tests/test_mel.py`)
- [ ] CLAUDE.md / 既存コードのスタイル準拠 (型ヒント、docstring 日本語 + 技術用語英語)
- [ ] 不正入力 (eps<=0, missing config key) で fail-fast (`ValueError`) するか確認
- [ ] パラメータ数 / メモリ消費: `LogMelSpectrogram` は学習可能パラメータを持たない (torchaudio の `MelScale` の filter bank は buffer)。`sum(p.numel() for p in m.parameters())` が 0 であること
- [ ] 参考実装をコピーしていない (Vocos `feature_extractors.py` を参考にしつつ、再構成して書く)
- [ ] `configs/*.yaml` の `mel:` セクション値が §2.3 と一致

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **librosa mel に統一** (`librosa.feature.melspectrogram`) | librosa 慣例で MCD/log F0 RMSE 等の評価系と整合、論文・他研究との数値比較が容易 | GPU 上で動かない (CPU 計算 → tensor 変換が必要)、Dataset の DataLoader worker 内で計算するとオーバーヘッド大、torchaudio の `MelSpectrogram` は GPU on-the-fly 計算可能 | torchaudio 路線は Vocos/WaveFit-PT 慣例。GPU 効率を優先。評価側 (MCD) は librosa 側で別途計算 (`src/wavenext2/eval/`) | M5 smoke で librosa vs torchaudio の数値差が品質に影響する場合 |
| **log mel ではなく生 mel (linear scale)** を出力 | downstream で必要に応じて log を取れる柔軟性 | HiFi-GAN/Vocos/WaveFit-PT 全てが log mel を入力にしている (慣例)、後続 train script で log を取り忘れるリスク | docs/architecture.md §6.5 で log mel を確定済み、慣例準拠 | (再評価しない) |
| **`torch.log10` (base 10) で正規化** | dB 表記との直感的整合 | Vocos `safe_log` は `torch.log` (natural log)、HiFi-GAN/BigVGAN も natural log。混在すると downstream の暗黙の前提が崩れる | natural log 路線で確定 (`docs/training.md` §1.2 "log type: natural log") | (再評価しない) |
| **Dynamic range compression を追加** (例: `log(1 + C * mel)` または `Tacotron 2 dynamic_range_compression`) | 低振幅領域での解像度向上、TTS 系では Tacotron 2 や Glow-TTS で採用例あり | Vocos / WaveFit-PT / HiFi-GAN は dynamic range compression を使っていない、論文準拠を優先 | 慣例準拠、本論文 (WaveNeXt 2) も dynamic range compression に言及なし | downstream モデルで低振幅 mel の表現力不足が確認された場合 (主観聴感で確認) |
| **`eps=1e-7` に変更 (Vocos 完全準拠)** | Vocos と完全数値一致 | HiFi-GAN 慣例の 1e-5 より下限が `log(1e-7) ≈ -16.1` まで広がり、勾配 outlier が増える可能性 | docs/training.md §1.2 で「Vocos 1e-7 より学習安定」のため 1e-5 を採用済み | M2 smoke で学習が不安定な場合 (1e-7 / 1e-6 にしぼって比較) |
| **`build_log_mel_from_config()` factory を提供せず、呼び出し側で `LogMelSpectrogram(**cfg["mel"])` を直接呼ぶ** | コード量が減る、明示性が高い | 必須 key の検証が呼び出し側に分散、YAML typo で `KeyError` ではなく `TypeError` が出る (デバッグしにくい) | factory は任意 (本チケット §2.2 に実装) で、SoT の強調と fail-fast (`ValueError`) を提供 | 呼び出し側の冗長性が問題化した場合 |
| **`MelSpectrogram` の代わりに `torchaudio.transforms.Spectrogram` + 手動 mel filter** | より細かい制御 (center=False, padding 方式変更等) が可能 | コード量増、torchaudio の MelSpectrogram で十分カバーされる | torchaudio.MelSpectrogram は安定 API、本論文の要求を全て満たす | center=True で T_mel ずれが品質に影響する場合 (center=False + manual pad へ移行) |
| **Multi-scale mel (複数 hop_length を同時に抽出)** | Diff モデルで multi-resolution conditioning ができる可能性 | 論文に記述なし、メモリ・計算量増 | 論文準拠、本論文では single mel | (再評価しない、研究範疇) |
| **`eps=1e-5` (HiFi-GAN) vs `eps=1e-7` (Vocos) の M1 再評価** | warm-start 互換性 (1e-7 = Vocos) vs 学習安定性 (1e-5 = HiFi-GAN) のトレードオフ | 1e-5 採用根拠は実測なし、Vocos 重み流用時に出力分布が乖離するリスク | **M1 段階では 1e-7 を採用 (本チケット §2.3 で反映済み)**、M2 smoke で発散時に 1e-5 へ上げる方針 | **M2 smoke で発散検知** (`docs/training.md` §1.2 の最終値も M5 で更新) |
| **DataLoader CPU bound 解消のため `scripts/extract_mel.py` で precompute 設計** | GAN 410h + Diff 32h × 4 の長期訓練で on-the-fly mel 計算が CPU bound 化するのを回避、ディスクキャッシュで GPU 待ち時間削減 | キャッシュ容量 (LibriTTS-R 460h × 128 mels × ~5 frames/sec ≈ 数十 GB)、precompute スクリプトの保守コスト | M1 では on-the-fly 路線、M5 smoke で DataLoader bottleneck を検知した場合に precompute へ移行 (T-M2.1 で `scripts/extract_mel.py` 雛形は提供済みの想定) | **M5 smoke で DataLoader bottleneck 検知** (GPU 使用率が 70% 切る等) |
| **`LogMelSpectrogram` の DI 化 (`mel_transform: nn.Module \| None = None`)** | mock 注入でテスト <10ms 完走、torchaudio 依存をテストから切り離し可能 | `__init__` 引数が増えてやや複雑化、Production 経路 (mel_transform=None) と Test 経路の分岐 | **本チケットで採用 (§2.2 実装に反映済み)**、Reviewer も DI 経路を確認すること | (再評価しない、本チケットで採用) |
| **stateless function `log_mel_spectrogram(audio, cfg) -> tensor` (Vocos `feature_extractors.py` 風 hybrid 案)** | Notebook / 評価スクリプトでモジュールインスタンスを保持しない簡易呼び出し、`functools.lru_cache(maxsize=2)` で GAN/Diff の 2 種をキャッシュ | dict が hashable でないため frozen tuple 変換 helper が必要、本格訓練ループでは class instance 経路の方がキャッシュヒット保証 | **本チケットで実装シグネチャ提示 (§2.2)、実装本体は M5 で評価系から呼び出し需要が確認できた時点で完成** | M5 で評価スクリプト (MCD / log F0 RMSE) からの呼び出し需要発生時 |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度は適切か**: 適切。size=S で `LogMelSpectrogram` クラス + factory + テストの 3 要素のみ。`torchaudio.transforms.MelSpectrogram` を wrap するだけのため複雑度が低く、独立チケットとして分離する価値が高い (T-M2.1 Dataset と分離することで mel SoT の責任境界が明確化)
- **別マイルストーンに移すべき部分はないか**: なし。M1 コア部品の構成要素として mel 抽出は妥当な配置。M2 Dataset に統合する案もあるが、SoT 原則を強調するため分離した方が良い
- **factory パターン全モジュール一貫化**: T-M1.2 STFT module にも `from_config()` factory を持たせるよう **M1 phase review で横断議論済み**。本チケットでは `LogMelSpectrogram.from_config()` を classmethod として採用 (§2.2 反映済み)。STFT 側にも同様に classmethod を持たせる方針を T-M1.2 へ申し送り (§9.1 参照)
- **mel パラメータ SoT (config 経由のみ)**: T-M0.2 の決定方針を docstring + `assert eps > 0` 等の runtime checks で強制 (§2.2 反映済み)。違反検出は **fail-fast** で行い、silent fallback (default 値で代用) はしない
- **インターフェース定義の見直し余地**:
  - `LogMelSpectrogram.__init__` に `eps` のデフォルト値だけは残す: **1e-7 (M1 暫定)** は Vocos warm-start 互換性で確定済みで、毎回 config に書く価値が低い (boilerplate 削減)。ただし他の mel パラメータは必須化する
  - factory は `LogMelSpectrogram.from_config(cfg)` の classmethod で確定 (§2.2 反映済み)。STFT module (T-M1.2) との一貫性のため
  - 出力 dtype: 入力が `float16` (AMP) の場合に内部計算を float32 に上げるか議論。本チケットでは torchaudio の default に従い、AMP 対応は M2.5 train_gan で再評価
- **再評価トリガー条件**: §8.1 末尾参照
- **librosa との整合度許容差**: M5 smoke 完了時に `librosa.feature.melspectrogram` 出力と RMS 差を測定し、品質に影響しない閾値 (例: 1e-3) を確認。乖離が大きい場合は librosa 路線への切り替えを検討

### 8.3 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `LogMelSpectrogram` 実装、`tests/test_mel.py` 19 件 pass (CPU/GPU)。学習パラメータ 0。`from_config` factory + DI (mel_transform 注入) 対応。
- **T_mel 確定 (実機)**: center=True で `T_mel = 1 + T_audio // hop_length`。24000 sample → **GAN 81 / Diff 94**。milestones の「GAN 80」は +1 のずれで実際は **81** (center padding 由来)。

想定外と対処:
1. **eps をチケットの 1e-7 (Vocos warm-start 用) でなく 1e-5 で確定**: 本実装は **scratch 学習で Vocos 重みを warm-start しない** (CLAUDE.md がコピー禁止) ため 1e-7 の根拠が無効。CLAUDE.md / open-questions §C3 / configs (SoT) は全て **1e-5**。コード既定も 1e-5 に統一し全体整合。教訓: **「暫定値」より SoT (config + 確定ドキュメント) を優先**。チケット §2.3/§6.2 の 1e-7 記述は superseded。
2. **config キー名の写像は呼び出し側責務**: configs は `hop`/`log_eps`、本モジュールは `hop_length`/`eps`。`from_config` は param 名 flat dict を受け、YAML→paramdict 写像は T-M2.5/T-M3.2 の config loader が行う設計に確定。`from_config` は `log_eps`→`eps` のみ alias 対応。
3. **stateless `log_mel_spectrogram(audio,cfg)` は実装せず M5 へ延期** (YAGNI、評価系から需要が出た時点で追加)。
4. **T_mel の SoT**: `1 + samples//hop` の式で厳密に導出できるため config に `T_mel_per_sec` フィールドは追加せず、式と実機値 (GAN 81/Diff 94) を §9.1 で申し送る。

次の似たタスクで応用できる教訓:
- 「暫定値」を採用する前に SoT (config/確定 docs) と矛盾しないか必ず照合する。warm-start 前提の値は scratch 学習では捨てる。
- center=True の系列長は実機で pin し、ドキュメントの概算 (80) を実値 (81) に更新する。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

- **インターフェース**:
  - `from wavenext2.data.mel import LogMelSpectrogram`
  - `LogMelSpectrogram.__init__(sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max, eps=1e-7, mel_transform=None)` 必須引数 7 つ + eps + DI 用 mel_transform
  - `LogMelSpectrogram.forward(audio: torch.Tensor) -> torch.Tensor`: 入力 `(B, T_audio)` 想定、出力 `(B, n_mels, T_mel)` (`T_mel` 値は実機で確定して `tests/test_mel.py` に pin)
  - `LogMelSpectrogram.from_config(cfg: dict) -> LogMelSpectrogram` classmethod factory (T-M1.2 STFT との一貫性のため `build_log_mel_from_config` 関数案から classmethod へ変更)
  - (M5 で評価需要発生時に完成予定) `log_mel_spectrogram(audio, cfg)` stateless function
- **設定値**:
  - YAML config `configs/{gan_wavenext2,diff_wavenext2}.yaml` の `mel:` セクションに `sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max` の 7 key を必ず含める (eps は省略可、その場合 1e-7 default = M1 暫定値)
- **注意事項**:
  - mel 抽出パラメータは **config 経由のみ** で渡す。**ハードコード禁止**。呼び出し側で `LogMelSpectrogram(n_fft=2048, ...)` のような literal 渡しを直接書かないこと
  - 入力 audio は `[-1, 1]` 範囲想定 (sox `norm` 後)。`[-1, 1]` 範囲外の入力は内部 assertion なしで通すが、出力 mel 値が常識を逸脱するリスクあり

#### T-M0.2 conftest.py への申し送り
- 本チケットのテストで使用する deterministic fixture を `tests/conftest.py` に追加:
  ```python
  @pytest.fixture(scope="session")
  def sample_audio_24khz_1sec():
      return torch.randn(2, 24000, generator=torch.Generator().manual_seed(42))
  ```
- `pytest_randomly` の seed 制御方針も T-M0.2 で確定する: CI では `--randomly-seed=42` を pin、ローカルでは任意 seed (random) を許容

#### T-M1.2 (STFT module) との一貫性
- M1 phase review で **factory パターンを全モジュール一貫化** する方針が確定。STFT module も `from_config(cls, cfg)` classmethod を持たせる方針 (本チケット §8.2 参照)
- 本チケットで `LogMelSpectrogram.from_config()` を classmethod として採用したため、T-M1.2 にも同様の API 提供を申し送り済み

#### M5 phase review への申し送り
- `eps=1e-7` (M1 暫定) vs `eps=1e-5` (HiFi-GAN 慣例) を **M5 phase review で smoke 結果を見て再評価**:
  - M2 smoke で発散しなければ 1e-7 確定、`docs/training.md` §1.2 を 1e-5 → 1e-7 に更新
  - M2 smoke で発散すれば 1e-5 に上げる (本チケット §2.3 表と `__init__` default を再変更)

#### T-M1.6 (Sub-model wrapper) への申し送り
- Sub-model 自体は `LogMelSpectrogram` を直接呼ばない設計。Dataset (T-M2.1) で precompute された log-mel を受け取る前提
- ただし `LogMelSpectrogram` の出力 shape `(B, 128, T_mel)` を sub-model の `forward(mel, y_prev)` の `mel` 引数として渡す経路は本クラスで担保される

#### T-M2.1 (Dataset) への申し送り
- `LibriTTSRDataset.__init__` で `LogMelSpectrogram.from_config(config["mel"])` をインスタンス化し、`__getitem__` で呼び出す
- 想定: `audio = sox_normalize(load(filelist[idx]))`; `log_mel = self.mel_transform(audio)`; `return log_mel, audio`
- GAN(hop=300) と Diff(hop=256) で config を切り替えることで同じ Dataset コードが両方に使える
- **`T_mel` 確定値の SoT**: 本チケット完了時に PoC 実機で `T_mel_per_sec` を確定し、`configs/{gan_wavenext2,diff_wavenext2}.yaml` の `mel.T_mel_per_sec: <値>` で SoT 化する。T-M2.1 は segment_length 計算で `cfg["mel"]["T_mel_per_sec"]` を参照する設計を採用すること (torchaudio バージョン依存の +1 ずれリスクを config 側で吸収)

#### T-M2.1 の `scripts/extract_mel.py` への申し送り
- 大規模データセット (LibriTTS-R 460h) で mel を pre-compute する場合、本クラスを `--n_fft 2048 --hop_length 300 ...` 等の CLI 引数で受け取りバッチ処理する想定
- precompute 結果は `data/cache/mel/{utterance_id}.npy` に保存

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M1.3 の Acceptance チェックボックス 3 項目を更新
  - [ ] `docs/tickets/index.md` の T-M1.3 ステータスを `📝 pending` → `✅ completed` に更新
  - [ ] (該当時) `docs/architecture.md` §6.5 のコード例を `LogMelSpectrogram` 経由の形式に補足
  - [ ] (該当時) `docs/training.md` §1.2 の値表を本チケットで確定した実機 T_mel 値で更新
  - [ ] (該当時) `configs/{gan_wavenext2,diff_wavenext2}.yaml` の `mel:` セクション値が §2.3 表と一致しない場合は同時にコミット

### 9.3 Open question として残ったもの
- center=True での T_mel の確定値 (`24000/300 + 1 = 81` or `24000/300 = 80`) の torchaudio バージョン依存性 → 実機テストで pin することで解決、本チケット完了時点では `docs/open-questions.md` への追記は不要
- librosa-mel との数値整合度の閾値 → M5 smoke 完了時に再評価、本チケット時点では結論を出さない (§8.1 「librosa mel に統一」の再評価トリガー)
- AMP (mixed precision) 環境での mel 計算の dtype → M2.5 train_gan で再評価
