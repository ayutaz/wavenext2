---
id: T-M2.1
title: LibriTTS-R Dataset + sox norm 正規化
milestone: M2
phase: M2
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M0.3, T-M1.3]
blocks: [T-M2.5, T-M3.2]
related_docs:
  - docs/milestones.md#m21-dataset-srcwavenext2datadatasetpy
  - docs/architecture.md
  - docs/training.md
---

# T-M2.1: LibriTTS-R Dataset + sox norm 正規化

> **マイルストーン**: [M2](../milestones.md#m2-gan-wavenext-2-作業量-large6-サブタスク) / **サブタスク**: [M2.1](../milestones.md#m21-dataset-srcwavenext2datadatasetpy)
> **依存**: [T-M0.3](T-M0.3-libritts-r.md), [T-M1.3](T-M1.3-mel-spectrogram.md) / **後続**: [T-M2.5](T-M2.5-train-gan.md), [T-M3.2](T-M3.2-train-diff.md)

## 1. タスク目的とゴール

### 目的
T-M0.3 で生成した LibriTTS-R filelist (TSV header 付き) を読み込み、訓練/検証/推論で共通利用できる `LibriTTSRDataset` を実装する。本 Dataset は以下を担う:

- LibriTTS-R 24 kHz の wav を `torchaudio.load` で読み出し、Vocos 方式の **sox `norm`** で正規化 (train: U(-6, -1) dB, val: -3 dB)
- 指定された `segment_length` で **反射 pad / random crop** を施し、mel と audio の時間長を整合させる
- `LogMelSpectrogram.from_config()` (T-M1.3) で log-mel を抽出して `(mel, audio)` ペアを返す
- GAN (segment_length=16384, hop=300) と Diff (segment_length=25600, hop=256) の両方を config 経由で切替可能

本 Dataset は T-M2.5 (GAN training) / T-M3.2 (Diff training) / T-M2.6 (smoke) / T-M5.1 (1 epoch) の **唯一の data source** であり、後続全タスクの品質を支配する。

### 確定事項
- **segment_length は config 経由のみ** で受け取る (GAN=16384 / Diff=25600)。**ハードコード禁止**
- **sox `norm` (`torchaudio.sox_effects.apply_effects_tensor`)** を採用 (Vocos `vocos/dataset.py` L42-43 完全準拠)。`torchaudio.functional.gain` 代替は §8.1 で却下根拠を明記
- **反射 pad** (短い wav に `torch.nn.functional.pad(..., mode="reflect")` 適用)、**random crop** (長い wav)
- **center=True 整合性**: `audio.shape[0] == mel.shape[1] * hop_length` (T-M1.6 §6.1 critical 項目で確定する `T_mel` 規約に合わせる)
- **filelist 形式**: T-M0.3 で確定した TSV (header: rel_path / n_samples / speaker_id / chapter_id / duration_sec / peak_dbfs / rms_dbfs) を `csv.DictReader` で読む
- **`n_samples` 列を BucketSampler に活用** (毎 epoch `torchaudio.info` を呼ばない)
- **mel 抽出は on-the-fly** が M1 デフォルト (precompute は §8.1 案で M5 phase review 再評価)
- **`mel` と `STFT-spec` の dynamic range mismatch 懸念** (T-M1.2 / T-M1.6 §6.1 から伝搬) は本チケット tests で grad norm を観測する
- **Windows での sox_effects 動作** (T-M0.1 §6 整合): torchaudio backend に sox_io が含まれることを前提

### ゴール
- [ ] `src/wavenext2/data/dataset.py` に `LibriTTSRDataset(Dataset)` が実装される
- [ ] `__init__(filelist_path, root_dir, segment_length, hop_length, mel_cfg, mode)` のシグネチャで GAN/Diff 両 config から生成可能
- [ ] mode="train" で sox norm gain が **U(-6, -1) dB** から sample される (`np.random.uniform(-6, -1)`)
- [ ] mode="val" で sox norm gain が **-3 dB** の固定値
- [ ] 短い wav は **反射 pad** で `segment_length` まで拡張、長い wav は **random crop**
- [ ] mel と audio の時間長整合: `audio.shape[0] == mel.shape[1] * hop_length` (center=True の `T_mel = floor(T_audio/hop)+1` 規約は T-M1.3 で pin 済み)
- [ ] `LogMelSpectrogram.from_config(mel_cfg)` (T-M1.3) を Dataset 内で利用
- [ ] `tests/test_dataset.py` 新規作成、Acceptance 全項目を pytest で網羅 (`uv run pytest tests/test_dataset.py` が pass)
- [ ] `from wavenext2.data.dataset import LibriTTSRDataset` が import 可能
- [ ] BucketSampler 実装の補助クラスとして `n_samples` 列を Dataset attribute 経由で公開
- [ ] `docs/milestones.md` §M2.1 Acceptance 4 項目をすべてクリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `src/wavenext2/data/dataset.py` (本体実装、T-M0.2 で TODO stub 配置済み → 本実装で書き換え)
  - `tests/test_dataset.py` (新規、`pytest.skip` placeholder 置換)
- 編集:
  - `src/wavenext2/data/__init__.py` (`__all__` に `LibriTTSRDataset` 追加)
  - `configs/gan_wavenext2.yaml` (data セクション: `segment_length: 16384` を確定)
  - `configs/diff_wavenext2.yaml` (data セクション: `segment_length: 25600` を確定)
- 編集 (ステータス更新):
  - `docs/milestones.md` §M2.1 の Acceptance チェックボックス
  - `docs/tickets/index.md` の T-M2.1 ステータス

### 2.2 主要構造

```python
# src/wavenext2/data/dataset.py
"""LibriTTS-R 用 Dataset.

T-M0.3 で生成した filelist (TSV header 付き) を読み込み、sox `norm` 正規化 +
反射 pad / random crop で固定長 segment を切り出し、log-mel と audio のペアを返す。

GAN (segment_length=16384, hop=300) と Diff (segment_length=25600, hop=256) の
両方を config 経由で切替可能。segment_length / mel パラメータは config 経由のみで
受け取り、ハードコードしない (T-M1.3 SoT 原則と整合)。

確定根拠:
- docs/architecture.md §6.5 / §6.6 (mel 抽出 + audio 正規化)
- docs/training.md §1.2 / §1.2.5 / §1.3 (Vocos sox norm + segment 長)
- docs/tickets/T-M0.3-libritts-r.md (filelist TSV header 仕様)
- docs/tickets/T-M1.3-mel-spectrogram.md (LogMelSpectrogram.from_config())
- docs/tickets/T-M1.6-sub-model.md (CONCAT_ORDER, center=True 整合性)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from wavenext2.data.mel import LogMelSpectrogram


class LibriTTSRDataset(Dataset):
    """LibriTTS-R loader + sox `norm` 正規化 + 反射 pad / random crop.

    Args:
        filelist_path: T-M0.3 で生成した TSV (header あり) の Path。
            列: rel_path, n_samples, speaker_id, chapter_id, duration_sec,
                peak_dbfs, rms_dbfs
        root_dir: LibriTTS-R 展開ルート (例: `D:/datasets/LibriTTS_R/`).
            filelist の rel_path とこの root を結合して absolute path を作る。
        segment_length: 切り出すサンプル数。GAN=16384, Diff=25600 を config で渡す。
            hop_length の倍数であることを要請 (`mel.shape[1] * hop == segment_length`
            または `(mel.shape[1] - 1) * hop == segment_length` を center=True 規約
            に応じて担保)。
        hop_length: mel 抽出の hop_length。GAN=300, Diff=256. segment_length と
            T_mel の整合性検証に使用。
        mel_cfg: T-M1.3 LogMelSpectrogram.from_config() に渡す dict.
        mode: "train" → sox norm gain U(-6, -1) dB, "val" → -3 dB 固定.
        sample_rate: 24000 Hz (LibriTTS-R 固定).
        seed: random crop / gain 再現用 (None なら毎回 random).

    Yields:
        __getitem__(idx) → (log_mel: (n_mels, T_mel), audio: (segment_length,))
    """

    SAMPLE_RATE: int = 24000  # LibriTTS-R 固定値、SoT 化

    def __init__(
        self,
        filelist_path: Path,
        root_dir: Path,
        segment_length: int,
        hop_length: int,
        mel_cfg: dict,
        mode: Literal["train", "val"] = "train",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # SoT 原則: hop_length / segment_length のハードコード禁止、必須引数化
        assert segment_length > 0
        assert hop_length > 0
        assert mode in {"train", "val"}, f"mode must be 'train' or 'val' (got {mode})"

        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.mode = mode
        self._rng = np.random.default_rng(seed)

        # filelist 読み込み (TSV header 付き)
        self.rows: list[dict] = self._load_filelist(filelist_path)
        # BucketSampler が使う n_samples 列を attribute で公開
        self.n_samples: list[int] = [int(r["n_samples"]) for r in self.rows]

        # LogMelSpectrogram (T-M1.3) を factory 経由でインスタンス化
        self.mel_transform = LogMelSpectrogram.from_config(mel_cfg)

        # mel_cfg の hop と Dataset の hop_length が一致することを検証 (SoT)
        if mel_cfg["hop_length"] != hop_length:
            raise ValueError(
                f"hop_length mismatch: dataset={hop_length} vs mel_cfg={mel_cfg['hop_length']}. "
                f"Both must come from the same YAML config section."
            )

    @staticmethod
    def _load_filelist(filelist_path: Path) -> list[dict]:
        """T-M0.3 で生成した TSV (header あり) を csv.DictReader で読む."""
        with open(filelist_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        if not rows:
            raise ValueError(f"filelist is empty: {filelist_path}")
        # 必須キーの存在チェック (T-M0.3 と整合)
        required = {"rel_path", "n_samples", "speaker_id"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"filelist missing required columns: {missing}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        wav_path = self.root_dir / row["rel_path"]

        # 1. 波形読み込み (mono 化)
        audio, sr = torchaudio.load(str(wav_path))   # (C, T_raw)
        if sr != self.SAMPLE_RATE:
            raise ValueError(f"sample rate mismatch: expected 24000, got {sr} for {wav_path}")
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)   # stereo → mono

        # 2. sox norm 正規化 (gain は mode で切替)
        gain_db = self._sample_gain_db()
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio, self.SAMPLE_RATE, [["norm", f"{gain_db:.2f}"]]
        )                                              # (1, T_raw)
        audio = audio.squeeze(0)                       # (T_raw,)

        # 3. 反射 pad / random crop で segment_length に揃える
        audio = self._fit_segment(audio)               # (segment_length,)

        # 4. log-mel 抽出 (on-the-fly)
        log_mel = self.mel_transform(audio.unsqueeze(0)).squeeze(0)  # (n_mels, T_mel)

        # 5. 時間長整合性チェック (center=True 規約は T-M1.3 で pin)
        self._assert_time_alignment(audio, log_mel)

        return log_mel, audio

    def _sample_gain_db(self) -> float:
        """sox norm gain を mode に応じて返す."""
        if self.mode == "train":
            # Vocos: gain_db = np.random.uniform(-1, -6)
            return float(self._rng.uniform(-6.0, -1.0))
        else:  # mode == "val"
            return -3.0

    def _fit_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """反射 pad (短い) / random crop (長い) で segment_length に整える."""
        T = audio.shape[0]
        if T < self.segment_length:
            # F.pad(mode="reflect") は (B, C, T) を想定するため (1, 1, T) に変形
            pad_len = self.segment_length - T
            audio = F.pad(audio.unsqueeze(0).unsqueeze(0), (0, pad_len), mode="reflect")
            audio = audio.squeeze(0).squeeze(0)
        elif T > self.segment_length:
            max_start = T - self.segment_length
            if self.mode == "train":
                start = int(self._rng.integers(0, max_start + 1))
            else:
                start = 0   # val は deterministic に先頭から
            audio = audio[start : start + self.segment_length]
        return audio

    def _assert_time_alignment(self, audio: torch.Tensor, log_mel: torch.Tensor) -> None:
        """center=True 規約 (`T_mel = floor(T_audio/hop) + 1`) に従い整合検証."""
        T_audio = audio.shape[0]
        T_mel = log_mel.shape[1]
        # T-M1.3 で pin した center=True 規約: T_mel = floor(T_audio / hop) + 1
        # ⇒ T_audio が hop の倍数なら T_mel * hop == T_audio + hop (1 hop 余剰)
        # ⇒ または (T_mel - 1) * hop == T_audio が成立
        # 本実装は (T_mel - 1) * hop == segment_length を期待 (T-M1.6 §6.1 critical)
        expected = (T_mel - 1) * self.hop_length
        if expected != T_audio:
            raise ValueError(
                f"audio/mel time mismatch: audio={T_audio}, mel={T_mel}, "
                f"hop={self.hop_length}, expected_audio=(T_mel-1)*hop={expected}. "
                f"T-M1.3 center=True 規約 + T-M1.6 §6.1 で pin した整合性違反"
            )

    @classmethod
    def from_config(
        cls, cfg: dict, mode: Literal["train", "val"] = "train", seed: int | None = None
    ) -> "LibriTTSRDataset":
        """Config dict から factory で生成 (T-M1.3 / T-M1.6 の from_config と整合).

        Args:
            cfg: 例 `{"filelist": "data/filelists/train.tsv",
                     "root_dir": "D:/datasets/LibriTTS_R",
                     "segment_length": 16384, "hop_length": 300,
                     "mel": {...}}`
            mode: "train" or "val".

        Returns:
            LibriTTSRDataset instance
        """
        return cls(
            filelist_path=Path(cfg["filelist"]),
            root_dir=Path(cfg["root_dir"]),
            segment_length=cfg["segment_length"],
            hop_length=cfg["hop_length"],
            mel_cfg=cfg["mel"],
            mode=mode,
            seed=seed,
        )
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | GAN 値 | Diff 値 | 出典 |
|---|---|---|---|
| sample_rate | 24000 | 24000 | docs/training.md §1.1, LibriTTS-R 固定 |
| segment_length | **16384** | **25600** | docs/training.md §1.3 / docs/milestones.md §M2.1 |
| hop_length | 300 | 256 | docs/training.md §1.2 |
| mode="train" gain | U(-6, -1) dB | U(-6, -1) dB | docs/training.md §1.2.5 / docs/architecture.md §6.6 (Vocos) |
| mode="val" gain | -3 dB | -3 dB | docs/training.md §1.2.5 (Vocos val/inference 固定値) |
| 短い wav 拡張 | 反射 pad | 反射 pad | 慣例 (HiFi-GAN / WaveFit-PT) |
| 長い wav 切り出し | random crop (train) / 先頭固定 (val) | 同左 | 慣例 |
| filelist 形式 | TSV (header) | TSV (header) | T-M0.3 |
| log type | natural log (`torch.log`) | 同左 | T-M1.3 (Vocos `safe_log`) |
| log eps | 1e-7 (M1 暫定) | 1e-7 (M1 暫定) | T-M1.3 (M2 smoke 後に 1e-5 へ上げる可能性) |
| center=True 整合 | `(T_mel - 1) * hop == T_audio` | 同左 | T-M1.3 / T-M1.6 §6.1 critical で pin |

### 2.4 アルゴリズム / 処理フロー

1. **`__init__`**:
   - 必須引数 (`segment_length`, `hop_length`, `mel_cfg`) を SoT に従い hardcode default なしで受領
   - `mode in {"train", "val"}` を assert
   - filelist TSV を `csv.DictReader` で読み、`rel_path / n_samples / speaker_id / ...` 列を保持
   - `n_samples` 列を `self.n_samples` (list[int]) として attribute 公開 (BucketSampler 用)
   - `LogMelSpectrogram.from_config(mel_cfg)` で T-M1.3 の factory 経由インスタンス化
   - **`mel_cfg["hop_length"] == hop_length`** を runtime check (SoT drift 防止)
2. **`__getitem__(idx)`**:
   - `row["rel_path"]` を `root_dir` と join → wav の absolute path
   - `torchaudio.load(wav_path)` → `(C, T_raw)`、sr 検証 (== 24000)
   - stereo → mono (`audio.mean(dim=0, keepdim=True)`)
   - `_sample_gain_db()` で mode に応じた gain (train=U(-6,-1), val=-3.0)
   - `torchaudio.sox_effects.apply_effects_tensor(audio, sr, [["norm", f"{gain_db:.2f}"]])`
   - `_fit_segment(audio)` で **反射 pad** (短い) / **random crop** (長い)
   - `self.mel_transform(audio.unsqueeze(0)).squeeze(0)` で `(n_mels, T_mel)`
   - `_assert_time_alignment(audio, log_mel)` で `(T_mel - 1) * hop == T_audio` を検証
3. **`_fit_segment`**:
   - `T < segment_length`: `F.pad(..., mode="reflect")` で末尾を反射 pad
   - `T == segment_length`: 何もしない
   - `T > segment_length`: train mode は `[start, start+segment_length]` のランダム crop、val mode は先頭固定 (`start=0`) で deterministic
4. **`_assert_time_alignment`**:
   - center=True 規約: `T_mel = floor(T_audio/hop) + 1` → `(T_mel - 1) * hop == T_audio` (T-M1.3 で pin)
   - 不一致なら `ValueError` (silent failure 防止)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `dataset.py` 実装 + `tests/test_dataset.py` 記述 | general-purpose |
| Reviewer | 1 | docs/training.md §1.2.5 / docs/architecture.md §6.6 整合性 + Vocos `vocos/dataset.py` との非コピー確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_dataset.py -v` 実行 + Windows での sox_effects 動作確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes**
  - T-M2.2 (Discriminator), T-M2.3 (Losses) と並列可能 (依存なし、ファイル独立)
- 並列実行する場合の最大並列数: 3 (M2.1 / M2.2 / M2.3 同時)
- 依存: T-M0.3 (filelist) と T-M1.3 (LogMelSpectrogram) 完了が前提

## 4. 提供範囲 (Scope)

### In Scope
- `LibriTTSRDataset(Dataset)` クラス本実装
- `__init__(filelist_path, root_dir, segment_length, hop_length, mel_cfg, mode, seed)` シグネチャ
- `from_config(cfg, mode, seed)` classmethod factory (T-M1.3 / T-M1.6 と一貫)
- TSV filelist の `csv.DictReader` 読み込み
- `n_samples` 列を attribute 公開 (BucketSampler 補助)
- sox `norm` 正規化 (train: U(-6, -1) dB, val: -3 dB)
- 反射 pad / random crop / val deterministic 切り出し
- `LogMelSpectrogram.from_config()` 利用
- center=True 整合性 runtime check
- `tests/test_dataset.py` 新規作成 (Acceptance 全項目)
- `configs/{gan,diff}_wavenext2.yaml` の `data:` セクション値確定 (segment_length, hop_length, root_dir, filelist key)

### Out of Scope
- BucketSampler 本実装 (T-M2.5 で `torch.utils.data.Sampler` 継承 + `n_samples` 列消費)
- LengthSampler / random sampler の選定 (§8.1 案、M5 で再評価)
- mel precompute スクリプト (`scripts/extract_mel.py`) - 本チケットは on-the-fly のみ実装、precompute は §8.1 で M5 phase review 再評価
- DataLoader 本体の構築 (T-M2.5 で `DataLoader(dataset, batch_size, num_workers, ...)`)
- webdataset / tar shard 化 (§8.1 M6 視点で再評価)
- Multi-GPU 対応 (DistributedSampler は T-M6.1 / T-M6.2 で導入)
- AMP / fp16 入力 (M2.5 train_gan で必要なら再評価)
- speaker-balanced batch sampler (§8.1, M2 smoke 後判断)

### Deliverable
- ファイル:
  - `src/wavenext2/data/dataset.py` (本実装)
  - `tests/test_dataset.py` (新規)
  - `src/wavenext2/data/__init__.py` (`__all__` 追加)
- 関数 / クラス:
  - `class LibriTTSRDataset(Dataset)`
  - `LibriTTSRDataset.from_config(cfg, mode, seed)` classmethod
- ドキュメント差分:
  - `docs/milestones.md` §M2.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M2.1 ステータス更新
  - `configs/{gan,diff}_wavenext2.yaml` の `data:` セクション (segment_length / hop_length / root_dir / filelist) 追加 or 更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_dataset.py`)

#### 基本 shape / 整合性
- [ ] `test_gan_getitem_shape`: GAN config (segment_length=16384, hop=300) で `__getitem__(0)` → `(mel.shape == (128, T_mel_gan), audio.shape == (16384,))`、`T_mel_gan` は T-M1.3 で pin した実機値
- [ ] `test_diff_getitem_shape`: Diff config (segment_length=25600, hop=256) で `(mel.shape == (128, T_mel_diff), audio.shape == (25600,))`
- [ ] `test_time_alignment_assert`: `(mel.shape[1] - 1) * hop_length == audio.shape[0]` (center=True 規約) を全 sample で確認
- [ ] `test_batch_collation`: `DataLoader(dataset, batch_size=4)` で 1 batch 取り出して `(B, 128, T_mel) + (B, segment_length)` の shape を確認

#### sox norm gain 検証
- [ ] `test_train_mode_gain_range`: mode="train" で 100 sample の peak が gain U(-6, -1) dB の理論値範囲 (`10**(-6/20) ≈ 0.501 ~ 10**(-1/20) ≈ 0.891`) に収まる (各 sample で `audio.abs().max()` を採取)
- [ ] `test_val_mode_peak_fixed`: mode="val" で `audio.abs().max() ≈ 10**(-3/20) ≈ 0.7079` (許容 ±0.02、sox norm は peak-based 正規化のため理論一致)
- [ ] `test_val_mode_deterministic`: mode="val" で同一 idx を 2 回呼んで `torch.equal(audio_1, audio_2)` (random crop が deterministic, gain も固定)

#### 反射 pad / random crop
- [ ] `test_short_wav_reflect_pad`: `segment_length` より短い wav (fixture で生成) → 反射 pad で `segment_length` ちょうどに伸ばす
- [ ] `test_long_wav_random_crop_train`: 長い wav + mode="train" で seed 違いの 2 回呼びで `not torch.equal(audio_1, audio_2)` (異なる region から crop)
- [ ] `test_long_wav_deterministic_crop_val`: 長い wav + mode="val" で 2 回呼び `torch.equal(audio_1, audio_2)` (先頭固定)

#### filelist / config 経由
- [ ] `test_filelist_tsv_header_required`: header なし TSV を渡して `ValueError`
- [ ] `test_filelist_empty_raises`: 空 TSV で `ValueError`
- [ ] `test_n_samples_attribute`: `dataset.n_samples` が len(dataset) と同じ長さ、すべて int
- [ ] `test_from_config_factory`: `LibriTTSRDataset.from_config(cfg, mode="train")` で生成可能
- [ ] `test_hop_length_mismatch_raises`: `mel_cfg["hop_length"] != hop_length` のとき `ValueError` (SoT drift 検出)

#### 不正入力
- [ ] `test_invalid_mode_raises`: `mode="test"` 等で `AssertionError`
- [ ] `test_wav_sample_rate_check`: 24000 以外の wav (mock) で `ValueError`

### 5.2 e2e / 結合テスト
- [ ] `uv run python -c "from wavenext2.data.dataset import LibriTTSRDataset; print(LibriTTSRDataset)"` で import 成立
- [ ] LibriTTS-R sample 数件 (test-clean から fixture) で `dataset = LibriTTSRDataset.from_config(cfg)` → `len(dataset) > 0` → `dataset[0]` が成功 (T-M0.3 完了後に実行可能、それまで `@pytest.mark.slow` で skip)
- [ ] **`test_mel_stft_grad_balance` (M2 phase review 追加)**: `mel` と `STFT-spec` の dynamic range mismatch 懸念 (T-M1.2 / T-M1.6 §6.1 から伝搬) を観測。`Conv1d.weight.grad[:, :128, :].norm() / Conv1d.weight.grad[:, 128:, :].norm()` を log、1% 未満なら T-M1.2 へフィードバック (本チケットでは観測のみ)

### 5.3 Acceptance criteria (`docs/milestones.md` §M2.1 より転記)
- [ ] train mode: 出力 audio の peak が gain に応じて変化
- [ ] val mode: 出力 audio の peak が ≈ `10**(-3/20)` ≈ 0.708
- [ ] segment_length より短い wav は反射 pad、長いものは random crop
- [ ] mel と audio の時間長整合: `audio.shape[0] == mel.shape[1] * hop_length` (center=True で `(T_mel - 1) * hop_length == T_audio` 規約、本チケットで pin)

### 5.4 テスト戦略 (M2 phase review 反映)
- **fixture 戦略**:
  - `tests/conftest.py` に短い (0.5 秒) / 長い (5 秒) / ちょうど (segment_length / 24000 秒) の 3 種類の wav fixture を session scope で生成 (tmp_path_factory + soundfile)
  - `tiny_filelist_tsv` fixture (header 付き、3 行) を tmp_path で生成
- **GPU/CPU 分離**: `@pytest.mark.cpu` で本チケットのテストは CPU only (mel 抽出は CPU 完結)
- **Windows での sox_effects 動作**: T-M0.1 §6 で torchaudio backend に sox_io が含まれることを CI で確認済み前提。本チケット tests は CI Windows runner で全 pass を確認
- **coverage 目標**: `src/wavenext2/data/dataset.py` の line coverage **90%** 以上 (error path 含む)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (実装着手前に解決)

- **CRITICAL: mel/STFT-spec dynamic range mismatch** (T-M1.2 / T-M1.6 §6.1 から伝搬):
  - `torch.cat([mel, stft_spec], dim=1)` 直後の `Conv1d(2176, 512, k=7)` で STFT-spec の振幅が大きく mel ch [0:128] が無視される懸念
  - **本チケットでは観測のみ** (test_mel_stft_grad_balance §5.2)、修正は T-M1.2 (STFT 側 LayerNorm) または T-M1.4 (入力 LayerNorm) の責務
  - **検知閾値**: `weight[:, :128, :].grad.norm() / weight[:, 128:, :].grad.norm() < 0.01` (mel side grad が STFT 側の 1% 未満) で alert

- **CRITICAL: DataLoader CPU bound** (M2 phase review):
  - on-the-fly mel 抽出は LibriTTS-R 460h の本格訓練 (M6, A100 410h) で **GPU 待ち** が発生するリスク
  - 緩和策 (本チケットでは未着手):
    - 案 A: `scripts/extract_mel.py` で precompute → ディスクキャッシュ (M5 phase review で再評価、§8.1 参照)
    - 案 B: `num_workers=8` で並列読み (T-M2.5 train_gan で設定)
    - 案 C: webdataset shard 化 (§8.1, M6 視点)
  - **本チケットでは on-the-fly + num_workers で対応**、M5 smoke で GPU 使用率 < 70% を検知した時点で案 A へ移行

- **CRITICAL: Windows での `sox_effects` 動作** (T-M0.1 §6 整合):
  - torchaudio backend に `sox_io` が含まれていない場合 `RuntimeError`
  - T-M0.1 Acceptance で `sox_io in backends` を CI 検証済みのはずだが、本チケット完了時に **再確認** (`torchaudio.list_audio_backends()`)
  - Windows 環境で `sox_io` が機能しない場合の fallback として `torchaudio.functional.gain` を §8.1 で代替案として残す

#### 通常項目

- **center=True 規約の T_mel +1 ずれ** (T-M1.3 §6.1):
  - `T_mel = floor(T_audio / hop) + 1` で `(T_mel - 1) * hop == T_audio` の規約を採用
  - 実機 PoC で T-M1.3 が確定する `T_mel_per_sec` を `configs/*.yaml` の `mel.T_mel_per_sec` に SoT 化し、本 Dataset は参照する設計を採用
  - 不整合検出は `_assert_time_alignment` で fail-fast

- **stereo wav の処理**:
  - LibriTTS-R は mono が多いが念のため `audio.mean(dim=0, keepdim=True)` で safety net (Vocos と同じ)

- **`np.random.uniform(-6, -1)` の seed 制御**:
  - test の deterministic 確認のため `np.random.default_rng(seed)` で生成
  - グローバル汚染 (`np.random.seed`) を避ける

- **`torchaudio.load` の dtype**:
  - 戻り値が `float32` で `[-1, 1]` 範囲を期待。一部 wav が int16 で読まれた場合 `[-32768, 32767]` に化けるリスク → torchaudio default で float32 化される (確認)

- **mel.shape[1] と segment_length の整合性 (config 制約)**:
  - GAN: `segment_length = 16384`, `hop = 300` → `T_mel = floor(16384/300) + 1 = 55` → `(T_mel - 1) * hop = 54 * 300 = 16200 ≠ 16384`
  - **`(T_mel - 1) * hop == segment_length`** を担保するには `segment_length` が `hop` の倍数 + 1 でない範囲で多少のズレが許容される必要あり
  - **解決**: `_fit_segment` で `segment_length` ぴったりに切り出し、mel は `floor(segment_length / hop) + 1` 個の frame を返す。本チケットでは `(T_mel - 1) * hop ≤ segment_length < T_mel * hop` の範囲で整合とする (assertion は `<= segment_length < T_mel * hop`)
  - **要再検証**: T-M1.3 完了後の `T_mel_per_sec` 実機値で本 assertion を pin (`tests/test_dataset.py::test_time_alignment_assert` で確定)

- **反射 pad で短い wav が壊れるリスク**:
  - `F.pad(mode="reflect")` は pad_len > T で失敗する (`RuntimeError`)。LibriTTS-R に segment_length より極端に短い wav (例: 0.1 秒) が混ざっていた場合 OOM 前にエラー → T-M0.3 で `--min-duration=1.0` (1 秒) フィルタ済み (`docs/tickets/T-M0.3-libritts-r.md` §2.3) のため通常起きない

- **`np.random` と `torch.utils.data.DataLoader` worker のシード分散**:
  - `num_workers > 0` で各 worker が同じ seed を持つと epoch 毎に同じ random crop になるリスク
  - **対応**: T-M2.5 で `worker_init_fn` を設定 (本チケットは責務外、`README` メモのみ)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` で関連項目はすべて確定済み
- 本チケットで決定する事項 (詳細は §1 確定事項):
  - **`(T_mel - 1) * hop == segment_length` か `T_mel * hop == segment_length` か**: center=True 規約に従い前者を採用 (T-M1.3 規約と整合)
  - **stereo → mono の処理**: `mean(dim=0)` (Vocos 慣例)
  - **val random crop**: deterministic に先頭固定 (`start=0`)
  - **filelist 読み込み**: `csv.DictReader(delimiter="\t")`
  - **`n_samples` を BucketSampler に活用**: attribute `dataset.n_samples` で公開

### 6.3 他チケットとの整合性
- **T-M0.3 (filelist)**: TSV header 列名 (`rel_path`, `n_samples`, `speaker_id`, ...) を完全準拠。本チケットで `csv.DictReader` で読む
- **T-M1.3 (LogMelSpectrogram)**:
  - `LogMelSpectrogram.from_config(mel_cfg)` 経由でインスタンス化
  - center=True 規約 (`T_mel = floor(T_audio/hop) + 1`) を前提とし、本 Dataset で `(T_mel - 1) * hop == segment_length` 整合性を runtime check
  - `T_mel_per_sec` を `configs/*.yaml` の `mel:` セクションに SoT 化することを T-M1.3 へ申し送り (T-M1.3 §9.1 経由で実施済み想定)
- **T-M1.6 (SubModelGAN/Diff)**:
  - center=True 整合性の規約 (`(T_mel - 1) * hop == T_audio`) は本 Dataset で担保。Sub-model 側は `audio.shape[0] == T_mel * hop` を assert していたら不整合 → T-M1.6 §6.1 critical の解決待ち
  - `CONCAT_ORDER = ("mel", "stft_spec")` (T-M1.6) の前提で Dataset は `(mel, audio)` のペアで返す
- **T-M2.5 (train_gan)**:
  - 想定: `DataLoader(dataset, batch_size=16, num_workers=8, sampler=BucketSampler(dataset.n_samples, batch_size=16))`
  - `worker_init_fn` で `np.random` seed を worker_id でずらす責務は T-M2.5
- **T-M3.2 (train_diff)**:
  - 同じ `LibriTTSRDataset` を使い、`segment_length=25600`, `hop_length=256` で config から生成
  - 4 sub-model 訓練で同じ Dataset を 4 回 instantiate (sub-model index は別 config で渡す)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] `docs/training.md` §1.2.5 / docs/architecture.md §6.6 (Vocos sox norm) と一致
- [ ] `docs/training.md` §1.3 segment_length (16384 / 25600) が config に hardcode されず引数経由
- [ ] T-M0.3 の filelist TSV header 列名と `csv.DictReader` 読み込み列名が一致
- [ ] `LogMelSpectrogram.from_config()` (T-M1.3) を経由しているか (直接 `LogMelSpectrogram(...)` の literal 渡しでないこと)
- [ ] `mel_cfg["hop_length"] == hop_length` の runtime check が入っている (SoT drift 防止)
- [ ] sox `norm` の gain 文字列が `f"{gain_db:.2f}"` (Vocos と同じ書式)
- [ ] 反射 pad が `F.pad(..., mode="reflect")` で実装され、tensor の次元変換 (unsqueeze/squeeze) が正しい
- [ ] random crop が `np.random.default_rng(seed)` 経由 (グローバル汚染なし)
- [ ] val mode の crop が deterministic (`start=0`)
- [ ] center=True 規約 (`(T_mel - 1) * hop == T_audio`) が `_assert_time_alignment` で fail-fast
- [ ] `from_config(cfg, mode, seed)` factory が T-M1.3 / T-M1.6 の `from_config` と一貫
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`)
- [ ] エラー処理: 不正 mode / 24000 != sr / hop_length mismatch / empty filelist で fail-fast
- [ ] Vocos `vocos/dataset.py` をコピーしていない (再構成)
- [ ] `n_samples` attribute が `list[int]` で公開され、BucketSampler が import 可能
- [ ] テストが GAN/Diff 両 config で gain / shape / 整合性を pass

## 8. ゼロから作り直すとしたら

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **precompute mel** (`scripts/extract_mel.py` で全 wav の log-mel を `.npy` キャッシュ) | DataLoader CPU 負荷激減、本格訓練 (M6 410h) で GPU 待ち減 | キャッシュ容量 (LibriTTS-R 460h × 128 mel × ~80 frames/sec ≈ 数十 GB)、sox norm を pre-apply するか on-the-fly かの判断必要、`scripts/extract_mel.py` の保守 | M2 smoke では on-the-fly で十分。**M5 phase review で DataLoader bottleneck (GPU < 70%) 検知時に移行** | **M5 smoke で GPU 利用率 < 70%** |
| **BucketSampler** (`n_samples` 列で長さ別 bucket、bucket 内 random) | 同一 batch 内の長さ揃いで pad ロス削減 | 実装複雑、`segment_length` で固定長切り出すため bucket の効用は限定的 | 固定長切り出し時は不要 (本チケットの segment_length 切り出しなら全 sample 同長)。**ただし M5 で可変長 batch 化する場合は採用** | M5 で可変長 batch 設計時 |
| **LengthSampler** (短いものから順次バッチ) | curriculum learning 効果 | LibriTTS-R は十分大きいデータセットなので curriculum 不要 | 慣例ではない、効果不明 | (再評価しない) |
| **random Sampler** (全 wav からランダム) | 最も単純 | 同左 | **本チケットでは `Sampler` を明示せず DataLoader default の random に従う**、T-M2.5 で BucketSampler を採用するか判断 | T-M2.5 実装時 |
| **`torchaudio.functional.gain` で sox norm 代替** | sox_effects 依存を排除 (Windows 環境で `sox_io` バックエンドが動かない場合の fallback) | gain 計算が peak-based でなく単純ゲインなので Vocos と数値非互換、品質劣化リスク | Vocos `vocos/dataset.py` 完全準拠で sox `norm` 採用、再現性優先 | **T-M0.1 Acceptance で sox_io が CI で fail した場合に切替** |
| **`librosa.util.normalize`** で peak 正規化 | numpy のみで動く | sox の dB 指定と完全互換でない、Vocos と数値ずれ | Vocos との数値完全一致を優先 | (再評価しない) |
| **webdataset 化** (`tar` shard) | M6 クラウド A100 訓練でストリーミング可、shard 並列で I/O 高速化 | shard 設計が必要、ローカル開発と非対称 | M5 完了時に I/O 律速が判明したら採用 | **M5 完了時に I/O 律速判明** |
| **HuggingFace `datasets` 使用** | filelist 不要、streaming 可 | LibriTTS-R mirror の存在確認必要、Vocos 慣例から外れる | T-M0.3 で openslr 経由を採用したため整合 | M0.3 mirror 採用時 |
| **`torchaudio.datasets.LIBRITTS_R`** (将来 torchaudio に追加されたら) | 公式 wrapper を使える | torchaudio 0.x 時点では未提供、依存バージョン上げが必要 | 自前実装で十分、torchaudio 1.x で追加されたら再評価 | torchaudio が `LIBRITTS_R` を提供開始した時点 |
| **on-the-fly + GPU 上で mel 計算** (`mel_transform.to("cuda")`) | DataLoader 不要で速い | バッチ全体を GPU メモリに乗せる必要があり OOM リスク、DataLoader の num_workers benefit を捨てる | num_workers 並列を活かす設計を優先 | M5 で num_workers 効果が薄い場合 |
| **stereo 対応 (mono 化せず 2ch のまま)** | データ情報量増 | LibriTTS-R は mono が大半、Vocos と非互換 | Vocos 慣例 (mean(dim=0)) を踏襲 | (再評価しない) |
| **dataset を `webdataset` 化 (M6 視点)** | 大規模分散訓練に最適 | shard 設計コスト | **M6 で本格訓練起動時に再評価**、本チケットでは見送り | M6 起動時に再評価 |
| **mel と audio を別ファイルキャッシュ** | mel cache + audio on-the-fly のハイブリッド | キャッシュ管理が複雑 | precompute するなら全て precompute、しないなら全て on-the-fly が単純 | (再評価しない) |
| **speaker-balanced batch sampler** | 1 batch 内で話者多様性、特定話者過学習防止 | 実装複雑、効果は smoke で確認後判断 | M2 smoke で speaker bias が観測されたら採用 | M2 smoke で speaker bias 観測 |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度**: 適切 (size=M)。LibriTTSRDataset 1 クラス + sox norm + 反射 pad/crop + mel 結合の thin layer。T-M0.3 (filelist) と T-M1.3 (mel) の集約点
- **factory パターン一貫化**: T-M1.3 / T-M1.6 と同じく `from_config(cls, cfg, mode, seed)` を採用。M1 phase review の横断方針と整合
- **SoT 強制**: `mel_cfg["hop_length"] == hop_length` の runtime check で SoT drift を fail-fast
- **責務分離**:
  - Dataset = 「filelist 読み込み + 正規化 + 切り出し + mel 抽出 + 整合性検証」までで止める
  - BucketSampler / DataLoader 設定 / worker_init_fn = T-M2.5 / T-M3.2 の責務
  - mel precompute = `scripts/extract_mel.py` の責務 (本チケットでは作らない)
  - post-filter dev set 切り出し = T-M3.4 の責務 (本チケットは `dev_postfilter.tsv` を読む Dataset としては未対応、T-M3.4 で `LibriTTSRDataset(filelist=dev_postfilter.tsv, mode="val")` を使う想定)

### 8.3 再評価トリガー条件まとめ
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| on-the-fly mel | M5 smoke | GPU 利用率 < 70% で precompute へ |
| Sampler 戦略 (random) | T-M2.5 / M5 | BucketSampler / LengthSampler 採用 |
| sox `norm` | T-M0.1 sox_io 検証 | Windows で fail なら `torchaudio.functional.gain` へ fallback |
| webdataset | M5 完了 / M6 起動 | I/O 律速判明時 |
| speaker-balanced batch | M2 smoke | speaker bias 観測時 |

### 8.4 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外: (未記入)
- 次の似たタスクで応用できる教訓: (未記入)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.5 (train_gan)
- **使用方法**:
  ```python
  from wavenext2.data.dataset import LibriTTSRDataset
  from torch.utils.data import DataLoader

  train_dataset = LibriTTSRDataset.from_config(cfg["data_train"], mode="train")
  val_dataset = LibriTTSRDataset.from_config(cfg["data_val"], mode="val")

  train_loader = DataLoader(
      train_dataset,
      batch_size=16,
      num_workers=8,
      shuffle=True,
      pin_memory=True,
      worker_init_fn=seed_worker,  # T-M2.5 で実装、np.random worker 分散
  )
  ```
- **重要事項**:
  - `dataset.n_samples` (list[int]) を BucketSampler に渡せる (本チケットで属性公開)
  - `worker_init_fn` の実装は T-M2.5 の責務 (本チケット未着手)
  - `cfg["data_train"]["segment_length"] = 16384`、`cfg["data_val"]["segment_length"] = 16384`
  - `cfg["data_train"]["mel"]` は GAN config (n_fft=2048, hop=300, win=1200)
  - DataLoader CPU bound 検知時の precompute 移行 (§8.1) を T-M5.1 で評価
- **想定 epoch step 数**: train.tsv ~145k 行 / batch_size=16 ≈ 9k step/epoch

#### T-M3.2 (train_diff)
- **使用方法**:
  ```python
  diff_dataset = LibriTTSRDataset.from_config(cfg["data_train"], mode="train")
  # cfg["data_train"]["segment_length"] = 25600
  # cfg["data_train"]["hop_length"] = 256
  # cfg["data_train"]["mel"] = {n_fft=1024, hop=256, win=1024, ...}
  ```
- **重要事項**:
  - `segment_length=25600` (FastDiff 慣例) で GAN と切替
  - 4 sub-model 訓練で同じ Dataset インスタンスを共有可能 (band sampler 側で noise level を切替)
  - batch_size=20 (FastDiff default)、num_workers=8

#### T-M2.6 (GAN smoke) / T-M3.5 (Diff smoke)
- **使用方法**: `LibriTTSRDataset.from_config(cfg, mode="val", seed=42)` で 1 utterance を deterministic 取得
- **想定**: smoke は val mode + seed 固定 + `dataset[0]` の 1 sample で over-fitting test

#### T-M3.4 (Post-filter)
- **使用方法**: `LibriTTSRDataset.from_config({"filelist": "data/filelists/dev_postfilter.tsv", "root_dir": ..., "segment_length": 24000, "hop_length": 256, "mel": ...}, mode="val")` で 200 utterances を deterministic 読み込み
- **重要事項**: post-filter fit は **whole utterance** を使うため segment_length は十分大きく設定 (例: 最大 15 秒 × 24000 = 360000)。固定長切り出しが不適切な場合は T-M3.4 で `LibriTTSRDataset` を継承して `_fit_segment` を override する設計

#### T-M5.1 (1 epoch smoke) / T-M5.2 (Diff smoke)
- **想定**: train.tsv 全量 (145k 行) で 1 epoch、`num_workers=8` で I/O bottleneck を観測
- **GPU 利用率 < 70%** で **§8.1 案「precompute mel」を起動**、`scripts/extract_mel.py` で全 wav の log-mel を `.npy` キャッシュ

#### T-M1.3 (LogMelSpectrogram) との連絡
- 本チケット完了時に `mel_cfg["hop_length"] == hop_length` の runtime check が動作することを T-M1.3 へフィードバック
- T-M1.3 で確定した `T_mel_per_sec` (実機 PoC 値) を `configs/*.yaml` の `mel:` セクションに SoT 化することを T-M1.3 §9.1 経由で要請済み想定
- center=True 規約 (`(T_mel - 1) * hop == segment_length`) を本 Dataset で fail-fast 検証

#### T-M1.6 (Sub-model) との連絡
- T-M1.6 §6.1 critical 項目 `y_prev.shape == T_mel * hop` か `(T_mel - 1) * hop` かの選択を確定する必要あり
- 本 Dataset は **`(T_mel - 1) * hop == segment_length` 規約** を採用するため、T-M1.6 / T-M2.4 (GAN モデル) で `y_prev = torch.zeros_like(audio)` を生成する際に `audio.shape[0] = (T_mel - 1) * hop` 想定で OK

### 9.2 設定値 (YAML)
- 完了時に `configs/{gan,diff}_wavenext2.yaml` の `data:` セクションに以下を追加:
  ```yaml
  # configs/gan_wavenext2.yaml
  data_train:
    filelist: data/filelists/train.tsv
    root_dir: ${LIBRITTSR_ROOT}  # 環境変数 or absolute path
    segment_length: 16384
    hop_length: 300
    mel:
      sample_rate: 24000
      n_fft: 2048
      hop_length: 300
      win_length: 1200
      n_mels: 128
      f_min: 20.0
      f_max: 12000.0
      eps: 1.0e-7  # M1 暫定
  data_val:
    filelist: data/filelists/val.tsv
    root_dir: ${LIBRITTSR_ROOT}
    segment_length: 16384
    hop_length: 300
    mel: { ... }
  ```
  Diff 側は `segment_length: 25600, hop_length: 256, mel: {n_fft: 1024, win_length: 1024, hop_length: 256, ...}`

### 9.3 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M2.1 の Acceptance 4 項目をチェック
  - [ ] `docs/tickets/index.md` の T-M2.1 ステータスを `pending` → `completed`
  - [ ] `configs/{gan,diff}_wavenext2.yaml` の `data:` セクション差分を commit (§9.2 と一致)
  - [ ] (該当時) `docs/training.md` §1.3 「セグメント長」表記の確定値を更新 (M1.3 phase review で T_mel pin 完了後)

### 9.4 Open question として残ったもの
- **DataLoader bottleneck 判明時の precompute 移行戦略**: M5 smoke 完了時に再評価 (§8.1 「precompute mel」)
- **BucketSampler / LengthSampler 採用判断**: T-M2.5 で `dataset.n_samples` を活用するか判断
- **Windows での sox_effects 動作確証**: T-M0.1 Acceptance で sox_io が CI 検証済みであることを本チケット完了時に再確認、fail なら §8.1 fallback (`torchaudio.functional.gain`) を採用
- **`webdataset` 化**: M6 本格訓練で I/O 律速が判明したら採用検討、`docs/open-questions.md` への追記は本チケット時点では不要
