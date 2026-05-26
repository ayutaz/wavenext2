---
id: T-M2.1
title: LibriTTS-R Dataset + sox norm 正規化
milestone: M2
phase: M2
status: completed
size: M
owner: claude
created: 2026-05-26
updated: 2026-05-27  # 実装完了 (sox→peak-norm, soundfile, time-alignment 修正)
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
- **`__getitem__` 戻り値は `Batch` TypedDict** (M2 phase review 採用昇格): `{"mel": Tensor, "audio": Tensor, "n_samples": int}` で M3 `noise_level` / M4 `speaker_id` 追加に前方互換
- **`return_mel: bool = True` 引数を予約** (M2 phase review 検討追加): M5 で CPU 律速判明時の MelOnGPU 第 3 の選択肢、実装は M5 で行う (本チケットでは引数だけ追加、`False` は `NotImplementedError`)
- **`seed_worker` テンプレ snippet を本チケット §9.1 で提供** (M2 phase review 責務移譲完結): T-M2.5 / T-M3.2 は import するだけ

### ゴール
- [ ] `src/wavenext2/data/dataset.py` に `LibriTTSRDataset(Dataset)` が実装される
- [ ] `__init__(filelist_path, root_dir, segment_length, hop_length, mel_cfg, mode, seed, return_mel=True)` のシグネチャで GAN/Diff 両 config から生成可能
- [ ] `__getitem__` 戻り値は **`Batch` TypedDict** (`{"mel", "audio", "n_samples"}`) — M2 phase review 採用昇格
- [ ] mode="train" で sox norm gain が **U(-6, -1) dB** から sample される (`np.random.uniform(-6, -1)`)
- [ ] mode="val" で sox norm gain が **-3 dB** の固定値
- [ ] 短い wav は **反射 pad** で `segment_length` まで拡張、長い wav は **random crop**
- [ ] mel と audio の時間長整合: `audio.shape[0] == mel.shape[1] * hop_length` (center=True の `T_mel = floor(T_audio/hop)+1` 規約は T-M1.3 で pin 済み)
- [ ] `LogMelSpectrogram.from_config(mel_cfg)` (T-M1.3) を Dataset 内で利用
- [ ] `tests/test_dataset.py` 新規作成、Acceptance 全項目を pytest で網羅 (`uv run pytest tests/test_dataset.py` が pass、**15 秒以内**)
- [ ] `from wavenext2.data.dataset import LibriTTSRDataset, Batch, seed_worker` が import 可能
- [ ] **`seed_worker(worker_id)` 関数** がモジュール末尾に配置され、T-M2.5 / T-M3.2 が import 可能 — M2 phase review 責務移譲完結
- [ ] BucketSampler 実装の補助クラスとして `n_samples` 列を Dataset attribute 経由で公開
- [ ] `return_mel: bool = True` 引数が `__init__` に追加され、`False` の場合は `NotImplementedError` (M5 で MelOnGPU 実装予約)
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

#### M2 phase review 追加テスト
- [ ] `test_getitem_returns_batch_typed_dict`: `__getitem__(0)` の戻り値が `dict` で `{"mel", "audio", "n_samples"}` キーを持つ (`Batch` TypedDict 整合性)
- [ ] `test_return_mel_false_raises_not_implemented`: `LibriTTSRDataset(..., return_mel=False)` の `__getitem__` が `NotImplementedError` (M5 で MelOnGPU 実装予約)
- [ ] `test_dataset_picklable`: `pickle.dumps(dataset)` が成功 (Windows `spawn` 経由の worker 起動 pickle-safe 確認)
- [ ] `test_seed_worker_function_exists`: `from wavenext2.data.dataset import seed_worker` で import 可能、`seed_worker(0)` 呼び出しが例外なし
- [ ] `test_seed_worker_isolates_rng_per_worker`: `seed_worker(0)` と `seed_worker(1)` で `np.random.rand()` 値が異なる (worker_id 分散)
- [ ] `test_gan_184_sample_excess`: GAN config で `audio.shape[0] == 16384`、`(mel.shape[1] - 1) * hop == 16200`、差 184 sample の存在を assert で確認 (Discriminator 側責務として明示)
- [ ] `test_diff_no_excess`: Diff config で `(mel.shape[1] - 1) * hop == segment_length == 25600` の **ぴったり整合** を確認

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

- **`np.random` と `torch.utils.data.DataLoader` worker のシード分散** (M2 phase review 昇格):
  - `num_workers > 0` で各 worker 内で `np.random.default_rng(seed=None)` 由来の seed が同一になり、**同じ epoch で同じ crop が出続ける** 罠 (各 worker が同じ seed sequence を持つため、batch ごとに同じ region を crop する)
  - **対応方針**: `worker_init_fn` で `epoch * num_workers + worker_id` を seed 化 (`np.random.seed(initial_seed + worker_id)` ではなく **epoch も加算** することが必須)
  - **テンプレ snippet を §9.1 に置く**: T-M2.5 / T-M3.2 はこれを import するだけにする (責務移譲を **本チケット §9.1 で完結**、T-M2.5 へは「snippet を import せよ」の連絡のみ)
  - **§9.1 連絡先**: T-M2.5 (GAN training), T-M3.2 (Diff training)

- **`(T_mel - 1) * hop != segment_length` の差 184 sample の責務** (M2 phase review 追加):
  - GAN: `segment_length=16384, hop=300` → `T_mel = floor(16384/300) + 1 = 55` → `(55 - 1) * 300 = 16200`、`16384 - 16200 = 184 sample` の余剰
  - Diff: `segment_length=25600, hop=256` → `T_mel = floor(25600/256) + 1 = 101` → `(101 - 1) * 256 = 25600`、**Diff は ぴったり** 整合 (256 が 25600 の約数)
  - **責務**: Discriminator / MR-STFT loss が末尾 184 sample を無視 (zero pad or crop) するかは別問題 (T-M2.2 / T-M2.3)、**Dataset 内** では以下を担保:
    - 戻り値 audio は **常に `segment_length` ちょうど** (16384 / 25600)
    - 戻り値 mel は **`floor(segment_length / hop) + 1`** frame (GAN=55 / Diff=101)
    - 整合性 assertion は `(T_mel - 1) * hop <= segment_length < T_mel * hop` の範囲 (GAN は **不等式**、Diff は **等式**)
  - **集計時の zero-pad / crop**: 末尾 184 sample は Discriminator / loss に渡る audio に含まれるが mel と整合しない領域。**T-M2.4 (Generator) / T-M2.2 (Discriminator) で末尾扱いを明示** する申し送りを §9.1 に追加

- **`torchaudio.sox_effects` の Linux CI 依存** (M2 phase review 追加):
  - Windows (T-M0.1 §6) では torchaudio backend に `sox_io` が同梱されるが、**Linux CI runner では `apt install libsox-dev` が必要**
  - `.github/workflows/test.yml` の OS matrix に Linux を含める場合は `apt-get install -y libsox-dev` step 追加が必須
  - **本チケットの責務**: T-M0.2 (CI 構築) への申し送り (§9.1)
  - **検知**: 本チケット tests を Linux CI で実行して `sox_io` backend 利用可能を確認

- **Dataset テスト実行時間上限 15s** (M2 phase review 追加):
  - M2 全体の unit test 目標 < 60s (smoke 除く) から逆算、本チケット tests は **15 秒以内** で完了させる
  - 各 test に `@pytest.mark.timeout(5)` を付与 (個別 5 秒)
  - 重い fixture (5 秒 wav 生成) は `session` scope で 1 回だけ生成

- **`spawn` vs `fork` の OS 別挙動** (M2 phase review 昇格):
  - Windows: `spawn` (default) → Dataset を **pickle-safe** にする必要あり (`mel_transform` が `register_buffer` を持つため、worker 起動毎に LogMelSpectrogram を再構築 → 起動コスト増)
  - Linux: `fork` (default) → pickle 不要、worker 起動高速
  - **責務**: T-M2.5 への申し送り (本チケット では `mel_transform` の `register_buffer` (mel filterbank 等) が pickle 可能であることを test で確認 → `test_dataset_picklable` 追加)
  - **§9.1 連絡先**: T-M2.5 / T-M3.2 へ `multiprocessing_context="spawn"` または `"fork"` の明示と `prefetch_factor` 設定の申し送り

- **`(T_mel-1)*hop != segment_length` 差の検証**: 上記 184 sample 余剰の責務確認

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

#### 採用昇格 (M2 phase review)
- **`__getitem__` 戻り値を `dataclass Batch` / `TypedDict` に統一** (現状 tuple → 破壊的変更コスト大):
  - 現状の `(log_mel, audio)` tuple は M3 で `noise_level`、M4 で `speaker_id` を追加する際に **後続全タスクの破壊的変更** になる
  - **採用**: `TypedDict` (Python 3.10+ で `total=False` が使える、最も軽量):
    ```python
    from typing import TypedDict
    class Batch(TypedDict, total=False):
        mel: torch.Tensor      # (n_mels, T_mel)
        audio: torch.Tensor    # (segment_length,)
        n_samples: int         # 元の wav の長さ (BucketSampler 用)
        # M3 で追加: noise_level: torch.Tensor
        # M4 で追加: speaker_id: int
    ```
  - **前方互換**: T-M2.5 / T-M3.2 / T-M3.4 で `batch["mel"], batch["audio"]` のみ参照、M3 で `noise_level` 追加時も既存コード無変更
  - **却下案**: `dataclass(frozen=True)` は collate_fn でフィールド追加が面倒、`NamedTuple` は順序依存
  - **§9.1 連絡先**: T-M2.5 / T-M3.2 / T-M3.4 で `Batch` TypedDict を共有 import

- **`worker_init_fn` テンプレ snippet を本チケット §9.1 に置く**:
  - 責務移譲: 「T-M2.5 の責務」で逃げず、本チケット §9.1 で **完成形 snippet** を提供する
  - T-M2.5 / T-M3.2 は snippet を `from wavenext2.data.dataset import seed_worker` で import するだけ
  - **本体実装**: `src/wavenext2/data/dataset.py` 末尾に `seed_worker(worker_id: int) -> None` を関数として置く (§9.1 でテンプレ snippet を明示)

#### 検討追加 (M2 phase review)
- **`MelOnGPU` モード予約** (`return_mel: bool = True`):
  - `__init__(..., return_mel: bool = True)` 引数を追加。`False` の場合 `__getitem__` は **audio のみ** 返す (mel は collate_fn 後 GPU で抽出)
  - M5 で CPU 律速 (GPU 利用率 < 70%) が判明した時の **第 3 の選択肢** (案 A precompute / 案 B num_workers / **案 D MelOnGPU**)
  - **再評価トリガー**: M5 smoke で CPU 律速かつ precompute がキャッシュ容量で困難な場合
  - **本チケットでの扱い**: `return_mel: bool = True` 引数だけ追加、`False` ブランチは `NotImplementedError` (M5 で実装)

- **`segment_length=24576` (Vocos 流)** をトレードオフ表に追加:
  - HiFi-GAN: 8192 / 本実装 GAN: 16384 / **Vocos**: 24576 / 本実装 Diff: 25600
  - **24576 = 2^14 + 2^13 = 16384 + 8192**、`hop=300` で `T_mel = 82+1 = 83 frame`、`hop=256` で `T_mel = 96+1 = 97 frame`
  - MR-STFT `n_fft=2048` で実質 **`24576 / 2048 = 12 frame** → 8 frame 以上確保、loss 収束が安定
  - **再評価トリガー**: **M5.1 で MR-STFT 収束が遅い場合** (loss curve が plateau)
  - **本チケットでは GAN=16384 維持**、Vocos 24576 は config 切替のみで対応可能 (`segment_length` を YAML 経由)

- **`rms_dbfs` 正規化案** (peak の代わりに RMS 正規化):
  - 現状 sox `norm` は **peak-based** で LibriTTS-R の RMS ばらつきに脆弱 (静かな utterance と大きな utterance で perceptual loudness が揃わない)
  - T-M0.3 で `peak_dbfs` / `rms_dbfs` 両方を filelist 列に持つのに **peak のみ** 使用は勿体ない
  - **却下根拠**: Vocos `vocos/dataset.py` L42-43 も **peak (`norm`)** 採用、再現性優先
  - **代替案**: `torchaudio.transforms.Loudness` (LUFS) や ITU-R BS.1770-4 RMS normalization
  - **再評価トリガー**: M5 smoke で perceptual quality (UTMOS) が低い場合、RMS 正規化との A/B test

#### 既存案

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
| **speaker-balanced batch sampler** | 1 batch 内で話者多様性、特定話者過学習防止 | 実装複雑、効果は smoke で確認後判断 | **M2 smoke は 1 sample over-fit のため speaker bias は観測不可能**。**T-M5.1 (1 epoch) 後の validation MR-STFT 話者分散検査** で判断 | **T-M5.1 (1 epoch) validation MR-STFT 話者分散検査時** |

### 8.2 思想 / 哲学の見直し
- **このサブタスクの粒度**: 適切 (size=M)。LibriTTSRDataset 1 クラス + sox norm + 反射 pad/crop + mel 結合の thin layer。T-M0.3 (filelist) と T-M1.3 (mel) の集約点
- **factory パターン一貫化**: T-M1.3 / T-M1.6 と同じく `from_config(cls, cfg, mode, seed)` を採用。M1 phase review の横断方針と整合
- **SoT 強制**: `mel_cfg["hop_length"] == hop_length` の runtime check で SoT drift を fail-fast
- **mel 抽出を Dataset 内に閉じ込めない哲学** (M2 phase review):
  - 現状: Dataset = 「filelist 読み込み + 正規化 + 切り出し + mel 抽出 + 整合性検証」を全て閉じ込めている
  - **問題**: CPU 律速が判明した時の **唯一の改善経路が precompute だけ** に縛られる (DataLoader 並列を増やしても CPU で mel 抽出は逃れられない)
  - **修正**: `return_mel: bool = True` 引数で raw audio のみ返すモードを **設計上残しておく** (実装は M5 で必要時に行う)
  - **第 3 の選択肢**: 案 A precompute / 案 B num_workers / **案 D MelOnGPU (collate_fn 後 GPU で mel)** という 3 つの逃げ道を確保
  - 哲学: 「**1 つの責務を 1 つの場所に閉じる**」より「**律速判明時に逃げ道を残す**」を優先 (Vocos は閉じ込めているが、本実装は M5/M6 の本格訓練で 410h かかるため逃げ道が重要)
- **責務分離**:
  - Dataset = 「filelist 読み込み + 正規化 + 切り出し + (オプショナル) mel 抽出 + 整合性検証」までで止める
  - BucketSampler / DataLoader 設定 = T-M2.5 / T-M3.2 の責務
  - `worker_init_fn` テンプレ snippet = **本チケット §9.1 で提供**、T-M2.5 / T-M3.2 は import するだけ (M2 phase review で責務移譲を完結)
  - mel precompute = `scripts/extract_mel.py` の責務 (本チケットでは作らない)
  - post-filter dev set 切り出し = T-M3.4 の責務 (本チケットは `dev_postfilter.tsv` を読む Dataset としては未対応、T-M3.4 で `LibriTTSRDataset(filelist=dev_postfilter.tsv, mode="val")` を使う想定)

### 8.3 再評価トリガー条件まとめ
| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| on-the-fly mel | M5 smoke | GPU 利用率 < 70% で precompute / MelOnGPU へ |
| `return_mel: bool` MelOnGPU 第 3 の選択肢 | M5 smoke | CPU 律速 + precompute がディスク容量で困難なら採用 |
| Sampler 戦略 (random) | T-M2.5 / M5 | BucketSampler / LengthSampler 採用 |
| sox `norm` (peak-based) | T-M0.1 sox_io 検証 / M5 UTMOS 低下 | Windows fail なら `torchaudio.functional.gain` へ fallback / 知覚品質低下なら `rms_dbfs` 正規化 |
| webdataset | M5 完了 / M6 起動 | I/O 律速判明時 |
| speaker-balanced batch | **T-M5.1 (1 epoch) validation MR-STFT 話者分散検査** | 話者分散が極端な場合に採用 (M2 smoke は 1 sample で不可能) |
| `segment_length=24576` (Vocos 流) | **M5.1 MR-STFT 収束が遅い場合** | GAN config を 16384 → 24576 へ切替 |
| `Batch` TypedDict 戻り値 | **本チケットで採用済** | M3 で `noise_level`、M4 で `speaker_id` 追加に前方互換 |

### 8.4 学んだこと (2026-05-27 実装完了後に追記)

実装結果:
- `LibriTTSRDataset` 実装、`tests/test_dataset.py` 15 件 pass (合成 wav + TSV)。`Batch` TypedDict、`seed_worker`、`n_samples` attribute、`from_config`、`return_mel=False` は NotImplementedError 予約。

チケット記述からの確定的変更 (M0.1 / 整合性検算より):
1. **sox `norm` → peak 正規化 + soundfile** (torchaudio 2.11 で sox_effects/load 廃止、T-M0.1 §9.1)。`gain = 10**(target_dbfs/20) / peak; audio *= gain` で sox norm と数値等価。`test_peak_normalize_val_target` で peak→10^(-3/20)=0.708 を検証。**チケット §2.2 の `torchaudio.load`/`sox_effects.apply_effects_tensor` は使わない**。
2. **time-alignment 式を修正 (重要)**: チケットの `(T_mel-1)*hop == T_audio` は **GAN segment 16384 (hop 300 の倍数でない) で破綻** (54*300=16200≠16384)。torchaudio center=True の正しい不変量は **`T_mel == 1 + T_audio//hop`**。これで全 segment 長に対応。`_assert_time_alignment` はこの式で検証。
   - **副次的帰結 (T-M2.4 申し送り)**: generator 出力長 = T_mel*hop は segment_length と **hop ぶんずれる** (center=True の +1 frame、Vocos は iSTFT で吸収するが WaveNeXt の linear head は T_mel*hop を出す)。**fixed-point / loss で generator 出力を segment_length に crop する責務は T-M2.4/T-M2.5**。
3. **極端に短い wav は反射 pad 不可** (reflect は pad<input が必要) → tile fallback を追加 (`test_very_short_audio_tiled`)。

次の似たタスクで応用できる教訓:
- 信号長の不変量はマジック式 (`(T-1)*hop`) を鵜呑みにせず、実フレームワーク (torchaudio center=True) の定義で検算する。segment_length が hop の倍数かで式が変わる。
- peak 正規化は crop 後だと peak が保たれない (crop が peak を含まない場合) ため、**正規化 → crop の順**にし、peak テストは正規化メソッド単体 or pad ケースで行う。

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M2.5 (train_gan)
- **使用方法**:
  ```python
  from wavenext2.data.dataset import LibriTTSRDataset, seed_worker, Batch

  train_dataset = LibriTTSRDataset.from_config(cfg["data_train"], mode="train")
  val_dataset = LibriTTSRDataset.from_config(cfg["data_val"], mode="val")

  train_loader = DataLoader(
      train_dataset,
      batch_size=16,
      num_workers=8,
      shuffle=True,
      pin_memory=True,
      worker_init_fn=seed_worker,           # 本チケットで提供する snippet
      prefetch_factor=2,                    # default、I/O 律速時は 4 へ
      persistent_workers=True,              # epoch 間で worker 再生成回避
      multiprocessing_context="spawn",      # Windows / Linux 共通
  )
  ```
- **`worker_init_fn` テンプレ snippet (本チケット §9.1 で提供、T-M2.5 / T-M3.2 はこれを import)**:
  ```python
  # src/wavenext2/data/dataset.py 末尾に配置
  import numpy as np
  import torch

  def seed_worker(worker_id: int) -> None:
      """DataLoader worker_init_fn: epoch * num_workers + worker_id で seed.

      `num_workers > 0` で各 worker が同じ epoch で同じ crop を出す罠を回避.
      `torch.initial_seed()` は DataLoader が `base_seed + worker_id` を毎 epoch
      振り直すため、これを numpy seed に再注入することで全 RNG を同期する.
      """
      worker_seed = torch.initial_seed() % 2**32
      np.random.seed(worker_seed)
      # Dataset 内の `self._rng = np.random.default_rng(seed)` は __init__ 時の
      # seed (None or 固定) を保持するが、worker fork/spawn 後は worker_seed で
      # 上書きする必要あり (DataLoader が `worker_init_fn` 経由で各 worker で再 seed)
      info = torch.utils.data.get_worker_info()
      if info is not None and hasattr(info.dataset, "_rng"):
          info.dataset._rng = np.random.default_rng(worker_seed)
  ```
- **`spawn` vs `fork` 挙動** (M2 phase review):
  - **Windows**: `spawn` (default) → Dataset は **pickle-safe** であること必須。`mel_transform` が `register_buffer` を持つため、worker 起動毎に LogMelSpectrogram を再構築 → 起動コスト増 (但し `persistent_workers=True` で epoch 間は再利用)
  - **Linux**: `fork` (default) → pickle 不要、worker 起動高速。但し CUDA initialized state の inherit に注意 (`fork` 後 CUDA 操作は禁止、`torchaudio.load` のみで OK)
  - **`multiprocessing_context="spawn"` を明示** することで OS 間の挙動差を吸収 (本チケット は test で `test_dataset_picklable` を追加して spawn 動作確認)
- **`prefetch_factor` 明示**:
  - default = 2 (各 worker が 2 batch 先読み)
  - I/O 律速 (M5 smoke で GPU 利用率 < 70%) 検知時は `prefetch_factor=4` へ
- **`Batch` TypedDict 共有** (M2 phase review 採用昇格):
  - `from wavenext2.data.dataset import Batch` で T-M2.5 / T-M3.2 / T-M3.4 が共通利用
  - 現状フィールド: `mel`, `audio`, `n_samples`
  - M3 で `noise_level` 追加、M4 で `speaker_id` 追加 (`TypedDict(total=False)` で前方互換)
- **重要事項**:
  - `dataset.n_samples` (list[int]) を BucketSampler に渡せる (本チケットで属性公開)
  - `cfg["data_train"]["segment_length"] = 16384`、`cfg["data_val"]["segment_length"] = 16384`
  - `cfg["data_train"]["mel"]` は GAN config (n_fft=2048, hop=300, win=1200)
  - DataLoader CPU bound 検知時の precompute 移行 (§8.1) を T-M5.1 で評価
  - 末尾 184 sample の Discriminator / MR-STFT loss での扱い (zero-pad / crop) は T-M2.2 / T-M2.3 で明示
- **想定 epoch step 数**: train.tsv ~145k 行 / batch_size=16 ≈ 9k step/epoch

#### T-M3.2 (train_diff)
- **使用方法**:
  ```python
  from wavenext2.data.dataset import LibriTTSRDataset, seed_worker, Batch

  diff_dataset = LibriTTSRDataset.from_config(cfg["data_train"], mode="train")
  # cfg["data_train"]["segment_length"] = 25600  ← config 経由で切替 (本チケットでは同じ Dataset クラス)
  # cfg["data_train"]["hop_length"] = 256
  # cfg["data_train"]["mel"] = {n_fft=1024, hop=256, win=1024, ...}
  ```
- **重要事項**:
  - **同じ Dataset クラス** で segment_length=25600 へ切替 (config 経由のみ、本体改修不要) — M2 phase review で明示
  - **`Batch` TypedDict 共有**: M3 で `noise_level: torch.Tensor` フィールドを `total=False` で追加 (既存 T-M2.5 コードに無影響)
  - `seed_worker` snippet を共有 import (本チケット §9.1 で提供)
  - `segment_length=25600` (FastDiff 慣例) で GAN と切替 (256 が 25600 の約数なので `(T_mel-1)*hop == segment_length` ぴったり整合、GAN の 184 sample 余剰問題なし)
  - 4 sub-model 訓練で同じ Dataset インスタンスを共有可能 (band sampler 側で noise level を切替)
  - batch_size=20 (FastDiff default)、num_workers=8、`multiprocessing_context="spawn"`

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

#### T-M0.2 (CI 構築) への申し送り (M2 phase review 追加)
- **Linux CI runner で `libsox-dev` 依存** が必要 (`torchaudio.sox_effects` が Linux runner で動作するため)
- **`.github/workflows/test.yml` に apt step 追加要請**:
  ```yaml
  # .github/workflows/test.yml に追加
  - name: Install sox dependency (Linux only)
    if: runner.os == 'Linux'
    run: |
      sudo apt-get update
      sudo apt-get install -y libsox-dev libsox-fmt-all
  ```
- **Windows runner**: torchaudio backend に `sox_io` が同梱されるため追加 install 不要 (T-M0.1 §6 で確認済み前提)
- **macOS runner**: 本プロジェクトでは未対応 (T-M0.1 で Windows + Linux のみサポート決定)
- **本チケットの test** は CI Windows + Linux runner の両方で pass することを確認 (T-M0.2 で CI matrix 設定)

#### T-M2.2 (Discriminator) / T-M2.3 (Loss) への申し送り (M2 phase review 追加)
- **GAN の末尾 184 sample 余剰** (`segment_length=16384, hop=300` → `(T_mel-1)*hop = 16200`、余剰 184 sample) の扱い:
  - audio は 16384 サンプル丸ごと渡るが mel と整合する領域は **先頭 16200 sample**
  - **Discriminator (MSD ×3)**: 全 16384 sample をそのまま処理 (mel との整合は不要、生波形を直接識別)
  - **MR-STFT loss (T-M2.3)**: STFT 計算は audio を直接使用するため 16384 全てが loss に寄与 (但し center=True / padded mode により末尾は reflect で扱われる)
  - **Generator 出力 (T-M2.4)**: `Generator` の出力長 = `T_mel * hop` か `(T_mel-1) * hop` か は T-M2.4 で決定 (本 Dataset は GT audio 側を **`segment_length=16384` ぴったり**で固定するため、Generator が 16200 出力なら **末尾 184 を zero-pad**、16384 出力なら整合不要)
- **Diff の場合**: `segment_length=25600, hop=256` で **`(T_mel-1)*hop = 25600`** ぴったり整合 (余剰なし)

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
- **DataLoader bottleneck 判明時の precompute 移行戦略**: M5 smoke 完了時に再評価 (§8.1 「precompute mel」 / 「MelOnGPU 第 3 の選択肢」)
- **BucketSampler / LengthSampler 採用判断**: T-M2.5 で `dataset.n_samples` を活用するか判断
- **Windows での sox_effects 動作確証**: T-M0.1 Acceptance で sox_io が CI 検証済みであることを本チケット完了時に再確認、fail なら §8.1 fallback (`torchaudio.functional.gain`) を採用
- **Linux CI での `libsox-dev` install**: T-M0.2 申し送りで `.github/workflows/test.yml` に apt step 追加 (M2 phase review 追加)
- **`webdataset` 化**: M6 本格訓練で I/O 律速が判明したら採用検討、`docs/open-questions.md` への追記は本チケット時点では不要
- **`segment_length=24576` (Vocos 流) 切替判断**: M5.1 MR-STFT 収束が遅い場合に GAN config を 16384 → 24576 へ (M2 phase review 追加)
- **`rms_dbfs` 正規化採用判断**: M5 smoke で UTMOS / perceptual quality が低下した場合に peak ベースから RMS ベースへ切替 (M2 phase review 追加)
- **speaker-balanced batch 採用判断**: T-M5.1 (1 epoch) validation MR-STFT 話者分散検査時 (M2 smoke は 1 sample のため不可能、M2 phase review でトリガー変更)
