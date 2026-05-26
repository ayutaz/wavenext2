"""LibriTTS-R 用 Dataset (peak 正規化 + 反射 pad / random crop + log-mel)。

T-M0.3 の filelist (TSV header 付き) を読み、波形を soundfile で読み込み、
**peak 正規化** (Vocos sox `norm` と数値等価: `gain = 10**(target_dbfs/20)/peak`、
train: peak U(-6,-1) dBFS / val: peak -3 dBFS) を施し、固定長 segment に
反射 pad / random crop して log-mel と audio のペアを返す。

GAN (segment_length=16384, hop=300) / Diff (segment_length=25600, hop=256) を
config 経由で切替。segment_length / mel パラメータは config 経由のみ (T-M1.3 SoT)。

音声 I/O は soundfile に統一 (torchaudio 2.11 は load/info/sox_effects 廃止、T-M0.1 §9.1)。

確定根拠: docs/architecture.md §6.5/§6.6、docs/training.md §1.2/§1.2.5/§1.3、
docs/open-questions.md §C4 (peak 正規化)、T-M0.3 (filelist)、T-M1.3 (LogMelSpectrogram)。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from wavenext2.data.mel import LogMelSpectrogram

__all__ = ["Batch", "LibriTTSRDataset", "seed_worker"]


class Batch(TypedDict):
    """__getitem__ / collate の戻り値。M3 noise_level / M4 speaker_id 追加に前方互換。"""

    mel: torch.Tensor  # (n_mels, T_mel)
    audio: torch.Tensor  # (segment_length,)
    n_samples: int  # 元 wav の総サンプル数 (BucketSampler 用)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker の seed を固定 (T-M2.5 / T-M3.2 が import)。"""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    import random as _random

    _random.seed(seed)


class LibriTTSRDataset(Dataset):
    """LibriTTS-R loader + peak 正規化 + 反射 pad / random crop。"""

    SAMPLE_RATE: int = 24000

    def __init__(
        self,
        filelist_path: Path,
        root_dir: Path,
        segment_length: int,
        hop_length: int,
        mel_cfg: dict,
        mode: Literal["train", "val"] = "train",
        seed: int | None = None,
        return_mel: bool = True,
    ) -> None:
        super().__init__()
        if segment_length <= 0 or hop_length <= 0:
            raise ValueError("segment_length and hop_length must be positive")
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode}")
        if not return_mel:
            raise NotImplementedError("return_mel=False (MelOnGPU) is reserved for M5")
        if mel_cfg["hop_length"] != hop_length:
            raise ValueError(
                f"hop_length mismatch: dataset={hop_length} vs mel_cfg={mel_cfg['hop_length']}"
            )

        self.root_dir = Path(root_dir)
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.mode = mode
        self._rng = np.random.default_rng(seed)

        self.rows = self._load_filelist(Path(filelist_path))
        self.n_samples: list[int] = [int(r["n_samples"]) for r in self.rows]
        self.mel_transform = LogMelSpectrogram.from_config(mel_cfg)

    @staticmethod
    def _load_filelist(filelist_path: Path) -> list[dict]:
        with filelist_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            raise ValueError(f"filelist is empty: {filelist_path}")
        missing = {"rel_path", "n_samples", "speaker_id"} - set(rows[0].keys())
        if missing:
            raise ValueError(f"filelist missing required columns: {missing}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Batch:
        row = self.rows[idx]
        wav_path = self.root_dir / row["rel_path"]

        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr != self.SAMPLE_RATE:
            raise ValueError(f"sample rate mismatch: expected 24000, got {sr} for {wav_path}")
        audio = torch.from_numpy(wav)
        if audio.dim() > 1:  # (T, C) → mono
            audio = audio.mean(dim=1)

        audio = self._peak_normalize(audio, self._sample_gain_db())
        audio = self._fit_segment(audio)
        mel = self.mel_transform(audio.unsqueeze(0)).squeeze(0)  # (n_mels, T_mel)
        self._assert_time_alignment(audio, mel)
        return {"mel": mel, "audio": audio, "n_samples": int(row["n_samples"])}

    def _sample_gain_db(self) -> float:
        return float(self._rng.uniform(-6.0, -1.0)) if self.mode == "train" else -3.0

    def _peak_normalize(self, audio: torch.Tensor, target_dbfs: float) -> torch.Tensor:
        """sox `norm -X dB` と数値等価な peak 正規化 (peak を target dBFS に)。"""
        peak = audio.abs().max()
        if peak < 1e-9:  # 無音はそのまま
            return audio
        return audio * (10.0 ** (target_dbfs / 20.0) / peak)

    def _fit_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """反射 pad (短い) / random crop (長い) で segment_length に整える。"""
        t = audio.shape[0]
        if t < self.segment_length:
            pad_len = self.segment_length - t
            if pad_len < t:
                audio = F.pad(audio.view(1, 1, t), (0, pad_len), mode="reflect").view(-1)
            else:  # reflect は pad<input が必要。極端に短い場合は tile
                reps = self.segment_length // t + 1
                audio = audio.repeat(reps)[: self.segment_length]
        elif t > self.segment_length:
            max_start = t - self.segment_length
            start = int(self._rng.integers(0, max_start + 1)) if self.mode == "train" else 0
            audio = audio[start : start + self.segment_length]
        return audio

    def _assert_time_alignment(self, audio: torch.Tensor, mel: torch.Tensor) -> None:
        """torchaudio center=True の不変量 T_mel == 1 + T_audio//hop を検証。

        ※ generator 出力長は T_mel*hop で segment_length と hop ぶんずれうる
        (center=True の +1 frame)。その crop は下流 (T-M2.4) 責務。
        """
        expected_t_mel = 1 + audio.shape[0] // self.hop_length
        if mel.shape[1] != expected_t_mel:
            raise ValueError(
                f"mel/audio mismatch: mel T={mel.shape[1]}, audio={audio.shape[0]}, "
                f"hop={self.hop_length}, expected T_mel=1+audio//hop={expected_t_mel}"
            )

    @classmethod
    def from_config(
        cls, cfg: dict, mode: Literal["train", "val"] = "train", seed: int | None = None
    ) -> LibriTTSRDataset:
        return cls(
            filelist_path=Path(cfg["filelist"]),
            root_dir=Path(cfg["root_dir"]),
            segment_length=cfg["segment_length"],
            hop_length=cfg["hop_length"],
            mel_cfg=cfg["mel"],
            mode=mode,
            seed=seed,
        )
