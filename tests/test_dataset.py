"""LibriTTSRDataset の unit テスト (T-M2.1)。合成 wav + TSV で検証。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from wavenext2.data.dataset import Batch, LibriTTSRDataset, seed_worker

SR = 24000
HOP = 256
SEG = 2560  # = 10 * HOP
MEL_CFG = dict(
    sample_rate=SR,
    n_fft=1024,
    hop_length=HOP,
    win_length=1024,
    n_mels=128,
    f_min=20.0,
    f_max=12000.0,
    eps=1e-5,
)


def _make_dataset(tmp_path: Path, lengths: list[int], mode="train", seed=0) -> LibriTTSRDataset:
    root = tmp_path / "LibriTTS_R"
    rows = []
    rng = np.random.default_rng(0)
    for i, n in enumerate(lengths):
        rel = f"train-clean-100/{i}/1/{i}_1_0000.wav"
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), (0.5 * rng.standard_normal(n)).astype("float32"), SR)
        rows.append(
            {
                "rel_path": rel,
                "n_samples": n,
                "speaker_id": str(i),
                "chapter_id": "1",
                "duration_sec": round(n / SR, 3),
                "peak_dbfs": -1.0,
                "rms_dbfs": -10.0,
            }
        )
    flp = tmp_path / "fl.tsv"
    with flp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return LibriTTSRDataset(flp, root, SEG, HOP, MEL_CFG, mode=mode, seed=seed)


# --- __getitem__ -------------------------------------------------------------
def test_getitem_shape_and_keys(tmp_path):
    ds = _make_dataset(tmp_path, [SEG * 2, SEG // 2])
    item: Batch = ds[0]
    assert set(item) == {"mel", "audio", "n_samples"}
    assert item["audio"].shape == (SEG,)
    assert item["mel"].shape == (128, 1 + SEG // HOP)  # (128, 11)
    assert item["n_samples"] == SEG * 2


def test_time_alignment(tmp_path):
    ds = _make_dataset(tmp_path, [SEG])
    item = ds[0]
    assert item["mel"].shape[1] == 1 + item["audio"].shape[0] // HOP


def test_short_audio_reflect_padded(tmp_path):
    ds = _make_dataset(tmp_path, [SEG // 2])
    assert ds[0]["audio"].shape == (SEG,)


def test_very_short_audio_tiled(tmp_path):
    ds = _make_dataset(tmp_path, [100])  # 極端に短い → tile fallback
    assert ds[0]["audio"].shape == (SEG,)


def test_long_audio_cropped(tmp_path):
    ds = _make_dataset(tmp_path, [SEG * 3])
    assert ds[0]["audio"].shape == (SEG,)


# --- peak 正規化 -------------------------------------------------------------
def test_peak_normalize_val_target():
    ds = LibriTTSRDataset.__new__(LibriTTSRDataset)
    out = ds._peak_normalize(torch.randn(1000) * 3.0, -3.0)
    assert abs(float(out.abs().max()) - 10 ** (-3 / 20)) < 1e-4  # ≈ 0.7079


def test_peak_normalize_silence():
    ds = LibriTTSRDataset.__new__(LibriTTSRDataset)
    z = torch.zeros(100)
    assert torch.equal(ds._peak_normalize(z, -3.0), z)


def test_val_gain_is_minus3(tmp_path):
    ds = _make_dataset(tmp_path, [SEG], mode="val")
    assert ds._sample_gain_db() == -3.0


def test_train_gain_in_range(tmp_path):
    ds = _make_dataset(tmp_path, [SEG], mode="train", seed=1)
    for _ in range(20):
        assert -6.0 <= ds._sample_gain_db() <= -1.0


def test_val_crop_deterministic(tmp_path):
    ds = _make_dataset(tmp_path, [SEG * 3], mode="val")
    assert torch.equal(ds[0]["audio"], ds[0]["audio"])  # val は start=0 固定


# --- config / error ----------------------------------------------------------
def test_hop_mismatch_raises(tmp_path):
    bad_mel = {**MEL_CFG, "hop_length": 300}
    root = tmp_path / "r"
    (root / "a").mkdir(parents=True)
    sf.write(str(root / "a/x.wav"), np.zeros(SEG, "float32"), SR)
    flp = tmp_path / "fl.tsv"
    flp.write_text("rel_path\tn_samples\tspeaker_id\na/x.wav\t2560\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hop_length mismatch"):
        LibriTTSRDataset(flp, root, SEG, HOP, bad_mel)


def test_return_mel_false_not_implemented():
    with pytest.raises(NotImplementedError):
        LibriTTSRDataset(Path("x"), Path("y"), SEG, HOP, MEL_CFG, return_mel=False)


def test_n_samples_attribute(tmp_path):
    ds = _make_dataset(tmp_path, [SEG, SEG * 2, SEG * 3])
    assert ds.n_samples == [SEG, SEG * 2, SEG * 3]


def test_from_config(tmp_path):
    _make_dataset(tmp_path, [SEG])  # creates fl.tsv + tree
    cfg = {
        "filelist": str(tmp_path / "fl.tsv"),
        "root_dir": str(tmp_path / "LibriTTS_R"),
        "segment_length": SEG,
        "hop_length": HOP,
        "mel": MEL_CFG,
    }
    ds2 = LibriTTSRDataset.from_config(cfg, mode="val")
    assert ds2.mode == "val" and ds2.segment_length == SEG


# --- DataLoader 結合 ---------------------------------------------------------
def test_dataloader_collate(tmp_path):
    ds = _make_dataset(tmp_path, [SEG, SEG * 2, SEG // 2, SEG])
    loader = DataLoader(ds, batch_size=2, worker_init_fn=seed_worker)
    batch = next(iter(loader))
    assert batch["mel"].shape == (2, 128, 1 + SEG // HOP)
    assert batch["audio"].shape == (2, SEG)
    assert batch["n_samples"].shape == (2,)  # default_collate が int を tensor 化
