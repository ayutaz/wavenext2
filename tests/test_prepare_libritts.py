"""scripts/prepare_libritts.py の unit + 合成データ e2e テスト (T-M0.3)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# scripts/ はパッケージではないのでパス指定で import
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("prepare_libritts", _SCRIPTS / "prepare_libritts.py")
pl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pl)


# --- unit: filter_duration ----------------------------------------------------
def test_filter_duration_partitions():
    rows = [
        {"rel_path": "a", "duration_sec": 0.3},
        {"rel_path": "b", "duration_sec": 1.0},
        {"rel_path": "c", "duration_sec": 5.0},
        {"rel_path": "d", "duration_sec": 20.0},
    ]
    kept, short, long_ = pl.filter_duration(rows, 1.0, 15.0)
    assert [r["rel_path"] for r in kept] == ["b", "c"]
    assert [r["rel_path"] for r in short] == ["a"]
    assert [r["rel_path"] for r in long_] == ["d"]
    # 件数保存則
    assert len(kept) + len(short) + len(long_) == len(rows)


def test_filter_duration_boundaries_inclusive():
    rows = [{"rel_path": "lo", "duration_sec": 1.0}, {"rel_path": "hi", "duration_sec": 15.0}]
    kept, short, long_ = pl.filter_duration(rows, 1.0, 15.0)
    assert len(kept) == 2 and not short and not long_


# --- unit: speaker_balanced_sample --------------------------------------------
def _make_pool():
    pool = []
    for spk in range(5):
        for utt in range(4):
            pool.append(
                {
                    "rel_path": f"train-clean-100/{spk}/0/{spk}_{utt}.wav",
                    "speaker_id": str(spk),
                    "n_samples": 1000 + utt * 100,  # utt=3 が最長
                }
            )
    return pool


def test_speaker_balanced_one_per_speaker():
    pool = _make_pool()
    selected, remaining = pl.speaker_balanced_sample(pool, n_speakers=3, seed=42)
    assert len(selected) == 3
    # 1 話者 1 件 (speaker_id ユニーク)
    assert len({r["speaker_id"] for r in selected}) == 3
    # 各選択は最長 utterance (n_samples=1300)
    assert all(r["n_samples"] == 1300 for r in selected)
    # remaining は pool から selected を除いたもの
    sel_keys = {r["rel_path"] for r in selected}
    assert len(remaining) == len(pool) - len(selected)
    assert sel_keys.isdisjoint({r["rel_path"] for r in remaining})


def test_speaker_balanced_deterministic():
    pool = _make_pool()
    a, _ = pl.speaker_balanced_sample(pool, 3, seed=42)
    b, _ = pl.speaker_balanced_sample(pool, 3, seed=42)
    assert [r["rel_path"] for r in a] == [r["rel_path"] for r in b]
    # 別 seed では (一般に) 選択が変わりうるが、件数は不変
    c, _ = pl.speaker_balanced_sample(pool, 3, seed=999)
    assert len(c) == 3


def test_speaker_balanced_caps_at_available():
    pool = _make_pool()  # 5 speakers
    selected, _ = pl.speaker_balanced_sample(pool, n_speakers=100, seed=1)
    assert len(selected) == 5  # 話者数で頭打ち


# --- unit: compute_dbfs -------------------------------------------------------
def test_compute_dbfs_known_values():
    # 振幅 0.5 の定数信号 → peak = rms = 0.5 → 20*log10(0.5) ≈ -6.02 dBFS
    wav = np.full(1000, 0.5, dtype=np.float32)
    peak, rms = pl.compute_dbfs(wav)
    assert abs(peak - (-6.02)) < 0.05
    assert abs(rms - (-6.02)) < 0.05


def test_compute_dbfs_silence_floor():
    wav = np.zeros(100, dtype=np.float32)
    peak, rms = pl.compute_dbfs(wav)
    assert peak == pl.DBFS_FLOOR and rms == pl.DBFS_FLOOR


# --- e2e: 合成 LibriTTS-R ツリー ----------------------------------------------
def _write_wav(path: Path, n_samples: int, sr: int = pl.SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sig = (0.3 * rng.standard_normal(n_samples)).astype(np.float32)
    sf.write(str(path), sig, sr)


def _build_synthetic(root: Path):
    sr = pl.SAMPLE_RATE
    # train-clean-100: speaker 100,101,102 × 3 utt (各 1.0s = kept)
    # train-clean-360: speaker 200,201,202 × 3 utt
    for subset, speakers in [
        ("train-clean-100", [100, 101, 102]),
        ("train-clean-360", [200, 201, 202]),
    ]:
        for spk in speakers:
            for utt in range(3):
                _write_wav(root / subset / str(spk) / "1" / f"{spk}_1_{utt:04d}.wav", sr)  # 1.0s
    # 長さフィルタ用に short / long を 1 件ずつ追加
    _write_wav(
        root / "train-clean-100" / "100" / "1" / "100_1_short.wav", int(0.3 * sr)
    )  # too short
    _write_wav(root / "train-clean-100" / "101" / "1" / "101_1_long.wav", int(3.0 * sr))  # too long
    # test-clean: 2 speakers × 2 utt = 4 (長さフィルタ非適用)
    for spk in [300, 301]:
        for utt in range(2):
            _write_wav(root / "test-clean" / str(spk) / "1" / f"{spk}_1_{utt:04d}.wav", sr)


def test_e2e_synthetic(tmp_path: Path):
    src = tmp_path / "LibriTTS_R"
    _build_synthetic(src)
    out = tmp_path / "filelists"

    summary = pl.run(
        src,
        out,
        val_size=3,
        dev_postfilter_size=2,
        min_duration=0.5,
        max_duration=2.0,
        check_disk=False,
    )

    # 件数: train 18 kept = train(13) + val(3) + pf(2)、short/long 各 1、test 4
    assert summary["n_too_short"] == 1
    assert summary["n_too_long"] == 1
    assert summary["n_val"] == 3
    assert summary["n_dev_postfilter"] == 2
    assert summary["n_train"] == 13
    assert summary["n_test"] == 4
    assert summary["n_bad_sample_rate"] == 0
    assert summary["n_errors"] == 0

    # 生成ファイルの存在 + header
    for name in ("train.tsv", "val.tsv", "test.tsv", "dev_postfilter.tsv"):
        p = out / name
        assert p.exists()
        with p.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
        assert header == pl.FILELIST_HEADER

    # val は 1 話者 1 件 (speaker ユニーク = 3)
    with (out / "val.tsv").open(encoding="utf-8") as f:
        val = list(csv.DictReader(f, delimiter="\t"))
    assert len({r["speaker_id"] for r in val}) == 3

    # 相対パス (POSIX, バックスラッシュなし)、絶対パス混入なし
    for r in val:
        assert "\\" not in r["rel_path"]
        assert not Path(r["rel_path"]).is_absolute()

    # 重複なし (train ∩ val ∩ pf == ∅)
    def paths(name):
        with (out / name).open(encoding="utf-8") as f:
            return {row["rel_path"] for row in csv.DictReader(f, delimiter="\t")}

    tr, va, pf = paths("train.tsv"), paths("val.tsv"), paths("dev_postfilter.tsv")
    assert tr.isdisjoint(va) and tr.isdisjoint(pf) and va.isdisjoint(pf)

    # stats.json schema
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["sample_rate"] == pl.SAMPLE_RATE
    assert stats["train"]["n_utterances"] == 13
    assert stats["filtered"]["n_too_short"] == 1
    assert len(stats["val"]["speakers"]) == 3
    assert "val_hash" in stats

    # audio_info.tsv: 全 wav (train 20 + test 4 = 24)
    with (out / "audio_info.tsv").open(encoding="utf-8") as f:
        info = list(csv.DictReader(f, delimiter="\t"))
    assert len(info) == 24
    assert all(int(r["sample_rate"]) == pl.SAMPLE_RATE for r in info)


def test_e2e_reproducible(tmp_path: Path):
    """同一 seed で 2 回実行 → val_hash 一致 (スナップショット安定性)."""
    src = tmp_path / "LibriTTS_R"
    _build_synthetic(src)
    s1 = pl.run(
        src,
        tmp_path / "o1",
        val_size=3,
        dev_postfilter_size=2,
        min_duration=0.5,
        max_duration=2.0,
        check_disk=False,
    )
    s2 = pl.run(
        src,
        tmp_path / "o2",
        val_size=3,
        dev_postfilter_size=2,
        min_duration=0.5,
        max_duration=2.0,
        check_disk=False,
    )
    assert s1["val_hash"] == s2["val_hash"]
