"""prepare_libritts.py: LibriTTS-R 取得後の filelist / stats 生成.

ユーザーが openslr (https://www.openslr.org/141/) からダウンロード・展開した
LibriTTS-R を入力に、後続 Dataset (T-M2.1) が読み込む filelist を生成する。

生成物 (--out-dir, 既定 data/filelists/):
  - train.tsv / val.tsv / test.tsv / dev_postfilter.tsv  (TSV header 付き, LF, 相対 POSIX path)
  - stats.json       (speaker/chapter/duration/dBFS 分布)
  - audio_info.tsv   (全 wav の sr/channels/duration、source of truth)

音声 I/O は soundfile に統一 (torchaudio 2.11 は info/sox_effects 廃止・load が TorchCodec 依存)。

使い方:
    uv run python scripts/prepare_libritts.py --src-dir <LibriTTS-R 展開ルート>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# --- 定数 ---------------------------------------------------------------------
FILELIST_HEADER = [
    "rel_path",
    "n_samples",
    "speaker_id",
    "chapter_id",
    "duration_sec",
    "peak_dbfs",
    "rms_dbfs",
]
TRAIN_SUBSETS = ["train-clean-100", "train-clean-360"]
TEST_SUBSET = "test-clean"  # openslr 公式 split。docs の "test-clean-100" 表記揺れは §9.4 参照
SAMPLE_RATE = 24000
DBFS_FLOOR = -120.0  # 無音 (peak/rms == 0) のときの下限値


# --- 入出力ヘルパ -------------------------------------------------------------
def find_subset_dir(src_dir: Path, subset: str) -> Path | None:
    """subset ディレクトリを探す。ハイフン版とアンダースコア版の両方を許容."""
    for name in (subset, subset.replace("-", "_")):
        d = src_dir / name
        if d.is_dir():
            return d
    return None


def collect_wavs(subset_dir: Path) -> list[Path]:
    """subset 配下の wav を再帰収集 (sorted で決定論的)."""
    return sorted(subset_dir.rglob("*.wav"))


def parse_ids_from_path(rel_path: str) -> tuple[str, str]:
    """`<subset>/<speaker>/<chapter>/<utt>.wav` から speaker/chapter を取り出す."""
    parts = Path(rel_path).parts
    if len(parts) >= 4:
        return parts[1], parts[2]
    return "unknown", "unknown"


def compute_dbfs(wav: np.ndarray) -> tuple[float, float]:
    """peak / RMS を dBFS で返す (無音は DBFS_FLOOR)."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono 化 (LibriTTS-R は基本 mono)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
    peak_dbfs = 20.0 * np.log10(peak) if peak > 0 else DBFS_FLOOR
    rms_dbfs = 20.0 * np.log10(rms) if rms > 0 else DBFS_FLOOR
    return round(peak_dbfs, 2), round(rms_dbfs, 2)


def build_audio_info(wav_paths: list[Path], src_dir: Path) -> tuple[list[dict], list[str]]:
    """全 wav の sr/channels/duration + peak/RMS dBFS を集計.

    Returns:
        (rows, errors) — rows は FILELIST_HEADER + sample_rate/channels を含む dict のリスト、
        errors は読めない / 24kHz でない wav のメッセージ.
    """
    rows: list[dict] = []
    errors: list[str] = []
    for wav_path in wav_paths:
        rel_path = wav_path.relative_to(src_dir).as_posix()
        try:
            info = sf.info(str(wav_path))
        except Exception as e:  # noqa: BLE001 — corruption を検出して列挙
            errors.append(f"unreadable: {rel_path} ({e})")
            continue
        if info.samplerate != SAMPLE_RATE:
            errors.append(f"sr!={SAMPLE_RATE}: {rel_path} (sr={info.samplerate})")
        wav, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        peak_dbfs, rms_dbfs = compute_dbfs(wav)
        speaker_id, chapter_id = parse_ids_from_path(rel_path)
        rows.append(
            {
                "rel_path": rel_path,
                "n_samples": int(info.frames),
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "duration_sec": round(info.frames / info.samplerate, 3),
                "peak_dbfs": peak_dbfs,
                "rms_dbfs": rms_dbfs,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
            }
        )
    return rows, errors


# --- 分割ロジック (単体テスト対象) --------------------------------------------
def filter_duration(
    rows: list[dict], min_sec: float, max_sec: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """duration_sec が [min_sec, max_sec] 外を除外。Returns (kept, too_short, too_long)."""
    kept, too_short, too_long = [], [], []
    for r in rows:
        d = r["duration_sec"]
        if d < min_sec:
            too_short.append(r)
        elif d > max_sec:
            too_long.append(r)
        else:
            kept.append(r)
    return kept, too_short, too_long


def speaker_balanced_sample(
    pool: list[dict], n_speakers: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """n_speakers 話者を seeded RNG で抽出し、各話者から最長 utterance を 1 件選択.

    各話者から「最長 (n_samples 最大、tie-break は rel_path)」を選ぶため決定論的で
    metric も安定する。Returns (selected, remaining = pool - selected).
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_speaker[r["speaker_id"]].append(r)

    speakers = sorted(by_speaker.keys())
    rng = random.Random(seed)
    n = min(n_speakers, len(speakers))
    chosen_speakers = sorted(rng.sample(speakers, n))

    selected: list[dict] = []
    selected_keys: set[str] = set()
    for spk in chosen_speakers:
        utt = max(by_speaker[spk], key=lambda r: (r["n_samples"], r["rel_path"]))
        selected.append(utt)
        selected_keys.add(utt["rel_path"])

    remaining = [r for r in pool if r["rel_path"] not in selected_keys]
    return selected, remaining


# --- 書き出し -----------------------------------------------------------------
def write_tsv(path: Path, rows: list[dict]) -> None:
    """TSV (header 付き, UTF-8, LF) で書き出し (FILELIST_HEADER 列のみ)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FILELIST_HEADER,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_audio_info(path: Path, rows: list[dict]) -> None:
    """audio_info.tsv (sr/channels を含む全列) を書き出し."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = FILELIST_HEADER + ["sample_rate", "channels"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _duration_histogram(rows: list[dict]) -> dict[str, int]:
    bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 15), (15, 1e9)]
    labels = ["0-1", "1-3", "3-5", "5-10", "10-15", "15+"]
    hist = dict.fromkeys(labels, 0)
    for r in rows:
        d = r["duration_sec"]
        for (lo, hi), label in zip(bins, labels):
            if lo <= d < hi:
                hist[label] += 1
                break
    return hist


def _split_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"n_utterances": 0, "n_speakers": 0, "n_chapters": 0, "duration_hours_total": 0.0}
    durations = np.array([r["duration_sec"] for r in rows])
    peaks = np.array([r["peak_dbfs"] for r in rows])
    rmss = np.array([r["rms_dbfs"] for r in rows])

    def pct(a, q):
        return round(float(np.percentile(a, q)), 3)

    return {
        "n_utterances": len(rows),
        "n_speakers": len({r["speaker_id"] for r in rows}),
        "n_chapters": len({(r["speaker_id"], r["chapter_id"]) for r in rows}),
        "duration_hours_total": round(float(durations.sum()) / 3600.0, 4),
        "duration_histogram": _duration_histogram(rows),
        "peak_dbfs": {
            "min": pct(peaks, 0),
            "p5": pct(peaks, 5),
            "p50": pct(peaks, 50),
            "p95": pct(peaks, 95),
            "max": pct(peaks, 100),
        },
        "rms_dbfs": {
            "min": pct(rmss, 0),
            "p5": pct(rmss, 5),
            "p50": pct(rmss, 50),
            "p95": pct(rmss, 95),
            "max": pct(rmss, 100),
        },
    }


def write_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _val_hash(val_rows: list[dict]) -> str:
    """val 選択の変更検知用 hash (rel_path の sorted を SHA-256)."""
    joined = "\n".join(sorted(r["rel_path"] for r in val_rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# --- コア処理 (テストから直接呼べる) ------------------------------------------
def run(
    src_dir: Path,
    out_dir: Path,
    *,
    val_size: int = 100,
    dev_postfilter_size: int = 200,
    val_seed: int = 42,
    postfilter_seed: int = 43,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    min_free_gb: float = 100.0,
    check_disk: bool = True,
) -> dict:
    """filelist 一式を生成し、サマリ dict を返す."""
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    if check_disk:
        free_gb = shutil.disk_usage(src_dir).free / 1e9
        if free_gb < min_free_gb:
            raise RuntimeError(f"disk free {free_gb:.1f} GB < required {min_free_gb} GB")

    # train subsets 収集
    train_wavs: list[Path] = []
    for subset in TRAIN_SUBSETS:
        d = find_subset_dir(src_dir, subset)
        if d is None:
            raise FileNotFoundError(f"train subset not found under {src_dir}: {subset}")
        train_wavs += collect_wavs(d)
    if not train_wavs:
        raise RuntimeError(f"no train wav found under {src_dir}")

    test_dir = find_subset_dir(src_dir, TEST_SUBSET)
    if test_dir is None:
        raise FileNotFoundError(f"test subset not found under {src_dir}: {TEST_SUBSET}")
    test_wavs = collect_wavs(test_dir)

    # audio_info (全 wav)
    all_rows, errors = build_audio_info(train_wavs + test_wavs, src_dir)
    test_set = {p.relative_to(src_dir).as_posix() for p in test_wavs}
    train_pool_all = [r for r in all_rows if r["rel_path"] not in test_set]
    test_rows = sorted(
        (r for r in all_rows if r["rel_path"] in test_set), key=lambda r: r["rel_path"]
    )

    # 長さフィルタ (train のみ)
    kept, too_short, too_long = filter_duration(train_pool_all, min_duration, max_duration)

    # dev_postfilter -> val -> train の順に hold-out (非重複)
    rng_pf = random.Random(postfilter_seed)
    pf_n = min(dev_postfilter_size, len(kept))
    dev_postfilter = rng_pf.sample(kept, pf_n)
    pf_keys = {r["rel_path"] for r in dev_postfilter}
    pool_after_pf = [r for r in kept if r["rel_path"] not in pf_keys]

    val_rows, train_rows = speaker_balanced_sample(pool_after_pf, val_size, val_seed)

    # 重複検証
    s_train = {r["rel_path"] for r in train_rows}
    s_val = {r["rel_path"] for r in val_rows}
    s_pf = {r["rel_path"] for r in dev_postfilter}
    assert s_train.isdisjoint(s_val), "train ∩ val != ∅"
    assert s_train.isdisjoint(s_pf), "train ∩ dev_postfilter != ∅"
    assert s_val.isdisjoint(s_pf), "val ∩ dev_postfilter != ∅"
    assert len(train_rows) + len(val_rows) + len(dev_postfilter) == len(kept), "split 件数不一致"

    # 書き出し
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "train.tsv", train_rows)
    write_tsv(out_dir / "val.tsv", val_rows)
    write_tsv(out_dir / "test.tsv", test_rows)
    write_tsv(out_dir / "dev_postfilter.tsv", dev_postfilter)
    write_audio_info(out_dir / "audio_info.tsv", all_rows)

    n_bad_sr = sum(1 for r in all_rows if r["sample_rate"] != SAMPLE_RATE)
    stats = {
        "sample_rate": SAMPLE_RATE,
        "train": _split_stats(train_rows),
        "val": {**_split_stats(val_rows), "speakers": sorted({r["speaker_id"] for r in val_rows})},
        "test": _split_stats(test_rows),
        "dev_postfilter": _split_stats(dev_postfilter),
        "filtered": {"n_too_short": len(too_short), "n_too_long": len(too_long)},
        "errors": {"n_unreadable_or_bad_sr": len(errors), "n_bad_sample_rate": n_bad_sr},
        "val_hash": _val_hash(val_rows),
    }
    write_stats(out_dir / "stats.json", stats)

    return {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "n_dev_postfilter": len(dev_postfilter),
        "n_too_short": len(too_short),
        "n_too_long": len(too_long),
        "n_errors": len(errors),
        "n_bad_sample_rate": n_bad_sr,
        "val_hash": stats["val_hash"],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LibriTTS-R filelist / stats 生成")
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="LibriTTS-R 展開ルート (train-clean-100/... を含む)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/filelists"))
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--dev-postfilter-size", type=int, default=200)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--postfilter-seed", type=int, default=43)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=100.0)
    parser.add_argument("--no-disk-check", action="store_true")
    args = parser.parse_args(argv)

    summary = run(
        args.src_dir,
        args.out_dir,
        val_size=args.val_size,
        dev_postfilter_size=args.dev_postfilter_size,
        val_seed=args.val_seed,
        postfilter_seed=args.postfilter_seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_free_gb=args.min_free_gb,
        check_disk=not args.no_disk_check,
    )

    print("[prepare_libritts] done")
    for k in (
        "n_train",
        "n_val",
        "n_test",
        "n_dev_postfilter",
        "n_too_short",
        "n_too_long",
        "n_errors",
        "n_bad_sample_rate",
    ):
        print(f"  {k}: {summary[k]}")
    print(f"  val_hash: {summary['val_hash']}")
    if summary["errors"]:
        print(f"  [WARN] {len(summary['errors'])} error(s); first 10:", file=sys.stderr)
        for msg in summary["errors"][:10]:
            print(f"    - {msg}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
