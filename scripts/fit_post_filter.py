"""fit_post_filter.py — Time-invariant spectral enhancement FIR の fit (T-M3.4).

dev set (200 utterances、T-M0.3 `dev_postfilter.tsv`、seed=43) で:
    1. 各 utt の GT 波形を Diff モデルで 4-step reverse sample → y_gen
    2. STFT 振幅差 |Y_gt| - |Y_gen| を全 (utt × frame) で平均 (Okamoto21 §3.3 解釈 B)
    3. iRFFT → fftshift → 512-tap linear-phase FIR を取得
    4. `post_filter/fir.npy` に保存、`fit_stats.json` に再現性 record (seed/git_sha/checkpoint_sha256)

未学習モデルでは FIR は無意味 (周波数応答テスト不合格)。**実 FIR の fit は M6.2 の Diff フル訓練後**
に実施し、本スクリプトはその reproducibility 用途 (T-M3.4 §8.1)。コア関数 `fit_post_filter` は
学習済み model を受け取る pure な集約処理で、`scripts/` 配下に CLI を置く。

参考: docs/architecture.md §5 / docs/training.md §4.3 / Okamoto21 §3.3。
checkpoint format は T-M3.2 / M6.2 で確定するため、`_load_model` は最小実装で後調整可。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

from wavenext2.data.mel import LogMelSpectrogram
from wavenext2.inference.infer_diff import reverse_sample
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss
from wavenext2.models.diff_wavenext2 import DiffWaveNext2

N_FFT_FIT = 512  # Okamoto21 §3.3 実験設定
HOP_FIT = 256
FIR_LENGTH = 512  # = n_fft (linear-phase 用)
POSTFILTER_SEED = 43  # T-M0.3 postfilter_seed (val seed=42 と非重複)
SAMPLE_RATE = 24000

# Diff 用 mel 設定 (architecture.md §3 / §7)。
_MEL_CFG = {
    "sample_rate": SAMPLE_RATE,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 12000,
}


def fit_post_filter(
    model: DiffWaveNext2,
    dev_audio_paths: list[Path],
    *,
    n_fft: int = N_FFT_FIT,
    hop: int = HOP_FIT,
    fir_length: int = FIR_LENGTH,
    seed: int = POSTFILTER_SEED,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """dev set 上で振幅差平均 → iRFFT → fftshift で linear-phase FIR を構築.

    Args:
        model:           学習済み DiffWaveNext2 (4 sub-models、eval 済 or 内部で eval)。
        dev_audio_paths: GT 波形パスのリスト (mel は runtime 抽出、mel-not-stored 戦略)。
        n_fft / hop:     fit 用 STFT 設定 (Okamoto21: 512 / 256)。
        fir_length:      FIR tap 数 (linear-phase 用に n_fft と一致)。
        seed:            reverse_sample の deterministic seed (CI 再現性)。
        device:          "cuda" / "cpu"。

    Returns:
        (fir, stats):
          fir:   (fir_length,) float32 numpy、fftshift 済 linear-phase FIR。
          stats: fit_stats.json に dump する dict。
    """
    mel_extractor = LogMelSpectrogram.from_config(_MEL_CFG).to(device)
    window = torch.hann_window(n_fft, device=device)
    n_bins = n_fft // 2 + 1
    diff_acc = np.zeros(n_bins, dtype=np.float64)  # 加算誤差防止に float64 蓄積
    n_frames_total = 0

    model.eval().to(device)
    with torch.no_grad():
        for gt_path in dev_audio_paths:
            y_gt, sr = torchaudio.load(str(gt_path))
            y_gt = y_gt.mean(dim=0)  # stereo → mono
            if sr != SAMPLE_RATE:
                y_gt = torchaudio.functional.resample(y_gt, sr, SAMPLE_RATE)
            y_gt = y_gt.to(device)

            mel = mel_extractor(y_gt.unsqueeze(0))  # (1, 128, T_mel)
            y_gen = reverse_sample(model, mel, seed=seed).squeeze(0).float()  # (T_audio,)

            t = min(y_gen.shape[0], y_gt.shape[0])  # hop padding 差を shorter に揃える
            y_gen, y_gt_c = y_gen[:t], y_gt[:t].float()

            s_gt = torch.stft(
                y_gt_c, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True
            ).abs()
            s_gen = torch.stft(
                y_gen, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True
            ).abs()

            mag_diff = (s_gt - s_gen).sum(dim=1)  # (n_bins,) frame 軸 sum
            diff_acc += mag_diff.cpu().numpy().astype(np.float64)
            n_frames_total += s_gt.shape[1]

    # 全 (utt × frame) 平均 (Okamoto21 §3.3 解釈 B)
    mean_diff_mag = (diff_acc / max(n_frames_total, 1)).astype(np.float32)
    fir = np.fft.irfft(mean_diff_mag, n=fir_length).astype(np.float32)
    fir = np.fft.fftshift(fir)  # linear-phase 用に中央化

    stats = {
        "seed": seed,
        "n_utt": len(dev_audio_paths),
        "n_frames_total": int(n_frames_total),
        "n_fft": n_fft,
        "hop": hop,
        "fir_length": fir_length,
        "fir_max_abs": float(np.abs(fir).max()),
        "mean_diff_mag_min": float(mean_diff_mag.min()),
        "mean_diff_mag_max": float(mean_diff_mag.max()),
        "mean_diff_mag_abs_mean": float(np.abs(mean_diff_mag).mean()),
    }
    return fir, stats


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256_dir(checkpoint_dir: Path) -> str:
    """checkpoint dir 内の *.pt を sorted 順で hash (再現性 record)。"""
    h = hashlib.sha256()
    for p in sorted(checkpoint_dir.glob("*.pt")):
        h.update(p.read_bytes())
    return h.hexdigest()


def _read_filelist(filelist: Path, data_root: Path) -> list[Path]:
    """TSV (header 付き) の rel_path 列から GT 波形の絶対パスを構築。"""
    paths: list[Path] = []
    lines = filelist.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        rel = line.split("\t")[0]
        paths.append(data_root / rel)
    return paths


def _load_model(checkpoint_dir: Path, device: str) -> DiffWaveNext2:
    """4 sub-model checkpoint (sub_{1..4}.pt) を DiffWaveNext2 に load。

    checkpoint format は T-M3.2 / M6.2 で確定するため最小実装。各ファイルは sub-model の
    state_dict (または "model" キー配下) を想定。format 確定後に調整する。
    """
    model = DiffWaveNext2.from_config({"sub_model_cfg": {}})
    for k in range(1, model.K + 1):
        ckpt_path = checkpoint_dir / f"sub_{k}.pt"
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.sub_models[k - 1].load_state_dict(state)
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit time-invariant post-filter FIR (T-M3.4).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--filelist", type=Path, default=Path("data/filelists/dev_postfilter.tsv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, default=Path("post_filter/fir.npy"))
    parser.add_argument("--stats-path", type=Path, default=Path("post_filter/fit_stats.json"))
    parser.add_argument("--n-fft", type=int, default=N_FFT_FIT)
    parser.add_argument("--hop", type=int, default=HOP_FIT)
    parser.add_argument("--fir-length", type=int, default=FIR_LENGTH)
    parser.add_argument("--seed", type=int, default=POSTFILTER_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = _load_model(args.checkpoint_dir, args.device)
    dev_paths = _read_filelist(args.filelist, args.data_root)
    fir, stats = fit_post_filter(
        model,
        dev_paths,
        n_fft=args.n_fft,
        hop=args.hop,
        fir_length=args.fir_length,
        seed=args.seed,
        device=args.device,
    )

    stats["git_sha"] = _git_sha()
    stats["checkpoint_sha256"] = _sha256_dir(args.checkpoint_dir)
    # MR-STFT (T-M2.3) は再利用可能性の確認のみ (full eval は M4.1 で実施)。
    _ = MultiResolutionSTFTLoss

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, fir)
    args.stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved FIR to {args.out_path} (max_abs={stats['fir_max_abs']:.4f})")
    print(f"Stats: {args.stats_path}")


if __name__ == "__main__":
    main()
