"""measure_rtf.py — GAN/Diff-WaveNeXt 2 の RTF 計測 (GPU A100 + CPU 1-core) (T-M4.3).

RTF = inference_time[s] / audio_length[s]。論文 Table 1〜3 と対比する。
- `model.synthesize(mel)` で GAN (fixed-point T 回) / Diff (4-step reverse) を横断計測
  (`getattr(model, "synthesize", model.forward)` で dispatch)。
- GPU: CUDA events (`torch.cuda.Event`) で GPU 実時間を計測 (perf_counter+sync は同期 overhead 込)。
- CPU: `time.perf_counter` の **median** + `torch.set_num_threads(1)` (OS スケジューラ揺らぎ対策)。
- iter 数 (GAN T / Diff 4-step) は model config に委ね、本関数は計測のみ。`n_warmup`/`n_measure`
  は計測ループ回数 (別概念)。`torch.set_num_threads` は finally で必ず復元 (グローバル状態保護)。
"""

from __future__ import annotations

import statistics
import time

import torch
from torch import nn

__all__ = ["measure_rtf"]


@torch.no_grad()
def measure_rtf(
    model: nn.Module,
    mels: list[torch.Tensor] | torch.Tensor,
    *,
    device: str = "cuda",
    sample_rate: int = 24000,
    hop_length: int | None = None,
    n_warmup: int = 5,
    n_measure: int = 100,
    compiled: bool = False,
) -> dict[str, float | str | int | bool]:
    """各 utterance を `model.synthesize(mel)` で 1 回ずつ合成し RTF を計測.

    Args:
        model:       GAN/Diff (`synthesize` alias、無ければ `forward` に fallback)。
        mels:        list[(1,128,T_mel)] (test-clean の長さ分布で 100 utt 推奨)。
        device:      "cuda" | "cpu"。
        sample_rate: 音声長秒の算出用 (24000)。
        hop_length:  None で `model.hop_length`。
        n_warmup:    計測から除外する空回し回数 (compile 時は増やす)。
        n_measure:   計測ループ回数。
        compiled:    torch.compile 適用済か (報告用)。

    Returns:
        {"rtf_mean", "rtf_std", "rtf_median", "n", "device", "compiled"} (T-M4.1 schema 統一)。
    """
    synth = getattr(model, "synthesize", model.forward)  # GAN/Diff 横断 dispatch
    model.eval().to(device)
    if isinstance(mels, torch.Tensor):
        mels = [mels]

    original_threads = torch.get_num_threads()
    try:
        if device == "cpu":
            torch.set_num_threads(1)  # CPU 1-core (OMP_NUM_THREADS=1 も別途必要、§6.1)
        hop = hop_length or model.hop_length

        # --- warmup (compile / CUDA context 初期化を計測から除外) ---
        for i in range(n_warmup):
            _ = synth(mels[i % len(mels)].to(device))
        if device == "cuda":
            torch.cuda.synchronize()

        # --- measure ---
        rtfs: list[float] = []
        for i in range(n_measure):
            mel = mels[i % len(mels)].to(device)
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = synth(mel)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0  # ms → s
            else:
                t0 = time.perf_counter()
                out = synth(mel)
                elapsed = time.perf_counter() - t0
            # RTF 定義に忠実に **実出力長** を使う (mel*hop 推定だと padding 由来 ±1 frame の bias)。
            # 出力長が取れない場合のみ mel*hop に fallback。
            out_len = out.shape[-1] if hasattr(out, "shape") else mel.shape[-1] * hop
            audio_sec = out_len / sample_rate
            rtfs.append(elapsed / audio_sec)
    finally:
        torch.set_num_threads(original_threads)  # グローバル状態を必ず復元

    return {
        "rtf_mean": statistics.fmean(rtfs),
        "rtf_std": statistics.stdev(rtfs) if len(rtfs) > 1 else 0.0,
        "rtf_median": statistics.median(rtfs),
        "n": n_measure,
        "device": device,
        "compiled": compiled,
    }
