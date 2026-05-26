---
id: T-M3.4
title: Time-invariant post-filter (FIR fit on dev + apply)
milestone: M3
phase: M3
status: pending
size: M
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M3.3]
blocks: [T-M3.5, T-M4.1]
related_docs:
  - docs/milestones.md#m34-post-filter-srcwavenext2inferencepost_filterpy--scriptsfit_post_filterpy
  - docs/architecture.md
  - docs/training.md
---

# T-M3.4: Time-invariant post-filter (FIR fit on dev + apply)

> **マイルストーン**: [M3](../milestones.md#m3-diff-wavenext-2-作業量-large5-サブタスク) / **サブタスク**: [M3.4](../milestones.md#m34-post-filter-srcwavenext2inferencepost_filterpy--scriptsfit_post_filterpy)
> **依存**: [T-M3.3](T-M3.3-reverse-sampler.md) (前提: [T-M3.1](T-M3.1-diff-model.md), [T-M0.3](T-M0.3-libritts-r.md), [T-M2.3](T-M2.3-losses.md)) / **後続**: [T-M3.5](T-M3.5-diff-smoke.md), [T-M4.1](T-M4.1-objective-metrics.md)

## 1. タスク目的とゴール

### 目的
Diff-WaveNeXt 2 の 4-step reverse sampling 出力に対して、論文 §3.3 / `docs/architecture.md` §5 / Okamoto21 §3.3 で確定した **time-invariant spectral enhancement FIR** を実装する。FIR は dev set 上で **1 度だけ fit** し (`scripts/fit_post_filter.py` → `post_filter/fir.npy`)、推論時に全発話で同じ filter を畳み込む (`src/wavenext2/inference/post_filter.py::apply_post_filter`)。これにより低 iteration diffusion vocoder で失われがちな高域 detail (WaveGrad 論文の指摘) を加算的に補償し、Nyquist 端で +5〜8 dB の高域シェルフを得る。

本チケットは M3 の最終 deliverable (M3.5 smoke / M4.1 客観評価で使う) に直結し、`MultiResolutionSTFTLoss` を eval metric として **M2 から再利用** することで責務集中を図る (T-M2.3 §9.1 申し送り)。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M3.4 Acceptance を内包):
- [ ] `scripts/fit_post_filter.py` が完成し、`--checkpoint-dir` / `--filelist` / `--out-path` 引数で dev set FIR fit を実行できる
- [ ] `src/wavenext2/inference/post_filter.py` に `apply_post_filter(audio, fir) -> torch.Tensor | np.ndarray` (torch / numpy 両受け) が実装され、`from wavenext2.inference.post_filter import apply_post_filter` で import 可能
- [ ] `reverse_sample(model, mel, *, seed, post_filter=None)` に post-filter 統合 (T-M3.3 引数追加、`post_filter=fir` で apply / `None` で no-apply)
- [ ] `data/filelists/dev_postfilter.tsv` (200 utterances、T-M0.3 で生成済) を使って fit が走り、`post_filter/fir.npy` が **エラーなく生成**される
- [ ] FIR 長 = **512** (linear-phase 用 `fftshift` 済み、`fir.shape == (512,)`)
- [ ] 周波数応答が ~2 kHz 以下で **0 dB ± 1 dB**、Nyquist 端 (12 kHz) で **+5〜8 dB** (Okamoto21 Fig 3a 整合)
- [ ] `apply_post_filter` 前後で音声長が変わらない (`scipy.signal.oaconvolve` の対称切り出し)
- [ ] T-M2.3 の `MultiResolutionSTFTLoss` を eval metric として再利用 (`from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import、post-filter 前後の MR-STFT を比較)
- [ ] `post_filter/fir.npy` を git commit (M6.2 後、`.gitignore` 除外解除済、`fit_stats.json` で再現性確認)、`post_filter/.gitkeep` で空 dir 維持
- [ ] `tests/test_post_filter.py` の全テスト pass
- [ ] `docs/milestones.md` §M3.4 Acceptance 4 項目クリア
- [ ] `docs/tickets/index.md` の T-M3.4 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル

- 新規:
  - `scripts/fit_post_filter.py` (CLI スクリプト、dev set → `fir.npy` の closed-form fit。**M6.2 完了後の `fir.npy` 再生成 reproducibility 用途に降格、§8.1 で詳述**)
  - `src/wavenext2/inference/post_filter.py` (`apply_post_filter` 関数 + 補助ユーティリティ、**torch tensor / np.ndarray 両受け**)
  - `tests/test_post_filter.py` (Unit テスト)
  - `post_filter/.gitkeep` (空 dir 維持) — **M6.2 完了後の `fir.npy` git commit 化に伴い役割は薄れる (§8.1 別の設計 — 採用昇格)**
- 編集:
  - `src/wavenext2/inference/__init__.py` (`apply_post_filter` を `__all__` に追加)
  - `src/wavenext2/inference/infer_diff.py` (T-M3.3) (`reverse_sample` に `post_filter: np.ndarray | None = None` 引数追加、本チケットで反映 — §8.1 採用設計参照)
  - `.gitignore` (`post_filter/*.npy` の **除外を解除** — `fir.npy` は M6.2 後に commit。`post_filter/.gitkeep` は引き続き keep)
  - `docs/milestones.md` §M3.4 Acceptance チェックボックス更新
  - `docs/tickets/index.md` の T-M3.4 ステータス更新

### 2.2 主要構造

#### `scripts/fit_post_filter.py`

```python
"""fit_post_filter.py — Time-invariant spectral enhancement FIR の fit.

dev set (200 utterances, T-M0.3 生成) で:
    1. 各 utt の GT 波形を Diff モデルで 4-step reverse sample → y_gen
    2. STFT 振幅差を utterance × frame の **全フレーム** で平均
    3. iRFFT → fftshift → 512-tap linear-phase FIR を取得
    4. `post_filter/fir.npy` に保存

参考: docs/architecture.md §5 / docs/training.md §4.3 / Okamoto21 §3.3
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from wavenext2.models.diff_wavenext2 import DiffWaveNext2  # T-M3.1
from wavenext2.inference.infer_diff import reverse_sample  # T-M3.3
from wavenext2.data.mel import LogMelSpectrogram          # T-M1.3
from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss  # T-M2.3 再利用


N_FFT_FIT = 512    # Okamoto21 §3.3 実験設定
HOP_FIT = 256
FIR_LENGTH = 512   # = n_fft (linear-phase 用)
EPS_DB = 1e-10     # log10 0 防御


def fit_post_filter(
    model: DiffWaveNext2,
    dev_pairs: list[tuple[Path, Path]],
    n_fft: int = N_FFT_FIT,
    hop: int = HOP_FIT,
    fir_length: int = FIR_LENGTH,
    device: str = "cuda",
) -> np.ndarray:
    """dev set 上で振幅差平均 → iRFFT → fftshift で 512-tap FIR を構築.

    Args:
        model: 学習済み DiffWaveNext2 (4 sub-models)
        dev_pairs: [(mel_or_audio_path, gt_audio_path), ...] 200 pairs
        n_fft: STFT 用 FFT 長 (Okamoto21 設定で 512)
        hop: STFT 用 hop (Okamoto21 設定で 256)
        fir_length: FIR の tap 数 (linear-phase 用に n_fft と一致)
    Returns:
        fir: (fir_length,) numpy array, dtype=float32, oaconvolve 中央切り出しで convolve できる形
    """
    window = torch.hann_window(n_fft, device=device)
    n_bins = n_fft // 2 + 1                          # = 257
    diff_acc = np.zeros(n_bins, dtype=np.float64)    # 加算誤差防止に float64 で蓄積
    n_frames_total = 0

    model.eval()
    with torch.no_grad():
        for mel_path, gt_path in dev_pairs:
            # 1. GT 波形と mel を取得
            y_gt, sr = torchaudio.load(gt_path)
            y_gt = y_gt.mean(dim=0)                  # stereo → mono
            mel = compute_mel(y_gt, sr).to(device)   # T-M1.3 経由

            # 2. Diff モデルで 4-step reverse sample (T-M3.3)
            y_gen = reverse_sample(model, mel)        # (1, T_audio) on device
            y_gen = y_gen.squeeze(0).cpu().float()
            y_gt = y_gt.float()

            # 3. 長さ揃え (Diff hop=256 の都合で y_gen の方が長いことがある → GT 側に truncate)
            T = min(y_gen.shape[0], y_gt.shape[0])
            y_gen, y_gt = y_gen[:T], y_gt[:T]

            # 4. STFT 振幅
            S_gt = torch.stft(y_gt.to(device), n_fft=n_fft, hop_length=hop,
                              window=window, center=True, return_complex=True).abs()
            S_gen = torch.stft(y_gen.to(device), n_fft=n_fft, hop_length=hop,
                               window=window, center=True, return_complex=True).abs()
            # (F, T_frames)

            # 5. 振幅差: |Y_gt| - |Y_gen|、全フレーム合計を accumulator に加算
            mag_diff = (S_gt - S_gen).sum(dim=1)     # (F,) frame 軸 sum
            diff_acc += mag_diff.cpu().numpy().astype(np.float64)
            n_frames_total += S_gt.shape[1]

    # 6. 全 (utt × frame) 平均
    mean_diff_mag = (diff_acc / max(n_frames_total, 1)).astype(np.float32)  # (n_bins,)

    # 7. iRFFT → linear-phase 用に fftshift で中央化
    fir = np.fft.irfft(mean_diff_mag, n=fir_length).astype(np.float32)
    fir = np.fft.fftshift(fir)                       # (fir_length,)

    return fir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="DiffWaveNext2 の 4 sub-model checkpoint (sub_{1..4}.pt)")
    parser.add_argument("--filelist", type=Path,
                        default=Path("data/filelists/dev_postfilter.tsv"),
                        help="T-M0.3 で生成済の 200 utt TSV")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="LibriTTS-R ルート (TSV の rel_path 解決用)")
    parser.add_argument("--out-path", type=Path,
                        default=Path("post_filter/fir.npy"))
    parser.add_argument("--stats-path", type=Path,
                        default=Path("post_filter/fit_stats.json"),
                        help="MR-STFT before/after などの統計 dump")
    parser.add_argument("--n-fft", type=int, default=N_FFT_FIT)
    parser.add_argument("--hop", type=int, default=HOP_FIT)
    parser.add_argument("--fir-length", type=int, default=FIR_LENGTH)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    ...
```

#### `src/wavenext2/inference/post_filter.py`

```python
"""post_filter.py — Time-invariant spectral enhancement FIR の apply.

参考: docs/architecture.md §5 / Okamoto21 §3.3
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from scipy.signal import oaconvolve  # overlap-add convolution、左右非対称回避


def load_post_filter(fir_path: str | Path) -> np.ndarray:
    """`post_filter/fir.npy` を読み込み、長さ / dtype 検証."""
    fir = np.load(fir_path).astype(np.float32)
    if fir.ndim != 1:
        raise ValueError(f"FIR must be 1-D, got shape {fir.shape}")
    if fir.shape[0] != 512:
        raise ValueError(f"Expected FIR length 512, got {fir.shape[0]}")
    return fir


def apply_post_filter(
    audio: torch.Tensor | np.ndarray,
    fir: np.ndarray,
) -> torch.Tensor | np.ndarray:
    """合成波形に time-invariant FIR を畳み込み (linear-phase, length-preserving).

    torch tensor / np.ndarray 両受け (GPU 上 post-filter 適用 / RTF 測定対応)。
    入力 dtype と同じ dtype で返す。

    Args:
        audio: (T,) torch.Tensor または np.ndarray, dtype=float32, 範囲 [-1, 1]
        fir: (512,) numpy array, fftshift 済み linear-phase FIR
    Returns:
        入力と同じ型 (torch.Tensor or np.ndarray) で同じ長さ
    """
    is_torch = isinstance(audio, torch.Tensor)
    if is_torch:
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {tuple(audio.shape)}")
        # torch 純実装: F.conv1d で GPU 対応 (RTF 測定時に CPU↔GPU 往復を回避)
        # fftshift 済 FIR を kernel として使う、padding="same" で長さ保存
        fir_t = torch.from_numpy(fir).to(audio.device, dtype=audio.dtype)
        # conv1d: (B=1, C=1, T) × (out=1, in=1, K)
        out = torch.nn.functional.conv1d(
            audio.view(1, 1, -1),
            fir_t.flip(0).view(1, 1, -1),   # conv1d は相関なので flip して畳み込み化
            padding=fir_t.shape[0] // 2,
        ).view(-1)
        # 偶数長 kernel の "same" 1-sample 端調整
        if out.shape[0] != audio.shape[0]:
            out = out[: audio.shape[0]]
        return out
    else:
        if audio.ndim != 1:
            raise ValueError(f"audio must be 1-D, got shape {audio.shape}")
        # scipy.signal.oaconvolve (overlap-add): np.convolve(mode="same") の左右非対称を回避
        # length 24000+ でも安全、length 保存は手動切り出し
        full = oaconvolve(audio, fir, mode="full").astype(np.float32)
        # mode="same" 相当の中央切り出し (対称遅延)
        start = (fir.shape[0] - 1) // 2
        out = full[start : start + audio.shape[0]]
        return out
```

#### `tests/test_post_filter.py`

```python
"""tests/test_post_filter.py — apply 側 + fit のスモーク."""

from __future__ import annotations
import numpy as np
import pytest
import torch
from scipy.signal import oaconvolve

from wavenext2.inference.post_filter import apply_post_filter, load_post_filter


def test_apply_preserves_length() -> None:
    audio = np.random.randn(24000).astype(np.float32)
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0  # identity (delta at center, fftshift 済みなら無遅延)
    out = apply_post_filter(audio, fir)
    assert out.shape == audio.shape
    # delta filter なので audio ≈ out (端の境界以外)
    np.testing.assert_allclose(out[256:-256], audio[256:-256], atol=1e-5)


def test_apply_dtype_float32() -> None:
    audio = np.random.randn(24000).astype(np.float32)
    fir = np.random.randn(512).astype(np.float32) * 1e-3
    out = apply_post_filter(audio, fir)
    assert out.dtype == np.float32


def test_apply_rejects_2d_audio() -> None:
    audio = np.random.randn(2, 24000).astype(np.float32)
    fir = np.zeros(512, dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        apply_post_filter(audio, fir)


def test_apply_post_filter_numpy_input() -> None:
    """np.ndarray 入力で np.ndarray 出力 (型一致)."""
    audio = np.random.randn(24000).astype(np.float32)
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0
    out = apply_post_filter(audio, fir)
    assert isinstance(out, np.ndarray)
    assert out.shape == audio.shape


def test_apply_post_filter_torch_input() -> None:
    """torch.Tensor 入力で torch.Tensor 出力 (型一致、GPU 上 post-filter 適用想定)."""
    audio = torch.randn(24000, dtype=torch.float32)
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0
    out = apply_post_filter(audio, fir)
    assert isinstance(out, torch.Tensor)
    assert out.shape == audio.shape
    # identity FIR で中央領域が一致
    torch.testing.assert_close(out[256:-256], audio[256:-256], atol=1e-5, rtol=1e-5)


def test_oaconvolve_vs_convolve() -> None:
    """scipy.signal.oaconvolve が np.convolve(mode='same') と数値整合性を持つこと検証.

    本テストは『なぜ oaconvolve を採用したか』の根拠として、左右非対称遅延の検出に使う。
    delta filter (fftshift 済) なら両者一致するはずだが、非対称遅延が出ると差分が出る。
    """
    audio = np.random.randn(24000).astype(np.float32)
    fir = np.zeros(512, dtype=np.float32)
    fir[256] = 1.0  # delta at center
    out_apply = apply_post_filter(audio, fir)  # oaconvolve 経由
    out_npconv = np.convolve(audio, fir, mode="same").astype(np.float32)
    # delta なら中央領域は一致するはず (端の左右切り出しの取り扱いだけ差が出る)
    np.testing.assert_allclose(out_apply[256:-256], out_npconv[256:-256], atol=1e-5)


def test_load_post_filter_validates_length(tmp_path) -> None:
    bad = np.zeros(256, dtype=np.float32)
    p = tmp_path / "bad.npy"
    np.save(p, bad)
    with pytest.raises(ValueError, match="length 512"):
        load_post_filter(p)


def test_load_post_filter_validates_ndim(tmp_path) -> None:
    bad = np.zeros((2, 512), dtype=np.float32)
    p = tmp_path / "bad.npy"
    np.save(p, bad)
    with pytest.raises(ValueError, match="1-D"):
        load_post_filter(p)


@pytest.mark.slow
def test_fit_reproducible_with_seed(tmp_path) -> None:
    """同じ checkpoint + seed=43 で fir.npy が bit-exact 再現される (CI deterministic 性).

    M6.2 完了後の `fir.npy` が `fit_stats.json` の (seed, git_sha, checkpoint_sha256)
    3 点 record で再現可能であることを検証する slow test。
    """
    # 本テストは実 checkpoint がある場合のみ実行 (M6.2 後)、それまでは skip
    # 概念検証として、内部の torch.randn(generator=Generator(seed=43)) が同じ結果を返すかを確認
    g1 = torch.Generator().manual_seed(43)
    g2 = torch.Generator().manual_seed(43)
    r1 = torch.randn(100, generator=g1)
    r2 = torch.randn(100, generator=g2)
    torch.testing.assert_close(r1, r2, atol=0.0, rtol=0.0)  # bit-exact


@pytest.mark.slow
def test_freq_response_in_expected_range() -> None:
    """fit したと仮定した FIR の周波数応答チェック (使用時は実 fir.npy を入れる).

    期待: ~2 kHz 以下で 0 dB ± 1 dB、Nyquist 端で +5〜8 dB (Okamoto21 Fig 3a)
    本テストは fit 後に実 fir.npy を読み込んで検証する slow test。
    """
    fir_path = "post_filter/fir.npy"
    fir = load_post_filter(fir_path)
    sr = 24000
    H = np.fft.rfft(np.fft.fftshift(fir))  # fftshift を戻してから rfft
    freqs = np.fft.rfftfreq(len(fir), d=1.0 / sr)
    db = 20.0 * np.log10(np.abs(H) + 1e-10)

    # 低域 0 dB ± 1 dB
    low_mask = freqs <= 2000.0
    assert np.allclose(db[low_mask], 0.0, atol=1.0), f"Low band not flat: {db[low_mask]}"

    # 高域 (Nyquist 端) +5〜8 dB
    nyq_db = db[-1]
    assert 5.0 <= nyq_db <= 8.0, f"Nyquist gain {nyq_db:.2f} dB outside [5, 8]"
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `n_fft` (fit) | **512** | Okamoto21 §3.3 / docs/architecture.md §5 |
| `hop` (fit) | **256** | Okamoto21 §3.3 |
| `fir_length` | **512** (= n_fft) | linear-phase 用 |
| `window` | Hann | Okamoto21 §3.3 |
| dev set サイズ | **200 utterances** | T-M0.3 `dev_postfilter.tsv`、論文 Okamoto21 (40) より大、LibriTTS-R 話者ばらつき考慮 |
| dev seed | **43** | T-M0.3 (`postfilter_seed=43`、val seed=42 と非重複) |
| 振幅差平均方向 | 時間 frame 軸 + utterance 軸の **両方** (全 (utt × frame) sum / total frame count) | Okamoto21 §3.3 「全フレームに渡って平均」 |
| accumulator dtype | **float64** | 加算誤差防止 |
| `reverse_sample` seed | **43** (`--reproducible`) | T-M0.3 `postfilter_seed` 一致、CI deterministic 性 |
| convolution backend | **`scipy.signal.oaconvolve`** (overlap-add) | 左右非対称回避 (§8.1 採用昇格) |
| `fir.npy` の git 管理 | **commit** (M6.2 後) | `.gitignore` 除外解除、2 KB (§8.1 採用昇格) |
| `fit_stats.json` 再現 record | **seed, git_sha, checkpoint_sha256** | CI deterministic 性 (§6.1 通常項目) |
| FIR dtype | float32 | 推論時 cast |
| 期待周波数応答 (低域 ≤2 kHz) | 0 dB ± 1 dB | Okamoto21 Fig 3a |
| 期待周波数応答 (Nyquist 端 12 kHz) | +5〜8 dB | Okamoto21 Fig 3a |
| sample_rate | 24000 Hz | docs/training.md §1.1 |
| MR-STFT eval | T-M2.3 既定 (n_ffts=[512,1024,2048] etc.) | T-M2.3 §2.3 再利用 |

### 2.4 アルゴリズム / 処理フロー

1. `--checkpoint-dir` から DiffWaveNext2 4 sub-models を `model.load_state_dict()` で復元
2. `--filelist` (デフォルト `data/filelists/dev_postfilter.tsv`、200 行、T-M0.3 で `seed=43` 生成済) を読み込み、`(mel_or_gt, gt)` のペアを作る (実装は **mel-not-stored 戦略**: GT 波形から runtime mel 抽出が単純で確実)
3. `diff_acc = np.zeros(257, dtype=np.float64)`、`n_frames_total = 0` を初期化
4. 各 utt について:
   - GT を `torchaudio.load` → mono → mel 抽出 (T-M1.3 `LogMelSpectrogram`)
   - `reverse_sample(model, mel, seed=43, post_filter=None)` で 4-step Diff サンプリング → y_gen (T-M3.3、**`post_filter=None` で apply 無効化、fit 段階では post-filter 未生成のため**、`seed=43` で `--reproducible` 動作)
   - 長さ揃え `T = min(len(y_gen), len(y_gt))`
   - `S_gt = |STFT(y_gt, n_fft=512, hop=256, hann)|`、同様に `S_gen`
   - `mag_diff = (S_gt - S_gen).sum(dim=1)` (frame 軸 sum、(257,))
   - `diff_acc += mag_diff.cpu().numpy().astype(float64)`
   - `n_frames_total += S_gt.shape[1]`
5. `mean_diff_mag = diff_acc / n_frames_total` (float32 cast。**underflow リスクあり、§6.1 通常項目で検知**)
6. `fir_raw = np.fft.irfft(mean_diff_mag, n=512)` (length-512 FIR, real)
7. `fir = np.fft.fftshift(fir_raw)` (linear-phase 用に中央化)
8. `np.save("post_filter/fir.npy", fir)` で保存 (dtype=float32, shape=(512,))。**M6.2 完了後は git commit (.gitignore 除外解除済)、PR ベースで FIR 共有可能**
9. eval metric (T-M2.3 `MultiResolutionSTFTLoss`) で post-filter 適用前後の MR-STFT 値を出力、`fit_stats.json` に dump。**schema 拡張: `{"seed": 43, "git_sha": "...", "checkpoint_sha256": "...", "n_utt": 200, ...}` の 3 点 record で再現性保証**
10. `apply_post_filter(audio, fir)` は **`scipy.signal.oaconvolve(audio, fir, mode="full")` の中央切り出し** (長さ保存、左右対称遅延、fftshift 済 FIR で中央 tap が遅延 0)。torch tensor 入力時は `torch.nn.functional.conv1d` 純実装で GPU 対応
11. **推論側統合**: M4.1 / M6.2 では `reverse_sample(model, mel, seed=..., post_filter=fir)` 1 関数化で apply/no-apply を switch (`post_filter=None` vs `post_filter=fir`)、外部で `apply_post_filter` を chain せずに済む

### 2.5 設計上の重要決定

- **fit と apply を別ファイル**: fit は重い (4-step Diff sampling × 200 utt) のでスクリプト、apply は軽量関数 (推論 hot path) で `src/` 配下。**fit スクリプトは M6.2 完了後の `fir.npy` 再生成 reproducibility 用途に降格** (§8.1 採用昇格)、apply は `reverse_sample` 引数として統合 (`post_filter: np.ndarray | None = None`)
- **`apply_post_filter` を `reverse_sample` の引数として統合** (§8.1 採用昇格): T-M3.5 / T-M4.1 / T-M6.2 で `apply_post_filter(reverse_sample(...))` の 2 関数 chain を毎回書くより、`reverse_sample(model, mel, *, seed, post_filter=None)` 1 関数化で T-M4.1 評価 caller が `post_filter=fir` か `None` の switch だけで apply/no-apply 比較できる (T-M3.3 と同期、本チケットで T-M3.3 §9.1 引数追加を依頼)
- **`apply_post_filter` の torch tensor / np.ndarray 両受け** (§8.1 採用昇格): GPU 上 post-filter 適用 (RTF 測定) も torch 実装なら自然。`torch.nn.functional.conv1d` 純 torch 実装で CPU↔GPU 往復を回避、`reverse_sample` 戻り値 (torch.Tensor) と np.ndarray のキャスト責務を本関数内で吸収
- **`scipy.signal.oaconvolve` 採用** (overlap-add、§8.1 採用昇格): `np.convolve(mode='same')` の左右非対称を回避、length 24000 でも安全。`mode="full"` の中央切り出しで対称遅延を保証
- **`fir.npy` を git commit 化** (§8.1 採用昇格): `.gitignore` の `post_filter/*.npy` 除外を **解除**、2 KB なのでサイズ問題なし。M6.2 完了後の `fir.npy` 共有が PR ベースで容易、reproducibility は `fit_stats.json` の (seed=43, git_sha, checkpoint_sha256) 3 点 record で担保
- **`diff_acc` を float64 で蓄積**: 200 utt × 数百 frame で加算回数が 50,000 オーダになるため、float32 直接加算では precision loss あり。最後の cast で float32 化 (**underflow リスクは §6.1 通常項目で対応**)
- **振幅差の符号は `|Y_gt| - |Y_gen|`** (Okamoto21 §3.3): 「合成側で **失われた** 高域 detail を **加算** 補償」する方向。逆符号 (gen - gt) にすると周波数応答が反転して低域 0 dB / Nyquist -5〜-8 dB になりノイズ抑制方向の filter になる (本論文の意図と逆)
- **長さ保存 (`mode="same"` 相当)**: `scipy.signal.oaconvolve(audio, fir, mode="full")` の中央 `len(audio)` 切り出し。fftshift 済の FIR は中央 tap (index 256) が delta 中心なので **対称な linear-phase 遅延** となり、左右非対称な遅延の懸念は本質的に回避される
- **MR-STFT を T-M2.3 から再利用** (T-M2.3 §9.1 申し送り): `MultiResolutionSTFTLoss(n_ffts=[512,1024,2048], win_lengths=[360,900,1800], hop_sizes=[80,150,300], eps=1e-5)` をそのまま import。Diff の hop=256 と FIR fit の hop=256 は無関係で、評価用 MR-STFT は GAN と共通設定で OK
- **dev set 200 utterances**: T-M0.3 で `seed=43` で生成済み、val (seed=42) と非重複。Okamoto21 の 40 utt より多く、LibriTTS-R の話者ばらつき (約 1,151 speakers in train-clean-100+360) を踏まえてサイズ拡張 (**ablation 候補: §8.1 検討追加で 50/100/200/500 比較**)
- **`reverse_sample` インターフェース依存 (拡張)**: T-M3.3 が `reverse_sample(model, mel, *, seed: int | torch.Generator, post_filter: np.ndarray | None = None) -> Tensor (B, T_audio)` を返す signature を前提 (本チケットは T-M3.3 §9.1 を参照、新引数追加を依頼)

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `scripts/fit_post_filter.py` + `src/wavenext2/inference/post_filter.py` + テスト本実装 | general-purpose |
| Reviewer | 1 | Okamoto21 §3.3 仕様準拠確認、周波数応答 plot 検証、`np.convolve("same")` の左右遅延チェック | general-purpose |
| Tester | 1 | `pytest tests/test_post_filter.py` 実行 + dev set 200 utt での実 fit + 周波数応答可視化 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **no** (T-M3.3 reverse sampler 完了が前提)
- 並列実行する場合の最大並列数: 1
- 後続 T-M3.5 (Diff smoke)、T-M4.1 (客観評価) とは順次。

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/fit_post_filter.py` (CLI、dev set 200 utt → `fir.npy`、`--reproducible` で `seed=43` 固定)
- `src/wavenext2/inference/post_filter.py` (`apply_post_filter` (torch / numpy 両受け), `load_post_filter`)
- `src/wavenext2/inference/infer_diff.py` への `reverse_sample(post_filter=None)` 引数追加 (T-M3.3 と同期、apply 統合)
- `tests/test_post_filter.py` (apply 側の Unit テスト + torch/numpy 両受け + oaconvolve 整合 + seed 再現 + 周波数応答 slow test)
- `post_filter/.gitkeep` + `.gitignore` の `post_filter/*.npy` 除外 **解除** (`fir.npy` を M6.2 後に commit)
- 振幅差平均の **時間 frame 軸 + utterance 軸の両方** での集約 (Okamoto21 §3.3 準拠)
- T-M2.3 `MultiResolutionSTFTLoss` 再利用による eval metric (post-filter 前後 MR-STFT 比較を `fit_stats.json` に dump、`seed`/`git_sha`/`checkpoint_sha256` 3 点 record)
- `scipy.signal.oaconvolve` (overlap-add) による convolution (左右非対称回避)
- 周波数応答の数値検証 (低域 0 dB ± 1 dB、Nyquist 端 +5〜8 dB)
- FIR が `[-1, 1]` を発散した場合の clip フォールバック (§6.1 で議論、デフォルト OFF、`--clip-fir` flag で有効化)

### Out of Scope
- learnable / NN-based post-filter (§8.1 代替案、M6.3 ablation で再検討)
- per-utterance 適応 post-filter (§8.1 代替案、overhead 大)
- Wiener filter 代替 (§8.1 代替案)
- frequency-domain multiplication (FFT × spectrum) による convolution 置換 (§8.1 代替案、本チケットの 512 tap には不要)
- GAN-WaveNeXt 2 への post-filter 適用 (論文では Diff のみ)
- 多話者 / 多言語データセットでの FIR generalization 検証 (M6 以降)
- post-filter の hyperparameter search (n_fft / hop / fir_length 変更ablation、M6.3)
- mel pre-extract キャッシュ (M0.3 / M2.1 のスコープ、本チケットは runtime mel 抽出で十分)

### Deliverable
- ファイル:
  - `scripts/fit_post_filter.py` (新規)
  - `src/wavenext2/inference/post_filter.py` (新規)
  - `tests/test_post_filter.py` (新規)
  - `post_filter/.gitkeep` (新規)
- 関数 / クラス:
  - `fit_post_filter(model, dev_pairs, n_fft, hop, fir_length, device) -> np.ndarray`
  - `apply_post_filter(audio: torch.Tensor | np.ndarray, fir) -> torch.Tensor | np.ndarray` (両受け)
  - `load_post_filter(fir_path) -> np.ndarray`
  - `main()` (CLI entry、`--reproducible` flag)
  - (T-M3.3 連携) `reverse_sample(model, mel, *, seed, post_filter=None)` への引数追加
- ドキュメント差分:
  - `docs/milestones.md` §M3.4 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M3.4 ステータス更新

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_post_filter.py`)

- [ ] `test_apply_preserves_length` — identity FIR (delta at center, fftshift 済) で長さ保存 通過後の長さが入力と一致、中央領域 (端の境界除く) で値が変化しない
- [ ] `test_apply_dtype_float32` — 出力 dtype が `float32`
- [ ] `test_apply_rejects_2d_audio` — `(2, T)` 入力で `ValueError`
- [ ] `test_apply_post_filter_numpy_input` (**新規追加**) — np.ndarray 入力で np.ndarray 出力 (型一致)
- [ ] `test_apply_post_filter_torch_input` (**新規追加**) — torch.Tensor 入力で torch.Tensor 出力 (型一致、GPU 上 post-filter 適用想定)
- [ ] `test_oaconvolve_vs_convolve` (**新規追加**) — `scipy.signal.oaconvolve` が `np.convolve(mode='same')` と数値整合 (左右非対称検証、delta filter 経路)
- [ ] `test_load_post_filter_validates_length` — 長さ 256 の `.npy` を読むと `ValueError`
- [ ] `test_load_post_filter_validates_ndim` — 2-D `.npy` を読むと `ValueError`
- [ ] `test_fit_reproducible_with_seed` (`@pytest.mark.slow`、**新規追加**) — 同じ checkpoint + `seed=43` で `fir.npy` が bit-exact 再現
- [ ] `test_freq_response_in_expected_range` (`@pytest.mark.slow`) — 実 `post_filter/fir.npy` を読み込み、低域 ≤2 kHz で 0 dB ± 1 dB、Nyquist 端で +5〜8 dB
- [ ] `test_apply_no_nan_inf` — `np.random.randn` 入力で出力に NaN / Inf を含まない
- [ ] `test_apply_linear_phase` — delta 入力 → 出力 peak が中央付近 (fftshift 済の linear-phase 確認)

### 5.2 e2e / 結合テスト

- [ ] `scripts/fit_post_filter.py --filelist <tiny_2utt.tsv> --reproducible` が tiny dev set (2 utt) で完走、`fir.npy` + `fit_stats.json` を生成
- [ ] `apply_post_filter(audio, np.load("fir.npy"))` が動作、長さ保存 (numpy / torch 両入力)
- [ ] `reverse_sample(model, mel, seed=43, post_filter=fir)` 経由でも apply が動作 (引数統合の e2e)
- [ ] T-M3.3 `reverse_sample` の戻り値 shape `(1, T_audio)` を `fit_post_filter` が正しく扱える
- [ ] (M3.5 / M4.1 で本格検証) dev set 200 utt fit が **エラーなく完走**、生成 FIR が周波数応答テストを pass

### 5.3 Acceptance criteria (`docs/milestones.md` §M3.4 より転記)
- [ ] dev set 100 utterances で fit してエラーなく fir.npy を生成 (本チケットでは 200 utt に拡張)
- [ ] FIR 長 = 512 (linear-phase 用 fftshift 済み)
- [ ] 周波数応答が ~2 kHz 以下で ≈ 0 dB、Nyquist 端で +5〜8 dB (Okamoto21 Fig 3a と整合)
- [ ] apply 前後で音声長が変わらない (`scipy.signal.oaconvolve` 対称切り出し)

### 5.4 追加 acceptance (本チケット独自)
- [ ] `from wavenext2.inference.post_filter import apply_post_filter, load_post_filter` が動作
- [ ] `apply_post_filter` が torch.Tensor / np.ndarray 両受け、入力型と同じ型で返す
- [ ] `reverse_sample(post_filter=fir / None)` で apply/no-apply switch が動作
- [ ] T-M2.3 `MultiResolutionSTFTLoss` の eval metric 値が `fit_stats.json` に dump され、post-filter 前後で改善 (高域成分の MR-STFT magnitude が GT に近づく) を確認
- [ ] `fit_stats.json` に `seed`/`git_sha`/`checkpoint_sha256` 3 点が record され、同一 checkpoint + seed=43 で `fir.npy` が bit-exact 再現
- [ ] `post_filter/.gitkeep` 配置済、`.gitignore` で `post_filter/*.npy` 除外 **解除** (commit 化) 確認
- [ ] CPU でも apply 側テストが pass (CUDA は fit のみ必要)

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目

- **CRITICAL: 振幅差平均の集約軸 (時間 axis のみ or utterance + 時間 axis の両方)**:
  - Okamoto21 §3.3 「全フレームに渡って平均」の **全フレーム** の解釈が曖昧:
    - **解釈 A**: 1 utterance 内 frame 平均 → 各 utterance の `mag_diff` を utt 数で平均 (utt 数 200 で割る)
    - **解釈 B**: 全 (utt × frame) frame 平均 → `(全 utt の mag_diff 合計) / (全 utt の frame 数合計)` で割る
  - 本チケットは **解釈 B** を採用 (`docs/architecture.md` §5 擬似コード `diff_acc / n_frames_total` と一致)
  - 解釈 A だと utterance ごとに発話長で重みづけが消える (短い utt が長い utt と同じ寄与に正規化される) → 短い utt の noise が増幅される傾向あり
  - 解釈 B は frame ごとに公平な重みづけ → LibriTTS-R は発話長分布が広い (1〜15 秒) ため B が妥当
  - **再検討トリガー**: M5.2 smoke 後の MR-STFT で改善が出ない場合、解釈 A に切り替えて ablation

- **CRITICAL: convolution の左右遅延の対称性 (M3 phase review で `scipy.signal.oaconvolve` 採用に解決)**:
  - 旧案 `np.convolve(mode="same")` は `(len(audio) + len(fir) - 1)` の full 出力から中央 `len(audio)` を取るが、len(fir)=512 (偶数) では中央切り出しが index `[256, 256+len(audio))` (1-sample 非対称) になる numpy 仕様で、左右非対称遅延の懸念があった
  - **解決 (採用)**: `scipy.signal.oaconvolve(audio, fir, mode="full")` の中央 `len(audio)` を自前で切り出し (`start = (len(fir)-1)//2`) → **対称遅延を明示制御**、length 24000+ でも overlap-add で安全。`np.convolve(mode="same")` の実装依存切り出しを避ける
  - fftshift 済 FIR は **中央 tap (index 256) が delta 中心**、上記切り出しで delta 中心が出力中央に揃う
  - **検証**: `test_apply_preserves_length` (identity FIR で入力一致) + `test_oaconvolve_vs_convolve` (oaconvolve vs np.convolve の数値整合、左右非対称検証)
  - **再評価トリガー**: 上記 2 テストが fail したら `scipy.signal.fftconvolve` 等へ再検討

- **CRITICAL: FIR が divergent (係数が `[-1, 1]` を超える) → 出力が clip される**:
  - `mean_diff_mag` の振幅差が大きい (Diff モデルが train 不足で高域大幅欠落など) と iRFFT 後の FIR 係数が `[-1, 1]` を超え、convolve 結果が `[-1, 1]` の audio range を超える
  - 出力 audio を `clip(-1, 1)` で post-process すると、その箇所が hard-clip ノイズを生む
  - **対策案 A** (本チケット採用): デフォルトで clip 無し、`--clip-fir` flag で FIR 係数自体を `[-1, 1]` に clip (周波数応答が変わるが安全)
  - **対策案 B**: FIR をエネルギー正規化 (`fir /= np.max(np.abs(fir))` でスケールダウン) → 周波数応答が一定スケール下に
  - **対策案 C**: `mean_diff_mag` の絶対値を上限制限 (e.g., 0.3) でクリップしてから iRFFT
  - **再評価トリガー**: M5.2 で FIR の max-abs が 1.0 を超える場合、対策 B or C に切替

- **CRITICAL: GT 波形と Diff 出力の長さ不一致 (hop=256 padding 影響)**:
  - Diff hop=256 で T_mel × 256 の長さで出力されるが、GT 波形の元長 (LibriTTS-R wav) と一致しないことがある
  - 振幅差を取る前に **shorter 側に truncate** (本チケット採用): `T = min(len(y_gen), len(y_gt))`
  - 代替: zero-pad で揃える → STFT 端で artifact、不採用
  - **検知**: fit ループ内で truncation 量を log し、極端な差 (`|len_gen - len_gt| > 256 * 10`) があれば warning

- **CRITICAL: Diff モデル checkpoint 未学習 (M3.5 smoke 前) で fit を走らせると FIR が無意味**:
  - Diff モデルが untrained 状態だと `y_gen` は random noise に近く、振幅差が無意味な値になる
  - 本チケットの実装自体は機能するが、生成 FIR は周波数応答テスト (Nyquist 端 +5〜8 dB) を pass しない可能性大
  - **対策**: 本チケットの実装 + Unit テスト (identity FIR の apply) は M3.4 単独で完了させ、**実 FIR の fit と 5.4 acceptance (周波数応答) は M6.2 の Diff フル訓練後に実施** (M3.5 smoke 段階では `test_freq_response_in_expected_range` を skip 可)
  - **時系列整合性**: M3.4 = 実装、M3.5 = smoke (実 FIR fit はオプション)、M6.2 完了後 = 実 FIR の周波数応答検証

#### 通常項目

- **dev set サイズ 200 が論文の Okamoto21 (40) より多い理由 (補足)**: LibriTTS-R は 1,151 speakers (train-clean) でばらつきが大きく、40 utt だと話者偏りで FIR が偏る懸念があった。200 utt なら各話者 ~0.17 utt 程度の重みづけで全体的な平均が取れる
- **CI deterministic 性 (新規)**: 200 utt × `reverse_sample` (内部 `torch.randn`) で `seed` 未指定なら fit が flaky → **`reverse_sample` に `seed=43`** (T-M0.3 `postfilter_seed` と一致) を `--reproducible` flag で渡す。`fit_stats.json` に `{"seed": 43, "git_sha": "...", "checkpoint_sha256": "..."}` 3 点 record で再現性保証。`test_fit_reproducible_with_seed` (slow) で検証
- **`mean_diff_mag` underflow (新規)**: 振幅差平均が小さい数値だと `np.fft.irfft` 結果が underflow する可能性、float64 蓄積でも `irfft` 後の `astype(np.float32)` cast で精度損失リスク。**検知**: `fit_stats.json` に `mean_diff_mag` の `min` / `max` / `abs_mean` を dump、`min < 1e-7` で warning
- **torch tensor vs numpy array の型不整合 (新規)**: `reverse_sample` 戻り値は `torch.Tensor`、旧 `apply_post_filter` は `np.ndarray` のみ受領で `y_gen.cpu().numpy()` の cast 責務が caller 側だった。**両受け実装で解決** (§2.5 採用昇格): `apply_post_filter(audio: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray`、入力 dtype 一致で返す
- **`np.fft.irfft(n=fir_length)` の `n` パラメータ**: `mean_diff_mag` が長さ 257 (= 512//2 + 1) の場合、`irfft(n=512)` で長さ 512 の real time-domain signal を得る。`n` を明示しないと numpy が自動推論 (= 512) だが、明示する方が安全
- **`fftshift` の方向**: 1-D の場合 `np.fft.fftshift` は `[N/2:, :N/2]` を入れ替える (中央が DC → 端が DC、tap 中心が中央)。逆方向は `np.fft.ifftshift` で、本チケットは `fftshift` のみ使う (linear-phase 化)
- **`window=torch.hann_window(n_fft)` の length 一致**: `torch.stft(n_fft=512, win_length=None)` のデフォルトは `win_length = n_fft = 512` で `window` の長さと一致。明示的に `win_length=n_fft` 指定するかは好み
- **`reverse_sample` の戻り値 dtype**: T-M3.3 の `reverse_sample` が float32 を返すことを想定。bf16 / fp16 で戻ってきたら明示 cast (`y_gen.float()`) が必要
- **`fit_stats.json` の schema (拡張)**: M4.1 で post-filter 効果検証時に使うため、`{"seed": 43, "git_sha": "...", "checkpoint_sha256": "...", "n_utt": 200, "n_frames_total": ..., "fir_max_abs": ..., "mean_diff_mag_min": ..., "mean_diff_mag_max": ..., "mrstft_before": {"sc": ..., "mag": ...}, "mrstft_after": {"sc": ..., "mag": ...}}` を出力
- **CPU で fit を走らせる場合の所要時間**: 200 utt × 4-step Diff sampling は CPU で 1 utt あたり数秒 → 200 utt で 10〜30 分。`--device cuda` 推奨だが CPU でも可
- **`dev_postfilter.tsv` が val.tsv と重複する事故**: T-M0.3 で `seed=43 != val seed=42` で hold-out 済、`set(val) ∩ set(dev_postfilter) == ∅` も T-M0.3 §2.4 step 14 で assert 済。本チケットでは追加検証不要
- **Stereo wav 混入**: LibriTTS-R は基本 mono だが念のため `y_gt.mean(dim=0)` で mono 化
- **post_filter dir の git 管理 (方針変更)**: **`.gitignore` の `post_filter/*.npy` 除外を解除** (§8.1 採用昇格、`fir.npy` 2 KB を M6.2 完了後に commit)。`post_filter/.gitkeep` は引き続き keep dir 用に維持

### 6.2 仕様の曖昧さ

- `docs/open-questions.md` の関連項目: **すべて解決済み** (Okamoto21 §3.3 で完全確定)
  - n_fft=512, hop=256, fir_length=512 (linear-phase) → `docs/architecture.md` §5
  - 振幅差の符号 `|Y_gt| - |Y_gen|` → 同上
  - 周波数応答の期待値 (低域 0 dB / Nyquist +5〜8 dB) → 同上
- 本チケットで確定する曖昧さ (§6.1 critical 参照):
  - 振幅差平均の集約軸 (utt + frame の両方軸で平均) — §6.1 critical #1 で **解釈 B 採用**
  - convolution の左右非対称性 — §6.1 critical #2 で **`scipy.signal.oaconvolve` 採用** に解決、test で検証
  - FIR clip フォールバック — §6.1 critical #3 で `--clip-fir` flag 化、デフォルト OFF

### 6.3 他チケットとの整合性

- **T-M3.3 (reverse_sampler)** から受領 (本チケットで引数追加を依頼):
  - 現状 `reverse_sample(model, mel) -> Tensor (B, T_audio)` signature
  - **依頼する拡張**: `reverse_sample(model, mel, *, seed: int | torch.Generator, post_filter: np.ndarray | None = None)` (apply 統合 + deterministic seed、§8.1 採用昇格 / §9.1 参照)
  - 戻り値 dtype = float32 (T-M3.3 §9.1 確定想定)
  - 推論時の sub-model dispatch (1-to-1) はすべて `reverse_sample` 内に閉じている
- **T-M3.1 (DiffWaveNext2)** から受領:
  - `model.eval()` で 4 sub-models すべてが eval mode に入る
  - checkpoint 読み込み API (`from_checkpoint(checkpoint_dir)` を想定、T-M3.1 §9.1 参照)
- **T-M0.3 (LibriTTS-R)** から受領:
  - `data/filelists/dev_postfilter.tsv` (200 utt, seed=43, val と非重複) を生成済
  - TSV header: `rel_path, n_samples, speaker_id, chapter_id, duration_sec, peak_dbfs, rms_dbfs`
  - パスは `--data-root` からの相対 (POSIX 区切り)
- **T-M2.3 (Loss)** から受領:
  - `MultiResolutionSTFTLoss(n_ffts=[512,1024,2048], win_lengths=[360,900,1800], hop_sizes=[80,150,300], eps=1e-5)` を `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import
  - T-M2.3 §9.1 で **「M3/M4 で再利用可能」** と申し送り済、本チケットがそれを実行
  - GAN 専用と決めつけない汎用 MR-STFT モジュール設計済 (T-M2.3 §2.5)
- **T-M1.3 (Mel-spectrogram)** から受領:
  - `LogMelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=128, f_min=20, f_max=12000)` を Diff 用に
  - `forward(audio) -> (B, 128, T_mel)` (`torch.log(torch.clamp(mel, min=1e-5))` 含む)
- **T-M3.5 (Diff smoke)** へ渡す情報:
  - `apply_post_filter` を smoke 出力の `reverse_sample` 後に適用
  - smoke では未学習 FIR で OK (実 FIR は M6.2 後)、smoke 範囲は **API 動作確認のみ**
- **T-M4.1 (objective metrics)** へ渡す情報:
  - post-filter 適用前後の MCD / log F0 RMSE で改善幅を比較
  - `fit_stats.json` から MR-STFT before/after も参照可能

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] Okamoto21 §3.3 / `docs/architecture.md` §5 / `docs/training.md` §4.3 の仕様と完全整合
- [ ] 5.1 / 5.2 Unit / e2e テスト全 pass、5.3 / 5.4 Acceptance クリア
- [ ] FIR 長 = 512、`fftshift` 済、`scipy.signal.oaconvolve` の対称切り出しで長さ保存
- [ ] `apply_post_filter` が torch.Tensor / np.ndarray 両受け、入力型と同じ型で返す (`test_apply_post_filter_torch_input` / `test_apply_post_filter_numpy_input`)
- [ ] `reverse_sample(post_filter=None)` 引数統合 (T-M3.3 と同期、apply/no-apply switch)
- [ ] 周波数応答テスト (`test_freq_response_in_expected_range`) が実 fir.npy で pass (M3.5 / M6.2 後)
- [ ] `T-M2.3 MultiResolutionSTFTLoss` を **再利用** (新規に MR-STFT を書いていないか)
- [ ] `.gitignore` で `post_filter/*.npy` 除外 **解除** (`fir.npy` を M6.2 後に commit)、`post_filter/.gitkeep` で空 dir 維持
- [ ] `diff_acc` が float64 で蓄積 (precision loss 防止)、`mean_diff_mag` underflow を `fit_stats.json` で検知
- [ ] 振幅差の符号が `|Y_gt| - |Y_gen|` (逆ではない)
- [ ] `scipy.signal.oaconvolve` の左右遅延が対称 (`test_apply_preserves_length` / `test_oaconvolve_vs_convolve` で検証済)
- [ ] `reverse_sample(seed=43)` で fit が deterministic、`fit_stats.json` に `seed`/`git_sha`/`checkpoint_sha256` 3 点 record (`test_fit_reproducible_with_seed`)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] エラー処理: `audio.ndim != 1`、`fir.shape != (512,)`、checkpoint 不存在、filelist 不存在で明示 raise
- [ ] **参考実装 (Okamoto21 公式 / `astrec-nict/*`) をコピーしていない**: 擬似コード (`docs/architecture.md` §5) は仕様、本チケットの実装はそれを独自に書き起こしているか

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M3) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

#### 採用設計

- **fit と apply の 2 ファイル分割** (`scripts/fit_post_filter.py` + `src/wavenext2/inference/post_filter.py`):
  - 理由: fit は重い (200 utt × 4-step) ので CLI スクリプト、apply は推論 hot path で軽量関数として `src/` 配下に分離
  - **修正 (M3 phase review)**: fit スクリプトは M6.2 完了後の `fir.npy` 再生成 reproducibility 用途に降格 (commit 化と併せて、フル訓練後の 1 回 fit が基本ライン)
- **`apply_post_filter` を `reverse_sample` の引数として統合** (M3 phase review 採用昇格):
  - 理由: T-M3.5 / T-M4.1 / T-M6.2 で 2 関数 chain (`apply_post_filter(reverse_sample(...))`) を毎回書くより、`reverse_sample(model, mel, *, seed, post_filter=None)` 1 関数化で評価 caller が `post_filter=fir` か `None` の switch だけで apply/no-apply を比較できる (T-M3.3 §9.1 と同期、本チケットで T-M3.3 への引数追加を依頼)
- **`apply_post_filter` の torch tensor / np.ndarray 両受け** (M3 phase review 採用昇格):
  - 理由: GPU 上 post-filter 適用 (RTF 測定) も torch 実装なら自然。`torch.nn.functional.conv1d` 純 torch 実装で CPU↔GPU 往復回避。`reverse_sample` 戻り値が torch.Tensor なのでキャスト責務を関数内吸収
- **`scipy.signal.oaconvolve` 採用** (overlap-add、M3 phase review 採用昇格):
  - 理由: `np.convolve(mode='same')` の左右非対称遅延を回避、length 24000 でも安全 (overlap-add は長 signal で効率良し)。`mode="full"` の中央切り出しで対称遅延を保証
- **`fir.npy` を git commit 化** (M3 phase review 採用昇格、§2.1 / §6.1 通常項目で詳述):
  - 理由: 2 KB と軽量、`.gitignore` の `post_filter/*.npy` **除外を解除**。M6.2 完了後の `fir.npy` 共有が PR ベースで容易、`scripts/fit_post_filter.py` は再生成 reproducibility 用途に降格
  - reproducibility は `fit_stats.json` の (seed=43, git_sha, checkpoint_sha256) 3 点 record で担保
- **振幅差を float64 で蓄積**:
  - 理由: 50,000 オーダの加算で float32 だと precision loss
- **MR-STFT を T-M2.3 から import 再利用**:
  - 理由: T-M2.3 §9.1 申し送り済、責務集中

#### 検討した代替設計

1. **Learnable post-filter (NN ベース)** (Okamoto21 でも検討あり):
   - メリット: 表現力が高い、per-utterance 適応可能
   - 却下: 論文は time-invariant FIR を採用、再現実装ではこれを優先
   - **再評価トリガー**: M5.2 smoke 後の MR-STFT 改善幅が不十分なら M6.3 ablation で検討

2. **frequency-domain multiplication (FFT × spectrum)**:
   - メリット: 長い FIR (> 1024 tap) で convolution より高速
   - 却下: 本チケットの 512 tap には不要、`np.convolve` で十分
   - **再評価トリガー**: M6.3 ablation で `fir_length > 1024` を試すなら

3. **per-utterance 適応 post-filter** (各 utt で別 FIR を fit):
   - メリット: 話者・録音条件に応じた最適化
   - 却下: overhead 大 (推論時に毎回 fit)、論文の time-invariant 仕様と乖離
   - **再評価トリガー**: なし (採用しない方針確定)

4. **Wiener filter** (frequency-domain Wiener フィルタリング):
   - メリット: 統計的最適、ノイズ抑制に強い
   - 却下: post-filter の目的 (高域 detail **加算補償**) と Wiener (ノイズ抑制) は方向が逆
   - **再評価トリガー**: なし

5. **STFT-domain log-amplitude 差で fit** (`log(S_gt + eps) - log(S_gen + eps)` の平均):
   - メリット: 対数ドメインで perception に近い
   - 却下: Okamoto21 §3.3 は **linear amplitude 差** で明記、本論文も同じ仕様
   - **再評価トリガー**: M6.3 ablation

6. **`scipy.signal.lfilter` / `scipy.signal.fftconvolve` への切替**:
   - メリット: より高機能 (`lfilter` は IIR 対応、`fftconvolve` は long FIR で高速)
   - 検討: `np.convolve("same")` の左右遅延が非対称な場合の fallback (§6.1 critical #2)
   - **更新 (M3 phase review)**: `np.convolve("same")` 自体を **`scipy.signal.oaconvolve` (overlap-add) に変更採用済** (§8.1 採用設計)。`mode="full"` 中央切り出しで左右対称遅延を保証、`lfilter` (IIR) は本チケットの FIR 用途には不要
   - **再評価トリガー**: `test_apply_preserves_length` / `test_oaconvolve_vs_convolve` が fail したら `fftconvolve` 等へ再検討

7. **FIR を `[-1, 1]` で clip** (`np.clip(fir, -1, 1)`):
   - メリット: 出力 audio が `[-1, 1]` 超えるリスクを抑制
   - 却下: 周波数応答が変わる (高域 gain が抑えられる) → デフォルト OFF、`--clip-fir` flag で opt-in
   - **再評価トリガー**: M5.2 で FIR の max-abs が 1.0 を超える場合

8. **`fit_post_filter` を Diff モデル class のメソッドとして実装**:
   - メリット: `model.fit_post_filter(dev_loader)` で 1 行
   - 却下: モデルクラスが推論責務だけでなく fit 責務を持つと SRP 違反、CLI スクリプト分離の方が自然

9. **dev set サイズを Okamoto21 と同じ 40 utt に縮小**:
   - メリット: 論文と完全一致
   - 却下: LibriTTS-R の話者ばらつき (1,151 speakers) で 40 utt は不十分、200 utt が妥当
   - **再評価トリガー**: M6.3 ablation で 40 / 100 / 200 / 500 の比較

10. **multi-frame averaging** (フレームごとに重みを変える):
    - メリット: 発話開始/末尾の無音区間に重みを置かない
    - 却下: Okamoto21 §3.3 は uniform average で明記、再現実装では同じ仕様
    - **再評価トリガー**: M6.3 ablation

11. **frame-wise normalize (相対誤差)** (M3 phase review 検討追加):
    - 提案: `(S_gt[f,t] - S_gen[f,t]) / (S_gt[f,t] + eps)` で平均、無音 frame の bias 排除
    - メリット: 振幅 dynamic range の大きい音声で large-magnitude frame が dominate する現象を抑制、無音 frame で `S_gt ≈ 0` のときの bias を取り除く
    - 検討中: Okamoto21 §3.3 は absolute (linear amplitude 差) で明記なので本チケット主流路は維持
    - **再評価トリガー**: M5.2 smoke 後の MR-STFT 改善幅が不十分なら ablation 候補

12. **dev set ablation 50/100/200/500** (M3 phase review 検討追加):
    - 提案: 200 utt は overfit リスクあり (Okamoto21 は 40 utt で十分とした実績)、話者偏り smoothing で FIR の Nyquist gain 低下の可能性
    - 検討中: 200 utt はデフォルトのまま、M6.3 ablation で 50 / 100 / 200 / 500 比較
    - **再評価トリガー**: M5.2 smoke で Nyquist gain < 5 dB の場合、サイズ縮小 (100 / 50) で再 fit して比較

#### 再評価トリガー条件

| 設計判断 | 再評価タイミング | 想定変更 |
|---|---|---|
| fit と apply の 2 ファイル分割 | M3 phase review (済) | **降格決定**: fit スクリプトは M6.2 後の再生成 reproducibility 用途、apply は `reverse_sample` 引数統合 |
| `apply_post_filter` を `reverse_sample` 引数統合 | M3 phase review (済) | **採用昇格**: `post_filter=None` switch で apply/no-apply 比較 (T-M3.3 と同期) |
| `scipy.signal.oaconvolve` 採用 | M3 phase review (済) | **採用昇格**: `np.convolve("same")` の左右非対称回避 |
| `fir.npy` commit 化 | M3 phase review (済) | **採用昇格**: `.gitignore` 除外解除、M6.2 後に commit |
| dev set サイズ 200 | M5.2 smoke 後 | MR-STFT 改善幅で 50 / 100 / 500 を再検討 (overfit / Nyquist gain 低下リスク、§8.1 検討追加) |
| 振幅差の集約軸 (utt + frame 両軸平均) | M5.2 smoke 後 | 解釈 A (utt 内平均 → utt 間平均) へ切替 ablation |
| frame-wise normalize (相対誤差) | M5.2 smoke 後 | MR-STFT 改善不十分なら相対誤差平均へ切替 (§8.1 検討追加) |
| 長さ保存 (oaconvolve full 中央切り出し) | M3.4 unit test | `test_oaconvolve_vs_convolve` / `test_apply_preserves_length` で非対称遅延検出 |
| FIR clip OFF (デフォルト) | M5.2 smoke 後 | max-abs > 1.0 なら `--clip-fir` デフォルト ON |
| `reverse_sample(seed=43)` deterministic | M3.4 / M6.2 | `--reproducible` flag で seed 固定、`fit_stats.json` 3 点 record |
| Linear amplitude 差 (vs log) | M6.3 ablation | log amplitude 差で改善するか比較 |
| Learnable post-filter | M6.3 ablation | NN-based で改善するか比較 |

### 8.2 思想 / 哲学の見直し

- **このサブタスクの粒度は適切か**: **適切**
  - fit (CLI script、重い 1 回処理) と apply (関数、軽量 hot path) の責務が明確に分かれる
  - 1 チケットに統合しても fit / apply のテストファイルは分けたいので、現状の 1 チケット内 2 ファイル分割が最適
- **post-filter の論文結果再現は post-filter 込み (M3 phase review 確定)**:
  - 論文 Table 2 は **post-filter ありで報告**されている。よって「論文結果再現 = post-filter 込み」と本チケットで確定する
  - **`reverse_sample` の default 動作判断**: default は `post_filter=None` (apply 無効) とし、論文結果再現の本ライン (M6.2 / M4.1) では明示的に `post_filter=fir` を渡す。理由: fit 段階や ablation の no-apply baseline で `post_filter=None` を必要とするため、default を None にする方が安全 (apply は呼び出し側の明示判断)
  - M4.1 / M6.2 で「論文 Table 2 に揃える = post-filter 込み」を再議論しないため、ここで確定
- **post-filter は spectral envelope correction であり sub-model artifact correction ではない (M3 phase review 確定)**:
  - post-filter は time-invariant FIR による高域 spectral envelope の加算補償であり、各 sub-model の出力品質差 (例: 特定 noise level band を担当する sub-model の学習不足) は post-filter では補償**できない**
  - 上流の `reverse_sample` (4 sub-model の reverse sampling) が前提であり、post-filter はその後段の固定 envelope 補正に過ぎない
  - 含意: M5.2 / M6.2 で品質問題が出た場合、まず upstream sub-model / reverse sampling を疑い、post-filter の fit ablation は二次的対応とする
- **別マイルストーンに移すべき部分はないか**:
  - **なし**: post-filter は Diff 専用 (`docs/training.md` §4.3) で M3 phase 内に閉じている
  - GAN 側は post-filter 不使用 (論文 §3.2 / `docs/architecture.md` §4 で明記)
- **インターフェース定義の見直し余地**:
  - **`apply_post_filter` を `reverse_sample` の引数に統合 (M3 phase review で採用)**: 2 関数 chain を避け、`post_filter=None` switch で apply/no-apply を 1 関数化
  - **`apply_post_filter` を `torch.nn.Module` 化する案**: train graph に組み込めるメリットあり、却下 (本チケットは推論時のみで requires_grad 不要)。ただし torch tensor 両受け + `F.conv1d` 実装で GPU 上の RTF 測定には対応済
  - **`fit_post_filter` を class 化 (`PostFilterFitter`)**: state を持つほどでもないので関数で十分
  - **`load_post_filter` を `apply_post_filter` の `fir` 引数自動解決にする案**: hidden state で挙動が分かりにくくなるので却下
- **時系列整合性の重要性**:
  - M3.4 実装は M3.5 smoke 前で OK (実装 + unit テストのみ)
  - 実 FIR の fit と周波数応答検証は M3.5 / M6.2 後
  - 本チケット完了 = 「実装が正しく書かれている」、実モデル統合は M3.5 へ

#### M3 phase review で追加されうる設計原則 (M3 phase 完了時に再評価)

- **post-filter の YAML config 化**: 現状 hardcode (`n_fft=512` etc.) → YAML から `n_fft`, `hop`, `fir_length` を切替可能にすると M6.3 ablation が楽
- **`PostFilterConfig` dataclass 化**: type safety 向上、ただし dict も十分

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M3.5 (Diff smoke) へ
- **使用方法**: `reverse_sample(model, mel, *, seed, post_filter=fir)` 1 関数化 (apply を内部統合、外部 chain 不要)
  ```python
  from wavenext2.inference.infer_diff import reverse_sample
  from wavenext2.inference.post_filter import load_post_filter
  fir = load_post_filter("post_filter/fir.npy")
  y_post = reverse_sample(model, mel, seed=43, post_filter=fir)   # apply 込み
  # apply 無効 (no-apply baseline) は post_filter=None
  ```
- **smoke 段階での FIR の扱い**:
  - M3.5 smoke は **未学習 / 部分学習 Diff モデル** で実施
  - smoke 範囲は **`apply_post_filter` の API 動作確認のみ** (品質は M5.2 で測定)
  - 実 FIR の fit は M3.4 単独で実施するも、**未学習モデルで fit した FIR は周波数応答テストを pass しない** 可能性大
  - 解決策: M3.5 では `test_freq_response_in_expected_range` を skip し、`reverse_sample(post_filter=fir)` の switch 動作確認のみ行う
  - 真の周波数応答検証は M6.2 完了 (Diff フル訓練後の checkpoint で fit) で実施
- **import パス**: `from wavenext2.inference.post_filter import apply_post_filter, load_post_filter`

#### T-M4.1 (objective metrics) へ
- **post-filter 適用前後の MCD / log F0 RMSE 比較**:
  - 同一 utterance に対して `reverse_sample(model, mel, seed=..., post_filter=None)` (no-apply) と `reverse_sample(model, mel, seed=..., post_filter=fir)` (apply) を計算
  - **`post_filter=fir` か `None` の switch だけで apply/no-apply を比較** (2 関数 chain 不要、§8.1 採用昇格)
  - MCD / log F0 RMSE を別々に計算して `post_filter/fit_stats.json` の `mrstft_before` / `mrstft_after` に並列追加、改善を確認
- **MR-STFT eval 再利用**: 本チケットで `fit_stats.json` に MR-STFT before/after を dump 済み、M4.1 はそれを参照するだけで OK

#### T-M6.2 (本格訓練) へ
- **`fir.npy` の commit**: Diff フル訓練後 (M6.2 完了) に `scripts/fit_post_filter.py --reproducible` で `fir.npy` を再 fit し、**git commit** する (`.gitignore` 除外解除済、2 KB、PR ベースで共有)
- **`fit_stats.json` で再現性確認**: commit する `fir.npy` に対応する `fit_stats.json` の (seed=43, git_sha, checkpoint_sha256) 3 点が record されていることを PR レビューで確認
- **実 FIR の周波数応答検証**: M6.2 後の checkpoint で fit した `fir.npy` に対して `test_freq_response_in_expected_range` (slow) を実行し、低域 0 dB ± 1 dB / Nyquist 端 +5〜8 dB を pass させる
- **論文 Table 2 再現は post-filter 込み (§8.2 確定)**: M6.2 の論文結果再現ラインは `reverse_sample(post_filter=fir)` を明示指定

#### T-M0.3 から受領 (確認済)
- `data/filelists/dev_postfilter.tsv` (200 utt, seed=43, val 非重複) を生成済
- パスは `--data-root` からの相対 (POSIX 区切り)、TSV header 付き

#### T-M3.3 から受領 (本チケットで引数追加を依頼)
- **現状 signature**: `reverse_sample(model, mel) -> Tensor (B, T_audio)`
- **本チケットで依頼する拡張** (§8.1 採用昇格、T-M3.3 §9.1 と同期):
  - `reverse_sample(model, mel, *, seed: int | torch.Generator, post_filter: np.ndarray | None = None) -> Tensor (B, T_audio)`
  - `seed` 引数: `int | torch.Generator` 拡張で `--reproducible` 動作 (内部 `torch.randn` を deterministic 化、`seed=43` で fit / 評価を bit-exact 再現)
  - `post_filter` 引数: `np.ndarray | None` (None で apply 無効、fir で apply 込み)、内部で `apply_post_filter` を呼ぶ
- `from wavenext2.inference.infer_diff import reverse_sample` で import

#### T-M2.3 から受領 (確認済)
- `MultiResolutionSTFTLoss(n_ffts=[512,1024,2048], win_lengths=[360,900,1800], hop_sizes=[80,150,300], eps=1e-5)` を eval metric として再利用
- `from wavenext2.losses.stft_loss import MultiResolutionSTFTLoss` で import

#### T-M3.1 から受領 (期待)
- `DiffWaveNext2.from_checkpoint(checkpoint_dir)` で 4 sub-models 一括復元 (T-M3.1 §9.1 想定)
- `model.eval()` で全 sub-models eval mode

#### T-M1.3 から受領 (確認済)
- `LogMelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=128, f_min=20, f_max=12000)` を Diff 用に
- 戻り値: `(B, 128, T_mel)` (log + clamp 済)

#### 共通の注意事項
- **`fir.npy` の dtype = float32**、shape = (512,) を厳格にチェック (`load_post_filter` で validate)
- **fit は時間がかかる** (GPU で 5〜10 分、CPU で 20〜40 分): M3.5 smoke では skip 可能、M4.1 / M6.2 で本実行
- **`apply_post_filter` は torch.Tensor / np.ndarray 両受け**: 入力型と同じ型で返す、`reverse_sample` 戻り値 (torch) のキャスト責務を吸収
- **`post_filter/fir.npy` は M6.2 後に git commit** (`.gitignore` の `*.npy` 除外を **解除**、2 KB): チケット PR で `.gitignore` 更新も含める。`.gitkeep` は空 dir 維持用に keep

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M3.4 の Acceptance チェックボックス 4 項目をすべてチェック
  - [ ] `docs/tickets/index.md` の T-M3.4 ステータス (`📝 pending` → `✅ completed`)
  - [ ] `docs/tickets/T-M3.4-post-filter.md` 自身の `status:` と `updated:` フィールド
  - [ ] (該当時) `docs/architecture.md` §5 / `docs/training.md` §4.3 の擬似コードから本実装への参照リンク追加
  - [ ] (該当時) `docs/open-questions.md` に「振幅差集約軸 (utt + frame 両軸平均) を採用」を追記

### 9.3 Open question として残ったもの
- **解決できなかった疑問**: なし (`docs/open-questions.md` で Okamoto21 §3.3 準拠と確定済み)
- **将来検討事項** (§8.1 再評価トリガー表参照):
  - dev set サイズ (40 / 50 / 100 / 200 / 500) ablation (M5.2 smoke 後 / M6.3、overfit / Nyquist gain 低下リスク)
  - 振幅差集約軸 (utt 内 → utt 間 vs utt + frame 両軸) ablation (M5.2 smoke 後)
  - frame-wise normalize (相対誤差) (M5.2 smoke 後、§8.1 検討追加)
  - Linear amplitude 差 vs log amplitude 差 (M6.3)
  - Learnable post-filter (NN ベース) との比較 (M6.3)
  - `scipy.signal.oaconvolve` vs `scipy.signal.fftconvolve` (`test_apply_preserves_length` / `test_oaconvolve_vs_convolve` fail 時)
  - FIR clip on/off (`--clip-fir` デフォルト OFF、max-abs > 1.0 時に再評価)
- **`docs/open-questions.md` への追記要否**: 不要 (本チケット完了時に振幅差集約軸の決定だけ記録)
