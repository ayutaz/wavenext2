# 再現実装計画

論文 "WaveNeXt 2" を再現するための計画と、参考実装の利用方針。

## 1. 依存スタック (推奨)

| 用途 | ライブラリ / ツール |
|---|---|
| パッケージ管理 / 実行 | [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`) |
| Python バージョン | 3.13 (論文指定なし、依存ライブラリ調査の結果 3.14 では pyworld 本家が cp314 wheel 未提供) |
| 深層学習 | PyTorch >= 2.10 (cp313 wheel 提供開始版) |
| 音声 I/O | torchaudio, soundfile |
| 信号処理 | librosa (mel/STFT は torchaudio で代替可) |
| ロガー | tensorboard / wandb |
| 設定管理 | hydra or yaml (任意) |
| データ | LibriTTS-R (huggingface or 公式 mirror) |

すべての Python スクリプト実行は `uv run python ...` 経由とする (`pip install` や global Python に依存しない)。

## 2. 参考にする既存実装

論文 footnote (Section 4.1) で論文著者が参照したものをそのまま利用する:

| 内容 | リポジトリ | 用途 |
|---|---|---|
| HiFi-GAN V1 / ParallelWaveGAN | https://github.com/kan-bayashi/ParallelWaveGAN | discriminator, STFT loss の参照 |
| WaveFit (PyTorch 非公式) | https://github.com/yukara-ikemiya/wavefit-pytorch | GAN-WaveNeXt 2 の訓練ループ・loss の参照 |
| FastDiff | https://github.com/Rongjiehuang/FastDiff | Diff 系のデータ流の参照 |
| Vocos | https://github.com/gemelo-ai/vocos | ConvNeXt block, STFT 分解の参照 |

これらをコピー流用するのではなく、**論文に合うように再構成** する。

## 3. 推奨ディレクトリ構成

```
wavenext2/
├── docs/                      # 本ドキュメント群
├── configs/
│   ├── gan_wavenext2.yaml
│   └── diff_wavenext2.yaml
├── src/wavenext2/
│   ├── data/
│   │   ├── dataset.py         # LibriTTS-R loader
│   │   └── mel.py             # log-mel-spec 計算
│   ├── models/
│   │   ├── convnext.py        # ConvNeXt block (GAN + Diff 共用 / Diff は additive bias 注入版)
│   │   ├── stft.py            # STFT module (Section 3.1)
│   │   ├── generator.py       # WaveNeXt-based generator (Conv1d → LN → ConvNeXt×8 → LN → Linear×2 → reshape → clip)
│   │   ├── sub_model.py       # Sub-model wrapper (Fig 2b、STFT module + generator)
│   │   ├── noise_embedding.py # Diff の sinusoidal embedding + FC×2 swish
│   │   ├── discriminator.py   # MSD ×3 (WaveFit-PT と同じ、MPD なし)
│   │   ├── gan_wavenext2.py   # GAN 全体モデル
│   │   └── diff_wavenext2.py  # Diffusion 全体モデル
│   ├── losses/
│   │   ├── stft_loss.py       # Multi-resolution STFT loss
│   │   ├── feature_matching.py
│   │   └── adversarial.py
│   ├── train/
│   │   ├── train_gan.py
│   │   └── train_diff.py
│   ├── inference/
│   │   ├── infer_gan.py
│   │   └── infer_diff.py
│   └── eval/
│       ├── compute_metrics.py # MCD, log F0 RMSE
│       └── run_utmos.py
├── scripts/
│   ├── prepare_libritts.sh
│   └── extract_mel.py
└── tests/
```

## 4. 実装フェーズ

### Phase 0: 環境準備
- [ ] uv インストール確認 (`uv --version`)、未導入なら winget / curl で導入
- [ ] `uv venv --python 3.13` で仮想環境作成
- [ ] `pyproject.toml` 作成 (`requires-python = ">=3.13,<3.14"`) + `uv add "torch>=2.10" torchaudio numpy scipy librosa soundfile pyyaml tensorboard matplotlib tqdm pymcd pyworld einops`
- [ ] `uv sync` で依存固定 (`uv.lock` を commit)
- [ ] `uv run python -c "import torch; print(torch.cuda.is_available())"` で CUDA 動作確認
- [ ] LibriTTS-R "train-clean-100" + "train-clean-360" + "test-clean" ダウンロード
- [ ] 24kHz リサンプリング確認
- [ ] mel 抽出 (n_mels=128, hop=300 と hop=256 の両方)

### Phase 1: コア部品
- [ ] `src/wavenext2/models/convnext.py`: ConvNeXt block 実装
  - dim=512, intermediate_dim=1536, kernel=7, LayerScale init=1e-6
  - GAN 版: 通常の ConvNeXt block
  - Diff 版: 入口に `Linear(512, dim)` の additive bias 注入を追加
- [ ] `src/wavenext2/models/stft.py`: 入力波形 → STFT-spec 変換 (Section 3.1 通り)
  - Hann window, center=True, normalized=False, onesided=True
  - n_fft/win_length/hop は mel-spec と同一値
  - 時間軸 truncation (mel-spec の T_mel に合わせる)
  - 実部 (全帯域 F bin) + 虚部 (DC/Nyquist 除外 F-2 bin) を channel concat
- [ ] `src/wavenext2/models/generator.py`: WaveNeXt-based generator (8 blocks)
  - `Conv1d(C_in, 512, k=7, p=3, bias=True)` → transpose → LN(512, eps=1e-6)
  - ConvNeXt × 8 (channels-last 形式で処理)
  - LN(512, eps=1e-6) (head 前の最終 LN、Vocos 慣例)
  - `Linear(512, n_fft+2, bias=True)` → `Linear(n_fft+2, hop, bias=False)`
  - reshape (B, T_mel, hop) → (B, T_mel*hop)
  - `torch.clip(-1, 1)`
  - 重み init: `trunc_normal_(std=0.02)`, bias は zero
- [ ] `src/wavenext2/models/noise_embedding.py`: Diff 用
  - sinusoidal embedding (dim=128, log(10000)/63 log-spaced)
  - FC1(128, 512) → SiLU → FC2(512, 512) → SiLU
- [ ] `src/wavenext2/models/sub_model.py`: 1 sub-model = STFT module + generator
- [ ] `src/wavenext2/data/mel.py`: Mel 抽出 (slaney scale + slaney norm + power=1 + 自然対数 + eps=1e-5)
- [ ] `src/wavenext2/data/dataset.py`: LibriTTS-R loader + sox `norm` 正規化 (train: U(-6,-1), val: -3 dB)

### Phase 2: GAN-WaveNeXt 2
- [ ] `src/wavenext2/models/discriminator.py`: MSD ×3 (WaveFit-PT と同一、MPD なし)
- [ ] `src/wavenext2/models/gan_wavenext2.py`: T 個の sub-model を直列に並べる
  - 初期入力 y_T = `torch.zeros_like(x_gt)`
- [ ] `src/wavenext2/losses/`: hinge GAN loss / FM L1 / MR-STFT (SC + Mag L1)
- [ ] `src/wavenext2/train/train_gan.py`:
  - Fixed-point iteration (ゼロ初期化、gain/denoising 制約なし)
  - Generator/Discriminator 交互更新
  - AdamW(lr=1e-4 G / 2e-4 D, β=[0.8, 0.99], wd=1e-3) + InverseLR
- [ ] スモークテスト: 1〜2 epoch 動作確認

### Phase 3: Diff-WaveNeXt 2
- [ ] `src/wavenext2/models/diff_wavenext2.py`: 4 sub-model を独立に扱える wrapper
- [ ] Noise schedule: `ᾱ = [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]` を固定で持つ
- [ ] Sub-model partition: point-specialized 1-to-1 (`docs/architecture.md` §5)
  - sub-model 1: band `[0.9929, 1.0]` (ᾱ ≈ 1e-4 周辺)
  - sub-model 2: band `[0.8246, 0.9929)` (ᾱ ≈ 2.8e-2 周辺)
  - sub-model 3: band `[0.4817, 0.8246)` (ᾱ ≈ 5.6e-1 周辺)
  - sub-model 4: band `[0, 0.4817)` (ᾱ ≈ 9.1e-1 周辺)
- [ ] `src/wavenext2/train/train_diff.py`: 各 sub-model を独立に MSE loss で訓練
  - Adam(lr=2e-4, β=[0.9, 0.98], wd=0)
- [ ] Post-filter (time-invariant spectral enhancement) 実装 (`docs/architecture.md` §5)
- [ ] スモークテスト

### Phase 4: 評価
- [ ] `src/wavenext2/inference/infer_gan.py`, `infer_diff.py`
- [ ] MCD, log F0 RMSE 算出
- [ ] UTMOS (https://github.com/sarulab-speech/UTMOS22), NISQA 連携
- [ ] RTF 測定 (GPU と CPU 1-core)

### Phase 5: 主観評価 (任意)
- 内部 MOS テスト orchestration (社内/個人での代替評価)

## 5. 設定ファイル雛形

`configs/gan_wavenext2.yaml`
```yaml
model:
  type: gan_wavenext2
  T: 4                       # sub-model 数 (= iterations)。論文推奨 T=4
  share_sub_models: false    # Table 1 から独立パラメータと判明
  initial_input: zeros       # 論文「initial input noise isn't required」→ ゼロベクトル
  sub_model:
    n_mels: 128
    sample_rate: 24000
    hop: 300
    n_fft: 2048
    win_length: 1200
    f_min: 20
    f_max: 12000
    mel:
      power: 1.0
      mel_scale: slaney
      norm: slaney
      log_eps: 1.0e-5        # 自然対数 + clamp(min=eps)
    stft_module:              # sub-model 入力側 STFT (Section 3.1)
      n_fft: 2048             # mel と同一
      win_length: 1200
      hop: 300
      window: hann
      center: true
      normalized: false
    convnext:
      n_blocks: 8
      embed_dim: 512
      intermediate_dim: 1536
      kernel_size: 7
      layer_scale_init: 1.0e-6
    head:                     # 元 WaveNeXt poster + wetdog 実装で確定
      linear1_out: 2050       # = n_fft + 2 (warm-start from Vocos 用)
      linear1_bias: true
      linear2_out: 300        # = hop_length
      linear2_bias: false
      output_clip: [-1.0, 1.0]
    init: trunc_normal_0.02   # 全 Conv1d/Linear の重み初期化
discriminator:
  type: msd_only              # WaveFit-PT 実装に従う (MPD は使わない)
  num_D: 3
  ndf: 16
  layers: 4
  downsampling_factor: 4
loss:
  adversarial: hinge          # WaveFit-PT 実装
  fm_loss: l1
  mrstft:
    fft_sizes: [512, 1024, 2048]
    win_lengths: [360, 900, 1800]
    hop_sizes: [80, 150, 300]
    eps: 1.0e-5
  weights:
    d_gan: 1.0
    d_fm: 10.0
    mrstft_sc: 2.5
    mrstft_mag: 2.5
    mel_mae: 0.0
train:
  batch_size: 16
  segment_length: 16384       # Vocos 流
  num_workers: 8
  max_steps: 2_000_000
  grad_clip_norm: 1.0
  optimizer:
    type: AdamW
    lr_g: 1.0e-4
    lr_d: 2.0e-4
    betas: [0.8, 0.99]
    weight_decay: 1.0e-3
  scheduler:
    type: InverseLR
    inv_gamma: 200000
    power: 0.5
    warmup: 0.999
  ema:
    enabled: false             # Vocos / WaveFit-PT 共に未使用
  validation:
    interval_steps: 10000
    metric: mrstft_total       # sc + mag の和を最小化
    num_utterances: 100
data:
  root: /path/to/LibriTTS-R
  subsets: [train-clean-100, train-clean-360]
  sample_rate: 24000
  audio_normalization:          # Vocos 流 sox `norm` peak 正規化
    method: sox_norm
    train_gain_db: [-6.0, -1.0] # uniform random
    val_gain_db: -3.0
```

`configs/diff_wavenext2.yaml`
```yaml
model:
  type: diff_wavenext2
  num_sub_models: 4
  noise_schedule: [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]   # ᾱ_1..ᾱ_4 (PDF 明示)
  partition: point_specialized                            # 1-to-1 mapping, see docs/architecture.md §5
  band_boundaries:                                         # √(1-ᾱ) 軸での train sampling band
    sub_model_1: [0.9929, 1.0]      # ᾱ ≈ 1e-4 周辺
    sub_model_2: [0.8246, 0.9929]   # ᾱ ≈ 2.8e-2 周辺
    sub_model_3: [0.4817, 0.8246]   # ᾱ ≈ 5.6e-1 周辺
    sub_model_4: [0.0, 0.4817]      # ᾱ ≈ 9.1e-1 周辺
  sub_model:
    n_mels: 128
    sample_rate: 24000
    hop: 256
    n_fft: 1024
    win_length: 1024
    f_min: 20
    f_max: 12000
    mel:
      power: 1.0
      mel_scale: slaney
      norm: slaney
      log_eps: 1.0e-5
    stft_module:
      n_fft: 1024              # mel と同一
      win_length: 1024
      hop: 256
      window: hann
      center: true
      normalized: false
    convnext:
      n_blocks: 8
      embed_dim: 512
      intermediate_dim: 1536
      kernel_size: 7
      layer_scale_init: 1.0e-6
    head:
      linear1_out: 1026         # = n_fft + 2
      linear1_bias: true
      linear2_out: 256          # = hop_length
      linear2_bias: false
      output_clip: [-1.0, 1.0]
    init: trunc_normal_0.02
    noise_level_embedding:       # FastDiff `module/FastDiff_model.py` で確定
      sinusoidal_dim: 128        # log(10000)/63 log-spaced, sin||cos
      fc1: [128, 512]
      fc2: [512, 512]
      activation: silu           # swish = SiLU
      injection: additive_bias   # NOT FiLM
      per_block_projection: true # 各 ConvNeXt block ごとに独立 Linear(512, 512)
  post_filter:
    enabled: true
    type: time_invariant_spectral_enhancement   # Okamoto21 §3
    n_fft: 512
    hop: 256
    window: hann
    fir_length: 512
    fit_dev_set_size: 200        # Okamoto21 は 40、LibriTTS-R では 100〜500
train:
  batch_size: 20
  segment_length: 25600          # FastDiff 準拠
  num_workers: 4
  max_steps_per_sub_model: 1_000_000
  grad_clip_norm: 1.0
  optimizer:
    type: Adam
    lr: 2.0e-4
    betas: [0.9, 0.98]
    weight_decay: 0.0
  ema:
    enabled: false               # FastDiff も未使用
  validation:
    interval_steps: 10000
    metric: mse_noise            # ε prediction の MSE
    num_utterances: 100
data:
  root: /path/to/LibriTTS-R
  subsets: [train-clean-100, train-clean-360]
  sample_rate: 24000
  audio_normalization:
    method: sox_norm
    train_gain_db: [-6.0, -1.0]
    val_gain_db: -3.0
```

## 6. 検証マイルストーン

訓練を本格化させる前に、以下のスモークが通っているか確認:

1. **形状チェック**: 各層の入出力 shape が想定通り
2. **過学習テスト**: 1 サンプルだけで 1000 step 訓練し、loss が 0 に近づき再構成できる
3. **STFT module の正当性**: 元波形 → STFT 入力に変換 → 戻したときに spectral content が保たれている
4. **GAN-WaveNeXt 2 の T=1 検証**: T=1 で WaveNeXt と挙動が一致するか (sanity check)
5. **Diff sub-model 単体動作**: 1 sub-model のみで 1 ステップ denoising できるか

## 7. 計算リソース見積もり (論文ベース)

| モデル | 訓練時間 (A100 単体) | 備考 |
|---|---|---|
| GAN-WaveNeXt 2 | ~410 時間 | WaveFit 同等 |
| Diff-WaveNeXt 2 | ~32 時間 | 4 sub-model 全体合計 |

個人 GPU で全モデル学習は非現実的なので、再現実装の Phase 1〜3 ではまず **アーキテクチャの正しさ** を 1 epoch スモークで担保し、本格訓練は GPU クラスタ確保後に進めることを推奨。

## 8. リスクと既知の不明点

詳細は `docs/open-questions.md` を参照。要点のみ:

| 項目 | 解決状況 | 対応 |
|---|---|---|
| `x_t` の拡散式 | ✅ **解決** | DDPM 標準形 `√ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (PDF Fig 1b 画像確認) |
| Generator output head | ✅ **解決** | `Linear(512, n_fft+2)` → `Linear(n_fft+2, hop, bias=False)` → reshape → `clip(-1, 1)` (元 WaveNeXt poster + `wetdog/wavenext_pytorch::heads.py`) |
| Generator input embedding | ✅ **解決** | `Conv1d(C_in, 512, k=7, p=3, bias=True)` → transpose → `LayerNorm(512, eps=1e-6)` (Vocos) |
| Generator 最終 activation | ✅ **解決** | `torch.clip(-1, 1)` (tanh ではない) |
| STFT module の n_fft/win/hop | ✅ **解決** | mel-spec と同一 (truncation で時間長を合わせるため) |
| Discriminator 詳細 | ✅ **解決** | MSD のみ × 3 (MPD なし)、WaveFit-PT |
| ConvNeXt 内部次元 | ✅ **解決** | embed=512, intermediate=1536, kernel=7, LayerScale init=1e-6 (Vocos) |
| Mel 正規化 | ✅ **解決** | `log(clamp(mel, min=1e-5))`, 自然対数, slaney scale/norm, power=1 |
| Audio 正規化 | ✅ **解決** | Vocos sox `norm` (train: U(-6,-1) dB, val: -3 dB) |
| Diff conditioning 注入 | ✅ **解決** | additive bias (FiLM ではない)、per-block 独立 `Linear(512, dim)`、block 入口で 1 回加算 (FastDiff) |
| Diff sinusoidal embedding | ✅ **解決** | 128 dim, `log(10000)/63` log-spaced, sin\|\|cos, FC×2 SiLU (FastDiff) |
| Optimizer / LR / scheduler | ✅ **解決** | GAN: AdamW + InverseLR, Diff: Adam |
| BDDM noise predictor の不在 | ✅ **解決** | 論文の 4 値固定スケジュールを直接使用 |
| time-invariant post-filter | ✅ **解決** | Okamoto21 §3.3: 振幅差平均 → 長さ 512 FIR (n_fft=512, hop=256) |
| 4 sub-models の noise range 境界 | ✅ **解決** | point-specialized 1-to-1 (band 境界は隣接 schedule 点の中点) |
| Diff reverse step 式 | ✅ **解決** | DDPM 標準形 (`docs/training.md` §4.2) |
| Fixed-point iteration の初期化 (GAN) | ✅ **解決** | `torch.zeros_like(x_gt)` (論文「initial input noise isn't required」) |
| EMA | ✅ **解決** | 不使用 (Vocos / WaveFit-PT 共に未使用) |
| Validation 基準 | ✅ **解決** | MR-STFT (sc + mag) 合計の最小、10k step ごと |

## 9. 次のアクション

1. `docs/` レビューを受けた上で実装開始 (Phase 0)
2. PyTorch プロジェクト雛形作成 (`src/`, `configs/`, `tests/`)
3. ConvNeXt block と STFT module のユニットテスト
4. データローダ + mel 抽出スクリプト
5. GAN-WaveNeXt 2 から実装着手 (Diff 側より部品が多く、検証パスが豊富なため)
