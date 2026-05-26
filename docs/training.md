# 訓練・推論手順

論文 Section 3.2 / 3.3、Section 4 を再現実装向けに整理。

## 1. データ前処理

### 1.1 共通
- データセット: LibriTTS-R "train-clean-100" + "train-clean-360"
- Sampling rate: 24,000 Hz
- 入力特徴量: 128-dim log-mel-spectrogram

### 1.2 Mel-spectrogram 抽出パラメータ
| 項目 | GAN-WaveNeXt 2 | Diff-WaveNeXt 2 | 確定元 |
|---|---|---|---|
| n_mels | 128 | 128 | WaveFit-PT |
| hop_length | **300** | **256** | 論文 |
| win_length | **1200** | **1024** | WaveFit-PT / Vocos |
| n_fft | **2048** | **1024** | WaveFit-PT / Vocos |
| f_min | **20** | **20** | WaveFit-PT |
| f_max | **12000** | **12000** | WaveFit-PT |
| power | **1.0** (magnitude) | **1.0** (magnitude) | Vocos |
| mel_scale | **slaney** | **slaney** | HiFi-GAN/BigVGAN 慣例 |
| norm | **slaney** | **slaney** | HiFi-GAN/BigVGAN 慣例 |
| padding | center | center | Vocos |
| log type | **natural log** | **natural log** | Vocos `safe_log` |
| log eps | **1e-5** | **1e-5** | HiFi-GAN 慣例 (Vocos の 1e-7 より安定) |

```python
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000, n_fft=n_fft, hop_length=hop,
    win_length=win_length, n_mels=128, f_min=20.0, f_max=12000.0,
    power=1.0, mel_scale="slaney", norm="slaney", center=True,
)(audio)
log_mel = torch.log(torch.clamp(mel, min=1e-5))
```

### 1.2.5 Audio 正規化 (Vocos 方式、確定)
```python
# 訓練時 (random gain)
gain_db = np.random.uniform(-1, -6)
audio, _ = torchaudio.sox_effects.apply_effects_tensor(
    audio, sr=24000, effects=[["norm", f"{gain_db:.2f}"]]
)
# 検証/推論時
gain_db = -3.0
audio, _ = torchaudio.sox_effects.apply_effects_tensor(
    audio, sr=24000, effects=[["norm", "-3.0"]]
)
```
- 出力: `float32`, `[-1, 1]`, mono (stereo の場合は `mean(dim=0)`)
- LUFS 正規化や peak clip は不要 (sox `norm` が peak-based)

### 1.3 切り出し
- 訓練時セグメント長: **16,384〜25,600 サンプル** (= 0.68〜1.07 秒 @24kHz)
  - GAN 側 (hop=300): mel-frame 54〜85
  - Diff 側 (hop=256): mel-frame 64〜100
- Vocos の `num_samples = 16384` または FastDiff の `25600` が参考値

## 2. GAN-WaveNeXt 2 の訓練 (Section 3.2)

### 2.1 訓練ループ
T iter (= sub-model 数) の fixed-point iteration を実行。

```python
# pseudo
for batch in dataloader:
    mel, x_gt = batch                 # mel-spec (B, 128, T_mel), ground-truth waveform (B, T_audio)
    y_T = torch.zeros_like(x_gt)      # 初期入力: ゼロベクトル (論文「initial input noise isn't required」)
                                       # → WaveFit の spectral-envelope shaped noise は使わない
    y_t = y_T
    for t in range(T, 0, -1):
        # sub-model はそれぞれ独自パラメータ
        n_t = sub_model[t](mel, y_t)
        y_t = y_t - n_t               # residual denoising
    y_0 = y_t

    # loss はすべて WaveFit と同一
    loss_stft = multi_resolution_stft_loss(y_0, x_gt)
    loss_adv  = adversarial_loss(D, y_0)
    loss_fm   = feature_matching_loss(D, y_0, x_gt)
    loss_G    = loss_stft + λ_adv * loss_adv + λ_fm * loss_fm

    # Discriminator
    loss_D = D_loss(D, y_0.detach(), x_gt)
```

### 2.2 重要な変更点 (WaveFit との差分)
- **denoising 制約は loss から除外**
  - WaveFit は loss 内に noise を波形に対して減衰させる制約があるが、本論文ではこれは不要と確認済み
- **gain adjustment module を削除**
  - STFT loss が代替として機能するため
- **初期入力ノイズの分布は単純化**
  - WaveFit のような spectral envelope ベースの noise initialization は不要

### 2.3 Discriminator (WaveFit-PyTorch 準拠で確定)
- **MSD のみ × 3 sub-discriminators** (MelGAN tradition)。MPD は **使用しない**
- 各 sub-discriminator: 7 layers (NLayerDiscriminator)
  - 入力 Conv: kernel=15, ch 1→16
  - downsampling layer × 4: kernel=41, stride=4, groups=`nf_prev//4`, ch 倍々 (上限 1024)
  - 終端 conv 2 つ + weight_norm
  - 全 LeakyReLU(0.2)
- 隣接 sub-discriminator 間: AvgPool1d(kernel=4, stride=2) で audio を downsample
- **Adversarial**: **hinge GAN loss** (LSGAN ではない)
- **Feature matching**: 中間特徴 L1 distance
- 詳細は `open-questions.md §確定事項 / Discriminator 仕様` 参照

### 2.4 訓練設定 (WaveFit-PT で確定)
- 訓練時間: 約 410 時間 / A100 (HiFi-GAN, WaveFit と同等) [PDF Table 2]
- **Optimizer (G)**: AdamW, lr=**1e-4**, betas=**[0.8, 0.99]**, weight_decay=**1e-3**
- **Optimizer (D)**: AdamW, lr=**2e-4** (G の 2 倍), betas=**[0.8, 0.99]**, weight_decay=**1e-3**
- **LR Scheduler**: InverseLR (`inv_gamma=200000`, `power=0.5`, `warmup=0.999`)
- **Gradient clip**: `max_grad_norm = 1.0`
- **Loss 重み**:
  - D-GAN: 1.0
  - D-Feature: 10.0
  - MRSTFT-SC: 2.5
  - MRSTFT-Mag: 2.5
  - Mel-MAE: 0.0
- **MR-STFT loss**:
  - n_ffts = [512, 1024, 2048]
  - win_sizes = [360, 900, 1800]
  - hop_sizes = [80, 150, 300]
  - sub-loss: Spectral Convergence (L2 ratio) + Magnitude L1 (log-amplitude)

## 3. Diff-WaveNeXt 2 の訓練 (Section 3.3)

### 3.1 各 sub-model は独立に訓練
- 4 step に分割された noise level range の **それぞれに対して 1 つの sub-model** を訓練
- 各 sub-model はそのレンジの noise level でしか訓練しないため、その狭い範囲に特化できる (高精度化)

### 3.2 訓練ループ (sub-model k に対して、k = 1..4) — point-specialized partition

各 sub-model は **1 つの schedule 点に対応した band** で訓練される (詳細は `docs/architecture.md` §5)。

```python
# Point-specialized band partition (1-to-1 with 4-step schedule)
SCHEDULE_ABAR = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])   # ᾱ_1..ᾱ_4
SCHEDULE_C    = torch.sqrt(1.0 - SCHEDULE_ABAR)                    # √(1-ᾱ): [0.99995, 0.9858, 0.6633, 0.3]

def get_band(k, K=4):
    """sub-model k (1-indexed) の訓練 band [L, U]"""
    # 隣接 schedule 点との中点で band を区切る
    c = SCHEDULE_C  # sorted descending: 0.99995, 0.9858, 0.6633, 0.3
    if k == 1:
        L = (c[0].item() + c[1].item()) / 2     # 0.9929
        U = 1.0
    elif k == K:
        L = 0.0
        U = (c[K-2].item() + c[K-1].item()) / 2 # 0.4817
    else:
        L = (c[k-1].item() + c[k].item()) / 2
        U = (c[k-2].item() + c[k-1].item()) / 2
    return L, U

# sub-model k in {1, 2, 3, 4}
k = ...                          # 訓練対象の sub-model index
L_k, U_k = get_band(k, K=4)
sub_model_k = sub_models[k - 1]  # 0-indexed in list

for batch in dataloader:
    mel, x_0 = batch  # (B, 128, T_mel), (B, 1, T_audio)

    # 1. 担当バンドから連続的に noise level を sampling
    c = torch.empty(B).uniform_(L_k, U_k)             # √(1-ᾱ)
    abar = 1.0 - c ** 2                                # ᾱ
    sqrt_abar = torch.sqrt(abar)
    sqrt_one_minus_abar = c

    # 2. 拡散式 (DDPM 標準形、PDF Fig 1b で確認):
    #    x_t = √ᾱ * x_0 + √(1-ᾱ) * ε
    eps = torch.randn_like(x_0)
    x_t = sqrt_abar.view(-1, 1, 1) * x_0 \
        + sqrt_one_minus_abar.view(-1, 1, 1) * eps

    # 3. sub_model k は連続 noise level c = √(1-ᾱ) を条件として受け取る
    #    (FastDiff 流の sinusoidal embedding + FC × 2 swish + per-block additive bias)
    eps_pred = sub_model_k(mel, x_t, sqrt_one_minus_abar)

    loss = mse_loss(eps_pred, eps)  # Fig 1b
    loss.backward()
    ...
```

**重要な点**:
- noise level は band 内で **一様分布から sampling** → 推論時の正確な schedule 点に対応できる
- conditioning 信号は `c = √(1-ᾱ)` (連続値)
- 4 個の sub-model はパラメータを共有しない (Table 1 のパラメータ数から判明)
- **推論時は band を経由せず schedule 点を直接使うため、1-to-1 マッピングで全 4 sub-model が deploy される**

### 3.3 Noise schedule (4 ステップ、固定)
```
a = [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]
```
- BDDM の noise schedule predictor で予測された値が論文に明示されている
- 再現実装では **この 4 値を固定として使えばよい** (predictor 自体の再現は不要)

### 3.4 訓練設定 (FastDiff 準拠で確定)
- 訓練時間: **32 時間** (FastDiff の 96 時間より大幅短縮) [PDF Table 2]
- A100 シングル GPU
- 4 sub-models をそれぞれ独立訓練 → 4 × 8時間/sub-model
- **Optimizer**: Adam, lr=**2e-4**, betas=**[0.9, 0.98]**, weight_decay=**0**
- **Gradient clip**: `clip_grad_norm = 1`
- **Segment length**: **25,600 samples** (FastDiff の値、約 1.07 秒)
- **Batch size**: 20 (FastDiff デフォルト)
- **Total steps per sub-model**: 1M 程度 (FastDiff の `max_updates = 1,000,000`)

### 3.5 Noise level conditioning (FastDiff `module/FastDiff_model.py` で確定)

**注入方式は additive bias** (FiLM ではない)。詳細は `docs/architecture.md` §5.4 参照。

```python
# 1. Sinusoidal embedding (sub-model forward あたり 1 回)
def sinusoidal_embedding(c, dim=128):
    half = dim // 2
    freq = np.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half) * -freq).to(c.device)
    e = c.unsqueeze(-1) * freq                              # (B, 64)
    return torch.cat([torch.sin(e), torch.cos(e)], dim=-1)  # (B, 128)

# 2. Shared embedding head (sub-model ごとに 1 セット)
e = F.silu(self.fc_t1(sinusoidal_embedding(c, 128)))       # Linear(128, 512)
e = F.silu(self.fc_t2(e))                                   # Linear(512, 512), → (B, 512)

# 3. Per-block additive bias (各 ConvNeXt block の入口で 1 回だけ加算)
for block in self.convnext_blocks:
    bias = block.fc_t(e).unsqueeze(-1)   # Linear(512, 512) per block, (B, 512, 1)
    x = x + bias                          # additive, broadcast over time
    x = block(x)                          # 通常の ConvNeXt forward
```

- Sinusoidal は 128 dim (`log(10000)/63` log-spaced、`sin || cos`)
- 各 sub-model は **独自の** sinusoidal head + 8 個の per-block `fc_t` projection を持つ (共有なし)
- bias は **block の入口で 1 回だけ** 加算 (内部の Linear ごとには再加算しない)

## 4. 推論手順

### 4.1 GAN-WaveNeXt 2
```python
B, T_mel = mel.shape[0], mel.shape[2]
T_audio = T_mel * hop_length                 # GAN: hop=300
y_t = torch.zeros(B, T_audio, device=mel.device)  # 初期入力: ゼロ (論文準拠)
for t in range(T, 0, -1):
    n_t = sub_model[t](mel, y_t)
    y_t = y_t - n_t
y_0 = y_t                                    # 最終出力 (clip(-1, 1) は sub-model 内部で適用済み)
```
- 反復数 T は 2〜5 を実験で比較 (Table 1)
- 推奨は **T = 4** (品質と速度のバランス、論文の MOS 結果より)

### 4.2 Diff-WaveNeXt 2 (BDDM reverse step、DDPM 標準形、point-specialized dispatch)

論文の 4 値は **ᾱ_t (cumulative)** として直接解釈 (PDF §3.3 本文・[Okamoto21] 規約に従う)。β_t は隣接 ᾱ_t から逆算。reverse step は BDDM の `bddm/sampler/sampler.py` DDPM ブランチに基づく。

**Dispatch は 1-to-1**: 推論順 t=1→4 で sub-model 1→4 を順に呼ぶ。

```python
# 論文の 4-step schedule: ᾱ_t = cumulative noise level (denoising 順 t=1..4)
abar = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])   # ᾱ_1..ᾱ_4
K = len(abar)  # = 4

# β を逆算:  ᾱ_t = ᾱ_{t-1} * (1 - β_t)
#           ⇒ β_t = 1 - ᾱ_t / ᾱ_{t-1}    (t >= 2)
#           ⇒ β_1 = 1 - ᾱ_1                (慣例: ᾱ_0 = 1)
beta = torch.zeros(K)
beta[0] = 1.0 - abar[0]
for t in range(1, K):
    beta[t] = 1.0 - abar[t] / abar[t-1]

sqrt_abar = torch.sqrt(abar)
sqrt_one_minus_abar = torch.sqrt(1.0 - abar)

# 後方分散 σ_t (BDDM/DDPM 形式、t=1 は σ=0)
# 注意: 推論順 t=1..4 (t=1 が純ノイズ、t=4 が目標) で t=4 が σ=0 (最後のステップ)
sigma = torch.zeros(K)
for t in range(K - 1):
    sigma[t] = torch.sqrt(beta[t+1] * (1.0 - abar[t]) / (1.0 - abar[t+1])) if abar[t+1] < 1 else torch.tensor(0.0)
sigma[K-1] = 0.0  # 最後のステップ (t=4) は decoder のように決定論的

# 推論ループ (denoising direction: t=1 (純ノイズ) → t=4 (ほぼクリーン))
x = torch.randn(B, T_audio)
for t in range(K):
    c_t = sqrt_one_minus_abar[t]                          # noise level conditioning
    k = t + 1                                              # sub-model 1-indexed, point-specialized 1-to-1

    eps_pred = sub_model[k](mel, x, c_t.expand(B))

    # DDPM 標準 reverse step (next ᾱ to denoise toward)
    # x_{t+1} を導出 (推論順での次ステップ)
    if t < K - 1:
        # 通常の DDPM step using β_{t+1}
        beta_next = beta[t+1]
        x = (1.0 / torch.sqrt(1.0 - beta_next)) * (
            x - (beta_next / sqrt_one_minus_abar[t]) * eps_pred
        )
        x = x + sigma[t] * torch.randn_like(x)
    else:
        # 最終ステップ: ε で完全に denoise
        x = (x - sqrt_one_minus_abar[t] * eps_pred) / sqrt_abar[t]

y_0 = post_filter(x)
```

**実装注意**:
- 上記は DDPM 標準形のシンプルな解釈。BDDM 公式実装 (`bddm/sampler/sampler.py`) の DDPM ブランチでは index 規約が異なる場合があるため、実装時は両方を試して品質を確認すること
- `t` index と `√(1-ᾱ)` 値の整合性チェックを必ず実施 (high noise → low noise の denoising 方向)

#### 数式 (LaTeX)
$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, c, t)\right) + \sigma_t z,\quad z\sim\mathcal{N}(0,I)
$$
$$
\sigma_t^2 = \beta_t \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t},\quad \sigma_0 = 0
$$

#### 推論 schedule の sub-model 割り当て (point-specialized 1-to-1)
論文の 4 値での `√(1-ᾱ_t)` と sub-model 割り当て (denoising 方向 t=1..4):

| t (denoising 順) | `ᾱ_t` | `√(1-ᾱ_t)` | sub-model k |
|---|---|---|---|
| 1 (最初、最高ノイズ) | 1.0e-4 | 0.99995 | 1 |
| 2 | 2.8e-2 | 0.9858 | 2 |
| 3 | 5.6e-1 | 0.6633 | 3 |
| 4 (最後、最低ノイズ) | 9.1e-1 | 0.3 | 4 |

→ 推論順では sub-model **1 → 2 → 3 → 4** を順に呼ぶ。**全 4 sub-model が deploy される** (Table 1 のパラメータ数 57.68M = 14.42M × 4 と整合)。

#### 4 ステップ完了後の post-filter
- §4.3 で実装した time-invariant spectral enhancement FIR を畳み込み (`y_post = conv(y_synth, fir)`)
- FIR は事前に dev set 上で 1 度だけ fit する

### 4.3 Time-invariant spectral enhancement post-filter ([Okamoto21] §3.3 で確定)

Okamoto+ ICASSP'21 preprint (https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2021_okamoto.pdf) から取得した完全な式。

#### Fit (dev set 上で 1 度だけ実施)
```python
# n_fft=512, hop=256 を使用 (Okamoto21 の実験設定)
diff_acc = np.zeros(257)
n_frames = 0
for orig, synth in dev_pairs:                # 数十〜数百発話
    mag_orig  = np.abs(stft(orig,  n_fft=512, hop=256, window="hann"))
    mag_synth = np.abs(stft(synth, n_fft=512, hop=256, window="hann"))
    diff_acc  += (mag_orig - mag_synth).sum(axis=1)
    n_frames  += mag_orig.shape[1]

mean_diff_mag = diff_acc / n_frames           # shape (257,)
fir = np.fft.irfft(mean_diff_mag, n=512)      # length 512 FIR
fir = np.fft.fftshift(fir)                    # 線形位相用に中央化
```

#### 推論時
```python
y_post = np.convolve(y_synth, fir, mode="same")
```

#### 特徴
- 全発話で **同じフィルタ** を使用 → time-invariant
- α やゲイン等の hyperparameter は不要
- スペクトル比型 (target/synth) ではなく、**振幅差の加算的補償**
- 周波数応答: 〜2 kHz 以下で 0 dB、Nyquist 端で +5〜8 dB (高域シェルフ)

## 5. 評価プロトコル (Section 4)

### 5.1 主観評価
- MOS (5-point scale)
- 20 名のネイティブ英語話者 (有償)
- ヘッドフォン使用、静音環境
- 各被験者が 120 サンプル評価 (20 utterances × 6 models)
- LibriTTS-R "test-clean-100" から抽出

### 5.2 客観評価
- **UTMOS** (UTokyo-SaruLab system, Interspeech 2022): 自動 MOS 推定
- **NISQA** (Mittag et al., Interspeech 2021): CNN-self-attention 系の品質予測
- **MCD** (Mel-Cepstral Distortion): スペクトル類似度
- **log F0 RMSE**: 韻律精度

### 5.3 速度評価
- **RTF (GPU)**: NVIDIA A100
- **RTF (CPU)**: AMD EPYC 7542、**1 core 限定**
- 4,824 utterances (LibriTTS-R test-clean-100) で平均

### 5.4 比較対象
- GAN: HiFi-GAN V1, WaveFit (2/3/4/5 iterations)
- Diffusion: FastDiff (with/without sub-modeling)
- 元 WaveNeXt (1 iteration、参考値)

## 6. Ablation (論文中の所見)

- Diff-WaveNeXt 2 で sub-modeling を **使わない** 場合、性能が著しく劣化することを確認
- 元 FastDiff アーキテクチャに sub-modeling だけ導入したものより、WaveNeXt 2 統一アーキテクチャの方が良い

## 6. Validation / Checkpoint 選択 (WaveFit-PT 慣例で確定)

| 項目 | 値 | 根拠 |
|---|---|---|
| validation 頻度 | 10,000 steps ごと | WaveFit-PT `configs/trainer/default.yaml` `n_step_test=10000` |
| best ckpt 基準 | MR-STFT (sc + mag) 合計の最小 | WaveFit-PT `metrics_for_best_ckpt` |
| EMA | **不使用** | Vocos / WaveFit-PT 共に使っていない |
| validation utterances 数 | 100 (慣例) | 軽量化のため |
| UTMOS/PESQ/MCD | 学習中は計算しない | 重いので最終評価のみ |

## 7. 解決済みの実装上の注意

| 過去の曖昧さ | 確定状況 | 確定根拠 |
|---|---|---|
| `x_t` 係数式 | ✅ `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (DDPM 標準形) | PDF Fig 1b 画像目視 |
| fixed-point の初期化 | ✅ `torch.zeros_like(x_gt)` (ゼロベクトル) | 論文「initial input noise isn't required」 |
| STFT loss の解像度 | ✅ n_ffts=[512,1024,2048], wins=[360,900,1800], hops=[80,150,300] | WaveFit-PT `src/loss/mrstft.py` |
| Generator output activation | ✅ `torch.clip(-1, 1)` (tanh ではない) | 元 WaveNeXt 実装 (`wetdog/wavenext_pytorch::heads.py`) |
| EMA | ✅ 不使用 | Vocos / WaveFit-PT |
| Audio 正規化 | ✅ Vocos sox `norm` (train: U(-6,-1), val: -3 dB) | Vocos `vocos/dataset.py` |
| Mel 正規化 | ✅ `log(clamp(mel, min=1e-5))`, slaney scale | Vocos `safe_log` + HiFi-GAN 慣例 |
| Diff conditioning 注入 | ✅ additive bias (per-block independent `Linear(512, dim)`) | FastDiff `module/FastDiff_model.py` |
| K=4 sub-model dispatch | ✅ point-specialized 1-to-1 | 論文 §3.3 + Table 1 (4 × 14.42M) |

詳細な未確定事項 (残るは予備実験項目のみ) は `docs/open-questions.md` を参照。
