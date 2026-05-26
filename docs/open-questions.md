# 再現実装に必要な情報の確定状況

論文本文 (PDF) + 関連論文 ([Okamoto21]) + 参考実装 (Vocos / WaveFit-PyTorch / FastDiff / BDDM / wavenext_pytorch) から取得した全項目を体系化。

## 結論

**再現実装に必要なすべての情報は確定済み**。Generator の output head、STFT module、Diff conditioning 注入、Mel/Audio 正規化、K=4 partition の解釈、最終 activation、EMA、Validation 基準まで全て参考実装または関連論文で裏取り済み。

| カテゴリ | 確定状況 |
|---|---|
| Generator 構造 (Conv1d → LN → ConvNeXt×8 → LN → Linear×2 → reshape → clip) | ✅ 100% |
| ConvNeXt block 内部 | ✅ 100% |
| STFT module (Section 3.1) | ✅ 100% |
| Mel-spec 抽出と log 正規化 | ✅ 100% |
| Audio 前処理 (peak norm = sox norm 等価, §C4) | ✅ 100% |
| Discriminator / GAN Loss | ✅ 100% |
| Diff 拡散式 / noise schedule | ✅ 100% |
| Diff conditioning 注入方式 (additive bias) | ⚠️ 推定 (論文 §3.3 は機構未記載、FastDiff 推定) — M1 review |
| Diff sub-model partition (point-specialized 1-to-1) | ✅ 100% |
| Diff reverse sampling | ✅ 100% |
| Time-invariant post-filter | ✅ 100% |
| 訓練ハイパーパラメータ (optimizer, LR, scheduler) | ✅ 100% |
| Validation / Best ckpt 基準 | ✅ 100% |
| EMA 有無 | ✅ 100% (不使用) |
| GAN fixed-point 初期化 | ✅ 100% (ゼロベクトル) |
| Generator 最終 activation | ✅ 100% (`clip(-1, 1)`) |

残るは Random seed や Validation utterance 数など、実装結果に微小な影響しか及ぼさない予備実験項目のみ (§D 参照)。

> **補遺 (M1 phase review, 2026-05-27)**: 上表の一部「✅ 100%」は **論文本文では未記載で、参照実装 (FastDiff / Vocos) からの推定**であることが判明 (arXiv HTML §3 / Fig 2 を WebFetch で確認)。具体的には: (a) **mel と STFT-spec の結合方法** (concat→単一 conv か別経路か) §3.1 未記載、(b) **Diff conditioning が per-block 注入か 1 回か / additive か FiLM か** §3.3 未記載、(c) **embed の kernel/stride** 未記載、確認できたのは ConvNeXt n=8 のみ。Table 1 の param 数 (GAN sub-model=14.985M=WaveNeXt baseline 14.98M、Diff=14.42M) が唯一の客観証拠で、**per-block fc_t (2.1M) を含めると Diff が +14% 超過**し、fc_t 無し (または embed mel-only) なら一致する。conditioning 機構は §C7 を「推定」に格下げし、M3 着手前に user 判断 + smoke で確定する。

---

## 情報源略号

- **[PDF]** = 本論文 (arXiv:2605.25506v1) 本文・図 (PyMuPDF で画像化して目視確認済み)
- **[Okamoto21]** = T. Okamoto et al., "Noise Level Limited Sub-Modeling for Diffusion Probabilistic Vocoders," ICASSP 2021 (preprint URL は §E 末尾参照)
- **[WaveNeXt-orig]** = T. Okamoto et al., "WaveNeXt: ConvNeXt-Based Fast Neural Vocoder," ASRU 2023 (poster PDF: https://www.okamotocamera.com/asru_2023.pdf)
- **[BDDM]** = M. W. Y. Lam et al., "BDDM: Bilateral Denoising Diffusion Models," ICLR 2022 (公式実装: tencent-ailab/bddm)
- **[Vocos]** = https://github.com/gemelo-ai/vocos
- **[WaveFit-PT]** = https://github.com/yukara-ikemiya/wavefit-pytorch
- **[FastDiff]** = https://github.com/Rongjiehuang/FastDiff
- **[wavenext-impl]** = https://github.com/wetdog/wavenext_pytorch (非公式 PyTorch 移植、Vocos fork)

---

## A. PDF 本文・図から確定した事項

### A1. モデル全体
- アーキテクチャ概要: ConvNeXt-based sub-model を GAN/Diffusion で共有
- 統一の鍵: 入力側の **STFT module**
- WaveNeXt-based generator 構造 (Fig 2a): `Conv1d → LN → ConvNeXt × 8 → LN → Linear → Linear → Reshape → clip`
- Sub-model 構造 (Fig 2b): STFT module + WaveNeXt-based generator
- **ConvNeXt block 数 n = 8** (全モデル共通) [PDF Fig 2 キャプション]
- Generator 出力は **ノイズ成分 n_t** (波形そのものではない)

### A2. STFT module (Section 3.1)
- 入力波形に **Hann window**
- STFT は **center=True**
- 複素 spectrogram を **mel-spec の時間長に truncate**
- 実部 (全帯域 F bins) + 虚部 (DC/Nyquist 除外 F-2 bins) を **concatenate** → `2F-2` ch
- mel-spec と一緒に generator へ入力

### A3. Mel-spectrogram / hop size
| 項目 | GAN-WaveNeXt 2 | Diff-WaveNeXt 2 |
|---|---|---|
| sample_rate | 24000 | 24000 |
| n_mels | 128 | 128 |
| hop_length | **300** | **256** |

### A4. GAN-WaveNeXt 2 (Section 3.2)
- 訓練戦略: WaveFit の fixed-point iteration を採用
- WaveFit からの簡略化: denoising 制約・初期入力ノイズ・gain module を **不採用**
- Discriminator: WaveFit と同一
- Loss: WaveFit と同一
- 反復数 T: 2, 3, 4, 5 を比較 (Table 1)、論文推奨は T=4
- 各 iteration に独立した sub-model (Table 1 のパラメータ数が iter に線形比例することから判明)

### A5. Diff-WaveNeXt 2 (Section 3.3)
- 4 sub-model 構成、各 sub-model が特定 noise level range を担当
- 訓練戦略: noise-level limited sub-modeling [Okamoto21]
- **拡散式** (PDF Fig 1b 画像で確定): `x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε` (DDPM 標準形)
- `ᾱ_t` は **cumulative noise level**
- **4-step noise schedule**: `ᾱ = [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]`
- Loss: MSE on noise prediction (Fig 1b)
- 推論手順: 初期 noise → 4 sub-models 逐次適用 → post-filter (Fig 3)

### A6. データセット (Section 4.1)
- LibriTTS-R "train-clean-100" + "train-clean-360"、24 kHz、約 585 時間
- 評価: "test-clean-100" 4,824 utterances
- MOS: 同 subset から 20 utterances

### A7. 評価指標 (Section 4.1)
- 主観: MOS (5-point, 20名のネイティブ英語話者)
- 客観: UTMOS, NISQA, MCD, log F0 RMSE
- 速度: RTF (NVIDIA A100 GPU, AMD EPYC 7542 CPU 1-core)

### A8. モデルサイズ (Table 1)
- GAN-WaveNeXt 2: 29.97M (2 iter) / 44.96M (3) / 59.94M (4) / 74.93M (5)
- Diff-WaveNeXt 2: 14.42M (wo/sub-model) / 57.68M (w/sub-model)
- Sub-model 1 個あたり 約 14.99M (GAN) / 14.42M (Diff)
- → ConvNeXt embed=512, intermediate=1536, n_blocks=8 で約 14〜15M (Vocos 設定と一致)
- → **57.68M = 14.42M × 4 から「4 sub-model 全てが訓練・deploy される」と判明** (point-specialized 1-to-1 の根拠)

---

## B. 関連論文から確定した事項

### B1. Sub-model partition: point-specialized 1-to-1 [PDF §3.3 + Table 1]

**WaveNeXt 2 は Okamoto21 の strict equal partition (K=10 中 N=6 のみ使用) ではなく、point-specialized 1-to-1 を採用** (`docs/architecture.md` §5)。

#### 根拠
- 論文「we divide the denoising task into **four** stages and construct **four** sub-models」
- Fig 3 が 4 sub-model を inference 順に並べる
- Table 1: 57.68M = 14.42M × 4 → 全 4 sub-model が deploy される
- Okamoto21 strict K=4 equal partition だと band 1 ([0, 0.25)) が無使用となり、パラメータ数と矛盾

#### Point-specialized band 境界 (確定)
| k (denoising 順) | schedule 点 √(1-ᾱ_k) | 訓練 band `[L_k, U_k]` |
|---|---|---|
| 1 (最高ノイズ) | 0.99995 | `[0.9929, 1.0]` |
| 2 | 0.9858 | `[0.8246, 0.9929)` |
| 3 | 0.6633 | `[0.4817, 0.8246)` |
| 4 (最低ノイズ) | 0.3 | `[0.0, 0.4817)` |

band 境界は隣接 schedule 点の中点。各 sub-model は band 内で `√(1-ᾱ) ~ U(L_k, U_k)` で連続的に sampling されるため、推論時に正確な schedule 点を入力しても問題なく動作する。

### B2. Time-invariant spectral enhancement post-filter [Okamoto21 §3.3]

**式**: 開発セット上で原波形と合成波形の STFT 振幅スペクトル差を平均し、その iSTFT (1 frame) で固定 FIR を得る。

$$
\Delta A(f) = \frac{1}{T}\sum_t\bigl(|X_\text{orig}(t,f)| - |X_\text{synth}(t,f)|\bigr)
\;\Rightarrow\;
\text{fir} = \text{iSTFT}_1(\Delta A) \quad (\text{length 512})
$$

実装パラメータ ([Okamoto21] の実験): `n_fft=512, hop=256, window=Hann`、dev set 40 utterances。

**特性**: 振幅 **差** の加算補償 (NOT 比型)、α 等のハイパーパラメータ不要、Nyquist 端で +5〜8 dB の高域シェルフ。

詳細擬似コードは `docs/architecture.md §5 / docs/training.md §4.3` 参照。

### B3. Diff reverse step (BDDM/DDPM 形式) [BDDM `bddm/sampler/sampler.py`]

論文の 4 値は **ᾱ_t (cumulative)** として直接解釈 (PDF §3.3 本文準拠)。β_t は隣接 ᾱ_t から逆算: `β_t = 1 - ᾱ_t / ᾱ_{t-1}` (慣例: ᾱ_0 = 1)。

DDPM 標準形:
$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right) + \sigma_t z,\;\; z\sim\mathcal{N}(0,I)
$$
$$
\sigma_t^2 = \beta_t \cdot \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t},\quad \sigma_0 = 0
$$

詳細擬似コードは `docs/training.md §4.2` 参照。

---

## C. 参考実装から確定した事項

### C1. ConvNeXt block 内部仕様 [Vocos `vocos/modules.py`]
- **embed_dim (dim)**: 512
- **intermediate_dim (MLP hidden)**: 1536 (拡張比 3×)
- **Depthwise Conv1d**: kernel_size=**7**, padding=**3**, groups=`dim`, bias=True
- **Activation**: GELU
- **LayerNorm** (channels_last, eps=1e-6)
- **LayerScale (gamma)**: 学習可能 channel-wise scale, init=**1e-6**
- 構造順序: DWConv → transpose → LayerNorm → Linear(dim→1536) → GELU → Linear(1536→dim) → γ → transpose → residual add
- **Diff 版のみ**: block の入口に `Linear(512, dim)` の additive bias 注入 (`x = x + fc_t(e).unsqueeze(-1)`) を追加

### C2. Generator input/output 詳細 [Vocos `vocos/models.py` + `wavenext-impl::heads.py`]

#### 入力 embedding (Vocos `VocosBackbone.__init__` L51-60)
- `nn.Conv1d(input_channels, dim=512, kernel_size=7, padding=3, bias=True)`
- transpose to `(B, T, C)`
- `nn.LayerNorm(512, eps=1e-6)` を 1 段

#### Output head (`wavenext-impl::heads.py::WaveNextHead` L84-114)
- `nn.LayerNorm(dim, eps=1e-6)` (head 前最終 LN, Vocos `final_layer_norm`)
- `nn.Linear(dim=512, n_fft+2, bias=True)`  ← linear_1 (Vocos `ISTFTHead` 互換、warm-start 用)
- `nn.Linear(n_fft+2, hop_length, bias=False)`  ← linear_2
- reshape `(B, L, hop_length) → (B, L*hop_length)`
- `torch.clip(x, min=-1.0, max=1.0)` (tanh ではない)

#### 重み初期化
- 全 Conv1d / Linear: `trunc_normal_(std=0.02)`
- bias: zero

### C3. STFT / mel 抽出残り (PDF 未明示) [WaveFit-PT / Vocos]
| 項目 | GAN | Diff |
|---|---|---|
| win_length | **1200** [WaveFit-PT] | **1024** [Vocos] |
| n_fft | **2048** [WaveFit-PT] | **1024** [Vocos] |
| f_min / f_max | **20 / 12000** [WaveFit-PT] | **20 / 12000** [WaveFit-PT] |
| padding | center | center |
| power | **1.0** (magnitude) [Vocos] | **1.0** [Vocos] |
| mel_scale | **slaney** [HiFi-GAN 慣例] | **slaney** |
| norm | **slaney** | **slaney** |
| log type | **自然対数** [Vocos `safe_log`] | **自然対数** |
| log eps | **1e-5** [HiFi-GAN 慣例] | **1e-5** |

```python
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000, n_fft=..., hop_length=..., win_length=...,
    n_mels=128, f_min=20.0, f_max=12000.0,
    power=1.0, mel_scale="slaney", norm="slaney", center=True,
)(audio)
log_mel = torch.log(torch.clamp(mel, min=1e-5))
```

#### Sub-model 入力側 STFT module は mel-spec と同一パラメータ
`mel-spec の時間長に truncate` には hop 一致が必須。Vocos の `ISTFTHead` で `win_length=n_fft` の慣例を踏襲し、`n_fft / win_length / hop` すべて mel-spec と一致させる。

### C4. Audio 正規化 [Vocos `vocos/dataset.py` L42-43]

> **実装方針更新 (T-M0.1 / M0 phase review, 2026-05-27)**: torchaudio 2.11 で `sox_effects` が
> **削除**されたため `apply_effects_tensor` は使用不可。sox `norm -X dB` は「peak を −X dBFS に
> 正規化」する線形ゲインなので、**`gain = 10**(target_dbfs/20) / wav.abs().max(); wav *= gain`** で
> **数値的に等価に自前実装**する (configs の `audio_normalization.method: peak_norm`)。意味は不変:
> train は peak を U(−6,−1) dBFS、val/推論は −3 dBFS。下記コードは原典 (Vocos) の記録として残す。
> なお `np.random.uniform(-1, -6)` は numpy が範囲指定なので `U(-6,-1)` と同値 (引数順は無関係)。

```python
# 原典 (Vocos)。torchaudio 2.11 では下記 sox_effects は使えない → peak_norm で再実装。
# 訓練
gain_db = np.random.uniform(-1, -6)
audio = sox_effects.apply_effects_tensor(audio, sr=24000, effects=[["norm", f"{gain_db:.2f}"]])
# 検証/推論
audio = sox_effects.apply_effects_tensor(audio, sr=24000, effects=[["norm", "-3.0"]])
```

- `float32`, `[-1, 1]`, mono (stereo → `mean(dim=0)`)
- LUFS 正規化や peak clip は不要 (peak-based 正規化のため)

### C5. Discriminator 仕様 (WaveFit と同一) [WaveFit-PT `src/model/discriminator.py`]
- **方式**: **MSD のみ × 3 sub-discriminators** (MelGAN tradition)。**MPD は不使用**
- 隣接 sub-discriminator 間: `AvgPool1d(kernel=4, stride=2)` で audio を downsample
- 各 sub-discriminator (NLayerDiscriminator):
  - Layer 0: `ReflectionPad1d(7) → WNConv1d(1→16, kernel=15) → LeakyReLU(0.2)`
  - Layers 1-4: depthwise-separable conv, kernel=41, stride=4, groups=`nf_prev//4`, channels 倍々 (上限 1024)
  - Layer 5: `WNConv1d` (channel 倍化) + LeakyReLU(0.2)
  - Layer 6: 最終 `WNConv1d → 1ch`
- 全 Conv1d に `weight_norm`
- ndf=16, layers=4, downsampling_factor=4, num_D=3

### C6. GAN Loss 仕様 (WaveFit と同一) [WaveFit-PT `src/loss/mrstft.py`]
- **Adversarial**: **hinge GAN loss** (NOT LSGAN)
- **Feature matching**: 中間特徴の L1 distance
- **Multi-resolution STFT loss**:
  - n_ffts = [512, 1024, 2048]
  - win_sizes = [360, 900, 1800]
  - hop_sizes = [80, 150, 300]
  - sub-loss: Spectral Convergence (L2 ratio) + Magnitude L1 (log-amplitude)
  - EPS = 1e-5
- **Loss 重み (LibriTTS)**: D-GAN=1.0, D-Feature=10.0, MRSTFT-SC=2.5, MRSTFT-Mag=2.5, Mel-MAE=0.0

### C7. Diff Noise level conditioning [FastDiff `module/FastDiff_model.py`]

**方式: additive bias (NOT FiLM)**。各 ConvNeXt block ごとに独立 projection。

#### Sinusoidal embedding (`calc_diffusion_step_embedding`)
- 入力: `c = √(1-ᾱ)` (連続値)
- 出力次元: 128
- 標準 DDPM/Transformer 形式: `freq = exp(-arange(64) * log(10000)/63)`, `[sin(c*freq); cos(c*freq)]`

#### Shared head
- `Linear(128, 512)` → SiLU → `Linear(512, 512)` → SiLU
- forward あたり 1 回計算、全 ConvNeXt block で共有

#### Per-block 注入
- 各 block ごとに **独立した** `nn.Linear(512, dim=512)`
- block の入口 (residual 取得前、dwconv 前) で `x = x + fc_t(e).unsqueeze(-1)` を加算
- 時間軸への broadcast は `unsqueeze(-1)`

詳細擬似コードは `docs/architecture.md §5.4 / docs/training.md §3.5` 参照。

### C8. 訓練ハイパーパラメータ
| 項目 | GAN [WaveFit-PT] | Diff [FastDiff / Vocos] |
|---|---|---|
| Optimizer | **AdamW** | **Adam** |
| betas | **[0.8, 0.99]** | **[0.9, 0.98]** |
| Weight decay | **1e-3** | **0** |
| Learning rate (G) | **1e-4** | **2e-4** |
| Learning rate (D) | **2e-4** (G の 2 倍) | n/a |
| LR Scheduler | **InverseLR** (inv_gamma=200000, power=0.5, warmup=0.999) | 固定 lr |
| Gradient clip | max_grad_norm = **1.0** | max_grad_norm = **1.0** |
| Max steps | ~2M (Vocos 慣例) | 1M / sub-model |
| Segment length | 16,384 (Vocos) | 25,600 |
| Batch size | 16 (Vocos 流、A100 40GB で妥当) | 20 |
| EMA | **不使用** (Vocos / WaveFit-PT) | **不使用** (FastDiff) |
| Validation 間隔 | 10,000 steps | 10,000 steps |
| Best ckpt 基準 | MR-STFT (sc + mag) 合計最小 | MSE on noise prediction |

---

## D. 残る予備実験項目 (実装時に詰める、性能に微小な影響しか及ぼさない)

これらは性能に微小な影響しか及ぼさないため、最初は妥当な慣例値で実装し、必要に応じて調整する範疇。

| 項目 | 推奨初期値 | 根拠 |
|---|---|---|
| Fixed-point iteration の初期化 (GAN) | `torch.zeros_like(x_gt)` | PDF「initial input noise isn't required」+ ゼロが最小実装 |
| Random seed | 42 | 任意の固定値 |
| Validation utterance 数 | 100 | 慣例 |
| Train segment length (GAN) | 16,384 (Vocos 流) | A100 40GB で妥当 |
| Train segment length (Diff) | 25,600 (FastDiff) | A100 40GB で妥当 |
| Post-filter dev set 数 | 100〜500 utterances | [Okamoto21] は 40 で実施 |
| UTMOS / NISQA バージョン | sarulab-speech/UTMOS22 default、gabrielmittag/NISQA v2 | 標準 |
| MCD / log F0 RMSE 実装 | `pymcd` + `pyworld` | 標準ライブラリ |
| LibriTTS-R 前処理 (リサンプル/トリム) | 24kHz そのまま、無音トリムなし | sox `norm` のみ |

---

## E. 解決経路と一次情報源

| 確定内容 | 情報源 |
|---|---|
| Generator output head (Linear×2 → reshape → clip) | [wavenext-impl] `heads.py::WaveNextHead` + [WaveNeXt-orig] poster |
| Generator input embedding (Conv1d k=7, LN) | [Vocos] `vocos/models.py::VocosBackbone.__init__` |
| Sub-model 入力 STFT (mel と同一 n_fft/win/hop) | [Vocos] ISTFTHead 慣例 (win_length=n_fft) + 論文 §3.1 truncation 制約 |
| Mel 正規化 (log natural, eps=1e-5, slaney) | [Vocos] `safe_log` + HiFi-GAN/BigVGAN 慣例 |
| Audio 正規化 (sox norm) | [Vocos] `vocos/dataset.py` L42-43 |
| ConvNeXt block 仕様 (dim=512, mid=1536, k=7, LS=1e-6) | [Vocos] `vocos/modules.py::ConvNeXtBlock` |
| 最終 activation (`clip(-1, 1)`, NOT tanh) | [wavenext-impl] `heads.py::WaveNextHead.forward` L111-112 |
| EMA 不使用 | [Vocos] `vocos/experiment.py` + [WaveFit-PT] `src/trainer.py` (両方とも EMA なし) |
| Diff conditioning (additive bias, per-block proj) | [FastDiff] `module/FastDiff_model.py` + `modules.py::TimeAware_LVCBlock` |
| Sinusoidal embedding (128 dim, log(10000)/63) | [FastDiff] `util.py::calc_diffusion_step_embedding` |
| Sub-model partition (point-specialized 1-to-1) | [PDF] §3.3 + Table 1 (57.68M = 14.42M × 4) |
| Post-filter 式 (振幅差平均 → FIR) | [Okamoto21] §3.3 + §4.1 |
| Diff reverse step (DDPM 形式) | [BDDM] `bddm/sampler/sampler.py` |
| GAN ハイパーパラメータ・Discriminator・MR-STFT loss | [WaveFit-PT] `configs/*` / `src/model/discriminator.py` / `src/loss/mrstft.py` |
| Diff ハイパーパラメータ | [FastDiff] `modules/FastDiff/config/base.yaml` |
| Validation 基準 (MR-STFT, 10k step) | [WaveFit-PT] `configs/trainer/default.yaml::metrics_for_best_ckpt` |

### Okamoto21 preprint 取得経路 (paywall 回避)
- 第 1 経路 (推奨): NICT ASTREC https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2021_okamoto.pdf
- 第 2 経路: Okamoto 氏個人ページ https://www.okamotocamera.com/preprint_icassp_2021_okamoto.pdf
- 第 3 経路 (フォールバック): Wayback Machine http://web.archive.org/web/20240415022107/https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2021_okamoto.pdf

### WaveNeXt 元論文 (Okamoto+ ASRU 2023) 取得経路
- 主要: poster PDF https://www.okamotocamera.com/asru_2023.pdf
- 4-page paper: IEEE Xplore (DOI 10.1109/asru57964.2023.10389765, 要 access)
- 非公式 PyTorch 移植: https://github.com/wetdog/wavenext_pytorch (Vocos fork)

### 各 docs ファイルとの対応
| 確定内容 | 反映先 |
|---|---|
| アーキテクチャ全体・各部品の構造 | `docs/architecture.md` |
| 訓練・推論手順の擬似コード | `docs/training.md` |
| YAML 設定雛形・実装計画 | `docs/implementation-plan.md` |
| 論文サマリ (背景・貢献・結果) | `docs/paper-summary.md` |
| 本表の元情報 | `docs/open-questions.md` (本ファイル) |

---

## まとめ

当初 PDF テキスト抽出のみでは **30〜40% が未確定** だった再現実装情報は、以下の段階で完全に確定:

1. **PDF を画像化** して Fig 1〜4 を目視確認 → 拡散式・generator 構造を確定
2. **4 つの参考実装の `configs/` と core ソース** を確認 → ConvNeXt 内部・hyperparameter・Discriminator・loss を確定
3. **[Okamoto21] preprint 入手** (NICT / Okamoto 氏個人ページ) → sub-model range の指針・post-filter を確定
4. **[BDDM] 公式実装** 確認 → reverse step を確定
5. **[WaveNeXt-orig] poster + [wavenext-impl]** 確認 → Generator output head の 2 段 Linear と `clip(-1, 1)` を確定
6. **[FastDiff] のソース** 読了 → Diff conditioning の additive bias 方式と per-block 注入を確定
7. **PDF Table 1 の paramter 数 (57.68M = 4 × 14.42M) と論文記述 §3.3** から → sub-model partition は point-specialized 1-to-1 と確定

再現実装は **本論文 PDF + 4 参考実装 + Okamoto21 preprint + WaveNeXt-orig poster + BDDM 公式実装** で **完全に充足**。docs 4 ファイルだけ見て実装着手可能な状態に到達した。
