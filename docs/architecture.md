# WaveNeXt 2 アーキテクチャ詳細

論文 Section 3 および Figure 2 / Figure 3 を再現実装に向けて整理した資料。

## 1. 全体像

WaveNeXt 2 のコア部品は 2 つの **sub-model** の集合で、GAN-WaveNeXt 2 と Diff-WaveNeXt 2 で共通の **sub-model 構造** を使う。

```
┌─────────────────────────────────────────────────────────────────┐
│  Sub-model (Fig. 2b) ── GAN/Diffusion 両方で共有 ──────────────  │
│                                                                  │
│   y_{t-1}  ──► STFT module ──► STFT-spec ──┐                     │
│                                              ├─► WaveNeXt-based │
│   mel-spectrogram (128-dim) ─────────────── ┘   generator ──► n_{t-1}
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

- 入力: 直前ステップの波形 `y_{t-1}` (またはノイズ `n`) と mel-spectrogram
- 出力: 現在ステップで除去すべきノイズ成分 `n_{t-1}`
- iteration を t = 1..T 回繰り返して最終波形 `y_0` を合成

## 2. WaveNeXt-based generator (Fig. 2a)

オリジナル WaveNeXt (Okamoto et al., ASRU 2023) の構造を踏襲した上で、出力が **波形そのもの** ではなく **ノイズ成分** になっている点が WaveNeXt 2 の改変点。

### 構造 (Fig. 2a + 元 WaveNeXt poster + Vocos ソースで確定)

```
Concat(Mel-spec [128 ch], STFT-spec [2F-2 ch])  ← shape: (B, C_in, T_mel)
  │  C_in = 128 + (2F-2)  ※ F = n_fft//2 + 1
  ▼
Conv1d(in=C_in, out=512, kernel=1, padding=0, bias=True)   ← embed (kernel=1 確定、下記注記)
  │
  ▼ transpose to (B, T_mel, 512)
LayerNorm(512, eps=1e-6)
  │
  ▼
ConvNeXt block × 8        ← n = 8 (全モデル共通) [PDF Fig 2]
  │  (Diff 版のみ各 block 入口で共有 noise emb (512次元) を射影なし additive bias で加算, §5.4)
  ▼
LayerNorm(512, eps=1e-6)  ← head 前の最終正規化 [Vocos 慣例]
  │
  ▼
Linear(512 → n_fft+2), bias=True           ← linear_1 [元 WaveNeXt poster + wetdog 実装で確定]
  │
  ▼
Linear(n_fft+2 → hop_length), bias=False   ← linear_2 [bias=False は元 WaveNeXt 実装で確定]
  │
  ▼ reshape (B, T_mel, hop_length) → (B, T_mel * hop_length)
torch.clip(x, min=-1.0, max=1.0)            ← tanh ではなく clip
  │
  ▼
Synthesized waveform (= ノイズ成分 n_{t-1})
```

> **embed kernel size の確定 (T-M1.4 実装時調査, 2026-05-27)**: 当初この図は kernel=7 だったが、
> concat 入力 (GAN 2176ch) に kernel=7 を適用すると embed だけで 7.8M、sub-model 計 ~22M となり、
> Table 1 の GAN sub-model 14.99M (2 iter=29.97M=2×, 5 iter=74.93M=5× の**厳密倍数**) と矛盾する
> (22M なら 2 iter=44M のはず)。ConvNeXt×8 (12.64M)+heads (1.67M)=14.31M が大半を占め embed には
> ~0.67M しか残らない。**embed kernel=1** (1×1 channel 射影 = 1.11M、GAN 計 15.43M = Table 1 +2.9%) を
> 採用。時間方向の文脈は後段 ConvNeXt block の depthwise kernel=7 が担うため kernel=1 で機能上問題なし。
> **関連事項 (2026-05-27 解決)**: Diff sub-model は per-block fc_t (8×0.263M=2.1M) を含めると 16.46M で
> Table 1 の 14.42M を +14% 超過した。エージェントチーム調査 + Table 1 連立復元の結果、per-block fc_t を
> **撤去**し共有 NoiseEmbedding を射影なしで各 block に additive 注入する方式に変更 (sub-model=14.354M, −0.46%)。
> 詳細は §5.4 / open-questions §C7。

### 設計上の意図
- **元 WaveNeXt の `linear_1` の出力次元 `n_fft+2` は Vocos の `ISTFTHead` の `Linear(dim → n_fft+2)` と一致**
  → 元 Vocos の事前学習重みで warm-start できるように設計されている
- **`linear_2` は `bias=False`**: 元 WaveNeXt 実装 (`wetdog/wavenext_pytorch`) で明示
- **重み初期化**: 全 Conv1d / Linear を `trunc_normal_(std=0.02)` で初期化、bias は zero
- **最終 activation は `tanh` ではなく `torch.clip(-1, 1)`** (HiFi-GAN とは異なる)

### 入力次元の具体値
| モデル | n_fft | F = n_fft/2+1 | STFT-spec ch (2F-2) | Conv1d in_ch | linear_1 out | linear_2 out (=hop) |
|---|---|---|---|---|---|---|
| GAN | 2048 | 1025 | 2048 | 128+2048=2176 | 2050 | 300 |
| Diff | 1024 | 513 | 1024 | 128+1024=1152 | 1026 | 256 |

### ConvNeXt block の内部構成 (Vocos `vocos/modules.py` で確定)
- **embed_dim (dim)**: 512
- **intermediate_dim (MLP hidden)**: 1536 (拡張比 3×)
- **Depthwise Conv1d**: kernel=7, padding=3, groups=dim, bias=True
- **Activation**: GELU
- **LayerNorm** (channels_last, eps=1e-6)
- **LayerScale γ**: 学習可能 channel-wise scale、init=`1e-6` (Vocos の `layer_scale_init_value` 既定値)
- **構造順序** (Vocos `vocos/modules.py::ConvNeXtBlock.forward`):
  ```
  residual = x
  x = dwconv(x)              # (B, C, T)
  x = x.transpose(1, 2)      # (B, T, C)
  x = LayerNorm(x)
  x = pwconv1(x)             # Linear(dim → 1536)
  x = GELU(x)
  x = pwconv2(x)             # Linear(1536 → dim)
  x = x * gamma              # LayerScale
  x = x.transpose(1, 2)      # (B, C, T)
  x = residual + x
  ```
- **Diff 版のみ追加**: 各 block の入口 (residual 取得前、dwconv 前) で `x = x + e.unsqueeze(-1)` の **射影なし additive bias** を加算 (`e`=共有 NoiseEmbedding 出力 512次元; per-block fc_t は撤去, 詳細は §5.4)

## 3. STFT module (統一フレームワークの要)

WaveNeXt 2 の最大の変更点。入力波形を STFT して generator にフィードバックすることで、GAN/Diffusion 両方の "前のステップの波形を条件にする" 構造を統一する。

### 処理手順 (Section 3.1)
1. 入力波形 `y_{t-1}` に **Hann window** を適用
2. STFT を **center=True** で計算
3. 結果の複素スペクトログラム時系列を **mel-spectrogram の時間長に合わせて truncate**
4. 実部 `R` と虚部 `I` を分離
5. **STFT-spec** を作成:
   - 実部: そのまま全帯域 (DC と Nyquist 含む) → bin 数 `F = n_fft/2 + 1`
   - 虚部: **DC と Nyquist を除外** → bin 数 `F - 2`
   - concatenate → channel 数 `2F - 2`
6. mel-spectrogram (128 ch) と STFT-spec を結合して generator へ入力

### STFT パラメータ (mel-spec と完全一致、確定)

`mel-spec の時間長に truncate` が成立するためには **hop が mel と一致** している必要がある。Vocos の `ISTFTHead` (`win_length = n_fft`) の慣例に従い、`n_fft / win_length / hop` すべて mel-spec の値と一致させる。

| 項目 | GAN-WaveNeXt 2 | Diff-WaveNeXt 2 |
|---|---|---|
| n_fft | 2048 | 1024 |
| win_length | 1200 | 1024 |
| hop_length | 300 | 256 |
| window | Hann | Hann |
| center | True | True |
| normalized | False | False |
| onesided | True | True |

```python
# 擬似コード (PyTorch)
def stft_module(y, n_fft, hop, win_length):
    window = torch.hann_window(win_length, device=y.device)
    Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length,
                   window=window, center=True, normalized=False,
                   onesided=True, return_complex=True)
    # Y: (B, F, T_stft) 複素テンソル, F = n_fft//2 + 1
    Y = Y[:, :, :T_mel]                       # mel の時間長に truncate
    real = Y.real                              # (B, F, T_mel)
    imag = Y.imag[:, 1:-1, :]                  # DC と Nyquist を除外、(B, F-2, T_mel)
    stft_spec = torch.cat([real, imag], dim=1) # (B, 2F-2, T_mel)
    return stft_spec
```

### 重要点
- 実部と虚部を分離して扱うのは Vocos と同様だが、Vocos は出力で iSTFT に渡すのに対し、WaveNeXt 2 では **入力側** に STFT が来る
- DC/Nyquist の虚部は常に 0 なので削除している
- WaveFit-PT は前ステップ波形の STFT 化は行わず生波形をそのまま入れる → **この STFT module は WaveNeXt 2 独自の設計**

## 4. GAN-WaveNeXt 2 (Fig. 1a)

### 訓練ループ (1 iter 内)
```
for t in T..1:
    n_t = sub_model_t(mel, y_t)        # sub-model がノイズを予測
    y_{t-1} = y_t - n_t                # 残差で denoising
loss = STFT_loss(y_0, x_gt)
       + adversarial_loss(D, y_0, x_gt)
       + feature_matching_loss(D, y_0, x_gt)
```
※実際の loss 構成は **WaveFit と同一** (論文中明記)。

### WaveFit からの簡略化 (Section 3.2)
- ❌ "denoising" 制約 (gradient sign を強制する WaveFit の制約) → 不要
- ❌ 初期入力ノイズ → 不要
- ❌ gain adjustment module → 不要 (STFT loss が代替)

### 反復数別パラメータ数 (Table 1)
| iter | params |
|---|---|
| 2 | 29.97M |
| 3 | 44.96M |
| 4 | 59.94M |
| 5 | 74.93M |

= 1 sub-model あたり約 14.99M (iter ごとに1サブモデル増える)

## 5. Diff-WaveNeXt 2 (Fig. 1b, Fig. 3)

### サブモデル分割 (Section 3.3 で確定: point-specialized 1-to-1)

#### 重要な解釈
論文は "we divide the denoising task into **four** stages and construct **four** sub-models" と明記し、Fig 3 では 4 sub-model が inference 順に並ぶ。Table 1 のパラメータ数 `57.68M = 14.42M × 4` も **全 4 sub-model が deploy される** ことを示す。

→ Okamoto21 (K=10 で N=6/N=25 step を使う) の strict equal partition (band 1 が無使用になり得る) ではなく、WaveNeXt 2 は **K = N = 4 で 1-to-1 マッピング** の point-specialized 構成を採用する。

#### Point-specialized 分割の確定式

- schedule: `ᾱ = [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]` (推論順 t=1..4、`t=1` が純ノイズ側)
- 対応 `√(1-ᾱ) = [0.99995, 0.9858, 0.6633, 0.3]`
- 各 sub-model k は schedule 点 k の `√(1-ᾱ_k)` を中心とする **band** で訓練される
- band 境界は **隣接 schedule 点の中点**

#### Band 境界 (確定)
| k (推論順) | schedule 点 √(1-ᾱ_k) | 訓練 band の `√(1-ᾱ)` 範囲 |
|---|---|---|
| 1 (最高ノイズ) | 0.99995 (ᾱ=1e-4) | `[(0.99995+0.9858)/2, 1.0] = [0.9929, 1.0]` |
| 2 | 0.9858 (ᾱ=2.8e-2) | `[(0.9858+0.6633)/2, (0.99995+0.9858)/2) = [0.8246, 0.9929)` |
| 3 | 0.6633 (ᾱ=5.6e-1) | `[(0.6633+0.3)/2, (0.9858+0.6633)/2) = [0.4817, 0.8246)` |
| 4 (最低ノイズ) | 0.3 (ᾱ=9.1e-1) | `[0.0, (0.6633+0.3)/2) = [0.0, 0.4817)` |

#### 訓練時の sampling 式 (sub-model k, k=1..4)
$$
\sqrt{1-\bar{\alpha}} \;\sim\; \mathcal{U}(L_k,\; U_k)
$$
ここで `[L_k, U_k]` は上表の band 範囲。

#### 連続 noise level conditioning
各 sub-model は `c = √(1-ᾱ)` を **continuous な条件** として受け取る (sinusoidal embedding + FC × 2 swish、FastDiff 同様)。band の中で uniform sampling されるため、推論時に schedule の正確な値が来ても問題ない。

#### 推論時の dispatch
推論で 4-step schedule を逆順 (denoising 方向 t=1→4) に辿るとき、ステップ k では **sub-model k を呼ぶ** だけ (1-to-1)。
- t=1 (ᾱ=1e-4, √(1-ᾱ)=0.99995) → sub-model 1
- t=2 (ᾱ=2.8e-2, √(1-ᾱ)=0.9858) → sub-model 2
- t=3 (ᾱ=5.6e-1, √(1-ᾱ)=0.6633) → sub-model 3
- t=4 (ᾱ=9.1e-1, √(1-ᾱ)=0.3) → sub-model 4

#### 代替解釈 (実装前注意)
Okamoto21 strict equal partition (`[k-1)/K, k/K)`) で実装する場合、4-step schedule では band 1 ([0, 0.25)) が無使用となる。論文の sub-model 数 4 と整合させるためには本セクションの point-specialized partition を採用。実験で品質に有意差がない場合は equal partition でも良い (Okamoto21 の K=10 慣例)。

### 訓練 (各 sub-model 単独)
- 訓練ターゲット noise level `a_t` を schedule から選択
- 拡散式 (**論文 Fig 1b の数式を画像から読み取って確定**):
  $$x_t = \sqrt{a_t} \cdot x_0 + \sqrt{1 - a_t} \cdot \epsilon$$
  - `a_t` は **cumulative noise level** (DDPM の $\bar{\alpha}_t$ に対応、累積積)
  - `x_0`: クリーン波形
  - `ε ~ N(0, I)`: ガウシアンノイズ
  - これは DDPM 標準形そのもの (pdftotext で √ が脱落していたために原文 `at·x0 + 1 - at·ε` のように見えていただけ)
- sub-model に mel-spectrogram と $x_t$ を入力し、$\epsilon$ を予測
- loss: MSE (Fig. 1b)

### 4-step noise schedule の値の解釈 (確定)

論文に与えられた 4 値: `[1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]`

PDF 本文 §3.3 と Okamoto21 preprint との突き合わせから、**これらは `ᾱ_t` (cumulative noise level)** として直接解釈する (Agent A/D の論文本文ベース確認)。

派生値:
```python
abar = np.array([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])          # 直接 ᾱ_t
one_minus_abar = 1.0 - abar                                  # [0.9999, 0.972, 0.44, 0.09]
sqrt_abar = np.sqrt(abar)                                     # [0.01, 0.167, 0.748, 0.954]
sqrt_one_minus_abar = np.sqrt(one_minus_abar)                 # [0.99995, 0.9858, 0.6633, 0.3]

# 【CRITICAL】β を逆算してはならない:
# 論文の schedule は denoising 順で ᾱ が増加列 [1e-4 → 9.1e-1] のため、
# β_t = 1 - ᾱ_t/ᾱ_{t-1} を隣接点に適用すると β[1] = 1 - 280 = -279 のような
# 負値になり α = 1-β > 1 で reverse step が発散する (連続 DDPM の 3-step
# サブサンプリングで隣接 β は物理的意味を持たない)。
# reverse step は β 不使用の x_0 予測経由式を使う (docs/training.md §4.2)。
```

`ᾱ` index 順 t=1..4:
- t=1: `ᾱ=1e-4` (純ノイズ、推論で **最初** に到達)
- t=2: `ᾱ=2.8e-2`
- t=3: `ᾱ=5.6e-1`
- t=4: `ᾱ=9.1e-1` (ほぼクリーン、推論で **最後** に到達)

#### 代替解釈 (BDDM 公式実装に従う場合)
tencent-ailab/bddm の `b_infer` 保存フォーマットでは同一の値列を **per-step β** として保存する慣例があり、`ᾱ = cumprod(1 - β)` で導出する。論文の数値を `b_infer` ファイルから取得した場合はこの解釈になる。両者は数学的に等価な再パラメトリゼーションなので、再現実装で予備実験して品質を確認する。

### Noise schedule (4 step)
```
a = [1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]
```
- BDDM (Lam et al. ICLR'22) ベースの noise schedule predictor で予測
- 推論時に4ステップでこの noise schedule を逆順にたどる

### 推論 (Fig. 3)
```
x ~ N(0, I)                              # 初期ノイズ (純ノイズ side, ᾱ=1e-4)
for t in 1..4:                            # denoising 順 (ᾱ 増加方向)
    eps_pred = sub_model_t(mel, x, c_t=√(1-ᾱ_t))
    x0_hat   = (x - √(1-ᾱ_t)·eps_pred) / √(ᾱ_t)            # x_0 推定 (β 不使用)
    x        = √(ᾱ_next)·x0_hat + √(1-ᾱ_next-σ²)·eps_pred + σ·z   # 次のクリーン側へ射影
y_0 = post_filter(x)                      # Time-invariant spectral enhancement
```
詳細な reverse step 式 (DDIM/DDPM 一般形、β 不使用、`σ² = η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`) は `docs/training.md` §4.2 参照。

### Noise level conditioning の注入方法 [FastDiff `modules/FastDiff/module/FastDiff_model.py` で確定]

#### Sinusoidal embedding (`calc_diffusion_step_embedding`)
```python
def sinusoidal_embedding(c, dim=128):
    """c: (B,) or (B,1) noise level scalar"""
    half = dim // 2                              # 64
    freq = np.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half) * -freq) # (half,)
    e = c.unsqueeze(-1) * freq                   # (B, half)
    return torch.cat([torch.sin(e), torch.cos(e)], dim=-1)  # (B, dim)
```
- 入力: `c = √(1-ᾱ)` (連続値)
- 出力次元: **128**
- 標準 DDPM/Transformer 形式 (log-spaced freq 10^[0..4])

#### Shared embedding head (sub-model ごとに 1 セット)
```python
self.fc_t1 = nn.Linear(128, 512)
self.fc_t2 = nn.Linear(512, 512)
# forward
e = F.silu(self.fc_t1(sinusoidal_embedding(c, 128)))
e = F.silu(self.fc_t2(e))                       # (B, 512)
```
- 中間次元: **512**
- 活性化: **Swish (SiLU)**
- forward あたり 1 回だけ計算 (全 ConvNeXt block で共有)

#### Additive bias 注入 (射影なし、NOT FiLM) — per-block fc_t は撤去

**設計判断 (2026-05-27、エージェントチーム調査 + Table 1 連立復元)**: 当初は FastDiff/DiffWave 流に
各 ConvNeXt block が独立 `Linear(512,512)` (per-block fc_t、8 block 計 2.1M) を持つ設計だったが、
論文 **Table 1 の Diff sub-model=14.42M を +14% 超過**し丸め誤差では説明不可能なため **撤去**。
共有 NoiseEmbedding (内部に `Linear(512,512)+SiLU` の学習射影を持つ) の 512次元出力を、各 block
入口で **射影なしの additive bias** として直接加算する。これで sub-model=14.354M (−0.46%) と Table 1
にほぼ一致する。

```python
class ConvNeXtBlockDiff(nn.Module):
    def __init__(self, dim=512):
        ...
        # per-block fc_t は持たない (Table 1 整合)

    def forward(self, x, e):                     # x:(B,C,T), e:(B,512)=共有 NoiseEmbedding 出力
        h = x + e.unsqueeze(-1)                   # additive, 射影なし, broadcast over T
        # 以降は通常の ConvNeXt block
        residual = h
        h = self.dwconv(h)
        h = self.norm(h.transpose(1,2))
        h = self.pwconv2(F.gelu(self.pwconv1(h)))
        h = h * self.gamma                       # LayerScale
        h = h.transpose(1,2)
        return residual + h
```

#### 重要な点
- **FiLM (scale+shift) ではなく additive bias**。注入形式は DiffWave/Okamoto21 (WaveNeXt 2 の
  sub-modeling 直系祖先) の「共有 step embedding を各層で additive」の骨格に一致。
- bias は **各 block の入口で加算** (per-block 注入は維持; DiffWave 同様、層をまたいで強化される)。
- **per-block の射影層は持たない**。共有 NoiseEmbedding の最終 `Linear(512,512)` が実質の共有 projection。
  per-block 独立 `Linear(512,512)` は Table 1 が数学的に許容しない (+14%) ため撤去 (open-questions §C7)。
- 各 sub-model k は **独自の** NoiseEmbedding head を持つ (sub-model 間で共有しない)。
- DiffWave は per-layer に軽量 `Linear(512, residual_ch)` を持つが、ConvNeXt は width=512 が大きく
  8×`Linear(512,512)`=2.1M が予算超過。共有射影 (+1.4%) より射影なし (−0.46%) が Table 1 に近いため後者を採用。
  - smoke で conditioning が弱い兆候 (noise level 不感) が出た場合は共有 `Linear(512,512)` 1 個 (+1.4%) の追加を ablation。

### Post-filter (Time-invariant spectral enhancement) [Okamoto21 §3.3 で確定]

NICT ASTREC が公開している preprint (https://ast-astrec.nict.go.jp/release/preprints/preprint_icassp_2021_okamoto.pdf) §3.3 から取得した完全な定式化。

#### 構築手順 (開発セット上で 1 回だけ実施)
1. 開発セットの各発話について、原波形と合成波形の **STFT 振幅スペクトル** を計算
   - Hann window、Okamoto21 の実験: `n_fft=512, hop=256` (24 kHz)
2. 全フレームに渡って **振幅差** を平均:
   $$\Delta A(f) = \frac{1}{T}\sum_t \bigl(|X_\text{orig}(t,f)| - |X_\text{synth}(t,f)|\bigr)$$
3. 平均振幅差の **iSTFT (1 frame)** を取り、長さ 512 の **固定 FIR インパルス応答** を得る
4. 推論時: 合成波形にこの FIR を畳み込む (time-invariant、全発話で同じフィルタ)

#### 性質
- スペクトル比 `(target/synth)^α` ではなく、**加算的な振幅差補償**
- α やゲインなどのハイパーパラメータは **不要** (STFT 設定と FIR 長のみ)
- 周波数応答は ~2 kHz 以下で 0 dB、Nyquist 端で +5〜8 dB の **高域シェルフ** (Okamoto21 Fig 3a)
- 動機: WaveGrad 論文 [Chen+ ICLR'21] の指摘 「低 iteration 数の diffusion vocoder では HF detail が失われる」を補正

#### 擬似コード (論文と完全一致)
```python
# fit (once, on dev set)
diff_acc = np.zeros(257)
n = 0
for orig, synth in dev_pairs:
    mag_diff = (np.abs(stft(orig,  n_fft=512, hop=256, window="hann"))
              - np.abs(stft(synth, n_fft=512, hop=256, window="hann")))
    diff_acc += mag_diff.sum(axis=1)
    n += mag_diff.shape[1]
mean_diff_mag = diff_acc / n              # shape (257,)
fir = np.fft.irfft(mean_diff_mag, n=512)  # length 512, time-invariant
fir = np.fft.fftshift(fir)                # linear-phase 用に中央化

# inference
y_post = np.convolve(y_synth, fir, mode="same")
```

#### Okamoto21 実験での開発セット規模 (参考)
- 40 utterances (Japanese female, 24 kHz)
- WaveNeXt 2 (LibriTTS-R) で再現する場合は、validation split から数百発話を使うのが妥当

### パラメータ数
- Diff-WaveNeXt 2 (w/ sub-model): 57.68M
- Diff-WaveNeXt 2 (wo/ sub-model): 14.42M
  → sub-model 1 個約 14.42M、× 4 で約 57.68M

## 6. Discriminator (GAN-WaveNeXt 2 のみ)

WaveFit と同じ構成 (論文 Section 4.1 明記)。WaveFit-PyTorch 参考実装 (`src/model/discriminator.py`) を確認した結果、以下の **MSD のみ** であり、HiFi-GAN 系の MPD は使用しない。

### 構造
- **MSD (Multi-Scale Discriminator)、3 sub-discriminators** (MelGAN 系統)
- 隣接 sub-discriminator 間: `AvgPool1d(kernel=4, stride=2)` で downsample してから次に渡す
- **MPD は使用しない**

### 各 sub-discriminator (NLayerDiscriminator)
- Layer 0: `ReflectionPad1d(7)` → `WNConv1d(1 → 16, kernel=15)` → `LeakyReLU(0.2)`
- Layers 1〜4: depthwise-separable conv
  - kernel = `stride*10 + 1` (stride=4 → kernel=41)
  - stride = 4 (`downsampling_factor`)
  - groups = `nf_prev // 4` (グループ畳み込み)
  - channels は層ごとに 2 倍 (上限 1024)
  - LeakyReLU(0.2)
- Layer 5: `WNConv1d` (channel 倍化) + LeakyReLU(0.2)
- Layer 6: `WNConv1d → 1ch` (出力)

### 設定
- ndf (base channels): **16**
- 内部 layers: **4** downsampling layers
- num_D (sub-discriminator 数): **3**
- 全 Conv1d に `weight_norm`

### Loss
- **Adversarial**: **hinge GAN loss** `(1 - out_fake).relu().mean()` (LSGAN ではない)
- **Feature matching**: 中間特徴の L1 distance
- **Multi-resolution STFT loss**:
  - n_ffts = [512, 1024, 2048]
  - win_sizes = [360, 900, 1800]
  - hop_sizes = [80, 150, 300]
  - sub-loss: Spectral Convergence + Magnitude (L1 on log-amplitude)
- Loss 重み (LibriTTS 設定):
  - D-GAN: 1.0
  - D-Feature: 10.0
  - MRSTFT-SC: 2.5
  - MRSTFT-Mag: 2.5
  - Mel L1 MAE: 0.0 (このデータでは無効化)

詳細は [yukara-ikemiya/wavefit-pytorch `src/model/discriminator.py`](https://github.com/yukara-ikemiya/wavefit-pytorch/blob/master/src/model/discriminator.py) と `src/loss/mrstft.py` を参照。

## 6.5. Mel-spectrogram 抽出と log 正規化 (確定)

[Vocos `vocos/feature_extractors.py` + Vocos `safe_log`] と [HiFi-GAN/BigVGAN 慣例] を統合した値で確定。

```python
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000,
    n_fft=2048 if model=="gan" else 1024,
    hop_length=300 if model=="gan" else 256,
    win_length=1200 if model=="gan" else 1024,
    n_mels=128,
    f_min=20.0,
    f_max=12000.0,
    power=1.0,                # magnitude (Vocos と同じ)
    mel_scale="slaney",       # HiFi-GAN/BigVGAN 慣例
    norm="slaney",            # per-band normalization
    center=True,
)
mel = mel_transform(audio)                       # (B, 128, T_mel)
log_mel = torch.log(torch.clamp(mel, min=1e-5))  # 自然対数, eps=1e-5
```

### 各パラメータの確定根拠
- `power=1.0`: Vocos が magnitude を使用 (`vocos/feature_extractors.py::MelSpectrogramFeatures`)
- `mel_scale="slaney"`, `norm="slaney"`: HiFi-GAN / BigVGAN / WaveFit-PT の慣例
- `log` は **自然対数** (`torch.log`), `log10` ではない
- eps = `1e-5`: HiFi-GAN 系で過大増幅を抑える慣例値 (Vocos の `1e-7` より大きいが学習安定)
- `f_min=20.0`, `f_max=12000.0`: WaveFit-PT `wavefit-3.yaml` 値

## 6.6. Audio 正規化 (Vocos 方式、確定)

LibriTTS-R 24kHz 波形に対して以下の正規化を適用 [Vocos `vocos/dataset.py` L42-43]:

> **実装は peak 正規化で再構成 (T-M2.1, 2026-05-27)**: torchaudio 2.11 で `sox_effects` 削除のため
> 下記 sox は使えない。`gain = 10**(target_dbfs/20) / audio.abs().max(); audio *= gain` で数値等価に
> 自前実装 (`src/wavenext2/data/dataset.py::_peak_normalize`)、I/O は soundfile。意味は不変。下記は原典記録。

```python
# 原典 (Vocos)。torchaudio 2.11 では sox_effects 不可 → peak 正規化で再実装。
import numpy as np

gain_db = np.random.uniform(-1, -6)  # 訓練時
audio_norm, _ = torchaudio.sox_effects.apply_effects_tensor(
    audio, sample_rate=24000, effects=[["norm", f"{gain_db:.2f}"]]
)
# 検証/推論時は gain_db=-3.0 固定
```

- データ型: `float32`, 範囲 `[-1, 1]`
- モノラル化: stereo の場合は `audio.mean(dim=0)` (Vocos と同じ)
- LUFS 正規化や peak clip は不要 (peak-based 正規化のため)

## 7. ハイパーパラメータまとめ

| 項目 | GAN-WaveNeXt 2 | Diff-WaveNeXt 2 |
|---|---|---|
| Sampling rate | 24,000 Hz | 24,000 Hz |
| Mel channels | 128 | 128 |
| Hop size | 300 | 256 |
| Window | Hann | Hann |
| Win length | 1200 | 1024 |
| FFT size | 2048 | 1024 |
| f_min / f_max | 20 / 12000 | 20 / 12000 |
| ConvNeXt embed_dim | 512 | 512 |
| ConvNeXt intermediate_dim | 1536 | 1536 |
| ConvNeXt kernel_size | 7 | 7 |
| ConvNeXt blocks per sub-model | 8 | 8 |
| Sub-model 数 | 2〜5 (実験で比較、推奨 4) | 4 (固定) |
| 出力 | ノイズ成分 nt | ノイズ成分 nt |
| 訓練 loss | hinge GAN + L1 FM + MR-STFT (SC+Mag) | MSE on noise |
| Discriminator | MSD ×3 (WaveFit と同一) | なし |
| Post-filter | なし | time-invariant spectral enhancement |
| Optimizer | AdamW(lr=1e-4 G / 2e-4 D, β=[0.8,0.99], wd=1e-3) | Adam(lr=2e-4, β=[0.9,0.98]) |
| LR scheduler | InverseLR (inv_gamma=200000, power=0.5, warmup=0.999) | 固定 lr |
| Gradient clip | max_grad_norm = 1.0 | max_grad_norm = 1.0 |

## 8. 再現実装で未確定の項目 (予備実験で詰める範疇)

論文・参考実装・関連論文を突き合わせた結果、**全てのアーキテクチャ・訓練レシピは確定済み**。残るのは予備実験で詰める性質の項目のみ。

| 項目 | 推奨初期値 | 根拠・コメント |
|---|---|---|
| fixed-point iteration の初期入力 y_T (GAN) | `torch.zeros_like(x_gt)` | 論文「initial input noise isn't required」+ Vocos/WaveFit-PT に近い最小実装 |
| K=4 vs Okamoto21 strict partition | **point-specialized 1-to-1** (§5 参照) | 論文 Table 1 が 4 sub-model deploy を示唆 |
| Random seed | 42 (任意の固定値) | 再現性のため固定 |
| Validation split サイズ | 100 utterances | 慣例 |
| Post-filter dev set サイズ | 100〜500 utterances | Okamoto21 は 40 で実施 |
| Generator EMA | **不使用** | Vocos / WaveFit-PT 共に未使用 |
| Best ckpt 選択基準 | MR-STFT (sc + mag) 合計最小 | WaveFit-PT 慣例、10k step ごと |

詳細は `docs/open-questions.md` を参照。
