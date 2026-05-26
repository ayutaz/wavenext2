# WaveNeXt 2 論文サマリー

## 書誌情報
- **タイトル**: WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders with Residual Denoising and Sub-Modeling for GAN and Diffusion Models
- **著者**: Wangzixi Zhou, Takuma Okamoto, Yamato Ohtani, Sakriani Sakti, Hisashi Kawai
- **所属**: Nara Institute of Science and Technology (NAIST) / National Institute of Information and Communications Technology (NICT)
- **arXiv ID**: 2605.25506v1 [eess.AS]
- **公開日**: 2026年5月25日
- **デモページ**: https://37integer.github.io/WAVENEXT-2

## 1. 研究背景と動機

### ニューラルボコーダの2大潮流
1. **GAN ベース** (HiFi-GAN, MS-FC-HiFi-GAN, Vocos, WaveNeXt など)
   - 利点: 低レイテンシ、高品質
   - 欠点: 訓練が不安定、計算リソースを大きく消費 (例: HiFi-GAN は dual V100 GPU で 2.5M ステップ訓練に 300時間超)
2. **Diffusion (DDPM) ベース** (DiffWave, WaveGrad, FastDiff, SpecGrad など)
   - 利点: 訓練が安定で速い
   - 欠点: 多段階反復のため CPU 推論が遅い、ステップ削減で品質劣化

### 既存研究のギャップ
- Vocos / WaveNeXt は強力な ConvNeXt 系 generator を持つが、**GAN 専用** であり multi-speaker での性能が制限される
- ステップ削減のために BDDM, noise-level limited sub-modeling などが提案されてきたが、**GAN/Diffusion 双方に適用できる統一フレームワーク** は存在しなかった

## 2. 提案: WaveNeXt 2

GAN ベースと Diffusion ベースの両方に同じ ConvNeXt-based generator を流用できる **統一ボコーダフレームワーク**。

### 主な貢献
1. **ConvNeXt-based residual denoising and sub-modeling**
   - 各 sub-model が時間ステップごとに段階的にノイズ除去を行う
   - 単一アーキテクチャを GAN/Diffusion の両ボコーダタイプに適用可能にする
2. **RTF の大幅改善**
   - GAN-WaveNeXt 2: HiFi-GAN, WaveFit, 元 WaveNeXt と同等品質を維持しつつ大幅に高速化
   - Diff-WaveNeXt 2: FastDiff より高速かつ品質も競合する
3. **訓練効率**
   - Diff-WaveNeXt 2 は **32時間** で訓練完了 (FastDiff の 96時間より大幅短縮)

## 3. 二つの派生モデル

### GAN-WaveNeXt 2
- 訓練戦略: WaveFit の fixed-point iteration を採用
- 通常の DDPM のような確率的ノイズ除去ではなく、**決定論的に目標波形へ導く**
- WaveFit からの簡略化:
  - "denoising" 制約のための初期入力ノイズは不要
  - gain adjustment module は不要 (STFT loss で代替)
- Discriminator と loss は WaveFit と同じ

### Diff-WaveNeXt 2
- 訓練戦略: noise-level limited sub-modeling [Okamoto et al. ICASSP'21] を採用
- denoising を **4 段階に分割**し、それぞれを担当する 4 つの独立 sub-model を学習
- 各 sub-model は mel-spectrogram と特定の noisy audio に conditioning される
  - $x_t = a_t x_0 + (1 - a_t) \epsilon$
  - $x_0$: クリーン波形, $\epsilon$: ガウシアンノイズ, $a_t$: 累積ノイズ係数
- noise schedule は **BDDM ベースの noise schedule predictor** で予測
- 推論時は noise から開始し、4 sub-models を順次適用
- Time-invariant spectral enhancement post-filtering (Okamoto et al. ICASSP'21) を併用

## 4. 実験設定

### データ
- LibriTTS-R "train-clean-100" + "train-clean-360" 結合、24kHz、585時間
- 評価: "test-clean-100" の 4,824 サンプル
- MOS: 同サブセットから 20 サンプル、ネイティブ英語話者 20名

### モデル
- WaveNeXt-based generator は ConvNeXt ブロック n = 8
- mel-spectrogram は 128 次元
- hop size:
  - GAN-WaveNeXt 2: **300** (HiFi-GAN, WaveFit に合わせる)
  - Diff-WaveNeXt 2: **256** (FastDiff に合わせる)
- Diff-WaveNeXt 2 noise schedule (4 step): `[1.0e-04, 2.8e-02, 5.6e-01, 9.1e-01]`

### 評価指標
- 主観評価: MOS (5段階、20名)
- 客観評価: UTMOS, NISQA, MCD (Mel-Cepstral Distortion), log F0 RMSE
- 推論速度: RTF on NVIDIA A100 GPU / AMD EPYC 7542 CPU (1 core)

### 訓練ハードウェア
- NVIDIA A100 GPU (40GB) シングル

## 5. 主要結果

### 速度 (RTF)
| モデル | RTF GPU | RTF CPU |
|---|---|---|
| WaveFit (5 iter) | 0.0213 | 4.28 |
| HiFi-GAN V1 | 0.0226 | 5.36 |
| GAN-WaveNeXt 2 (5 iter) | **0.0066** | **0.20** |
| FastDiff (w/ sub-model) | 0.0625 | 0.80 |
| Diff-WaveNeXt 2 (w/ sub-model) | **0.0282** | **0.16** |

- GAN-WaveNeXt 2 (5 iter): WaveFit 比で GPU **-70%**、CPU **-90%** RTF
- HiFi-GAN 比でも GPU **-40%**、CPU **-75%** RTF
- Diff-WaveNeXt 2 (w/ sub-model): FastDiff (w/ sub-model) 比で GPU **-36%**、CPU **-80%** RTF

### 品質
- GAN-WaveNeXt 2: UTMOS / NISQA が WaveFit と同等、4 iter で MOS は WaveFit 5 iter および HiFi-GAN と同等
- Diff-WaveNeXt 2 (w/ sub-model): log F0 RMSE が HiFi-GAN より低い (0.12 ± 0.01)
- MCD では GAN-WaveNeXt 2 が HiFi-GAN を上回る (spectral fidelity が優位)

### 訓練時間 (Table 2)
| モデル | 訓練時間 |
|---|---|
| GAN-WaveNeXt 2 | 410 時間 |
| HiFi-GAN | 270 時間 |
| WaveFit | 410 時間 |
| Diff-WaveNeXt 2 | **32 時間** |
| FastDiff | 96 時間 |

### 課題
- sub-model 数の増加に比例して **総パラメータ数が増大**
  - GAN-WaveNeXt 2 (5 iter): 74.93M
  - Diff-WaveNeXt 2 (w/ sub-model): 57.68M

## 6. 結論 (論文の主張)
- WaveNeXt 2 は **GAN/Diffusion 双方に対応する最初の ConvNeXt-based 統一フレームワーク**
- multi-speaker での元 WaveNeXt の弱点を克服
- 用途別の推奨:
  - 高品質重視 → GAN-WaveNeXt 2
  - リソース制約環境での速度・訓練コスト重視 → Diff-WaveNeXt 2

## 7. 主要参照論文 (再現実装で必読)
- [9] WaveNeXt (Okamoto et al., ASRU 2023)
- [8] Vocos (Siuzdak, ICLR 2024)
- [11] WaveFit (Koizumi et al., SLT 2023)
- [16] FastDiff (Huang et al., IJCAI 2022)
- [21] Noise-level limited sub-modeling (Okamoto et al., ICASSP 2021)
- [22] BDDM (Lam et al., ICLR 2022)
- [23] ConvNeXt (Liu et al., CVPR 2022)
- [3] HiFi-GAN (Kong et al., NeurIPS 2020)

## 8. 公開済み参考実装 (論文 footnote)
- HiFi-GAN V1: https://github.com/kan-bayashi/ParallelWaveGAN
- WaveFit (非公式): https://github.com/yukara-ikemiya/wavefit-pytorch
- FastDiff: https://github.com/Rongjiehuang/FastDiff
- Vocos: https://github.com/gemelo-ai/vocos
