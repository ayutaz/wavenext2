# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリは何か

論文 **"WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders with Residual Denoising and Sub-Modeling for GAN and Diffusion Models"** (Zhou et al., arXiv:2605.25506v1) の **PyTorch 再現実装プロジェクト**。

- 論文 URL: https://arxiv.org/abs/2605.25506
- デモ: https://37integer.github.io/WAVENEXT-2

現時点ではコードは未着手で、論文内容を整理したドキュメント (`docs/`) のみが存在する。

## 重要ドキュメント (実装前に必読)

| ファイル | 内容 |
|---|---|
| `docs/paper-summary.md` | 論文全体のサマリ。背景・貢献・結果 |
| `docs/architecture.md` | WaveNeXt-based generator, STFT module, sub-model の構造詳細 |
| `docs/training.md` | GAN-WaveNeXt 2 / Diff-WaveNeXt 2 の訓練・推論手順、loss、noise schedule |
| `docs/implementation-plan.md` | 再現実装の進め方、ディレクトリ構成、設定ファイル雛形、リスク |
| `docs/open-questions.md` | 確定状況一覧 (全項目解決済み)、情報源、解決経路 |
| `docs/milestones.md` | M0〜M7 の具体的マイルストーン、各 deliverable・acceptance・依存関係 |

新しいタスクに着手する前に、最低限 `paper-summary.md` と `milestones.md` を読むこと。実装の具体的なステップは `milestones.md`、論文の事実確認は `open-questions.md`、構造の詳細は `architecture.md` を参照。

## プロジェクトの構造的要点

論文の核となるアイデアは、**同じ ConvNeXt-based sub-model 構造** を:
- **GAN-WaveNeXt 2** : T 個直列に並べて fixed-point iteration で訓練 (WaveFit ベースだが簡略化)
- **Diff-WaveNeXt 2** : 4 個用意し各 sub-model を異なる noise level range に特化させて独立訓練

の **2 通りに使い分ける** 統一フレームワークである、という点。

統一を可能にする鍵は sub-model 入力側に置かれた **STFT module** (前ステップの波形を STFT 表現化して mel と結合)。`docs/architecture.md` §3 参照。

## 推奨実装順序 (全工程 Claude Code が実装)

詳細なマイルストーンは `docs/milestones.md` (M0〜M7) 参照。要点:

1. **M0**: 環境整備 + LibriTTS-R 取得 (small)
2. **M1**: コア部品 (ConvNeXt block / STFT module / Mel / Generator / Noise embedding / Sub-model) と各ユニットテスト (large)
3. **M2**: GAN-WaveNeXt 2 (Discriminator, Loss, fixed-point training, smoke) (large)
4. **M3**: Diff-WaveNeXt 2 (4 sub-model, point-specialized partition, reverse sampling, post-filter, smoke) (large)
5. **M4**: 評価インフラ (MCD, log F0 RMSE, UTMOS, NISQA, RTF) (medium)
6. **M5**: 統合スモーク (1 epoch GAN + 1 epoch Diff 1 sub-model) (small)
7. **M6**: 本格訓練 (GAN 410h + Diff 32h on A100、Claude Code は起動・監視のみ)
8. **M7**: 主観評価 (任意、人間の評価者必須)

### ユーザー操作が必要な箇所 (それ以外は Claude Code 完結)
- **M0.3**: LibriTTS-R のライセンス同意とダウンロード認証 (またはダウンロード済みパスを Claude Code に伝える)
- **M6**: A100 GPU クラスタの確保・SSH 認証・課金設定
- **M7**: 評価者の手配と評価結果 CSV の提供

## 論文上で曖昧だった箇所 (調査済み、全て確定)

詳細は `docs/open-questions.md` 参照。主要な確定事項:

- **Diffusion 式**: DDPM 標準形 `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (PDF Fig 1b 画像確認)
- **BDDM noise schedule predictor**: 再現不要 (4 値固定スケジュールを直接使用)
- **GAN fixed-point 初期化**: `torch.zeros_like(x_gt)` (論文「initial input noise isn't required」)
- **Generator output head**: `Linear(512, n_fft+2)` → `Linear(n_fft+2, hop, bias=False)` → reshape → `clip(-1, 1)` (元 WaveNeXt poster + wetdog 実装)
- **STFT loss / Discriminator**: WaveFit-PT (`yukara-ikemiya/wavefit-pytorch`) 完全準拠。**MSD ×3 のみ、MPD なし**
- **Diff sub-model partition**: point-specialized 1-to-1 mapping (band 境界は隣接 schedule 点の中点、`docs/architecture.md` §5)
- **Diff conditioning 注入**: additive bias (FiLM ではない)、per-block 独立 `Linear(512, 512)` (FastDiff)
- **Mel 正規化**: `torch.log(clamp(mel, min=1e-5))`, slaney scale/norm, power=1.0
- **Audio 正規化**: Vocos sox `norm` (train: U(-6,-1) dB, val: -3 dB)
- **EMA**: 不使用 (Vocos / WaveFit-PT 共に未使用)

## 参考実装 (論文 footnote)

| 用途 | リポジトリ |
|---|---|
| HiFi-GAN, discriminator, STFT loss | https://github.com/kan-bayashi/ParallelWaveGAN |
| WaveFit (GAN 系の訓練ループ参考) | https://github.com/yukara-ikemiya/wavefit-pytorch |
| FastDiff (Diffusion 系の参考) | https://github.com/Rongjiehuang/FastDiff |
| Vocos (ConvNeXt block, STFT 分解) | https://github.com/gemelo-ai/vocos |

これらは **コピー流用しない**。論文記述に合うよう再構成する。

## データ

- LibriTTS-R "train-clean-100" + "train-clean-360" (24 kHz, 計約 460 時間)
- 評価: "test-clean-100" 4,824 utterances
- 入力: 128-dim log-mel-spectrogram
- hop size はモデルによって異なる: GAN は 300 (HiFi-GAN/WaveFit に揃える)、Diff は 256 (FastDiff に揃える)

## 計算リソース見積もり

- GAN-WaveNeXt 2: A100 単体で約 410 時間
- Diff-WaveNeXt 2: A100 単体で約 32 時間 (4 sub-model 合計)

個人 GPU での全モデル本格訓練は非現実的。Phase 1〜3 ではスモークテストでアーキテクチャの正しさを担保するに留め、本格訓練は GPU クラスタ確保後に行うこと。

## 論文テキストの取得

論文 PDF はリポジトリには含まれていない。必要な場合は arXiv から取得して使う:

```bash
# PDF 取得
curl -L -o paper.pdf https://arxiv.org/pdf/2605.25506
# テキスト抽出 (pdftotext が使える環境の場合)
pdftotext -layout paper.pdf paper.txt
# 画像化 (PyMuPDF を使う場合、Section 3 等の数式確認用)
python -c "import fitz; doc = fitz.open('paper.pdf'); [page.get_pixmap(dpi=200).save(f'page_{i+1}.png') for i, page in enumerate(doc)]"
```

取得した PDF / 画像 / 抽出テキストは **リポジトリには commit しない** (`.gitignore` で除外、本リポジトリは公開予定のため著作物を同梱しない)。
