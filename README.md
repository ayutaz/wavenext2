# WaveNeXt 2 - PyTorch 再現実装

論文 **"WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders with Residual Denoising and Sub-Modeling for GAN and Diffusion Models"** (Zhou et al., arXiv:2605.25506v1, 2026) の非公式 PyTorch 再現実装プロジェクト。

- **論文**: https://arxiv.org/abs/2605.25506
- **デモ (公式)**: https://37integer.github.io/WAVENEXT-2
- **ステータス**: ドキュメント整備完了、実装着手前 (2026-05 時点)

## このリポジトリは何か

WaveNeXt 2 は ConvNeXt-based generator を **GAN ベースと Diffusion ベースの両方** に流用できる統一ボコーダフレームワーク。本リポジトリでは、論文 PDF と参考実装 (Vocos / WaveFit / FastDiff / BDDM / WaveNeXt) を突き合わせて再現実装に必要な情報を `docs/` にまとめ、PyTorch で M0〜M7 のマイルストーンに沿って実装を進める。

実装の核となるアイデア:
- 同じ ConvNeXt-based **sub-model** を、
  - **GAN-WaveNeXt 2**: T 個直列に並べて fixed-point iteration で訓練
  - **Diff-WaveNeXt 2**: 4 個用意し各 sub-model を異なる noise level range に特化させて独立訓練
- 統一を可能にする鍵は sub-model 入力側に置かれた **STFT module** (前ステップの波形を STFT 表現化して mel と結合)

## ドキュメント

実装着手前にまず読むべき順序:

| ファイル | 内容 |
|---|---|
| [`docs/paper-summary.md`](docs/paper-summary.md) | 論文全体のサマリ (背景・貢献・結果) |
| [`docs/milestones.md`](docs/milestones.md) | **M0〜M7 の具体的マイルストーン**、各 deliverable・acceptance |
| [`docs/architecture.md`](docs/architecture.md) | 各部品の構造詳細 (generator, STFT module, sub-model, ConvNeXt block) |
| [`docs/training.md`](docs/training.md) | 訓練・推論手順の擬似コード、loss、noise schedule |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | ディレクトリ構成、YAML 設定雛形、リスク |
| [`docs/open-questions.md`](docs/open-questions.md) | 確定状況一覧、情報源と解決経路 |

## セットアップ (M0)

### Python 環境

```bash
# 推奨: Python 3.10+
pip install -r requirements.txt   # ※ M0.1 で作成予定
```

依存: PyTorch 2.x, torchaudio, librosa, soundfile, pymcd, pyworld 等。

### データセット

LibriTTS-R をユーザー自身でダウンロード:

1. https://www.openslr.org/141/ にアクセスしライセンス (CC BY 4.0) に同意
2. `train_clean_100.tar.gz`, `train_clean_360.tar.gz`, `test_clean.tar.gz` をダウンロード
3. 展開先のパスを `configs/*.yaml` の `data.root` に設定

LibriTTS-R は本リポジトリには **含まれない** (約 50 GB)。

### 訓練

実装完了後 (M2 / M3 以降):

```bash
# GAN-WaveNeXt 2
python -m src.train.train_gan --config configs/gan_wavenext2.yaml

# Diff-WaveNeXt 2 (4 sub-models を順次)
for k in 1 2 3 4; do
    python -m src.train.train_diff --config configs/diff_wavenext2.yaml --sub-model $k
done
```

### 推論

```bash
python -m src.inference.infer_gan --ckpt checkpoints/gan/best.pt --mel path/to/mel.npy --out output.wav
python -m src.inference.infer_diff --ckpt-dir checkpoints/diff/ --mel path/to/mel.npy --out output.wav
```

## 参考実装

論文 footnote (Section 4.1) で言及されている公式・準公式実装。**本リポジトリは これらをコピー流用せず**、論文記述に合わせて再構成する。

| 用途 | リポジトリ |
|---|---|
| HiFi-GAN, MR-STFT loss | https://github.com/kan-bayashi/ParallelWaveGAN |
| WaveFit (GAN 系訓練ループ) | https://github.com/yukara-ikemiya/wavefit-pytorch |
| FastDiff (Diffusion 系) | https://github.com/Rongjiehuang/FastDiff |
| Vocos (ConvNeXt block, STFT 分解) | https://github.com/gemelo-ai/vocos |
| WaveNeXt 非公式 PyTorch | https://github.com/wetdog/wavenext_pytorch |
| BDDM (reverse step) | https://github.com/tencent-ailab/bddm |

## 計算リソース見積もり

| モデル | 訓練時間 (A100 単体) | 備考 |
|---|---|---|
| GAN-WaveNeXt 2 | ~410 時間 | WaveFit と同等 |
| Diff-WaveNeXt 2 | ~32 時間 | 4 sub-model 合計 |

個人 GPU での全モデル本格訓練は非現実的。マイルストーン M5 までは smoke でアーキテクチャの正しさを担保し、本格訓練 (M6) は GPU クラスタ確保後に行う。

## 引用

本リポジトリを利用する場合は元論文を引用してください:

```bibtex
@article{zhou2026wavenext2,
  title={WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders with Residual Denoising and Sub-Modeling for GAN and Diffusion Models},
  author={Zhou, Wangzixi and Okamoto, Takuma and Ohtani, Yamato and Sakti, Sakriani and Kawai, Hisashi},
  journal={arXiv preprint arXiv:2605.25506},
  year={2026}
}
```

本リポジトリ自体を引用する場合は [`CITATION.cff`](CITATION.cff) を参照。

## ライセンス

本リポジトリのコード・ドキュメントは **Apache License 2.0** で公開する。詳細は [`LICENSE`](LICENSE) 参照。

論文・参考実装の各著作権はそれぞれの著者・組織に帰属する。本リポジトリは論文 PDF や参考実装のソースコードを **同梱しない**。

## 注意事項

- 本実装は **非公式** であり、論文の著者・所属組織 (NAIST / NICT) とは無関係です
- LibriTTS-R 等のデータセットライセンスは各データセット側に従ってください
- 主観評価 (MOS) を行う場合、被験者保護・倫理審査は各実施機関の規程に従うこと
