# WaveNeXt 2 - PyTorch 再現実装

論文 **"WaveNeXt 2: ConvNeXt-Based Fast Neural Vocoders with Residual Denoising and Sub-Modeling for GAN and Diffusion Models"** (Zhou et al., arXiv:2605.25506v1, 2026) の非公式 PyTorch 再現実装プロジェクト。

- **論文**: https://arxiv.org/abs/2605.25506
- **デモ (公式)**: https://37integer.github.io/WAVENEXT-2
- **ステータス** (2026-05-28): M0〜M4 実装完了 + M5 (統合スモーク) 足場実装済み。M5 実行・M6 本格訓練・M7 主観評価・LibriTTS-R 取得 (T-M0.3) は GPU / データ認証を要するため未実施 (進捗は [`docs/tickets/index.md`](docs/tickets/index.md))

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

本プロジェクトは [uv](https://docs.astral.sh/uv/) で依存・実行環境を管理する。Python は **3.13** を採用 (論文に Python バージョン指定なし、依存ライブラリ調査の結果 3.13 が全公式 wheel 完備の最新版)。

```bash
# uv インストール (未導入時)
#   Windows: winget install --id=astral-sh.uv -e
#   macOS / Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存セットアップ (pyproject.toml + uv.lock から復元、M0.1 で作成)
uv sync

# バージョン確認
uv run python --version    # Python 3.13.x
```

依存: PyTorch >= 2.10 (Python 3.13 対応版), torchaudio, librosa, soundfile, pyworld 等 (MCD は外部依存を使わず librosa MFCC + 自前 DTW で実装)。`requires-python` は `>=3.13.13,<3.14`。

テスト (CPU、`slow` / `gpu` マーカーは既定で除外):

```bash
uv run pytest -q
```

### データセット

LibriTTS-R をユーザー自身でダウンロード:

1. https://www.openslr.org/141/ にアクセスしライセンス (CC BY 4.0) に同意
2. `train_clean_100.tar.gz`, `train_clean_360.tar.gz`, `test_clean.tar.gz` をダウンロード
3. 展開先のパスを `configs/*.yaml` の `data.root` / `data.root_dir` に設定
4. filelist を生成: `uv run python scripts/prepare_libritts.py --src-dir <LibriTTS-R 展開ルート>`

LibriTTS-R は本リポジトリには **含まれない** (約 50 GB)。

### 訓練

`pyproject.toml` の `[project.scripts]` (`train-gan` / `train-diff`) または `python -m wavenext2.train.*` で起動する (本格訓練は GPU 必須、M6)。

```bash
# GAN-WaveNeXt 2
uv run train-gan --config configs/gan_wavenext2.yaml

# Diff-WaveNeXt 2 (4 sub-models を順次)
for k in 1 2 3 4; do
    uv run train-diff --config configs/diff_wavenext2.yaml --sub-model $k
done
```

### 推論

推論用の CLI は未実装 (`src/wavenext2/inference/infer_gan.py` は scaffold のスタブ)。現状は Python API から呼ぶ:

- GAN: `wavenext2.models.gan_wavenext2.GANWaveNext2.synthesize(mel)` (zeros 初期化 + T 回 fixed-point iteration)
- Diff: `wavenext2.inference.infer_diff.reverse_sample(model, mel, ...)` (4 sub-model の reverse sampling)

学習済み checkpoint の客観評価 (MCD / log F0 RMSE) は `scripts/eval_gan_checkpoint.py` / `scripts/eval_diff_checkpoint.py` を使う:

```bash
uv run python scripts/eval_gan_checkpoint.py --config configs/gan_wavenext2_1epoch.yaml \
    --ckpt checkpoints/gan_1epoch/best.pt --out eval_results/gan_1epoch.json
uv run python scripts/eval_diff_checkpoint.py --config configs/diff_wavenext2_1epoch.yaml \
    --ckpt-dir checkpoints/diff_1epoch --out eval_results/diff_1epoch.json
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
