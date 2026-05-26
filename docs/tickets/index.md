# チケット一覧

`docs/milestones.md` の各サブタスクに対する個別チケット。1 チケット = 1 マイルストーン・サブタスク。

**凡例**:
- ステータス: `📝 pending` / `🚧 in_progress` / `🔍 in_review` / `✅ completed` / `⛔ blocked`
- サイズ: `S` (small) / `M` (medium) / `L` (large)

## 進捗サマリ

| マイルストーン | チケット数 | pending | in_progress | in_review | completed | フェーズレビュー |
|---|---|---|---|---|---|---|
| M0 | 3 | 3 | 0 | 0 | 0 | 未実施 |
| M1 | 6 | 6 | 0 | 0 | 0 | 未実施 |
| M2 | 6 | 6 | 0 | 0 | 0 | 未実施 |
| M3 | 5 | 5 | 0 | 0 | 0 | 未実施 |
| M4 | 3 | 3 | 0 | 0 | 0 | 未実施 |
| M5 | 2 | 2 | 0 | 0 | 0 | 未実施 |
| M6 | 3 | 3 | 0 | 0 | 0 | 未実施 |
| M7 | 1 | 1 | 0 | 0 | 0 | 未実施 |
| **計** | **29** | **29** | **0** | **0** | **0** | — |

> **注**: チケット数が milestones.md のサブタスク総数 (28) と +1 ずれる場合があるのは、`M0` を `M0.1 / M0.2 / M0.3` の 3 チケットに分割しているため、または将来サブタスクが追加されたためです。最新は下表で確認。

## M0: 環境整備とデータ準備

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M0.1 | [Python 環境](T-M0.1-python-env.md) | S | 📝 | — | — |
| T-M0.2 | [ディレクトリ scaffold](T-M0.2-scaffold.md) | S | 📝 | — | — |
| T-M0.3 | [LibriTTS-R 取得](T-M0.3-libritts-r.md) | S | 📝 | T-M0.1, T-M0.2 | — |

## M1: コア部品 (sub-model 構成要素)

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M1.1 | [ConvNeXt block](T-M1.1-convnext-block.md) | M | 📝 | T-M0.2 | — |
| T-M1.2 | [STFT module](T-M1.2-stft-module.md) | M | 📝 | T-M0.2 | — |
| T-M1.3 | [Mel-spectrogram 抽出](T-M1.3-mel-spectrogram.md) | S | 📝 | T-M0.2 | — |
| T-M1.4 | [Generator](T-M1.4-generator.md) | M | 📝 | T-M1.1 | — |
| T-M1.5 | [Noise embedding (Diff)](T-M1.5-noise-embedding.md) | S | 📝 | T-M0.2 | — |
| T-M1.6 | [Sub-model wrapper](T-M1.6-sub-model.md) | M | 📝 | T-M1.2, T-M1.3, T-M1.4, T-M1.5 | — |

## M2: GAN-WaveNeXt 2

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M2.1 | [Dataset](T-M2.1-dataset.md) | M | 📝 | T-M0.3, T-M1.3 | — |
| T-M2.2 | [Discriminator (MSD×3)](T-M2.2-discriminator.md) | M | 📝 | T-M0.2 | — |
| T-M2.3 | [Loss 関数](T-M2.3-losses.md) | M | 📝 | T-M0.2 | — |
| T-M2.4 | [GAN モデル](T-M2.4-gan-model.md) | M | 📝 | T-M1.6 | — |
| T-M2.5 | [Training script](T-M2.5-train-gan.md) | L | 📝 | T-M2.1, T-M2.2, T-M2.3, T-M2.4 | — |
| T-M2.6 | [Smoke training](T-M2.6-gan-smoke.md) | S | 📝 | T-M2.5 | — |

## M3: Diff-WaveNeXt 2

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M3.1 | [Diff モデル](T-M3.1-diff-model.md) | M | 📝 | T-M1.6 | — |
| T-M3.2 | [Training script](T-M3.2-train-diff.md) | L | 📝 | T-M2.1, T-M3.1 | — |
| T-M3.3 | [Reverse sampler](T-M3.3-reverse-sampler.md) | M | 📝 | T-M3.1 | — |
| T-M3.4 | [Post-filter](T-M3.4-post-filter.md) | M | 📝 | T-M3.3 | — |
| T-M3.5 | [Smoke training](T-M3.5-diff-smoke.md) | S | 📝 | T-M3.2, T-M3.3 | — |

## M4: 評価インフラ

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M4.1 | [客観評価 (MCD / logF0RMSE)](T-M4.1-objective-metrics.md) | M | 📝 | T-M0.1 | — |
| T-M4.2 | [UTMOS / NISQA 連携](T-M4.2-utmos-nisqa.md) | M | 📝 | T-M0.1 | — |
| T-M4.3 | [RTF 測定](T-M4.3-rtf.md) | S | 📝 | T-M2.4, T-M3.1 | — |

## M5: 統合スモークテスト

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M5.1 | [GAN 1 epoch 訓練](T-M5.1-gan-1epoch.md) | S | 📝 | T-M2.6, T-M4.* | — |
| T-M5.2 | [Diff 1 sub-model 1 epoch](T-M5.2-diff-1epoch.md) | S | 📝 | T-M3.5, T-M4.* | — |

## M6: 本格訓練 (要 GPU)

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M6.1 | [GAN フル訓練 (410h)](T-M6.1-gan-full-training.md) | L | 📝 | T-M5.1 | — |
| T-M6.2 | [Diff 4 sub-model 訓練 (32h)](T-M6.2-diff-full-training.md) | L | 📝 | T-M5.2 | — |
| T-M6.3 | [Ablation 比較 (任意)](T-M6.3-ablation.md) | L | 📝 | T-M6.1, T-M6.2 | — |

## M7: 主観評価 (任意)

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M7.1 | [内部 MOS テスト](T-M7.1-mos-test.md) | M | 📝 | T-M6.1, T-M6.2 | — |

## フェーズ完了後のレビューログ

「ゼロから作り直すとしたら」の §8 セクションを各フェーズ完了時にエージェントチームで再評価する。実施記録は以下:

| マイルストーン | 完了日 | レビュー実施日 | 主な更新点 |
|---|---|---|---|
| M0 | — | — | — |
| M1 | — | — | — |
| M2 | — | — | — |
| M3 | — | — | — |
| M4 | — | — | — |
| M5 | — | — | — |
| M6 | — | — | — |
| M7 | — | — | — |
