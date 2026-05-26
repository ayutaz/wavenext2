# チケット一覧

`docs/milestones.md` の各サブタスクに対する個別チケット。1 チケット = 1 マイルストーン・サブタスク。

**凡例**:
- ステータス: `📝 pending` / `🚧 in_progress` / `🔍 in_review` / `✅ completed` / `⛔ blocked`
- サイズ: `S` (small) / `M` (medium) / `L` (large)

## 進捗サマリ

| マイルストーン | チケット数 | pending | in_progress | in_review | completed | フェーズレビュー |
|---|---|---|---|---|---|---|
| M0 | 3 | 2 | 0 | 0 | 1 | ✅ 2026-05-26 |
| M1 | 6 | 6 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M2 | 6 | 6 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M3 | 5 | 5 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M4 | 3 | 3 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M5 | 2 | 2 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M6 | 3 | 3 | 0 | 0 | 0 | ✅ 2026-05-27 |
| M7 | 1 | 1 | 0 | 0 | 0 | ✅ 2026-05-27 |
| **計** | **29** | **28** | **0** | **0** | **1** | 全 8 フェーズ完了 |

> **チケット作成ステータス**: 全 29 チケットが作成済み + フェーズレビュー (architect / ML / DevOps の 3 視点) 完了。`status` 列の `pending` は **実装着手前** を意味し、チケット文書自体は完成している。

> **注**: チケット数が milestones.md のサブタスク総数 (28) と +1 ずれる場合があるのは、`M0` を `M0.1 / M0.2 / M0.3` の 3 チケットに分割しているため、または将来サブタスクが追加されたためです。最新は下表で確認。

## M0: 環境整備とデータ準備

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M0.1 | [Python 環境](T-M0.1-python-env.md) | S | ✅ | — | claude |
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

> **実装着手前に解決した CRITICAL (2026-05-27)**:
> 1. **Diff reverse step の β 負値** — 論文 schedule は denoising 順で ᾱ 増加列のため `β[1]=1-280=-279` で発散。`docs/training.md` §4.2 で **β 不使用の x_0 予測経由 DDIM/DDPM 一般形** (`σ²=η²·(1-ᾱ_next)/(1-ᾱ_t)·(1-ᾱ_t/ᾱ_next)`, η で stochastic 切替) に確定。T-M3.3 §2 コードも β-free に更新済み (✅)。
> 2. **RTF/param の T 取り違え** — `paper-summary.md` の 0.0066/0.20/74.93M は **5 iter (T=5)** 値と確定 (74.93M=5×14.99M)。品質 (MOS) は 4 iter で HiFi-GAN 同等。`milestones.md` L488 / T-M6.1 §6.1 を T=5 基準に修正済み (✅)。

| マイルストーン | 完了日 | レビュー実施日 | 主な更新点 |
|---|---|---|---|
| M0 | — | 2026-05-26 | `src/wavenext2/` パッケージ化採用 (T-M0.2)、scaffold で Docker/CI/pre-commit/`.gitattributes`/`.env.example` stub を commit (T-M0.2)、filelist を TSV header 付き化 + speaker-balanced val + `dev_postfilter.tsv` 同時生成 + stats.json/audio_info.tsv 出力 (T-M0.3)、再現性関連リスク 6 件追加 (T-M0.1)、Critical: docs/training.md の `test-clean-100` 表記揺れを T-M0.3 実装時に解決予定 |
| M1 | — | 2026-05-26 | T-M1.1 `forward(x, *, cond=None)` keyword-only 確定 + `fc_t.bias=0` zero init、T-M1.2 dynamic range mismatch (critical) + iSTFT 予約、T-M1.3 `eps=1e-7` Vocos 互換 (暫定) + precompute 設計、T-M1.4 `enable_grad_ckpt` / `final_activation` / `block_factory` 引数化 + `linear_2` clip 飽和懸念、T-M1.5 `c * 1000` rescale (M3 smoke 最優先 ablation) + `fc1.bias=0` 検討、T-M1.6 **Critical** 戻り値仕様 (n_t vs y_{t-1}) を実装時に再確認 + `from_config()` factory 採用、横断: `tests/conftest.py` fixture 戦略を T-M0.2 へ申し送り |
| M2 | — | 2026-05-26 | T-M2.1 `Batch` TypedDict + `worker_init_fn` snippet + `MelOnGPU` 予約、T-M2.2 `DiscriminatorOutput` NamedTuple + defensive `unsqueeze(1)` + API 非依存 weight_norm test、T-M2.3 `compute_total_loss` を `(total, unweighted_dict)` tuple 化 + `sorted(losses.keys())` 順序固定 + MR-STFT を M3/M4 で再利用、T-M2.4 **`return_intermediates=True` を v1 から導入** + atomic `best.pt` rename + `BaseVocoder` 抽出を M3.1 着手前判断、T-M2.5 **`train_gan_step` 公開関数化** + SIGTERM/SIGINT handler + 決定論性 env、T-M2.6 pytest single-source + synthetic CPU smoke を nightly CI |
| M3 | — | 2026-05-26 | **T-M3.3 CRITICAL: β_t 負値問題発見** (`β[1]=1-2.8e-2/1e-4=-279` → `1/√(1-β)` で NaN 必発、論文 4 点 schedule は連続 DDPM のサブサンプリングで隣接 β 無意味 → α_t + skip-aware σ で書き換え必須)、T-M3.1 `from_config(only_sub_model=k)` lazy 化 + `synthesize` alias、T-M3.2 bf16 で c 精度 fp32 強制 + 3-point validation + `c_rescale` 切替、T-M3.4 `apply_post_filter` を `reverse_sample` 引数統合 + torch/numpy 両受け + `fir.npy` commit 化、T-M3.5 **sub-model 4 を primary smoke に昇格** (conditioning 感度 ~70 倍) + mock reverse で β 問題早期検知 |
| M4 | — | 2026-05-26 | **統一 eval facade `eval/runner.py::evaluate()` を T-M4.1 に新設** (3 指標を 1 entry point、`EvalResult` + `eval_results/*.json` 永続化)、3 チケット戻り値 schema 統一 (`{metric}_mean/std`, `n`, `n_skipped`)、T-M4.1 相対比較主軸 + pymcd `adv_dtw` mode、T-M4.2 speechmos default + fairseq opt-in 降格 + GT UTMOS≥3.8 緩和、T-M4.3 CUDA events 測定 + two-tier CI + thread 復元。**cross-ticket: GAN 側 `synthesize(mel)` alias が T-M2.4 に必要 (T-M4.3 §9 申し送り、実装時に対応)** |
| M5 | — | 2026-05-26 | **gate を品質でなく divergence gate に** (1 epoch=33k step では未達定常、loss 方向条件 + finite + NaN なしで判定)、`EvalResult.gate_passed` 二層構造 (自動 + 聴感人間)、共通 `run_smoke.py --mode {gan,diff}` orchestrator、**gate 判定者 = user GO/NO-GO** (M6 課金前)、resume 完全性 (optim/sched/step/RNG)、T-M5.2 **mock reverse では β 負値検出に不十分 → sub-model 1 を加え実 eps_pred で 2-step reverse**、cos<0.5 期待値 (0.99 は最低線)、bf16 fp32 強制を gate 前提に格上げ |
| M6 | — | 2026-05-27 | **T-M6.1 CRITICAL: RTF 参照値の T 取り違え** (paper-summary L89/L113 の 0.0066/0.20/74.93M は 5 iter の値、T=4 acceptance にすると誤判定 → milestones L548/paper-summary 矛盾解消を申し送り)、共通 `scripts/orchestrate.py` + `eval/report.py` 一本化 (3 回再発明回避)、cloud run dir を resume 単一情報源、SWA 採用昇格、cloud provider コスト比較 (on-demand $40/h vs spot $0.6/h)、uv.lock cross-platform 検証、T-M6.2 post-filter fit は reverse 安定が前提 (β 連結) + 4 sub-model spot 分散、T-M6.3 T sweep step 非対称の偽 trend (wall-clock 等価で揃える) + sub-modeling 優先 |
| M7 | — | 2026-05-27 | **統計検定の誤用発見** (paired 設計に Mann-Whitney は誤用 → Wilcoxon signed-rank が正、t-test は論文対比補助)、**LibriTTS-R CC BY 4.0 帰属表示義務** (評価 web app で合成音声=改変物を配信、attribution 必須)、少人数では CMOS 主軸 (ACR 従)、bootstrap (BCa) CI + Spearman 相関 + ICC/α 信頼性、Holm 多重比較補正、webMUSHRA 全面委譲、`mos_results/` を個人データとして別格扱い (data minimization) |
