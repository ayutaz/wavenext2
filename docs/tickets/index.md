# チケット一覧

`docs/milestones.md` の各サブタスクに対する個別チケット。1 チケット = 1 マイルストーン・サブタスク。

**凡例**:
- ステータス: `📝 pending` / `🚧 in_progress` / `🔍 in_review` / `✅ completed` / `⛔ blocked`
- サイズ: `S` (small) / `M` (medium) / `L` (large)

## 進捗サマリ

| マイルストーン | チケット数 | pending | in_progress | in_review | completed | フェーズレビュー |
|---|---|---|---|---|---|---|
| M0 | 3 | 0 | 0 | 1 | 2 | ✅ 2026-05-26 |
| M1 | 6 | 0 | 0 | 0 | 6 | ✅ 2026-05-26 |
| M2 | 6 | 0 | 0 | 0 | 6 | ✅ 2026-05-26 |
| M3 | 5 | 3 | 0 | 0 | 2 | ✅ 2026-05-26 |
| M4 | 3 | 3 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M5 | 2 | 2 | 0 | 0 | 0 | ✅ 2026-05-26 |
| M6 | 3 | 3 | 0 | 0 | 0 | ✅ 2026-05-27 |
| M7 | 1 | 1 | 0 | 0 | 0 | ✅ 2026-05-27 |
| **計** | **29** | **12** | **0** | **1** | **16** | 全 8 フェーズ完了 |

> **チケット作成ステータス**: 全 29 チケットが作成済み + フェーズレビュー (architect / ML / DevOps の 3 視点) 完了。`status` 列の `pending` は **実装着手前** を意味し、チケット文書自体は完成している。

> **注**: チケット数が milestones.md のサブタスク総数 (28) と +1 ずれる場合があるのは、`M0` を `M0.1 / M0.2 / M0.3` の 3 チケットに分割しているため、または将来サブタスクが追加されたためです。最新は下表で確認。

## M0: 環境整備とデータ準備

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M0.1 | [Python 環境](T-M0.1-python-env.md) | S | ✅ | — | claude |
| T-M0.2 | [ディレクトリ scaffold](T-M0.2-scaffold.md) | S | ✅ | — | claude |
| T-M0.3 | [LibriTTS-R 取得](T-M0.3-libritts-r.md) | S | 🔍 | T-M0.1, T-M0.2 | claude |

## M1: コア部品 (sub-model 構成要素)

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M1.1 | [ConvNeXt block](T-M1.1-convnext-block.md) | M | ✅ | T-M0.2 | claude |
| T-M1.2 | [STFT module](T-M1.2-stft-module.md) | M | ✅ | T-M0.2 | claude |
| T-M1.3 | [Mel-spectrogram 抽出](T-M1.3-mel-spectrogram.md) | S | ✅ | T-M0.2 | claude |
| T-M1.4 | [Generator](T-M1.4-generator.md) | M | ✅ | T-M1.1 | claude |
| T-M1.5 | [Noise embedding (Diff)](T-M1.5-noise-embedding.md) | S | ✅ | T-M0.2 | claude |
| T-M1.6 | [Sub-model wrapper](T-M1.6-sub-model.md) | M | ✅ | T-M1.2, T-M1.3, T-M1.4, T-M1.5 | claude |

## M2: GAN-WaveNeXt 2

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M2.1 | [Dataset](T-M2.1-dataset.md) | M | ✅ | T-M0.3, T-M1.3 | claude |
| T-M2.2 | [Discriminator (MSD×3)](T-M2.2-discriminator.md) | M | ✅ | T-M0.2 | claude |
| T-M2.3 | [Loss 関数](T-M2.3-losses.md) | M | ✅ | T-M0.2 | claude |
| T-M2.4 | [GAN モデル](T-M2.4-gan-model.md) | M | ✅ | T-M1.6 | claude |
| T-M2.5 | [Training script](T-M2.5-train-gan.md) | L | ✅ | T-M2.1, T-M2.2, T-M2.3, T-M2.4 | claude |
| T-M2.6 | [Smoke training](T-M2.6-gan-smoke.md) | S | ✅ | T-M2.5 | claude |

## M3: Diff-WaveNeXt 2

| ID | チケット | サイズ | ステータス | 依存 | 担当 |
|---|---|---|---|---|---|
| T-M3.1 | [Diff モデル](T-M3.1-diff-model.md) | M | ✅ | T-M1.6 | claude |
| T-M3.2 | [Training script](T-M3.2-train-diff.md) | L | 📝 | T-M2.1, T-M3.1 | — |
| T-M3.3 | [Reverse sampler](T-M3.3-reverse-sampler.md) | M | ✅ | T-M3.1 | claude |
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
| M0 | 2026-05-27 (impl) | 2026-05-26 (ticket) / 2026-05-27 (impl) | **[ticket review]** `src/wavenext2/` パッケージ化採用 (T-M0.2)、scaffold で Docker/CI/pre-commit/`.gitattributes`/`.env.example` stub を commit (T-M0.2)、filelist を TSV header 付き化 + speaker-balanced val + `dev_postfilter.tsv` 同時生成 + stats.json/audio_info.tsv 出力 (T-M0.3)、再現性関連リスク 6 件追加 (T-M0.1)、~~Critical: `test-clean-100` 表記揺れ~~ → ✅ 解決 (2026-05-27): 実装は openslr `test-clean` で確定、training.md に注記。<br>**[impl review 2026-05-27]** 3 視点 agent review 実施。(1) **torchaudio 2.11 I/O sunset** → soundfile-first を §8.1 に昇格・open-questions C4 を peak-norm に更新 (sox `norm` と数値等価検証済)。(2) **pymcd 除外** (Windows cp313 ビルド不可) → MCD は T-M4.1 で `mel-cepstral-distance` or 自前実装。(3) **Python 3.13.13 下限** (3.13.8 は torch import 不能)。(4) val 選択 (最長 utt) に median 代替案 H 追加 (M4 で再評価)。(5) **M1 申し送り**: min-duration 1.0s < Diff segment 1.067s → 反射 pad 必須 (T-M2.1/M3)、CI は cu128 を Linux DL・GPU テストは skip 前提。agent の「lint.yml 不在」指摘は誤検出 (実在・追跡済を検証)。 |
| M1 | 2026-05-27 (impl) | 2026-05-26 (ticket) / 2026-05-27 (impl) | **[ticket review]** T-M1.1 `forward(x, *, cond=None)` keyword-only 確定、T-M1.2 dynamic range mismatch、T-M1.3 eps、T-M1.4 引数化、T-M1.5 `c*1000` rescale、T-M1.6 戻り値仕様 + factory。<br>**[impl review 2026-05-27]** 6 チケット実装完了 (全 unit テスト pass)。3 視点 agent review 実施。**重大発見**: (1) **embed kernel=1 に確定** — kernel=7 over 2176 は 22M で Table 1 14.99M と矛盾 (2iter=29.97M=2× の厳密倍数で棄却)、kernel=1 で GAN 15.43M=+2.9% (architecture.md L34 修正)。(2) **論文 §3.1/§3.3 は mel/STFT 結合方法と conditioning 機構を未記載** (concat も per-block fc_t も FastDiff/Vocos 推定) → open-questions §C7 を「推定」に格下げ。(3) **per-block fc_t (2.1M) で Diff sub-model が Table 1 14.42M を +14% 超過**、fc_t 無しなら一致 → ✅ **解決 (2026-05-27, user 委譲のエージェントチーム調査)**: 3 体 agent (FastDiff/OSS 機構・Table 1 連立復元・Okamoto 系列慣行) を並行調査・議論。**per-block fc_t を撤去**し共有 NoiseEmbedding を射影なしで各 block に additive 注入 (sub-model=14.354M, −0.46%)。Okamoto21 (DiffWave 流 per-block additive) の骨格に一致。convnext.py/generator.py/noise_embedding.py + 全 docs (§C7/§5.4/CLAUDE.md) 更新済、185 tests pass。M3 着手のブロッカー解消。(4) Diff の ε は clip 不可 → generator `final_activation="none"` 追加。(5) SubModelGAN→n_t (clip)、減算は T-M2.4。<br>**[M2 申し送り]** fixed-point は `y = y - sub(mel,y)` (減算)、zeros 初期化は T-M2.4、grad ckpt stub 未実装 (OOM 時要実装)、mel/STFT grad balance を smoke で計測。 |
| M2 | 2026-05-28 (impl) | 2026-05-26 (ticket) / 2026-05-28 (impl) | **[ticket review]** T-M2.1 `Batch` TypedDict + `worker_init_fn` snippet、T-M2.2 NamedTuple + defensive `unsqueeze(1)`、T-M2.3 `compute_total_loss` tuple 化、T-M2.4 `return_intermediates` v1 導入 + `BaseVocoder` 抽出を M3.1 着手前判断、T-M2.5 `train_gan_step` 公開関数化、T-M2.6 pytest single-source。<br>**[impl review 2026-05-28]** 6 チケット実装完了 (230 passed + synthetic smoke `-m slow` pass)。3 視点 agent review 実施。**ML 正当性**: fixed-point 符号/初期化・hinge 符号・autocast 境界・InverseLR・loss 重みは docs と一致、重大バグなし。**🔴 実バグ修正**: 本番 yaml `loss.mrstft.fft_sizes`→`n_ffts` (`MultiResolutionSTFTLoss(**cfg)` が TypeError だった)。**重大発見 (記録済)**: 出力 head `clip(-1,1)` の飽和域勾配 0 による高 lr 学習凍結 (G 単独でも再現)、実訓練は lr 1e-4+warmup で安全側・発散時 tanh fallback。**M3 着手前の確定設計判断**: (1) **BaseVocoder/IterativeVocoder は抽出しない (GAN=反復子 vs Diff=ステートレス dispatch で 2 並列実装)** — T-M2.4 §8.3 確定。(2) **`train_gan_step`↔`train_diff_step` の signature 統一は不可能** (共通は「dict 返却 step 関数」原則のみ、fixture も別建て) — T-M2.5 §8.1 記述を T-M3.2 着手前に訂正。(3) **`utils/training_loop.py` 共通化 (checkpoint/iter_forever/validation) を M3.2 着手前に実施** (未実施)。(4) **config nested→flat マッピング**: `hop`/`hop_length`・`n_mels`/`mel_channels` が既定値一致で偶然動作 → M3.2/T-M5.1 着手前に yaml キー名統一 or from_config rename マップ集約。(5) **GAN embed +2.9% は許容確定** (別経路 embed 不採用、open-questions §C7)。**M3 ticket 取りこぼし**: T-M3.1/T-M3.2 擬似コードが `from_config(cfg, mode="diff")` を残置 (実装は `mode=` なし)、着手前に削除。**テスト欠落 (M3.2 前 or T-M5.1 で補完)**: `keep_last_n` rolling delete 未実装 (M6 disk full リスク)、config-drift 検知未実装、resume RNG 再現性/run_validation/SIGTERM emergency save 未テスト (synthetic で追加可)。 |
| M3 | — | 2026-05-26 | **T-M3.3 CRITICAL: β_t 負値問題発見** (`β[1]=1-2.8e-2/1e-4=-279` → `1/√(1-β)` で NaN 必発、論文 4 点 schedule は連続 DDPM のサブサンプリングで隣接 β 無意味 → α_t + skip-aware σ で書き換え必須)、T-M3.1 `from_config(only_sub_model=k)` lazy 化 + `synthesize` alias、T-M3.2 bf16 で c 精度 fp32 強制 + 3-point validation + `c_rescale` 切替、T-M3.4 `apply_post_filter` を `reverse_sample` 引数統合 + torch/numpy 両受け + `fir.npy` commit 化、T-M3.5 **sub-model 4 を primary smoke に昇格** (conditioning 感度 ~70 倍) + mock reverse で β 問題早期検知 |
| M4 | — | 2026-05-26 | **統一 eval facade `eval/runner.py::evaluate()` を T-M4.1 に新設** (3 指標を 1 entry point、`EvalResult` + `eval_results/*.json` 永続化)、3 チケット戻り値 schema 統一 (`{metric}_mean/std`, `n`, `n_skipped`)、T-M4.1 相対比較主軸 + pymcd `adv_dtw` mode、T-M4.2 speechmos default + fairseq opt-in 降格 + GT UTMOS≥3.8 緩和、T-M4.3 CUDA events 測定 + two-tier CI + thread 復元。**cross-ticket: GAN 側 `synthesize(mel)` alias が T-M2.4 に必要 (T-M4.3 §9 申し送り、実装時に対応)** |
| M5 | — | 2026-05-26 | **gate を品質でなく divergence gate に** (1 epoch=33k step では未達定常、loss 方向条件 + finite + NaN なしで判定)、`EvalResult.gate_passed` 二層構造 (自動 + 聴感人間)、共通 `run_smoke.py --mode {gan,diff}` orchestrator、**gate 判定者 = user GO/NO-GO** (M6 課金前)、resume 完全性 (optim/sched/step/RNG)、T-M5.2 **mock reverse では β 負値検出に不十分 → sub-model 1 を加え実 eps_pred で 2-step reverse**、cos<0.5 期待値 (0.99 は最低線)、bf16 fp32 強制を gate 前提に格上げ |
| M6 | — | 2026-05-27 | **T-M6.1 CRITICAL: RTF 参照値の T 取り違え** (paper-summary L89/L113 の 0.0066/0.20/74.93M は 5 iter の値、T=4 acceptance にすると誤判定 → milestones L548/paper-summary 矛盾解消を申し送り)、共通 `scripts/orchestrate.py` + `eval/report.py` 一本化 (3 回再発明回避)、cloud run dir を resume 単一情報源、SWA 採用昇格、cloud provider コスト比較 (on-demand $40/h vs spot $0.6/h)、uv.lock cross-platform 検証、T-M6.2 post-filter fit は reverse 安定が前提 (β 連結) + 4 sub-model spot 分散、T-M6.3 T sweep step 非対称の偽 trend (wall-clock 等価で揃える) + sub-modeling 優先 |
| M7 | — | 2026-05-27 | **統計検定の誤用発見** (paired 設計に Mann-Whitney は誤用 → Wilcoxon signed-rank が正、t-test は論文対比補助)、**LibriTTS-R CC BY 4.0 帰属表示義務** (評価 web app で合成音声=改変物を配信、attribution 必須)、少人数では CMOS 主軸 (ACR 従)、bootstrap (BCa) CI + Spearman 相関 + ICC/α 信頼性、Holm 多重比較補正、webMUSHRA 全面委譲、`mos_results/` を個人データとして別格扱い (data minimization) |
