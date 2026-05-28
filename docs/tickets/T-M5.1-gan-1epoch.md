---
id: T-M5.1
title: GAN 統合スモーク (train-clean-100 1 epoch)
milestone: M5
phase: M5
status: in_review
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-28
depends_on: [T-M2.6]
blocks: [T-M6.1]
related_docs:
  - docs/milestones.md#m51-1-epoch-訓練
  - docs/training.md
---

# T-M5.1: GAN 統合スモーク (train-clean-100 1 epoch)

> **マイルストーン**: [M5](../milestones.md#m5-統合スモークテスト-作業量-smallwall-clock-は-gpu-数時間) / **サブタスク**: [M5.1](../milestones.md#m51-1-epoch-訓練)
> **依存**: [T-M2.6](T-M2.6-gan-smoke.md) (前提: [T-M2.5](T-M2.5-train-gan.md), [T-M4.1](T-M4.1-objective-metrics.md)) / **後続**: [T-M6.1](T-M6.1-gan-full-training.md)

## 1. タスク目的とゴール

### 目的
T-M2.6 の 1 sample overfit (architectural sanity) を通過した GAN-WaveNeXt 2 を、**実データ train-clean-100 (約 33k step = 1 epoch) で訓練**して、複数話者・実音声に対して **汎化的に収束するか** を最小コストで検証する。本チケットは M6 本格訓練 (410h × A100) に進む前の **最後の gate** であり、ここで「validation MR-STFT が十分下がる」「聴感破綻なし」を確認できなければ M6 を起動しない。

T-M2.6 が見られなかった「収束性」「汎化」「validation 品質」を、**新規実装ゼロ** (既存 `train_gan.py` をそのまま起動) で確認する。同時に M5 phase review として、M1〜M2 で先送りした再評価項目 (eps=1e-7 vs 1e-5 / speaker-balanced sampler / segment_length / D 強すぎ) を実測データで判断する。

### ゴール
- [ ] 既存 `train_gan.py` を `configs/gan_wavenext2_1epoch.yaml` (max_steps≈33k, validation on) で起動し **OOM なし完走**
- [ ] 1 epoch 後の generator が validation utterances に対して **MR-STFT (sc+mag total) < 初期値の 30%**
- [ ] 生成音声の聴感: 明らかな破綻なし (ノイズだらけ・全部 0・全部無音 ではない)
- [ ] **`evaluate()` facade (T-M4.1)** で validation を 1 entry point 評価 → MCD / log F0 RMSE / UTMOS / MR-STFT が出る
- [ ] TensorBoard に loss curve / sample audio / mel visualization が記録され、divergence (NaN) なし
- [ ] M5 phase review 再評価 4 項目 (eps / sampler / segment_length / D update 比) の判断結果を §8.3 に追記し、最適 config を T-M6.1 へ申し送り
- [ ] `docs/milestones.md` §M5.1 Acceptance 3 項目クリア、`docs/tickets/index.md` の T-M5.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `configs/gan_wavenext2_1epoch.yaml` (smoke と本番の中間 config。max_steps≈33k, validation/checkpoint on、batch=16)
  - `scripts/eval_gan_checkpoint.py` (薄い driver: checkpoint をロード → `evaluate()` facade を呼ぶだけ。重複実装しない)
- 編集:
  - `docs/milestones.md` §M5.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M5.1 ステータス更新 + M5 フェーズレビューログ行
  - (再評価で確定したら) `configs/gan_wavenext2.yaml` (eps / segment_length / D update 比を本番 config へ反映)
  - (eps を 1e-5 に上げる判断なら) `src/wavenext2/data/mel.py` の `eps` default (T-M1.3 §8.3)

> **重要 (DRY)**: 訓練ロジックは T-M2.5 `train_gan.py` を **そのまま** 使う (新規 train コードを書かない)。評価は T-M4.1 `evaluate()` を呼ぶだけ。本チケットの新規ファイルは「中間 config」と「薄い eval driver」に限定する。

### 2.2 主要構造

#### 訓練起動 (既存 CLI、新規コードなし)
```bash
# 1 epoch 訓練 (約 33k step、validation/checkpoint 10k step ごと)
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2_1epoch.yaml
# divergence / OOM 時は latest checkpoint から resume
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2_1epoch.yaml --resume checkpoints/gan_1epoch/step_30000.pt
# OOM 回避が必要なら bf16
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2_1epoch.yaml --amp
```

#### `evaluate()` facade による validation (T-M4.1 §9.1 推奨 entry point)
```python
# scripts/eval_gan_checkpoint.py (薄い driver)
from wavenext2.eval.runner import evaluate, EvalResult
from wavenext2.models.gan_wavenext2 import GANWaveNext2
from wavenext2.data.dataset import LibriTTSRDataset

def main() -> None:  # Windows spawn 対策で if __name__ ガード下から呼ぶ
    G = GANWaveNext2.from_config(cfg["model"])
    G.load_state_dict(torch.load(ckpt_path, map_location="cpu")["G_state_dict"])
    G.eval()
    val_ds = LibriTTSRDataset(cfg["data"]["val_filelist"], mode="val")
    # 1 entry point: 合成 → GT ペア化 → MCD/logF0/UTMOS/MR-STFT を一括 (グルーコード不要)
    result: EvalResult = evaluate(
        G, val_ds,
        metrics=["mcd", "log_f0_rmse", "mrstft", "utmos"],
        post_filter=None,          # GAN は post-filter 不要
        seed=43,
        save_to="eval_results/gan_1epoch.json",
    )
    print(result.summary)          # {mcd_mean, log_f0_rmse_mean, utmos_mean, mrstft_sc_mean, ...}
```

> **前提 (T-M4.3 §9 / M4 phase review 申し送り)**: `evaluate()` は内部で GAN を「合成」する際に `G.synthesize(mel)` または forward を呼ぶ。T-M2.4 に `synthesize(mel)` alias が無い場合は本チケット着手時に T-M2.4 へフィードバック (薄い wrapper を追加)。

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `max_steps` | ≈ 33,000 (train-clean-100 の 1 epoch) | docs/milestones.md §M5.1 (33k step) |
| `batch_size` | 16 (OOM なら 8、§6) | docs/implementation-plan.md §5 / T-M2.5 §2.3 |
| `segment_length` | 16384 (GAN 既定、§6 で 24576 と比較) | T-M2.1 §2.3 (SoT: GAN=16384) |
| `hop_length` / `n_fft` / `win_length` | 300 / 2048 / 1200 | docs/architecture.md / T-M1.6 |
| lr_g / lr_d / betas / wd | 1e-4 / 2e-4 / [0.8,0.99] / 1e-3 | T-M2.5 §2.3 (そのまま流用) |
| scheduler | InverseLR (inv_gamma=200000, power=0.5, warmup=0.999) | T-M2.5 §2.3 |
| grad_clip_norm | 1.0 | T-M2.5 §2.3 |
| validation.interval_steps | 10000 (1 epoch 中 ~3 回) | T-M2.5 §2.3 |
| validation.num_utterances | 100 (T-M0.3 hold-out, speaker-balanced) | T-M2.5 §2.3 / T-M0.3 |
| mel eps | **1e-7 (M1 暫定)** → 発散時 1e-5 (§6) | T-M1.3 §2.3 / §8.3 |
| **MR-STFT 合格閾値** | **final < init × 0.30** | docs/milestones.md §M5.1 |
| amp | fp32 default、`--amp` で bf16 (OOM 回避) | T-M2.5 §2.3 |
| EMA | 不使用 | docs/open-questions.md |
| eval metrics | mcd, log_f0_rmse, mrstft, utmos | T-M4.1 §9.1 |

> **注 (init MR-STFT の測り方)**: T-M2.6 と同様、step 0〜10 は InverseLR warmup 域 (lr ≈ 1e-7) で実質未更新のため、`init` は **最初の validation (step ~0 直後 or step 1k)** または step 10〜20 平均で測る。30% 判定の baseline は「訓練前 (random init) の validation MR-STFT」を採用するのが最も解釈が明快 (§8.1)。

### 2.4 アルゴリズム / 処理フロー
1. `configs/gan_wavenext2_1epoch.yaml` を用意 (max_steps≈33k、validation/checkpoint on、segment/eps は再評価対象)
2. (任意) 訓練前に random-init generator の validation MR-STFT を `evaluate()` で測り `init` baseline として記録
3. `python -m wavenext2.train.train_gan --config ...` を **`run_in_background=true`** で起動 (GPU 数時間)
4. TensorBoard で loss curve / `loss_D` (D 強すぎ検知) / sample audio / mel を監視。NaN 検知時は lr 半減 or eps=1e-5 で resume
5. 10k step ごとの validation で MR-STFT total をログ、`best.pt` 更新
6. 1 epoch 完走後、`scripts/eval_gan_checkpoint.py` で `best.pt` を `evaluate()` facade 評価 → MCD / log F0 RMSE / UTMOS / MR-STFT を `eval_results/gan_1epoch.json` に永続化
7. 合格判定: `final_val_mrstft / init_val_mrstft < 0.30` かつ 生成波形が `[-1,1]`・非無音・非全0
8. 聴感確認: best.pt の sample audio (4 utterance) を人間 or 簡易チェックで破綻なしと確認
9. M5 phase review: §6 の 4 再評価項目を実測で判断し §8.3 に記録、最適 config を T-M6.1 へ申し送り

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `configs/gan_wavenext2_1epoch.yaml` + `scripts/eval_gan_checkpoint.py` (薄い driver) | general-purpose |
| Operator/Monitor | 1 | 訓練を background 起動・TensorBoard 監視・OOM/divergence 時 resume | general-purpose |
| Reviewer | 1 | Acceptance 検証 + M5 phase review 4 項目判断 + T-M6.1 申し送り作成 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M5.2 (Diff 1 epoch) と GPU を共有しなければ並列可。ただし単一 GPU なら順次)
- 並列実行する場合の最大並列数: 2 (T-M5.1 / T-M5.2、GPU が 2 枚あれば)
- 後続 T-M6.1 は本チケット pass が gate

## 4. 提供範囲 (Scope)

### In Scope
- `configs/gan_wavenext2_1epoch.yaml` (max_steps≈33k の中間 config)
- 既存 `train_gan.py` の **1 epoch 起動・監視** (新規 train コードなし)
- `evaluate()` facade (T-M4.1) による validation 評価 driver (`scripts/eval_gan_checkpoint.py`、薄い wrapper)
- MR-STFT < init × 30% の合格判定
- 生成波形 `[-1,1]`・非無音・非全0 の sanity 判定
- 聴感破綻なしの確認 (sample audio)
- M5 phase review 4 項目 (eps / speaker-balanced sampler / segment_length / D update 比) の実測判断
- T-M6.1 への最適 config / hyperparameter 申し送り

### Out of Scope
- **2M step 本格訓練** (T-M6.1)
- **新規 train ロジック / loss / model** (すべて T-M2.x で完成済前提、本チケットは起動のみ)
- **Diff 1 epoch** (T-M5.2)
- **MCD / log F0 RMSE / UTMOS / RTF の指標実装本体** (T-M4.1 / T-M4.2 / T-M4.3、本チケットは `evaluate()` を呼ぶだけ)
- **論文 Table との絶対値対比** (M6、本チケットは「init 比 30%」と相対傾向のみ)
- **Multi-GPU / DDP** (T-M6.1)
- **NISQA** (任意。`evaluate()` の metrics に追加可だが必須は UTMOS まで)
- **hyperparameter search** (再評価は §6 の 4 項目に限定、grid search はしない)

### Deliverable
- ファイル:
  - `configs/gan_wavenext2_1epoch.yaml` (新規)
  - `scripts/eval_gan_checkpoint.py` (新規、薄い driver)
  - `checkpoints/gan_1epoch/best.pt` (訓練成果物、commit しない)
  - `eval_results/gan_1epoch.json` (評価結果、`.gitignore` 済)
- 関数 / クラス: なし (既存 `train_gan.main` / `evaluate` を再利用)
- ドキュメント差分:
  - `docs/milestones.md` §M5.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M5.1 ステータス + M5 フェーズレビューログ
  - §8.3 に再評価判断結果を追記

## 5. テスト項目

### 5.1 Unit テスト
- [ ] `configs/gan_wavenext2_1epoch.yaml` が `load_config` でパース可能、必須 key (train/validation/checkpoint/logging/data/model) が揃う
- [ ] `scripts/eval_gan_checkpoint.py` が `--ckpt` / `--config` を受け、`if __name__ == "__main__"` ガード下で `evaluate()` を呼ぶ (Windows spawn 安全)
- [ ] dummy/小 checkpoint + 小 val subset で `evaluate(G, ds, metrics=["mcd","log_f0_rmse","mrstft"])` が `EvalResult` を返す (API 動作のみ、値の妥当性は不問)

### 5.2 e2e / 結合テスト (本チケットの本体、GPU 必須・`@pytest.mark.slow @pytest.mark.gpu`)
- [ ] **33k step (1 epoch) を OOM なし完走** (batch=16、OOM 時は 8 or `--amp` で再試行)
- [ ] **validation MR-STFT total < 初期値の 30%** (init = 訓練前 random-init validation、または step 10〜20 平均)
- [ ] 生成波形が `[-1, 1]` 範囲内 (clip 機能)、**無音でない・全 0 でない** (`y.abs().max() > ε_floor` かつ `y.std() > ε_floor`)
- [ ] `evaluate()` facade で **MCD / log F0 RMSE / UTMOS** が finite な値で出る (GT 比で MCD は有限・小さめ、UTMOS は破綻なしの目安 > 2.5)
- [ ] TensorBoard に **loss curve (loss_G/loss_D/各 sub-loss) / sample audio / mel visualization** が記録、NaN/Inf 出現なし

### 5.3 Acceptance criteria (`docs/milestones.md` §M5.1 より転記)
- [ ] OOM なしで完走
- [ ] 1 epoch 後の generator が validation utterances に対して MR-STFT loss < 初期値の 30%
- [ ] 生成音声の聴感: 明らかな破綻なし (ノイズだらけ・全部 0 ではない)

### 5.4 追加 acceptance (本チケット独自)
- [ ] `evaluate()` 1 entry point 呼び出しで MCD / log F0 RMSE / UTMOS / MR-STFT が `eval_results/gan_1epoch.json` に永続化
- [ ] M5 phase review 4 項目 (eps / sampler / segment_length / D update 比) の判断が §8.3 に記録
- [ ] T-M6.1 へ「pass した config + 最適 hyperparameter」が §9.1 で明文化

## 6. 懸念事項

> **本チケットは M6 本格訓練 (410h) 前の最後の gate**。ここで先送りした再評価項目を実測データで決着させる。

### 6.1 技術的リスク (M5 phase review 再評価項目)

#### 【重要】統合バグの分類 (smoke pass ≠ 統合層検証済み)
T-M2.6 smoke は **単一プロセス・固定 device** で回るが、1 epoch は以下を **初めて連続で踏む**: DataLoader worker (Windows spawn)・AMP autocast・checkpoint resume。smoke が pass していても **統合層は未検証** であり、以下の統合バグを §5.2 e2e で初めて検出しうる:
- **device mismatch**: checkpoint を `map_location="cpu"` でロード後 `.cuda()` を忘れる / optimizer state が CPU に残る → 起動時 `RuntimeError: tensor on different device`
- **config mismatch**: `gan_wavenext2_1epoch.yaml` と本番 `gan_wavenext2.yaml` の key drift (片方にしか無い key、default 値のズレ) → 静かに別挙動。両 config の diff を §7 で明示確認
- **AMP dtype mismatch**: `--amp` (bf16) autocast 下で fp32 前提の loss/STFT が dtype 不一致 → `expected scalar type` or 静かな精度劣化
> 位置づけ: 「smoke pass 済でも統合層 (worker/AMP/resume) は未検証」。M5.1 fail 時は architecture (優先度低) より **これら統合バグ** を先に疑う。

#### 【重要】resume 完全性 (checkpoint からの状態復元)
OOM/divergence resume が成立するには、checkpoint に **optimizer / scheduler / step counter / RNG state** が含まれ完全復元される必要がある。特に:
- **InverseLR warmup 途中 resume で lr がリセットされると収束判定が壊れる** (warmup=0.999 の途中で lr が step 0 相当に戻ると見かけ上 loss が再上昇し gate 誤判定)
- step counter が復元されないと scheduler の現在 lr が再計算できない、RNG state が無いと resume 後の data 順 / dropout が変わり再現性喪失
- 確認: 同一 step で resume 前後の lr / loss が連続することを 1 度検証 (§5.2 に追加候補)。確認結果を T-M6.1 へ申し送り (§9.1)

#### 再評価 4 項目 (M1〜M2 から M5 へ持ち越し)
- **eps=1e-7 vs 1e-5 の再評価 (T-M1.3 §8.3 / §6.1 トリガー)**:
  - M1 では Vocos warm-start 互換性のため `eps=1e-7` を暫定採用。**1 epoch で発散 (NaN / loss 振動拡大) したら `eps=1e-5` (HiFi-GAN 慣例) に上げて resume**
  - 判断: 発散しなければ 1e-7 確定 → `docs/training.md` §1.2 を 1e-7 に更新。発散すれば 1e-5 へ (`mel.py` default + 本番 config 反映)
  - **代理判定 (33k step では NaN が出ないことも多く判断材料不足)**: 1 epoch では eps=1e-7 でも NaN に至らないことが多く「発散の有無」だけでは 2M step (T-M6.1) の安全性を判断しきれない。そこで NaN を待たず **`grad_norm の p99`** (clip 前の上位パーセンタイル、1e-7 で外れ値が膨らむか) と **`loss_G の振動幅`** (移動平均からの偏差) を代理指標とし、これらが 1e-5 比で顕著に大きければ 1e-7 を危険と判断して 1e-5 を申し送る
- **speaker-balanced batch sampler の再評価 (T-M2.1 §8.1, 「M2 smoke 後判断」→ M5 へ)**:
  - validation MR-STFT を **話者ごとに分解**し、話者間で分散が大きい (特定話者だけ品質が悪い) なら speaker-balanced sampler を T-M6.1 で導入
  - 判断: 話者分散が小さければ random sampler 継続、大きければ T-M6.1 で sampler 変更を申し送り
- **D 強すぎ問題 (T-M2.5 §6.1 critical)**:
  - `loss_D < 0.01` が連続 1000 step 続く (TensorBoard 警告ログ) なら D が強すぎて G が学習しない兆候
  - **failure mode: `loss_adv` が 0 に張り付く**: D が完璧に G を見破ると G の adversarial 勾配が消え、G は FM (feature matching) / MR-STFT loss のみで学習する。loss curve は下がり続けるように見えるが **音質は頭打ち** になる (adversarial 成分が効かず GAN の旨味が出ない)。`loss_adv → 0` への張り付きを TensorBoard で監視し divergence gate の `loss_adv ∈ (0.5,2.0)` 域 (§8.1) を外れたら検知
  - 対応: **D を 2 step に 1 回更新** (1:2 → 1:1 へ) or D lr を下げる。判断結果を T-M6.1 へ申し送り
- **segment_length=16384 vs 24576 (T-M2.1 §2.3)**:
  - MR-STFT 収束が遅い (1 epoch で 30% に届かない) なら Vocos 流に segment を伸ばす。**ただし T-M2.1 の SoT は GAN=16384 / Diff=25600** であり、prompt の「24576」は中間案。長い segment は OOM リスク増のため batch との trade-off で判断
  - **high-res STFT underfit の懸念**: hop=300 で `segment=16384` は約 54 frame 相当だが、MR-STFT の **high-res 項 (n_fft=2048, win_length=1800)** は 1 window が segment の 11% を占め、segment 内に **1 window 弱しか入らない**。high-res STFT 項が segment 短すぎで underfit し、その項だけ下がらない可能性がある
  - 判断手順: MR-STFT total でなく **high-res STFT 項単独の curve** を分けてログし、その項が下がらないかを見る。下がらなければ **16384 vs 24576 を比較** (24576 で high-res 項が改善するか) して segment 延長を T-M6.1 へ申し送り

#### その他リスク
- **OOM (T=4 × batch=16 × sub-model 4 段、T-M1.4 §6.1)**:
  - 緩和優先順位: ① `--amp` (bf16、activation 半減) → ② `model.enable_grad_ckpt=true` (T-M1.4) → ③ batch_size=8 へ半減
  - 検知: 起動直後 数 step で `torch.cuda.OutOfMemoryError`。`run_in_background` のログを Monitor で確認
- **T-M1.6 戻り値仕様 (n_t vs y_{t-1}) が正しいか**:
  - 本来 T-M2.6 smoke で確定済みのはず。**もし 1 epoch で全く収束しない (MR-STFT が下がらない) なら、smoke が偽陽性だった疑い** → T-M1.6 §6.1 / T-M2.4 へ最優先で遡及 (`y = y - n_t` の符号、初期化 `y_T = zeros`)
- **divergence (NaN) (T-M2.6 §6.1)**:
  - grad_clip=1.0 が効いているか確認、効いていれば lr 半減 or eps=1e-5 で resume。AMP は最初 fp32 (bf16 は OOM 時のみ)
- **1 epoch (33k step) では収束に不十分な可能性**:
  - smoke は overfit (10%) だが 1 epoch 汎化は 30% 閾値 (緩め)。それでも届かない場合は §8.1「fixed 10k step gate」or epoch 数増を検討、ただし安易に閾値を緩めない (M6 で無駄打ちを防ぐ gate の意味が薄れる)
- **GPU リソース**:
  - A100 1 枚で数時間、または ローカル GPU (RTX 3090/4090) で半日〜1 日。**ユーザー操作** (GPU 環境確保) が必要 (§9)
- **wall-clock 実レンジ (GPU 確保判断材料)**:
  - 33k step の実時間は GPU で 3〜10x 変動する (A100 で数時間 / RTX 3090 で半日〜 / Colab T4 でさらに長い)。**実測した 33k step の wall-clock を記録**し、T-M6.1 の 2M step (≈ 60x) との比例で本格訓練の所要時間 (= GPU 確保規模・課金見積もり) を user に提示する。これが M6 起動の GO/NO-GO 判断材料になる (§8.2、§9.1)
  - wall-clock が過大なら fixed-step / budget-cap gate へ切替 (§8.1)
- **TensorBoard ログの保管・提示**:
  - `logs/` のログ寿命 (上書き / ローテーション) を確認し、gate 判定に使った loss curve が消えないようにする。user への提示は **`tensorboard --inspect` のサマリ** または **scalar の CSV export** で行い (TensorBoard UI を直接見せられない前提)、divergence gate の 3 指標 (loss_G/loss_adv/loss_D) を数値で添える
  - background 起動した訓練プロセスの **stdout/stderr は Monitor で追う** (各行が通知される)
- **background プロセス死活監視 (crash 検知)**:
  - OOM 以外の crash (DataLoader worker の死、Windows spawn 失敗、segfault) は `torch.cuda.OutOfMemoryError` のような明示例外を出さずプロセスが静かに死ぬことがある。**プロセスの生存** (PID / exit code) と **ログの最終更新時刻の停滞** を死活監視の手段とし、停止を検知したら最新 checkpoint から resume (resume 完全性は本節「resume 完全性」参照)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 関連項目: すべて確定済み
- 本チケット固有の判断:
  - **MR-STFT 30% 判定の `init` 定義**: 「訓練前 random-init generator の validation MR-STFT」を採用 (最も解釈が明快、§8.1)。代替で step 10〜20 平均も可
  - **聴感破綻なしの定量化**: `y.abs().max() ∈ (ε, 1.0]` (無音/全0でない・clip 機能) + UTMOS > 2.5 を目安、最終判断は人間の聴取

### 6.3 他チケットとの整合性
- **T-M2.5 (train_gan)**: `main()` CLI / `train_gan_step` / checkpoint resume / validation 機構をそのまま使う。本チケットは config 差分のみ
- **T-M2.6 (smoke)**: smoke pass = architectural sanity のみ。本チケットで収束性を初めて確認 (T-M2.6 §9.1)。smoke pass しているので fail 時はまず hyperparameter / データ前処理を疑い、architecture バグの優先度は低い
- **T-M4.1 (evaluate facade)**: `evaluate(model, dataset, metrics, post_filter, *, seed, save_to) -> EvalResult` を呼ぶ。`GANWaveNext2` の合成 alias (`synthesize(mel)`) が必要 (M4 phase review 申し送り) → 無ければ T-M2.4 へフィードバック
- **T-M2.1 (dataset)**: val は `mode="val"` で固定 gain (-3 dB) + `shuffle=False` の deterministic。speaker-balanced は本チケットで再評価

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 訓練ロジックを新規実装せず T-M2.5 `train_gan.py` を起動している (DRY)
- [ ] 評価が `evaluate()` facade 1 entry point で、グルーコードを書いていない (T-M4.1 §9.1)
- [ ] Acceptance criteria 全項目クリア (5.3 / 5.4)
- [ ] `configs/gan_wavenext2_1epoch.yaml` が smoke (T-M2.6) / 本番 (T-M6.1) と差分明示 (max_steps / validation on)
- [ ] OOM 緩和 (amp / grad_ckpt / batch 半減) の手順が config / 起動コマンドに反映
- [ ] M5 phase review 4 項目の判断が §8.3 に実測値付きで記録
- [ ] divergence / D 強すぎの検知ログが TensorBoard で確認可能
- [ ] T-M6.1 への申し送り (最適 config + hyperparameter) が §9.1 で具体的
- [ ] 参考実装をコピーしていない (起動・評価のみで新規ロジックなし)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M5 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **1 epoch でなく fixed step (10k step) で gate** | epoch 長 (データ量) に依存せず gate 基準が安定、wall-clock 予測可能 | 「1 epoch 完走」の達成感がない、データ全体を見ない | milestones.md が「1 epoch (33k step)」と明記。ただし fixed-step の方が gate として健全 | M6 本格訓練前 (本チケット完了時) に「33k step ≈ 1 epoch」が wall-clock 過大なら 10k step gate へ |
| **train-clean-100 でなく小 subset (10h)** | iteration が高速、複数 config を試せる | 汎化検証が弱い (話者数が減る) | M5 は汎化 gate なので 100h が妥当。ただし高速 iteration 用に subset を別途用意する価値あり | eps / segment_length の再評価で複数 config を回す必要が出たとき |
| **multiple seeds で variance 確認** | 1 epoch 結果の運要素 (seed 依存) を排除、gate の信頼性向上 | GPU 時間が seed 数倍 | M5 は単一 seed で「明らかな破綻なし」を見る粒度。variance は M6 で確認 | M6.1 で seed 依存の品質差が疑われたとき |
| **30% でなく絶対値閾値 (init 非依存)** | init 測定不要、再現性が高い | 絶対 MR-STFT 値の妥当ラインが未知 (backend 依存) | init 比は smoke (10%) と一貫し解釈が明快 | M6 で複数 run の MR-STFT 絶対値分布が判明したら絶対閾値へ |
| **checkpoint 間隔 10k → 2.5k step** (1 epoch 中 ~12 回) | OOM/divergence は終盤に起きやすく resume コストを削減、最後の健全 ckpt が近い | ストレージ消費増、I/O オーバーヘッド微増 | (検討中) M5 起動時に採用判断。GPU 数時間規模なら 2.5k が安全側 | 1 epoch 中盤以降で OOM/divergence が頻発し 10k 前 resume が高コストと判明したとき |
| **wall-clock budget gate** (例: 4h cap) | GPU が A100/3090/Colab T4 で 33k step 実時間が 3〜10x 変動するため step 数でなく時間で頭打ちを切れる | 遅い GPU で 1 epoch 未満で打ち切られ gate 判定不能になりうる | (検討中) step gate と併用候補。実 GPU 確定後に cap 値を決める | ローカル/Colab で 33k step が想定外に長時間化したとき (§6.1 wall-clock 実レンジ参照) |
| **`auto_oom_fallback` の config 化** (`[bf16, grad_ckpt, batch//2]`) | background 起動中 OOM 検知 → 自動で次段の緩和を適用し再起動、人手介入レス | fallback 適用で batch/精度が変わると gate baseline がずれる (要記録) | (検討中) Operator 手動 fallback (§6.1) の自動化候補。orchestrator に組込む | OOM が起動初期に頻発し手動 resume が煩雑なとき。T-M5.2 と共通化 (§9.1) |

#### 採用設計
- **既存 `train_gan.py` を中間 config で起動するだけ** (新規コードゼロ、最小コストの gate)
- **`evaluate()` facade 1 entry point 評価** (グルーコードを書かない、T-M4.1 の設計意図に乗る)
- **`init` = 訓練前 random-init validation MR-STFT** (smoke の step 10〜20 平均より解釈が明快)
- **【昇格】gate を「品質 gate」でなく「divergence gate」に**: 1 epoch (33k step ≈ 本番 2M の 1.6%) では HiFi-GAN/Vocos でも init 比 50〜70% しか落ちず、`MR-STFT < init×30%` は M2.6 overfit (10%) からの安直な外挿で未達が定常化する。gate を以下の **方向条件 (divergence の不在)** に変更する:
  - `loss_G` が単調減少傾向 (1 epoch スパンで右肩下がり)
  - `loss_adv` が学習中域 (0.5〜2.0) に留まる
  - `loss_D` が 0.01 を割らない (D 強すぎ検出、§6.1)
  - 生成波形が finite かつ `[-1,1]` 範囲内かつ 非無音かつ 非全 0
  - NaN/Inf が出現しない
  > MR-STFT 30% は **参考指標 (目安)** に降格。1 epoch で「品質が出たか」でなく「発散していないか」だけを gate とする (§8.2 哲学)。
- **【昇格】gate の自動 pass/fail と人間判断の二層構造**: `evaluate()` が機械可読 verdict (`EvalResult.gate_passed: bool`) を返し、**自動項目** (閾値・finite・clip・NaN) は自動判定、**聴感のみ** 人間に委ねる。これにより gate 判定の大部分が再現可能・自動化可能になる
- **【昇格】共通 `run_smoke.py --mode {gan,diff}` orchestrator**: T-M5.1/T-M5.2 の骨格 (config 起動 → background 監視 → `evaluate()` → gate 判定 → JSON 永続化 → 申し送り生成) は同型。**gate 判定ロジック・JSON schema を 1 箇所に集約**し、GAN/Diff 差分は `--mode` と config だけにする (DRY、T-M5.2 と共通化 §9.1)

#### 再評価トリガー
- **M6 本格訓練前**: 33k step ≈ 1 epoch の wall-clock が過大なら fixed 10k step gate へ切替。30% 閾値が緩すぎ/厳しすぎなら M6.1 の収束曲線を見て調整
- **divergence gate の方向条件** (loss_G 傾向・loss_adv 域・loss_D 下限) が実 GPU で誤検知 (健全なのに fail) するなら閾値域を M5 起動時に微調整

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切。M5 は「実データで汎化的に収束するか」を最小コストで見る gate。size=S (起動 + config + 薄い eval driver) が妥当
- **smoke (M2.6) との役割分担**: M2.6 = architectural sanity (1 sample overfit)、M5.1 = 汎化収束 (実データ 1 epoch)。**M5.1 fail 時はまず hyperparameter / データ前処理を疑う** (smoke pass 済なので architecture バグ優先度は低い、T-M2.6 §8.2)
- **別マイルストーン移行**: なし。M5 (統合スモーク) に置くのが自然
- **gate の意味を緩めない**: 30% 閾値を安易に緩めると M6 で 410h を無駄打ちするリスク。閾値を緩めるより epoch/step を増やす方が gate として健全
- **【重要】「品質 gate でなく divergence gate」哲学**: 1 epoch (本番 2M step の 1.6%) で品質を判断するのは早計。M5.1 が問うべきは「この config で本番を回したとき発散しないか」だけであり、品質 (MCD/UTMOS/MR-STFT 絶対値) は M6 で判断する。divergence (NaN・loss 振動拡大・D 暴走・無音/全0 出力) の **不在** を確認できれば gate pass とする (§8.1 採用設計)
- **【重要】gate 判定者 = user の GO/NO-GO**: M6 は **GPU 課金が発生するユーザー操作**。したがって gate の最終判定は **user 承認 (GO/NO-GO) を明示的に要する**。Claude Code (Reviewer) の役割は判定を下すことではなく、「divergence gate の自動項目 (3 指標 + finite/clip/NaN) の pass/fail + sample audio 4 本」を提示し **GO/NO-GO 推奨を出す** ことに留める。最終的に課金前のボタンを押すのは user
- **CI 不可 (GPU runner 必須)**: 本チケットの e2e は GPU を要するため通常の CI には乗らない (`@pytest.mark.slow @pytest.mark.gpu`)。GPU runner が無い限り自動 CI ゲートにはできず、user 環境での手動起動 + Reviewer の GO/NO-GO 推奨が実体

### 8.3 学んだこと (2026-05-28 足場実装、実行は user-gated)

- **M5 は M6/M7 同様のユーザー操作必須境界**: 実 1 epoch 訓練は GPU 数時間 + T-M0.3 実 LibriTTS-R + M6 課金前の **user GO/NO-GO 判定** (§8.2) で、Claude では実行不可。自律実装の範囲は「非ゲートの足場」= divergence gate ロジック + 共通 orchestrator + 中間 config + 薄い eval driver + テスト。実行・判定は user。
- **divergence gate を機械判定可能な純関数に**: `eval/gate.py::gan_divergence_gate(loss_g, loss_adv, loss_d, output) -> GateResult` で「発散の不在」(loss_G 単調減少傾向・loss_adv∈(0.5,2.0)・loss_D≥0.01・波形 finite/[-1,1]/非無音・NaN なし) を自動判定。傾向判定は前半平均>後半平均で step 揺らぎに頑健、履歴 <4 は pass 寄り。MR-STFT<30% は品質目安に降格 (1 epoch=本番の 1.6% で品質は測れない、§8.2)。
- **共通 orchestrator `run_smoke.py --mode {gan,diff}`** (M5 review §8.1): metrics JSON → gate 判定 → GO/NO-GO レポート → JSON 永続化、gate fail で exit 1。GAN/Diff 差分は `--mode` と loss キーのみ。訓練起動自体は GPU 要のため orchestrator は起動せず、訓練+eval が出力した metrics を判定する設計 (テスト可能・再現可能)。
- **再評価 4 項目 (eps/sampler/segment_length/D update 比) は実データ必須のため未確定**: いずれも 1 epoch の実 loss curve / grad_norm p99 / 話者分散を見て判断する項目で、実行時に user 環境で決着 → T-M6.1 へ申し送り。
- **18 tests pass (gate 各 fail mode / orchestrator / 中間 config parse)、ruff clean**。
- 教訓: GPU/データ/課金が絡む gate は「判定ロジックを純関数化して機械判定を自動化・テスト可能にし、実行と最終承認だけ user に残す」と、コードは完成・実行は委譲を両立できる。divergence gate (発散の不在) は品質 gate より早期・安価・解釈明快。
- **eps=1e-7 vs 1e-5 判断**: TBD (実行時の発散有無 + grad_norm p99 / loss_G 振動幅で確定)
- **speaker-balanced sampler 判断**: TBD (validation MR-STFT の話者分散で確定)
- **D update 比判断**: TBD (loss_D < 0.01 連続有無で確定)
- **segment_length 判断**: TBD (MR-STFT 収束速度で確定)
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M6.1 (GAN フル訓練) へ
- **使用方法**: smoke (M5.1) が pass した `configs/gan_wavenext2_1epoch.yaml` を **そのまま max_steps=2M に拡張**して本格訓練
  ```bash
  # T-M6.1 で実行 (max_steps を 2M に、validation/checkpoint 間隔は据え置き)
  uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml
  ```
- **引き継ぐもの (divergence gate を pass した config)**: 本チケットで **divergence gate** (loss_G 単調減少・loss_adv ∈ (0.5,2.0)・loss_D ≥ 0.01・波形 finite/[-1,1]/非無音・NaN なし、§8.1) を pass した `configs/gan_wavenext2_1epoch.yaml` を引き継ぐ。MR-STFT 30% は品質目安に降格 (gate ではない)
- **引き継ぐ最適 hyperparameter** (本チケットの再評価で確定):
  - `eps`: 1e-7 (発散せず代理指標 grad_norm p99 / loss_G 振動幅も健全なら) or 1e-5 (発散 or 代理指標が危険なら) — §8.3 に記録
  - `segment_length`: 16384 (high-res STFT 項含め収束十分なら) or 24576 (high-res 項が underfit で遅いなら、OOM と trade-off) — §8.3
  - **D update 比**: 1:1 (loss_D 健全なら) or 1:2/D-lr 減 (D 強すぎ / loss_adv 0 張り付きなら) — §8.3
  - **batch_size / amp / grad_ckpt / `auto_oom_fallback`**: OOM 回避で確定した組み合わせ (例: batch=16 + bf16) と、採用したなら `auto_oom_fallback: [bf16, grad_ckpt, batch//2]` の config 値 (§8.1)
  - **speaker-balanced sampler**: 要 (話者分散大なら) / 不要 (random sampler 継続) — §8.3
- **resume 完全性 (確認済みを引き継ぐ)**: optimizer/scheduler/step counter/RNG state が checkpoint に含まれ完全復元される (特に InverseLR warmup 途中 resume で lr がリセットされない) ことを本チケットで確認済みである旨を申し送る。T-M6.1 (2M step・長時間) では resume が頻発するため前提として重要 (§6.1)
- **評価方法**: T-M6.1 も `evaluate()` facade で full 4824 utt 評価。論文 Table 対比は相対比較主軸、結果は `eval_results/*.json` 永続化 (T-M4.1 §9.1)
- **gate の意味 / 判定者**: 本チケットが pass しなければ T-M6.1 を **起動しない** (410h の無駄打ち防止)。pass 判定は **Claude Code が GO/NO-GO 推奨を出し、最終承認は user** (課金前、§8.2)

#### T-M5.2 (Diff 1 epoch) と共通
- **`run_smoke.py --mode {gan,diff}` orchestrator**: gate 判定ロジック・JSON schema を 1 箇所に集約する共通骨格 (§8.1)。GAN/Diff で差分は `--mode` と config のみ。本チケットと T-M5.2 で実装を共有
- **`EvalResult.gate_passed: bool`**: `evaluate()` が返す機械可読 verdict (自動項目の pass/fail)。GAN/Diff 共通の API として T-M4.1 に依存

#### ユーザー操作 (必須)
- **GPU リソース確保**: A100 1 枚 (数時間) または ローカル GPU (RTX 3090/4090 で半日〜1 日)。SSH / 課金設定はユーザー側 (T-M6 と同様だが M5 は数時間規模で軽い)
- **gate の GO/NO-GO 承認 (M6 課金前)**: M6 は GPU 課金が発生するため、Claude Code が提示する「divergence gate 3 指標 + sample audio 4 本 + 33k step wall-clock 実測」を見て、**user が M6 起動の GO/NO-GO を承認**する (§8.2)

#### 失敗時のフィードバック方向
- **MR-STFT が 30% に届かない / 発散**:
  1. hyperparameter (lr, batch, warmup, eps) を疑う (smoke pass 済なので最優先)
  2. データ前処理 (mel 正規化 eps, audio gain -3dB) を疑う
  3. それでも全く下がらない → **T-M1.6 §6.1 (戻り値 n_t vs y_{t-1}) / T-M2.4 (fixed-point 符号) / T-M2.3 (hinge GAN 符号)** へ遡及 (smoke が偽陽性だった疑い)
- **OOM**: T-M1.4 §6.1 (enable_grad_ckpt) / T-M2.5 §6.1 (amp) へ
- **D 強すぎ**: T-M2.5 §8.1 (D update 比) へ

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M5.1 Acceptance チェックボックス 3 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M5.1 ステータスを `📝 pending` → `✅ completed`、M5 進捗サマリ + フェーズレビューログ M5 行を更新
  - [ ] (発散時) `src/wavenext2/data/mel.py` の `eps` default と `docs/training.md` §1.2 を 1e-5 に更新
  - [ ] (収束時) `docs/training.md` §1.2 の eps を 1e-7 確定に更新
  - [ ] §8.3 に再評価 4 項目の判断結果を実測値付きで追記

### 9.3 Open question として残ったもの
- 30% 閾値が経験的に妥当か (M6.1 収束曲線で確認、緩すぎ/厳しすぎなら §8.1 で調整)
- 33k step (1 epoch) で汎化 gate に十分か (不足なら fixed-step gate or epoch 増、§8.1)
- segment_length 16384 vs 24576 のどちらが本番に最適か (本チケットで初期判断、M6.1 で最終確認)
