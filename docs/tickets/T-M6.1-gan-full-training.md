---
id: T-M6.1
title: GAN-WaveNeXt 2 フル訓練 (A100 約 410h、2M step)
milestone: M6
phase: M6
status: pending
size: L
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M5.1]
blocks: [T-M6.3, T-M7.1]
related_docs:
  - docs/milestones.md#m61-gan-wavenext-2-フル訓練
  - docs/training.md
---

# T-M6.1: GAN-WaveNeXt 2 フル訓練 (A100 約 410h、2M step)

> **マイルストーン**: [M6](../milestones.md#m6-本格訓練-claude-code-は起動監視のみ-wall-clock-a100-で約-442-時間) / **サブタスク**: [M6.1](../milestones.md#m61-gan-wavenext-2-フル訓練)
> **依存**: [T-M5.1](T-M5.1-gan-1epoch.md) (前提: [T-M2.5](T-M2.5-train-gan.md), [T-M4.1](T-M4.1-objective-metrics.md), [T-M4.3](T-M4.3-rtf.md)) / **後続**: [T-M6.3](T-M6.3-ablation.md), [T-M7.1](T-M7.1-mos-test.md)

## 1. タスク目的とゴール

### 目的
T-M5.1 の **divergence gate** (1 epoch ≈ 33k step、loss 方向条件 + finite + NaN なし) を通過し **user が GO を出した** `configs/gan_wavenext2_1epoch.yaml` の最適 config を **そのまま max_steps=2M に拡張**して、GAN-WaveNeXt 2 を **A100 単体で約 410 時間 (2M step) フル訓練**し、論文 Table 1〜3 と対比可能な品質の `checkpoints/gan/best.pt` を得る。

本チケットは **新規実装ゼロ** (既存 `train_gan.py` を起動するだけ) を原則とし、唯一の例外として **RTF 測定 (T-M4.3) のために `GANWaveNext2.synthesize(mel)` alias を T-M2.4 へ実装する** (期限は **best.pt 生成後・RTF 測定前**。RTF は完走後タスクなので 410h 起動をブロックしない、§9 cross-ticket)。Claude Code は訓練の **起動・長時間監視・自動 resume・checkpoint ごとの validation・best 更新** を担い、品質判定 (論文 Table 対比) と最終成果物の確定を行う。**GPU 確保・課金・SSH・M5.1 gate の GO/NO-GO 承認は user 操作** (§9.3)。

T-M5.1 が「発散しないか」だけを見た gate であったのに対し、本チケットは初めて **品質 (UTMOS / NISQA / MCD / log F0 RMSE / RTF)** を論文と対比する。ただし backend 差により絶対値の一致は期待せず、**相対比較主軸** (T-M4.1 §8.2) で合否を判断する。

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M6.1 Acceptance を内包):
- [ ] 既存 `train_gan.py` を `configs/gan_wavenext2.yaml` (max_steps=2M, validation/checkpoint on) で **`run_in_background=true` で起動**し、**2M step 完走 (or early stop)**
- [ ] validation MR-STFT loss が **プラトー** (収束曲線が右肩下がりから平坦に遷移、移動平均の傾きが閾値以下)
- [ ] 客観評価 (`evaluate()` facade、full 4824 utt): **UTMOS / NISQA / MCD / log F0 RMSE が論文 Table 1〜3 と概ね一致 (±10%)**、ただし **backend 差を考慮した相対比較主軸** (GT≈0 / GAN < Diff 等の順序関係、T-M4.1 §8.2)
- [ ] **RTF が論文と一致** (`measure_rtf` を A100 + 1-core CPU で測定。**参照値 0.0066/0.20 は paper-summary では T=5 の値 → 測定 T と参照行 T を一致させてから対比**、§6.1 CRITICAL / §5)
- [ ] divergence (NaN) / OOM / GPU preemption を検知したら **最新 checkpoint から完全 state 復元で自動 resume** (optimizer/scheduler/step/RNG)
- [ ] checkpoint 保存ごとに validation を `evaluate()` で実行し、MR-STFT 最小で `best.pt` を **atomic rename** 更新 (T-M2.4 申し送り)
- [ ] **`GANWaveNext2.synthesize(mel)` alias** (RTF 測定で必要) が T-M2.4 に実装され、`getattr(model, "synthesize", model.forward)` dispatch で動作 (**期限: best.pt 生成後・RTF 測定前**。RTF は完走後タスクなので 410h 起動はブロックしない、§6.1 / §9 cross-ticket)
- [ ] 評価結果が `eval_results/gan_full.json` に永続化 (論文 Table 対比時の再計算回避)、TensorBoard ログ / checkpoint が cloud (S3/GCS) に sync
- [ ] `docs/milestones.md` §M6.1 Acceptance 4 項目クリア、`docs/tickets/index.md` の T-M6.1 ステータス更新

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - (原則なし。本チケットは既存 `train_gan.py` の起動・監視が本体)
  - (任意) `scripts/sync_checkpoints.sh` (S3/GCS への checkpoint / TensorBoard ログ rsync、cron or save hook から呼ぶ薄い wrapper、§6.1)
- 編集:
  - `configs/gan_wavenext2.yaml` (T-M2.5 で生成済の本番 config に、T-M5.1 で確定した最適 hyperparameter を反映: eps / segment_length / D update 比 / batch / amp / `auto_oom_fallback`。max_steps=2M は据え置き)
  - `src/wavenext2/models/gan_wavenext2.py` (**T-M2.4 への cross-ticket**: `synthesize(mel)` alias を **best.pt 生成後・RTF 測定前**までに追加。本チケットでは「RTF 測定時に未実装なら T-M2.4 にフィードバックして追加させる」。訓練起動はブロックしない)
  - `docs/milestones.md` §M6.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M6.1 ステータス更新 + M6 フェーズレビューログ行

> **重要 (DRY)**: 訓練ロジックは T-M2.5 `train_gan.py` を **そのまま** 使う (新規 train コードを書かない)。評価は T-M4.1 `evaluate()` facade を呼ぶだけ。T-M5.1 で pass した `gan_wavenext2_1epoch.yaml` の差分を本番 `gan_wavenext2.yaml` に取り込み、max_steps だけ 2M にする。本チケットの新規ファイルは「cloud sync の薄い script」に限定する。

### 2.2 主要構造

#### 訓練起動 (既存 CLI、新規コードなし、background 実行)
```bash
# 2M step フル訓練 (validation/checkpoint 10k step ごと)。run_in_background=true で起動
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml
# divergence / OOM / preemption 時は latest checkpoint から完全 state 復元で自動 resume
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml --resume checkpoints/gan/step_1500000.pt
# divergence (NaN) 復帰時は lr 半減 / eps=1e-5 / AMP 無効化を config 側で調整して resume (§6.1)
uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2_recover.yaml --resume checkpoints/gan/step_1500000.pt
```

#### RTF 測定で必要な `synthesize(mel)` alias (T-M2.4 cross-ticket、§9)
```python
# src/wavenext2/models/gan_wavenext2.py に追加 (best.pt 生成後・RTF 測定前まで。訓練起動はブロックしない)
class GANWaveNext2(nn.Module):
    ...
    @torch.no_grad()
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """RTF 測定 / 評価用の単一引数合成 alias.

        forward(mel, audio_length) の audio_length=None auto-infer 版。
        measure_rtf (T-M4.3) が getattr(model, "synthesize", model.forward) で
        引数 1 個 (synth(mel)) で呼ぶため、第 2 引数必須の forward では TypeError になる。
        """
        return self.forward(mel, audio_length=None)
```

#### `evaluate()` facade による full 4824 utt 評価 (T-M4.1 §9.1 推奨 entry point)
```python
# scripts/eval_gan_checkpoint.py (T-M5.1 で作成済の薄い driver を full set に向ける)
def main() -> None:  # Windows spawn 対策で if __name__ ガード下から
    G = GANWaveNext2.from_config(cfg["model"])
    G.load_state_dict(torch.load("checkpoints/gan/best.pt", map_location="cpu")["G_state_dict"])
    G.eval()
    test_ds = LibriTTSRDataset(cfg["data"]["test_filelist"], mode="val")  # test-clean 4824 utt
    result: EvalResult = evaluate(
        G, test_ds,
        metrics=["mcd", "log_f0_rmse", "mrstft", "utmos", "nisqa"],
        post_filter=None,          # GAN は post-filter 不要
        seed=43,
        save_to="eval_results/gan_full.json",
    )
    print(result.summary)          # 論文 Table 1〜3 と相対比較 (T-M4.1 §8.2)

# RTF は GT 不要 (model のみ)。measure_rtf を A100 GPU + 1-core CPU で別途
from wavenext2.eval.measure_rtf import measure_rtf
rtf_gpu = measure_rtf(G, mels_100, device="cuda", n_measure=100)   # 論文 GPU 0.0066 対比
rtf_cpu = measure_rtf(G, mels_100, device="cpu",  n_measure=100)   # 論文 CPU 0.20 対比 (OMP_NUM_THREADS=1)
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `max_steps` | **2,000,000** | docs/implementation-plan.md §5 / T-M2.5 §2.3 |
| wall-clock 見積もり | **A100 単体で約 410 時間** | docs/milestones.md §M6.1 |
| `batch_size` | 16 (OOM 時 8、§6.1) | T-M2.5 §2.3 / T-M5.1 で確定 |
| `segment_length` | 16384 (T-M5.1 で high-res STFT 項を見て確定、24576 もありうる) | T-M2.1 §2.3 (SoT: GAN=16384) / T-M5.1 §8.3 |
| `hop_length` / `n_fft` / `win_length` | 300 / 2048 / 1200 | docs/architecture.md / T-M1.6 |
| lr_g / lr_d / betas / wd | 1e-4 / 2e-4 / [0.8,0.99] / 1e-3 | T-M2.5 §2.3 |
| scheduler | InverseLR (inv_gamma=200000, power=0.5, warmup=0.999) | T-M2.5 §2.3 |
| grad_clip_norm | 1.0 (NaN 復帰時も維持、§6.1) | T-M2.5 §2.3 |
| mel eps | 1e-7 or 1e-5 (T-M5.1 §8.3 で確定したもの) | T-M1.3 §2.3 / T-M5.1 |
| D update 比 | 1:1 or 1:2 (T-M5.1 §8.3 で確定したもの) | T-M2.5 §8.1 / T-M5.1 |
| validation.interval_steps | 10000 (2M step 中 ~200 回) | T-M2.5 §2.3 |
| checkpoint.interval_steps | 10000 (`keep_last_n=5`、best.pt 除外) | T-M2.5 §2.3 |
| validation.num_utterances | 100 (訓練中 validation)、4824 (完走後 full eval) | T-M2.5 §2.3 / docs/milestones.md §M6.1 |
| amp | fp32 default、OOM 時 `--amp` (bf16)、NaN 時は無効化 (§6.1) | T-M2.5 §2.3 |
| EMA | 不使用 | docs/open-questions.md |
| eval metrics (full) | mcd, log_f0_rmse, mrstft, utmos, nisqa | T-M4.1 §9.1 / docs/milestones.md §M6.1 |
| **論文 RTF (要 T 確定)** | **0.0066 (GPU) / 0.20 (CPU) は paper-summary L89/L113 では 5 iter=T=5 の値**。T=4 で測るなら paper-summary の T=4 行を参照に差し替える (§6.1 CRITICAL) | docs/paper-summary.md L89,L113 / docs/milestones.md §M6.1 (要矛盾解消) |
| **論文 param 数 (要 T 確定)** | **74.93M は T=5、T=4 は ~59.94M**。測定 T と一致する行を参照 (§6.1 CRITICAL) | docs/paper-summary.md L89,L113 |
| RTF iter 数 (T) | model config の T。**参照 RTF/param の T と必ず一致させる** (T=4 で測るなら論文 T=4 行、T=5 を見るなら config を T=5) (§6.1 CRITICAL) | T-M4.3 §8.2 / docs/paper-summary.md |
| 論文 Table 一致許容 | ±10% (絶対値) かつ 相対順序 (主軸) | docs/milestones.md §M6.1 / T-M4.1 §8.2 |

### 2.4 アルゴリズム / 処理フロー
1. **(前提) M5.1 gate GO 確認**: user が T-M5.1 の divergence gate 3 指標 + sample audio + 33k step wall-clock 実測を見て **GO/NO-GO 承認** (課金前)。NO-GO なら本チケットを起動しない
2. **(cross-ticket) `synthesize(mel)` alias 確認**: `GANWaveNext2.synthesize(mel)` が無ければ T-M2.4 へフィードバックして追加させる (RTF 測定で必須、§9)
3. **config 確定**: T-M5.1 §8.3 で確定した最適 hyperparameter (eps / segment_length / D update 比 / batch / amp / `auto_oom_fallback`) を `configs/gan_wavenext2.yaml` に反映、max_steps=2M
4. **(user 操作) GPU 環境準備**: A100 環境 (AWS p4 / Lambda Labs / RunPod / 社内クラスタ) の確保・SSH・課金・接続設定
5. `uv run python -m wavenext2.train.train_gan --config configs/gan_wavenext2.yaml` を **`run_in_background=true`** で起動
6. **長時間監視 (数時間〜数日おき)**: TensorBoard ログを `tensorboard --inspect` / scalar CSV export / Monitor (stdout 各行) で確認。divergence gate 3 指標 (loss_G 傾向 / loss_adv ∈ (0.5,2.0) / loss_D ≥ 0.01) + NaN/Inf を継続チェック。background プロセスの生存 (PID/exit code) とログ最終更新時刻の停滞で死活監視
7. **自動 resume**: divergence (NaN) / OOM / GPU preemption (SIGTERM) を検知したら最新の健全 checkpoint から完全 state 復元で resume (§6.1)。NaN 復帰時は lr 半減 / eps=1e-5 / AMP 無効化を config で調整
8. **checkpoint ごと validation**: 10k step ごとに 100 utt validation を `evaluate()` で実行、MR-STFT 最小で `best.pt` を atomic rename 更新
9. **cloud sync**: checkpoint / TensorBoard ログを S3/GCS に定期 sync (preemption / disk full 対策)
10. **完走 (or early stop) 後 full eval**: `best.pt` を `evaluate()` facade で test-clean 4824 utt 評価 → UTMOS / NISQA / MCD / log F0 RMSE を `eval_results/gan_full.json` に永続化
11. **RTF 測定 (T を揃えてから)**: まず `docs/paper-summary.md` L89/L113 で **参照 RTF/param の T (5 iter=T=5)** を確認し、本チケットの T=4 に対しては論文 T=4 行を参照に差し替える (§6.1 CRITICAL)。その上で `measure_rtf(G, mels_100, device="cuda")` (A100) と `device="cpu"` (1-core, OMP_NUM_THREADS=1) を測定、**測定 T と参照行 T が一致した状態で対比** (compile on/off 両報告、T-M4.3 §8.1)
12. **合否判定**: validation MR-STFT プラトー + 客観指標が論文 Table と相対整合 (±10% or 相対順序) + RTF 一致。結果を §8.3 に記録し T-M6.3 / T-M7.1 へ申し送り

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Operator/Monitor | 1 | `train_gan.py` を background 起動・長時間 TensorBoard 監視・divergence/OOM/preemption 時の自動 resume・cloud sync | general-purpose |
| Evaluator | 1 | checkpoint ごと validation + 完走後 full 4824 utt `evaluate()` + RTF 測定 (A100/CPU) + 論文 Table 相対比較 | Explore |
| Reviewer | 1 | Acceptance 検証 + `synthesize` alias cross-ticket 確認 + M6 phase review + T-M6.3/T-M7.1 申し送り作成 | general-purpose |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **条件付き yes** (T-M6.2 (Diff フル訓練) と GPU を共有しなければ並列可。単一 A100 なら順次。Diff は約 32h と軽いので GAN 410h の合間に挟める)
- 並列実行する場合の最大並列数: 2 (T-M6.1 / T-M6.2、GPU が複数あれば)
- 後続 T-M6.3 (ablation) / T-M7.1 (MOS) は本チケットの `best.pt` が gate

## 4. 提供範囲 (Scope)

### In Scope
- 既存 `train_gan.py` の **2M step background 起動・長時間監視・自動 resume** (新規 train コードなし)
- T-M5.1 で確定した最適 hyperparameter の `configs/gan_wavenext2.yaml` への反映 (max_steps=2M)
- **`GANWaveNext2.synthesize(mel)` alias** の T-M2.4 への追加 (期限: best.pt 生成後・RTF 測定前。RTF 測定で必須、訓練起動はブロックしない、cross-ticket)
- divergence (NaN) / OOM / GPU preemption の検知と完全 state 復元 resume
- checkpoint 保存ごとの `evaluate()` validation + best.pt atomic rename 更新
- 完走後の full 4824 utt 客観評価 (`evaluate()` facade、UTMOS/NISQA/MCD/log F0 RMSE)
- RTF 測定 (`measure_rtf`、A100 GPU + 1-core CPU、T=4、compile on/off 両報告)
- 論文 Table 1〜3 との相対比較 (±10% or 相対順序、T-M4.1 §8.2)
- checkpoint / TensorBoard ログの cloud (S3/GCS) sync (薄い script)
- `eval_results/gan_full.json` への評価結果永続化

### Out of Scope
- **新規 train ロジック / loss / model 本体** (すべて T-M2.x で完成済前提、本チケットは起動のみ)
- **MCD / log F0 RMSE / UTMOS / NISQA / RTF の指標実装本体** (T-M4.1 / T-M4.2 / T-M4.3、本チケットは `evaluate()` / `measure_rtf` を呼ぶだけ)
- **Diff フル訓練** (T-M6.2)
- **ablation (T=2,3,4,5 比較 / with-without)** (T-M6.3、本チケットは T=4 single config の best.pt 生成のみ)
- **主観評価 (MOS)** (T-M7.1)
- **multi-GPU DDP / accelerate** (本チケットは single A100。§8.1 でゼロから作り直す場合の代替として記載、本格採用は別 PR)
- **torch.compile / ONNX / TorchScript での高速化** (RTF は compile on/off 報告に留め、最適化本体は別 PR)
- **論文 Table との数値対比レポートの体裁整形** (T-M6.3 で比較表・プロット生成)
- **GPU クラスタ確保 / SSH / 課金 / GO-NOGO 承認** (user 操作、§9.3)

### Deliverable
- ファイル:
  - `checkpoints/gan/best.pt` (訓練成果物、commit しない。T-M6.3 / T-M7.1 のベース)
  - `eval_results/gan_full.json` (full 4824 utt 評価結果、`.gitignore` 済)
  - `configs/gan_wavenext2.yaml` (T-M5.1 確定 hyperparameter 反映済、max_steps=2M)
  - (任意) `scripts/sync_checkpoints.sh` (cloud sync の薄い script)
  - `src/wavenext2/models/gan_wavenext2.py` の `synthesize(mel)` alias (T-M2.4 cross-ticket)
- 関数 / クラス: `GANWaveNext2.synthesize(mel)` (cross-ticket)。それ以外は既存 `train_gan.main` / `evaluate` / `measure_rtf` を再利用
- ドキュメント差分:
  - `docs/milestones.md` §M6.1 Acceptance チェックボックス更新
  - `docs/tickets/index.md` T-M6.1 ステータス + M6 フェーズレビューログ
  - §8.3 に訓練結果 (収束 / 論文対比 / RTF 実測) を追記

## 5. テスト項目

> **本チケットの本体は GPU 必須・長時間の e2e (`@pytest.mark.slow @pytest.mark.gpu`)**。CI には乗らず、user の A100 環境での実起動 + Operator/Evaluator の監視・測定が実体。

### 5.1 Unit テスト (起動前の軽量確認)
- [ ] `configs/gan_wavenext2.yaml` が `load_config` でパース可能、`max_steps == 2_000_000`、必須 key (train/validation/checkpoint/logging/data/model/discriminator) が揃う
- [ ] `configs/gan_wavenext2.yaml` と T-M5.1 の `gan_wavenext2_1epoch.yaml` の diff が **max_steps のみ** (+ T-M5.1 で確定した hyperparameter 反映分)、key drift なし
- [ ] **`GANWaveNext2.synthesize(mel)` が引数 1 個で呼べる** (`synth = getattr(G, "synthesize", G.forward); synth(mel)` が TypeError にならず、`forward(mel, audio_length=None)` 相当の出力)
- [ ] `measure_rtf` の dispatch (`getattr(model, "synthesize", model.forward)`) が `GANWaveNext2` で `synthesize` を選択 (T-M4.3 §5.1 `test_dispatch_*` 相当)

### 5.2 e2e / 結合テスト (GPU 必須・`@pytest.mark.slow @pytest.mark.gpu`、本チケット本体)
- [ ] **2M step を完走 (or early stop)** (A100 約 410h、OOM 時は batch=8 / `--amp` / grad_ckpt で再開)
- [ ] **resume が optimizer / scheduler / step counter / RNG state を完全復元** (同一 step で resume 前後の lr / loss が連続、特に InverseLR が step=1.5M 相当の lr を復元し step 0 に戻らない、§6.1)
- [ ] **validation MR-STFT がプラトー** (収束曲線の移動平均の傾きが閾値以下、右肩下がりから平坦へ遷移)
- [ ] full 4824 utt `evaluate()` で **UTMOS / NISQA / MCD / log F0 RMSE が finite かつ論文 Table と相対整合** (±10% or 相対順序、GT≈0 / 破綻なしの UTMOS 目安 > 3.0)
- [ ] **RTF が論文と一致 (T を揃えてから)**: 参照値 0.0066 (GPU) / 0.20 (CPU) は paper-summary L89/L113 では **5 iter=T=5** の値 → **T=4 で測るなら論文 T=4 行に差し替え**、**T=5 と比べるなら config を T=5 に** (§6.1 CRITICAL)。`measure_rtf(G, device="cuda")` / `device="cpu"` (1-core) を測定 T と参照行の **T が一致した状態** で対比 (同一 GPU 種別 A100 でのみ絶対対比有効、compile on/off 両報告)
- [ ] TensorBoard に loss curve (loss_G/loss_adv/loss_D/各 sub-loss) / sample audio / mel が記録、2M step 全域で NaN/Inf 出現なし (出たら resume で復帰した記録)
- [ ] checkpoint / TensorBoard ログが S3/GCS に sync され、preemption 後も復元可能

### 5.3 Acceptance criteria (`docs/milestones.md` §M6.1 より転記)
- [ ] 2M step 完走 (or early stop)
- [ ] validation MR-STFT loss がプラトー
- [ ] 客観評価: UTMOS, NISQA, MCD, log F0 RMSE が論文 Table 1〜3 と概ね一致 (±10%、ただし backend 差を考慮した相対比較主軸 T-M4.1 §8.2)
- [ ] RTF が論文と一致 (**測定 T と参照行の T を一致させること**。0.0066/0.20 は paper-summary では T=5 の値 → T=4 を測るなら論文 T=4 行を参照、§6.1 CRITICAL)

### 5.4 追加 acceptance (本チケット独自)
- [ ] `GANWaveNext2.synthesize(mel)` alias が実装され RTF 測定 dispatch で動作 (cross-ticket、期限: best.pt 生成後・RTF 測定前)
- [ ] divergence/OOM/preemption からの自動 resume が完全 state 復元で成功した記録が残る
- [ ] `eval_results/gan_full.json` に full 4824 utt の per-utterance + summary が永続化 (論文 Table 対比の再計算回避)
- [ ] T-M6.3 / T-M7.1 へ best.pt と評価結果が §9.1 で明文化

## 6. 懸念事項

> **本チケットは GPU 課金が発生する最重量タスク (A100 約 410h)**。divergence / OOM / preemption での無駄打ち防止と、backend 差による論文 Table 不一致の扱いが要点。

### 6.1 技術的リスク

#### 【CRITICAL】RTF 参照値の T 取り違え (論文と一致と誤判定する)
- **問題**: `docs/paper-summary.md` L89/L113 の **RTF 0.0066 (GPU) / 0.20 (CPU) と 74.93M params は 5 iter (T=5) の値**。一方、本チケット §2.3 (L129)/§5.2 (L213) と `docs/milestones.md` L548 は同じ値を **T=4 の acceptance** にしている。T=4 は ~59.94M params で RTF も iter 数に比例して小さくなるはず。このまま **T=4 で測ると論文の T=5 値 (0.0066/0.20) に届かず「論文と一致せず」と誤判定**する (実際は別 T を比べているだけ)
- **修正方針 (どちらかに統一)**: ① **T=4 を測るなら** `docs/paper-summary.md` の **T=4 行** (≈59.94M, T=4 相当の RTF) を参照値に引く / ② **T=5 との一致を見るなら** model config を **T=5** にして測る。本チケットは「best.pt は T=4 single config」なので原則 ① (T=4 行を参照)、ただし論文 Table と直接対比したいなら ② も検討
- **着手手順**: 本チケット実装時に **`docs/paper-summary.md` L89/L113 の RTF/param 表を再確認し、T と iter 数の対応 (5 iter ⇔ T=? / T=4 ⇔ 何 M params・何 RTF) を確定**してから RTF acceptance を書き直す。確定前に RTF を測定しない (測っても誤判定するため)
- **検知**: 測定した T=4 の params が ~59.94M なのに RTF 参照値が 74.93M (T=5) 由来の 0.0066/0.20 のまま → 取り違えのサイン。params と RTF の T が一致しているか必ずクロスチェック
- **申し送り**: §9 で `docs/milestones.md` L548 / `docs/paper-summary.md` L89,L113 の矛盾解消 (RTF/param が 5 iter か T=4 か) を後続へ明記。両ドキュメントの修正提案を本チケット実装時に出す

#### 【重要】410h 訓練の予算超過
- A100 単体で約 410h (≒ 17 日連続)。クラウドなら課金が大きく、preemption / 障害で再開のたびにコスト増
- **緩和**: ① **checkpoint averaging (SWA)** で終盤の複数 checkpoint を平均し品質を補完 (2M step 完走前でも近い品質を出せる)、② **early stop** (validation MR-STFT プラトー検知で 2M 未満でも打ち切り)、③ cloud sync で再開コスト最小化
- **検知**: validation MR-STFT の移動平均の傾きが閾値以下になったら early stop 候補として user に提示
- **再評価トリガー**: 予算確定時 (§8.1)。fixed-step (2M) でなく budget-cap / early-stop ベースへ切替

#### 【重要】divergence (NaN) からの復帰
- 2M step の長時間では eps=1e-7 の外れ値膨張 (T-M5.1 §6.1 代理指標で監視済) や rare batch で NaN/Inf が出うる
- **緩和**: ① `grad_clip_norm=1.0` が効いているか確認 (維持)、② **AMP (bf16) 無効化** して fp32 に戻す、③ **lr 半減**、④ **eps=1e-5** に上げる — のいずれかを config で調整して最新の健全 checkpoint から resume
- **検知**: TensorBoard で loss_G/loss_adv/loss_D の NaN/Inf、divergence gate 3 指標 (loss_adv ∈ (0.5,2.0) / loss_D ≥ 0.01) の逸脱。Monitor で stdout の `nan` 文字列
- **resume 完全性 (前提)**: T-M5.1 で確認済 (optimizer/scheduler/step/RNG 完全復元、InverseLR warmup 途中でも lr リセットなし)。2M step では resume が頻発するため最重要

#### 【重要】GPU preemption (spot instance)
- コスト削減で spot/preemptible instance を使うと予告付き / 突然の中断が起きる
- **緩和**: T-M2.5 実装済の **SIGTERM/SIGINT handler** (`_emergency_save` → `emergency_step_N.pt`) + **atomic checkpoint** (`best.pt.tmp` → `os.replace`)。preemption 通知 (SIGTERM) で緊急保存 → 別ノードで `--resume emergency_step_N.pt`
- **検知**: プロセスの exit / SIGTERM 受信、ログ最終更新時刻の停滞 (T-M5.1 §6.1 死活監視)。検知したら自動で別ノード起動 + resume
- **注意**: emergency save は通常の checkpoint 間隔 (10k step) とは別。resume 後 step counter が正しく継続するか確認

#### 【重要】backend 差で論文 Table と絶対値が合わない
- MCD は backend / MFCC order / DTW mode で系統差、UTMOS/NISQA は実装 (speechmos vs fairseq) で値が変わる (T-M4.1 §8.2 / T-M4.2)
- **緩和**: **相対比較主軸** (T-M4.1 §8.2)。論文 Table の絶対値一致 (±10%) は努力目標とし、合否は **自系列内の順序関係** (GT≈0、GAN < Diff、with-FM > without-FM 等) で判断。`eval_results/gan_full.json` にどの backend で測ったか記録
- **検知**: 絶対値が ±10% を外れても相対順序が論文と整合すれば pass 寄り。乖離が大きければ backend / mode を T-M4.1 §8.1 に従い再検討
- **補強 (自前ベースライン)**: 論文の絶対値は backend 不明で当てにならないため、**同一 backend で HiFi-GAN / WaveFit を自前で再測**し、GAN-WaveNeXt 2 がその相対序列 (論文 Table の trend) に届くかで判定する方が再現性判定に有効 (§8.2 品質哲学)

#### 【重要】mode collapse / musical-noise の検出 (UTMOS は破綻に鈍感)
- GAN vocoder は長時間訓練で mode collapse / 周期的 musical-noise (metallic artifact) を起こしうるが、**UTMOS / NISQA はこうした局所的破綻に鈍感**で高スコアのまま破綻を見逃す
- **緩和**: gate に **破綻検出を追加** — ① validation sample audio のスペクトログラム目視 (定期的に harmonic/musical-noise を確認)、② MR-STFT の高域成分 / spectral flatness の異常値監視、③ 同一 backend で再測した HiFi-GAN/WaveFit との相対序列に届くか (UTMOS 単独で pass 判定しない)
- **検知**: UTMOS は高いのに MR-STFT 高域が劣化 / sample audio に metallic tone → mode collapse 疑い。検知したら D 強すぎ回復策 (下記) や lr 調整で対処

#### 【重要】Discriminator が強すぎる場合の回復策 (lr 半減のみでは不足)
- 長時間訓練で D が G を圧倒すると loss_adv が (0.5,2.0) を超えて発散方向、G の学習が止まる (divergence gate 逸脱)
- **緩和 (lr 半減だけでなく拡充)**: ① **D の lr を下げる** (lr_d↓、G ではなく D を弱める)、② **feature matching (FM) loss の重みを上げる** (adversarial 項の支配を緩和)、③ **D update 比を 1:2 → 1:1 に戻す** (D の更新頻度を下げる) — を段階的に試して resume。従来の「G lr 半減」と組み合わせる
- **検知**: loss_D が極小 (≪0.01) かつ loss_adv が上振れ → D 優勢。divergence gate 3 指標で監視

#### 【重要】best.pt race condition
- validation improvement 頻発期 (訓練序盤) や cloud sync との競合で `best.pt` が部分書き込み / 破損するリスク (T-M2.4 §6.1)
- **緩和**: T-M2.4/T-M2.5 実装済の **atomic rename** (`torch.save(state, "best.pt.tmp")` → `os.replace("best.pt.tmp", "best.pt")`、POSIX/Windows 共通アトミック)。cloud sync は `best.pt` の os.replace 完了後に行う (sync 中の中間状態を読まない)
- **検知**: `best.pt` ロード時に key 欠落 / shape mismatch が出ないか起動時に確認

#### 【重要】uv.lock cross-platform 問題の継承 (環境再現失敗は致命的)
- T-M0.1 §6 で指摘の通り、**Windows 開発 → Linux A100 で wheel が分岐**する (`uv.lock` は platform-specific marker で解決を分ける)。最重量課金タスク (410h) の起動環境で torch / CUDA wheel の不整合が出ると、起動失敗・無駄打ち・原因切り分けに時間を溶かす
- **緩和**: ① **Linux 上で `uv sync --frozen` を事前検証** (本番起動前に A100 ノードと同 OS/arch で lock が解決し torch+CUDA が入ることを確認)、② **Dockerfile で環境固定** (base image + `uv sync --frozen` を焼き込み、ノード間で同一イメージを使う)。spot で別ノード再起動するたびに環境を作り直す前提なので、再現性は orchestrator の前提条件
- **検知**: 起動時に `import torch; torch.cuda.is_available()` が False / wheel 解決エラー / lock のプラットフォーム分岐警告

#### その他リスク
- **OOM (T=4 × batch=16 × sub-model 4 段)**: 緩和優先順位 ① `--amp` (bf16) → ② `model.enable_grad_ckpt=true` → ③ batch_size=8。T-M5.1 で確定した組み合わせ (+ `auto_oom_fallback`) を踏襲。長時間訓練中に rare に長い utterance で OOM する可能性も監視
- **disk full**: 2M step / 10k step ごと save = 200 ckpt × 数百 MB。`keep_last_n=5` で rolling delete (best.pt 除外) + cloud sync 後にローカル削除
- **checkpoint 容量と S3 転送コスト**: 200 ckpt × 数百 MB + sample audio 数 GB を S3/GCS に sync すると **egress 課金 / sync 帯域** が無視できない。GPU 課金 (§8.1) に隠れがちだが、頻繁 sync + 大量 ckpt で転送費が嵩む。`keep_last_n=5` + best.pt + emergency のみ sync し、全 ckpt を毎回 push しない。sync 間隔と保持数で帯域を制御
- **ログ・checkpoint の cloud sync (S3/GCS)**: 長時間訓練ではローカルディスク / instance 寿命に依存しないよう定期 sync 必須。TensorBoard ログも sync し、gate 判定 / 論文対比に使った scalar が消えないようにする。sync は atomic write 完了後
- **死活監視の通知経路 (人間への到達手段)**: ログ停滞 / プロセス死 / NaN を検知しても、**Claude Code セッションが継続していない時間帯**は誰も気づけない。**Slack / email alert** (ログ最終更新時刻の停滞・exit code・budget alert) を通知経路として用意し、無人時間帯でも課金者 (user) に到達させる。検知だけでなく「誰にどう届くか」を明記 (§8.2 無人運用 orchestrator)
- **RTF が A100 以外で乖離**: ローカル GPU (RTX) / Colab で測ると論文 (A100) と乖離 (T-M4.3 §6.1)。**最終 RTF は A100 上で測定**、`device` 名を記録、同一 GPU 種別でのみ絶対対比。compile on/off 両報告で論文条件を推定。**かつ T (iter 数) を論文の参照行と必ず一致させる** (上記 CRITICAL)
- **`synthesize(mel)` alias の実装期限 (410h 起動を alias でブロックしない)**: T-M2.4 に alias が無いと `measure_rtf` 内の `synth(mel)` が TypeError (T-M4.3 §6.1) になるが、**RTF 測定は 2M step 完走後のタスク**である。alias 未実装を理由に 410h の訓練起動を遅らせるのは設計倒錯 (alias は訓練に不要)。**真の実装期限は「best.pt 生成後・RTF 測定前」**。訓練起動はブロックせず、完走を待つ間に T-M2.4 へ追加すればよい (§9 cross-ticket で期限を訂正)
- **長時間 TensorBoard ログの肥大**: sample audio (24kHz × 数秒 × 4 utt × 200 validation) で数 GB。`num_audio_samples=4` 限定 + global_step 間引き (T-M2.5 §6.1)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 関連項目: すべて確定済み
- 本チケット固有の判断:
  - **MR-STFT プラトー判定の定量化**: validation MR-STFT total の移動平均 (例: 直近 10 validation = 100k step) の傾きが閾値以下を「プラトー」とする。early stop もこの基準を流用
  - **論文 Table ±10% vs 相対順序のどちらを gate にするか**: **相対順序を主軸**、±10% は努力目標 (T-M4.1 §8.2)。backend 差で絶対値が合わなくても相対順序が整合すれば pass 寄りと判断

### 6.3 他チケットとの整合性
- **T-M5.1 (gate)**: 本チケットは T-M5.1 が pass し user が GO を出した config を引き継ぐ。divergence gate を通過済の `gan_wavenext2_1epoch.yaml` の hyperparameter を本番 config に反映 (T-M5.1 §9.1)
- **T-M2.5 (train_gan)**: `main()` CLI / `train_gan_step` / SIGTERM handler / atomic checkpoint / resume 機構をそのまま使う。本チケットは max_steps=2M の config 差分のみ
- **T-M2.4 (GAN model)**: **`synthesize(mel)` alias を best.pt 生成後・RTF 測定前までに追加** (RTF 測定で必須だが完走後タスクなので訓練起動はブロックしない、T-M4.3 §9 / 本チケット §9 cross-ticket)。atomic best.pt rename も T-M2.4 §6.1 で要請済
- **T-M4.1 (evaluate facade)**: `evaluate(model, dataset, metrics, post_filter, *, seed, save_to) -> EvalResult` を full 4824 utt で呼ぶ。論文 Table 対比は相対比較主軸、`eval_results/gan_full.json` 永続化
- **T-M4.3 (RTF)**: `measure_rtf(model, mels, *, device, ...)` を A100 GPU + 1-core CPU で呼ぶ。GAN は `synthesize(mel)` dispatch、iter 数 (T=4) は model config 依存で論文と揃える
- **T-M6.3 (ablation) / T-M7.1 (MOS)**: 本チケットの `best.pt` を T=2,3,4,5 比較のベース / 主観評価サンプル生成に渡す (§9.1)

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 訓練ロジックを新規実装せず T-M2.5 `train_gan.py` を起動している (DRY)
- [ ] 評価が `evaluate()` facade 1 entry point + `measure_rtf` で、グルーコードを書いていない
- [ ] `configs/gan_wavenext2.yaml` が T-M5.1 確定 hyperparameter を反映し max_steps=2M、key drift なし
- [ ] **`GANWaveNext2.synthesize(mel)` alias が best.pt 生成後・RTF 測定前までに T-M2.4 に追加され、RTF dispatch で動作** (alias 未実装で 410h 起動をブロックしていない)
- [ ] 2M step 完走 (or early stop)、validation MR-STFT プラトー
- [ ] divergence/OOM/preemption からの自動 resume が完全 state 復元 (optimizer/scheduler/step/RNG、InverseLR lr 連続)
- [ ] full 4824 utt 客観評価が論文 Table と相対整合 (±10% or 相対順序、自前ベースライン再測との相対序列)、**RTF は測定 T と参照行 T を一致させて対比** (0.0066/0.20 は T=5 値、§6.1 CRITICAL)
- [ ] best.pt が atomic rename、cloud sync が atomic write 完了後
- [ ] M6 phase review の結果が §8.3 に実測値付きで記録
- [ ] T-M6.3 / T-M7.1 への申し送り (best.pt + 評価結果) が §9.1 で具体的
- [ ] 参考実装をコピーしていない (起動・評価のみで新規ロジックなし、`synthesize` alias は薄い wrapper)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M6 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

| 別案 | メリット | デメリット | 採用しなかった理由 | 再評価トリガー |
|---|---|---|---|---|
| **2M step fixed でなく early stop ベース** | validation プラトー検知で打ち切り、410h の無駄を削減・予算を最小化 | 「2M step 完走」の論文準拠感がない、early stop 基準の調整が要る | milestones.md が「2M step (or early stop)」と明記。fixed step は論文再現の基準として明快 | **予算確定時** (§6.1)。クラウド課金が高ければ early-stop / budget-cap を主軸に |
| **multi-GPU DDP (accelerate) で wall-clock 短縮** | 4×A100 で **410h → 約 100h** に短縮、preemption リスク期間も短縮 | DDP 化の実装・デバッグコスト、rank0-only checkpoint / sampler 分割の罠、single-GPU 結果との一致検証 | T-M2.5 は single-GPU 前提 (DDP は別 PR)。本チケットは既存 train_gan.py 起動が原則 | **wall-clock 410h が運用上過大** と判明したとき。accelerate 導入を別 PR で |
| **checkpoint averaging (SWA) を最初から** | 終盤の複数 checkpoint 平均で品質を底上げ・分散低減、early stop と相性良 | SWA の bn 再計算 (vocoder は bn なしなので軽い) / averaging window の調整 | 本チケットは品質補完策 (§6.1) として後付け想定。最初から組むと config が複雑化 | **2M step 完走前に品質を出したい / 予算超過時** (§6.1) |
| **spot instance + 自動 resume orchestrator** | コスト最小 (spot は on-demand の 1/3 程度)、preemption → 別ノード自動 resume で無人運用 | orchestrator (preemption 検知 → ノード起動 → resume) の実装、spot 中断頻度依存 | 本チケットは SIGTERM handler + atomic checkpoint (T-M2.5) で半自動。完全 orchestrator は別途 | **長時間訓練で手動 resume が頻発し煩雑なとき**。T-M6.2 と共通化 |
| **段階的解像度 / progressive segment_length** | 序盤短 segment で高速・終盤長 segment で high-res STFT 改善 (T-M5.1 §6.1 underfit 懸念に対処) | カリキュラム設計が論文外、収束挙動の検証コスト | 論文は固定 segment。T-M5.1 で segment を確定する方針 | **T-M5.1 で high-res STFT 項が underfit と判明し固定 segment で改善しないとき** |
| **cloud provider コスト比較で調達先を選ぶ (重要・桁 2 つ違う)** | 410h の総額が **provider 選定で桁 2 つ動く** (下表)。「2M fixed vs early-stop」の節約より調達先選定が支配的 | provider ごとに SSH/環境/spot 中断挙動が異なる、安価な provider は可用性・信頼性が劣ることも | 本チケットは GPU 調達を user 操作に委ねている (§9.3)。ただし **どの provider を選ぶかでコストが支配的に決まる**ため検討対象に格上げ | **予算確定時 / GPU 調達時** (user 操作)。下表で on-demand/spot を比較 |
| **wall-clock-cap (410h で打ち切り、step は事後記録)** | 「2M step」の典拠は `implementation-plan.md` の独自値のみで論文/WaveFit-PT に 2M の典拠なし。**wall-clock 410h を上限**にすれば課金が確定し step は結果として記録される | step ベースの論文準拠感がない、ハード/provider で 410h あたり step 数が変動 | fixed-step (2M) / early-stop と並ぶ **第 3 軸**。2M の典拠が弱いので有力な代替 | **予算確定時** (§6.1)。課金上限を先に決めたいとき。fixed-step/early-stop と並べて選ぶ |
| **tmux / nohup / systemd で detach 起動** | `run_in_background` は **ローカルプロセス前提で SSH 断とともに死ぬ**。tmux/nohup/systemd なら SSH 切断後も訓練が生き残る | tmux session 管理 / systemd unit 作成の手間、ログ tail の経路設計 (§8.2) | 暗黙に `run_in_background` を使う前提だったが、リモート A100 では SSH 断で死ぬため明示的に detach 手段が要る | **リモート A100 を SSH 越しに起動する全ケース** (実質必須)。§8.2 SSH/remote 監視と一体 |

#### コスト表 (provider × on-demand/spot、A100 80GB、410h 概算)
> 価格は時期・リージョン・在庫で変動する目安。**桁感の把握**が目的 (正確な見積もりは user が調達時に確認)。

| provider / プラン | 単価 (概算) | 410h 総額 (概算) | 備考 |
|---|---|---|---|
| AWS p4de (on-demand, A100 80GB ×8 ノードの 1 枚換算) | ≈ $40/h | ≈ **$16,000** | フルマネージド・高信頼だが最も高い。8GPU ノード課金前提で 1 枚あたりは更に注意 |
| Lambda Labs (on-demand) | ≈ $1.5/h | ≈ **$600** | 単 GPU 借りやすい・中信頼 |
| RunPod / vast.ai (spot/interruptible) | ≈ $0.6–1/h | ≈ **$250–410** | 最安だが preemption 前提 (要 orchestrator + cloud sync)、可用性ばらつき |

- **含意**: spot orchestrator (§8.1 spot 行) + provider 選定で **$16k → $250–410** まで下げられる。**コスト支配要因は step 数より調達先**。budget alert (§8.2) を必ず併設

#### 採用昇格 (M6.1/M6.2/M6.3 で共通化・本チケットで設計確定)
> M6 フェーズレビューで「3 チケットが同じものを 3 回再発明する」と判明したため、別案から採用設計へ格上げ。実装本体は共通先に置き、本チケット (GAN) はそれを利用する側になる。

- **共通 `scripts/orchestrate.py` (launch + state file + resume + cloud sync)**: M6.1/M6.2/M6.3 が **launch + monitor + resume + cloud sync** を三者三様に書くのを防ぐため、共通 orchestrator に抽出。GAN は **1 job として渡す** (config + run_id を渡すだけ)。state file (現在 step / 最新健全 ckpt / cloud run dir) を単一情報源にし、SIGTERM→sync→別ノード再起動を無人で回す (§8.2 無人運用)。T-M6.2/T-M6.3 と共通 (§9)
- **対比レポート生成器を `eval/report.py` に一本化**: T-M4.1 が論文 Table 対比を M6 へ deferred したため、M6.1/M6.2/M6.3 が対比表を三者三様に書く懸念。**`eval/report.py` に昇格** (T-M4.1 facade 側に置く) し、`eval_results/*.json` を読んで論文 Table との相対比較表・プロットを生成する単一実装にする。本チケットはこれを呼ぶだけ (§9)
- **cloud run dir を resume の単一情報源 (`s3://.../runs/<run_id>/`)**: 「local latest vs S3 vs emergency_step_N」のどれが正かの取り違えを避けるため、**cloud run dir (`s3://.../runs/<run_id>/`) を resume の単一情報源**にする。save hook は **atomic write 完了後に push**、resume はこの run dir の最新健全 ckpt を引く。local は cache 扱い。T-M6.2/T-M6.3 と共通 (§9)
- **SWA を採用設計に格上げ**: EMA 不使用方針 (`docs/open-questions.md`) と **SWA は直交** — SWA は **post-hoc の重み平均** (訓練中の shadow パラメータ不要、訓練後に終盤 checkpoint を平均するだけ) であり、EMA を使わない方針に抵触しない。長時間 GAN の **validation 分散を埋める**安価な手段として、品質補完 (§6.1) でなく**標準採用**に格上げ。bn 再計算は vocoder に bn が無いため不要

#### 採用設計
- **既存 `train_gan.py` を共通 `scripts/orchestrate.py` 経由で 1 job として起動** (新規 train コードゼロ、T-M5.1 確定 config を max_steps=2M に拡張)。launch + monitor + resume + cloud sync は orchestrate.py に集約 (M6.1/M6.2/M6.3 共通、上記採用昇格)
- **リモート A100 へは tmux/nohup/systemd で detach 起動** (SSH 断で死なない、§8.1 / §8.2)
- **`evaluate()` facade + `measure_rtf` で full 評価、論文 Table 対比は `eval/report.py` に一本化** (グルーコードを書かない、T-M4.1/T-M4.3 の設計意図に乗る)
- **RTF は測定 T と参照行 T を一致させてから対比** (0.0066/0.20 は paper-summary では T=5 値、§6.1 CRITICAL)
- **唯一の model 側新規コードは `GANWaveNext2.synthesize(mel)` alias** (T-M2.4 cross-ticket、RTF 測定の薄い wrapper、期限は best.pt 生成後・RTF 測定前)
- **論文 Table 対比は相対比較主軸 + 自前ベースライン (HiFi-GAN/WaveFit を同一 backend で再測)** (T-M4.1 §8.2 / §8.2 品質哲学)。backend 差で絶対値が合わなくても相対序列で合否判断
- **divergence/OOM/preemption は SIGTERM handler + atomic checkpoint + cloud run dir からの完全 state resume で対処** (T-M2.5 実装済 + cloud run dir 単一情報源)
- **SWA を標準採用** (終盤 checkpoint の post-hoc 平均で validation 分散を埋める、EMA 不使用方針と直交)。**予算超過は early stop / wall-clock-cap** (fixed 2M を堅持しつつ逃げ道を用意)
- **mode collapse / musical-noise は sample audio + spectral 監視で検出** (UTMOS 単独で pass 判定しない、§6.1)

#### 再評価トリガー
- **予算確定時 / GPU 調達時**: 2M fixed → early stop / **wall-clock-cap (410h)**、single-GPU → multi-GPU DDP、on-demand → spot + orchestrator、**provider 選定 (コスト表で桁 2 つ差)** を判断 (§8.1 表 + コスト表)。budget alert を併設 (§8.2)
- **M6 完了時**: 論文 Table との乖離が backend 起因か実装起因かを判定、乖離が大きければ T-M4.1/T-M4.2 backend を再検討

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切。M6.1 は「実データ full 訓練で論文品質を出す」最重量タスク。size=L (起動 + 長時間監視 + 自動 resume + full eval + RTF) が妥当
- **gate (T-M5.1) との役割分担**: T-M5.1 = divergence の不在 (1 epoch、品質は問わない)、T-M6.1 = 品質 (full 訓練、論文 Table 相対対比)。M6.1 で品質が出なければ T-M5.1 で見落とした hyperparameter / 統合バグを疑う前に、**まず 2M step が本当に必要量か** (early stop 曲線) を確認
- **「絶対値でなく相対比較」哲学**: 論文 Table の絶対値再現は backend 差で困難 (T-M4.1 §8.2)。再現の合否は **自系列内の順序関係** (GT≈0 / GAN < Diff) で判断し、論文値は参考に留める。±10% は努力目標
- **判定者 = Claude Code の品質判定 + user の予算判断**: T-M5.1 の gate (課金前 GO/NO-GO) は user 判断だったが、本チケット起動後の品質判定 (プラトー / 論文対比 / RTF) は Claude Code が下せる。ただし **early stop / 予算超過時の打ち切り判断は user (課金者)** に委ねる
- **CI 不可 (GPU + 410h)**: 本チケットの e2e は A100 で数百時間を要し CI に乗らない。user 環境での実起動 + Operator/Evaluator の監視・測定が実体 (`@pytest.mark.slow @pytest.mark.gpu`)
- **SSH/remote 監視の技術設計 (`run_in_background` / Monitor はローカル前提)**: 本チケットが暗黙に使う `run_in_background` / Monitor は **Claude Code が動くローカルマシンのプロセス前提**で、リモート A100 のプロセスには直接届かない。リモートへどう監視を届けるかを明記する必要がある — ① **ssh wrapper を `run_in_background` で起動し、リモートの tmux/nohup ログを `ssh host tail -f` でローカルに流して Monitor で各行を読む**、または ② **remote tmux + 定期 pull** (cloud run dir の TensorBoard scalar / state file を定期取得して判定)。訓練本体は detach (tmux/nohup/systemd) でローカル SSH 断と切り離す (§8.1)
- **無人運用 orchestrator の思想 (410h = 17 日連続)**: 410h は約 17 日連続で、**Claude Code セッションがその間継続する前提は非現実的**。前提を「セッションは継続しない」に置くなら、**SIGTERM → cloud sync → 別ノード再起動 を無人で回す orchestrator** (cron + state file、`scripts/orchestrate.py` に集約) が本質的に必要。さらに **budget alert (AWS Budgets / 自動停止 lambda)** を「**課金者 = user の安全弁**」として哲学化する — Claude Code の死活監視が届かない時間帯でも、予算上限超過で自動的にインスタンスを止め、暴走課金を防ぐ。検知 (§6.1) と通知経路 (Slack/email) と自動停止 (budget) の三層で無人運用を成立させる
- **品質哲学 (論文絶対値より自前ベースラインとの相対序列)**: 論文 Table の絶対値は backend 不明で再現困難 (T-M4.1 §8.2)。再現性判定にはむしろ **同一 backend で再測した自前ベースライン (HiFi-GAN / WaveFit) との相対序列** の方が有効 — 「GAN-WaveNeXt 2 が自前 HiFi-GAN/WaveFit を論文 Table の trend 通りに上回るか」を主判定にし、論文絶対値は参考に留める。UTMOS は破綻に鈍感 (§6.1) なので、相対序列 + sample audio 目視で総合判定

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- **2M step 完走 vs early stop の実際**: TBD (収束曲線で確定)
- **論文 Table との乖離幅と相対順序の整合**: TBD (full eval 後)
- **RTF 実測 (A100 GPU / 1-core CPU、compile on/off)**: TBD (**測定 T と参照行 T を一致させた上での**論文対比。0.0066/0.20 は paper-summary では T=5 値、§6.1 CRITICAL)
- **paper-summary RTF/param の T (5 iter か T=4 か) 確定結果**: TBD (本チケット実装時に確認し両ドキュメント修正提案、§9)
- **divergence/OOM/preemption の発生頻度と resume の信頼性**: TBD
- 想定外: TBD
- 教訓: TBD

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M6.3 (ablation) へ
- **best.pt を T=2,3,4,5 比較のベースに**: 本チケットの `checkpoints/gan/best.pt` (T=4) を ablation matrix の基準点とする。T-M6.3 は T を変えて再訓練 (or sub-model 段数を変えて) し、論文 Table 1〜3 の trend (T 増で品質向上・RTF 悪化) を再現
- **使用方法**: `configs/gan_wavenext2.yaml` の `model.T` を 2/3/4/5 に変えて `train_gan.py` を再起動、各 best.pt を `evaluate()` + `measure_rtf` で評価。本チケットの T=4 結果を比較表の 1 行として流用
- **評価方法**: `eval_results/gan_full.json` (T=4) を読んで再計算せず比較 (T-M4.1 §9.1)

#### T-M7.1 (MOS) へ
- **生成サンプルを主観評価に**: 本チケットの `best.pt` (T=4) で生成した test-clean サンプルを MOS テスト (20 utt × 6 models) のうち GAN-WaveNeXt 2 (T=4) の系列として提供
- **Acceptance 連携**: T-M7.1 の「GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN」判定に本 best.pt のサンプルを使う

#### ドキュメント矛盾解消 (本チケット実装時に確定・両ドキュメント修正提案)
- **`docs/milestones.md` L548 / `docs/paper-summary.md` L89,L113 の矛盾解消** (RTF/param が **5 iter か T=4 か**): paper-summary L89/L113 では RTF 0.0066/0.20 と 74.93M params が **5 iter (T=5)** の値だが、milestones.md L548 と本チケットはこれを **T=4 acceptance** にしている (§6.1 CRITICAL)。**本チケット実装時 (RTF 測定前) に paper-summary の RTF/param 表を再確認し、T と iter 数の対応を確定**して、milestones.md L548 と paper-summary L89/L113 (および本チケット §2.3/§5.2) を **整合させる修正を提案**する。確定するまで RTF acceptance を「論文と一致」と書かない

#### T-M2.4 へ申し送り (cross-ticket、RTF 測定前に対応必須)
- **`GANWaveNext2.synthesize(mel)` alias を実装** (**期限: best.pt 生成後・RTF 測定前**。RTF は 2M step 完走後のタスクなので 410h 訓練起動は alias 未実装でもブロックしない — 完走を待つ間に追加すればよい): RTF 測定 (`measure_rtf`、T-M4.3) が `getattr(model, "synthesize", model.forward)` で `synth(mel)` を引数 1 個で呼ぶため、`forward(mel, audio_length)` (第 2 引数必須) では TypeError。`synthesize` 側で `audio_length=None` auto-infer を実装すること。これが無いと GAN の RTF 測定が不可 (T-M4.3 §9 / M4 phase review 申し送り)

#### T-M6.2 (Diff フル訓練) / T-M6.3 (ablation) と共通 (M6 で 1 度だけ作る)
- **共通 `scripts/orchestrate.py`**: launch + state file + resume + cloud sync を集約し、GAN/Diff/ablation を各 1 job として渡す (§8.1 採用昇格)。3 チケットで再発明しない。本チケット (最重量 GAN) が最初の利用者になり設計を固める
- **`eval/report.py` (対比レポート生成器)**: `eval_results/*.json` を読んで論文 Table 相対比較表・プロットを生成する単一実装 (T-M4.1 facade 側に昇格、§8.1)。M6.1/M6.2/M6.3 が三者三様に対比表を書くのを防ぐ
- **cloud run dir を resume の単一情報源** (`s3://.../runs/<run_id>/`): 「local latest vs S3 vs emergency_step_N」の取り違え回避。save hook は atomic write 完了後に push、resume は run dir の最新健全 ckpt を引く (§8.1)

#### ユーザー操作 (必須、§9.3 と重複)
- **GPU provider 選定 (コスト支配要因、§8.1 コスト表参照)**: AWS p4de (≈$16k) / Lambda Labs (≈$600) / RunPod・vast.ai spot (≈$250–410) で **410h 総額が桁 2 つ変わる**。on-demand/spot とあわせて user が調達先を選定 (約 410h、spot なら中断前提で orchestrator + cloud sync が前提)
- **budget alert 設定 (課金者の安全弁、§8.2)**: AWS Budgets / 自動停止 lambda 等で予算上限を設定し、無人時間帯の暴走課金を自動で止める。Slack/email 通知経路も併設 (§6.1 死活監視)
- **SSH 認証・課金設定**: user 側で実施。リモートは tmux/nohup/systemd で detach 起動 (§8.1)
- **Claude Code が訓練起動・監視できる接続設定**: detach 起動 + リモートログを tail/pull できる環境 (§8.2 SSH/remote 監視)
- **M5.1 gate の GO/NO-GO 承認**: 課金前に T-M5.1 の divergence gate 結果を見て GO/NO-GO (§8.2)

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M6.1 の Acceptance チェックボックス 4 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M6.1 ステータスを `📝 pending` → `✅ completed`、M6 進捗サマリ + フェーズレビューログ M6 行を更新
  - [ ] §8.3 に訓練結果 (収束 / 論文 Table 相対対比 / RTF 実測 / divergence 頻度) を実測値付きで追記
  - [ ] (該当時) `docs/training.md` §2 に最終 hyperparameter / RTF 実測値を反映

### 9.3 Open question として残ったもの
- **【CRITICAL 起源】RTF/param の参照値は 5 iter (T=5) か T=4 か** — paper-summary L89/L113 では 5 iter の値、milestones.md L548 / 本チケットは T=4 acceptance。本チケット実装時に確定し両ドキュメント修正 (§6.1 CRITICAL / §9.1)
- **2M step が必要量か** (early stop / wall-clock-cap で十分か) — 収束曲線で確認、過剰なら §8.1 early-stop / wall-clock-cap 主軸へ (2M の典拠は implementation-plan.md の独自値のみ)
- **論文 Table との絶対値乖離が backend 起因か実装起因か** — 自前ベースライン (HiFi-GAN/WaveFit 同一 backend 再測) との相対序列が整合すれば backend 起因と判断、乖離大なら T-M4.1/T-M4.2 backend 再検討
- **RTF が A100 で論文と一致するか (T を揃えた上で)** — 0.0066/0.20 は T=5 値なので測定 T と参照行 T を一致させてから対比、compile on/off 両報告で論文条件を推定 (T-M4.3 §8.1)、乖離あれば warmup/sync/iter 数を見直し
- **multi-GPU DDP / spot orchestrator / provider 選定** — wall-clock 410h / 予算が過大なら §8.1 の DDP / spot / コスト表 (桁 2 つ差) を別 PR・user 調達時に検討
