---
id: T-M6.2
title: Diff-WaveNeXt 2 4 sub-model フル訓練 (A100 約 32h)
milestone: M6
phase: M6
status: pending
size: L
owner: -
created: 2026-05-26
updated: 2026-05-26
depends_on: [T-M5.2]
blocks: [T-M6.3, T-M7.1]
related_docs:
  - docs/milestones.md#m62-diff-wavenext-2-4-sub-model-訓練
  - docs/training.md
---

# T-M6.2: Diff-WaveNeXt 2 4 sub-model フル訓練 (A100 約 32h)

> **マイルストーン**: [M6](../milestones.md#m6-本格訓練-claude-code-は起動監視のみ-wall-clock-a100-で約-442-時間) / **サブタスク**: [M6.2](../milestones.md#m62-diff-wavenext-2-4-sub-model-訓練)
> **依存**: [T-M5.2](T-M5.2-diff-1epoch.md) (前提: T-M3.2, T-M3.3, T-M3.4, T-M4.1) / **後続**: [T-M6.3](T-M6.3-ablation.md), [T-M7.1](T-M7.1-mos-test.md)

## 1. タスク目的とゴール

### 目的
Diff-WaveNeXt 2 の **4 sub-model を本番スケールで独立訓練** (各 1M step、合計約 32 時間 on A100) し、4-step reverse sampling + post-filter を通した **論文 Table 1〜3 相当の客観品質** を再現する。T-M5.2 (1 sub-model 1 epoch divergence gate) が pass した「学習可能性・収束性・OOM 耐性・β 負値解決・conditioning 生存」を前提に、本チケットは **(a) 4 sub-model の本番訓練起動・監視・resume**、**(b) 訓練済み 4 checkpoint の統合ロード**、**(c) post-filter fit (`scripts/fit_post_filter.py`)**、**(d) 推論パイプライン (`reverse_sample(post_filter=fir)`) を組み立て test-clean 全 4,824 utterance を `evaluate()` で評価** を担う。

T-M5.2 から昇格する確認軸:
- M5.2 (gate): 「**1 sub-model** が実データで OOM なし完走 + conditioning 生存 + MSE 収束 + 実 eps_pred t=1→2 2-step reverse で β NaN なし」(divergence gate、M6 課金前の唯一の入口)
- **M6.2 (本番): 「**4 sub-model 全部** が 1M step 完走 + **full 4-step reverse** が NaN なし安定 (β 負値が本番でも解決済み) + post-filter 後の UTMOS/NISQA/MCD/log F0 RMSE が論文と相対整合 + RTF (4-step) が論文と一致」(本番品質・4 sub-model 統合)**

GAN 側 T-M6.1 と異なる点 (sibling、M6 共通の起動・監視・resume 思想は共有):
- 訓練対象が **4 sub-model** (各独立 ckpt `checkpoints/diff/sub_{1,2,3,4}.pt`、順次 or 4 GPU 並列)
- gate ではなく **本番品質再現** (論文値との相対比較が acceptance)
- **config-shape mismatch の統合バグ初検証地点** (T-M5.2 §8.2): M5.2 で検証した config は 1 sub-model 形状 (`only_sub_model=k`)、本チケットで初めて **4 sub-model 一括 config** に統合し、4 ckpt を 1 `DiffWaveNext2` にロード (`from_config(only_sub_model=None)`)
- **post-filter fit を本番 4 sub-model で初実施** (T-M3.4 は実装のみ、M3.5 smoke では未学習 FIR でAPI 確認のみ)

### ゴール
完了したと判断できる具体的な状態 (`docs/milestones.md` §M6.2 Acceptance を内包):
- [ ] **4 sub-model それぞれが 1M step 完走** (or early stop)、各独立 checkpoint `checkpoints/diff/sub_{1,2,3,4}.pt` が生成される (順次 4 回 `--sub-model {1,2,3,4}` or `scripts/train_diff_all.py` orchestrator、4 GPU あれば並列)
- [ ] **4 ckpt の統合ロードが成功**: `DiffWaveNext2.from_config(cfg["model"], only_sub_model=None)` で 4 sub-model 全部を instantiate し、`sub_{1,2,3,4}.pt` の `model_state_dict` から各 `sub_models[k-1]` を正しくロード (config-shape mismatch の統合バグ初検証、§6.1)
- [ ] **full 4-step reverse sampling が NaN / divergence なく安定** (β 負値問題が **本番訓練済み 4 sub-model でも** 解決済み、`reverse_sample(model, mel, seed=43)` で test-clean 全 4,824 utt が NaN なし、§6.1 最優先 blocker)
- [ ] **post-filter fit が完走**: `scripts/fit_post_filter.py --checkpoint-dir checkpoints/diff/` を **seed=43 deterministic** で自動実行し `post_filter/fir.npy` を生成 (本番 4 sub-model で初 fit)、周波数応答が低域 0 dB ± 1 dB / Nyquist 端 +5〜8 dB (T-M3.4 §5.3)
- [ ] **post-filter apply 後の UTMOS / NISQA / MCD / log F0 RMSE が論文と概ね一致** (相対比較主軸、`evaluate(model, test_dataset, metrics=["utmos","nisqa","mcd","log_f0_rmse"], post_filter=fir)` で test-clean 全 4,824 utt 評価、`eval_results/*.json` 永続化、§5.3)
- [ ] **RTF (4-step) が論文と一致** (T-M4.3、GPU / CPU 1-core で測定、論文値との相対整合)
- [ ] TensorBoard に 4 sub-model それぞれの MSE loss / lr / noise level histogram が NaN/Inf なく記録され、各 sub-model の学習曲線が単調減少・プラトー到達 (per-sub-model 単調性で判定、桁差は正常、§6.1)
- [ ] 訓練の中断・再開 (resume) が Adam state / step / RNG state を完全復元し、4 sub-model の順次訓練が偽の連続性なく進む (§6.1)
- [ ] `docs/milestones.md` §M6.2 Acceptance 3 項目クリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規:
  - `scripts/train_diff_all.py` (orchestrator: 4 sub-model を順次 or 並列に `train_diff.main(config, sub_model_k=k)` で起動、各 sub-model の完了状態を state file で管理、resume 対応)
  - `scripts/eval_diff_full.py` (薄い driver: 4 ckpt を `from_config(only_sub_model=None)` で統合ロード → post-filter fit → `evaluate()` で test-clean 4824 utt 評価 → 論文 Table 対比レポート生成)
  - `configs/diff_wavenext2_full.yaml` (**4 sub-model 一括 config**、`max_steps=1_000_000`、本番 batch_size / segment_length、`c_rescale` は M5.2 で確定した値を継承)
- 編集:
  - `docs/milestones.md` §M6.2 Acceptance チェックボックス
  - `docs/tickets/index.md` T-M6.2 ステータス + M6 進捗サマリ
  - `.gitignore` (`post_filter/*.npy` の除外解除は T-M3.4 で済、本チケットで `fir.npy` を実際に commit)
  - (該当時) `docs/training.md` に 4 sub-model 各 1M step の wall-clock 実測値を追記
- **新規モデル / 訓練ロジックコードなし** — T-M3.2 (`train_diff.py` / `train_diff_step`)、T-M3.1 (`DiffWaveNext2.from_config`)、T-M3.3 (`reverse_sample`)、T-M3.4 (`fit_post_filter` / `apply_post_filter`)、T-M4.1 (`evaluate`) を **流用** するのが本チケットの主旨 (M6 は起動・監視・統合・評価のみ)

### 2.2 主要構造

#### `scripts/train_diff_all.py` — 4 sub-model orchestrator (薄い driver)

```python
"""train_diff_all.py — Diff-WaveNeXt 2 の 4 sub-model 本番訓練 orchestrator.

4 sub-model を順次 (single GPU) または並列 (multi-GPU) に T-M3.2 train_diff.main で
起動し、各 sub-model の完了状態を state file で管理する。M6 は起動・監視・resume のみ。

- 順次: --sub-models 1,2,3,4 を 1 GPU で逐次 (合計 ~32h)
- 並列: 4 GPU あれば各 sub-model を別 GPU に割り当て (~8h、§8.1)
- resume: state file (どの sub-model まで完了したか) を読み、未完了 sub-model の
  latest checkpoint から再開 (cloud preemption / 中断対策)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from wavenext2.train.train_diff import main as train_diff_main  # T-M3.2
from wavenext2.utils.config import load_config


def run_all_sub_models(
    config_path: str,
    sub_models: list[int],          # [1, 2, 3, 4] (or サブセットで部分再開)
    parallel: bool = False,         # 4 GPU 並列 (§8.1)
    state_path: Path = Path("checkpoints/diff/_train_state.json"),
) -> None:
    """4 sub-model を順次 or 並列に訓練起動し、完了状態を state file で管理.

    各 sub-model:
      - configs/diff_wavenext2_full.yaml (4 sub-model 一括 config) + only_sub_model=k で
        lazy instantiation (T-M3.2、メモリ 1/4)
      - max_steps=1M、checkpoints/diff/sub_{k}.pt に best を保存 (atomic rename)
      - 中断時は latest step_{step}_sub_{k}.pt から resume (Adam/step/RNG 完全復元)
    """
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}
    for k in sub_models:
        if k in state["done"]:
            continue  # 完了済みは skip (resume)
        # T-M3.2 train_diff.main を sub-model k で起動 (順次は blocking、並列は subprocess)
        train_diff_main(config_path=config_path, sub_model_k=k, resume_path=_latest_ckpt(k))
        state["done"].append(k)
        state_path.write_text(json.dumps(state))
```

#### `scripts/eval_diff_full.py` — 統合ロード + post-filter + 評価 (薄い driver)

```python
"""eval_diff_full.py — 4 sub-model 統合ロード → post-filter fit → test-clean 全評価.

config-shape mismatch の統合バグ初検証地点 (T-M5.2 §8.2): 1 sub-model 形状で
検証した config を 4 sub-model 一括 config に統合し、4 ckpt を 1 DiffWaveNext2 にロード。
"""

from __future__ import annotations
import argparse
from pathlib import Path

import torch
import numpy as np

from wavenext2.models.diff_wavenext2 import DiffWaveNext2     # T-M3.1
from wavenext2.inference.infer_diff import reverse_sample     # T-M3.3
from wavenext2.inference.post_filter import load_post_filter  # T-M3.4
from wavenext2.eval.runner import evaluate, EvalResult        # T-M4.1
from wavenext2.utils.config import load_config


def load_four_sub_models(config_path: str, ckpt_dir: Path, device: str = "cuda") -> DiffWaveNext2:
    """4 ckpt (sub_{1,2,3,4}.pt) を 1 DiffWaveNext2 に統合ロード.

    本番では only_sub_model=None で 4 sub-model 全部 instantiate し、各 sub_{k}.pt の
    model_state_dict から sub_models[k-1] のパラメータをロードする。
    (config-shape mismatch: M5.2 は only_sub_model=k 単体、本チケットで初めて 4 一括)
    """
    cfg = load_config(config_path)
    model = DiffWaveNext2.from_config(cfg["model"], only_sub_model=None).to(device)
    for k in (1, 2, 3, 4):
        ckpt = torch.load(ckpt_dir / f"sub_{k}.pt", map_location="cpu")
        # sub_{k}.pt は only_sub_model=k で訓練されたため model_state_dict は
        # sub_models.{k-1}.* のみを含む。strict=False で該当 sub-model のみロード。
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # missing には他 3 sub-model の key が出る (期待通り)、unexpected は空であること
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diff_wavenext2_full.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/diff/"))
    parser.add_argument("--fir", type=Path, default=Path("post_filter/fir.npy"))
    parser.add_argument("--test-filelist", type=Path, default=Path("data/filelists/test.tsv"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_four_sub_models(args.config, args.checkpoint_dir, args.device)

    # 1. post-filter fit は別途 scripts/fit_post_filter.py (seed=43) で生成済を読む
    fir = load_post_filter(args.fir)

    # 2. test-clean 全 4824 utt を post-filter apply / no-apply 両方で評価
    res_with: EvalResult = evaluate(
        model, test_dataset,
        metrics=["utmos", "nisqa", "mcd", "log_f0_rmse", "rtf"],
        post_filter=fir, seed=43, save_to="eval_results/diff_full_postfilter.json",
    )
    res_without: EvalResult = evaluate(
        model, test_dataset,
        metrics=["utmos", "nisqa", "mcd", "log_f0_rmse"],
        post_filter=None, seed=43, save_to="eval_results/diff_full_nopostfilter.json",
    )
    # 3. 論文 Table 1〜3 との相対対比レポート (T-M6.3 ablation の入力にもなる)
```

#### 4 sub-model の本番訓練起動 (CLI、Claude Code が `run_in_background=true` で起動)

```bash
# 方式 A: orchestrator (推奨、resume / state 管理込み)
uv run python scripts/train_diff_all.py --config configs/diff_wavenext2_full.yaml --sub-models 1,2,3,4

# 方式 B: 4 回直接起動 (順次、single GPU)
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2_full.yaml --sub-model 1
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2_full.yaml --sub-model 2
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2_full.yaml --sub-model 3
uv run python -m wavenext2.train.train_diff --config configs/diff_wavenext2_full.yaml --sub-model 4

# 方式 C: 4 GPU 並列 (§8.1、~8h)
CUDA_VISIBLE_DEVICES=0 uv run python -m wavenext2.train.train_diff ... --sub-model 1 &
CUDA_VISIBLE_DEVICES=1 uv run python -m wavenext2.train.train_diff ... --sub-model 2 &
# ...

# 訓練完了後: post-filter fit (seed=43 deterministic) を自動実行
uv run python scripts/fit_post_filter.py --checkpoint-dir checkpoints/diff/ --reproducible

# 推論 + 評価
uv run python scripts/eval_diff_full.py --checkpoint-dir checkpoints/diff/ --fir post_filter/fir.npy
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `max_steps` (per sub-model) | **1,000,000** | docs/training.md §3.4 (FastDiff `max_updates`) |
| sub-model 数 | **4** | docs/training.md §3.1, Table 1 |
| 訓練時間 (合計) | **約 32 時間** on A100 単体 (= 4 × ~8h/sub-model) | docs/training.md §3.4 [PDF Table 2] |
| Optimizer | `Adam(lr=2e-4, betas=[0.9, 0.98], weight_decay=0.0)` | docs/training.md §3.4, T-M3.2 |
| Scheduler | **不使用** (固定 lr) | docs/training.md §3.4, T-M3.2 |
| Loss | `F.mse_loss(eps_pred, eps)` (noise prediction) | docs/training.md §3.2 (Fig 1b), T-M3.2 |
| `grad_clip_norm` | 1.0 | docs/training.md §3.4 |
| `batch_size` | 20 | docs/training.md §3.4 (FastDiff default) |
| `segment_length` | 25600 | docs/training.md §3.4 (Diff hop=256) |
| `hop_length` | 256 | docs/training.md §1.2 |
| `c_rescale` | M5.2 で確定 (1.0 default / 1000.0) | T-M5.2 §9.1, T-M1.5 |
| amp dtype | fp32 (default) / bf16 (`--amp`、c/abar は fp32 強制) | T-M3.2 §6.1 |
| `NOISE_SCHEDULE_ABAR` | `[1e-4, 2.8e-2, 5.6e-1, 9.1e-1]` | docs/training.md §3.3, T-M3.1 |
| `BAND_BOUNDS` | `[(0.9929,1.0),(0.8246,0.9929),(0.4817,0.8246),(0.0,0.4817)]` | docs/architecture.md §5, T-M3.1 |
| reverse step | DDPM 標準形 + α_t + skip-aware σ (β 負値回避) | docs/training.md §4.2, T-M3.3 |
| post-filter seed | **43** (deterministic) | T-M3.4 (`postfilter_seed`、val seed=42 と非重複) |
| post-filter n_fft / hop / fir_length | 512 / 256 / 512 | T-M3.4, Okamoto21 §3.3 |
| eval metrics | UTMOS, NISQA, MCD, log F0 RMSE, RTF | docs/training.md §5.2, T-M4.1/4.2/4.3 |
| eval 対象 | test-clean 全 **4,824 utterance** | docs/milestones.md §M0.3 / §M4.1 |
| eval seed | 43 (`evaluate()` deterministic) | T-M4.1 |

### 2.4 アルゴリズム / 処理フロー

1. **GPU リソース確保 + GO/NO-GO 承認** (ユーザー操作、M5.2 gate pass が前提、§9)
2. **config 統合**: `configs/diff_wavenext2_1epoch.yaml` (1 sub-model 形状) → `configs/diff_wavenext2_full.yaml` (4 sub-model 一括 config、`max_steps=1M`) に変換 (**config-shape mismatch の統合バグ初検証**、§6.1)
3. **4 sub-model 訓練起動** (Claude Code が `run_in_background=true`): orchestrator (`train_diff_all.py`) or 4 回直接起動。各 sub-model は `only_sub_model=k` で lazy instantiate (T-M3.2、メモリ 1/4)、1M step、`checkpoints/diff/sub_{k}.pt` に best 保存
4. **監視**: 数時間〜数日おきに TensorBoard で各 sub-model の MSE loss / lr / noise level histogram を確認、per-sub-model 単調性 (単調減少・プラトー) を判定 (桁差は正常、§6.1)。divergence / NaN / OOM 検知時は latest checkpoint から自動 resume
5. **4 sub-model 完走後、統合ロード**: `from_config(only_sub_model=None)` で 4 sub-model 全部 instantiate → `sub_{1,2,3,4}.pt` を `strict=False` で各 sub-model にロード (`eval_diff_full.load_four_sub_models`、統合バグ検証)
6. **full 4-step reverse の安定性確認** (最優先 blocker): `reverse_sample(model, mel, seed=43)` を test-clean サブセットで回し **NaN / divergence なし** を確認 (β 負値が本番でも解決済みか)。NaN が出たら T-M3.3 を最優先で再オープン
7. **post-filter fit** (Claude Code が自動実行、seed=43 deterministic): `scripts/fit_post_filter.py --checkpoint-dir checkpoints/diff/ --reproducible` で本番 4 sub-model から `post_filter/fir.npy` を生成、周波数応答検証 (低域 0 dB ± 1 dB / Nyquist +5〜8 dB)、`fir.npy` を git commit
8. **推論パイプライン組み立て + 評価**: `reverse_sample(model, mel, seed=43, post_filter=fir)` で test-clean 全 4,824 utt を合成 → `evaluate()` で UTMOS / NISQA / MCD / log F0 RMSE / RTF を計算、apply / no-apply 両系列を `eval_results/*.json` に永続化
9. **論文 Table 1〜3 との相対対比レポート** 生成 (絶対値でなく自系列内の順序関係・相対整合で合否判定、§6.1)
10. **完了報告**: 変更ファイル一覧 / 通過 acceptance / 既知懸念 / 次マイルストーン (T-M6.3 ablation / T-M7.1 MOS) 概要をユーザーに提示

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Launcher/Monitor | 1 | 4 sub-model 訓練起動 (orchestrator)、TensorBoard 監視、divergence/OOM 検知→resume、post-filter fit 自動実行 | general-purpose |
| Reviewer | 1 | per-sub-model 学習曲線 / full reverse NaN なし / post-filter 周波数応答 / 論文相対整合の確認 + 遡及調査 (T-M5.2 / T-M3.3 / T-M1.5) | general-purpose |
| Evaluator | 1 | 4 ckpt 統合ロード + `evaluate()` で test-clean 4824 utt 評価 + 論文 Table 対比レポート | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **partial** (T-M6.1 GAN フル訓練とは GPU リソース次第。別 GPU / 別時間帯なら並列可、同一単体 GPU では逐次)
- 並列実行する場合の最大並列数: 4 sub-model を **4 GPU で完全並列** すれば 32h → ~8h (§8.1)。single GPU なら 4 sub-model 逐次 (~32h)
- 後続 T-M6.3 (ablation)、T-M7.1 (MOS) とは順次 (本チケットの post-filter 適用済みサンプルが入力)

## 4. 提供範囲 (Scope)

### In Scope
- `scripts/train_diff_all.py` (4 sub-model orchestrator、順次 / 並列起動、state file resume 管理)
- `scripts/eval_diff_full.py` (4 ckpt 統合ロード driver + `evaluate()` 呼び出し + 論文 Table 対比レポート)
- `configs/diff_wavenext2_full.yaml` (4 sub-model 一括 config、`max_steps=1M`)
- 4 sub-model 各 1M step の本番訓練起動・監視・resume (T-M3.2 `train_diff.main` 流用)
- 4 checkpoint (`sub_{1,2,3,4}.pt`) の **統合ロード** (`from_config(only_sub_model=None)` + `strict=False`、config-shape mismatch 検証)
- post-filter fit の自動実行 (`scripts/fit_post_filter.py`、seed=43 deterministic、本番 4 sub-model で初 fit、`fir.npy` commit)
- 推論パイプライン (`reverse_sample(post_filter=fir)`) 組み立て + test-clean 全 4,824 utt を `evaluate()` で評価 (T-M4.1 facade)
- full 4-step reverse の NaN / divergence なし安定性確認 (β 負値が本番でも解決済みの実証)
- 論文 Table 1〜3 との相対対比レポート (`eval_results/*.json` 永続化)

### Out of Scope
- **新規モデル / 訓練ロジック / loss / reverse / post-filter / eval 実装** — すべて T-M3.x / T-M4.x で実装済、本チケットは流用のみ
- ablation (with/without post-filter, with/without sub-modeling, sub-model 数変更) → **T-M6.3** (本チケットは with-sub-model + post-filter の本命系列のみ)
- 主観 MOS テスト → **T-M7.1** (本チケットは post-filter 適用済み生成サンプルを渡すのみ)
- GAN-WaveNeXt 2 フル訓練 → **T-M6.1** (sibling)
- BDDM による noise schedule 学習 (4 値固定スケジュールを使う、§8.1)
- learnable NN post-filter (§8.1、M6.3 検討)
- GPU クラスタ確保・SSH 認証・課金設定 (ユーザー操作、§9)
- DDP / multi-node 訓練 (4 sub-model は独立なので data-parallel 不要、§8.1 の 4 GPU 並列は sub-model ごと 1 GPU)

### Deliverable
- ファイル: `scripts/train_diff_all.py`, `scripts/eval_diff_full.py`, `configs/diff_wavenext2_full.yaml`
- 成果物: `checkpoints/diff/sub_{1,2,3,4}.pt` (4 訓練済み checkpoint)、`post_filter/fir.npy` (本番 FIR、commit)、`eval_results/diff_full_{postfilter,nopostfilter}.json` (評価結果)
- ドキュメント差分: `docs/milestones.md` §M6.2、`docs/tickets/index.md`、(該当時) `docs/training.md` wall-clock 実測値

## 5. テスト項目 (本番訓練のため検証項目)

### 5.1 検証項目 (本番訓練・統合・評価の確認)
- [ ] **4 sub-model 完走、各独立 checkpoint**: `checkpoints/diff/sub_{1,2,3,4}.pt` が 4 つ生成され、各 `model_state_dict` が `sub_models.{k-1}.*` を含む。各 sub-model 1M step 完走 (or プラトー early stop)
- [ ] **4-step reverse が NaN なし安定 (最優先 blocker、β 負値解決確認)**: 統合ロードした 4 sub-model で `reverse_sample(model, mel, seed=43)` を test-clean 全 4,824 utt に対し実行し **NaN / Inf / divergence なし**、出力が `[-1, 1]` 内。β 負値問題 (T-M3.3 CRITICAL) が **本番訓練済み 4 sub-model でも** 解決済みであることの実証 (mock や 2-step では不十分、full 4-step)
- [ ] **config-shape mismatch の統合ロード成功**: `from_config(only_sub_model=None)` + 4 ckpt `strict=False` ロードで、`load_state_dict` の `unexpected` が空、各 sub-model のパラメータが正しくロード (1 sub-model config → 4 sub-model 一括 config の変換が問題なく通る、§6.1)
- [ ] **post-filter fit が deterministic**: `scripts/fit_post_filter.py --reproducible` (seed=43) を 2 回実行して `fir.npy` が bit-exact 一致、`fit_stats.json` に (seed=43, git_sha, checkpoint_sha256) record、`fir.npy` を git commit
- [ ] **post-filter 周波数応答**: `fir.npy` が低域 ≤2 kHz で 0 dB ± 1 dB、Nyquist 端 (12 kHz) で +5〜8 dB (Okamoto21 Fig 3a、T-M3.4 §5.3)
- [ ] **post-filter apply 後の UTMOS/NISQA/MCD/log F0 RMSE が論文と相対整合**: `evaluate(..., post_filter=fir)` の summary が論文 Table 1〜3 の Diff-WaveNeXt 2 (w/ sub-model) 値と相対的に整合 (絶対値でなく順序関係・相対傾向、§6.1)。apply で MCD↓ / 高域 detail 改善を no-apply 系列と比較確認
- [ ] **RTF (4-step) が論文と一致**: T-M4.3 で GPU (A100) / CPU 1-core の RTF を測定、論文値と相対整合 (4-step diffusion の速度特性)
- [ ] **per-sub-model 学習曲線**: 4 sub-model それぞれの MSE loss が TensorBoard で単調減少・プラトー到達。sub-model 1 (c≈1) と sub-model 4 (c≈0.3) で MSE が桁違いなのは **正常**、桁差でなく **各 sub-model の単調性** で判定 (§6.1)
- [ ] **resume 完全性**: 訓練中断→再開で Adam state / step counter / RNG state が完全復元され、4 sub-model の順次訓練が偽の連続性なく進む

### 5.2 e2e
- [ ] 4 sub-model 訓練が OOM / NaN なく完走 (各 exit code 0)
- [ ] `scripts/eval_diff_full.py` が 4 ckpt 統合ロード → post-filter apply → `evaluate()` で `EvalResult` を返し `eval_results/*.json` を生成 (T-M4.1 facade)
- [ ] post-filter apply / no-apply の 2 系列が出て差分 (改善幅) を取れる (T-M3.4 / T-M4.1 連携)

### 5.3 Acceptance criteria (`docs/milestones.md` §M6.2 より転記)
- [ ] 全 sub-model が完走
- [ ] 4-step reverse sampling が安定 (NaN / divergence なし)
- [ ] post-filter 後の UTMOS, NISQA, MCD, log F0 RMSE が論文と概ね一致

## 6. 懸念事項

### 6.1 技術的リスク

#### Critical 項目 (最優先順)

- **CRITICAL #1: β 負値問題が本番でも解決済みか (T-M3.3 CRITICAL、最優先 blocker)**:
  - `β[1]=1-2.8e-2/1e-4=-279` → `1/√(1-β)` で NaN 必発の問題が T-M3.3 で α_t + skip-aware σ により解決され、T-M5.2 gate で **sub-model 1 の実 eps_pred t=1→2 2-step reverse** で NaN なしを確認済み。**本チケットは初めて 4 sub-model 全部訓練済みで full 4-step reverse を回す**ため、sub-model 1→2→3→4 の全遷移 (high noise → low noise) で NaN が出ないことを test-clean 全 4,824 utt で実証する
  - **2-step (M5.2) では不十分な理由**: M5.2 は sub-model 1 のみ訓練して t=1→2 の 1 遷移だけ確認。本番は t=1→2→3→4 の 3 遷移すべてを実 eps_pred で通すため、sub-model 2→3 / 3→4 遷移の σ 計算経路は本チケットで初検証
  - **検知**: full 4-step reverse の出力に NaN/Inf があれば即 NO-GO、T-M3.3 を最優先で再オープン (`_compute_ddpm_coefficients` の σ index / α-based formula を再確認)
  - **gate 順序**: ① full reverse NaN なし → ② post-filter 周波数応答正常 → ③ 論文相対整合 の順。NaN が出れば品質が出ていても NO-GO
  - **post-filter fit 元 (reverse 出力) の安定性が前提 (M6 レビュー追加、CRITICAL #1 に連結)**: FIR は dev 200 utt で fit するが、**fit に使う synth は 4-step reverse (seed=43) の出力**である。reverse が sub-model ばらつき (とりわけ β 負値の残滓や sub-model 4 の不安定) で unstable だと、FIR がそのばらつきを吸収してしまい test で逆効果になる。すなわち「**reverse 出力が安定している**」ことが post-filter fit の前提条件であり、これは CRITICAL #1 (β 負値解決) の解決と明示的に連結する。**gate 順序の ① (full reverse NaN なし) を通過しないと ② (post-filter) に進めない**のはこの依存関係による (NaN がなくても reverse 出力の分散が大きければ FIR fit が不安定になりうる点に注意)

- **CRITICAL #2: config-shape mismatch の統合バグ (T-M5.2 §8.2 から、本チケットが初検証地点)**:
  - M5.2 で検証した `diff_wavenext2_1epoch.yaml` は `sub_model_idx=k` 単体・`only_sub_model=k` lazy instantiation 形状。**本チケットで初めて (a) 4 sub-model 一括 config への変換、(b) 4 ckpt の統合ロード (`from_config(only_sub_model=None)` で 4 sub-model 全部 instantiate + `sub_{1,2,3,4}.pt` を `strict=False` ロード) を行う**ため、1epoch config がそのまま 4 sub-model config に通る保証はない
  - **想定される統合バグ**: ① `sub_{k}.pt` の `model_state_dict` が `sub_models.{k-1}.*` を含むが、`only_sub_model=k` 訓練時の None placeholder が state_dict に影響していないか、② 4 ckpt ロード後に `_validate_band_bounds()` / `synthesize` alias が 4 sub-model 揃った状態で正しく動くか、③ config の sub_model_cfg が 4 個分一括で正しく展開されるか
  - **検知**: `load_state_dict(strict=False)` の `missing` に他 3 sub-model の key が出る (期待通り) / `unexpected` が空であること、統合後に各 sub-model のパラメータが訓練済み値を保持していることを sanity check
  - **再評価トリガー**: 統合ロードで key mismatch / shape error が出たら T-M3.1 (`from_config` / state_dict 構造) を再オープン

#### 通常項目

- **per-sub-model loss scale 不均衡 (T-M3.1 / T-M5.2 §6.1)**: sub-model 1 (c≈1、ε≈x_t) と sub-model 4 (c≈0.3) で MSE が一桁違うのは **正常** (band ごとに noise level が異なるため)。桁差そのものを異常とせず、**各 sub-model の学習曲線の単調性** (単調減少・プラトー到達) で baseline を pin。**lr 調整の余地**: sub-model 1 と 4 で収束速度が大きく違う場合、per-sub-model に lr を変える (e.g., sub-model 4 のみ lr↑) ablation を検討するが、まず一律 lr=2e-4 で本番を回し M5.2 単調性 baseline と比較
- **`c * 1000` rescale が必要だったか (M5.2 の結果次第、T-M1.5)**: M5.2 で sub-model 4 の conditioning が cos<0.5 を満たし `c_rescale=1.0` で十分と確定していればそのまま継承。M5.2 で `c_rescale=1000.0` に切り替えていれば本チケットも 1000.0 で統一。**本番で 4 sub-model 全部に同じ rescale 値を適用** (sub-model 間で rescale を変えると conditioning スケールが不整合になる)
- **post-filter の dev set 200 utt で overfit (T-M3.4)**: FIR は dev set 200 utt の振幅差平均で fit するため、test-clean に対し generalize しない懸念。**検知**: post-filter apply で MCD が **改善でなく悪化** する場合 overfit を疑い、dev set サイズを増やす (T-M3.4 §8.1 で 50/100/200/500 比較) か、低域フラット制約を強める。time-invariant な spectral tilt 補正なので utterance 依存性は本質的に小さいが、本番 4 sub-model の FIR で初めて検証
- **4 sub-model の cloud sync (順次訓練の中断・再開)**: 4 sub-model を 1 GPU で順次 ~32h 回す間に cloud preemption / 課金切れで中断する懸念。`train_diff_all.py` の state file (どの sub-model まで完了したか) + 各 sub-model の latest checkpoint resume で対応。**SIGTERM handler** (T-M3.2 / T-M2.5 `register_sigterm_handler`) で preemption 時に emergency save。checkpoint を cloud storage に定期 sync する運用を推奨
- **state file も cloud sync 対象 (M6 レビュー追加)**: `train_diff_all.py` の state file (どの sub-model まで完了したか) がローカルディスク前提だと、**preemption でインスタンスごと消えて completion 状態をロスト**し、完了済み sub-model を再訓練する無駄打ちが起きる。state file (`checkpoints/diff/_train_state.json`) も checkpoint と同じく cloud storage に sync 対象に含める。spot instance に分散する設計 (§8.1 昇格) では各 sub-model の完了通知を中央 state に集約する必要があるため特に重要
- **checkpoint / eval の cloud 退避 (M6 レビュー追加)**: 本チケットは `fir.npy` の git commit のみ明記しているが、**checkpoint (`sub_{1,2,3,4}.pt`) / eval 結果 (`eval_results/*.json`) の cloud sync が GAN (T-M6.1) ほど明記されていない**。preemption / 課金切れで訓練済み checkpoint を失うと ~8h/sub-model の再訓練になるため、checkpoint・eval も cloud storage に退避する運用を T-M6.1 と共通の `scripts/orchestrate.py` の cloud sync で担保する (§8.1 昇格)
- **uv.lock cross-platform (M6 レビュー追加、T-M6.1 と共通、T-M0.1 §6 継承)**: ローカル開発 (Windows / macOS) と cloud A100 (Linux) で `uv.lock` が解決する wheel が異なる懸念。torch / torchaudio / soundfile 等の platform-specific wheel が Linux で正しく解決されるか、本番起動前に cloud (Linux) 上で `uv sync` を検証する。T-M0.1 §6 から継承し T-M6.1 と共通の検証項目
- **post-filter fit の deterministic 性**: `fit_post_filter.py` 内の `reverse_sample(seed=43)` が deterministic でないと fit が flaky。T-M3.4 で `--reproducible` (seed=43) + `fit_stats.json` の (seed, git_sha, checkpoint_sha256) 3 点 record により担保済。本チケットで 2 回 fit して bit-exact 一致を確認
- **OOM (4 sub-model のいずれか)**: M5.2 で OOM 耐性確認済 (batch_size=20, segment_length=25600, `only_sub_model=k` lazy で 1/4 メモリ)。本番でも `only_sub_model=k` で 1 sub-model のみ instantiate するため OOM リスクは M5.2 と同等。4 GPU 並列時は各 GPU が 1 sub-model なのでメモリは単体と同じ
- **論文 Table の絶対値一致は期待しない (T-M4.1 §8.2)**: MCD は backend / MFCC order / DTW mode で系統差、UTMOS/NISQA は推定器バージョン依存。**相対比較主軸** (GT≈高品質、Diff w/ sub-model + post-filter が w/o より良い、論文の順序関係を再現) で合否判定。`eval_results/*.json` を読んで自系列内の相対傾向で判断

### 6.2 仕様の曖昧さ
- `docs/open-questions.md`: すべて確定済み。本チケットは T-M3.x / T-M4.x で実装・M5.2 で gate 済みのものを本番スケールで回すのみ。新規の仕様判断は **(a) 4 sub-model 一括 config の構造、(b) 4 ckpt 統合ロードの strict 戦略** のみで、いずれも §6.1 CRITICAL #2 / §2.2 で確定

### 6.3 他チケットとの整合性
- **T-M5.2 (1 sub-model 1 epoch gate)** から受領: divergence gate pass (β NaN なし + conditioning 生存 + MSE 収束) が起動の前提。`c_rescale` 確定値 / OOM 回避策 / resume 完全性 / per-sub-model 単調性 baseline を継承。**config-shape mismatch (1 sub-model → 4 sub-model 一括) は M5.2 §9.1 で本チケットを初統合地点と明記**
- **T-M3.2 (train_diff)** から受領: `train_diff.main(config, sub_model_k=k)` / `train_diff_step` / `only_sub_model=k` lazy instantiation / atomic checkpoint / SIGTERM handler を流用
- **T-M3.3 (reverse_sample)** から受領: `reverse_sample(model, mel, *, seed=43, post_filter=None)` で full 4-step + 1-to-1 dispatch、β 負値回避 (α_t + skip-aware σ)。本チケットが β 解決を本番 4 sub-model で実証
- **T-M3.4 (post-filter)** から受領: `scripts/fit_post_filter.py` (seed=43 deterministic) / `apply_post_filter` / `reverse_sample(post_filter=fir)` 統合 / `fir.npy` commit。本番 4 sub-model で初 fit
- **T-M4.1 (evaluate facade)** から受領: `evaluate(model, dataset, metrics=[...], post_filter=fir, seed=43, save_to=...) -> EvalResult` で test-clean 4824 utt を 1 entry point 評価、`eval_results/*.json` 永続化、相対比較主軸
- **T-M6.1 (GAN フル訓練、sibling)** と整合: M6 共通の起動 (`run_in_background`) / 監視 (TensorBoard) / resume / GO-NO-GO 思想を共有。`run_smoke.py` / orchestrator パターンを M5 から継承

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] T-M3.2 / T-M3.3 / T-M3.4 / T-M4.1 を **重複実装せず流用** (orchestrator / eval driver は薄い wrapper、DRY)
- [ ] Acceptance criteria 全項目クリア (§5.3): 4 sub-model 完走 + full reverse NaN なし + post-filter 後の論文相対整合
- [ ] **gate 判定順序の遵守 (§6.1)**: ① full 4-step reverse NaN なし (β 負値解決の本番実証) → ② post-filter 周波数応答正常 → ③ 論文相対整合。NaN が出れば品質が出ていても NO-GO
- [ ] **config-shape mismatch の統合ロード**: `from_config(only_sub_model=None)` + 4 ckpt `strict=False` で `unexpected` 空、各 sub-model パラメータが訓練済み値を保持 (§6.1 CRITICAL #2)
- [ ] **β 負値が本番でも解決済み**: 訓練済み 4 sub-model で full 4-step reverse (t=1→2→3→4) が test-clean 全 4,824 utt で NaN なし (T-M3.3 解決の本番実証)
- [ ] post-filter fit が deterministic (seed=43、2 回 bit-exact 一致)、周波数応答が低域 0 dB / Nyquist +5〜8 dB、`fir.npy` commit 済
- [ ] post-filter apply / no-apply 比較で MCD↓ / 高域改善を確認 (overfit でないこと)
- [ ] per-sub-model 学習曲線が単調減少・プラトー到達 (桁差でなく単調性で判定、§6.1)
- [ ] resume 完全性: Adam state / step / RNG 復元、4 sub-model 順次訓練が偽の連続性なし
- [ ] `evaluate()` facade (T-M4.1) を使用、論文 Table 対比は相対比較主軸 (`eval_results/*.json` 永続化)
- [ ] RTF (4-step) が論文と相対整合 (T-M4.3)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring、`from __future__ import annotations`)
- [ ] 参考実装 (FastDiff) コピーしていない (orchestrator / driver は一から記述)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**M6 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら

> **M6 フェーズレビュー反映 (2026-05-26)**: 以下 3 案を **採用に昇格** (採用設計に統合) — ①共通 `scripts/orchestrate.py` への launch+monitor+resume+cloud sync 抽出 (M6.1/M6.2/M6.3 で 3 回再発明されるのを防ぐ)、②対比レポート生成器を `eval/report.py` に一本化 (T-M6.1 と共通、T-M4.1 facade 側へ昇格)、③4 sub-model を別々の安い single-GPU spot instance に分散 (embarrassingly parallel、DDP 不要)。加えて 2 案を **検討に追加** — post-filter fit の中間品質可視化 / sub-model 4 への step・lr 厚配分。

#### 採用に昇格した設計 (M6 レビュー)

| 昇格案 | 内容 | 効果 | T-M6.1 / 他チケットとの関係 |
|---|---|---|---|
| **共通 `scripts/orchestrate.py` に抽出** | M6.1/M6.2/M6.3 で 3 回再発明される launch+monitor+resume+cloud sync を 1 つの薄い infra 層に抽出。Diff は **4 job として渡す** (`train_diff_all.py` の state-file orchestrator と M6.1 の監視ループを統合) | 重複実装排除 (DRY)、resume / cloud sync ロジックを 1 箇所で保守、GAN/Diff で挙動を揃える | T-M6.1 と共通モジュール。`train_diff_all.py` は `orchestrate.py` に 4 job (sub-model 1-4) を渡す薄い caller に縮小 |
| **対比レポート生成器を `eval/report.py` に一本化** | 論文 Table 1〜3 対比レポート生成を `eval_diff_full.py` 内に閉じず `eval/report.py` に切り出し、T-M6.1 (GAN) と共通化、T-M4.1 facade 側へ昇格 | GAN/Diff で同一フォーマットのレポート、M6.3 ablation も同じ生成器を再利用 | T-M6.1 と共通。T-M4.1 facade の一部として昇格 (`evaluate()` の出力を `report.py` が消費) |
| **4 sub-model を別々の安い single-GPU spot instance に分散** | 4 sub-model 独立 = embarrassingly parallel。DDP 不要、各 ~8h、中断時は **当該 sub-model のみ再投入**。GAN (単一 DDP job) より遥かに spot 親和性が高い | spot 価格でコスト圧縮、preemption の影響を 1 sub-model に局所化 (他 3 sub-model は無傷)、4 GPU 並列より柔軟 (バラバラの安いインスタンスで可) | GAN との **本質的な差**: GAN は DDP で全 GPU 同期が必要で spot 中断に弱いが、Diff は 4 独立 job なので spot で 1 本落ちても局所再投入で済む |
| **BDDM で noise schedule を学習** (4 値固定でなく) | 論文の BDDM noise schedule predictor を完全再現、schedule 最適化で品質↑の可能性 | predictor の再実装コスト大、論文は 4 値を明示しており再現不要 (open-questions.md 確定) | 論文の 4 値固定スケジュールを直接使うのが open-questions.md の確定方針 (predictor 再現不要) | **固定スケジュールで論文品質に届かず schedule が原因と疑われるとき** (M6.3 ablation) |
| **sub-model 数を 2 / 6 / 8 に変える** (ablation) | step 数と品質のトレードオフ探索、論文の 4-step 選択の妥当性検証 | 訓練コスト N 倍、band partition / schedule の再設計が必要 | 論文は 4 sub-model (4-step) を採用、本チケットは本命再現に集中 | **T-M6.3 ablation で sub-model 数の影響を測るとき** |
| **post-filter を learnable NN に** | dev set fit の time-invariant FIR より表現力が高い、utterance 適応も可能 | 訓練コスト増、論文は time-invariant FIR (Okamoto21) を採用、推論 hot path が重くなる | 論文準拠の time-invariant FIR が確定方針 (T-M3.4)、推論速度 (RTF) も維持 | **T-M6.3 で post-filter の品質寄与が小さく learnable で改善余地があるとき** |
| **per-sub-model lr チューニング** (sub-model 1 vs 4 で lr 変更) | loss scale 不均衡 (sub-model 1 と 4 で MSE 桁違い) に対し収束速度を揃えられる | ハイパーパラメータ探索コスト、論文は一律 lr=2e-4 | まず一律 lr で M5.2 単調性 baseline と比較。桁差は正常なので lr 変更は最終手段 | **sub-model 1 or 4 が一律 lr で収束しない / プラトーが極端に遅いとき** (§6.1) |
| **post-filter fit を各 sub-model 完成ごとに暫定 FIR で中間品質可視化** (M6 レビュー検討追加) | 4 本逐次 ~32h で sub-model 4 (最難・最終 denoise 段) の不調が全工程後まで分からないのは無駄打ちリスク。各 sub-model 完成時点で暫定 FIR を fit して中間品質を可視化すれば早期に異常を検知できる | 暫定 FIR は他 sub-model 未完のため代用入力 (x_{k-1} を GT 由来) が必要で、本番 FIR とは別物。可視化用の追加コード | 本番 fit は 4 sub-model 揃ってから (本命)。ただし **sub-model 4 単体の reverse (t=4 のみ、x_3 を GT 由来で代用) を早期 sanity に**回す価値がある (sub-model 4 が最難なため) | **4 本逐次訓練で sub-model 4 の不調を早期検知したいとき / 無駄打ちリスクを下げたいとき** |
| **sub-model 4 に step / lr を厚く配る** (難易度予測に基づく配分、M6 レビュー検討追加) | sub-model 4 は c≈0.3 の低ノイズ域 (= 高周波 detail 担当) で **最難**。「4 本均等に 1M step」でなく難易度予測に基づき sub-model 4 に step / lr を厚配分すれば、最終 denoise 段の品質を底上げできる | 均等配分からの逸脱はハイパーパラメータ探索コスト、過配分で過学習リスク | まず 4 本均等 1M step + 一律 lr で M5.2 baseline と比較。sub-model 4 のプラトーが他より浅い / 高域 detail が出ない場合に厚配分へ | **sub-model 4 が均等配分で品質不足 (高域 detail 不足 / post-filter 過補正破綻) のとき** (§6.1 / §8.2) |
| **checkpoint averaging (SWA)** | 複数 checkpoint 平均で品質を底上げ (M6 予算超過時の品質補完、リスク表) | 4 sub-model それぞれで averaging が必要、best.pt 選択と二重管理 | まず best validation MSE の `sub_{k}.pt` を使う。EMA は不使用 (open-questions.md 確定) | **本番訓練が 1M step 未達で中断し品質補完が必要なとき** (リスク表) |

#### 採用設計
- **4 sub-model を `only_sub_model=k` lazy instantiation で 1 つずつ本番訓練** (T-M3.2 流用、メモリ 1/4)、single GPU では順次 (~32h)、4 GPU あれば並列 (~8h)
- **共通 `scripts/orchestrate.py` に launch+monitor+resume+cloud sync を抽出** (M6 レビュー昇格): M6.1/M6.2/M6.3 横断の薄い infra 層。`train_diff_all.py` は **4 job (sub-model 1-4) を `orchestrate.py` に渡す薄い caller** に縮小。T-M6.1 と共有
- **4 sub-model を別々の安い single-GPU spot instance に分散** (M6 レビュー昇格): embarrassingly parallel、DDP 不要、各 ~8h、中断時は **当該 sub-model のみ再投入** (preemption の影響を局所化)。GAN (DDP) より spot 親和性が高い
- **訓練済み 4 ckpt を `from_config(only_sub_model=None)` + `strict=False` で統合ロード** (config-shape mismatch の初検証)
- **post-filter fit は本番 4 sub-model で初実施** (seed=43 deterministic、`fir.npy` commit)
- **`evaluate()` facade 1 entry point で test-clean 全評価** (T-M4.1、相対比較主軸)
- **対比レポート生成器を `eval/report.py` に一本化** (M6 レビュー昇格): 論文 Table 対比を `eval_diff_full.py` に閉じず `eval/report.py` (T-M4.1 facade 側) に切り出し、T-M6.1 (GAN) / T-M6.3 (ablation) と共通化
- **gate 順序**: ① full reverse NaN なし → ② post-filter 周波数応答 → ③ 論文相対整合 (§6.1)

#### 再評価トリガー
- **T-M6.3 ablation 着手前**: with/without post-filter, with/without sub-modeling, sub-model 数 (2/4/6/8), BDDM schedule 学習, learnable post-filter を比較するか再判断
- **論文品質未達時**: schedule (BDDM) / band partition (strict equal vs point-specialized, architecture.md §5 注記) / post-filter dev set サイズ を §8.1 の代替案で ablation

### 8.2 思想 / 哲学の見直し
- **gate 判定者 = user の GO/NO-GO (T-M5.1 / T-M5.2 と整合)**: M6.2 は A100 32h の課金。M5.2 gate が technical に pass しても、本番 32h (or 4 GPU 8h) を回すかは user の GPU 予算とセットの承認事項。本チケットが technical に完了しても、T-M6.3 ablation まで回すかは予算依存で user が判断
- **config-shape mismatch を本チケットの責務として明示 (T-M5.2 §8.2 から)**: M5.2 は 1 sub-model 形状で gate、**4 sub-model 統合は本チケットで初検証**。「1epoch config がそのまま M6.2 に通る保証はない」ことを §6.1 CRITICAL #2 で明示し、統合バグを本チケットの主要リスクとして扱う。これは「M5 gate が pass = M6 が必ず通る」という誤解を防ぐ責務境界の明確化
- **M5.2 (gate) と M6.2 (本番) の責務分離**: M5.2 = 1 sub-model で学習可能性・OOM・β 解決を **安く** 確認 / M6.2 = 4 sub-model 本番品質 + 統合 + post-filter + 全評価。粒度は適切。本チケットの核は「4 sub-model 統合後の full 4-step reverse が NaN を出さない (β 解決の本番実証)」と「post-filter 後の論文相対整合」
- **本格訓練という位置づけ**: M6.2 は Claude Code が起動・監視・統合・評価を担い、wall-clock の制約 (A100 32h) があるため訓練自体は `run_in_background`。M5.2 が pass = アーキテクチャの正しさは担保済で、本チケットは「論文同等の品質が出るか」を本番スケールで検証する最終段
- **post-filter の責務が M3.4 (実装) と M6.2 (本番 fit) に分離**: M3.4 は API + 未学習 FIR の動作確認、M6.2 で本番 4 sub-model から実 FIR を fit + commit。time-invariant FIR が test-clean に generalize するかは本チケットで初検証 (§6.1 overfit 懸念)
- **sub-model 間の品質ばらつきを許容する哲学 (M6 レビュー追加)**: 4 sub-model は band ごとに noise level が異なり MSE が桁違いになる (§6.1)。これを異常とせず **各 sub-model の単調性で判定** する一方、品質ばらつき自体は許容する。ただし **sub-model 4 (最難・最終 denoise 段、高周波 detail 担当) が弱いと post-filter が過補正で破綻する依存関係** を明記する: post-filter は reverse 出力の spectral tilt を補正するが、sub-model 4 由来の高域不足を FIR が過剰に持ち上げると test で逆効果になる (§6.1 CRITICAL #1 / overfit 懸念と連結)。「ばらつき許容」と「sub-model 4 だけは品質を担保する」は両立する責務境界
- **GAN (T-M6.1) とのインフラ共通化 (M6 レビュー追加)**: orchestrator (`scripts/orchestrate.py`) / cloud sync / 通知を GAN・Diff 両チケットで共有する **薄い infra 層** として M6 横断で設計する。M6.1/M6.2/M6.3 で launch+monitor+resume+cloud sync を 3 回再発明しない (§8.1 昇格)。GAN は DDP 単一 job、Diff は 4 独立 job という **訓練形態の差は orchestrator が job リストとして吸収** し、その上の起動・監視・resume・通知ロジックは共通化する

### 8.3 学んだこと (チケット完了後に追記)
- (実装完了後に追記)
- **β 負値が本番 4 sub-model でも解決済みか**: TBD (full 4-step reverse t=1→2→3→4 で test-clean 全 utt NaN なしを確認)
- **config-shape mismatch の統合バグ有無**: TBD (4 ckpt `strict=False` 統合ロードで unexpected 空・パラメータ保持を確認)
- **`c * 1000` rescale が最終的に必要だったか**: TBD (M5.2 確定値を本番で継続使用し品質確認)
- **post-filter が test-clean に generalize したか**: TBD (dev 200 utt fit の FIR で apply/no-apply の MCD 差を確認)
- **4 sub-model 本番 wall-clock 実測**: TBD (single GPU 順次 ~32h / 4 GPU 並列 ~8h の実測、docs/training.md に追記)
- **論文 Table 1〜3 との相対整合**: TBD (UTMOS/NISQA/MCD/log F0 RMSE/RTF の自系列内順序関係)
- 想定外: TBD
- 教訓: TBD
- 再評価トリガー: **T-M6.3 ablation 着手前** (post-filter / sub-modeling / schedule の代替案を ablation するか再判断)

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報

#### T-M6.1 (GAN フル訓練、sibling) と共通 (M6 レビュー追加)
M6 横断で共有する薄い infra 層 (§8.1 / §8.2 昇格)。GAN・Diff のどちらが先に着手しても、以下は **両チケットで共通実装・共通検証** する:
- **`scripts/orchestrate.py`**: launch+monitor+resume+cloud sync の共通 orchestrator。Diff は 4 job (sub-model 1-4)、GAN は 1 DDP job を job リストとして渡す。`train_diff_all.py` はこれを呼ぶ薄い caller に縮小
- **`eval/report.py`**: 論文 Table 1〜3 対比レポート生成器を一本化 (T-M4.1 facade 側へ昇格)。GAN/Diff/ablation (M6.3) が同一フォーマットを再利用
- **cloud run dir**: checkpoint / state file / eval 結果 / log を退避する cloud storage の run ディレクトリ構成を GAN・Diff で共通化 (preemption 耐性、§6.1)
- **uv.lock Linux 検証**: cloud A100 (Linux) 上で `uv sync` が正しく解決するかを本番起動前に両チケットで検証 (T-M0.1 §6 継承、§6.1)

#### T-M6.3 (ablation) へ
> 本チケットの **4 sub-model (point-specialized) + post-filter 本命系列が M6.3 ablation の比較ベース** (with/without post-filter, with/without sub-modeling の基準点)。M6.3 はここから派生系列を引く。
- **with/without post-filter 比較**: 本チケットで `evaluate(..., post_filter=fir)` と `post_filter=None` の 2 系列を `eval_results/diff_full_{postfilter,nopostfilter}.json` に永続化済。M6.3 はこれを読んで post-filter の品質寄与を定量化 (再計算不要)
- **with/without sub-modeling 比較**: 本チケットの 4 sub-model (point-specialized) を baseline とし、M6.3 で single-model (sub-modeling なし) / strict equal partition (Okamoto21、architecture.md §5 注記) と対比
- **訓練済み 4 ckpt + FIR**: `checkpoints/diff/sub_{1,2,3,4}.pt` + `post_filter/fir.npy` を ablation の起点に流用
- **sub-model 数 ablation (2/4/6/8)**: band partition / schedule の再設計が必要 (§8.1)、本チケットの 4 を基準
- **per-sub-model 単調性 baseline**: 4 sub-model の MSE 学習曲線 (桁差は正常、単調性で pin) を ablation 比較の baseline に

#### T-M7.1 (MOS) へ
- **post-filter 適用済み生成サンプル**: `reverse_sample(model, mel, seed=43, post_filter=fir)` で生成した test-clean サンプルを MOS テスト用に提供。Diff-WaveNeXt 2 (w/ sub-model + post-filter) が MOS テストの本命系列
- **deterministic 生成**: seed=43 固定で MOS 用サンプルが再現可能 (評価者間で同一サンプル)
- **acceptance 連携**: T-M7.1 の「Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)」は本チケットの post-filter 適用済みサンプルで評価

#### ユーザー操作 (必須)
- **GPU クラスタ確保 + GO/NO-GO 承認**: A100 単体 ~32h (or 4 GPU 並列 ~8h)。M5.2 gate pass 後、本番訓練 (課金) へ進むかは user の GPU 予算とセットの承認事項 (M6.1 と同様)。SSH 認証・課金設定・接続設定は user 側で実施

#### 失敗時のフィードバック方向 / 遡及調査優先順位 (acceptance を満たせない場合、最優先順)
1. **β 負値問題 (T-M3.3)** — full 4-step reverse が NaN を出す場合。本番訓練済み 4 sub-model で sub-model 1→2→3→4 の全遷移を確認したうえで `_compute_ddpm_coefficients` の σ index / α-based formula を再オープン (最優先 blocker)。**reverse 出力の安定は post-filter fit の前提** (§6.1 CRITICAL #1 連結): reverse が unstable だと FIR がばらつきを吸収し test で逆効果になるため、post-filter の不調 (5) を疑う前にまず reverse 安定 (β 解決) を確認する
2. **T-M5.2** — 1 sub-model gate に戻り、本番 config (4 sub-model 一括) の統合バグ / `c_rescale` / OOM / resume 完全性を再確認。**β 負値 / reverse 安定が post-filter fit の前提**であることを M5.2 gate と連結して再評価
3. **noise embedding rescale (T-M1.5)** — conditioning が本番でも効かない場合 (`c_rescale` 1.0 → 1000.0 を 4 sub-model 一律で再訓練)。sub-model 4 (低ノイズ域) の conditioning が効かないと reverse が不安定→post-filter 過補正破綻 (§8.2) に連鎖するため、rescale も post-filter fit の前提要因
4. **T-M3.1** — 4 ckpt 統合ロードで key/shape mismatch が出る場合 (`from_config(only_sub_model=None)` / state_dict 構造)
5. **T-M3.4** — post-filter が overfit / 周波数応答が範囲外の場合 (dev set サイズ / 集約軸 / clip フォールバック)。**ただし post-filter の不調は reverse 出力の不安定 (1/2/3) が真因のことがある** (§6.1 CRITICAL #1): post-filter 単体を疑う前に reverse 出力の安定性 (β 解決・rescale・sub-model 4 品質) を先に切り分ける

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M6.2 Acceptance チェックボックス 3 項目を ☑ に更新
  - [ ] `docs/tickets/index.md` の T-M6.2 ステータス更新 + M6 進捗サマリ
  - [ ] `docs/training.md` に 4 sub-model 各 1M step の wall-clock 実測値 (single GPU 順次 / 4 GPU 並列) を追記
  - [ ] (該当時) `docs/tickets/index.md` のフェーズレビューログ M6 行を更新

### 9.3 Open question として残ったもの
- 1 sub-model config → 4 sub-model 一括 config への変換 / 4 ckpt 統合ロードが問題なく通るか → 本チケットで初統合・初検証 (§6.1 CRITICAL #2)
- β 負値が本番訓練済み 4 sub-model の full 4-step reverse でも解決済みか → 本チケットで test-clean 全 utt 実証 (§6.1 CRITICAL #1)
- `c * 1000` rescale が本番でも必要か → M5.2 確定値を継続使用し品質で確認
- time-invariant post-filter が test-clean に generalize するか (dev 200 utt fit) → apply/no-apply 比較で確認、overfit なら dev set 拡張 (§6.1)
- 論文 Table 1〜3 の絶対値に届くか / 相対傾向の再現で十分か → 相対比較主軸 (§6.1)、乖離が大きければ §8.1 代替案 (schedule / partition / post-filter) を T-M6.3 で ablation
- `docs/open-questions.md` への追記要否: acceptance 未達時のみ再オープン (現時点では不要)
