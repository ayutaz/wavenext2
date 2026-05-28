---
id: T-M4.3
title: RTF 測定 (GPU A100 + CPU 1-core)
milestone: M4
phase: M4
status: completed
size: S
owner: claude
created: 2026-05-26
updated: 2026-05-28
depends_on: [T-M2.4, T-M3.1]
blocks: []
related_docs:
  - docs/milestones.md#m43-rtf-測定-srcwavenext2evalmeasure_rtfpy
  - docs/training.md
---

# T-M4.3: RTF 測定 (GPU A100 + CPU 1-core)

> **マイルストーン**: [M4](../milestones.md#m4-評価インフラ-作業量medium3-サブタスク) / **サブタスク**: [M4.3](../milestones.md#m43-rtf-測定-srcwavenext2evalmeasure_rtfpy)
> **依存**: [T-M2.4](T-M2.4-gan-model.md), [T-M3.1](T-M3.1-diff-model.md) / **後続**: なし

## 1. タスク目的とゴール

### 目的
GAN-WaveNeXt 2 / Diff-WaveNeXt 2 の **推論速度 (Real-Time Factor)** を GPU (A100) と CPU (1-core) の両方で計測する関数 `measure_rtf` を実装する。論文 Table 1〜3 の RTF (GAN T=4: GPU 0.0066 / CPU 0.20) との対比を可能にする。

RTF = (推論時間 [秒]) / (合成音声長 [秒])。1 未満ならリアルタイムより速い。

### ゴール
完了したと判断できる具体的な状態:
- [ ] `src/wavenext2/eval/measure_rtf.py` に `measure_rtf(...)` が実装され、`from wavenext2.eval.measure_rtf import measure_rtf` で import 可能
- [ ] **`model.synthesize(mel)` 経由で GAN/Diff を横断計測** (Diff は T-M3.1 で `synthesize` alias 採用済み、GAN は T-M2.4 で `synthesize(mel)` alias 追加が必要 §9、`getattr(model, "synthesize", model.forward)` で dispatch)
- [ ] GPU (CUDA 利用可能時) と CPU (`torch.set_num_threads(1)`) の両方で計測でき、結果を dict (`{"rtf_mean", "rtf_std", "rtf_median", "n", "device", "compiled"}`、T-M4.1 schema 統一) で返す
- [ ] **GPU は CUDA events (`torch.cuda.Event`)**、**CPU は `time.perf_counter` の median** で計測 (warmup 後、§8.1)
- [ ] 100 utterances で median 中心に集計 (CPU は OS スケジューラ揺らぎがあるため median 主指標)
- [ ] `tests/test_measure_rtf.py` が `uv run pytest tests/test_measure_rtf.py` で全 pass
- [ ] `docs/milestones.md` §M4.3 Acceptance criteria 2 項目クリア

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `src/wavenext2/eval/measure_rtf.py`, `tests/test_measure_rtf.py`
- 編集: `src/wavenext2/eval/__init__.py` (`__all__` に `measure_rtf` 追加), `docs/milestones.md` §M4.3, `docs/tickets/index.md`

### 2.2 主要構造

```python
"""measure_rtf.py — GAN/Diff-WaveNeXt 2 の RTF 計測 (GPU A100 + CPU 1-core).

RTF = inference_time[s] / audio_length[s]。論文 Table 1〜3 と対比する。
- model.synthesize(mel) で GAN (fixed-point) / Diff (reverse sample) を横断計測
- GPU: CUDA events (torch.cuda.Event) で GPU 実時間を計測 (perf_counter+sync は同期 overhead を含む)
- CPU: time.perf_counter の median + torch.set_num_threads(1) + OMP_NUM_THREADS=1 (T-M0.1 連携)
- iter 数 (GAN T 回 / Diff 4 step) は model config に従う。n_warmup/n_measure は計測ループ回数 (別概念)
"""
from __future__ import annotations
import time
import statistics
import torch
import torch.nn as nn


@torch.no_grad()
def measure_rtf(
    model: nn.Module,
    mels: list[torch.Tensor] | torch.Tensor,   # 100 utterances (各 (1, 128, T_mel)、論文と同じ長さ分布)
    *,
    device: str = "cuda",                        # "cuda" | "cpu"
    sample_rate: int = 24000,
    hop_length: int | None = None,               # None → model.hop_length
    n_warmup: int = 5,                            # compile 適用時は増やす (§6.1)
    n_measure: int = 100,
    compiled: bool = False,                       # torch.compile 適用済みか (両報告用 §8.1)
) -> dict[str, float | str | int | bool]:
    """各 utterance を model.synthesize(mel) で 1 回ずつ合成し RTF を計測.

    GPU は CUDA events、CPU は perf_counter の median を採用 (§8.1)。
    iter 数 (T 回 / 4 step) は model 側に委ね、本関数は計測のみ。
    Returns:
        {"rtf_mean", "rtf_std", "rtf_median", "n", "device", "compiled"} (T-M4.1 schema 統一)
    """
    synth = getattr(model, "synthesize", model.forward)   # GAN/Diff 横断 dispatch
    #   NOTE: GAN は synthesize(mel) alias 必須 (forward(mel, audio_length) は第 2 引数必須、§6.1)
    model.eval().to(device)
    original_threads = torch.get_num_threads()            # グローバル状態を退避 (§6.1)
    try:
        if device == "cpu":
            torch.set_num_threads(1)                      # CPU 1-core 制限
        # torch.backends.cudnn.benchmark=False 推奨 (autotune 揺れ回避、§8.2)
        hop = hop_length or model.hop_length

        # --- warmup (compile/cache/CUDA context 初期化を計測から除外) ---
        for i in range(n_warmup):
            _ = synth(mels[i % len(mels)].to(device))
        if device == "cuda":
            torch.cuda.synchronize()

        # --- measure ---
        rtfs: list[float] = []
        for i in range(n_measure):
            mel = mels[i % len(mels)].to(device)
            audio_sec = (mel.shape[-1] * hop) / sample_rate
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = synth(mel)
                end.record()
                torch.cuda.synchronize()                  # event 完了待ち
                elapsed = start.elapsed_time(end) / 1000.0  # ms → s
            else:
                t0 = time.perf_counter()
                _ = synth(mel)
                elapsed = time.perf_counter() - t0        # OS スケジューラ揺らぎ → median 併用
            rtfs.append(elapsed / audio_sec)
    finally:
        torch.set_num_threads(original_threads)           # 必ず復元 (§6.1)

    mean = statistics.fmean(rtfs)
    std = statistics.stdev(rtfs) if len(rtfs) > 1 else 0.0
    median = statistics.median(rtfs)
    return {"rtf_mean": mean, "rtf_std": std, "rtf_median": median,
            "n": n_measure, "device": device, "compiled": compiled}
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `sample_rate` | 24000 | docs/architecture.md §3 |
| `hop_length` (GAN) | 300 | docs/architecture.md §3 (HiFi-GAN/WaveFit 揃え) |
| `hop_length` (Diff) | 256 | docs/architecture.md §3 (FastDiff 揃え) |
| `n_warmup` | 5 (compile 適用時は増やす) | §6 GPU warmup / compile |
| `n_measure` | 100 utterances | docs/milestones.md §M4.3 |
| GPU 計時 | CUDA events (`torch.cuda.Event`) | §8.1 (perf_counter+sync は同期 overhead 含む) |
| CPU 計時 | `time.perf_counter` の median | §8.1 (OS スケジューラ揺らぎ) |
| iter 数 (GAN T / Diff step) | model config に従う (T=4 or 5 / 4-step) | §8.2 (計測ループ回数とは別概念) |
| `cudnn.benchmark` | False 推奨 | §8.2 (autotune 揺れ回避) |
| 論文 RTF (GAN T=4) | GPU 0.0066 / CPU 0.20 | docs/milestones.md §M6.1 |

### 2.4 アルゴリズム / 処理フロー
1. `synth = getattr(model, "synthesize", model.forward)` で GAN/Diff 横断 dispatch を決定 (GAN は `synthesize(mel)` alias 必須、§6.1)
2. `original_threads = torch.get_num_threads()` を退避、`model.eval().to(device)`、CPU なら `torch.set_num_threads(1)` (`cudnn.benchmark=False` 推奨)
3. **warmup**: `n_warmup` 回 `synth(mel)` を空回し (計測対象外、compile/CUDA context 初期化を除外)、GPU なら `synchronize()`
4. **measure**: `n_measure` 回ループ。**GPU は CUDA events** (`start.record()` → `synth(mel)` → `end.record()` → `synchronize()` → `start.elapsed_time(end)/1000`)、**CPU は `perf_counter`** で計測 → 経過秒 / 音声長秒 を記録。iter 数は model 側 (T 回 / 4 step) に委ねる
5. `finally:` で `torch.set_num_threads(original_threads)` 復元、mean / std / **median** を集計し schema (`rtf_mean, rtf_std, rtf_median, n, device, compiled`) で返す

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | `measure_rtf.py` 実装 + `tests/test_measure_rtf.py` 記述 | general-purpose |
| Reviewer | 1 | warmup / sync / 1-core 制限の正しさ + GAN/Diff dispatch 整合性確認 | general-purpose |
| Tester | 1 | `uv run pytest tests/test_measure_rtf.py -v` + 実スレッド数確認 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: **yes** (T-M4.1 / T-M4.2 と独立、依存は T-M2.4 / T-M3.1)
- 並列実行する場合の最大並列数: 3 (M4 の 3 チケット並列)

## 4. 提供範囲 (Scope)

### In Scope
- `measure_rtf(model, mels, *, device, ...)` 関数 (GPU + CPU 両対応)
- `synthesize` alias 経由の GAN/Diff 横断 dispatch (`getattr` フォールバック、GAN は `synthesize(mel)` alias 前提)
- GPU は CUDA events 計測、CPU は `perf_counter` median + `torch.set_num_threads(1)` (`finally` で復元)
- mean / std / median 集計と schema 統一返却 (`compiled` 含む)
- `tests/test_measure_rtf.py` (5 節すべて)、`__init__.py` re-export

### Out of Scope
- 訓練・推論ロジック本体 (T-M2.4 GAN / T-M3.1 + T-M3.3 Diff の責務)
- `torch.profiler` による layer 別 breakdown (§8 代替)
- ONNX / TorchScript / torch.compile での高速化測定 (§8 代替)
- batch RTF / streaming 実装 (§6、本チケットは batch_size=1)
- 論文値との最終対比レポート (T-M6.1 / T-M6.2 で実施)

### Deliverable
- ファイル: `src/wavenext2/eval/measure_rtf.py`, `tests/test_measure_rtf.py`, `src/wavenext2/eval/__init__.py` (re-export)
- 関数: `measure_rtf(model, mels, *, device, ...) -> dict`
- ドキュメント差分: `docs/milestones.md` §M4.3 Acceptance チェック, `docs/tickets/index.md` ステータス

## 5. テスト項目

### 5.1 Unit テスト (`tests/test_measure_rtf.py`、CPU テストは default tier / GPU テストは `@pytest.mark.gpu`)
- [ ] `test_cpu_single_thread`: `device="cpu"` で計測中 `torch.get_num_threads() == 1` (実測スレッド数確認、`torch.set_num_threads(1)` が機能)
- [ ] `test_thread_count_restored`: `measure_rtf(device="cpu")` 呼び出し**後**に `torch.get_num_threads()` が呼び出し前の値に戻っている (teardown / `finally` 復元、グローバル状態破壊なし §6.1)
- [ ] `test_rtf_median_finite`: 軽量 mock model で 100 utterances 計測し `rtf_mean` / `rtf_median` が finite かつ正
- [ ] `test_warmup_excluded`: warmup あり/なしで集計値を比較、1st iteration の異常値が除外されている (warmup の `n` 回が計測に含まれない)
- [ ] `test_cuda_events_measurement` (`@pytest.mark.gpu`): `device="cuda"` で **CUDA events (`torch.cuda.Event`) 経由の計測が動作** し RTF が finite (perf_counter ではなく events を使っている)
- [ ] `test_gpu_synchronize` (`@pytest.mark.gpu`): `device="cuda"` で `synchronize()` を経た RTF が finite (非同期 kernel の完了を待っている、sync 漏れの誤差がない)
- [ ] `test_dispatch_gan`: `synthesize` を持たない GAN mock で `getattr(model, "synthesize", model.forward)` が `forward` を選択
- [ ] `test_dispatch_diff`: `synthesize` を持つ Diff mock で `synthesize` が選択される
- [ ] `test_rtf_formula`: 既知の固定 sleep を仕込んだ mock (CPU) で `rtf ≈ sleep_time / audio_sec` を assert (RTF = 推論時間/音声長秒)
- [ ] `test_rtf_schema`: 戻り値が `{rtf_mean, rtf_std, rtf_median, n, device, compiled}` 命名規則に一致 (T-M4.1 schema 統一、`compiled` は bool)

### 5.2 e2e / 結合テスト
- [ ] `test_gan_t4` (`@pytest.mark.slow`): `GANWaveNext2(T=4).eval()` (T-M2.4) で CPU 計測が finite (`synthesize` または `forward` 経由)
- [ ] `test_diff_4step` (`@pytest.mark.slow`): `DiffWaveNext2().synthesize` (= reverse_sample alias, T-M3.1/T-M3.3) で CPU 計測が finite
- [ ] **GAN (T=4) と Diff (4-step) 両方で計測できる** (`synthesize` 横断 dispatch が両モデルで動作)

### 5.3 Acceptance criteria (`docs/milestones.md` §M4.3 より転記)
- [ ] CPU 1-core 制限が `torch.set_num_threads(1)` で機能 (かつ呼び出し後に元の thread 数へ復元)
- [ ] 100 utterances で安定した RTF (median 中心) が出る

### 5.4 テスト戦略
- **mock model**: `nn.Module` に `synthesize` / `forward` + `hop_length` 属性を持たせた軽量スタブで CPU テスト (実 model は重いため slow marker)
- **two-tier**: **GPU テストは `@pytest.mark.gpu`** (CI では skip、self-hosted A100 runner で実行 §8.1)、**CPU テストは default tier** で常時実行。**`@pytest.mark.slow`**: 実 model 結合テスト
- **CI 時間目標**: ~20 秒以内 (mock 中心、GPU 系は CI 対象外)

## 6. 懸念事項

### 6.1 技術的リスク
- **GPU warmup 不足**: 1st iteration は CUDA context 初期化 / カーネルキャッシュ / (torch.compile 時) compile で異常値 → **warmup `n` 回を必ず計測から除外**。`test_warmup_excluded` で検知
- **`torch.cuda.synchronize()` 漏れ**: GPU カーネルは非同期発行のため、sync なしだと `perf_counter()` が発行時刻のみ計測し過小評価 → **計測区間の前後で `synchronize()`**。`test_gpu_synchronize` で検知
- **CPU 1-core 制限が OS / BLAS レベルで効かない**: `torch.set_num_threads(1)` は PyTorch intra-op のみ。OpenMP/MKL は別経路のため **`OMP_NUM_THREADS=1` 環境変数も必要** (T-M0.1 §6 の再現性 env 設定と連携、`scripts` / CI で export)。`test_cpu_single_thread` で実測確認
- **batch_size=1 測定**: RTF は通常 streaming / 1 発話想定のため batch_size=1 で測る。batch 一括処理の throughput は別指標 (§8)
- **A100 がない環境**: ローカル GPU (RTX 系) / Colab で測ると論文 (A100) 値と乖離 → 戻り値に `device` 名を含め、**論文対比は同一 GPU 種別でのみ有効**と明記。最終 RTF 報告は T-M6.1 / T-M6.2 で A100 上で実施 (§8.1 two-tier: GPU は self-hosted A100 runner)
- **`torch.compile` の compile 時間 / warmup 不足**: 適用時は compile が 1 回目に発生 → **warmup に含めて計測から除外**。ただし compile は 1 回で数十秒かかり **`n_warmup=5` では完了に不足しうる** → compile 適用時は `n_warmup` を増やす。compile on/off 両報告 (§8.1)、戻り値 `compiled: bool` (§6.3)
- **`torch.set_num_threads(1)` のグローバル状態破壊**: `torch.set_num_threads` はプロセス全体に効くため、関数内で 1 に設定すると後続のテスト/呼び出しが 1-thread のままになる → **冒頭で `original = torch.get_num_threads()` を退避し、`finally:` 節で `torch.set_num_threads(original)` で復元**。`test_thread_count_restored` で検知
- **GAN synthesize dispatch の引数不一致 (重要、T-M2.4 依存契約)**: `getattr(model, "synthesize", model.forward)` が GAN の `forward(mel, audio_length)` (T-M2.4 で第 2 引数 `audio_length` 必須) に当たると、本関数内の `synth(mel)` 呼び出しが引数 1 個で **TypeError**。→ **GAN 側に `synthesize(mel)` alias (audio_length=None auto-infer) が必要**。M2 実装時に `GANWaveNext2.synthesize(mel)` を追加すること (§9 で T-M2.4 へ申し送り)。それまで GAN 計測は不可
- **`time.perf_counter` の CPU 計測で OS スケジューラ揺らぎ**: CPU wall-time は OS スケジューラの影響で外れ値が混じる → **mean だけでなく median も併記** (`rtf_median`)。CPU 経路は median を主指標とする (§8.1)

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 関連: なし (RTF は純粋な計測タスク、論文事実依存なし)
- **Diff の `synthesize` alias 前提**: T-M3.1 §2.2 で `synthesize = staticmethod(reverse_sample)` (T-M3.3 完成後に有効化) を採用。未有効化期間は `getattr(model, "synthesize", model.forward)` が `forward` にフォールバックするが、Diff の `forward(mel, x_t, c, k)` は単一 sub-model 用で RTF 計測に不適 → **T-M3.3 完了後に Diff 計測を有効化**

### 6.3 他チケットとの整合性
- **T-M2.4 (GAN)**: `model.eval()` 後 `synthesize(mel)` を計測。**現状 GAN は `forward(mel, audio_length)` で第 2 引数必須** → `synthesize(mel)` alias (audio_length auto-infer) を M2 実装時に追加する必要 (§6.1 dispatch 引数不一致、§9 申し送り)。`torch.no_grad()` 内 (本関数 decorator)
- **T-M3.1 / T-M3.3 (Diff)**: `synthesize` (= `reverse_sample` alias) を計測。**T-M3.3 CRITICAL β_t 負値問題** (index.md M3 レビューログ) が未解決だと reverse_sample が NaN → RTF 計測前に T-M3.3 smoke 通過が前提
- **T-M0.1 (Python 環境)**: CPU 1-core には `OMP_NUM_THREADS=1` 環境変数も必要 (§6.1)、再現性 env 設定と連携
- **T-M4.1 (戻り値 schema 統一)**: M4.1 が定義した命名規則に合わせ、戻り値を **`rtf_mean`, `rtf_std`, `rtf_median`, `n`, `device`, `compiled: bool`** に統一 (旧 `n_utterances` → `n`、`rtf_median` / `compiled` を追加)。また本関数は **`evaluate()` facade (T-M4.1) の backend** として呼ばれるが、**RTF は GT 不要で model のみ受ける** (MCD/F0/UTMOS/NISQA は GT 波形が要る) ため、facade からの入力形 (signature) が他メトリクスと異なる点を明記

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:
- [ ] warmup の `n` 回が計測から除外されている (1st iteration 異常値対策、compile 時は warmup 増)
- [ ] **GPU は CUDA events (`torch.cuda.Event`)** で計測 (perf_counter+sync の同期 overhead を避けている)、event 後に `synchronize()`
- [ ] **CPU は `time.perf_counter` の median** を主指標 (OS スケジューラ揺らぎ対策、`rtf_median` 返却)
- [ ] **`torch.set_num_threads(1)` を `finally:` で元の値に復元** (グローバル状態破壊なし)
- [ ] CPU で `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` 連携、`cudnn.benchmark=False` 推奨が明記されている
- [ ] `getattr(model, "synthesize", model.forward)` で GAN/Diff 横断 dispatch (**GAN は `synthesize(mel)` alias 必須**、§9 申し送り確認)
- [ ] iter 数 (T 回 / 4 step) を本関数で制御せず model config に委ねている
- [ ] RTF = 推論時間 / 音声長秒 の計算式が正しい
- [ ] 戻り値 schema が `{rtf_mean, rtf_std, rtf_median, n, device, compiled}` (T-M4.1 統一) に一致
- [ ] Acceptance criteria 全項目クリア、Unit テスト全 pass (GPU 系は `@pytest.mark.gpu` で分離)
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`、`@torch.no_grad()`、`*` で keyword-only)
- [ ] CPU でテストが pass (CUDA 不要、`@pytest.mark.gpu` で分離)
- [ ] 参考実装をコピーしていない

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M4) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら
- **採用設計**: `measure_rtf(model, mels, *, device, ...)` 単一関数 + `synthesize` alias dispatch + warmup/sync。理由: GAN/Diff を同一 API で計測でき (`getattr` 横断)、T-M6.1 / T-M6.2 の最終 RTF 報告が薄い wrapper で済む。
- **採用昇格 (M4 フェーズレビューで代替案 → 採用へ)**:
  - **CUDA events (`torch.cuda.Event`) で GPU 測定**: `time.perf_counter` + `torch.cuda.synchronize()` は CPU↔GPU 同期 overhead を計測値に含めてしまう。CUDA events (`start.record()` / `end.record()` → `synchronize()` → `start.elapsed_time(end)`) が GPU 実時間計測の正攻法。**GPU は CUDA events、CPU は `time.perf_counter` の median** をそれぞれ採用。device ごとに計時経路を分岐する
  - **two-tier 実行設計**: 実 RTF (特に GPU) は **self-hosted A100 runner** で回し、CI では GPU テストを `@pytest.mark.gpu` で skip。A100 不在環境 (ローカル RTX / Colab) では `device` 名を戻り値に記録し **相対値のみ報告** (論文 A100 値との絶対対比はしない)。CPU テストは default tier で常時実行
- **検討追加 (M4 フェーズレビュー、要否は M6 で判断)**:
  - **compile on/off 両方を報告**: 論文 RTF (GPU 0.0066) は ConvNeXt の高速性が売りで、`torch.compile` 有無で 2〜3 倍変わりうる。論文が compile 適用済みか否かは不明なため、**両条件を報告**して論文条件を推定する。戻り値 `compiled: bool` で区別 (§6.3)
  - **utterance 長分布で RTF 正規化**: `n_measure=100` 固定ループだと utterance 長のばらつきで RTF 分散が増える (短 utt は固定 overhead の比率が大きく RTF が膨らむ)。**論文と同じ長さ分布** (test-clean の実分布) で測ることで論文 Table と整合させる
- **代替案 (採用せず)**:
  1. **`torch.profiler` で layer 別 breakdown**: ボトルネック特定に有用だが overhead で RTF 自体が歪む。本チケットは wall-time のみ、profiler は M6 最適化時に別途
  2. **ONNX export → onnxruntime 計測**: deployment 想定の実測値だが export パイプラインが重い。M6 以降に検討
  3. **batch RTF (streaming ではなく batch 一括)**: throughput 指標。RTF (1 発話) とは別物として将来追加

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切 (size=S)。RTF は単一計測関数に閉じる。GAN/Diff の synthesize 統一があるため横断計測が自然
- **インターフェース**: `device` を文字列引数にしたが、将来 multi-GPU では `torch.device` を受ける拡張余地
- **iter 数を model 設定に委ねる前提 (M4 フェーズレビュー)**: RTF を支配するのは反復数 ── GAN は fixed-point T 回 (論文 T=4 or 5)、Diff は 4 step。**同一 T で比較しないと論文 Table と整合しない**。`n_warmup` / `n_measure` (計測ループ回数) とは別概念であり、本関数は **iter 数を制御せず model config に従う** (`synthesize` 内部が T 回 / 4 step を実行) ことを明記。論文対比時は model 側の T を論文と揃える責務が呼び出し側 (T-M6.1/T-M6.2) にある
- **`cudnn.benchmark` の扱い (M4 フェーズレビュー)**: `torch.backends.cudnn.benchmark=True` だと warmup 後も最初の数回 autotune が走り RTF が揺れる (warmup 回数との相互作用)。**RTF 測定時は `benchmark=False` 推奨**。`benchmark=True` を使うなら autotune 完了に十分な warmup 回数を確保すること。§6.1 に技術リスクとして併記

### 8.3 学んだこと (2026-05-28 実装完了)

- **GANWaveNext2.synthesize(mel) alias を本チケットで追加 (T-M2.4 申し送りの実行)**: §6.1/§9 の予告通り、GAN の `forward(mel, audio_length)` は第 2 引数必須ではないが (audio_length=None で auto-infer)、横断 dispatch を明示化するため `synthesize(mel) = forward(mel)` を追加。Diff は infer_diff の import 副作用で `synthesize = reverse_sample` が注入済み。`getattr(model, "synthesize", model.forward)` で両者を統一計測。
- **RTF は GT 不要のため evaluate() facade backend に載せない**: MCD/F0/UTMOS/NISQA は GT 波形ペアを取るが RTF は model のみ。facade の `backend(pairs, sr)` signature と入力形が異なるため standalone 関数として提供 (§6.3 の指摘通り)。T-M6.1/6.2 が直接 `measure_rtf(model, mels)` を呼ぶ。
- **mock で計測ロジックを完全検証 (GPU 不要)**: 固定 sleep mock で RTF 式 (sleep/audio_sec)、thread 観測 mock で 1-thread 制限と finally 復元、synthesize/forward mock で dispatch を CPU default tier で検証。CUDA events 経路は `@pytest.mark.gpu` で deselect (self-hosted A100 runner、§8.1 two-tier)。
- **8 tests pass (slow GAN 結合は deselect)、ruff clean**。
- 教訓: 速度計測は「計測ロジックの正しさ」(warmup 除外・thread 制限・式・dispatch・状態復元) と「実機での絶対値」を分離でき、前者は mock + CPU で網羅、後者は GPU runner に委ねる two-tier が効率的。グローバル状態 (`set_num_threads`) を触る関数は finally 復元 + 復元テストが必須。

#### 再評価トリガー
- **M6 本格訓練後の最終 RTF 報告** (T-M6.1 GPU 0.0066 / CPU 0.20 対比) 時に、A100 実測値と本関数の整合性を再評価。乖離があれば warmup 回数 / sync 位置を見直す

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **インターフェース**: `measure_rtf(model, mels, *, device="cuda"|"cpu", sample_rate=24000, hop_length=None, n_warmup=5, n_measure=100, compiled=False) -> dict`。戻り値 `{"rtf_mean", "rtf_std", "rtf_median", "n", "device", "compiled"}` (T-M4.1 schema 統一)
- **T-M2.4 (GAN) へ申し送り (重要)**: 本関数は `synth(mel)` を引数 1 個で呼ぶため、**`GANWaveNext2.synthesize(mel)` alias が必要** (現状 `forward(mel, audio_length)` は第 2 引数 `audio_length` 必須 → `synth(mel)` が TypeError)。`synthesize` 側で `audio_length=None` auto-infer を実装すること。**M2 実装時に追加**。これが無いと GAN の RTF 計測が不可
- **T-M3.1 (Diff) から**: `DiffWaveNext2.synthesize` (= `reverse_sample` alias, T-M3.3 完成後に有効化) は既に採用済み。そのまま計測対象に渡す
- **T-M4.1 から**: 本関数は `evaluate()` facade の backend として呼ばれ、戻り値 schema (`rtf_mean/rtf_std/rtf_median/n/device/compiled`) を facade の命名規則に統一済み。ただし **RTF は GT 不要で model のみ受ける** (他メトリクスと入力形が異なる)
- **T-M0.1 から**: CPU 1-core 計測には `OMP_NUM_THREADS=1` 環境変数も必要 (`torch.set_num_threads(1)` 単独では BLAS/OpenMP が multi-thread)
- **T-M6.1 / T-M6.2 (本格訓練) へ**: 訓練完了後、本関数で **最終 RTF を論文と対比** (GAN T=4: GPU 0.0066 / CPU 0.20)。**compile on/off 両方を報告** (論文条件推定 §8.1)。**two-tier** で GPU は self-hosted A100 runner、同一 GPU 種別 (A100) でのみ論文対比が有効。論文対比時は model 側の T を論文と揃える (iter 数は model config 依存 §8.2)
- **注意事項 (後続が踏む罠)**: warmup 除外漏れ → 1st iteration 異常値で過大評価 / GPU で CUDA events 未使用 (perf_counter+sync) → 同期 overhead 混入 / `synchronize()` 漏れ → GPU 過小評価 / `OMP_NUM_THREADS` 未設定 → CPU が multi-thread で過小評価 / `set_num_threads` 復元漏れ → 後続テストが 1-thread のまま / GAN `synthesize` alias 欠如 → TypeError

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M4.3 の Acceptance チェックボックス 2 項目
  - [ ] `docs/tickets/index.md` の T-M4.3 ステータスを `📝 pending` → `✅ completed`

### 9.3 Open question として残ったもの
- **Diff の正確な論文 RTF 値**: `docs/training.md` §5.3 に Diff (w/ sub-model, 4-step) の RTF が明記されているか実装時に確認、なければ T-M6.2 で実測のみ報告
- **A100 以外での測定値の解釈基準**: ローカル GPU 測定を論文対比にどう使うか、T-M6 で方針確定
