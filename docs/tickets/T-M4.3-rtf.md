---
id: T-M4.3
title: RTF 測定 (GPU A100 + CPU 1-core)
milestone: M4
phase: M4
status: pending
size: S
owner: -
created: 2026-05-26
updated: 2026-05-26
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
- [ ] **`model.synthesize(mel)` 経由で GAN/Diff を横断計測** (T-M3.1 で `synthesize` alias 採用済み、`getattr(model, "synthesize", model.forward)` で dispatch)
- [ ] GPU (CUDA 利用可能時) と CPU (`torch.set_num_threads(1)`) の両方で計測でき、結果を dict (`{"rtf_mean", "rtf_std", "device", "n_utterances", ...}`) で返す
- [ ] GPU warmup (最初の数 iteration を除外) + `torch.cuda.synchronize()` で正確な計測
- [ ] 100 utterances 平均で **stable** (`rtf_std < rtf_mean * 0.1`)
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
- GPU: warmup 後 torch.cuda.synchronize() で正確に
- CPU: torch.set_num_threads(1) + OMP_NUM_THREADS=1 (T-M0.1 連携) で 1-core 制限
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn


@torch.no_grad()
def measure_rtf(
    model: nn.Module,
    mels: list[torch.Tensor] | torch.Tensor,   # 100 utterances (各 (1, 128, T_mel))
    *,
    device: str = "cuda",                        # "cuda" | "cpu"
    sample_rate: int = 24000,
    hop_length: int | None = None,               # None → model.hop_length
    n_warmup: int = 5,
    n_measure: int = 100,
) -> dict[str, float | str | int]:
    """各 utterance を model.synthesize(mel) で 1 回ずつ合成し RTF を計測.

    Returns:
        {"rtf_mean", "rtf_std", "device", "n_utterances", "n_threads", "stable"}
    """
    synth = getattr(model, "synthesize", model.forward)   # GAN/Diff 横断 dispatch
    model.eval().to(device)
    if device == "cpu":
        torch.set_num_threads(1)                          # CPU 1-core 制限
    hop = hop_length or model.hop_length

    # --- warmup (1st iteration の compile/cache 効果を計測から除外) ---
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
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = synth(mel)
        if device == "cuda":
            torch.cuda.synchronize()                      # 非同期 kernel の完了待ち
        rtfs.append((time.perf_counter() - t0) / audio_sec)

    t = torch.tensor(rtfs)
    mean, std = t.mean().item(), t.std().item()
    return {"rtf_mean": mean, "rtf_std": std, "device": device,
            "n_utterances": n_measure, "n_threads": torch.get_num_threads(),
            "stable": std < mean * 0.1}
```

### 2.3 使用するハイパーパラメータ / 定数

| 名前 | 値 | 出典 |
|---|---|---|
| `sample_rate` | 24000 | docs/architecture.md §3 |
| `hop_length` (GAN) | 300 | docs/architecture.md §3 (HiFi-GAN/WaveFit 揃え) |
| `hop_length` (Diff) | 256 | docs/architecture.md §3 (FastDiff 揃え) |
| `n_warmup` | 5 (最初の数 iteration を除外) | §6 GPU warmup |
| `n_measure` | 100 utterances | docs/milestones.md §M4.3 |
| stable 閾値 | `rtf_std < rtf_mean * 0.1` | docs/milestones.md §M4.3 |
| 論文 RTF (GAN T=4) | GPU 0.0066 / CPU 0.20 | docs/milestones.md §M6.1 |

### 2.4 アルゴリズム / 処理フロー
1. `synth = getattr(model, "synthesize", model.forward)` で GAN/Diff 横断 dispatch を決定
2. `model.eval().to(device)`、CPU なら `torch.set_num_threads(1)`
3. **warmup**: `n_warmup` 回 `synth(mel)` を空回し (計測対象外)、GPU なら `synchronize()`
4. **measure**: `n_measure` 回ループ。各回 `synchronize()` → `perf_counter()` → `synth(mel)` → `synchronize()` → 経過時間 / 音声長秒 を記録
5. mean / std を集計、`stable = std < mean * 0.1` を返す

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
- `synthesize` alias 経由の GAN/Diff 横断 dispatch (`getattr` フォールバック)
- GPU warmup + `torch.cuda.synchronize()`、CPU `torch.set_num_threads(1)`
- mean / std 集計と `stable` 判定
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

### 5.1 Unit テスト (`tests/test_measure_rtf.py`)
- [ ] `test_cpu_single_thread`: `device="cpu"` で `torch.get_num_threads() == 1` (実測スレッド数確認、`torch.set_num_threads(1)` が機能)
- [ ] `test_rtf_stable`: 軽量 mock model で 100 utterances 計測し `result["stable"] is True` (`rtf_std < rtf_mean * 0.1`)
- [ ] `test_warmup_excluded`: warmup あり/なしで mean を比較、1st iteration の異常値が除外されている (warmup の `n` 回が計測に含まれない)
- [ ] `test_gpu_synchronize` (`@pytest.mark.gpu`): `device="cuda"` で `torch.cuda.synchronize()` を経た RTF が finite かつ非同期計測より大きい (sync 漏れの誤差がない)
- [ ] `test_dispatch_gan`: `synthesize` を持たない GAN mock で `getattr(model, "synthesize", model.forward)` が `forward` を選択
- [ ] `test_dispatch_diff`: `synthesize` を持つ Diff mock で `synthesize` が選択される
- [ ] `test_rtf_formula`: 既知の固定 sleep を仕込んだ mock で `rtf == sleep_time / audio_sec` を assert (RTF = 推論時間/音声長秒)
- [ ] `test_return_dict_keys`: 戻り値に `rtf_mean / rtf_std / device / n_utterances / stable` が含まれる

### 5.2 e2e / 結合テスト
- [ ] `test_gan_t4` (`@pytest.mark.slow`): `GANWaveNext2(T=4).eval()` (T-M2.4) で CPU 計測が finite (`synthesize` または `forward` 経由)
- [ ] `test_diff_4step` (`@pytest.mark.slow`): `DiffWaveNext2().synthesize` (= reverse_sample alias, T-M3.1/T-M3.3) で CPU 計測が finite
- [ ] **GAN (T=4) と Diff (4-step) 両方で計測できる** (`synthesize` 横断 dispatch が両モデルで動作)

### 5.3 Acceptance criteria (`docs/milestones.md` §M4.3 より転記)
- [ ] CPU 1-core 制限が `torch.set_num_threads(1)` で機能
- [ ] 100 utterances 平均で stable な RTF が出る (std < mean*0.1)

### 5.4 テスト戦略
- **mock model**: `nn.Module` に `synthesize` / `forward` + `hop_length` 属性を持たせた軽量スタブで CPU テスト (実 model は重いため slow marker)
- **`@pytest.mark.gpu`**: CUDA 限定 (CPU 環境では skip)、**`@pytest.mark.slow`**: 実 model 結合テスト
- **CI 時間目標**: ~20 秒以内 (mock 中心)

## 6. 懸念事項

### 6.1 技術的リスク
- **GPU warmup 不足**: 1st iteration は CUDA context 初期化 / カーネルキャッシュ / (torch.compile 時) compile で異常値 → **warmup `n` 回を必ず計測から除外**。`test_warmup_excluded` で検知
- **`torch.cuda.synchronize()` 漏れ**: GPU カーネルは非同期発行のため、sync なしだと `perf_counter()` が発行時刻のみ計測し過小評価 → **計測区間の前後で `synchronize()`**。`test_gpu_synchronize` で検知
- **CPU 1-core 制限が OS / BLAS レベルで効かない**: `torch.set_num_threads(1)` は PyTorch intra-op のみ。OpenMP/MKL は別経路のため **`OMP_NUM_THREADS=1` 環境変数も必要** (T-M0.1 §6 の再現性 env 設定と連携、`scripts` / CI で export)。`test_cpu_single_thread` で実測確認
- **batch_size=1 測定**: RTF は通常 streaming / 1 発話想定のため batch_size=1 で測る。batch 一括処理の throughput は別指標 (§8)
- **A100 がない環境**: ローカル GPU (RTX 系) / Colab で測ると論文 (A100) 値と乖離 → 戻り値に `device` 名を含め、**論文対比は同一 GPU 種別でのみ有効**と明記。最終 RTF 報告は T-M6.1 / T-M6.2 で A100 上で実施
- **`torch.compile` の compile 時間**: 適用時は compile が 1 回目に発生 → **warmup に含めて計測から除外** (warmup を compile 完了に十分な回数にする)。本チケット v1 は compile 非適用、§8 で測定検討

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` 関連: なし (RTF は純粋な計測タスク、論文事実依存なし)
- **Diff の `synthesize` alias 前提**: T-M3.1 §2.2 で `synthesize = staticmethod(reverse_sample)` (T-M3.3 完成後に有効化) を採用。未有効化期間は `getattr(model, "synthesize", model.forward)` が `forward` にフォールバックするが、Diff の `forward(mel, x_t, c, k)` は単一 sub-model 用で RTF 計測に不適 → **T-M3.3 完了後に Diff 計測を有効化**

### 6.3 他チケットとの整合性
- **T-M2.4 (GAN)**: `model.eval()` 後 `synthesize` (未定義なら `forward(mel)` で auto-infer audio_length) を計測。`torch.no_grad()` 内 (本関数 decorator)
- **T-M3.1 / T-M3.3 (Diff)**: `synthesize` (= `reverse_sample` alias) を計測。**T-M3.3 CRITICAL β_t 負値問題** (index.md M3 レビューログ) が未解決だと reverse_sample が NaN → RTF 計測前に T-M3.3 smoke 通過が前提
- **T-M0.1 (Python 環境)**: CPU 1-core には `OMP_NUM_THREADS=1` 環境変数も必要 (§6.1)、再現性 env 設定と連携

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:
- [ ] warmup の `n` 回が計測から除外されている (1st iteration 異常値対策)
- [ ] GPU 計測区間の前後で `torch.cuda.synchronize()` が呼ばれている
- [ ] CPU で `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1` 連携が明記されている
- [ ] `getattr(model, "synthesize", model.forward)` で GAN/Diff 横断 dispatch
- [ ] RTF = 推論時間 / 音声長秒 の計算式が正しい
- [ ] `stable` 判定 (`std < mean * 0.1`) が実装されている
- [ ] Acceptance criteria 全項目クリア、Unit テスト全 pass
- [ ] CLAUDE.md スタイル準拠 (型ヒント、docstring 英文 + 日本語、`from __future__ import annotations`、`@torch.no_grad()`、`*` で keyword-only)
- [ ] CPU でテストが pass (CUDA 不要、`@pytest.mark.gpu` で分離)
- [ ] 参考実装をコピーしていない

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (M4) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら
- **採用設計**: `measure_rtf(model, mels, *, device, ...)` 単一関数 + `synthesize` alias dispatch + warmup/sync。理由: GAN/Diff を同一 API で計測でき (`getattr` 横断)、T-M6.1 / T-M6.2 の最終 RTF 報告が薄い wrapper で済む。
- **代替案 (採用せず)**:
  1. **`torch.profiler` で layer 別 breakdown**: ボトルネック特定に有用だが overhead で RTF 自体が歪む。本チケットは wall-time のみ、profiler は M6 最適化時に別途
  2. **ONNX export → onnxruntime 計測**: deployment 想定の実測値だが export パイプラインが重い。M6 以降に検討
  3. **TorchScript / torch.compile での高速化測定**: compile 後 RTF は論文超えの可能性。warmup 設計を流用して M6 で測定
  4. **batch RTF (streaming ではなく batch 一括)**: throughput 指標。RTF (1 発話) とは別物として将来追加

### 8.2 思想 / 哲学の見直し
- **粒度**: 適切 (size=S)。RTF は単一計測関数に閉じる。GAN/Diff の synthesize 統一があるため横断計測が自然
- **インターフェース**: `device` を文字列引数にしたが、将来 multi-GPU では `torch.device` を受ける拡張余地

### 8.3 学んだこと (チケット完了後に追記)
- 想定外: TBD
- 教訓: TBD

#### 再評価トリガー
- **M6 本格訓練後の最終 RTF 報告** (T-M6.1 GPU 0.0066 / CPU 0.20 対比) 時に、A100 実測値と本関数の整合性を再評価。乖離があれば warmup 回数 / sync 位置を見直す

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- **インターフェース**: `measure_rtf(model, mels, *, device="cuda"|"cpu", sample_rate=24000, hop_length=None, n_warmup=5, n_measure=100) -> dict`。戻り値 `{"rtf_mean", "rtf_std", "device", "n_utterances", "n_threads", "stable"}`
- **T-M2.4 (GAN) から**: `GANWaveNext2(T=4).synthesize` (未定義なら `forward(mel)` で audio_length auto-infer) を計測対象に渡す
- **T-M3.1 (Diff) から**: `DiffWaveNext2.synthesize` (= `reverse_sample` alias, T-M3.3 完成後に有効化) を計測対象に渡す
- **T-M0.1 から**: CPU 1-core 計測には `OMP_NUM_THREADS=1` 環境変数も必要 (`torch.set_num_threads(1)` 単独では BLAS/OpenMP が multi-thread)
- **T-M6.1 / T-M6.2 (本格訓練) へ**: 訓練完了後、本関数で **最終 RTF を論文と対比** (GAN T=4: GPU 0.0066 / CPU 0.20)。同一 GPU 種別 (A100) でのみ論文対比が有効
- **注意事項 (後続が踏む罠)**: warmup 除外漏れ → 1st iteration 異常値で過大評価 / `synchronize()` 漏れ → GPU 過小評価 / `OMP_NUM_THREADS` 未設定 → CPU が multi-thread で過小評価

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` §M4.3 の Acceptance チェックボックス 2 項目
  - [ ] `docs/tickets/index.md` の T-M4.3 ステータスを `📝 pending` → `✅ completed`

### 9.3 Open question として残ったもの
- **Diff の正確な論文 RTF 値**: `docs/training.md` §5.3 に Diff (w/ sub-model, 4-step) の RTF が明記されているか実装時に確認、なければ T-M6.2 で実測のみ報告
- **A100 以外での測定値の解釈基準**: ローカル GPU 測定を論文対比にどう使うか、T-M6 で方針確定
