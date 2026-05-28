# 再現実装マイルストーン

調査結果 (`docs/open-questions.md`) を踏まえた、WaveNeXt 2 PyTorch 再現実装の工程表。

**実装担当**: 全工程を Claude Code が担当する。ユーザーは acceptance criteria のレビューと、人間にしかできない作業 (M0.3 LibriTTS-R ライセンス同意・ダウンロード認証、M6 GPU クラスタ確保、M7 主観評価の人手集め) のみ実施する。

各マイルストーンには:
- **Deliverable**: 完成物 (Claude Code が作成)
- **Acceptance criteria**: 完了判定条件 (Claude Code が自動テストで確認)
- **Dependencies**: 前提となる別のマイルストーン
- **作業量**: Claude Code セッションでの実装規模感 (small/medium/large)
- **ユーザー操作**: ユーザー側が実施すべき作業 (該当時のみ)

`Acceptance criteria` をすべて満たした時点で次マイルストーンに進む。

---

## 全体フロー

```
M0 (環境整備) ─► M1 (コア部品) ┬─► M2 (GAN)  ─┐
                                └─► M3 (Diff) ─┤
                                  M4 (評価) ────┤
                                                ▼
                                              M5 (統合スモーク)
                                                ▼
                                              M6 (本格訓練 ※要GPU)
                                                ▼
                                              M7 (主観評価 ※要人手)
```

- M0〜M5 は Claude Code が連続実装可能
- M6 は wall-clock time の制約 (A100 で GAN 410h + Diff 32h) があり Claude Code は訓練スクリプトを起動・監視するのみ
- M7 は人間が評価する必要があり Claude Code は集計と統計検定のみ実施

---

## M0: 環境整備とデータ準備 (作業量: small)

### M0.1 Python 環境
**チケット**: [T-M0.1](tickets/T-M0.1-python-env.md)

**Deliverable**: `pyproject.toml` (uv 管理) + `uv.lock`

- 管理ツール: [uv](https://docs.astral.sh/uv/) (`pip` / `poetry` / `conda` は使わない)
- Python: **3.13** を採用。論文に Python バージョン指定はない。依存ライブラリ調査の結果、最新の 3.14 では `pyworld` 本家が cp314 wheel を提供せず (Cython build が必要)、`librosa` 0.11.0 の PyPI classifier も 3.13 までしか明示していないため、全依存が公式 wheel で揃う 3.13 を採用。将来 pyworld 等が cp314 wheel を出した時点で 3.14 へのアップグレードを検討する。
- `pyproject.toml` の `requires-python = ">=3.13,<3.14"`
- 主な依存:

```
torch >= 2.1
torchaudio >= 2.1
numpy
scipy
librosa            # mel/STFT は torchaudio で代替可だが eval で使う
soundfile
pyyaml             # config 管理
tensorboard        # ログ
matplotlib         # 波形・mel 可視化
tqdm
                   # pymcd は Windows cp313 でビルド不能 (pysptk が MSVC 要求) のため除外。MCD は T-M4.1 で確定
pyworld            # log F0 RMSE 用 (cp313 wheel あり)
einops             # tensor reshape 用 (任意)
```

**Acceptance** (Claude Code が Bash で実行確認、2026-05-27 完了):
- [x] `uv sync` が成功 → `.venv/` と `uv.lock` が生成 (89 packages)
- [x] `uv run python --version` が `3.13.13` を表示 (3.13.8 は torch 2.11 import 不能の回帰のため下限 `>=3.13.13`)
- [x] `uv run python -c "import torch; print(torch.cuda.is_available())"` で True (torch 2.11.0+cu128 >= 2.10.0、RTX 4070 Ti SUPER)
- [x] ~~torchaudio `list_audio_backends` に sox_io~~ → **torchaudio 2.11 で backend dispatch / sox_effects / info が廃止**。音声 I/O は `soundfile` (1.2.2) に統一。sox `norm` は手動 peak 正規化で数値等価に再実装 (T-M0.1 §8.3 / §9.1 参照)
- [x] `uv run python -c "import pyworld, librosa; print(pyworld.__version__, librosa.__version__)"` でエラーなし (pyworld 0.3.5 / librosa 0.11.0)

**ユーザー操作**: 不要 (Claude Code が `uv venv --python 3.13.13` → `uv sync` まで実施)。CUDA は driver 596.21 で cu128 wheel が動作 (CUDA toolkit の別途インストールは不要、wheel 同梱ランタイムで充足)。

### M0.2 ディレクトリ scaffold
**チケット**: [T-M0.2](tickets/T-M0.2-scaffold.md)

**Deliverable**: `src/`, `configs/`, `tests/`, `scripts/`, `checkpoints/`, `logs/` の作成

```bash
src/wavenext2/
  data/{__init__.py, dataset.py, mel.py}
  models/{__init__.py, convnext.py, stft.py, generator.py, sub_model.py,
          noise_embedding.py, discriminator.py, gan_wavenext2.py, diff_wavenext2.py}
  losses/{__init__.py, adversarial.py, feature_matching.py, stft_loss.py}
  train/{__init__.py, train_gan.py, train_diff.py}
  inference/{__init__.py, infer_gan.py, infer_diff.py, post_filter.py}
  eval/{__init__.py, compute_metrics.py, run_utmos.py, run_nisqa.py, measure_rtf.py}
  utils/{__init__.py, config.py, logging.py, scheduler.py}
configs/{gan_wavenext2.yaml, diff_wavenext2.yaml}
tests/{test_convnext.py, test_stft_module.py, test_generator.py,
       test_sub_model.py, test_dataset.py, test_post_filter.py}
scripts/{prepare_libritts.py, extract_mel.py, fit_post_filter.py}
```

**Acceptance** (2026-05-27 完了):
- [x] 上記ツリーが作成され、各 `__init__.py` が `__all__` 予約コメント付きで存在 (`src/wavenext2/` パッケージ化)
- [x] `wavenext2.{models,data,losses,train,inference,eval,utils}` 全 import OK (editable install)
- [x] `pytest tests/ --collect-only` で 6 placeholder collect (exit 0)、実行で全 skip
- [x] `ruff check` / `ruff format --check` 通過、scaffold.py 再実行で冪等 (51 skip)
- [x] 追加 stub: Docker / `.devcontainer` / CI (`.github/workflows`) / `.pre-commit-config.yaml` / `.gitattributes` / `.env.example` / `.gitkeep` x7

**ユーザー操作**: 不要。

### M0.3 LibriTTS-R 取得
**チケット**: [T-M0.3](tickets/T-M0.3-libritts-r.md)

**Deliverable**: `scripts/prepare_libritts.py` (24 kHz 確認、必要なら resample、wav リストを生成)

- "train-clean-100" + "train-clean-360" + "test-clean" をローカル展開
- 音声ファイル一覧 (`data/filelists/{train,val,test,dev_postfilter}.tsv`) + `stats.json` + `audio_info.tsv` を生成 (TSV header 付き)
- validation は train から speaker-balanced で 100 utterances を hold-out

**Acceptance** (スクリプト+テスト完成、2026-05-27。実データ実行はユーザー DL 待ち):
- [x] `scripts/prepare_libritts.py` 実装 + `tests/test_prepare_libritts.py` 9 件 pass (synthetic e2e)
- [ ] `train.tsv` に約 145k 行 (train-clean-100 + 360 で約 460h) ← 実データ実行で確認
- [ ] `test.tsv` に約 4,824 行 (openslr `test-clean`) ← 実データ実行で確認
- [ ] 任意の wav を `soundfile` で読めて sample_rate=24000 を確認 (torchaudio 2.11 で load/info 廃止のため soundfile に統一)

**ユーザー操作** (必須):
1. LibriTTS-R のライセンス (CC BY 4.0) に同意し、https://www.openslr.org/141/ から `train_clean_100.tar.gz`, `train_clean_360.tar.gz`, `test_clean.tar.gz` をダウンロード
2. ダウンロード先のパスを Claude Code に伝える (例: `D:\datasets\LibriTTS_R\`)
3. Claude Code が展開・filelist 生成・整合性検証を自動実施

代替案: Claude Code が `wget` / `curl` で直接ダウンロードできる場合は、ユーザー操作なしで完結可能 (要約 50 GB のディスク空き)。

---

## M1: コア部品 (sub-model の構成要素) (作業量: large、6 サブタスク)

### M1.1 ConvNeXt block (`src/wavenext2/models/convnext.py`)
**チケット**: [T-M1.1](tickets/T-M1.1-convnext-block.md)

**Deliverable**: GAN/Diff 両対応の `ConvNeXtBlock` クラス

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim=512, intermediate_dim=1536, kernel_size=7,
                 layer_scale_init=1e-6, conditioning_dim=None):
        # conditioning_dim is not None → Diff 用 (additive bias 注入)
        ...
    def forward(self, x, cond=None):  # x: (B, dim, T), cond: (B, conditioning_dim) or None
        if cond is not None:
            x = x + self.fc_t(cond).unsqueeze(-1)
        # 標準 ConvNeXt forward
        ...
```

**Acceptance** (`tests/test_convnext.py`、2026-05-27 完了、42 件 pass):
- [x] GAN モード (`conditioning_dim=None`): `block(x)` で入出力 shape `(B, 512, T)` 一致
- [x] Diff モード (`conditioning_dim=512`): `block(x, cond=...)` で動作、cond なし呼び出しは ValueError (cond は keyword-only)
- [x] パラメータ数: GAN 版 **1,580,544**/block (= ×8 で 12.64M)、Diff 版 +**262,656** (`Linear(512,512)`)
- [x] gradient flow 確認: 全パラメータ + cond に grad

### M1.2 STFT module (`src/wavenext2/models/stft.py`)
**チケット**: [T-M1.2](tickets/T-M1.2-stft-module.md)

**Deliverable**: 波形 → STFT-spec (2F-2 ch) の変換モジュール

```python
class STFTModule(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        ...
    def forward(self, y, T_mel):  # y: (B, T_audio), T_mel: int
        # → (B, 2F-2, T_mel)
        ...
```

**Acceptance** (`tests/test_stft_module.py`、2026-05-27 完了、21 件 pass):
- [x] GAN 設定 (n_fft=2048, win=1200, hop=300): 入力 `(B, T_mel*300)` → 出力 `(B, 2048, T_mel)` (2F-2=n_fft)
- [x] Diff 設定 (n_fft=1024, win=1024, hop=256): 入力 `(B, T_mel*256)` → 出力 `(B, 1024, T_mel)`
- [x] 時間長 truncation 後の T 次元が指定 T_mel と完全一致
- [x] 実部・虚部の DC/Nyquist 扱いが正しい (虚部の最初/最後 bin が削除されている)
- [x] 単純な正弦波で round-trip テスト: 複素 magnitude が 440Hz bin (38±1) に集中

### M1.3 Mel-spectrogram 抽出 (`src/wavenext2/data/mel.py`)
**チケット**: [T-M1.3](tickets/T-M1.3-mel-spectrogram.md)

**Deliverable**: `MelSpectrogram` クラス

```python
class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=24000, n_fft, hop_length, win_length,
                 n_mels=128, f_min=20, f_max=12000, eps=1e-5):
        # power=1, mel_scale="slaney", norm="slaney", center=True
        ...
    def forward(self, audio):  # (B, T_audio) → (B, 128, T_mel)
        return torch.log(torch.clamp(self.mel(audio), min=eps))
```

**Acceptance** (`tests/test_mel.py`、2026-05-27 完了、19 件 pass):
- [x] GAN 設定で 1 秒 (24000 samples) 入力 → mel shape `(B, 128, 81)` (center=True: 1+24000//300=**81**、概算 80 から +1)
- [x] Diff 設定で 1 秒入力 → mel shape `(B, 128, 94)` (1+24000//256=94)
- [x] 出力範囲が `[log(1e-5), log(max)]` ≈ `[-11.51, ?]` に収まる (eps=1e-5 確定、SoT)

### M1.4 Generator (`src/wavenext2/models/generator.py`)
**チケット**: [T-M1.4](tickets/T-M1.4-generator.md)

**Deliverable**: WaveNeXt-based generator

```python
class WaveNextGenerator(nn.Module):
    def __init__(self, input_channels, dim=512, intermediate_dim=1536,
                 n_blocks=8, kernel_size=7, n_fft, hop_length,
                 conditioning_dim=None):
        # Conv1d(in=input_channels, out=dim, k=1, p=0)  ← embed kernel=1 確定 (T-M1.4 調査)
        # LayerNorm(dim, eps=1e-6)
        # ConvNeXt × n_blocks (cond optional)
        # LayerNorm(dim, eps=1e-6)
        # Linear(dim, n_fft+2, bias=True)
        # Linear(n_fft+2, hop_length, bias=False)
        # init: trunc_normal_(std=0.02), bias=zero
        ...
    def forward(self, x, cond=None):  # x: (B, input_channels, T_mel)
        ...
        return torch.clip(audio, -1.0, 1.0)  # (B, T_mel * hop_length)
```

**Acceptance** (`tests/test_generator.py`、2026-05-27 完了、18 件 pass):
- [x] GAN 設定 (input_channels=2176, hop=300, n_fft=2048): 入力 `(B, 2176, 80)` → 出力 `(B, 24000)`
- [x] Diff 設定 (input_channels=1152, hop=256, n_fft=1024): 入力 `(B, 1152, 94)` → 出力 `(B, 24064)`
- [x] 出力範囲が `[-1, 1]` に収まる (clip が機能、`*1e6` でも飽和)
- [x] パラメータ数: GAN **15.43M** (embed kernel=1、Table 1 14.99M +2.9% ✓)。Diff Generator は per-block fc_t 撤去後 **14.03M** (+NoiseEmbedding 0.33M で sub-model 14.354M、Table 1 14.42M −0.46% ✓、2026-05-27 確定 §C7)
- [ ] 重み初期化: `Conv1d.weight.std() ≈ 0.02`, `Linear.bias` がゼロ

### M1.5 Noise embedding (`src/wavenext2/models/noise_embedding.py`) [Diff のみ]
**チケット**: [T-M1.5](tickets/T-M1.5-noise-embedding.md)

**Deliverable**: sinusoidal + FC×2 SiLU の noise level embedding

```python
def sinusoidal_embedding(c: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """c: (B,) noise level → (B, dim)"""

class NoiseEmbedding(nn.Module):
    def __init__(self, sinusoidal_dim=128, mid_dim=512, out_dim=512):
        # FC1(128, 512) → SiLU → FC2(512, 512) → SiLU
        ...
    def forward(self, c):  # (B,) → (B, 512)
        ...
```

**Acceptance** (`tests/test_noise_embedding.py`、2026-05-27 完了、15 件 pass):
- [x] `c = torch.tensor([0.5])` で出力 shape `(1, 512)`、param 数 328,704
- [x] 同じ c に対する出力が deterministic
- [x] 異なる c に対する出力が異なる (cosine similarity < 0.99)
- [x] freq の log-spaced 確認: `freq[0] / freq[-1] ≈ 10000`。`input_rescale=1.0` 既定 (M3.5 で 1000 ablation 可)

### M1.6 Sub-model wrapper (`src/wavenext2/models/sub_model.py`)
**チケット**: [T-M1.6](tickets/T-M1.6-sub-model.md)

**Deliverable**: `SubModelGAN` と `SubModelDiff`

```python
class SubModelGAN(nn.Module):
    """STFT module + Generator (no conditioning)"""
    def __init__(self, mel_channels=128, n_fft, hop, win_length, ...):
        ...
    def forward(self, mel, y_prev):  # mel: (B, 128, T_mel), y_prev: (B, T_audio)
        stft_spec = self.stft_module(y_prev, T_mel=mel.shape[2])  # (B, 2F-2, T_mel)
        x = torch.cat([mel, stft_spec], dim=1)                     # (B, C_in, T_mel)
        return self.generator(x)                                    # (B, T_audio)

class SubModelDiff(nn.Module):
    """STFT module + Generator (with noise level conditioning)"""
    def forward(self, mel, x_t, c):  # c: (B,) noise level
        cond = self.noise_embedding(c)                              # (B, 512)
        stft_spec = self.stft_module(x_t, T_mel=mel.shape[2])
        x = torch.cat([mel, stft_spec], dim=1)
        return self.generator(x, cond=cond)
```

**Acceptance** (`tests/test_sub_model.py`、2026-05-27 完了、18 件 pass):
- [x] GAN: `mel(B, 128, 80) + y_prev(B, 24000) → n_t(B, 24000)` (clip[-1,1]、減算 y_{t-1}=y_t-n_t は T-M2.4)
- [x] Diff: `mel(B, 128, 94) + x_t(B, 24064) + c(B,) → ε_pred(B, 24064)` (clip なし=final_activation "none")
- [x] パラメータ数: GAN 15.43M (Table 1 14.99M +2.9%)。Diff は per-block fc_t 撤去後 14.354M (Table 1 14.42M −0.46% ✓、2026-05-27 確定 §C7)
- [x] forward + backward 動作、CONCAT_ORDER=("mel","stft_spec")、from_config 対応

---

## M2: GAN-WaveNeXt 2 (作業量: large、6 サブタスク)

### M2.1 Dataset (`src/wavenext2/data/dataset.py`)
**チケット**: [T-M2.1](tickets/T-M2.1-dataset.md)

**Deliverable**: LibriTTS-R loader + **peak 正規化** (sox `norm` と数値等価、soundfile I/O)

```python
class LibriTTSRDataset(Dataset):
    def __init__(self, filelist_path, root_dir, segment_length, hop_length, mel_cfg,
                 mode="train", seed=None, return_mel=True):
        # mode: "train" → peak U(-6, -1) dBFS, "val" → -3 dBFS
        ...
    def __getitem__(self, idx) -> Batch:  # {"mel": (128,T_mel), "audio": (seg,), "n_samples": int}
        ...
```

**Acceptance** (`tests/test_dataset.py`、2026-05-27 完了、15 件 pass):
- [x] train mode: peak 正規化 gain が U(-6, -1) dBFS、val mode: peak ≈ `10**(-3/20)` ≈ 0.708
- [x] segment_length より短い wav は反射 pad (極端に短いものは tile)、長いものは random crop (val は先頭固定)
- [x] mel と audio の時間長整合: **`mel.shape[1] == 1 + audio.shape[0]//hop`** (center=True、チケットの `(T_mel-1)*hop` 式は GAN 16384 で破綻するため修正)。generator 出力 (T_mel*hop) の crop は T-M2.4 責務
- [x] Batch TypedDict / seed_worker / n_samples attribute / from_config

### M2.2 Discriminator (`src/wavenext2/models/discriminator.py`)
**チケット**: [T-M2.2](tickets/T-M2.2-discriminator.md)

**Deliverable**: MSD × 3 (WaveFit-PT 完全準拠、MPD なし)

**Acceptance** (`tests/test_discriminator.py`、2026-05-27 完了、12 件 pass):
- [x] 入力 `(B, 1, T)` (or `(B, T)`) で 3 つの sub-discriminator から `SubDiscOutput(logits, features)` の list を返す
- [x] 各 sub-discriminator の中間特徴 6 個を返す (FM loss 用)、channel 進行 [16,64,256,1024,1024,1024]
- [x] AvgPool1d(k4,s2,p1) で隣接 sub-discriminator 間 downsample、weight_norm 全 Conv1d、Tanh/Sigmoid 不在 (hinge 互換)

### M2.3 Loss 関数 (`src/wavenext2/losses/`)
**チケット**: [T-M2.3](tickets/T-M2.3-losses.md)

**Deliverable**:
- `adversarial.py`: hinge GAN loss (`HingeGANLoss`)
- `feature_matching.py`: L1 FM loss
- `stft_loss.py`: Multi-resolution STFT (SC + Mag L1, eps=1e-5)

**Acceptance** (`tests/test_losses.py`、2026-05-27 完了、15 件 pass):
- [x] hinge GAN: `D_loss = relu(1 - D(real)).mean() + relu(1 + D(fake)).mean()` (sub-D 平均)、`G_loss = -D(fake).mean()`
- [x] FM: 3 sub-D × 6 layer の全中間特徴の L1 平均 (real 側 detach)
- [x] MR-STFT: 3 resolution `[512,1024,2048]` × `[360,900,1800]` × `[80,150,300]`、各 SC + log-mag L1、eps=1e-5
- [x] 重み: D-GAN=1.0, D-FM=10.0, MRSTFT-SC=2.5, MRSTFT-Mag=2.5 (`compute_total_loss`、sorted 加算順、unweighted dict 返し)

### M2.4 GAN モデル (`src/wavenext2/models/gan_wavenext2.py`)
**チケット**: [T-M2.4](tickets/T-M2.4-gan-model.md)

**Deliverable**: T 個の sub-model を直列に並べた fixed-point iteration generator

```python
class GANWaveNext2(nn.Module):
    def __init__(self, T=4, sub_model_cfg=...):
        self.sub_models = nn.ModuleList([SubModelGAN(**sub_model_cfg) for _ in range(T)])
    def forward(self, mel, x_gt_shape):
        y = torch.zeros(*x_gt_shape, device=mel.device)  # 初期 y_T = zeros
        for t in range(self.T, 0, -1):
            n_t = self.sub_models[t-1](mel, y)
            y = y - n_t
        return y
```

**Acceptance** (`tests/test_gan_wavenext2.py`、2026-05-28 完了、23 件 pass):
- [x] パラメータ数が Table 1 と一致 (T=4 で 61.71M = 4×15.43M、Table 1 59.94M +2.9%。T=1..5 で厳密 T 線形を検証)
- [x] forward → backward が動作 (T=4 で全 4 sub-model の全 param に grad)
- [x] T=1 で `y_0 = -sub_model(mel, zeros)` (パターン A: `y_{t-1}=y_t-n_t`、追加 clamp なし)。`return_intermediates` / `audio_length=None` auto-infer / `enable_grad_ckpt` (T-M2.4 レベル checkpoint) / `from_config` も実装

### M2.5 Training script (`src/wavenext2/train/train_gan.py`)
**チケット**: [T-M2.5](tickets/T-M2.5-train-gan.md)

**Deliverable**: AdamW + InverseLR + hinge GAN の交互更新ループ

```python
# Optimizer
opt_G = AdamW(G.parameters(), lr=1e-4, betas=[0.8, 0.99], weight_decay=1e-3)
opt_D = AdamW(D.parameters(), lr=2e-4, betas=[0.8, 0.99], weight_decay=1e-3)
# Scheduler (InverseLR: inv_gamma=200000, power=0.5, warmup=0.999)
# Grad clip: max_norm = 1.0
```

**Acceptance** (`tests/test_train_gan.py` + `tests/test_inverse_lr.py`、2026-05-28 完了、21 件 pass):
- [x] 1 step 実行で `loss_G`, `loss_D` が finite (`train_gan_step` 公開関数、synthetic batch で検証)
- [x] checkpoint 保存・復元が動作 (step/opt/sch/RNG 完全復元 + atomic best.pt rename)
- [x] TensorBoard に loss / 各 sub-loss / lr が記録される (`log_scalars`、sample audio は real-data の T-M5.1 で確認)
- 補足: InverseLR を step=0/inv_gamma/10×inv_gamma で数式 pin。CLI は click 非依存のため argparse 採用 (§8.3)。`build_loaders`/`run_validation`/`main` の real-data 経路は T-M5.1 (1 epoch) で実走確認。

### M2.6 Smoke training (overfitting test)
**チケット**: [T-M2.6](tickets/T-M2.6-gan-smoke.md)

**Deliverable**: 1 サンプルだけで 1000 step 訓練して loss が下がることを確認

**Acceptance** (`tests/test_train_gan_overfit.py` + `scripts/smoke_gan_synthetic.py`、2026-05-28):
- [x] **synthetic gate (CPU/GPU, データ不要)**: 合成 1 sample overfit で MR-STFT が減少 (best/init=0.84 < 0.92)、finite、勾配が流れることを確認 → 訓練 stack の sanity 担保
- [ ] 実 LibriTTS-R 1 utterance × 1000 step overfit (MR-STFT < 初期値 10%) は **T-M5.1 と併走で実データ取得後に実走** (`test_smoke_completes`、slow+gpu で skip 中)
- ⚠️ **発見 (要 M2 phase review / M5 注視)**: 出力 head の `clip(-1,1)` は飽和域で勾配 0。高 lr (≥5e-4) では generator が ±1 を超え **凍結** (loss 不変)。低 lr (1e-4) なら飽和前に学習継続。実訓練 (lr 1e-4 + InverseLR warmup) は安全側だが、発散時は `final_activation="tanh"` fallback を検討 (architecture.md generator 注記)。

---

## M3: Diff-WaveNeXt 2 (作業量: large、5 サブタスク)

### M3.1 Diff モデル (`src/wavenext2/models/diff_wavenext2.py`)
**チケット**: [T-M3.1](tickets/T-M3.1-diff-model.md)

**Deliverable**: 4 sub-model を独立に扱える wrapper + point-specialized partition

```python
class DiffWaveNext2(nn.Module):
    NOISE_SCHEDULE_ABAR = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])
    BAND_BOUNDS = [(0.9929, 1.0), (0.8246, 0.9929), (0.4817, 0.8246), (0.0, 0.4817)]

    def __init__(self, sub_model_cfg):
        self.sub_models = nn.ModuleList([SubModelDiff(**sub_model_cfg) for _ in range(4)])

    def sample_noise_level(self, k, batch_size):
        L, U = self.BAND_BOUNDS[k-1]
        return torch.empty(batch_size).uniform_(L, U)

    def reverse_sample(self, mel):  # 推論
        ...
```

**Acceptance**:
- [x] 4 sub-model それぞれが独立してパラメータを保持 (`nn.ModuleList`、重み非共有、`test_param_independence`)
- [x] `sample_noise_level(k=1)` の値が `[0.9929, 1.0]` に収まる (k=2,3,4 も同様、`test_sample_noise_level_range`)
- [x] パラメータ総数 = 4 × 14.354M = 57.42M (fc_t 撤去後の実装値、Table 1 57.68M に対し −0.46%。±1% 以内で `test_param_count_total`)

> **実装メモ (2026-05-28)**: `from_config` は `mode=` 引数を取らない実 API に合わせた (チケット擬似コードの `SubModelDiff.from_config(cfg, mode="diff")` は誤り、M2 review 申し送り済)。`NOISE_SCHEDULE_ABAR` は buffer `noise_schedule_abar` への device-aware property alias として公開 (T-M3.3 が `model.NOISE_SCHEDULE_ABAR.to(device)` で参照)。`from_config(only_sub_model=k)` で 1 sub-model のみ lazy instantiate (他は None placeholder)。45 tests pass。

### M3.2 Training script (`src/wavenext2/train/train_diff.py`)
**チケット**: [T-M3.2](tickets/T-M3.2-train-diff.md)

**Deliverable**: 各 sub-model を独立に MSE loss で訓練

```python
# 1 sub-model ずつ訓練 (sub-model index は CLI 引数 or config から指定)
# Optimizer: Adam(lr=2e-4, betas=[0.9, 0.98], wd=0)
# Loss: MSE on noise prediction
```

**Acceptance**:
- [ ] sub-model 1 を 100 step 訓練して loss が単調減少
- [ ] noise level が band 内で uniform sampling されていることを TensorBoard で確認
- [ ] 4 つの sub-model それぞれ独立に checkpoint 保存

### M3.3 Reverse sampler (`src/wavenext2/inference/infer_diff.py`)
**チケット**: [T-M3.3](tickets/T-M3.3-reverse-sampler.md)

**Deliverable**: x_0 予測経由 DDIM/DDPM 一般形 (β-free) 4-step sampling + 1-to-1 dispatch

```python
@torch.no_grad()
def reverse_sample(model, mel, *, seed=None, eta=1.0):
    # 詳細は docs/training.md §4.2 / T-M3.3。β は一切計算しない (denoising 順で ᾱ 増加列のため
    # β=1-ᾱ_t/ᾱ_{t-1} が負値になり発散)。x_0 予測経由で σ²≥0 が構造的に保証される。
    abar = model.NOISE_SCHEDULE_ABAR            # [1e-4, 2.8e-2, 5.6e-1, 9.1e-1]
    x = randn(...)
    for t in 1..K:                              # 1-to-1 dispatch: sub_models[t-1]
        eps = model.sub_models[t-1](mel, x, c_t=sqrt(1-abar_t))
        x0_hat = ((x - sqrt(1-abar_t)*eps) / sqrt(abar_t)).clamp(-1, 1)
        if t < K:
            sigma2 = eta**2 * (1-abar_next)/(1-abar_t) * (1 - abar_t/abar_next)
            x = sqrt(abar_next)*x0_hat + sqrt(clamp(1-abar_next-sigma2,0))*eps + (eta>0: sigma*z)
        else:
            x = x0_hat                          # 最終 step は ᾱ_next=1
    return x.clamp(-1, 1)
```

**Acceptance**:
- [x] mel → 4 step で `[-1, 1]` 範囲の波形が出力される (`test_output_range` / `test_with_real_diff_model`)
- [x] 同じ mel + 同じ seed で deterministic (`test_same_seed_deterministic` / `test_seed_with_external_generator`)

> **実装メモ (2026-05-28)**: `_compute_ddpm_coefficients` (β-free pure helper) + `reverse_sample(model, mel, *, seed=None, eta=1.0)` + `eval_mode(model)` context manager を実装。`seed` は `int | torch.Generator | None` の 3 形式 (global RNG 非汚染)。`eta=1.0` DDPM / `eta=0.0` DDIM を 1 引数切替。`infer_diff.py` 末尾の import 副作用で `DiffWaveNext2.synthesize = reverse_sample` を注入 (T-M4.3 が GAN/Diff 横断で `model.synthesize(mel)` 計測可能)。37 tests pass (mock で call_count==1 / dispatch order / σ²≥0 / β-free 検証)。全体 312 passed。

### M3.4 Post-filter (`src/wavenext2/inference/post_filter.py` + `scripts/fit_post_filter.py`)
**チケット**: [T-M3.4](tickets/T-M3.4-post-filter.md)

**Deliverable**: time-invariant spectral enhancement FIR の fit と apply

```python
# scripts/fit_post_filter.py
def fit_post_filter(model, dev_loader, n_fft=512, hop=256, fir_length=512):
    # 振幅差平均 → iRFFT → fftshift → fir.npy 保存
    ...

# src/wavenext2/inference/post_filter.py
def apply_post_filter(audio, fir):
    return np.convolve(audio, fir, mode="same")
```

**Acceptance**:
- [ ] dev set 100 utterances で fit してエラーなく fir.npy を生成
- [ ] FIR 長 = 512 (linear-phase 用 fftshift 済み)
- [ ] 周波数応答が ~2 kHz 以下で ≈ 0 dB、Nyquist 端で +5〜8 dB (Okamoto21 Fig 3a と整合)
- [ ] apply 前後で音声長が変わらない (`mode="same"`)

### M3.5 Smoke training
**チケット**: [T-M3.5](tickets/T-M3.5-diff-smoke.md)

**Deliverable**: sub-model 1 のみで 1000 step 訓練 + 過学習テスト

**Acceptance**:
- [ ] 1 utterance で 1000 step 後、MSE loss が初期値の 5% 以下
- [ ] 同じ utterance に対する reverse sample が GT に近い (MR-STFT loss で比較)

---

## M4: 評価インフラ (作業量: medium、3 サブタスク)

### M4.1 客観評価スクリプト (`src/wavenext2/eval/compute_metrics.py`)
**チケット**: [T-M4.1](tickets/T-M4.1-objective-metrics.md)

**Deliverable**: MCD, log F0 RMSE の自動計算

```python
def compute_mcd(y_true, y_pred, sr=24000):
    # pymcd 利用
def compute_log_f0_rmse(y_true, y_pred, sr=24000):
    # pyworld で F0 抽出 → log RMSE
```

**Acceptance**:
- [ ] 同一音声に対して MCD ≈ 0, log F0 RMSE ≈ 0
- [ ] LibriTTS-R test-clean-100 全 4824 utterance を 1 GPU で 30 分以内に処理

### M4.2 UTMOS / NISQA 連携
**チケット**: [T-M4.2](tickets/T-M4.2-utmos-nisqa.md)

**Deliverable**: `src/wavenext2/eval/run_utmos.py`, `src/wavenext2/eval/run_nisqa.py`

- UTMOS: https://github.com/sarulab-speech/UTMOS22 を git submodule
- NISQA: https://github.com/gabrielmittag/NISQA を pip install

**Acceptance**:
- [ ] GT 音声で UTMOS ≈ 4.0±0.2, NISQA ≈ 4.5±0.3 (LibriTTS-R は高品質なため)

### M4.3 RTF 測定 (`src/wavenext2/eval/measure_rtf.py`)
**チケット**: [T-M4.3](tickets/T-M4.3-rtf.md)

**Deliverable**: GPU (A100) と CPU (1 core) での RTF 計測

**Acceptance**:
- [ ] CPU 1-core 制限が `torch.set_num_threads(1)` で機能
- [ ] 100 utterances 平均で stable な RTF が出る (std < mean*0.1)

---

## M5: 統合スモークテスト (作業量: small、wall-clock は GPU 数時間)

### M5.1 1 epoch 訓練
**チケット**: [T-M5.1](tickets/T-M5.1-gan-1epoch.md)

**Deliverable**: GAN-WaveNeXt 2 を train-clean-100 で 1 epoch (約 33k step) 訓練

**Acceptance**:
- [ ] OOM なしで完走
- [ ] 1 epoch 後の generator が validation utterances に対して MR-STFT loss < 初期値の 30%
- [ ] 生成音声の聴感: 明らかな破綻なし (ノイズだらけ・全部 0 ではない)

### M5.2 Diff-WaveNeXt 2 1 sub-model 1 epoch
**チケット**: [T-M5.2](tickets/T-M5.2-diff-1epoch.md)

**Deliverable**: sub-model 1 を train-clean-100 で 1 epoch 訓練

**Acceptance**:
- [ ] OOM なしで完走
- [ ] noise level conditioning が機能 (異なる noise level で異なる出力)

---

## M6: 本格訓練 (Claude Code は起動・監視のみ、wall-clock: A100 で約 442 時間)

### M6.1 GAN-WaveNeXt 2 フル訓練
**チケット**: [T-M6.1](tickets/T-M6.1-gan-full-training.md)

**Deliverable**: A100 単体で約 410 時間訓練 → `checkpoints/gan/best.pt`

**Acceptance**:
- [ ] 2M step 完走 (or early stop)
- [ ] validation MR-STFT loss がプラトー
- [ ] 客観評価: UTMOS, NISQA, MCD, log F0 RMSE が論文 Table 1〜3 と概ね一致 (±10%)
- [ ] RTF が論文と一致 (**論文の RTF/param Table は 5 iter 値**: T=5 で GPU 0.0066 / CPU 0.20 / 74.93M)。T=4 は param 59.94M で品質 (MOS) が HiFi-GAN / WaveFit 5 iter と同等、RTF は T=5 より速い (論文に T=4 の RTF 直接値なし)

**Claude Code の役割**:
- 訓練を `bash` で `run_in_background=true` で起動
- 数時間〜数日おきに TensorBoard ログを `tensorboard --inspect` で確認
- divergence や OOM を検知したら自動再開 (resume from latest checkpoint)
- checkpoint 保存ごとに validation を実行して best を更新

**ユーザー操作** (必須):
1. A100 GPU が使える環境 (クラウド AWS p4/Lambda Labs/RunPod 等、または社内クラスタ) を準備
2. 環境への SSH 認証・課金設定をユーザー側で行う
3. Claude Code がそこから訓練を起動できるよう接続設定を済ませる

### M6.2 Diff-WaveNeXt 2 4 sub-model 訓練
**チケット**: [T-M6.2](tickets/T-M6.2-diff-full-training.md)

**Deliverable**: 各 sub-model を 1M step、合計約 32 時間 → `checkpoints/diff/sub_{1,2,3,4}.pt`

**Acceptance**:
- [ ] 全 sub-model が完走
- [ ] 4-step reverse sampling が安定 (NaN / divergence なし)
- [ ] post-filter 後の UTMOS, NISQA, MCD, log F0 RMSE が論文と概ね一致

**Claude Code の役割**:
- 4 つの sub-model を順次 (またはマルチ GPU あれば並列) 訓練起動
- 訓練完了後に post-filter fit (`scripts/fit_post_filter.py`) を自動実行
- 推論パイプラインを組み立てて test-clean-100 全 4824 utterance を評価

**ユーザー操作**: M6.1 と同様 (GPU リソース確保のみ)。

### M6.3 Ablation (任意)
**チケット**: [T-M6.3](tickets/T-M6.3-ablation.md)

**Deliverable**: 比較表 + プロット
- T=2,3,4,5 (GAN), with/without post-filter (Diff), with/without sub-modeling (Diff)
- 論文 Table 1〜3 の trend を再現

**Claude Code の役割**: ablation matrix を生成、各 config で訓練起動、論文との対比を Markdown でレポート生成。

---

## M7: 主観評価 (任意、人間が必須、Claude Code は集計のみ)

### M7.1 内部 MOS テスト
**チケット**: [T-M7.1](tickets/T-M7.1-mos-test.md)

- 20 utterances × 6 models = 120 sample
- 内部・少人数 (5〜10名) で評価
- 論文 MOS と相関を確認

**Acceptance**:
- [ ] GAN-WaveNeXt 2 (T=4) の MOS ≥ HiFi-GAN
- [ ] Diff-WaveNeXt 2 (w/ sub-model) の MOS ≥ FastDiff (w/ sub-model)

**Claude Code の役割**:
- 評価用 web app または Google Forms 用の音声ペア・質問票を自動生成
- 結果の CSV を受け取って統計検定 (paired t-test / Mann-Whitney) と CI 計算
- 論文との比較表を生成

**ユーザー操作**:
- 評価者の手配 (内部メンバー or 外部クラウドソーシング)
- 評価結果 CSV を Claude Code に渡す

---

## 依存関係グラフ

```
M0 (環境・データ)  ← ユーザー操作: LibriTTS-R DL
 ├─► M1 (コア部品)
 │    ├─► M2 (GAN)
 │    └─► M3 (Diff)
 ├─► M4 (評価インフラ)
 │
 └────────► M5 (統合スモーク、smoke train)
                              ▼
                            M6 (本格訓練)  ← ユーザー操作: GPU クラスタ確保
                              ▼
                            M7 (主観評価、任意)  ← ユーザー操作: 評価者手配
```

- M0 → 全マイルストーンの前提
- M1 → M2, M3 の前提
- M2, M3, M4 は M1 完了後に Claude Code が逐次実装 (一度に一つのファイル編集のため厳密な並列は不可だが、機能境界がきれいなので作業順序の自由度は高い)
- M5 → M6 への gate (smoke が通らないと本格訓練しない)
- ユーザー操作が必要なのは M0.3, M6, M7 のみ

---

## リスクと緩和策

| リスク | 影響 | 緩和 |
|---|---|---|
| GPU メモリ不足 (sub-model × T で OOM) | M2 完走できない | gradient checkpointing 導入、batch_size を 8 に半減 |
| LibriTTS-R ダウンロード失敗 | M0 で詰まる | huggingface ミラー、Common Voice で代替 |
| UTMOS / NISQA の Python バージョン非互換 | M4 で詰まる | venv で分離 |
| post-filter で発散 (FIR が divergent) | Diff 品質劣化 | clip FIR を `[-1, 1]` に制限、fit dev set を 500 に増やす |
| Diff partition の point-specialized 解釈が誤り | 品質劣化 | strict equal partition (Okamoto21) を ablation で比較 (`docs/architecture.md §5 注記`) |
| 訓練が divergent (NaN) | 全フェーズ | grad clip 1.0、AMP fp16 を最初は無効化、lr を半減して再開 |
| 本格訓練の予算超過 | M6 中断 | M5 で smoke が通った時点で論文同等を期待しない、checkpoint averaging で品質を補う |

---

## 次のアクション

ユーザーが「実装着手」と指示した時点で Claude Code が以下を順次実行:

1. **M0.1 + M0.2 を一気に実施**: `requirements.txt`, `pyproject.toml`, `src/` scaffold をすべて作成
2. **M0.3 開始時にユーザーに確認**: LibriTTS-R を Claude Code がダウンロードして良いか (約 50 GB)、または既にダウンロード済みのパスを教えてもらうか
3. **M1.1 → M1.6 を順次実装**: 各部品ごとにユニットテスト (`tests/test_*.py`) を書いて pytest が通ることを Bash で確認しながら進める
4. **M1 完了時にユーザーに報告**: パラメータ数や shape が論文と整合しているか提示してから M2/M3 に進む
5. **M2, M3, M4 を順次実装**: 各 deliverable が完成するごとに smoke test を実行
6. **M5 (統合スモーク) の前にユーザーに報告**: smoke で品質チェックする前に、Claude Code が確認した bug や懸念点をまとめて提示
7. **M5 が pass したら M6 への移行をユーザーに提案**: GPU クラスタ調達の判断はユーザーに委ねる
8. **M6 が完走したら客観評価レポートを自動生成**、M7 は希望時のみ実施

各マイルストーン完了時に Claude Code が:
- 変更ファイル一覧
- 通過した acceptance criteria
- 既知の懸念点
- 次マイルストーンの作業概要

を簡潔に報告する。
