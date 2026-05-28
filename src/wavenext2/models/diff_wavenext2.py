"""diff_wavenext2.py — 4 個の SubModelDiff を point-specialized partition で保持する Diff-WaveNeXt 2.

論文 Fig 1b / Section 3.3 / Fig 3 の 4 sub-model 独立訓練 + 4-step reverse sampling を実装する。
- 4 個の `SubModelDiff` を独立に保持 (重み共有なし; Table 1 のパラメータ数 57.68M = 14.42M × 4
  に対し、fc_t 撤去後の本実装は 57.42M = 14.354M × 4 で −0.46%。docs/open-questions.md §C7)
- noise schedule ᾱ = [1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1] (denoising 順 t=1..4、t=1 が純ノイズ側)
- point-specialized band partition: 隣接 schedule 点 √(1-ᾱ) の中点で band を区切る

GAN (`GANWaveNext2`) との共通親クラス (`BaseVocoder` / `IterativeVocoder`) は **抽出しない**
(T-M3.1 §6.1 / §8 で確定)。GAN は逐次 fixed-point iteration (状態あり)、Diff は band 独立呼び出し
(状態なし) で forward signature が本質的に異なるため、2 並列実装とする。共通化は sub-model level の
`from_config()` factory のみ。

詳細は docs/architecture.md §5, docs/training.md §3 / §4.2 を参照。
"""

from __future__ import annotations

import torch
from torch import nn

from wavenext2.models.sub_model import SubModelDiff

__all__ = ["DiffWaveNext2"]


class DiffWaveNext2(nn.Module):
    """Diff-WaveNeXt 2: 4 個の SubModelDiff を point-specialized partition で並列保持.

    各 sub-model は 4-step 固定 noise schedule の各点に 1-to-1 で対応する band を担当する。
    訓練時は `forward(mel, x_t, c, k)` で単一 sub-model k を band 独立に呼ぶ (4 個一括 forward
    の API は持たない)。推論 (4-step reverse sampling) は T-M3.3 の `reverse_sample` が責務で、
    本クラスの `reverse_sample` は v1 ではスタブ (NotImplementedError)。

    Args:
        sub_model_cfg: `SubModelDiff.from_config` に渡す dict (None で既定値)。
            例: {"mel_channels": 128, "n_fft": 1024, "hop_length": 256, "win_length": 1024,
                 "sinusoidal_dim": 128, "cond_dim": 512}。
    """

    # 4-step 固定 noise schedule (denoising 方向 t=1..4)。
    # t=1: 純ノイズ側 (ᾱ=1e-4)、t=4: ほぼクリーン (ᾱ=9.1e-1)。
    # docs/architecture.md §5 / docs/training.md §3.3 / docs/open-questions.md §解決済み。
    NOISE_SCHEDULE_ABAR_DEFAULT: tuple[float, ...] = (1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1)

    # Point-specialized band partition (√(1-ᾱ) 軸、隣接点の中点で区切る)。
    # k=1 (最高ノイズ) → k=4 (最低ノイズ) の順。docs/architecture.md §5 Band 境界表で確定。
    # 中点計算: c = √(1-ᾱ) = [0.99995, 0.9858, 0.6633, 0.3]
    #   (c[0]+c[1])/2 = 0.99287 → 0.9929、(c[1]+c[2])/2 = 0.82455 → 0.8246、
    #   (c[2]+c[3])/2 = 0.48165 → 0.4817
    BAND_BOUNDS: list[tuple[float, float]] = [
        (0.9929, 1.0),     # k=1: 中心 √(1-ᾱ_1) = 0.99995
        (0.8246, 0.9929),  # k=2: 中心 √(1-ᾱ_2) = 0.9858
        (0.4817, 0.8246),  # k=3: 中心 √(1-ᾱ_3) = 0.6633
        (0.0, 0.4817),     # k=4: 中心 √(1-ᾱ_4) = 0.3
    ]

    # sub-model 数 (4 で固定、論文 §3.3)
    K: int = 4

    def __init__(self, sub_model_cfg: dict | None = None) -> None:
        super().__init__()
        self._validate_band_bounds()
        cfg = dict(sub_model_cfg or {})
        # 重み非共有: K=4 個の独立 instance (Table 1 の 57.68M = 14.42M × 4 から確定)。
        self.sub_models = nn.ModuleList([SubModelDiff.from_config(cfg) for _ in range(self.K)])

        # noise schedule を state_dict 互換で固定保持 (推論時の整合性確保、§6.1)。
        self.register_buffer(
            "noise_schedule_abar",
            torch.tensor(self.NOISE_SCHEDULE_ABAR_DEFAULT, dtype=torch.float32),
            persistent=True,
        )

        # audio_length 検算用に hop をキャッシュ (全 sub-model で同一)。
        self.hop_length: int = self.sub_models[0].hop_length

    @property
    def NOISE_SCHEDULE_ABAR(self) -> torch.Tensor:  # noqa: N802 (論文表記に合わせた定数風 alias)
        """`noise_schedule_abar` buffer への alias (T-M3.3 reverse_sampler が参照)。

        buffer なので `.to(device)` が自動同期される。class attribute ではなく buffer 経由で
        返すことで、device-aware かつ load_state_dict 整合性を担保する。
        """
        return self.noise_schedule_abar

    @classmethod
    def _validate_band_bounds(cls) -> None:
        """BAND_BOUNDS が point-specialized partition (隙間/重複なし、全体被覆) と整合するか検証.

        - 隣接 band の境界が一致 (k の lower == k+1 の upper)
        - 全体カバレッジ: k=1 の upper == 1.0、k=K の lower == 0.0
        - 各 band で lower < upper
        """
        bb = cls.BAND_BOUNDS
        if len(bb) != cls.K:
            raise ValueError(f"BAND_BOUNDS must have {cls.K} entries, got {len(bb)}")
        if bb[0][1] != 1.0:
            raise ValueError(f"BAND_BOUNDS[0] upper must be 1.0, got {bb[0][1]}")
        if bb[-1][0] != 0.0:
            raise ValueError(f"BAND_BOUNDS[-1] lower must be 0.0, got {bb[-1][0]}")
        for k in range(cls.K):
            lo, hi = bb[k]
            if lo >= hi:
                raise ValueError(f"BAND_BOUNDS[{k}] must satisfy lower < upper, got {bb[k]}")
        for k in range(cls.K - 1):
            # k (0-indexed) の lower == k+1 の upper で隙間/重複なし。
            if bb[k][0] != bb[k + 1][1]:
                raise ValueError(
                    f"BAND_BOUNDS gap/overlap at k={k + 1}/{k + 2}: "
                    f"bb[{k}][0]={bb[k][0]} != bb[{k + 1}][1]={bb[k + 1][1]}"
                )

    def sample_noise_level(self, k: int, batch_size: int) -> torch.Tensor:
        """sub-model k (1-indexed) の band [L_k, U_k) 内で uniform sampling.

        Args:
            k:          sub-model index (1..K=4)。
            batch_size: 出力サンプル数。

        Returns:
            (batch_size,) tensor of √(1-ᾱ) ∈ [L_k, U_k)。device/dtype は呼び出し側で `.to()`。
        """
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        lo, hi = self.BAND_BOUNDS[k - 1]
        return torch.empty(batch_size).uniform_(lo, hi)

    def get_band(self, k: int) -> tuple[float, float]:
        """sub-model k (1-indexed) の band [L_k, U_k) を返す (T-M3.2 から参照可)."""
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        return self.BAND_BOUNDS[k - 1]

    def forward(
        self,
        mel: torch.Tensor,
        x_t: torch.Tensor,
        c: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """単一 sub-model k の forward (訓練時に使用、T-M3.2 から呼ばれる).

        Args:
            mel: (B, 128, T_mel) log-mel。
            x_t: (B, T_audio = T_mel * hop_length) noised waveform。
            c:   (B,) noise level √(1-ᾱ) ∈ band [L_k, U_k)。
            k:   呼び出す sub-model index (1..K=4)。kwarg 推奨 (positional 渡しも可)。

        Returns:
            (B, T_audio) 予測ノイズ ε_θ (MSE ターゲット、Fig 1b)。

        Note:
            複数 sub-model を一括で呼ぶ API は提供しない (band 独立訓練の設計上 1 sub-model ずつ呼ぶ)。
            推論 (reverse sampling) は T-M3.3 の `reverse_sample` で別実装。
        """
        if not 1 <= k <= self.K:
            raise ValueError(f"k must be in [1, {self.K}], got {k}")
        sub = self.sub_models[k - 1]
        if sub is None:
            raise RuntimeError(
                f"sub_models[{k - 1}] is None (lazy instantiation). "
                f"from_config(only_sub_model=...) で対象 sub-model のみ instantiate した場合は "
                f"その k のみ forward 可能。"
            )
        return sub(mel, x_t, c)

    def reverse_sample(self, mel: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """4-step DDPM 逆プロセスで mel から波形を合成 (推論時に使用).

        T-M3.3 の reverse_sampler に詳細実装を委譲。本チケット v1 ではスタブ。
        詳細は docs/training.md §4.2 を参照。
        """
        raise NotImplementedError(
            "reverse_sample is implemented in T-M3.3. "
            "Use wavenext2.inference.infer_diff.reverse_sample(model, mel)."
        )

    @classmethod
    def from_config(cls, cfg: dict, only_sub_model: int | None = None) -> DiffWaveNext2:
        """YAML config から DiffWaveNext2 を生成 (T-M1.6 §8.2 factory パターン一貫化).

        受理キー: `sub_model_cfg` (または `sub_model`)。その他のキーは無視 (config drift 耐性)。

        Args:
            cfg: dict with key "sub_model_cfg" (or "sub_model")。
            only_sub_model: 指定時は対象 sub-model (1-indexed) のみを instantiate し、他は
                None placeholder で `sub_models` に格納する (state_dict 互換維持)。T-M3.2 で
                1 sub-model のみ訓練するときの OOM 回避 (4×14.354M weight + optimizer state を
                削減) 用。default=None で全 4 sub-model を instantiate。
        """
        cfg = dict(cfg)
        sub_model_cfg = cfg.get("sub_model_cfg") or cfg.get("sub_model") or {}
        if only_sub_model is None:
            return cls(sub_model_cfg=sub_model_cfg)
        if not 1 <= only_sub_model <= cls.K:
            raise ValueError(f"only_sub_model must be in [1, {cls.K}], got {only_sub_model}")
        # lazy instantiation: 指定 sub-model 1 個のみ Module、他は None。
        instance = cls(sub_model_cfg=sub_model_cfg)
        for i in range(cls.K):
            if (i + 1) != only_sub_model:
                instance.sub_models[i] = None  # type: ignore[index]
        return instance
