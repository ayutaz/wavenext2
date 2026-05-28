"""gan_wavenext2.py — T 個の SubModelGAN を直列化した GAN-WaveNeXt 2 generator (T-M2.4).

論文 Fig 1a / Section 3.2 の **fixed-point iteration** (WaveFit ベースの residual denoising) を実装する。

- T 個の `SubModelGAN` を独立に保持 (重み共有なし; Table 1 のパラメータ数が T 線形比例することから確定)
- 初期 `y_T = torch.zeros(...)` (論文「initial input noise isn't required」, docs/open-questions §D)
- 逆順 t = T, T-1, ..., 1 で `y_{t-1} = y_t - n_t` (`n_t = sub_models[t-1](mel, y_t)`)

**戻り値仕様 (確定, パターン A)**: `SubModelGAN.forward` は残差ノイズ `n_t` を返す (generator 内で
clip[-1,1] 済)。減算 `y_{t-1} = y_t - n_t` は本クラスが行う (docs/architecture.md §2/§4,
docs/training.md §2.1)。**追加の clamp は行わない** (training.md §2.2「clip は sub-model 内部で
適用済み」)。よって `y_0 ∈ [-1,1]` は T=1 または訓練収束時に成立し、未訓練の T>1 では残差累積で
一時的に範囲外を取りうる (収束に伴い target ∈ [-1,1] へ)。却下案 (B) は §8 (ticket) 参照。

詳細: docs/architecture.md §2/§4、docs/training.md §2、docs/tickets/T-M2.4-gan-model.md。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from wavenext2.models.sub_model import SubModelGAN

__all__ = ["GANWaveNext2"]


class GANWaveNext2(nn.Module):
    """GAN-WaveNeXt 2 generator: T 個の SubModelGAN を直列化した fixed-point iterator.

    Args:
        T: iteration 回数 / sub-model 数。論文 Table 1 は T=2,3,4,5 を比較、推奨 T=4。
        sub_model_cfg: `SubModelGAN.from_config` に渡す dict (None で既定値)。
            例: {"n_fft": 2048, "hop_length": 300, "win_length": 1200}。
        enable_grad_ckpt: True で各 sub-model 呼び出しを `torch.utils.checkpoint`
            (use_reentrant=False) で包み、訓練 backward 時の activation memory を削減する
            (T 直列の OOM 対策、24GB GPU 向け)。eval / no_grad 時は素通し。
    """

    def __init__(
        self,
        T: int = 4,
        sub_model_cfg: dict | None = None,
        enable_grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        self.T = T
        self.enable_grad_ckpt = enable_grad_ckpt
        cfg = dict(sub_model_cfg or {})
        # 重み非共有: T 個の独立 instance (Table 1 の T 線形パラメータ数から確定)
        self.sub_models = nn.ModuleList([SubModelGAN.from_config(cfg) for _ in range(T)])
        # audio_length 検算用に hop をキャッシュ (全 sub-model で同一)
        self.hop_length: int = self.sub_models[0].hop_length

    def forward(
        self,
        mel: torch.Tensor,
        audio_length: int | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Fixed-point iteration による波形合成.

        Args:
            mel: (B, 128, T_mel) log-mel-spectrogram。
            audio_length: 出力波形長。None のとき `T_mel * hop_length` で auto-infer。
                明示時は同値でなければ ValueError。
            return_intermediates: True で `[y_T, y_{T-1}, ..., y_0]` (長さ T+1) を返す
                (T-M2.5 の中間 y_t loss 用)。False で最終 y_0 のみ。

        Returns:
            (B, audio_length) の y_0 (return_intermediates=False)、
            または長さ T+1 の list[Tensor] (return_intermediates=True)。
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be (B, 128, T_mel), got {tuple(mel.shape)}")
        b, _, t_mel = mel.shape
        expected = t_mel * self.hop_length
        if audio_length is None:
            audio_length = expected
        elif audio_length != expected:
            raise ValueError(
                f"audio_length ({audio_length}) must equal T_mel * hop_length "
                f"({t_mel} * {self.hop_length} = {expected})"
            )

        y = torch.zeros(b, audio_length, device=mel.device, dtype=mel.dtype)
        intermediates: list[torch.Tensor] = [y] if return_intermediates else []
        # 逆順 t = T → 1。sub_models は 0-indexed のため t-1 で参照。
        use_ckpt = self.enable_grad_ckpt and self.training and torch.is_grad_enabled()
        for t in range(self.T, 0, -1):
            sub = self.sub_models[t - 1]
            n_t = checkpoint(sub, mel, y, use_reentrant=False) if use_ckpt else sub(mel, y)
            y = self._residual_update(y, n_t)
            if return_intermediates:
                intermediates.append(y)
        return intermediates if return_intermediates else y

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """推論用 alias (T-M4.3 RTF が GAN/Diff 横断で `model.synthesize(mel)` を呼ぶ)。

        `forward(mel)` は audio_length=None で `T_mel*hop` を auto-infer するため、第 2 引数
        必須問題を回避する thin wrapper。Diff 側は infer_diff が同名 alias を注入する。
        """
        return self.forward(mel)

    @staticmethod
    def _residual_update(y: torch.Tensor, n_t: torch.Tensor) -> torch.Tensor:
        """残差更新の単一情報源 (SoT)。パターン (A): `y_{t-1} = y_t - n_t`。

        却下案 (B) (`SubModelGAN` が denoised 波形 y_{t-1} を直接返し `return n_t` とする) は
        docs/architecture.md §2「出力は波形そのものではなくノイズ成分」と矛盾するため不採用
        (T-M2.4 §8)。
        """
        return y - n_t

    @classmethod
    def from_config(cls, cfg: dict) -> GANWaveNext2:
        """dict から生成 (T-M2.5 / T-M4.3 の生成経路統一用)。

        受理キー: `T`, `sub_model_cfg` (または `sub_model`), `enable_grad_ckpt`。
        その他のキーは無視 (config drift 耐性)。
        """
        cfg = dict(cfg)
        return cls(
            T=cfg.get("T", 4),
            sub_model_cfg=cfg.get("sub_model_cfg") or cfg.get("sub_model"),
            enable_grad_ckpt=cfg.get("enable_grad_ckpt", False),
        )
