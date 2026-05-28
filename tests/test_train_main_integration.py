"""train_gan.main / train_diff.main の CPU 統合テスト (M3 review H1 ギャップ補完).

実 LibriTTS-R / GPU を使わず、load_config と build_loaders を monkeypatch で tiny config +
合成 loader に差し替えて `main()` の訓練ループ本体 (loop / step / logging / validation /
checkpoint scheduling / resume / SIGTERM 登録 / writer) を CPU で実行する。これにより
M3 review で T-M5.2 へ繰延していた「main()/build_loaders 全面未テスト」を実データ非依存で閉じる。
"""

from __future__ import annotations

import torch

import wavenext2.train.train_diff as td
import wavenext2.train.train_gan as tg

# --- tiny config (実 model を作らず高速化) -------------------------------------
_DIFF_CFG = {
    "seed": 0,
    "model": {
        "sub_model": {
            "n_mels": 32,
            "n_fft": 128,
            "hop": 64,
            "win_length": 128,
            "convnext": {
                "embed_dim": 64,
                "intermediate_dim": 128,
                "n_blocks": 2,
                "kernel_size": 3,
            },
            "noise_level_embedding": {"sinusoidal_dim": 16, "fc2": [64, 64]},
        },
        "noise_emb": {"c_rescale": 1.0},
    },
    "train": {
        "grad_clip_norm": 1.0,
        "max_steps": 6,
        "optimizer": {"lr": 2.0e-4, "betas": [0.9, 0.98], "weight_decay": 0.0},
    },
    "validation": {"interval_steps": 3, "num_utterances": 2},
    "checkpoint": {"interval_steps": 3},
    "logging": {"tensorboard_dir": "", "scalar_interval_steps": 1, "histogram_interval_steps": 2},
}


def _diff_batch(b: int = 1, t_mel: int = 8) -> dict:
    return {"mel": torch.randn(b, 32, t_mel), "audio": torch.randn(b, t_mel * 64)}


def test_train_diff_main_runs_loop(monkeypatch, tmp_path) -> None:
    cfg = {**_DIFF_CFG}
    cfg["checkpoint"] = {**cfg["checkpoint"], "dir": str(tmp_path / "ckpt")}
    cfg["logging"] = {**cfg["logging"], "tensorboard_dir": str(tmp_path / "tb")}
    monkeypatch.setattr(td, "load_config", lambda _p: cfg)
    loader = [_diff_batch() for _ in range(3)]
    monkeypatch.setattr(td, "build_loaders", lambda _cfg, _k: (loader, loader))

    # max_steps=6, validation/checkpoint interval=3 → loop / validation / checkpoint 分岐を踏む。
    td.main(["--config", "x.yaml", "--sub-model", "4"])

    # best (sub_4.pt) と step checkpoint が保存される (validation で best 更新)。
    assert (tmp_path / "ckpt" / "sub_4.pt").exists()


def test_train_diff_main_resume(monkeypatch, tmp_path) -> None:
    cfg = {**_DIFF_CFG, "train": {**_DIFF_CFG["train"], "max_steps": 3}}
    cfg["checkpoint"] = {**cfg["checkpoint"], "dir": str(tmp_path / "ckpt"), "interval_steps": 1}
    cfg["logging"] = {**cfg["logging"], "tensorboard_dir": str(tmp_path / "tb")}
    cfg["validation"] = {"interval_steps": 0, "num_utterances": 0}
    monkeypatch.setattr(td, "load_config", lambda _p: cfg)
    loader = [_diff_batch() for _ in range(2)]
    monkeypatch.setattr(td, "build_loaders", lambda _cfg, _k: (loader, loader))

    td.main(["--config", "x.yaml", "--sub-model", "1"])
    ckpt = tmp_path / "ckpt" / "step_2_sub_1.pt"
    assert ckpt.exists()
    # resume で step / RNG / optimizer が復元され例外なく再開・完走する。
    td.main(["--config", "x.yaml", "--sub-model", "1", "--resume", str(ckpt)])


def test_train_diff_main_debug_single_step(monkeypatch, tmp_path) -> None:
    cfg = {**_DIFF_CFG}
    cfg["checkpoint"] = {**cfg["checkpoint"], "dir": str(tmp_path / "c")}
    cfg["logging"] = {**cfg["logging"], "tensorboard_dir": str(tmp_path / "t")}
    monkeypatch.setattr(td, "load_config", lambda _p: cfg)
    loader = [_diff_batch()]
    monkeypatch.setattr(td, "build_loaders", lambda _cfg, _k: (loader, loader))
    td.main(["--config", "x.yaml", "--sub-model", "2", "--debug"])  # 1 step で break


# --- GAN main -----------------------------------------------------------------
_GAN_CFG = {
    "seed": 0,
    "model": {
        "T": 1,
        "sub_model": {"n_fft": 256, "hop": 64, "win_length": 256, "dim": 32,
                      "intermediate_dim": 64, "n_blocks": 1, "kernel_size": 3},
    },
    "discriminator": {"num_D": 1, "ndf": 4, "layers": 2, "downsampling_factor": 4},
    "loss": {
        "mrstft": {"n_ffts": [128], "win_lengths": [120], "hop_sizes": [32], "eps": 1.0e-5},
        "weights": {"d_gan": 1.0, "d_fm": 10.0, "mrstft_sc": 2.5, "mrstft_mag": 2.5},
    },
    "train": {
        "grad_clip_norm": 1.0,
        "max_steps": 4,
        "optimizer": {"lr_g": 1.0e-4, "lr_d": 2.0e-4, "betas": [0.8, 0.99],
                      "weight_decay": 1.0e-3},
        "scheduler": {"type": "InverseLR", "inv_gamma": 200000, "power": 0.5, "warmup": 0.999},
    },
    "validation": {"interval_steps": 2},
    "checkpoint": {"interval_steps": 2},
    "logging": {"tensorboard_dir": "", "scalar_interval_steps": 1},
}


def _gan_batch(b: int = 1, t_mel: int = 12) -> dict:
    return {"mel": torch.randn(b, 128, t_mel), "audio": torch.randn(b, t_mel * 64)}


def test_train_gan_main_runs_loop(monkeypatch, tmp_path) -> None:
    cfg = {**_GAN_CFG}
    cfg["checkpoint"] = {**cfg["checkpoint"], "dir": str(tmp_path / "ckpt")}
    cfg["logging"] = {**cfg["logging"], "tensorboard_dir": str(tmp_path / "tb")}
    monkeypatch.setattr(tg, "load_config", lambda _p: cfg)
    loader = [_gan_batch() for _ in range(2)]
    monkeypatch.setattr(tg, "build_loaders", lambda _cfg: (loader, loader))

    # max_steps=4, validation/checkpoint interval=2 → loop / validation / best / checkpoint を踏む。
    tg.main(["--config", "x.yaml"])
    assert (tmp_path / "ckpt" / "best.pt").exists()
