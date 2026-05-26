"""scaffold.py: wavenext2 パッケージの空ツリー / スタブ一括生成 (冪等).

T-M0.2 (ディレクトリ scaffold) の実体。`Path.exists()` チェックで既存ファイルを
上書きしないため、何度実行しても安全 (後続チケットで実装済みのモジュールを壊さない)。

使い方:
    uv run python scripts/scaffold.py
"""

from __future__ import annotations

from pathlib import Path

# リポジトリ ROOT (= このファイルの 1 つ上)
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# テンプレート
# ---------------------------------------------------------------------------
MODULE_STUB = '''"""{desc}

TODO: {ticket} で実装。詳細は docs/architecture.md / docs/milestones.md を参照。
"""

# このファイルは scaffold 段階のスタブです。実装は後続チケットで行います。
'''

INIT_STUB = '''"""wavenext2{dotted} public API.

TODO: 後続チケットで `__all__` に re-export を追加。
"""

# __all__ = []  # 後続チケットで明示
'''

SCRIPT_STUB = '''"""{desc}

TODO: {ticket} で実装。
使い方 (実装後): uv run python scripts/{name}
"""

# このファイルは scaffold 段階のスタブです。実装は後続チケットで行います。

if __name__ == "__main__":
    raise SystemExit("{name} は未実装です ({ticket} で実装)。")
'''

TEST_STUB = '''"""{module} のテスト (placeholder).

TODO: {ticket} で実テストに置き換え。
"""

import pytest


def test_placeholder():
    pytest.skip("{ticket} で実装")
'''

CONFTEST_STUB = '''"""pytest 共有 fixture の置き場 (現状は空).

TODO: M1.x で synthetic audio / mel / config fixture を追加。
横断 fixture 戦略は T-M1.x フェーズレビューの申し送り参照。
"""
'''

SEED_STUB = '''"""乱数 seed の一括設定ユーティリティ.

TODO: M1.x で実装。torch / numpy / random / CUDA (manual_seed_all) を一括設定し、
`deterministic=True` のとき cudnn.deterministic / benchmark を切り替える。

想定 API:
    def set_seed(seed: int, deterministic: bool = False) -> None: ...
"""

# このファイルは scaffold 段階のスタブです。実装は後続チケットで行います。
'''

# ---------------------------------------------------------------------------
# 生成対象の定義
# ---------------------------------------------------------------------------
# サブパッケージ (dotted は __init__ docstring 用、"" は root)
SUBPACKAGES = ["", ".data", ".models", ".losses", ".train", ".inference", ".eval", ".utils"]

# モジュールスタブ: 相対パス -> (説明, 担当チケット)
MODULES: dict[str, tuple[str, str]] = {
    "data/dataset.py": ("LibriTTS-R Dataset + peak 正規化 + 反射 pad / random crop.", "T-M2.1"),
    "data/mel.py": (
        "log-mel-spectrogram 抽出。n_fft/hop_length/win_length/n_mels は "
        "config 経由のみで受け取る関数 API として実装 (GAN hop=300 / Diff hop=256 の取り違え防止)。",
        "T-M1.3",
    ),
    "models/convnext.py": ("ConvNeXt block (Diff conditioning は additive bias で注入).", "T-M1.1"),
    "models/stft.py": ("STFT module (前ステップ波形を STFT 表現化し mel と結合).", "T-M1.2"),
    "models/generator.py": (
        "WaveNeXt generator (ConvNeXt backbone + Linear output head).",
        "T-M1.4",
    ),
    "models/sub_model.py": ("Sub-model wrapper (STFT + mel 結合 + generator).", "T-M1.6"),
    "models/noise_embedding.py": (
        "Diffusion noise-level embedding (c のスカラ→ベクトル埋め込み).",
        "T-M1.5",
    ),
    "models/discriminator.py": ("MSD x3 discriminator (MPD なし、WaveFit-PT 準拠).", "T-M2.2"),
    "models/gan_wavenext2.py": (
        "GAN-WaveNeXt 2 (T 個 sub-model 直列 + fixed-point iteration).",
        "T-M2.4",
    ),
    "models/diff_wavenext2.py": (
        "Diff-WaveNeXt 2 (4 sub-model, point-specialized partition).",
        "T-M3.1",
    ),
    "losses/adversarial.py": ("MSD hinge GAN loss (generator / discriminator).", "T-M2.3"),
    "losses/feature_matching.py": ("feature matching loss.", "T-M2.3"),
    "losses/stft_loss.py": ("multi-resolution STFT loss (M3/M4 でも再利用).", "T-M2.3"),
    "train/train_gan.py": (
        "GAN-WaveNeXt 2 訓練ループ (fixed-point, train_gan_step 公開関数).",
        "T-M2.5",
    ),
    "train/train_diff.py": (
        "Diff-WaveNeXt 2 訓練ループ (sub-model 独立訓練, MSE on noise).",
        "T-M3.2",
    ),
    "inference/infer_gan.py": ("GAN 推論 (zeros 初期化 + T 回 fixed-point iteration).", "T-M2.4"),
    "inference/infer_diff.py": ("Diff 推論 (beta-free x0 予測 reverse sampling).", "T-M3.3"),
    "inference/post_filter.py": ("post-filter (time-invariant FIR の fit / apply).", "T-M3.4"),
    "eval/compute_metrics.py": ("MCD / log-F0 RMSE などの客観指標.", "T-M4.1"),
    "eval/run_utmos.py": ("UTMOS (subprocess 隔離).", "T-M4.2"),
    "eval/run_nisqa.py": ("NISQA (subprocess 隔離).", "T-M4.2"),
    "eval/measure_rtf.py": ("RTF 測定 (GPU CUDA events + CPU 1-core).", "T-M4.3"),
    "utils/config.py": ("YAML config の読み込み / 検証.", "T-M0.2 基盤 (実装は利用側チケット)"),
    "utils/logging.py": ("TensorBoard / stdout ロギング.", "T-M0.2 基盤 (実装は利用側チケット)"),
    "utils/scheduler.py": ("LR scheduler ヘルパ.", "T-M2.5"),
}

# scripts スタブ: ファイル名 -> (説明, 担当チケット)
SCRIPTS: dict[str, tuple[str, str]] = {
    "prepare_libritts.py": ("LibriTTS-R 取得後の filelist / stats 生成.", "T-M0.3"),
    "extract_mel.py": ("mel-spectrogram の事前計算 (cache).", "T-M2.1"),
    "fit_post_filter.py": ("post-filter FIR を dev_postfilter から fit.", "T-M3.4"),
}

# テストスタブ: ファイル名 -> (対象モジュール, 担当チケット)
TESTS: dict[str, tuple[str, str]] = {
    "test_convnext.py": ("ConvNeXt block", "M1.1"),
    "test_stft_module.py": ("STFT module", "M1.2"),
    "test_generator.py": ("Generator", "M1.4"),
    "test_sub_model.py": ("Sub-model", "M1.6"),
    "test_dataset.py": ("Dataset", "M2.1"),
    "test_post_filter.py": ("Post-filter", "M3.4"),
}

# .gitkeep を置く空ディレクトリ
GITKEEP_DIRS = [
    "checkpoints",
    "logs",
    "data",
    "data/filelists",
    "data/raw",
    "data/cache",
    "data/processed",
]


def write_if_absent(path: Path, content: str) -> bool:
    """既存ファイルは上書きしない。生成したら True、skip したら False。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(ROOT).as_posix()
    if path.exists():
        print(f"  skip (exists): {rel}")
        return False
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"  create:        {rel}")
    return True


def main() -> None:
    created = 0
    print("[scaffold] generating wavenext2 package tree (idempotent)")

    pkg_root = ROOT / "src" / "wavenext2"

    # __init__.py
    for sub in SUBPACKAGES:
        rel = sub.lstrip(".").replace(".", "/")  # ".data" -> "data", "" -> ""
        init_path = pkg_root / rel / "__init__.py" if rel else pkg_root / "__init__.py"
        created += write_if_absent(init_path, INIT_STUB.format(dotted=sub))

    # utils/seed.py (特別な docstring)
    created += write_if_absent(pkg_root / "utils" / "seed.py", SEED_STUB)

    # モジュールスタブ
    for rel, (desc, ticket) in MODULES.items():
        created += write_if_absent(pkg_root / rel, MODULE_STUB.format(desc=desc, ticket=ticket))

    # scripts スタブ (scaffold.py 自身は除く)
    for name, (desc, ticket) in SCRIPTS.items():
        created += write_if_absent(
            ROOT / "scripts" / name, SCRIPT_STUB.format(desc=desc, ticket=ticket, name=name)
        )

    # tests スタブ
    created += write_if_absent(ROOT / "tests" / "conftest.py", CONFTEST_STUB)
    for name, (module, ticket) in TESTS.items():
        created += write_if_absent(
            ROOT / "tests" / name, TEST_STUB.format(module=module, ticket=ticket)
        )

    # .gitkeep
    for d in GITKEEP_DIRS:
        created += write_if_absent(ROOT / d / ".gitkeep", "")

    print(f"[scaffold] done: {created} files created")


if __name__ == "__main__":
    main()
