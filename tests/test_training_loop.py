"""M3 review: utils/training_loop.py 共通プリミティブの直接テスト.

M2/M3 review で「共通プリミティブが間接的にしかテストされていない」と指摘された穴を補完
(atomic_save の原子性 / RNG state 往復 / iter_forever の epoch またぎ / SIGTERM handler)。
これらは M6 cluster 訓練 (preemption 多) の resume・耐障害の土台。
"""

from __future__ import annotations

import signal

import numpy as np
import pytest
import torch

from wavenext2.utils.training_loop import (
    atomic_save,
    capture_rng_state,
    iter_forever,
    register_sigterm_handler,
    restore_rng_state,
)


# --------------------------------------------------------------------------- #
# atomic_save
# --------------------------------------------------------------------------- #
def test_atomic_save_roundtrip(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    obj = {"step": 7, "w": torch.arange(3)}
    atomic_save(obj, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["w"], torch.arange(3))


def test_atomic_save_no_tmp_left(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    atomic_save({"x": 1}, path)
    # `.tmp` 中間ファイルが残っていない (os.replace で rename 済)。
    assert not (tmp_path / "ckpt.pt.tmp").exists()
    assert path.exists()


def test_atomic_save_creates_parent(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "ckpt.pt"
    atomic_save({"x": 1}, path)
    assert path.exists()


def test_atomic_save_overwrites(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    atomic_save({"v": 1}, path)
    atomic_save({"v": 2}, path)
    assert torch.load(path, weights_only=False)["v"] == 2


# --------------------------------------------------------------------------- #
# capture / restore_rng_state (resume 再現性の核心)
# --------------------------------------------------------------------------- #
def test_rng_state_roundtrip_torch() -> None:
    torch.manual_seed(123)
    state = capture_rng_state()
    before = torch.randn(5)
    # RNG を進めてから restore → 同じ乱数列が再現される。
    _ = torch.randn(10)
    restore_rng_state(state)
    after = torch.randn(5)
    assert torch.equal(before, after)


def test_rng_state_roundtrip_numpy() -> None:
    np.random.seed(7)
    state = capture_rng_state()
    before = np.random.randn(5)
    _ = np.random.randn(10)
    restore_rng_state(state)
    after = np.random.randn(5)
    np.testing.assert_array_equal(before, after)


def test_capture_rng_state_keys() -> None:
    state = capture_rng_state()
    assert {"torch", "cuda", "numpy", "python"} <= set(state)


def test_restore_rng_state_none_noop() -> None:
    # None / 空 dict は no-op (例外を出さない)。
    restore_rng_state(None)
    restore_rng_state({})


def test_restore_rng_state_partial() -> None:
    # 欠損キー (torch のみ) でも他を触らず復元できる。
    torch.manual_seed(5)
    partial = {"torch": torch.get_rng_state()}
    before = torch.randn(3)
    _ = torch.randn(3)
    restore_rng_state(partial)
    assert torch.equal(torch.randn(3), before)


# --------------------------------------------------------------------------- #
# iter_forever (epoch またぎ)
# --------------------------------------------------------------------------- #
def test_iter_forever_crosses_epoch() -> None:
    loader = [1, 2, 3]  # len 3
    it = iter_forever(loader)
    got = [next(it) for _ in range(7)]  # len を超えて回る
    assert got == [1, 2, 3, 1, 2, 3, 1]


# --------------------------------------------------------------------------- #
# register_sigterm_handler
# --------------------------------------------------------------------------- #
def test_register_sigterm_handler_calls_save_and_exits() -> None:
    calls: list[int] = []
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_int = signal.getsignal(signal.SIGINT)
    try:
        register_sigterm_handler(lambda: calls.append(1))
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        # signal を実送信せず handler を直接呼ぶ (Windows でも安全)。
        with pytest.raises(SystemExit):
            handler(signal.SIGTERM, None)
        assert calls == [1]  # save_fn が呼ばれた
        # SIGINT にも同じ handler が登録される。
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)
