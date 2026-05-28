"""T-M3.1: DiffWaveNext2 (4 sub-model, point-specialized partition) の unit テスト.

docs/tickets/T-M3.1-diff-model.md §5 を実装する。パラメータ許容値は fc_t 撤去後の実装値
(1 sub-model = 14.354M, ×4 = 57.42M) を基準とし、Table 1 (14.42M / 57.68M) への近接 (±1%)
も別途検証する (docs/open-questions.md §C7)。
"""

from __future__ import annotations

import math

import pytest
import torch

from wavenext2.models import DiffWaveNext2
from wavenext2.models.sub_model import SubModelDiff

# Diff 設定 (architecture.md §3 / §7): n_fft=1024, hop=256, win=1024。
SUB_CFG = {
    "mel_channels": 128,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "sinusoidal_dim": 128,
    "cond_dim": 512,
}

# 実装値 (計測): 1 sub-model = 14_354_434、×4 = 57_417_736。
PARAM_PER_SUB = 14_354_434
PARAM_TOTAL = 4 * PARAM_PER_SUB
TABLE1_PER_SUB = 14.42e6  # 論文 Table 1 (Diff wo/ sub-model)
TABLE1_TOTAL = 57.68e6    # 論文 Table 1 (Diff w/ sub-model)


@pytest.fixture(scope="module")
def model() -> DiffWaveNext2:
    """~57.42M params の重い init を module 単位で再利用."""
    torch.manual_seed(0)
    return DiffWaveNext2(sub_model_cfg=SUB_CFG)


def _make_io(b: int = 2, t_mel: int = 32, hop: int = 256):
    mel = torch.randn(b, 128, t_mel)
    x_t = torch.randn(b, t_mel * hop)
    return mel, x_t


# --------------------------------------------------------------------------- #
# パラメータ数
# --------------------------------------------------------------------------- #
def test_param_count_total(model: DiffWaveNext2) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert n == PARAM_TOTAL
    # Table 1 (57.68M) への近接 ±1% (fc_t 撤去で −0.46%)。
    assert abs(n - TABLE1_TOTAL) / TABLE1_TOTAL < 0.01


def test_param_count_per_sub_model(model: DiffWaveNext2) -> None:
    for k in range(4):
        n = sum(p.numel() for p in model.sub_models[k].parameters())
        assert n == PARAM_PER_SUB
        assert abs(n - TABLE1_PER_SUB) / TABLE1_PER_SUB < 0.01


def test_param_count_is_4x_single(model: DiffWaveNext2) -> None:
    total = sum(p.numel() for p in model.parameters())
    per = sum(p.numel() for p in model.sub_models[0].parameters())
    # 重み非共有なら厳密に 4 倍 (shared なら total == per)。
    assert total == 4 * per


def test_param_independence(model: DiffWaveNext2) -> None:
    assert id(model.sub_models[0]) != id(model.sub_models[1])
    # sub_models[0] の weight を変えても sub_models[1] に伝播しない (独立 instance)。
    p0 = next(model.sub_models[0].parameters())
    p1 = next(model.sub_models[1].parameters())
    assert p0.data_ptr() != p1.data_ptr()


# --------------------------------------------------------------------------- #
# 4 sub-model 独立性 (milestones §M3.1 Acceptance #1)
# --------------------------------------------------------------------------- #
def test_four_sub_models_independent(model: DiffWaveNext2) -> None:
    assert len(model.sub_models) == 4
    for sub in model.sub_models:
        assert isinstance(sub, SubModelDiff)


def test_independent_grad(model: DiffWaveNext2) -> None:
    model.zero_grad(set_to_none=True)
    mel, x_t = _make_io()
    c = model.sample_noise_level(2, mel.shape[0])
    out = model(mel, x_t, c, k=2)
    out.sum().backward()
    # sub_models[1] (k=2) のみ grad が流れる。
    assert any(p.grad is not None for p in model.sub_models[1].parameters())
    for j in (0, 2, 3):
        assert all(p.grad is None for p in model.sub_models[j].parameters())
    model.zero_grad(set_to_none=True)


def test_sub_model_state_dict_separate(model: DiffWaveNext2) -> None:
    keys = model.state_dict().keys()
    for i in range(4):
        assert any(k.startswith(f"sub_models.{i}.") for k in keys)


# --------------------------------------------------------------------------- #
# sample_noise_level の band 範囲 (milestones §M3.1 Acceptance #2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("k", "lo", "hi"),
    [(1, 0.9929, 1.0), (2, 0.8246, 0.9929), (3, 0.4817, 0.8246), (4, 0.0, 0.4817)],
)
def test_sample_noise_level_range(model: DiffWaveNext2, k: int, lo: float, hi: float) -> None:
    s = model.sample_noise_level(k, 2000)
    assert s.shape == (2000,)
    assert float(s.min()) >= lo
    assert float(s.max()) < hi


def test_sample_noise_level_covers_band(model: DiffWaveNext2) -> None:
    # uniform なら band の両端付近にもサンプルが出る (粗い被覆チェック、scipy 非依存)。
    lo, hi = model.get_band(3)
    s = model.sample_noise_level(3, 5000)
    width = hi - lo
    assert float(s.min()) < lo + 0.05 * width
    assert float(s.max()) > hi - 0.05 * width
    # 平均が中点付近 (uniform の期待値)。
    assert abs(float(s.mean()) - (lo + hi) / 2) < 0.05 * width


@pytest.mark.parametrize("k", [0, 5, -1])
def test_sample_noise_level_invalid_k(model: DiffWaveNext2, k: int) -> None:
    with pytest.raises(ValueError):
        model.sample_noise_level(k, 4)


def test_sample_noise_level_deterministic(model: DiffWaveNext2) -> None:
    torch.manual_seed(42)
    a = model.sample_noise_level(1, 4)
    torch.manual_seed(42)
    b = model.sample_noise_level(1, 4)
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# BAND_BOUNDS 数値検証
# --------------------------------------------------------------------------- #
def test_band_bounds_no_gap_no_overlap() -> None:
    bb = DiffWaveNext2.BAND_BOUNDS
    for k in range(3):
        assert bb[k][0] == bb[k + 1][1]  # k の lower == k+1 の upper (隙間/重複なし)


def test_band_bounds_full_coverage() -> None:
    bb = DiffWaveNext2.BAND_BOUNDS
    assert bb[0][1] == 1.0
    assert bb[-1][0] == 0.0


def test_band_bounds_centers_match_schedule() -> None:
    # band の中心 (近似) が √(1-ᾱ) schedule 点に対応 (architecture.md §5)。
    abar = DiffWaveNext2.NOISE_SCHEDULE_ABAR_DEFAULT
    c = [math.sqrt(1.0 - a) for a in abar]  # [0.99995, 0.9858, 0.6633, 0.3]
    expected_centers = [0.99995, 0.9858, 0.6633, 0.3]
    for ci, ei in zip(c, expected_centers, strict=True):
        assert ci == pytest.approx(ei, abs=1e-3)


def test_band_bounds_midpoints() -> None:
    # 境界が隣接 schedule 点の中点に一致 (±1e-3、丸め誤差吸収)。
    abar = DiffWaveNext2.NOISE_SCHEDULE_ABAR_DEFAULT
    c = [math.sqrt(1.0 - a) for a in abar]
    bb = DiffWaveNext2.BAND_BOUNDS
    assert bb[0][0] == pytest.approx((c[0] + c[1]) / 2, abs=1e-3)
    assert bb[1][0] == pytest.approx((c[1] + c[2]) / 2, abs=1e-3)
    assert bb[2][0] == pytest.approx((c[2] + c[3]) / 2, abs=1e-3)


# --------------------------------------------------------------------------- #
# noise schedule buffer (state_dict 整合性)
# --------------------------------------------------------------------------- #
def test_noise_schedule_buffer_register(model: DiffWaveNext2) -> None:
    assert "noise_schedule_abar" in model.state_dict()


def test_noise_schedule_values(model: DiffWaveNext2) -> None:
    expected = torch.tensor([1.0e-4, 2.8e-2, 5.6e-1, 9.1e-1])
    assert torch.allclose(model.noise_schedule_abar, expected, atol=1e-7)
    # NOISE_SCHEDULE_ABAR alias (T-M3.3 が参照) も同値。
    assert torch.allclose(model.NOISE_SCHEDULE_ABAR, expected, atol=1e-7)


def test_noise_schedule_dtype(model: DiffWaveNext2) -> None:
    assert model.noise_schedule_abar.dtype == torch.float32


def test_noise_schedule_load_state_dict(model: DiffWaveNext2) -> None:
    fresh = DiffWaveNext2(sub_model_cfg=SUB_CFG)
    fresh.load_state_dict(model.state_dict())
    assert torch.allclose(fresh.noise_schedule_abar, model.noise_schedule_abar)


# --------------------------------------------------------------------------- #
# forward(mel, x_t, c, k)
# --------------------------------------------------------------------------- #
def test_forward_shape(model: DiffWaveNext2) -> None:
    mel, x_t = _make_io(b=2, t_mel=32)
    c = model.sample_noise_level(2, 2)
    out = model(mel, x_t, c, k=2)
    assert out.shape == (2, 32 * 256)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_forward_all_k(model: DiffWaveNext2, k: int) -> None:
    mel, x_t = _make_io(b=1, t_mel=16)
    c = model.sample_noise_level(k, 1)
    out = model(mel, x_t, c, k=k)
    assert out.shape == (1, 16 * 256)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("k", [0, 5])
def test_forward_invalid_k(model: DiffWaveNext2, k: int) -> None:
    mel, x_t = _make_io(b=1, t_mel=16)
    c = torch.full((1,), 0.5)
    with pytest.raises(ValueError):
        model(mel, x_t, c, k=k)


def test_forward_deterministic(model: DiffWaveNext2) -> None:
    mel, x_t = _make_io(b=1, t_mel=16)
    c = torch.full((1,), 0.5)
    model.eval()
    with torch.no_grad():
        a = model(mel, x_t, c, k=1)
        b = model(mel, x_t, c, k=1)
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# reverse_sample スタブ
# --------------------------------------------------------------------------- #
def test_reverse_sample_not_implemented(model: DiffWaveNext2) -> None:
    mel, _ = _make_io()
    with pytest.raises(NotImplementedError):
        model.reverse_sample(mel)


# --------------------------------------------------------------------------- #
# _validate_band_bounds (内部 invariant)
# --------------------------------------------------------------------------- #
def test_validate_band_bounds_pass() -> None:
    DiffWaveNext2._validate_band_bounds()  # default で例外なし


def test_validate_band_bounds_detects_gap(monkeypatch) -> None:
    bad = [(0.99, 1.0), (0.80, 0.95), (0.48, 0.80), (0.0, 0.48)]  # 0.99 != 0.95 で隙間
    monkeypatch.setattr(DiffWaveNext2, "BAND_BOUNDS", bad)
    with pytest.raises(ValueError):
        DiffWaveNext2._validate_band_bounds()


# --------------------------------------------------------------------------- #
# from_config factory
# --------------------------------------------------------------------------- #
def test_from_config() -> None:
    m = DiffWaveNext2.from_config({"sub_model_cfg": SUB_CFG})
    assert isinstance(m, DiffWaveNext2)
    assert len(m.sub_models) == 4


def test_from_config_sub_model_alias() -> None:
    # "sub_model" キーも受理 (config drift 耐性)。
    m = DiffWaveNext2.from_config({"sub_model": SUB_CFG})
    assert isinstance(m, DiffWaveNext2)


def test_from_config_empty() -> None:
    m = DiffWaveNext2.from_config({})
    assert isinstance(m, DiffWaveNext2)


def test_from_config_lazy_instantiation() -> None:
    m = DiffWaveNext2.from_config({"sub_model_cfg": SUB_CFG}, only_sub_model=1)
    assert isinstance(m.sub_models[0], SubModelDiff)
    for i in (1, 2, 3):
        assert m.sub_models[i] is None
    # None placeholder は state_dict に出力されない (sub_models.0.* のみ)。
    keys = m.state_dict().keys()
    assert any(k.startswith("sub_models.0.") for k in keys)
    for i in (1, 2, 3):
        assert not any(k.startswith(f"sub_models.{i}.") for k in keys)


@pytest.mark.parametrize("k", [0, 5])
def test_from_config_lazy_invalid_k(k: int) -> None:
    with pytest.raises(ValueError):
        DiffWaveNext2.from_config({"sub_model_cfg": SUB_CFG}, only_sub_model=k)


def test_from_config_lazy_param_count() -> None:
    m = DiffWaveNext2.from_config({"sub_model_cfg": SUB_CFG}, only_sub_model=3)
    n = sum(p.numel() for p in m.parameters())
    # 1 sub-model 分のみ (OOM 回避効果)。buffer (4 float) は無視できる誤差。
    assert n == PARAM_PER_SUB


def test_from_config_lazy_forward_only_target() -> None:
    m = DiffWaveNext2.from_config({"sub_model_cfg": SUB_CFG}, only_sub_model=3)
    mel, x_t = _make_io(b=1, t_mel=16)
    c = m.sample_noise_level(3, 1)
    out = m(mel, x_t, c, k=3)
    assert out.shape == (1, 16 * 256)
    # None の sub-model を呼ぶと RuntimeError。
    with pytest.raises(RuntimeError):
        m(mel, x_t, c, k=1)


# --------------------------------------------------------------------------- #
# import / 結合
# --------------------------------------------------------------------------- #
def test_importable() -> None:
    from wavenext2.models import DiffWaveNext2 as Imported

    assert Imported is DiffWaveNext2


def test_get_band(model: DiffWaveNext2) -> None:
    assert model.get_band(1) == (0.9929, 1.0)
    with pytest.raises(ValueError):
        model.get_band(0)
