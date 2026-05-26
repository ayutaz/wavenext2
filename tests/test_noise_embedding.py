"""NoiseEmbedding / sinusoidal_embedding の unit テスト (T-M1.5)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wavenext2.models.noise_embedding import NoiseEmbedding, sinusoidal_embedding

SNAP_DIR = Path(__file__).parent / "snapshots"


# --- sinusoidal_embedding -----------------------------------------------------
def test_sinusoidal_shape():
    assert sinusoidal_embedding(torch.tensor([0.5]), dim=128).shape == (1, 128)
    assert sinusoidal_embedding(torch.rand(8), dim=128).shape == (8, 128)


def test_sinusoidal_even_dim_required():
    with pytest.raises(ValueError, match="even"):
        sinusoidal_embedding(torch.tensor([0.5]), dim=127)


def test_sinusoidal_freq_log_spaced():
    # freq[0]=exp(0)=1, freq[-1]=exp(-log(10000))=1e-4 → 比は厳密に 10000
    half = 64
    log_scale = math.log(10000.0) / (half - 1)
    freq = torch.exp(-log_scale * torch.arange(half))
    assert abs(float(freq[0] / freq[-1]) - 10000.0) < 1.0


def test_sinusoidal_concat_order_sin_then_cos():
    # c=0 → sin(0)=0 (前半 64), cos(0)=1 (後半 64)
    emb = sinusoidal_embedding(torch.tensor([0.0]), dim=128)
    assert torch.allclose(emb[0, :64], torch.zeros(64), atol=1e-6)
    assert torch.allclose(emb[0, 64:], torch.ones(64), atol=1e-6)


def test_sinusoidal_sum_of_squares_is_half():
    # sin²+cos² = 1 を half 個分 → ノルム² = half = 64 (任意の c で不変)
    for c in (0.1, 0.5, 0.9):
        emb = sinusoidal_embedding(torch.tensor([c]), dim=128)
        assert abs(float((emb**2).sum()) - 64.0) < 1e-3


def test_sinusoidal_snapshot():
    emb = sinusoidal_embedding(torch.tensor([0.5, 0.9]), dim=128)
    summary = {"mean": round(float(emb.mean()), 6), "std": round(float(emb.std()), 6)}
    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / "noise_emb_sinusoidal.json"
    if not path.exists():
        path.write_text(json.dumps(summary, indent=2) + "\n")
        pytest.skip("baseline created")
    ref = json.loads(path.read_text())
    for k in ("mean", "std"):
        assert abs(summary[k] - ref[k]) <= 1e-4


# --- NoiseEmbedding -----------------------------------------------------------
def test_forward_shape(device):
    m = NoiseEmbedding().to(device)
    assert m(torch.tensor([0.5], device=device)).shape == (1, 512)
    assert m(torch.rand(8, device=device)).shape == (8, 512)


def test_param_count():
    # FC1 (128*512+512) + FC2 (512*512+512) = 328,704
    assert (
        sum(p.numel() for p in NoiseEmbedding().parameters()) == 128 * 512 + 512 + 512 * 512 + 512
    )


def test_deterministic():
    m = NoiseEmbedding().eval()
    c = torch.rand(4)
    assert torch.allclose(m(c), m(c))


def test_distinguishes_inputs():
    m = NoiseEmbedding().eval()
    with torch.no_grad():
        e1 = m(torch.tensor([0.1]))
        e2 = m(torch.tensor([0.9]))
    assert float(F.cosine_similarity(e1, e2)) < 0.99


def test_gradient_flow():
    m = NoiseEmbedding()
    m(torch.rand(4)).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, name


def test_silu_activation():
    assert isinstance(NoiseEmbedding().act, torch.nn.SiLU)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_follows_input(dtype):
    out = NoiseEmbedding().to(dtype)(torch.rand(4, dtype=dtype))
    assert out.dtype == dtype


def test_input_rescale_changes_output():
    # input_rescale=1000 (DDPM step 相当 ablation) で出力が変わることを確認
    c = torch.tensor([0.3, 0.7])
    torch.manual_seed(0)
    m1 = NoiseEmbedding(input_rescale=1.0).eval()
    torch.manual_seed(0)
    m1000 = NoiseEmbedding(input_rescale=1000.0).eval()
    # 同一重み (同 seed) でも入力 rescale が違えば sinusoidal が変わる
    assert not torch.allclose(m1(c), m1000(c))
