"""ConvNeXtBlock の unit テスト (T-M1.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wavenext2.models.convnext import ConvNeXtBlock

SNAP_DIR = Path(__file__).parent / "snapshots"

# 論文/Vocos 整合の理論パラメータ数
# gamma(512) + dwconv(7*512+512) + LN(2*512) + pwconv1(512*1536+1536) + pwconv2(1536*512+512)
GAN_PARAMS = 512 + (7 * 512 + 512) + 2 * 512 + (512 * 1536 + 1536) + (1536 * 512 + 512)  # 1,580,544
FC_T_PARAMS = 512 * 512 + 512  # 262,656


# --- shape ---------------------------------------------------------------------
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("t", [32, 80, 94, 256])
def test_shape_gan_mode(batch, t, device):
    block = ConvNeXtBlock(conditioning_dim=None).to(device)
    x = torch.randn(batch, 512, t, device=device)
    assert block(x).shape == (batch, 512, t)


@pytest.mark.parametrize("t", [32, 94, 256])
def test_shape_diff_mode(t, device):
    block = ConvNeXtBlock(conditioning_dim=512).to(device)
    x = torch.randn(2, 512, t, device=device)
    cond = torch.randn(2, 512, device=device)
    assert block(x, cond=cond).shape == (2, 512, t)


def test_shape_small_dim():
    block = ConvNeXtBlock(dim=64, intermediate_dim=192)
    assert block(torch.randn(2, 64, 40)).shape == (2, 64, 40)


# --- parameter count -----------------------------------------------------------
def test_param_count_gan():
    block = ConvNeXtBlock(conditioning_dim=None)
    assert sum(p.numel() for p in block.parameters()) == GAN_PARAMS


def test_param_count_diff_delta():
    gan = sum(p.numel() for p in ConvNeXtBlock(conditioning_dim=None).parameters())
    diff = sum(p.numel() for p in ConvNeXtBlock(conditioning_dim=512).parameters())
    assert diff - gan == FC_T_PARAMS


# --- gradient flow -------------------------------------------------------------
def test_gradient_flow_gan(device):
    block = ConvNeXtBlock(conditioning_dim=None).to(device)
    x = torch.randn(2, 512, 64, device=device)
    block(x).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().max() > 0, name


def test_gradient_flow_diff(device):
    block = ConvNeXtBlock(conditioning_dim=512).to(device)
    x = torch.randn(2, 512, 64, device=device)
    cond = torch.randn(2, 512, device=device, requires_grad=True)
    block(x, cond=cond).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, name
    assert cond.grad is not None and cond.grad.abs().max() > 0


# --- determinism / init --------------------------------------------------------
def test_deterministic():
    block = ConvNeXtBlock(conditioning_dim=None)
    x = torch.randn(2, 512, 80)
    block.eval()
    assert torch.equal(block(x), block(x))


def test_residual_dominance_at_init():
    # LayerScale γ=1e-6 のため初期化直後は block(x) ≈ x
    block = ConvNeXtBlock(conditioning_dim=None)
    x = torch.randn(2, 512, 80)
    assert (block(x) - x).abs().max() < 1e-2


def test_layer_scale_init_value():
    block = ConvNeXtBlock(conditioning_dim=None)
    assert block.gamma.abs().max() < 1e-5


def test_dwconv_is_depthwise():
    block = ConvNeXtBlock(dim=512)
    assert block.dwconv.groups == block.dim


# --- conditioning 挙動 ---------------------------------------------------------
def test_bias_broadcasts_over_time():
    block = ConvNeXtBlock(conditioning_dim=512)
    cond = torch.randn(2, 512)
    for t in (16, 100):
        assert block(torch.randn(2, 512, t), cond=cond).shape == (2, 512, t)


def test_cond_changes_output():
    block = ConvNeXtBlock(conditioning_dim=512)
    block.eval()
    x = torch.randn(2, 512, 64)
    out1 = block(x, cond=torch.zeros(2, 512))
    out2 = block(x, cond=torch.ones(2, 512))
    assert (out1 - out2).abs().max() > 1e-6


# --- error handling ------------------------------------------------------------
def test_error_diff_missing_cond():
    block = ConvNeXtBlock(conditioning_dim=512)
    with pytest.raises(ValueError, match="cond=None"):
        block(torch.randn(2, 512, 32))


def test_error_gan_extra_cond():
    block = ConvNeXtBlock(conditioning_dim=None)
    with pytest.raises(ValueError, match="without conditioning"):
        block(torch.randn(2, 512, 32), cond=torch.randn(2, 512))


def test_error_invalid_dim():
    with pytest.raises(ValueError, match="must be positive"):
        ConvNeXtBlock(dim=0)


def test_error_invalid_kernel():
    with pytest.raises(ValueError, match="odd"):
        ConvNeXtBlock(kernel_size=6)


def test_cond_is_keyword_only():
    # forward(x, *, cond=None) — cond を positional で渡すと TypeError
    block = ConvNeXtBlock(conditioning_dim=512)
    with pytest.raises(TypeError):
        block(torch.randn(2, 512, 32), torch.randn(2, 512))


# --- snapshot (drift 検知。CPU・要約統計を tolerant 比較) -----------------------
def _summary(t: torch.Tensor) -> dict:
    t = t.detach()
    return {
        "mean": round(float(t.mean()), 5),
        "std": round(float(t.std()), 5),
        "l2": round(float(t.norm()), 4),
    }


def _check_snapshot(name: str, out: torch.Tensor):
    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / f"{name}.json"
    summary = _summary(out)
    if not path.exists():
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        pytest.skip(f"snapshot baseline created: {path.name}")
    ref = json.loads(path.read_text(encoding="utf-8"))
    for k in ("mean", "std", "l2"):
        assert abs(summary[k] - ref[k]) <= max(1e-3, abs(ref[k]) * 1e-3), f"{name}.{k} drift"


def test_snapshot_gan():
    torch.manual_seed(0)
    block = ConvNeXtBlock(conditioning_dim=None)
    block.eval()
    _check_snapshot("convnext_gan", block(torch.randn(2, 512, 80)))


def test_snapshot_diff():
    torch.manual_seed(0)
    block = ConvNeXtBlock(conditioning_dim=512)
    block.eval()
    _check_snapshot("convnext_diff", block(torch.randn(2, 512, 80), cond=torch.randn(2, 512)))
