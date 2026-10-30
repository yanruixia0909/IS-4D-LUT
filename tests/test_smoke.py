"""Quick sanity checks for the IS-4D LUT implementation.

Run with:  python tests/test_smoke.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from is4dlut.config import LossConfig, ModelConfig  # noqa: E402
from is4dlut.datasets import SyntheticPairs  # noqa: E402
from is4dlut.losses import IS4DLUTLoss, soft_histogram, wasserstein_1d  # noqa: E402
from is4dlut.models import IS4DLUT, QuadrilinearLUT  # noqa: E402


def test_lut_identity():
    """An identity 4D LUT must reproduce the input colours exactly."""
    torch.manual_seed(0)
    lut = QuadrilinearLUT(n_grid=9, n_basis=3)
    weights = torch.softmax(torch.randn(1, 3), dim=1)
    img = torch.rand(1, 3, 24, 24)
    intensity = torch.rand(1, 1, 24, 24)
    out = lut(img, intensity, weights)
    err = float((out - img).abs().max())
    assert out.shape == img.shape, out.shape
    assert err < 1e-5, f"identity interpolation error too large: {err}"
    print(f"[ok] 4D LUT identity interpolation, max error = {err:.2e}")


def test_histogram_tools():
    x = torch.rand(1, 3, 16, 16)
    h = soft_histogram(x, bins=16)
    assert h.shape == (1, 3, 16)
    assert torch.allclose(h.sum(-1), torch.ones(1, 3), atol=1e-4)
    assert float(wasserstein_1d(h, h)) < 1e-6
    print("[ok] differentiable histogram + Wasserstein distance")


def test_model_forward_and_loss():
    torch.manual_seed(0)
    cfg = ModelConfig(n_grid=9, n_basis=3, base_channels=8, num_levels=2)
    model = IS4DLUT(cfg)
    criterion = IS4DLUTLoss(LossConfig(hist_bins=16, lambda_sa=0.05))

    low = torch.rand(1, 3, 48, 48)
    high = torch.rand(1, 3, 48, 48)
    out = model(low)

    for key in ("Io", "Ir", "Fd", "Fp", "lut", "region_logits"):
        assert key in out, key
    assert out["Io"].shape == low.shape
    assert out["lut"].shape == (1, 3, 9, 9, 9, 9)

    loss, stats = criterion.generator(out, high)
    assert torch.isfinite(loss), "generator loss is not finite"
    loss.backward()
    assert any(p.grad is not None for p in model.parameters()), "no gradients"

    d_loss, _ = criterion.discriminator(out, high)
    assert torch.isfinite(d_loss)
    print(f"[ok] forward + backward, total loss = {float(loss):.4f}")


def test_synthetic_dataset():
    ds = SyntheticPairs(length=2, size=32, seed=1)
    item = ds[0]
    assert item["low"].shape == (3, 32, 32)
    assert item["high"].shape == (3, 32, 32)
    assert 0.0 <= float(item["low"].min()) and float(item["low"].max()) <= 1.0
    print("[ok] synthetic low/normal pairs")


if __name__ == "__main__":
    test_lut_identity()
    test_histogram_tools()
    test_model_forward_and_loss()
    test_synthetic_dataset()
    print("\nAll smoke tests passed.")
