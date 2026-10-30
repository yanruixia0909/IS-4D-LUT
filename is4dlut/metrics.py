"""Evaluation metrics: PSNR, SSIM and CIELAB colour difference (dE_ab)."""
import numpy as np
import torch
import torch.nn.functional as F

from .losses import ssim as _ssim


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    return float(-10.0 * np.log10(max(mse, eps)))


@torch.no_grad()
def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(_ssim(pred.clamp(0, 1), target.clamp(0, 1)))


def delta_e_ab(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean CIELAB (a,b) colour difference; inputs are RGB uint8 images."""
    import cv2

    a = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    da = a[..., 1] - b[..., 1]
    db = a[..., 2] - b[..., 2]
    return float(np.sqrt(da ** 2 + db ** 2).mean())


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = None) -> dict:
    """Paired evaluation returning PSNR / SSIM of the final output."""
    model.eval()
    psnrs, ssims = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        out = model(low)["Io"]
        psnrs.append(psnr(out, high))
        ssims.append(ssim_score(out, high))
    model.train()
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim": float(np.mean(ssims)) if ssims else 0.0,
    }
