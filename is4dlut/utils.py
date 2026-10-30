"""Small utilities: device handling, unicode-safe image IO, colour helpers."""
import os
import random

import cv2
import numpy as np
import torch


# ----------------------------------------------------------------------
# reproducibility / device
# ----------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------
# image IO (unicode / Chinese paths are supported)
# ----------------------------------------------------------------------
def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"cannot decode image: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(path)


# ----------------------------------------------------------------------
# tensor <-> image
# ----------------------------------------------------------------------
def img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 BGR -> (1,3,H,W) float RGB in [0,1]."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) float RGB in [0,1] -> HWC uint8 BGR."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().float().clamp(0, 1).cpu()
    rgb = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# colour helpers (all operate on RGB tensors)
# ----------------------------------------------------------------------
def luminance(x: torch.Tensor) -> torch.Tensor:
    """RGB (...,3,H,W) -> Y (...,1,H,W) using BT.601 weights."""
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device)
    return (x * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def rgb_to_hsv_torch(x: torch.Tensor) -> torch.Tensor:
    """Differentiable-ish RGB->HSV (H in [0,1], S in [0,1], V in [0,1])."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    maxc = torch.max(x, dim=1, keepdim=True)[0]
    minc = torch.min(x, dim=1, keepdim=True)[0]
    delta = maxc - minc + 1e-8
    v = maxc
    s = delta / (maxc + 1e-8)
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta
    h = torch.zeros_like(maxc)
    h = torch.where(maxc == r, bc - gc, h)
    h = torch.where(maxc == g, 2.0 + rc - bc, h)
    h = torch.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return torch.cat([h, s, v], dim=1)


# ----------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
