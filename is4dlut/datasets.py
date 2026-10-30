"""Datasets: paired folders (LOL-style) and an offline synthetic generator.

The synthetic generator exists so that the whole pipeline can be trained and
evaluated without downloading LOL / FiveK / SMID / SDSD.
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------
# procedural images used by the synthetic dataset
# ----------------------------------------------------------------------
def _smooth_field(rng: np.random.RandomState, size: int, cells: int = 6) -> np.ndarray:
    low = rng.rand(cells, cells, 3).astype(np.float32)
    return cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)


def make_procedural_image(rng: np.random.RandomState, size: int = 128) -> np.ndarray:
    """Colourful, structured image in [0,1] (H,W,3) used as a pseudo GT."""
    img = _smooth_field(rng, size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    for _ in range(rng.randint(2, 5)):
        cx, cy = rng.rand(2)
        radius = rng.uniform(0.08, 0.3)
        color = rng.rand(3).astype(np.float32)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        alpha = rng.uniform(0.4, 0.9)
        img[mask] = (1 - alpha) * img[mask] + alpha * color
    if rng.rand() < 0.5:                      # vertical stripes as texture
        period = rng.randint(4, 14)
        stripes = (np.sin(xx * size / period * np.pi) > 0).astype(np.float32)
        img = 0.8 * img + 0.2 * stripes[..., None]
    img += rng.randn(size, size, 3).astype(np.float32) * 0.02
    return np.clip(img, 0, 1)


def degrade_low_light(gt: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Inverse-gamma + gain + noise, roughly following LOL-like degradation."""
    gamma = rng.uniform(1.8, 3.0)
    gain = rng.uniform(0.25, 0.55)
    low = np.power(np.clip(gt, 0, 1), gamma) * gain
    low = low + rng.randn(*low.shape).astype(np.float32) * rng.uniform(0.005, 0.02)
    if rng.rand() < 0.5:                      # slight blur, sensor-like
        low = cv2.GaussianBlur(low, (3, 3), 0)
    return np.clip(low, 0, 1)


class SyntheticPairs(Dataset):
    """Deterministic on-the-fly low/normal pairs (no external data needed)."""

    def __init__(self, length: int = 16, size: int = 128, seed: int = 0) -> None:
        self.length = int(length)
        self.size = int(size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.RandomState(self.seed * 100003 + index)
        gt = make_procedural_image(rng, self.size)
        low = degrade_low_light(gt, rng)
        to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1).copy()).float()
        return {"low": to_t(low), "high": to_t(gt), "name": f"synthetic_{index:04d}"}


# ----------------------------------------------------------------------
# real paired / unpaired data
# ----------------------------------------------------------------------
def _list_images(folder: str):
    if folder is None or not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if os.path.splitext(n)[1].lower() in IMG_EXTS)


def _imread_float(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def find_low_high_dirs(root: str):
    """Accept several common layouts and return (low_dir, high_dir)."""
    candidates = [
        ("low", "high"), ("input", "target"), ("input", "gt"),
        (os.path.join("our485", "low"), os.path.join("our485", "high")),
        (os.path.join("train", "low"), os.path.join("train", "high")),
        (os.path.join("Train", "Input"), os.path.join("Train", "GT")),
    ]
    for low, high in candidates:
        low_dir, high_dir = os.path.join(root, low), os.path.join(root, high)
        if _list_images(low_dir) and _list_images(high_dir):
            return low_dir, high_dir
    raise FileNotFoundError(
        f"no low/high image folders found under {root}; expected e.g. root/low and root/high"
    )


class PairedFolderDataset(Dataset):
    def __init__(self, root: str, crop_size: int = 256, use_flip: bool = True,
                 limit: int = None, full_image: bool = False) -> None:
        self.low_dir, self.high_dir = find_low_high_dirs(root)
        self.low_files = _list_images(self.low_dir)
        self.high_files = _list_images(self.high_dir)
        n = min(len(self.low_files), len(self.high_files))
        self.n = n if limit is None else min(n, limit)
        self.crop_size = crop_size
        self.use_flip = use_flip
        self.full_image = full_image

    def __len__(self) -> int:
        return self.n

    def _crop(self, low: np.ndarray, high: np.ndarray):
        h, w = low.shape[:2]
        size = min(self.crop_size, h, w)
        top = 0 if size >= h else random.randint(0, h - size)
        left = 0 if size >= w else random.randint(0, w - size)
        return low[top:top + size, left:left + size], high[top:top + size, left:left + size]

    def __getitem__(self, index: int):
        low = _imread_float(self.low_files[index])
        high = _imread_float(self.high_files[index])
        if low.shape != high.shape:
            high = cv2.resize(high, (low.shape[1], low.shape[0]),
                              interpolation=cv2.INTER_AREA)
        if not self.full_image:
            low, high = self._crop(low, high)
            if self.use_flip and random.random() < 0.5:
                low, high = low[:, ::-1], high[:, ::-1]
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()
        return {"low": to_t(low), "high": to_t(high),
                "name": os.path.splitext(os.path.basename(self.low_files[index]))[0]}


class UnpairedDataset(Dataset):
    """Folder of low-light images without ground truth (DICM / MEF / LIME / NPE)."""

    def __init__(self, root: str) -> None:
        self.files = _list_images(root)
        if not self.files:
            raise FileNotFoundError(f"no images found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        img = _imread_float(self.files[index])
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return {"low": t, "name": os.path.splitext(os.path.basename(self.files[index]))[0]}

