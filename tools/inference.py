"""Run IS-4D LUT on images (file or folder).

Examples
--------
python tools/inference.py --ckpt checkpoints/demo/is4dlut_last.pth --input some.jpg --out-dir results/infer
python tools/inference.py --ckpt checkpoints/paper/is4dlut_last.pth --input D:/data/DICM --out-dir results/DICM
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import Config  # noqa: E402
from is4dlut.datasets import IMG_EXTS  # noqa: E402
from is4dlut.engine import load_checkpoint  # noqa: E402
from is4dlut.models import IS4DLUT  # noqa: E402
from is4dlut.utils import (get_device, imread_unicode, imwrite_unicode,  # noqa: E402
                           img_to_tensor, tensor_to_img)


def parse_args():
    p = argparse.ArgumentParser("IS-4D LUT inference")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--out-dir", default="results/inference")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-size", type=int, default=0,
                   help="optional resize of the long side (0 = keep original)")
    p.add_argument("--suffix", default="_is4dlut")
    return p.parse_args()


def collect(path: str):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, n) for n in os.listdir(path)
                      if os.path.splitext(n)[1].lower() in IMG_EXTS)
    raise FileNotFoundError(path)


def build_model(ckpt: dict, device):
    cfg = Config.from_dict(ckpt["config"]) if ckpt.get("config") else Config()
    model = IS4DLUT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model = build_model(ckpt, device)

    files = collect(args.input)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[inference] {len(files)} image(s) -> {args.out_dir}")

    for path in files:
        img = imread_unicode(path)
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{name}{args.suffix}.png")
        x = img_to_tensor(img)
        if args.max_size and max(x.shape[-2:]) > args.max_size:
            scale = args.max_size / float(max(x.shape[-2:]))
            new_h = max(int(round(x.shape[-2] * scale)), 8)
            new_w = max(int(round(x.shape[-1] * scale)), 8)
            x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="area")
        x = x.to(device)
        with torch.no_grad():
            y = model.enhance(x)
        imwrite_unicode(out_path, tensor_to_img(y))
        print(f"  {os.path.basename(path)} -> {out_path}")


if __name__ == "__main__":
    main()
