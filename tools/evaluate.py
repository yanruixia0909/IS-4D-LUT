"""Evaluate a checkpoint on a paired dataset (PSNR / SSIM).

    python tools/evaluate.py --ckpt checkpoints/is4dlut_last.pth --data-root D:/datasets/LOL/test
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import Config  # noqa: E402
from is4dlut.datasets import PairedFolderDataset  # noqa: E402
from is4dlut.engine import load_checkpoint  # noqa: E402
from is4dlut.metrics import evaluate  # noqa: E402
from is4dlut.models import IS4DLUT  # noqa: E402
from is4dlut.utils import get_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("IS-4D LUT evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", default=None, help="optional json output")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = Config.from_dict(ckpt["config"]) if ckpt.get("config") else Config()

    model = IS4DLUT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = PairedFolderDataset(args.data_root, full_image=True, limit=args.limit)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    metrics = evaluate(model, loader, device)
    print(f"[evaluate] {len(ds)} images from {args.data_root}")
    print(f"  PSNR = {metrics['psnr']:.3f} dB")
    print(f"  SSIM = {metrics['ssim']:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
