"""End-to-end demo: train a small IS-4D LUT on synthetic data, then enhance
held-out images and report PSNR / SSIM before and after.

    python tools/demo.py --iters 6 --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import Config, demo_config  # noqa: E402
from is4dlut.datasets import SyntheticPairs  # noqa: E402
from is4dlut.engine import load_checkpoint, train  # noqa: E402
from is4dlut.metrics import psnr, ssim_score  # noqa: E402
from is4dlut.models import IS4DLUT  # noqa: E402
from is4dlut.utils import (get_device, imwrite_unicode, set_seed,  # noqa: E402
                           tensor_to_img)


def parse_args():
    p = argparse.ArgumentParser("IS-4D LUT demo")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--n-grid", type=int, default=None)
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--num-levels", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = demo_config()
    if args.iters is not None:
        cfg.train.total_iters = args.iters
    if args.n_grid is not None:
        cfg.model.n_grid = args.n_grid
    if args.base_channels is not None:
        cfg.model.base_channels = args.base_channels
    if args.num_levels is not None:
        cfg.model.num_levels = args.num_levels
    if args.device is not None:
        cfg.train.device = args.device
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
        cfg.train.ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    if args.seed is not None:
        cfg.train.seed = args.seed

    set_seed(cfg.train.seed)
    summary = train(cfg)

    device = get_device(cfg.train.device)
    ckpt = load_checkpoint(summary["ckpt"], map_location=device)
    model = IS4DLUT(Config.from_dict(ckpt["config"]).model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SyntheticPairs(cfg.train.synthetic_val, cfg.train.synthetic_size,
                        seed=cfg.train.seed + 12345)
    out_dir = cfg.train.out_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        low = item["low"].unsqueeze(0).to(device)
        gt = item["high"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(low)["Io"]

        p_before, p_after = psnr(low, gt), psnr(out, gt)
        s_before, s_after = ssim_score(low, gt), ssim_score(out, gt)
        rows.append({"name": item["name"], "psnr_before": p_before,
                     "psnr_after": p_after, "ssim_before": s_before,
                     "ssim_after": s_after})

        low_img, out_img, gt_img = (tensor_to_img(low), tensor_to_img(out),
                                    tensor_to_img(gt))
        gap = np.full((low_img.shape[0], 6, 3), 255, np.uint8)
        grid = np.concatenate([low_img, gap, out_img, gap, gt_img], axis=1)
        imwrite_unicode(os.path.join(out_dir, f"comparison_{i:02d}.png"), grid)
        print(f"[{item['name']}] PSNR {p_before:.2f} -> {p_after:.2f} dB | "
              f"SSIM {s_before:.4f} -> {s_after:.4f}")

    means = {k: float(np.mean([r[k] for r in rows]))
             for k in ("psnr_before", "psnr_after", "ssim_before", "ssim_after")}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_image": rows, "mean": means}, f, indent=2, ensure_ascii=False)

    print("\n=== mean over held-out synthetic images ===")
    print(f"  PSNR: {means['psnr_before']:.2f} -> {means['psnr_after']:.2f} dB")
    print(f"  SSIM: {means['ssim_before']:.4f} -> {means['ssim_after']:.4f}")
    print(f"  results: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
