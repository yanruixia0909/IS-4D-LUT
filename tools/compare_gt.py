"""Build a low / enhanced / GT comparison grid and report PSNR+SSIM.

    python tools/compare_gt.py --ckpt checkpoints/lolv1/is4dlut_last.pth `
        --data-root path/to/LOLv1/Test --out-dir results/lolv1_compare --num 5
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import Config  # noqa: E402
from is4dlut.datasets import _imread_float, find_low_high_dirs  # noqa: E402
from is4dlut.engine import load_checkpoint  # noqa: E402
from is4dlut.metrics import psnr, ssim_score  # noqa: E402
from is4dlut.models import IS4DLUT  # noqa: E402
from is4dlut.utils import (get_device, imwrite_unicode, tensor_to_img)  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("low / ours / GT comparison")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="results/compare")
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = Config.from_dict(ckpt["config"]) if ckpt.get("config") else Config()
    model = IS4DLUT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    low_dir, high_dir = find_low_high_dirs(args.data_root)
    lows = {os.path.splitext(n)[0]: n for n in os.listdir(low_dir)}
    highs = {os.path.splitext(n)[0]: n for n in os.listdir(high_dir)}
    names = sorted(set(lows) & set(highs))[: args.num]
    if not names:
        raise FileNotFoundError("no matched low/high pairs found")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for name in names:
        low = _imread_float(os.path.join(low_dir, lows[name]))
        gt = _imread_float(os.path.join(high_dir, highs[name]))
        x = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)["Io"]
        y = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))).unsqueeze(0).to(device)

        p_in, p_out = psnr(x, y), psnr(out, y)
        s_in, s_out = ssim_score(x, y), ssim_score(out, y)
        rows.append({"name": name, "psnr_in": p_in, "psnr_out": p_out,
                     "ssim_in": s_in, "ssim_out": s_out})
        print(f"{name}: PSNR {p_in:6.2f} -> {p_out:6.2f} dB | SSIM {s_in:.4f} -> {s_out:.4f}")

        a, b, c = tensor_to_img(x), tensor_to_img(out), tensor_to_img(y)
        gap = np.full((a.shape[0], 6, 3), 255, np.uint8)
        grid = np.concatenate([a, gap, b, gap, c], axis=1)
        imwrite_unicode(os.path.join(args.out_dir, f"{name}_compare.png"), grid)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    print(f"\nmean: PSNR {mean['psnr_in']:.2f} -> {mean['psnr_out']:.2f} dB | "
          f"SSIM {mean['ssim_in']:.4f} -> {mean['ssim_out']:.4f}")
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"per_image": rows, "mean": mean}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
