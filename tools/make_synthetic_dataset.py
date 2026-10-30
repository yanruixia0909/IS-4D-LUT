"""Write synthetic low/normal image pairs to disk (LOL-like layout).

    python tools/make_synthetic_dataset.py --out data/synthetic_pairs --num 8 --size 128

Creates ``<out>/low/*.png`` and ``<out>/high/*.png`` so that the real-data
training path (``PairedFolderDataset``) can be exercised without downloads.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.datasets import degrade_low_light, make_procedural_image  # noqa: E402
from is4dlut.utils import imwrite_unicode  # noqa: E402


def main():
    p = argparse.ArgumentParser("synthetic paired dataset")
    p.add_argument("--out", default="data/synthetic_pairs")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    low_dir = os.path.join(args.out, "low")
    high_dir = os.path.join(args.out, "high")
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    for i in range(args.num):
        rng = np.random.RandomState(args.seed * 100003 + i)
        gt = make_procedural_image(rng, args.size)
        low = degrade_low_light(gt, rng)
        name = f"synth_{i:03d}.png"
        imwrite_unicode(os.path.join(high_dir, name),
                        (gt[..., ::-1] * 255 + 0.5).astype(np.uint8))   # RGB -> BGR
        imwrite_unicode(os.path.join(low_dir, name),
                        (low[..., ::-1] * 255 + 0.5).astype(np.uint8))

    print(f"[ok] {args.num} pairs written to {os.path.abspath(args.out)}")
    print(f"     low : {low_dir}")
    print(f"     high: {high_dir}")


if __name__ == "__main__":
    main()
