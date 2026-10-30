"""Train IS-4D LUT.

Examples
--------
# paper-scale setting (N=33, M=3, batch 1, lr 1e-4, 1e5 iters)
python tools/train.py --data-root D:/datasets/LOL --preset paper --device cuda

# quick smoke run on the synthetic data
python tools/train.py --preset demo --synthetic --iters 20 --device cuda
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import PRESETS  # noqa: E402
from is4dlut.engine import train  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("IS-4D LUT training")
    p.add_argument("--preset", default="paper", choices=sorted(PRESETS.keys()))
    p.add_argument("--data-root", default=None, help="folder that contains low/ and high/")
    p.add_argument("--val-root", default=None, help="optional separate validation folder")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="use the built-in synthetic low-light pairs")
    p.add_argument("--real-data", action="store_true",
                   help="force the paired folder loader (overrides a synthetic preset)")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-grid", type=int, default=None, help="N (bins per LUT axis)")
    p.add_argument("--n-basis", type=int, default=None, help="M (number of basis LUTs)")
    p.add_argument("--base-channels", type=int, default=None)
    p.add_argument("--num-levels", type=int, default=None)
    p.add_argument("--n-regions", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--val-images", type=int, default=None,
                   help="dump N low/ours/GT grids at every validation (0 = off)")
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--lr-decay-start", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-gan", action="store_true",
                   help="disable the semantic-guided adversarial loss (lambda_sa=0)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PRESETS[args.preset]()

    if args.data_root is not None:
        cfg.train.data_root = args.data_root
    if args.val_root is not None:
        cfg.train.val_root = args.val_root
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.ckpt_dir is not None:
        cfg.train.ckpt_dir = args.ckpt_dir
    if args.synthetic:
        cfg.train.synthetic = True
    if args.real_data:
        cfg.train.synthetic = False
    if args.iters is not None:
        cfg.train.total_iters = args.iters
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.crop_size is not None:
        cfg.train.crop_size = args.crop_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.n_grid is not None:
        cfg.model.n_grid = args.n_grid
    if args.n_basis is not None:
        cfg.model.n_basis = args.n_basis
    if args.base_channels is not None:
        cfg.model.base_channels = args.base_channels
    if args.num_levels is not None:
        cfg.model.num_levels = args.num_levels
    if args.n_regions is not None:
        cfg.model.n_regions = args.n_regions
    if args.val_limit is not None:
        cfg.train.val_limit = args.val_limit
    if args.val_images is not None:
        cfg.train.val_images = args.val_images
    if args.log_every is not None:
        cfg.train.log_every = args.log_every
    if args.val_every is not None:
        cfg.train.val_every = args.val_every
    if args.save_every is not None:
        cfg.train.save_every = args.save_every
    if args.lr_decay_start is not None:
        cfg.train.lr_decay_start = args.lr_decay_start
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.device is not None:
        cfg.train.device = args.device
    if args.no_amp:
        cfg.train.amp = False
    if args.no_gan:
        cfg.loss.lambda_sa = 0.0

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
