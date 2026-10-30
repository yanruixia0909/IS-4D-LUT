"""Measure inference latency of the full enhancement pipeline.

Protocol used in the paper: batch size 1, FP32, fixed resolution, image loading
and result saving excluded, 50 warm-up iterations followed by 200 timed runs
synchronised with CUDA events; mean and standard deviation are reported.

Examples
--------
python tools/benchmark.py --ckpt checkpoints/lolv1/is4dlut_last.pth --size 400x600
python tools/benchmark.py --ckpt checkpoints/lolv1/is4dlut_last.pth --size 256 --runs 200
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from is4dlut.config import Config  # noqa: E402
from is4dlut.engine import load_checkpoint  # noqa: E402
from is4dlut.models import IS4DLUT  # noqa: E402
from is4dlut.utils import count_parameters, get_device  # noqa: E402


def parse_size(text: str):
    if "x" in text.lower():
        h, w = text.lower().split("x")
        return int(h), int(w)
    return int(text), int(text)


def parse_args():
    p = argparse.ArgumentParser("IS-4D LUT inference benchmark")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", default="400x600", help="'HxW' or a single int")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out", default=None, help="optional json output path")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    h, w = parse_size(args.size)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg = Config.from_dict(ckpt["config"]) if ckpt.get("config") else Config()
    model = IS4DLUT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # fixed input, deterministic; image IO is intentionally excluded
    torch.manual_seed(0)
    x = torch.rand(1, 3, h, w, device=device)

    def run_once():
        with torch.no_grad():
            model.enhance(x)

    for _ in range(max(args.warmup, 0)):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.runs):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            start.record()
            run_once()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            import time
            t0 = time.perf_counter()
            run_once()
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times, dtype=np.float64)
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == "cuda" else 0.0
    result = {
        "checkpoint": args.ckpt,
        "device": str(device),
        "precision": "fp32",
        "batch_size": 1,
        "resolution": [h, w],
        "params_M": round(count_parameters(model) / 1e6, 2),
        "warmup": args.warmup,
        "runs": args.runs,
        "mean_ms": round(float(times.mean()), 2),
        "std_ms": round(float(times.std(ddof=1)) if len(times) > 1 else 0.0, 2),
        "median_ms": round(float(np.median(times)), 2),
        "p90_ms": round(float(np.percentile(times, 90)), 2),
        "min_ms": round(float(times.min()), 2),
        "max_ms": round(float(times.max()), 2),
        "fps": round(1000.0 / float(times.mean()), 2),
        "peak_mem_MB": round(float(peak), 1),
    }
    for k, v in result.items():
        print(f"{k:>12s}: {v}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
