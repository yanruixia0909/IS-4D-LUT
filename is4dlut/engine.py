"""Training / validation engine."""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .datasets import PairedFolderDataset, SyntheticPairs
from .losses import IS4DLUTLoss
from .metrics import evaluate
from .models import IS4DLUT
from .utils import (AverageMeter, count_parameters, get_device, imwrite_unicode,
                    set_seed, tensor_to_img)


def build_datasets(cfg: Config):
    if cfg.train.synthetic:
        train_ds = SyntheticPairs(cfg.train.synthetic_train, cfg.train.synthetic_size,
                                  seed=cfg.train.seed)
        val_ds = SyntheticPairs(cfg.train.synthetic_val, cfg.train.synthetic_size,
                                seed=cfg.train.seed + 12345)
        return train_ds, val_ds

    train_ds = PairedFolderDataset(cfg.train.data_root, cfg.train.crop_size,
                                   cfg.train.use_flip)
    val_root = cfg.train.val_root or cfg.train.data_root
    val_ds = PairedFolderDataset(val_root, cfg.train.crop_size, use_flip=False,
                                 limit=cfg.train.val_limit, full_image=True)
    return train_ds, val_ds


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path: str, model, criterion, opt_g, opt_d, it: int, cfg: Config):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "disc": criterion.state_dict() if criterion.use_gan else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "iter": it,
        "config": cfg.to_dict(),
    }, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


@torch.no_grad()
def save_val_images(model, dataset, device, out_dir: str, it: int, num: int,
                    verbose: bool = True):
    """Dump ``low | ours | GT`` grids for the first ``num`` validation images."""
    if num <= 0 or len(dataset) == 0:
        return []
    was_training = model.training
    model.eval()
    folder = os.path.join(out_dir, f"iter{it:07d}")
    saved = []
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)
        out = model(low)["Io"]
        a, b, c = tensor_to_img(low), tensor_to_img(out), tensor_to_img(high)
        gap = np.full((a.shape[0], 6, 3), 255, np.uint8)
        path = os.path.join(folder, f"{item['name']}_compare.png")
        imwrite_unicode(path, np.concatenate([a, gap, b, gap, c], axis=1))
        saved.append(path)
    if was_training:
        model.train()
    if verbose:
        print(f"  [img] {len(saved)} comparison(s) -> {folder}")
    return saved


def train(cfg: Config, resume: str = None, verbose: bool = True) -> dict:
    device = get_device(cfg.train.device)
    set_seed(cfg.train.seed)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.train.num_workers)

    model = IS4DLUT(cfg.model).to(device)
    criterion = IS4DLUTLoss(cfg.loss).to(device)

    betas = (cfg.train.beta1, cfg.train.beta2)
    opt_g = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=betas,
                             weight_decay=cfg.train.weight_decay)
    opt_d = None
    if criterion.use_gan:
        opt_d = torch.optim.Adam(criterion.disc_parameters(), lr=cfg.train.lr, betas=betas)

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_iter = 0
    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if ckpt.get("disc") is not None and criterion.use_gan:
            criterion.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        if opt_d is not None and ckpt.get("opt_d") is not None:
            opt_d.load_state_dict(ckpt["opt_d"])
        start_iter = int(ckpt.get("iter", 0))

    if verbose:
        print(f"[IS-4D LUT] device={device} params={count_parameters(model)/1e6:.2f}M "
              f"N={cfg.model.n_grid} M={cfg.model.n_basis} "
              f"train={len(train_ds)} val={len(val_ds)}")
    return _run_loop(cfg, device, model, criterion, opt_g, opt_d, train_loader,
                     val_loader, scaler_g, scaler_d, start_iter, verbose)


def _run_loop(cfg, device, model, criterion, opt_g, opt_d, train_loader, val_loader,
              scaler_g, scaler_d, start_iter, verbose):
    use_amp = scaler_g.is_enabled()
    model.train()
    data_iter = _cycle(train_loader)
    meter = AverageMeter()
    tic = time.time()
    last_stats, val_metrics = {}, {}
    img_iter = None

    for it in range(start_iter + 1, cfg.train.total_iters + 1):
        batch = next(data_iter)
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        # ---------------- discriminator ----------------
        if criterion.use_gan:
            for _ in range(max(cfg.train.n_critic, 1)):
                with torch.no_grad():
                    out_d = model(low)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss_d, _ = criterion.discriminator(out_d, high)
                if not torch.isfinite(loss_d):
                    print("[warn] non-finite discriminator loss, step skipped")
                    continue
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(criterion.disc_parameters(),
                                                   cfg.train.grad_clip)
                scaler_d.step(opt_d)
                scaler_d.update()

        # ---------------- generator ----------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(low)
            loss_g, last_stats = criterion.generator(out, high)
        if not torch.isfinite(loss_g):
            print("[warn] non-finite generator loss, step skipped")
            if use_amp:
                scaler_g.update()
            continue
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        if cfg.train.grad_clip and cfg.train.grad_clip > 0:
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler_g.step(opt_g)
        scaler_g.update()

        # ---------------- lr schedule ----------------
        if it > cfg.train.lr_decay_start:
            span = max(cfg.train.total_iters - cfg.train.lr_decay_start, 1)
            ratio = max(0.0, 1.0 - (it - cfg.train.lr_decay_start) / span)
            for group in opt_g.param_groups:
                group["lr"] = cfg.train.lr * ratio
            if opt_d is not None:
                for group in opt_d.param_groups:
                    group["lr"] = cfg.train.lr * ratio

        meter.update(last_stats.get("total", 0.0))
        if verbose and (it % cfg.train.log_every == 0 or it == 1):
            elapsed = time.time() - tic
            detail = " ".join(f"{k}={v:.4f}" for k, v in last_stats.items())
            print(f"iter {it:6d}/{cfg.train.total_iters} | {detail} | "
                  f"avg={meter.avg:.4f} | {elapsed / max(it - start_iter, 1):.2f}s/it")

        if it % cfg.train.val_every == 0:
            val_metrics = evaluate(model, val_loader, device)
            if verbose:
                print(f"  [val] PSNR={val_metrics['psnr']:.2f} "
                      f"SSIM={val_metrics['ssim']:.4f}")
            if cfg.train.val_images:
                save_val_images(model, val_loader.dataset, device, cfg.train.out_dir,
                                it, cfg.train.val_images, verbose)
                img_iter = it

        if it % cfg.train.save_every == 0:
            path = os.path.join(cfg.train.ckpt_dir, f"is4dlut_iter{it:07d}.pth")
            save_checkpoint(path, model, criterion, opt_g, opt_d, it, cfg)
            if verbose:
                print(f"  [ckpt] {path}")

    final_path = os.path.join(cfg.train.ckpt_dir, "is4dlut_last.pth")
    save_checkpoint(final_path, model, criterion, opt_g, opt_d,
                    cfg.train.total_iters, cfg)
    cfg.save(os.path.join(cfg.train.out_dir, "config.json"))
    if cfg.train.val_images and img_iter != cfg.train.total_iters:
        save_val_images(model, val_loader.dataset, device, cfg.train.out_dir,
                        cfg.train.total_iters, cfg.train.val_images, verbose)

    summary = {"ckpt": final_path, "iterations": cfg.train.total_iters,
               "last_loss": last_stats, "val": val_metrics}
    with open(os.path.join(cfg.train.out_dir, "train_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[done] checkpoint -> {final_path}")
    return summary

