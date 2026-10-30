"""Losses of IS-4D LUT (Sec. 3.3, Eq. 11-13).

L_total1 = alpha_s L_s + alpha_m L_m + alpha_mse L_mse + alpha_ssim L_ssim
L_total2 = beta L_recon + lambda_sc L_sc + lambda_sa L_sa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig
from .models.lut4d import lut_monotonicity, lut_total_variation


# ----------------------------------------------------------------------
# SSIM
# ----------------------------------------------------------------------
def _gaussian_window(size: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = g[:, None] * g[None, :]
    return win.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM of (B,C,H,W) tensors in [0, data_range].

    Computed in fp32: under AMP the fp16 backward of the SSIM terms can overflow.
    """
    dtype = pred.dtype
    pred = pred.float()
    target = target.float()
    b, c, h, w = pred.shape
    size = min(11, h, w)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    win = _gaussian_window(size, 1.5, c, pred.device, pred.dtype)
    pad = size // 2
    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2) + 1e-8)
    return ssim_map.mean().to(dtype)


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


# ----------------------------------------------------------------------
# differentiable histograms (semantic-guided colour histogram loss L_sc)
# ----------------------------------------------------------------------
def soft_histogram(x: torch.Tensor, bins: int = 32,
                   weights: torch.Tensor = None) -> torch.Tensor:
    """Differentiable histogram of (B,C,H,W) values in [0,1].

    Returns (B,C,bins) probabilities.  ``weights`` (B,1,h,w) softly selects a
    semantic region before the histogram is accumulated.
    """
    pos = x.unsqueeze(-1) * (bins - 1)                       # (B,C,H,W,bins)
    centres = torch.arange(bins, device=x.device, dtype=x.dtype)
    kernel = (1.0 - (pos - centres).abs()).clamp(min=0.0)
    kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
    if weights is not None:
        if weights.shape[-2:] != x.shape[-2:]:
            weights = F.interpolate(weights, size=x.shape[-2:], mode="area")
        kernel = kernel * weights.unsqueeze(-1)               # (B,C,H,W,bins)
    hist = kernel.sum(dim=(2, 3))                             # (B,C,bins)
    return hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)


def wasserstein_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """L1 distance between the cumulative distributions (Wasserstein-1)."""
    return (p.cumsum(-1) - q.cumsum(-1)).abs().mean()


def histogram_equalization_consistency(pred: torch.Tensor, target: torch.Tensor,
                                       bins: int = 32) -> torch.Tensor:
    """Auxiliary constraint used in the paper to stabilise uneven histograms."""
    hp = soft_histogram(pred, bins).cumsum(-1)
    ht = soft_histogram(target, bins).cumsum(-1)
    b, c = pred.shape[:2]
    pos = pred.unsqueeze(-1) * (bins - 1)
    centres = torch.arange(bins, device=pred.device, dtype=pred.dtype)
    kernel = (1.0 - (pos - centres).abs()).clamp(min=0.0)
    kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
    map_p = (kernel * hp.view(b, c, 1, 1, bins)).sum(-1)
    map_t = (kernel * ht.view(b, c, 1, 1, bins)).sum(-1)
    return (map_p - map_t).abs().mean()


class SemanticHistogramLoss(nn.Module):
    def __init__(self, bins: int = 32, he_weight: float = 0.1) -> None:
        super().__init__()
        self.bins = bins
        self.he_weight = he_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                region_logits: torch.Tensor) -> torch.Tensor:
        dtype = pred.dtype
        pred = pred.float()
        target = target.float()
        region_logits = region_logits.float()
        size = (max(pred.shape[-2] // 4, 8), max(pred.shape[-1] // 4, 8))
        masks = torch.softmax(F.interpolate(region_logits, size=size, mode="area"), dim=1)
        p = F.interpolate(pred, size=size, mode="area")
        t = F.interpolate(target, size=size, mode="area")
        loss = pred.new_zeros(())
        n_regions = masks.shape[1]
        for k in range(n_regions):
            mk = masks[:, k:k + 1]
            loss = loss + wasserstein_1d(soft_histogram(p, self.bins, mk),
                                         soft_histogram(t, self.bins, mk))
        loss = loss / n_regions
        loss = loss + self.he_weight * histogram_equalization_consistency(p, t, self.bins)
        return loss.to(dtype)


# ----------------------------------------------------------------------
# semantic-guided adversarial loss L_sa
# ----------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (global / local)."""

    def __init__(self, in_ch: int = 3, base: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for i in range(1, n_layers):
            nxt = base * min(2 ** i, 8)
            layers += [nn.Conv2d(ch, nxt, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(nxt, affine=True),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # InstanceNorm is numerically unstable in fp16 (tiny variance -> inf);
        # the discriminator is small, so it is evaluated in fp32.
        dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            out = self.net(x.float())
        return out.to(dtype)


def semantic_weight_map(region_logits: torch.Tensor, size) -> torch.Tensor:
    """Saliency weight in [1, 2] derived from the semantic region assignment."""
    prob = torch.softmax(F.interpolate(region_logits, size=size, mode="bilinear",
                                       align_corners=False), dim=1)
    k = prob.shape[1]
    if k == 1:
        return torch.ones_like(prob)
    saliency = (prob.max(dim=1, keepdim=True)[0] - 1.0 / k) / (1.0 - 1.0 / k)
    return 1.0 + saliency


class GANLoss(nn.Module):
    def __init__(self, gan_type: str = "hinge") -> None:
        super().__init__()
        self.gan_type = gan_type

    def d_loss(self, real: torch.Tensor, fake: torch.Tensor,
               weight: torch.Tensor = None) -> torch.Tensor:
        if self.gan_type == "hinge":
            loss = F.relu(1.0 - real) + F.relu(1.0 + fake)
        elif self.gan_type == "lsgan":
            loss = (real - 1.0) ** 2 + fake ** 2
        else:
            loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
                   F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
            return 0.5 * loss
        if weight is not None:
            w = F.interpolate(weight, size=loss.shape[-2:], mode="area")
            loss = loss * w
        return 0.5 * loss.mean()

    def g_loss(self, fake: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        if self.gan_type == "lsgan":
            loss = (fake - 1.0) ** 2
        else:
            loss = -fake
        if weight is not None:
            w = F.interpolate(weight, size=loss.shape[-2:], mode="area")
            loss = loss * w
        return loss.mean()


def _local_crop(x: torch.Tensor, size: int = 64) -> torch.Tensor:
    h, w = x.shape[-2:]
    size = min(size, h, w)
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return x[..., top:top + size, left:left + size]


class IS4DLUTLoss(nn.Module):
    """Full objective ``L_total = L_total1 + L_total2`` with two discriminators."""

    def __init__(self, cfg: LossConfig = None) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.hist = SemanticHistogramLoss(self.cfg.hist_bins)
        self.gan = GANLoss(self.cfg.gan_type)
        self.use_gan = self.cfg.lambda_sa > 0
        if self.use_gan:
            self.d_global = PatchDiscriminator(3, 32, 3)
            self.d_local = PatchDiscriminator(3, 32, 3)
        else:
            self.d_global = None
            self.d_local = None

    # ------------------------------------------------------------------
    def disc_parameters(self):
        if not self.use_gan:
            return []
        return list(self.d_global.parameters()) + list(self.d_local.parameters())

    def generator(self, out: dict, gt: torch.Tensor):
        c = self.cfg
        l_s = lut_total_variation(out["lut"])
        l_m = lut_monotonicity(out["lut"])
        l_mse = F.mse_loss(out["Ir"], gt)
        l_ssim = 1.0 - ssim(out["Ir"], gt)
        l_recon = charbonnier(out["Io"], gt)
        l_sc = self.hist(out["Io"], gt, out["region_logits"])

        total = (c.alpha_smooth * l_s + c.alpha_mono * l_m +
                 c.alpha_mse * l_mse + c.alpha_ssim * l_ssim +
                 c.beta_recon * l_recon + c.lambda_sc * l_sc)

        stats = {"smooth": l_s.item(), "mono": l_m.item(), "mse": l_mse.item(),
                 "ssim_loss": l_ssim.item(), "recon": l_recon.item(),
                 "sc": l_sc.item()}

        if self.use_gan:
            weight = semantic_weight_map(out["region_logits"], out["Io"].shape[-2:])
            fake_g = self.d_global(out["Io"])
            l_sa_g = self.gan.g_loss(fake_g, weight)
            io_local = _local_crop(out["Io"])
            fake_l = self.d_local(io_local)
            l_sa_l = self.gan.g_loss(fake_l, _local_crop(weight))
            l_sa = l_sa_g + l_sa_l
            total = total + c.lambda_sa * l_sa
            stats["sa"] = l_sa.item()

        stats["total"] = total.item()
        return total, stats

    def discriminator(self, out: dict, gt: torch.Tensor):
        c = self.cfg
        weight = semantic_weight_map(out["region_logits"], out["Io"].shape[-2:])
        d_real = self.d_global(gt.detach())
        d_fake = self.d_global(out["Io"].detach())
        loss_g = self.gan.d_loss(d_real, d_fake, weight)

        gt_local = _local_crop(gt.detach())
        fake_local = _local_crop(out["Io"].detach())
        w_local = _local_crop(weight)
        d_real_l = self.d_local(gt_local)
        d_fake_l = self.d_local(fake_local)
        loss_l = self.gan.d_loss(d_real_l, d_fake_l, w_local)

        loss = 0.5 * (loss_g + loss_l)
        return loss, {"d": loss.item()}

