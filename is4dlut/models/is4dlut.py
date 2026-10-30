"""Full IS-4D LUT network (LGM + SFM)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .lgm import LGM
from .sfm import SFM


class IS4DLUT(nn.Module):
    """Intensity- and Semantics-aware 4D LUT for low-light enhancement."""

    def __init__(self, cfg: ModelConfig = None, **kwargs) -> None:
        super().__init__()
        cfg = cfg or ModelConfig(**kwargs)
        self.cfg = cfg
        self.num_levels = cfg.num_levels
        self.lgm = LGM(
            n_grid=cfg.n_grid,
            n_basis=cfg.n_basis,
            base_channels=cfg.base_channels,
            num_levels=cfg.num_levels,
            intensity_from=cfg.intensity_from,
        )
        self.sfm = SFM(
            base_channels=cfg.base_channels,
            num_levels=cfg.num_levels,
            n_regions=cfg.n_regions,
        )

    # ------------------------------------------------------------------
    def _pad(self, x: torch.Tensor):
        factor = 2 ** self.num_levels
        h, w = x.shape[-2:]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        x_pad, (h, w) = self._pad(x)
        out = self.lgm(x_pad)
        delta, region_logits, _ = self.sfm(out["Ir"], x_pad)
        out["delta"] = delta
        out["region_logits"] = region_logits
        io = (out["Ir"] + delta).clamp(0.0, 1.0)
        out["Io"] = io[..., :h, :w]
        out["Ir"] = out["Ir"][..., :h, :w]
        out["Fp"] = out["Fp"][..., :h, :w]
        out["Fd"] = out["Fd"][..., :h, :w]
        return out

    @torch.no_grad()
    def enhance(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["Io"]
