"""Semantic-Aware Fusion Module (SFM), Sec. 3.2.

A semantic encoder builds features from [I_r, I_L]; an image encoder builds
features from I_r.  Starting from the coarsest scale, SAEM injects semantics into
the image features and AFM fuses both modalities with self/cross attention.
The final feature is projected back to the image domain:

    I_o = R(F_s)                                                     (Eq. 10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .afm import AFM
from .blocks import Encoder, UpBlock, conv_block
from .saem import SAEM


class SFM(nn.Module):
    def __init__(self, base_channels: int = 32, num_levels: int = 3,
                 n_regions: int = 3) -> None:
        super().__init__()
        self.num_levels = num_levels
        channels = [base_channels * (2 ** i) for i in range(num_levels + 1)]
        self.img_encoder = Encoder(3, base_channels, num_levels)
        self.sem_encoder = Encoder(6, base_channels, num_levels)
        self.saem = nn.ModuleList([SAEM(channels[i]) for i in range(num_levels + 1)])
        self.afm = nn.ModuleList([AFM(channels[i]) for i in range(num_levels + 1)])
        self.up_img = nn.ModuleList(
            [UpBlock(channels[i + 1], channels[i]) for i in reversed(range(num_levels))]
        )
        self.up_sem = nn.ModuleList(
            [UpBlock(channels[i + 1], channels[i]) for i in reversed(range(num_levels))]
        )
        self.reconstruct = nn.Sequential(
            conv_block(channels[0], channels[0]),
            nn.Conv2d(channels[0], 3, 3, 1, 1),
        )
        self.region_head = nn.Sequential(
            conv_block(channels[0], channels[0]),
            nn.Conv2d(channels[0], n_regions, 1),
        )

    def forward(self, ir: torch.Tensor, il: torch.Tensor):
        f_img = self.img_encoder(ir)
        f_sem = self.sem_encoder(torch.cat([ir, il], dim=1))

        x = f_img[-1]
        y = f_sem[-1]
        for step, level in enumerate(reversed(range(self.num_levels))):
            x = self.up_img[step](x)
            y = self.up_sem[step](y)
            if x.shape[-2:] != f_img[level].shape[-2:]:
                x = F.interpolate(x, size=f_img[level].shape[-2:], mode="bilinear",
                                  align_corners=False)
            if y.shape[-2:] != f_sem[level].shape[-2:]:
                y = F.interpolate(y, size=f_sem[level].shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = x + f_img[level]
            y = y + f_sem[level]
            y = self.saem[level](x, y)          # SAEM: semantics -> image features
            x = self.afm[level](x, y)           # AFM: bidirectional fusion

        delta = self.reconstruct(x)             # R(F_s) as a residual image
        region_logits = self.region_head(y)     # K semantic regions
        return delta, region_logits, x
