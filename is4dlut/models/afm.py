"""Attention Fusion Module (AFM), Sec. 3.2.

Self-attention is applied to the image and to the semantic features, followed by
bidirectional cross-attention.  The six maps are concatenated and fused with
"Conv -> ReLU -> Conv -> Sigmoid", the sigmoid acting as a channel gate that is
applied to the summed features.
"""
import math

import torch
import torch.nn as nn

from .blocks import conv_block


class ChannelAttention(nn.Module):
    """Scaled dot-product attention over channels (C x C matrices)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = 1.0 / math.sqrt(channels)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query.shape
        dtype = query.dtype
        # the C x C attention is computed in fp32 (AMP fp16 matmul can overflow)
        with torch.cuda.amp.autocast(enabled=False):
            q = self.q(query.float()).flatten(2)
            k = self.k(key.float()).flatten(2)
            v = self.v(key.float()).flatten(2)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
            out = torch.bmm(attn, v).view(b, c, h, w)
        return out.to(dtype)


class AFM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_img = ChannelAttention(channels)
        self.self_sem = ChannelAttention(channels)
        self.cross_img = ChannelAttention(channels)
        self.cross_sem = ChannelAttention(channels)
        self.fuse = nn.Sequential(
            conv_block(channels * 6, channels * 2),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )
        self.out = conv_block(channels, channels)

    def forward(self, f_img: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        f1s = self.self_img(f_img, f_img)
        f2s = self.self_sem(f_sem, f_sem)
        f1c = self.cross_img(f_img, f_sem)
        f2c = self.cross_sem(f_sem, f_img)
        gate = self.fuse(torch.cat([f_img, f_sem, f1s, f2s, f1c, f2c], dim=1))
        return self.out((f_img + f_sem) * gate + f1c + f2c)
