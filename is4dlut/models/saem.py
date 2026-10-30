"""Semantic-Aware Embedding Module (SAEM), Eq. 8-9 (Sec. 3.2)."""
import math

import torch
import torch.nn as nn

from .blocks import LayerNorm2d


class SAEM(nn.Module):
    """Channel-wise semantic-aware attention + feed-forward refinement.

        F_bc  = Conv( F~bs * Softmax( Wk(F~bi) Wq(F~bs)^T / C ) )       (Eq. 9)
        F_att = F_bc * LN(F_bc + F_bs) + LN(F_bc + F~bs)                (Eq. 8)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.wi = nn.Conv2d(channels, channels, 1)
        self.ws = nn.Conv2d(channels, channels, 1)
        self.wk = nn.Conv2d(channels, channels, 1)
        self.wq = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.ln1 = LayerNorm2d(channels)
        self.ln2 = LayerNorm2d(channels)

    def forward(self, f_img: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_img.shape
        dtype = f_img.dtype
        # fp32 for the C x C attention + LayerNorm: under AMP the fp16 matmul of
        # large channel features can overflow and produce NaN.
        with torch.cuda.amp.autocast(enabled=False):
            fi = self.wi(f_img.float())
            fs = self.ws(f_sem.float())

            k = self.wk(fi).flatten(2)                     # (B,C,HW)
            q = self.wq(fs).flatten(2)                     # (B,C,HW)
            attn = torch.softmax(torch.bmm(k, q.transpose(1, 2)) / math.sqrt(c), dim=-1)
            bc = self.proj(torch.bmm(attn, fs.flatten(2)).view(b, c, h, w))

            out = bc * self.ln1(bc + f_sem.float()) + self.ln2(bc + fs)
        return out.to(dtype)
