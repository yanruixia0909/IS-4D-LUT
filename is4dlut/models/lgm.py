"""LUT Generation Module (LGM), Sec. 3.1.

    F_t = E(I_L)                     contextual features
    F_d = D(F_t)                     intensity image
    F_p = I_L + F_d                  lookup prior
    F_c = Concat(F_t, F_d)           (used to predict the fusion coefficients)
    L_4D = Phi(F_c)                  image-adaptive 4D LUT
    I_r = Interp(L_4D, F_p)          refined enhanced image
"""
import torch
import torch.nn as nn

from ..utils import luminance
from .blocks import Encoder, IntensityDecoder
from .lut4d import QuadrilinearLUT


class LGM(nn.Module):
    def __init__(self, n_grid: int = 33, n_basis: int = 3,
                 base_channels: int = 32, num_levels: int = 3,
                 intensity_from: str = "prior") -> None:
        super().__init__()
        self.encoder = Encoder(3, base_channels, num_levels)
        self.intensity_decoder = IntensityDecoder(base_channels, num_levels)
        deepest = base_channels * (2 ** num_levels)
        self.rho = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(deepest, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_basis),
        )
        self.lut = QuadrilinearLUT(n_grid, n_basis)
        self.intensity_from = intensity_from

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)
        fd = torch.sigmoid(self.intensity_decoder(feats))        # (B,1,H,W)
        fp = (x + fd).clamp(0.0, 1.0)                            # lookup prior
        weights = torch.softmax(self.rho(feats[-1]), dim=1)      # (B,M)
        if self.intensity_from == "intensity":
            d_in = fd
        else:
            d_in = luminance(fp)
        fused = self.lut.fuse(weights)                           # (B,3,N,N,N,N)
        ir = self.lut.sample(fused, fp, d_in)                    # (B,3,H,W)
        return {
            "Ir": ir,
            "Fd": fd,
            "Fp": fp,
            "Ft": feats,
            "weights": weights,
            "lut": fused,
        }
