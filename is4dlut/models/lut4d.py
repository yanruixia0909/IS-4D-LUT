"""Intensity-aware 4D LUT with quadrilinear interpolation (Sec. 3.1, Eq. 1-7).

The LUT has ``M`` basis tables ``psi_m`` of shape (3, N, N, N, N):
the four input axes are (r_in, g_in, b_in, d_in) and the three outputs are
(r_out, g_out, b_out).  An image-adaptive table is obtained by fusing the
basis tables with the coefficients predicted from the image features:

    L_4D = sum_m  omega_m * psi_m                                   (Eq. 1)

Sampling a colour tuple (r, g, b, d) follows Eq. 3-6: the continuous index
is x = v * (N - 1), the 16 surrounding vertices are gathered and combined
with quadrilinear interpolation.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrilinearLUT(nn.Module):
    def __init__(self, n_grid: int = 33, n_basis: int = 3,
                 identity_init: bool = True) -> None:
        super().__init__()
        self.n_grid = int(n_grid)
        self.n_basis = int(n_basis)
        basis = torch.empty(n_basis, 3, n_grid, n_grid, n_grid, n_grid)
        if identity_init:
            axis = torch.linspace(0.0, 1.0, n_grid)
            rr, gg, bb = torch.meshgrid(axis, axis, axis, indexing="ij")
            ident = torch.stack([rr, gg, bb], dim=0)            # (3,N,N,N)
            ident = ident.unsqueeze(-1).expand(3, n_grid, n_grid, n_grid, n_grid)
            basis[:] = ident.unsqueeze(0).expand_as(basis).clone()
        else:
            nn.init.uniform_(basis, 0.0, 1.0)
        self.basis = nn.Parameter(basis)

    # ------------------------------------------------------------------
    def fuse(self, weights: torch.Tensor) -> torch.Tensor:
        """weights: (B, M) -> fused table (B, 3, N, N, N, N).

        Always computed in fp32: the backward of the fusion sums contributions
        from every pixel, which overflows fp16 (AMP).
        """
        return torch.einsum("bm,mcxyzw->bcxyzw", weights.float(), self.basis)

    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor, intensity: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """img, intensity: (B,3,H,W) / (B,1,H,W) in [0,1]; weights: (B,M).

        The result is returned in fp32 (see ``sample``): keeping it in fp32
        avoids fp16 overflow in the backward of the SSIM / MSE losses under AMP.
        """
        lut = self.fuse(weights)
        return self.sample(lut, img, intensity)

    def sample(self, lut: torch.Tensor, img: torch.Tensor,
               intensity: torch.Tensor) -> torch.Tensor:
        """fp32 quadrilinear sampling (stable under AMP)."""
        img = img.float()
        intensity = intensity.float()
        outs = []
        for b in range(lut.shape[0]):
            outs.append(self._sample_single(lut[b], img[b], intensity[b]))
        return torch.stack(outs, dim=0)

    # ------------------------------------------------------------------
    def _sample_single(self, lut: torch.Tensor, img: torch.Tensor,
                       intensity: torch.Tensor) -> torch.Tensor:
        """lut: (3,N,N,N,N); img: (3,H,W); intensity: (1,H,W)."""
        n = self.n_grid
        _, h, w = img.shape
        coords = torch.stack([img[0], img[1], img[2], intensity[0]], dim=-1)
        coords = coords.clamp(0.0, 1.0) * (n - 1)
        base = torch.floor(coords)
        frac = coords - base
        idx = [base[..., i].long().reshape(-1) for i in range(4)]
        fr = [frac[..., i].reshape(-1) for i in range(4)]

        def gather(offset):
            i = (idx[0] + offset[0]).clamp_(0, n - 1)
            j = (idx[1] + offset[1]).clamp_(0, n - 1)
            k = (idx[2] + offset[2]).clamp_(0, n - 1)
            l = (idx[3] + offset[3]).clamp_(0, n - 1)
            return lut[:, i, j, k, l]                      # (3, H*W)

        # separable interpolation: u -> k -> j -> i
        plane_uv = {}
        for d_i, d_j, d_k in itertools.product((0, 1), repeat=3):
            v0 = gather((d_i, d_j, d_k, 0))
            v1 = gather((d_i, d_j, d_k, 1))
            plane_uv[(d_i, d_j, d_k)] = v0 + (v1 - v0) * fr[3]

        plane_jk = {}
        for d_i, d_j in itertools.product((0, 1), repeat=2):
            a = plane_uv[(d_i, d_j, 0)]
            b = plane_uv[(d_i, d_j, 1)]
            plane_jk[(d_i, d_j)] = a + (b - a) * fr[2]

        plane_ij = {}
        for d_i in (0, 1):
            a = plane_jk[(d_i, 0)]
            b = plane_jk[(d_i, 1)]
            plane_ij[d_i] = a + (b - a) * fr[1]

        out = plane_ij[0] + (plane_ij[1] - plane_ij[0]) * fr[0]
        return out.view(3, h, w)


def lut_total_variation(lut: torch.Tensor) -> torch.Tensor:
    """4D smoothness regulariser L_s (Sec. 3.3).

    ``lut`` is (B, 3, N, N, N, N); the penalty is the mean absolute difference
    between neighbouring vertices along every one of the four axes.
    """
    loss = lut.new_zeros(())
    for dim in range(2, 6):
        diff = lut.diff(dim=dim)
        loss = loss + diff.abs().mean()
    return loss / 4.0


def lut_monotonicity(lut: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """4D monotonicity constraint L_m (Sec. 3.3).

    Moving along the intensity axis must not decrease the output luminance:
    ``relu(-(Y_{l+1} - Y_l))``.  ``lut``: (B, 3, N, N, N, N).
    """
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=lut.dtype, device=lut.device)
    y = (lut * weights.view(1, 3, 1, 1, 1, 1, 1)).sum(dim=1)   # (B,N,N,N,N)
    dy = y.diff(dim=-1)
    return F.relu(-dy + eps).mean()
