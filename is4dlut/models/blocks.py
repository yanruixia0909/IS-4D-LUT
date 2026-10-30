"""Basic building blocks used by the IS-4D LUT network."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension for (B, C, H, W) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def conv_block(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
               activation: bool = True) -> nn.Sequential:
    """Conv (+ReLU). No normalisation is used: the paper's diagram only shows
    Conv / ReLU / Sigmoid / Pooling."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2,
                        bias=False)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x), inplace=True)


class DownBlock(nn.Module):
    """Stride-2 convolution followed by residual refinement."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.refine = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(F.relu(self.down(x), inplace=True))


class UpBlock(nn.Module):
    """Bilinear up-sampling followed by a 3x3 convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class Encoder(nn.Module):
    """Multi-scale encoder returning features at every level.

    Level 0 keeps the input resolution, each following level halves it.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32, num_levels: int = 3) -> None:
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(num_levels + 1)]
        self.stem = conv_block(in_ch, channels[0])
        self.downs = nn.ModuleList(
            [DownBlock(channels[i], channels[i + 1]) for i in range(num_levels)]
        )
        self.channels = channels

    def forward(self, x: torch.Tensor):
        feats = [self.stem(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        return feats


class IntensityDecoder(nn.Module):
    """Decoder that turns the deepest feature into a single-channel image."""

    def __init__(self, base_ch: int = 32, num_levels: int = 3) -> None:
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(num_levels + 1)]
        self.ups = nn.ModuleList(
            [UpBlock(channels[i + 1], channels[i]) for i in reversed(range(num_levels))]
        )
        self.head = nn.Conv2d(channels[0], 1, 3, 1, 1)

    def forward(self, feats):
        x = feats[-1]
        for idx, up in enumerate(self.ups):
            x = up(x)
            skip = feats[len(feats) - 2 - idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = x + skip
        return self.head(x)
