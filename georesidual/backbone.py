"""Unmodified mathematical backbone definitions, independently packaged."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1

class SpectralGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * (1.0 + self.gate(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.block(x) + self.residual(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels: int = 158, base: int = 32):
        super().__init__()
        self.spectral_gate = SpectralGate(in_channels)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x = self.spectral_gate(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        x = self.dec3(b, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.out(x)

class MaskDiscriminator(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(base * 8 * 4 * 4, 1)

    def forward(self, mask):
        return self.classifier(self.features(mask).flatten(1))
