from __future__ import annotations

import math

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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, hidden_channels * 4, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMUNet(nn.Module):
    def __init__(self, in_channels: int = 158, base: int = 32, steps: int = 10):
        super().__init__()
        self.steps = steps
        self.group_channels = math.ceil(in_channels / steps)
        self.padded_channels = self.group_channels * steps
        self.spectral_embed = nn.Conv2d(self.group_channels, base, 3, padding=1, bias=False)
        self.cell = ConvLSTMCell(base, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        if x.shape[1] < self.padded_channels:
            x = F.pad(x, (0, 0, 0, 0, 0, self.padded_channels - x.shape[1]))
        groups = x.view(x.shape[0], self.steps, self.group_channels, x.shape[2], x.shape[3])
        h = torch.zeros((x.shape[0], self.cell.hidden_channels, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for step in range(self.steps):
            embedded = F.relu(self.spectral_embed(groups[:, step]), inplace=True)
            h, c = self.cell(embedded, (h, c))
        e1 = h
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        x = self.dec3(b, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.out(x)


def grid_adjacency(size: int) -> torch.Tensor:
    nodes = size * size
    matrix = torch.zeros((nodes, nodes), dtype=torch.float32)
    for row in range(size):
        for col in range(size):
            node = row * size + col
            matrix[node, node] = 1.0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < size and 0 <= cc < size:
                    matrix[node, rr * size + cc] = 1.0
    degree = matrix.sum(dim=1)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


class GraphBlock(nn.Module):
    def __init__(self, channels: int, nodes_per_side: int = 8):
        super().__init__()
        self.register_buffer("adjacency", grid_adjacency(nodes_per_side), persistent=False)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        nodes = x.flatten(2).transpose(1, 2)
        y = torch.matmul(self.adjacency, nodes)
        nodes = self.norm1(nodes + F.gelu(self.linear1(y)))
        y = torch.matmul(self.adjacency, nodes)
        nodes = self.norm2(nodes + F.gelu(self.linear2(y)))
        return nodes.transpose(1, 2).reshape(b, c, h, w)


class GCNUNet(nn.Module):
    def __init__(self, in_channels: int = 158, base: int = 32):
        super().__init__()
        self.spectral_gate = SpectralGate(in_channels)
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.graph = GraphBlock(base * 8, 8)
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
        b = self.graph(self.bottleneck(self.pool(e3)))
        x = self.dec3(b, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.out(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        return tokens, h, w


class EfficientAttention(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h: int, w: int):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        source = x
        if self.sr_ratio > 1:
            source = x.transpose(1, 2).reshape(b, c, h, w)
            source = self.sr(source).flatten(2).transpose(1, 2)
            source = self.norm(source)
        kv = self.kv(source).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class MixFFN(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, h: int, w: int):
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = F.gelu(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)


class MixTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim)

    def forward(self, x, h: int, w: int):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class SegFormerB0(nn.Module):
    def __init__(self, in_channels: int = 158):
        super().__init__()
        self.spectral_gate = SpectralGate(in_channels)
        dims = [32, 64, 160, 256]
        heads = [1, 2, 5, 8]
        sr = [8, 4, 2, 1]
        self.embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(in_channels, dims[0], 7, 4),
                OverlapPatchEmbed(dims[0], dims[1], 3, 2),
                OverlapPatchEmbed(dims[1], dims[2], 3, 2),
                OverlapPatchEmbed(dims[2], dims[3], 3, 2),
            ]
        )
        self.blocks = nn.ModuleList(
            [nn.ModuleList([MixTransformerBlock(dims[i], heads[i], sr[i]) for _ in range(2)]) for i in range(4)]
        )
        decoder_dim = 128
        self.projections = nn.ModuleList([nn.Conv2d(dim, decoder_dim, 1) for dim in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, 1, 1),
        )

    def forward(self, x):
        x = self.spectral_gate(x)
        input_size = x.shape[-2:]
        features = []
        for embed, blocks in zip(self.embeds, self.blocks):
            tokens, h, w = embed(x)
            for block in blocks:
                tokens = block(tokens, h, w)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)
            features.append(x)
        target = features[0].shape[-2:]
        decoded = [
            F.interpolate(proj(feature), size=target, mode="bilinear", align_corners=False)
            for proj, feature in zip(self.projections, features)
        ]
        logits = self.fuse(torch.cat(decoded, dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


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


def create_model(name: str, in_channels: int = 158) -> nn.Module:
    factories = {
        "unet": UNet,
        "convlstm_unet": ConvLSTMUNet,
        "gcn_unet": GCNUNet,
        "segformer_b0": SegFormerB0,
        "gan_unet": UNet,
    }
    if name not in factories:
        raise ValueError(f"Unknown model {name}")
    return factories[name](in_channels=in_channels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
