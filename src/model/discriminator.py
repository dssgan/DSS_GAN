
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size=4, num_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B) if self.group_size is not None else B
        F_ = self.num_features

        y = x.reshape(G, -1, F_, C // F_, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4], keepdim=True).squeeze(2)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)


class DiscriminatorBlock(nn.Module):
    """StyleGAN2-ADA residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)

        if downsample or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = None

        self.resample = nn.AvgPool2d(2) if downsample else None

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        y = F.leaky_relu(self.conv2(y), negative_slope=0.2)

        if self.resample is not None:
            y = self.resample(y)

        if self.skip is not None:
            x = self.skip(x)
        if self.resample is not None:
            x = self.resample(x)

        return (y + x) * (1 / math.sqrt(2))


class Discriminator(nn.Module):

    def __init__(self, resolution=256, num_classes=10, base_ch=96, channel_max=512, mbstd_group_size=8, **kwargs):
        super().__init__()
        self.resolution = resolution
        self.num_classes = num_classes

        assert resolution in [128, 256, 512], f"Resolution must be 128, 256, or 512, got {resolution}"

        def nf(stage):
            return min(base_ch * (2 ** stage), channel_max)

        # Build channel dict based on resolution
        if resolution == 128:
            channels = {
                '128': nf(0),
                '64':  nf(1),
                '32':  nf(2),
                '16':  nf(3),
                '8':   nf(3),
                '4':   nf(3),
            }
        elif resolution == 256:
            channels = {
                '256': nf(0),
                '128': nf(1),
                '64':  nf(2),
                '32':  nf(3),
                '16':  nf(3),
                '8':   nf(3),
                '4':   nf(3),
            }
        else:  # 512
            channels = {
                '512': nf(0),
                '256': nf(1),
                '128': nf(2),
                '64':  nf(3),
                '32':  nf(3),
                '16':  nf(3),
                '8':   nf(3),
                '4':   nf(3),
            }

        self.from_rgb = nn.Conv2d(3, channels[str(resolution)], kernel_size=1, bias=True)

        if resolution == 512:
            self.block_512 = DiscriminatorBlock(channels['512'], channels['256'], downsample=True)
            self.block_256 = DiscriminatorBlock(channels['256'], channels['128'], downsample=True)
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'],  downsample=True)
        elif resolution == 256:
            self.block_256 = DiscriminatorBlock(channels['256'], channels['128'], downsample=True)
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'],  downsample=True)
        else:  # 128
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'],  downsample=True)

        self.block_64  = DiscriminatorBlock(channels['64'],  channels['32'],  downsample=True)
        self.block_32  = DiscriminatorBlock(channels['32'],  channels['16'],  downsample=True)
        self.block_16  = DiscriminatorBlock(channels['16'],  channels['8'],   downsample=True)
        self.block_8   = DiscriminatorBlock(channels['8'],   channels['4'],   downsample=True)

        self.mbstd    = MiniBatchStdDev(group_size=mbstd_group_size, num_features=1)
        self.block_4  = nn.Conv2d(channels['4'] + 1, channels['4'], kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels['4'], channels['4'], kernel_size=4, padding=0, bias=True)

        self.fc    = nn.Linear(channels['4'], 1, bias=True)
        self.embed = nn.Embedding(num_classes, channels['4'])

        print(f"\n{'='*60}")
        print(f"STYLEGAN2-ADA DISCRIMINATOR — {resolution}x{resolution}")
        print(f"{'='*60}")
        ch_str = " | ".join(f"{k}={v}" for k, v in channels.items())
        print(f"Channels: {ch_str}")
        print(f"mbstd group_size={mbstd_group_size}")
        if resolution == 512:
            print(f"512→256 block: NEW  |  256→128 block: NEW  |  rest: unchanged")
        elif resolution == 256:
            print(f"256→128 block: NEW  |  rest: unchanged from 128x128 baseline")
        print(f"{'='*60}\n")

    def forward(self, x, y):
        # x: [B, 3, resolution, resolution]
        h = F.leaky_relu(self.from_rgb(x), negative_slope=0.2)

        if self.resolution == 512:
            h = self.block_512(h)   # 512 → 256
            h = self.block_256(h)   # 256 → 128
            h = self.block_128(h)   # 128 → 64
        elif self.resolution == 256:
            h = self.block_256(h)   # 256 → 128
            h = self.block_128(h)   # 128 → 64
        else:  # 128
            h = self.block_128(h)   # 128 → 64

        h = self.block_64(h)    # 64  → 32
        h = self.block_32(h)    # 32  → 16
        h = self.block_16(h)    # 16  → 8
        h = self.block_8(h)     # 8   → 4

        h = self.mbstd(h)
        h = F.leaky_relu(self.block_4(h),  negative_slope=0.2)
        h = F.leaky_relu(self.conv_out(h), negative_slope=0.2)
        h = h.flatten(1)

        out  = self.fc(h).squeeze(1)
        cond = (self.embed(y) * h).sum(dim=1) / math.sqrt(h.shape[1])
        return out + cond
