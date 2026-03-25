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
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if (downsample or in_ch != out_ch) else None
        self.resample = nn.AvgPool2d(2) if downsample else None

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), 0.2)
        y = F.leaky_relu(self.conv2(y), 0.2)
        if self.resample is not None:
            y = self.resample(y)
        if self.skip is not None:
            x = self.skip(x)
        if self.resample is not None:
            x = self.resample(x)
        return (y + x) * (1 / math.sqrt(2))


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, base_ch=64, channel_max=384, mbstd_group_size=8, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        def nf(s): return min(base_ch * (2 ** s), channel_max)

        ch = {
            '256': nf(0), '128': nf(1), '64': nf(2),
            '32': nf(3), '16': nf(3), '8': nf(3), '4': nf(3),
        }

        self.from_rgb  = nn.Conv2d(3, ch['256'], 1, bias=True)
        self.block_256 = DiscriminatorBlock(ch['256'], ch['128'])
        self.block_128 = DiscriminatorBlock(ch['128'], ch['64'])
        self.block_64  = DiscriminatorBlock(ch['64'],  ch['32'])
        self.block_32  = DiscriminatorBlock(ch['32'],  ch['16'])
        self.block_16  = DiscriminatorBlock(ch['16'],  ch['8'])
        self.block_8   = DiscriminatorBlock(ch['8'],   ch['4'])
        self.mbstd     = MiniBatchStdDev(group_size=mbstd_group_size)
        self.block_4   = nn.Conv2d(ch['4'] + 1, ch['4'], 3, padding=1, bias=True)
        self.conv_out  = nn.Conv2d(ch['4'], ch['4'], 4, padding=0, bias=True)
        self.fc        = nn.Linear(ch['4'], 1, bias=True)
        self.embed     = nn.Embedding(num_classes, ch['4'])

    def forward(self, x, y):
        h = F.leaky_relu(self.from_rgb(x), 0.2)
        h = self.block_256(h)
        h = self.block_128(h)
        h = self.block_64(h)
        h = self.block_32(h)
        h = self.block_16(h)
        h = self.block_8(h)
        h = self.mbstd(h)
        h = F.leaky_relu(self.block_4(h), 0.2)
        h = F.leaky_relu(self.conv_out(h), 0.2)
        h = h.flatten(1)
        out  = self.fc(h).squeeze(1)
        cond = (self.embed(y) * h).sum(dim=1) / math.sqrt(h.shape[1])
        return out + cond
