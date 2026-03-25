import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


class DiffAug(nn.Module):
    def __init__(self, brightness=0.5, contrast=0.5, flip_prob=0.5, p_brightness=0.5, p_contrast=0.5):
        super().__init__()
        self.brightness   = brightness
        self.contrast     = contrast
        self.flip_prob    = flip_prob
        self.p_brightness = p_brightness
        self.p_contrast   = p_contrast

    def forward(self, x):
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        if torch.rand(1, device=device).item() < self.flip_prob:
            x = torch.flip(x, dims=[3])

        if self.brightness > 0:
            apply = (torch.rand(B, 1, 1, 1, device=device) < self.p_brightness).to(dtype)
            delta = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.brightness
            x = (x + apply * delta).clamp(-1, 1)

        if self.contrast > 0:
            apply  = (torch.rand(B, 1, 1, 1, device=device) < self.p_contrast).to(dtype)
            mean   = x.mean(dim=[2, 3], keepdim=True)
            factor = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * self.contrast
            x = ((x - mean) * (1 - apply + apply * factor) + mean).clamp(-1, 1)

        return x
