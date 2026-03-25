import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class FiLM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta  = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, y):
        g = self.gamma(y).view(-1, self.num_features, 1, 1)
        b = self.beta(y).view(-1, self.num_features, 1, 1)
        return g * x + b


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.mamba.named_parameters():
            if 'dt_proj' in name or 'x_proj' in name:
                (nn.init.xavier_uniform_(p, gain=0.5) if p.dim() >= 2
                 else nn.init.normal_(p, std=0.02))
            elif 'conv1d' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.8)
            elif 'in_proj' in name or 'out_proj' in name:
                (nn.init.xavier_uniform_(p, gain=0.8) if p.dim() >= 2
                 else nn.init.zeros_(p))

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class DirectionalWeighting(nn.Module):
    def __init__(self, num_directions, z_routing_dim=None, temperature=2.0):
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(num_directions))
        self.temperature  = temperature
        self.use_z = z_routing_dim is not None
        if self.use_z:
            self.z_to_w = nn.Linear(z_routing_dim, num_directions)
            nn.init.zeros_(self.z_to_w.weight)
            nn.init.zeros_(self.z_to_w.bias)

    def forward(self, z_dir=None, class_bias=None):
        w = self.base_weights
        if self.use_z and z_dir is not None:
            w = w.unsqueeze(0) + self.z_to_w(z_dir)
        else:
            w = w.unsqueeze(0)
        if class_bias is not None:
            w = w + class_bias
        return F.softmax(w / self.temperature, dim=-1)


class StyleGAN2Block(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.channels = channels
        self.style  = nn.Embedding(num_classes, channels)
        nn.init.zeros_(self.style.weight)
        self.weight = nn.Parameter(
            torch.randn(channels, channels, 3, 3) * (2 / (channels * 9)) ** 0.5
        )
        self.bias           = nn.Parameter(torch.zeros(channels))
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        B, C, H, W = x.shape
        style  = self.style(y).view(B, 1, C, 1, 1)
        weight = self.weight.unsqueeze(0) * (style + 1)
        demod  = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod
        x      = x.reshape(1, B * C, H, W)
        weight = weight.reshape(B * self.channels, C, 3, 3)
        out    = F.conv2d(x, weight, padding=1, groups=B)
        out    = out.reshape(B, self.channels, H, W) + self.bias.view(1, -1, 1, 1)
        if self.training and self.noise_strength.item() > 0:
            out = out + self.noise_strength * torch.randn(B, 1, H, W, device=out.device, dtype=out.dtype)
        return F.leaky_relu(out, 0.2)


class DLRBlock(nn.Module):
    def __init__(
        self, ch, d_state=16, d_conv=4, expand=2, spatial_size=8,
        scan_directions=None, use_z_routing=False, z_routing_dim=None,
        z_mlp_hidden=None, temperature=2.0, routing_clip=None,
        residual_routing=False, num_classes=None, class_embed_dim=128,
    ):
        super().__init__()
        self.ch              = ch
        self.spatial_size    = spatial_size
        self.routing_clip    = routing_clip
        self.residual_routing = residual_routing
        self.num_classes     = num_classes
        self.scan_directions = scan_directions or ['row_fwd', 'row_bwd', 'col_fwd', 'col_bwd']
        self.num_directions  = len(self.scan_directions)
        self.use_z_routing   = use_z_routing
        self.z_routing_dim   = z_routing_dim

        if use_z_routing:
            assert z_routing_dim is not None
            assert z_routing_dim % self.num_directions == 0
            self.z_chunk_dim = z_routing_dim // self.num_directions
            z_mlp_hidden = z_mlp_hidden or self.z_chunk_dim * 2

            if num_classes is not None:
                self.class_embeds = nn.ModuleDict({
                    d: nn.Embedding(num_classes, class_embed_dim)
                    for d in self.scan_directions
                })
                for emb in self.class_embeds.values():
                    nn.init.normal_(emb.weight, 0, 0.25)
                self.class_routing_proj = nn.Linear(class_embed_dim, self.num_directions)
                nn.init.normal_(self.class_routing_proj.weight, std=0.001)
                nn.init.zeros_(self.class_routing_proj.bias)
                self.routing_alpha = nn.Parameter(torch.full((self.num_directions,), 0.01))

            input_dim = self.z_chunk_dim + (class_embed_dim if num_classes else 0)
            self.z_projections = nn.ModuleDict({
                d: nn.Sequential(
                    nn.Linear(input_dim, z_mlp_hidden), nn.GELU(),
                    nn.Linear(z_mlp_hidden, 2 * ch)
                )
                for d in self.scan_directions
            })

        self.mamba_blocks = nn.ModuleDict({
            d: MambaBlock(ch, d_state, d_conv, expand)
            for d in self.scan_directions
        })

        self.direction_weighting = DirectionalWeighting(
            num_directions=self.num_directions,
            z_routing_dim=z_routing_dim if use_z_routing else None,
            temperature=temperature,
        )

        if 'diag_left' in self.scan_directions or 'diag_right' in self.scan_directions:
            H = W = spatial_size
            dl, dr = [], []
            for k in range(H + W - 1):
                i0, i1 = min(k, H - 1), max(0, k - (W - 1))
                for i in range(i0, i1 - 1, -1):
                    dl.append(i * W + (k - i))
                for i in range(i1, i0 + 1):
                    dr.append(i * W + (k - i))
            self.register_buffer('diag_left_idx',  torch.tensor(dl, dtype=torch.long))
            self.register_buffer('diag_right_idx', torch.tensor(dr, dtype=torch.long))

        self.last_direction_weights = None
        self.last_gamma_abs_mean    = None
        self.last_beta_abs_mean     = None

    def _rotate(self, x, k):
        return torch.rot90(x, k=k, dims=[2, 3]) if k else x

    def _scan(self, h, direction):
        B, C, H, W = h.shape
        if direction == 'row_fwd':
            return h.view(B, C, H * W).transpose(1, 2)
        elif direction == 'row_bwd':
            return h.flip([3]).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'col_fwd':
            return h.permute(0, 1, 3, 2).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'col_bwd':
            return h.flip([2]).permute(0, 1, 3, 2).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'diag_left':
            return h.view(B, C, H * W).transpose(1, 2)[:, self.diag_left_idx, :]
        elif direction == 'diag_right':
            return h.view(B, C, H * W).transpose(1, 2)[:, self.diag_right_idx, :]
        raise ValueError(f"Unknown direction: {direction}")

    def _unscan(self, seq, direction, H, W):
        B, N, C = seq.shape
        if direction == 'row_fwd':
            return seq.transpose(1, 2).view(B, C, H, W)
        elif direction == 'row_bwd':
            return seq.transpose(1, 2).view(B, C, H, W).flip([3])
        elif direction == 'col_fwd':
            return seq.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)
        elif direction == 'col_bwd':
            return seq.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2).flip([2])
        elif direction == 'diag_left':
            flat = torch.zeros(B, H * W, C, device=seq.device, dtype=seq.dtype)
            flat[:, self.diag_left_idx, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        elif direction == 'diag_right':
            flat = torch.zeros(B, H * W, C, device=seq.device, dtype=seq.dtype)
            flat[:, self.diag_right_idx, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        raise ValueError(f"Unknown direction: {direction}")

    def forward(self, h, z_dir=None, y=None, rot_k=None, prev_outputs=None):
        B, C, H, W = h.shape
        h_rot = self._rotate(h, rot_k)

        if self.use_z_routing and z_dir is not None:
            class_bias = None
            if self.num_classes is not None and y is not None and hasattr(self, 'class_routing_proj'):
                cls_emb = sum(self.class_embeds[d](y) for d in self.scan_directions) / self.num_directions
                cls_b   = self.class_routing_proj(cls_emb)
                alpha   = torch.abs(self.routing_alpha).unsqueeze(0)
                class_bias = alpha * cls_b
            dir_weights = self.direction_weighting(z_dir, class_bias=class_bias)
        else:
            dir_weights = self.direction_weighting()

        self.last_direction_weights = (
            dir_weights.mean(0) if dir_weights.dim() == 2 else dir_weights
        ).detach().cpu()

        if hasattr(self, 'routing_alpha'):
            self.last_routing_alpha = self.routing_alpha.detach().cpu()

        outputs = []
        for idx, direction in enumerate(self.scan_directions):
            seq = self._scan(h_rot, direction)

            if self.use_z_routing and z_dir is not None:
                z_chunk = z_dir[:, idx * self.z_chunk_dim:(idx + 1) * self.z_chunk_dim]
                if self.num_classes is not None and y is not None:
                    dlr_input = torch.cat([z_chunk, self.class_embeds[direction](y)], dim=1)
                else:
                    dlr_input = z_chunk

                gamma_raw, beta_raw = self.z_projections[direction](dlr_input).chunk(2, dim=1)
                if self.residual_routing and self.routing_clip is not None:
                    gamma = torch.tanh(gamma_raw) * self.routing_clip + 1.0
                    beta  = torch.tanh(beta_raw)  * self.routing_clip
                elif self.routing_clip is not None:
                    gamma = gamma_raw.clamp(-self.routing_clip, self.routing_clip)
                    beta  = beta_raw.clamp(-self.routing_clip, self.routing_clip)
                else:
                    gamma, beta = gamma_raw + 1.0, beta_raw

                seq = gamma.unsqueeze(1) * seq + beta.unsqueeze(1)
                if idx == 0:
                    self.last_gamma_abs_mean = gamma.abs().mean().item()
                    self.last_beta_abs_mean  = beta.abs().mean().item()

            seq = self.mamba_blocks[direction](seq)
            outputs.append(self._unscan(seq, direction, H, W))

        if dir_weights.dim() == 1:
            result = sum(w * o for w, o in zip(dir_weights, outputs))
        else:
            result = torch.zeros_like(outputs[0])
            for idx, o in enumerate(outputs):
                result = result + dir_weights[:, idx:idx+1, None, None] * o

        return self._rotate(result, -rot_k if rot_k else None)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim               = config['z_dim']
        self.num_classes         = config['num_classes']
        self.use_vit_residual    = config.get('use_vit_residual', False)
        self.token_residual_weight   = config.get('token_residual_weight', 0.3)
        self.spatial_residual_weight = config.get('spatial_residual_weight', 0.3)
        self.temperature         = config.get('temperature', 2.0)
        self.use_multiscale_skips = config.get('use_multiscale_skips', False)
        self.skip_8_to_64        = config.get('skip_8_to_64', 0.0)
        self.skip_16_to_128      = config.get('skip_16_to_128', 0.0)
        self.use_blur_upsample   = config.get('use_blur_upsample', False)
        self.use_dense_connections = config.get('use_dense_connections', False)
        self.use_lightweight_attention = config.get('use_lightweight_attention', False)
        self.attention_stages    = config.get('attention_stages', [])
        self.attention_residual_weight = config.get('attention_residual_weight', 0.1)
        self.use_grouped_conv    = config.get('use_grouped_conv', False)
        self.film_enabled        = config.get('film_enabled', {k: False for k in ['8x8','16x16','32x32','64x64']})
        self.z_tok_dim           = config.get('z_tok_dim', self.z_dim // 2)
        self.z_dir_dim           = self.z_dim - self.z_tok_dim
        self.rotation_modes      = config.get('rotation_modes', ['none', 'rot180'])
        self.rotation_mode_to_k  = {'none': None, 'rot90': 1, 'rot180': 2, 'rot270': 3}
        self.use_pixel_shuffle_32 = config.get('use_pixel_shuffle_32', False)
        self.use_pixel_shuffle_64 = config.get('use_pixel_shuffle_64', False)
        self.z_mlp_hidden        = config.get('z_mlp_hidden', {})

        cfg_tok   = config['mamba_tokens']
        cfg_8     = config['mamba_8x8']
        cfg_16    = config['mamba_16x16']
        cfg_32    = config['mamba_32x32']
        cfg_64    = config['mamba_64x64']
        cfg_128   = config['mamba_128x128']
        cfg_out   = config['output']

        ch8   = config['ch_8x8']
        ch16  = config['ch_16x16']
        ch32  = config['ch_32x32']
        ch64  = config['ch_64x64']
        ch128 = config['ch_128x128']
        ch256 = config['ch_256x256']
        self.ch_8x8 = ch8; self.ch_16x16 = ch16; self.ch_32x32 = ch32
        self.ch_64x64 = ch64; self.ch_128x128 = ch128; self.ch_256x256 = ch256

        def dlr(ch, cfg, spatial, z_mlp_key):
            return DLRBlock(
                ch=ch, d_state=cfg['d_state'], d_conv=cfg['d_conv'], expand=cfg['expand'],
                spatial_size=spatial, scan_directions=cfg['scan_directions'],
                use_z_routing=cfg['use_z_routing'], z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=self.z_mlp_hidden.get(z_mlp_key, 256),
                temperature=self.temperature, routing_clip=cfg.get('routing_clip'),
                residual_routing=cfg.get('residual_routing', False),
                num_classes=self.num_classes, class_embed_dim=128,
            )

        self.z_to_tokens  = nn.Linear(self.z_tok_dim, 64 * ch8)
        self.pos_enc64    = LearnedPositionalEncoding(64, ch8)
        self.mamba_tokens = nn.ModuleList([
            MambaBlock(ch8, cfg_tok['d_state'], cfg_tok['d_conv'], cfg_tok['expand'])
            for _ in range(cfg_tok['depth'])
        ])
        self.class_embed  = nn.Embedding(self.num_classes, ch8)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        self.mamba_8x8   = nn.ModuleList([dlr(ch8,  cfg_8,  8,  '8x8')  for _ in range(cfg_8['depth'])])
        self.film_8x8    = FiLM(ch8, self.num_classes)
        self.up_8_to_16  = self._up(ch8, ch16)

        self.mamba_16x16 = nn.ModuleList([dlr(ch16, cfg_16, 16, '16x16') for _ in range(cfg_16['depth'])])
        self.film_16x16  = FiLM(ch16, self.num_classes)
        self.up_16_to_32 = self._up(ch16, ch32)

        self.ch_32x32_shuffled = ch32 * 4 if self.use_pixel_shuffle_32 else ch32
        mamba_s32 = 16 if self.use_pixel_shuffle_32 else 32
        if self.use_pixel_shuffle_32:
            self.pixel_unshuffle_32 = nn.PixelUnshuffle(2)
            self.pixel_shuffle_32   = nn.PixelShuffle(2)
        self.mamba_32x32 = nn.ModuleList([dlr(self.ch_32x32_shuffled, cfg_32, mamba_s32, '32x32') for _ in range(cfg_32['depth'])])
        self.film_32x32  = FiLM(ch32, self.num_classes)
        self.up_32_to_64 = self._up(ch32, ch64)

        self.ch_64x64_shuffled = ch64 * 4 if self.use_pixel_shuffle_64 else ch64
        mamba_s64 = 32 if self.use_pixel_shuffle_64 else 64
        if self.use_pixel_shuffle_64:
            self.pixel_unshuffle_64 = nn.PixelUnshuffle(2)
            self.pixel_shuffle_64   = nn.PixelShuffle(2)
        self.mamba_64x64 = nn.ModuleList([dlr(self.ch_64x64_shuffled, cfg_64, mamba_s64, '64x64') for _ in range(cfg_64['depth'])])
        self.film_64x64  = FiLM(ch64, self.num_classes)

        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            self.skip_proj_8_to_64 = nn.Sequential(nn.Upsample(scale_factor=8, mode='nearest'), nn.Conv2d(ch8, ch64, 1))
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            self.skip_proj_16_to_128 = nn.Sequential(nn.Upsample(scale_factor=8, mode='nearest'), nn.Conv2d(ch16, ch128, 1))

        self.up_64_to_128  = self._up(ch64, ch128)
        self.mamba_128x128 = nn.ModuleList([
            DLRBlock(
                ch=ch128, d_state=cfg_128['d_state'], d_conv=cfg_128['d_conv'], expand=cfg_128['expand'],
                spatial_size=128, scan_directions=cfg_128['scan_directions'],
                use_z_routing=cfg_128['use_z_routing'], z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=512, temperature=self.temperature,
                routing_clip=cfg_128.get('routing_clip'), residual_routing=cfg_128.get('residual_routing', False),
                num_classes=self.num_classes, class_embed_dim=128,
            )
            for _ in range(cfg_128['depth'])
        ])
        self.film_128x128 = FiLM(ch128, self.num_classes)

        self.up_128_to_256    = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch128, ch256, 3, padding=1), nn.GELU())
        self.style_refine_256 = StyleGAN2Block(ch256, self.num_classes)

        refine_layers, in_ch = [], ch256
        for out_ch in cfg_out.get('refine_channels', [128]):
            refine_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU()]
            in_ch = out_ch
        self.refine     = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()
        final_in        = cfg_out.get('refine_channels', [ch256])[-1]
        self.final_conv = nn.Conv2d(final_in, cfg_out['final_ch'], 3, padding=1)
        self.to_rgb     = nn.Conv2d(cfg_out['final_ch'], 3, 1)

        self.directions_w   = []
        self.dlr_gamma_abs  = []
        self.dlr_beta_abs   = []
        self.routing_alphas = []

    def _up(self, in_ch, out_ch):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GELU())

    def _run_stage(self, h, blocks, z_dir, y, rot_k):
        h_in = h
        for blk in blocks:
            h = blk(h, z_dir=z_dir, y=y, rot_k=rot_k)
            if hasattr(blk, 'last_direction_weights') and blk.last_direction_weights is not None:
                self.directions_w.append(blk.last_direction_weights)
            if hasattr(blk, 'last_routing_alpha'):
                self.routing_alphas.append(blk.last_routing_alpha)
            if hasattr(blk, 'last_gamma_abs_mean') and blk.last_gamma_abs_mean is not None:
                self.dlr_gamma_abs.append(blk.last_gamma_abs_mean)
                self.dlr_beta_abs.append(blk.last_beta_abs_mean)
        if self.use_vit_residual:
            h = h + self.spatial_residual_weight * h_in
        return h

    def forward(self, z, y):
        B = z.size(0)
        z_tok, z_dir = torch.split(z, [self.z_tok_dim, self.z_dir_dim], dim=1)
        self.directions_w = []; self.dlr_gamma_abs = []; self.dlr_beta_abs = []; self.routing_alphas = []

        rot_k = None
        if self.training:
            mode  = self.rotation_modes[torch.randint(len(self.rotation_modes), (1,)).item()]
            rot_k = self.rotation_mode_to_k.get(mode)

        t = self.z_to_tokens(z_tok).view(B, 64, self.ch_8x8)
        t = self.pos_enc64(t)
        t_in = t
        for blk in self.mamba_tokens:
            t = blk(t)
        if self.use_vit_residual:
            t = t + self.token_residual_weight * t_in
        t = t + self.class_embed(y).unsqueeze(1)
        h = t.transpose(1, 2).contiguous().view(B, self.ch_8x8, 8, 8)

        h = self._run_stage(h, self.mamba_8x8, z_dir, y, rot_k)
        h_8 = h
        if self.film_enabled.get('8x8'):
            h = self.film_8x8(h, y)

        h = self.up_8_to_16(h)
        h = self._run_stage(h, self.mamba_16x16, z_dir, y, rot_k)
        h_16 = h
        if self.film_enabled.get('16x16'):
            h = self.film_16x16(h, y)

        h = self.up_16_to_32(h)
        if self.use_pixel_shuffle_32:
            h = self.pixel_unshuffle_32(h)
        h = self._run_stage(h, self.mamba_32x32, z_dir, y, rot_k)
        if self.use_pixel_shuffle_32:
            h = self.pixel_shuffle_32(h)
        if self.film_enabled.get('32x32'):
            h = self.film_32x32(h, y)

        h = self.up_32_to_64(h)
        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            h = h + self.skip_8_to_64 * self.skip_proj_8_to_64(h_8)
        if self.use_pixel_shuffle_64:
            h = self.pixel_unshuffle_64(h)
        h = self._run_stage(h, self.mamba_64x64, z_dir, y, rot_k)
        if self.use_pixel_shuffle_64:
            h = self.pixel_shuffle_64(h)
        if self.film_enabled.get('64x64'):
            h = self.film_64x64(h, y)

        h = self.up_64_to_128(h)
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            h = h + self.skip_16_to_128 * self.skip_proj_16_to_128(h_16)
        h = self._run_stage(h, self.mamba_128x128, z_dir, y, rot_k)
        if self.film_enabled.get('128x128'):
            h = self.film_128x128(h, y)

        h = self.up_128_to_256(h)
        h = self.style_refine_256(h, y)
        h = self.refine(h)
        h = F.gelu(self.final_conv(h))
        return torch.tanh(self.to_rgb(h))

    def enable_weight_logging(self):
        pass
