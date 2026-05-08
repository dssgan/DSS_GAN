import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        gamma = self.gamma(y).view(-1, self.num_features, 1, 1)
        beta  = self.beta(y).view(-1, self.num_features, 1, 1)
        return gamma * x + beta


class ConditionalBN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn    = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])
        self.num_features = num_features

    def forward(self, x, y):
        out = self.bn(x)
        gamma_beta = self.embed(y)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta  = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos


class RealVisionMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim   = dim
        self.norm  = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self._init_mamba_weights()

    def _init_mamba_weights(self):
        for name, param in self.mamba.named_parameters():
            if 'dt_proj' in name or 'x_proj' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0, std=0.02)
            elif 'conv1d' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)
            elif 'in_proj' in name or 'out_proj' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)
                elif param.dim() == 1:
                    nn.init.zeros_(param)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class DirectionalWeighting(nn.Module):
    def __init__(self, num_directions, z_routing_dim=None, temperature=2.0):
        super().__init__()
        self.base_weights  = nn.Parameter(torch.ones(num_directions))
        self.temperature   = temperature
        self.use_z_modulation = z_routing_dim is not None
        if self.use_z_modulation:
            self.z_to_weights = nn.Linear(z_routing_dim, num_directions)
            nn.init.zeros_(self.z_to_weights.weight)
            nn.init.zeros_(self.z_to_weights.bias)

    def forward(self, z_dir=None, class_routing_bias=None):
        weights = self.base_weights
        if self.use_z_modulation and z_dir is not None:
            z_mod   = self.z_to_weights(z_dir)
            weights = weights.unsqueeze(0) + z_mod          # (B, num_dir)
        else:
            weights = weights.unsqueeze(0)                  # (1, num_dir)
        if class_routing_bias is not None:
            weights = weights + class_routing_bias          # hybrid: additive, scaled by alpha
        return F.softmax(weights / self.temperature, dim=-1)


class LightweightAttention(nn.Module):
    def __init__(self, ch, spatial_size):
        super().__init__()
        self.ch          = ch
        self.spatial_size = spatial_size
        self.norm = nn.LayerNorm(ch)
        self.qkv  = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        qkv    = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        scale  = C ** 0.5
        attn   = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn   = attn.softmax(dim=-1)
        out    = torch.matmul(attn, v)
        out    = self.proj(out)
        return out.transpose(1, 2).contiguous().view(B, C, H, W)


class BlurUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=[1, 3, 3, 1], groups=1):
        super().__init__()
        self.groups   = groups
        k = torch.tensor(kernel, dtype=torch.float32)
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        self.register_buffer('blur_kernel', k)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if groups > 1:
            assert out_ch % groups == 0 and in_ch % groups == 0
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        self.act  = nn.GELU()

    def forward(self, x):
        x  = self.upsample(x)
        B, C, H, W = x.shape
        kernel = self.blur_kernel[None, None, :, :].repeat(C, 1, 1, 1)
        # Use F.pad explicitly to guarantee same spatial size regardless of kernel size
        k = self.blur_kernel.size(0)
        pad_total = k - 1
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        x = F.pad(x, (pad_l, pad_r, pad_l, pad_r), mode='reflect')
        x = F.conv2d(x, kernel, padding=0, groups=C)
        return self.act(self.conv(x))



class RealVisionMamba2x2OnMap(nn.Module):
    def __init__(
        self,
        ch,
        d_state=16,
        d_conv=4,
        expand=2,
        spatial_size=8,
        scan_directions=None,
        use_z_routing=False,
        z_routing_dim=None,
        z_mlp_hidden=None,
        random_corner=False,
        temperature=2.0,
        routing_clip=None,
        residual_routing=False,
        use_dense=False,
        dense_compression=0.5,
        num_classes=None,
        class_embed_dim=128,
    ):
        super().__init__()
        self.ch              = ch
        self.spatial_size    = spatial_size
        self.random_corner   = random_corner
        self.routing_clip    = routing_clip
        self.residual_routing = residual_routing
        self.use_dense       = use_dense
        self.dense_compression = dense_compression
        self.num_classes     = num_classes
        self.class_embed_dim = class_embed_dim

        if scan_directions is None:
            scan_directions = ['row_fwd', 'row_bwd', 'col_fwd', 'col_bwd']
        self.scan_directions  = scan_directions
        self.num_directions   = len(scan_directions)

        self.use_z_routing  = use_z_routing
        self.z_routing_dim  = z_routing_dim

        if self.use_z_routing:
            assert z_routing_dim is not None
            assert z_routing_dim % self.num_directions == 0
            self.z_chunk_dim = z_routing_dim // self.num_directions
            if z_mlp_hidden is None:
                z_mlp_hidden = self.z_chunk_dim * 2

            if num_classes is not None:
                self.class_embeds = nn.ModuleDict()
                for direction in self.scan_directions:
                    self.class_embeds[direction] = nn.Embedding(num_classes, class_embed_dim)
                    nn.init.normal_(self.class_embeds[direction].weight, 0, 0.25)

                # Hybrid DLR: class embedding → routing logits bias
                # Small init → routing starts dominated by z_dir, alpha learned
                self.class_routing_proj = nn.Linear(class_embed_dim, self.num_directions)
                nn.init.normal_(self.class_routing_proj.weight, std=0.001)
                nn.init.zeros_(self.class_routing_proj.bias)
                # Per-direction learnable scale; starts near zero → z_dir dominates
                self.routing_alpha = nn.Parameter(torch.full((self.num_directions,), 0.01))

            input_dim = self.z_chunk_dim + (class_embed_dim if num_classes else 0)
            self.z_projections = nn.ModuleDict()
            for direction in self.scan_directions:
                self.z_projections[direction] = nn.Sequential(
                    nn.Linear(input_dim, z_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(z_mlp_hidden, 2 * ch)
                )

        self.mamba_blocks = nn.ModuleDict()
        for direction in self.scan_directions:
            self.mamba_blocks[direction] = RealVisionMambaBlock(ch, d_state, d_conv, expand)

        if self.use_dense:
            self.dense_compress = nn.Conv2d(ch * 2, ch, 1)

        self.direction_weighting = DirectionalWeighting(
            num_directions=self.num_directions,
            z_routing_dim=z_routing_dim if use_z_routing else None,
            temperature=temperature
        )

        if 'diag_left' in self.scan_directions or 'diag_right' in self.scan_directions:
            H = W = self.spatial_size
            diag_left_idx  = []
            diag_right_idx = []
            for k in range(H + W - 1):
                i_start = min(k, H - 1)
                i_end   = max(0, k - (W - 1))
                for i in range(i_start, i_end - 1, -1):
                    j = k - i
                    diag_left_idx.append(i * W + j)
                for i in range(i_end, i_start + 1):
                    j = k - i
                    diag_right_idx.append(i * W + j)
            self.register_buffer('diag_left_indices',  torch.tensor(diag_left_idx,  dtype=torch.long))
            self.register_buffer('diag_right_indices', torch.tensor(diag_right_idx, dtype=torch.long))

        self.last_direction_weights = None
        self.last_gamma_abs_mean    = None
        self.last_beta_abs_mean     = None

    def apply_rotation(self, x, k):
        if k is None or k == 0:
            return x
        return torch.rot90(x, k=k, dims=[2, 3])

    def scan_sequence(self, h, direction):
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
            flat = h.view(B, C, H * W).transpose(1, 2)
            return flat[:, self.diag_left_indices, :]
        elif direction == 'diag_right':
            flat = h.view(B, C, H * W).transpose(1, 2)
            return flat[:, self.diag_right_indices, :]
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def unscan_sequence(self, seq, direction, H, W):
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
            flat[:, self.diag_left_indices, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        elif direction == 'diag_right':
            flat = torch.zeros(B, H * W, C, device=seq.device, dtype=seq.dtype)
            flat[:, self.diag_right_indices, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def forward(self, h, z_dir=None, y=None, rot_k=None, prev_outputs=None):
        B, C, H, W = h.shape
        assert C == self.ch

        if self.use_dense and prev_outputs is not None and len(prev_outputs) > 0:
            h = self.dense_compress(torch.cat([prev_outputs[-1], h], dim=1))

        h_rot = self.apply_rotation(h, rot_k)

        if self.use_z_routing and z_dir is not None:
            # Hybrid DLR: class provides a per-direction routing bias scaled by learned alpha
            class_routing_bias = None
            if self.num_classes is not None and y is not None and hasattr(self, 'class_routing_proj'):
                # Use mean class embedding across directions as routing signal
                # shape: (B, class_embed_dim) → (B, num_directions)
                cls_emb_mean = sum(
                    self.class_embeds[d](y) for d in self.scan_directions
                ) / self.num_directions                             # (B, class_embed_dim)
                cls_bias = self.class_routing_proj(cls_emb_mean)   # (B, num_directions)
                alpha = torch.abs(self.routing_alpha).unsqueeze(0) # (1, num_directions)
                class_routing_bias = alpha * cls_bias              # (B, num_directions)
            dir_weights = self.direction_weighting(z_dir, class_routing_bias=class_routing_bias)
        else:
            dir_weights = self.direction_weighting()

        if dir_weights.dim() == 1:
            self.last_direction_weights = dir_weights.detach().cpu()
        else:
            self.last_direction_weights = dir_weights.mean(0).detach().cpu()

        # Log routing_alpha for monitoring
        if hasattr(self, 'routing_alpha'):
            self.last_routing_alpha = self.routing_alpha.detach().cpu()

        outputs = []
        for idx, direction in enumerate(self.scan_directions):
            seq = self.scan_sequence(h_rot, direction)

            if self.use_z_routing and z_dir is not None:
                z_chunk = z_dir[:, idx * self.z_chunk_dim:(idx + 1) * self.z_chunk_dim]

                if self.num_classes is not None and y is not None:
                    class_emb = self.class_embeds[direction](y)
                    dlr_input = torch.cat([z_chunk, class_emb], dim=1)
                else:
                    dlr_input = z_chunk

                gamma_beta = self.z_projections[direction](dlr_input)
                gamma, beta = gamma_beta.chunk(2, dim=1)

                if self.residual_routing:
                    if self.routing_clip is not None:
                        gamma = torch.tanh(gamma) * self.routing_clip + 1.0
                        beta  = torch.tanh(beta)  * self.routing_clip
                    else:
                        gamma = gamma + 1.0
                else:
                    if self.routing_clip is not None:
                        gamma = torch.clamp(gamma, -self.routing_clip, self.routing_clip)
                        beta  = torch.clamp(beta,  -self.routing_clip, self.routing_clip)

                gamma = gamma.unsqueeze(1)
                beta  = beta.unsqueeze(1)
                seq   = gamma * seq + beta

                if idx == 0:
                    self.last_gamma_abs_mean = gamma.abs().mean().item()
                    self.last_beta_abs_mean  = beta.abs().mean().item()

            seq = self.mamba_blocks[direction](seq)
            out = self.unscan_sequence(seq, direction, H, W)
            outputs.append(out)

        if dir_weights.dim() == 1:
            result = sum(w * out for w, out in zip(dir_weights, outputs))
        else:
            result = torch.zeros_like(outputs[0])
            for idx, out in enumerate(outputs):
                w      = dir_weights[:, idx:idx+1, None, None]
                result = result + w * out

        result = self.apply_rotation(result, -rot_k if rot_k is not None else None)
        return result



class NoiseInjection(nn.Module):
    """Cascaded modulation: noise → class → feature."""
    def __init__(self, channels, num_classes, strength=0.1):
        super().__init__()
        self.strength    = strength
        self.noise_gamma = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.noise_beta  = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.class_to_gamma_beta = nn.Sequential(
            nn.Embedding(num_classes, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        nn.init.zeros_(self.class_to_gamma_beta[2].weight)
        nn.init.constant_(self.class_to_gamma_beta[2].bias, 0.0)
        self.class_to_gamma_beta[2].bias.data[0] = 1.0

    def forward(self, x, y):
        if self.training:
            B, C, H, W = x.shape
            noise   = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
            noise_1 = self.noise_gamma * noise + self.noise_beta
            gamma_beta = self.class_to_gamma_beta[2](
                F.gelu(self.class_to_gamma_beta[0](y))
            )
            gamma = gamma_beta[:, 0:1].view(-1, 1, 1, 1)
            beta  = gamma_beta[:, 1:2].view(-1, 1, 1, 1)
            noise_2 = gamma * noise_1 + beta
            return x + self.strength * noise_2
        return x


class StyleGAN2Block(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.channels = channels
        self.style    = nn.Embedding(num_classes, channels)
        nn.init.zeros_(self.style.weight)  # zeros → style+1=1 at init, no signal collapse
        self.weight = nn.Parameter(
            torch.randn(channels, channels, 3, 3) * (2 / (channels * 3 * 3)) ** 0.5
        )
        self.bias           = nn.Parameter(torch.zeros(channels))
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        B, C_in, H, W = x.shape
        C_out = self.channels

        style  = self.style(y).view(B, 1, C_in, 1, 1)
        weight = self.weight.unsqueeze(0) * (style + 1)
        demod  = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod

        x      = x.reshape(1, B * C_in, H, W)
        weight = weight.reshape(B * C_out, C_in, 3, 3)
        out    = F.conv2d(x, weight, padding=1, groups=B)
        out    = out.reshape(B, C_out, H, W) + self.bias.view(1, C_out, 1, 1)

        if self.training and self.noise_strength.item() > 0:
            out = out + self.noise_strength * torch.randn(B, 1, H, W, device=out.device, dtype=out.dtype)

        return F.leaky_relu(out, 0.2)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ── global config ──────────────────────────────────────
        self.resolution          = config.get('resolution', 512)  # NEW: 128, 256, or 512
        self.z_dim               = config['z_dim']
        self.num_classes         = config['num_classes']
        self.use_vit_residual    = config.get('use_vit_residual', False)
        self.token_residual_weight   = config.get('token_residual_weight', 0.3)
        self.spatial_residual_weight = config.get('spatial_residual_weight', 0.3)
        self.temperature         = config.get('temperature', 2.0)
        self.noise_strength      = config.get('noise_strength', 0.01)

        self.use_multiscale_skips = config.get('use_multiscale_skips', False)
        self.skip_8_to_64         = config.get('skip_8_to_64', 0.0)
        self.skip_16_to_128       = config.get('skip_16_to_128', 0.0)

        self.use_blur_upsample   = config.get('use_blur_upsample', False)
        self.blur_kernel         = config.get('blur_kernel', [1, 3, 3, 1])
        self.use_dense_connections = config.get('use_dense_connections', False)
        self.dense_compression   = config.get('dense_compression', 0.5)
        self.use_lightweight_attention = config.get('use_lightweight_attention', False)
        self.attention_stages    = config.get('attention_stages', [])
        self.attention_residual_weight = config.get('attention_residual_weight', 0.1)
        self.use_grouped_conv    = config.get('use_grouped_conv', False)
        self.conv_groups         = config.get('conv_groups', 1)

        self.film_enabled = config.get('film_enabled', {
            '8x8': False, '16x16': False, '32x32': False, '64x64': False
        })

        self.z_tok_dim  = config.get('z_tok_dim', self.z_dim // 2)
        self.z_dir_dim  = self.z_dim - self.z_tok_dim

        self.cfg_tokens  = config['mamba_tokens']
        self.cfg_8x8     = config['mamba_8x8']
        self.cfg_16x16   = config['mamba_16x16']
        self.cfg_32x32   = config['mamba_32x32']
        self.cfg_64x64   = config['mamba_64x64']
        
        if self.resolution >= 256:
            self.cfg_128x128 = config['mamba_128x128']
        if self.resolution >= 512:
            self.cfg_256x256 = config['mamba_256x256']
            
        self.cfg_out     = config['output']

        self.ch_8x8    = config['ch_8x8']
        self.ch_16x16  = config['ch_16x16']
        self.ch_32x32  = config['ch_32x32']
        self.ch_64x64  = config['ch_64x64']
        
        self.ch_128x128 = config['ch_128x128']
        if self.resolution >= 256:
            self.ch_256x256 = config.get('ch_256x256', 196)
        if self.resolution >= 512:
            self.ch_512x512 = config.get('ch_512x512', 48)

        self.use_pixel_shuffle_32 = config.get('use_pixel_shuffle_32', False)
        self.use_pixel_shuffle_64 = config.get('use_pixel_shuffle_64', False)
        self.z_mlp_hidden = config.get('z_mlp_hidden', {})

        self.rotation_modes = config.get('rotation_modes', ['none', 'rot180'])
        self.rotation_mode_to_k = {'none': None, 'rot90': 1, 'rot180': 2, 'rot270': 3}

        self.z_to_tokens  = nn.Linear(self.z_tok_dim, 64 * self.ch_8x8)
        self.pos_enc64    = LearnedPositionalEncoding(64, self.ch_8x8)
        self.mamba_tokens = nn.ModuleList([
            RealVisionMambaBlock(
                dim=self.ch_8x8,
                d_state=self.cfg_tokens['d_state'],
                d_conv=self.cfg_tokens['d_conv'],
                expand=self.cfg_tokens['expand']
            )
            for _ in range(self.cfg_tokens['depth'])
        ])
        self.class_embed = nn.Embedding(self.num_classes, self.ch_8x8)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        self.mamba_8x8 = nn.ModuleList([
            RealVisionMamba2x2OnMap(
                ch=self.ch_8x8,
                d_state=self.cfg_8x8['d_state'],
                d_conv=self.cfg_8x8['d_conv'],
                expand=self.cfg_8x8['expand'],
                spatial_size=8,
                scan_directions=self.cfg_8x8['scan_directions'],
                use_z_routing=self.cfg_8x8['use_z_routing'],
                z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=self.z_mlp_hidden.get('8x8', 256),
                random_corner=self.cfg_8x8['random_corner'],
                temperature=self.temperature,
                routing_clip=self.cfg_8x8.get('routing_clip'),
                residual_routing=self.cfg_8x8.get('residual_routing', False),
                use_dense=self.cfg_8x8.get('use_dense', False),
                dense_compression=self.dense_compression,
                num_classes=self.num_classes,
                class_embed_dim=128,
            )
            for _ in range(self.cfg_8x8['depth'])
        ])
        self.film_8x8 = FiLM(self.ch_8x8, self.num_classes)
        if self.cfg_8x8.get('noise', False):
            self.noise_8x8 = NoiseInjection(self.ch_8x8, self.num_classes, self.noise_strength)

        self.up_8_to_16 = self._make_upsample(self.ch_8x8, self.ch_16x16)
        self.mamba_16x16 = nn.ModuleList([
            RealVisionMamba2x2OnMap(
                ch=self.ch_16x16,
                d_state=self.cfg_16x16['d_state'],
                d_conv=self.cfg_16x16['d_conv'],
                expand=self.cfg_16x16['expand'],
                spatial_size=16,
                scan_directions=self.cfg_16x16['scan_directions'],
                use_z_routing=self.cfg_16x16['use_z_routing'],
                z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=self.z_mlp_hidden.get('16x16', 384),
                random_corner=self.cfg_16x16['random_corner'],
                temperature=self.temperature,
                routing_clip=self.cfg_16x16.get('routing_clip'),
                residual_routing=self.cfg_16x16.get('residual_routing', False),
                use_dense=self.cfg_16x16.get('use_dense', False),
                dense_compression=self.dense_compression,
                num_classes=self.num_classes,
                class_embed_dim=128,
            )
            for _ in range(self.cfg_16x16['depth'])
        ])
        self.film_16x16 = FiLM(self.ch_16x16, self.num_classes)
        if self.cfg_16x16.get('noise', False):
            self.noise_16x16 = NoiseInjection(self.ch_16x16, self.num_classes, self.noise_strength)

        self.up_16_to_32 = self._make_upsample(self.ch_16x16, self.ch_32x32)

        if self.use_pixel_shuffle_32:
            self.pixel_unshuffle_32  = nn.PixelUnshuffle(2)
            self.pixel_shuffle_32    = nn.PixelShuffle(2)
            self.ch_32x32_shuffled   = self.ch_32x32 * 4
            mamba_spatial_32 = 16
        else:
            self.ch_32x32_shuffled = self.ch_32x32
            mamba_spatial_32 = 32

        self.mamba_32x32 = nn.ModuleList([
            RealVisionMamba2x2OnMap(
                ch=self.ch_32x32_shuffled,
                d_state=self.cfg_32x32['d_state'],
                d_conv=self.cfg_32x32['d_conv'],
                expand=self.cfg_32x32['expand'],
                spatial_size=mamba_spatial_32,
                scan_directions=self.cfg_32x32['scan_directions'],
                use_z_routing=self.cfg_32x32['use_z_routing'],
                z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=self.z_mlp_hidden.get('32x32', 512),
                random_corner=self.cfg_32x32['random_corner'],
                temperature=self.temperature,
                routing_clip=self.cfg_32x32.get('routing_clip'),
                residual_routing=self.cfg_32x32.get('residual_routing', False),
                use_dense=self.cfg_32x32.get('use_dense', False),
                dense_compression=self.dense_compression,
                num_classes=self.num_classes,
                class_embed_dim=128,
            )
            for _ in range(self.cfg_32x32['depth'])
        ])
        if self.use_lightweight_attention and '32x32' in self.attention_stages:
            self.attn_32x32 = LightweightAttention(self.ch_32x32, 32)
        self.film_32x32 = FiLM(self.ch_32x32, self.num_classes)
        if self.cfg_32x32.get('noise', False):
            self.noise_32x32 = NoiseInjection(self.ch_32x32, self.num_classes, self.noise_strength)

        self.up_32_to_64 = self._make_upsample(self.ch_32x32, self.ch_64x64)

        if self.use_pixel_shuffle_64:
            self.pixel_unshuffle_64  = nn.PixelUnshuffle(2)
            self.pixel_shuffle_64    = nn.PixelShuffle(2)
            self.ch_64x64_shuffled   = self.ch_64x64 * 4
            mamba_spatial_64 = 32
        else:
            self.ch_64x64_shuffled = self.ch_64x64
            mamba_spatial_64 = 64

        self.mamba_64x64 = nn.ModuleList([
            RealVisionMamba2x2OnMap(
                ch=self.ch_64x64_shuffled,
                d_state=self.cfg_64x64['d_state'],
                d_conv=self.cfg_64x64['d_conv'],
                expand=self.cfg_64x64['expand'],
                spatial_size=mamba_spatial_64,
                scan_directions=self.cfg_64x64['scan_directions'],
                use_z_routing=self.cfg_64x64['use_z_routing'],
                z_routing_dim=self.z_dir_dim,
                z_mlp_hidden=self.z_mlp_hidden.get('64x64', 640),
                random_corner=self.cfg_64x64['random_corner'],
                temperature=self.temperature,
                routing_clip=self.cfg_64x64.get('routing_clip'),
                residual_routing=self.cfg_64x64.get('residual_routing', False),
                use_dense=self.cfg_64x64.get('use_dense', False),
                dense_compression=self.dense_compression,
                num_classes=self.num_classes,
                class_embed_dim=128,
            )
            for _ in range(self.cfg_64x64['depth'])
        ])
        if self.use_lightweight_attention and '64x64' in self.attention_stages:
            self.attn_64x64 = LightweightAttention(self.ch_64x64, 64)
        self.film_64x64 = FiLM(self.ch_64x64, self.num_classes)
        if self.cfg_64x64.get('noise', False):
            self.noise_64x64 = NoiseInjection(self.ch_64x64, self.num_classes, self.noise_strength)

        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            self.skip_proj_8_to_64 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='nearest'),
                nn.Conv2d(self.ch_8x8, self.ch_64x64, 1),
            )
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            self.skip_proj_16_to_128 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='nearest'),
                nn.Conv2d(self.ch_16x16, self.ch_128x128, 1),
            )

        self.up_64_to_128 = self._make_upsample(self.ch_64x64, self.ch_128x128)
        
        if self.resolution == 128:
            self.style_refine_128 = StyleGAN2Block(self.ch_128x128, self.num_classes)
        else:
            self.mamba_128x128 = nn.ModuleList([
                RealVisionMamba2x2OnMap(
                    ch=self.ch_128x128,
                    d_state=self.cfg_128x128['d_state'],
                    d_conv=self.cfg_128x128['d_conv'],
                    expand=self.cfg_128x128['expand'],
                    spatial_size=128,
                    scan_directions=self.cfg_128x128['scan_directions'],
                    use_z_routing=self.cfg_128x128['use_z_routing'],
                    z_routing_dim=self.z_dir_dim,
                    z_mlp_hidden=512,
                    random_corner=self.cfg_128x128['random_corner'],
                    temperature=self.temperature,
                    routing_clip=self.cfg_128x128.get('routing_clip'),
                    residual_routing=self.cfg_128x128.get('residual_routing', False),
                    use_dense=False,
                    dense_compression=self.dense_compression,
                    num_classes=self.num_classes,
                    class_embed_dim=128,
                )
                for _ in range(self.cfg_128x128['depth'])
            ])
            self.film_128x128 = FiLM(self.ch_128x128, self.num_classes)

        if self.resolution == 256:
            self.up_128_to_256 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.ch_128x128, self.ch_256x256, 3, padding=1),
                nn.GELU()
            )
            self.style_refine_256 = StyleGAN2Block(self.ch_256x256, self.num_classes)
            
        elif self.resolution == 512:
            self.up_128_to_256 = self._make_upsample(self.ch_128x128, self.ch_256x256)
            
            self.mamba_256x256 = nn.ModuleList([
                RealVisionMamba2x2OnMap(
                    ch=self.ch_256x256,
                    d_state=self.cfg_256x256['d_state'],
                    d_conv=self.cfg_256x256['d_conv'],
                    expand=self.cfg_256x256['expand'],
                    spatial_size=256,
                    scan_directions=self.cfg_256x256['scan_directions'],
                    use_z_routing=self.cfg_256x256['use_z_routing'],
                    z_routing_dim=self.z_dir_dim,
                    z_mlp_hidden=512,
                    random_corner=self.cfg_256x256['random_corner'],
                    temperature=self.temperature,
                    routing_clip=self.cfg_256x256.get('routing_clip'),
                    residual_routing=self.cfg_256x256.get('residual_routing', False),
                    use_dense=False,
                    dense_compression=self.dense_compression,
                    num_classes=self.num_classes,
                    class_embed_dim=128,
                )
                for _ in range(self.cfg_256x256['depth'])
            ])
            self.film_256x256 = FiLM(self.ch_256x256, self.num_classes)

        if self.resolution == 512:
            self.up_256_to_512 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.ch_256x256, self.ch_512x512, 3, padding=1),
                nn.GELU()
            )
            self.style_refine_512 = StyleGAN2Block(self.ch_512x512, self.num_classes)


        if self.resolution == 128:
            final_ch_in = self.ch_128x128
        elif self.resolution == 256:
            final_ch_in = self.ch_256x256
        else:  # 512
            final_ch_in = self.ch_512x512
            
        refine_channels = self.cfg_out.get('refine_channels', [64])
        refine_layers = []
        in_ch = final_ch_in
        for out_ch in refine_channels:
            refine_layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            refine_layers.append(nn.GELU())
            in_ch = out_ch
        self.refine    = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()
        final_in_ch    = refine_channels[-1] if refine_channels else final_ch_in
        self.final_conv = nn.Conv2d(final_in_ch, self.cfg_out['final_ch'], 3, padding=1)
        self.to_rgb     = nn.Conv2d(self.cfg_out['final_ch'], 3, 1)

        # Logging
        self._log_weights_this_epoch = False
        self.directions_w   = []
        self.dlr_gamma_abs  = []
        self.dlr_beta_abs   = []
        self.routing_alphas = []  


    def _make_upsample(self, in_ch, out_ch):
        if self.use_blur_upsample:
            groups = self.conv_groups if self.use_grouped_conv else 1
            return BlurUpsample(in_ch, out_ch, self.blur_kernel, groups)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU()
        )

    def _run_mamba_stage(self, h, blocks, z_dir, y, rot_k):
        prev = []
        if self.use_vit_residual:
            h_before = h
            for blk in blocks:
                h = blk(h, z_dir=z_dir, y=y, rot_k=rot_k, prev_outputs=prev)
                prev.append(h.clone())
                if hasattr(blk, 'last_direction_weights'):
                    self.directions_w.append(blk.last_direction_weights)
                if hasattr(blk, 'last_routing_alpha'):
                    self.routing_alphas.append(blk.last_routing_alpha)
                if hasattr(blk, 'last_gamma_abs_mean') and blk.last_gamma_abs_mean is not None:
                    self.dlr_gamma_abs.append(blk.last_gamma_abs_mean)
                    self.dlr_beta_abs.append(blk.last_beta_abs_mean)
            h = h + self.spatial_residual_weight * h_before
        else:
            for blk in blocks:
                h = blk(h, z_dir=z_dir, y=y, rot_k=rot_k, prev_outputs=prev)
                prev.append(h.clone())
                if hasattr(blk, 'last_direction_weights'):
                    self.directions_w.append(blk.last_direction_weights)
                if hasattr(blk, 'last_routing_alpha'):
                    self.routing_alphas.append(blk.last_routing_alpha)
                if hasattr(blk, 'last_gamma_abs_mean') and blk.last_gamma_abs_mean is not None:
                    self.dlr_gamma_abs.append(blk.last_gamma_abs_mean)
                    self.dlr_beta_abs.append(blk.last_beta_abs_mean)
        return h


    def forward(self, z, y):
        B = z.size(0)
        z_tok, z_dir = torch.split(z, [self.z_tok_dim, self.z_dir_dim], dim=1)

        self.directions_w  = []
        self.dlr_gamma_abs = []
        self.dlr_beta_abs  = []
        self.routing_alphas = []

        if self.training:
            idx      = torch.randint(0, len(self.rotation_modes), (1,), device=z.device).item()
            rot_mode = self.rotation_modes[idx]
            rot_k    = self.rotation_mode_to_k.get(rot_mode, None)
        else:
            rot_k = None

        # Stage 0: tokens → 8×8
        t = self.z_to_tokens(z_tok).view(B, 64, self.ch_8x8)
        t = self.pos_enc64(t)
        if self.use_vit_residual:
            t_before = t
            for blk in self.mamba_tokens:
                t = blk(t)
            t = t + self.token_residual_weight * t_before
        else:
            for blk in self.mamba_tokens:
                t = blk(t)
        t = t + self.class_embed(y).unsqueeze(1)
        h = t.transpose(1, 2).contiguous().view(B, self.ch_8x8, 8, 8)

        h = self._run_mamba_stage(h, self.mamba_8x8, z_dir, y, rot_k)
        h_8x8 = h
        if self.film_enabled.get('8x8', False):
            h = self.film_8x8(h, y)
        if hasattr(self, 'noise_8x8'):
            h = self.noise_8x8(h, y)

        h = self.up_8_to_16(h)
        h = self._run_mamba_stage(h, self.mamba_16x16, z_dir, y, rot_k)
        h_16x16 = h
        if self.film_enabled.get('16x16', False):
            h = self.film_16x16(h, y)
        if hasattr(self, 'noise_16x16'):
            h = self.noise_16x16(h, y)

        h = self.up_16_to_32(h)
        if self.use_pixel_shuffle_32:
            h = self.pixel_unshuffle_32(h)
        h = self._run_mamba_stage(h, self.mamba_32x32, z_dir, y, rot_k)
        if self.use_pixel_shuffle_32:
            h = self.pixel_shuffle_32(h)
        if hasattr(self, 'attn_32x32'):
            h = h + self.attention_residual_weight * self.attn_32x32(h)
        if self.film_enabled.get('32x32', False):
            h = self.film_32x32(h, y)
        if hasattr(self, 'noise_32x32'):
            h = self.noise_32x32(h, y)

        h = self.up_32_to_64(h)
        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            h = h + self.skip_8_to_64 * self.skip_proj_8_to_64(h_8x8)
        if self.use_pixel_shuffle_64:
            h = self.pixel_unshuffle_64(h)
        h = self._run_mamba_stage(h, self.mamba_64x64, z_dir, y, rot_k)
        if self.use_pixel_shuffle_64:
            h = self.pixel_shuffle_64(h)
        if hasattr(self, 'attn_64x64'):
            h = h + self.attention_residual_weight * self.attn_64x64(h)
        if self.film_enabled.get('64x64', False):
            h = self.film_64x64(h, y)
        if hasattr(self, 'noise_64x64'):
            h = self.noise_64x64(h, y)

        h = self.up_64_to_128(h)
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            h = h + self.skip_16_to_128 * self.skip_proj_16_to_128(h_16x16)
        
        if self.resolution == 128:
            h = self.style_refine_128(h, y)
        else:
            h = self._run_mamba_stage(h, self.mamba_128x128, z_dir, y, rot_k)
            if self.film_enabled.get('128x128', False):
                h = self.film_128x128(h, y)

        if self.resolution == 256:
            h = self.up_128_to_256(h)
            h = self.style_refine_256(h, y)
        elif self.resolution == 512:
            h = self.up_128_to_256(h)
            h = self._run_mamba_stage(h, self.mamba_256x256, z_dir, y, rot_k)
            if self.film_enabled.get('256x256', False):
                h = self.film_256x256(h, y)

        if self.resolution == 512:
            h = self.up_256_to_512(h)
            h = self.style_refine_512(h, y)

        # toRGB
        h = self.refine(h)
        h = F.gelu(self.final_conv(h))
        x = torch.tanh(self.to_rgb(h))
        return x

    def enable_weight_logging(self):
        self._log_weights_this_epoch = True