import os
import json

CLASSES = sorted(["cat", "dog", "wild"])

IMG_SIZE = 128
RESOLUTION = IMG_SIZE
OUT_DIR = f"afhq128_class_3dir"

DATASET = "afhq"

NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 128
EPOCHS = 300
SEED = 42

# ============================================================
# FiLM CONTROL
# ============================================================
FILM_ENABLED = {
    '8x8':   False,
    '16x16': False,
    '32x32': False,
    '64x64': False,
}

if all(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.3, '16x16': 0.3, '32x32': 0.15, '64x64': 0.1}
    VARIANT_NAME = "A_FiLM_all"
elif not any(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.8, '16x16': 0.8, '32x32': 0.5, '64x64': 0.3}
    VARIANT_NAME = "C_NoFiLM_StyleGAN128"
else:
    ROUTING_CLIP = {'8x8': 0.5, '16x16': 0.5, '32x32': 0.3, '64x64': 0.2}
    VARIANT_NAME = "Custom"

print(f"\n{'='*60}")
print(f"🧪 VARIANT: {VARIANT_NAME}  |  RESOLUTION: 128x128")
print(f"   Architecture: Mamba→64 + StyleGAN2→128")
print(f"   FiLM: 8={FILM_ENABLED['8x8']}, 16={FILM_ENABLED['16x16']}, 32={FILM_ENABLED['32x32']}, 64={FILM_ENABLED['64x64']}")
print(f"   Routing clip: {ROUTING_CLIP}")
print(f"{'='*60}\n")


LATENT_BASE = 92
LATENT_DIR = 20
SCAN_DIRECTIONS = ['row_fwd','col_bwd','diag_left']

NUM_DIR = len(SCAN_DIRECTIONS)
Z_DIM = LATENT_BASE + LATENT_DIR * NUM_DIR

# ============================================================
# LEARNING RATE & SCHEDULER
# ============================================================
LR_G = 9e-5
LR_D = 3e-5
BETAS = (0.0, 0.99)

LR_SCHEDULER = {
    'type': 'None',
    'warmup_epochs': 20,
    'min_lr_g': 1e-6,
    'min_lr_d': 5e-7,
    'T_max': EPOCHS,
}

# ============================================================
# TRAINING STEPS
# ============================================================
D_STEPS = 1
G_STEPS = 1

GRADIENT_CLIP_G = 10
GRADIENT_CLIP_D = 15

# ============================================================
# REGULARIZATION
# ============================================================
R1_GAMMA = 5
R1_INTERVAL = 4

PL_WEIGHT = 0.0
PL_INTERVAL = 4
PL_START_EPOCH = 10000

# ============================================================
# EMA
# ============================================================
EMA_DECAY_1 = 0.999
EMA_DECAY_2 = 0.9995
EMA_SWITCH_IMAGES = 1_000_000

# ============================================================
# AUGMENTATION
# ============================================================
AUG_CONFIG = {
    'brightness': 0.1,
    'contrast': 0.1,
    'flip_prob': 0.5,
}

AUG_CONFIG_G = {
    'brightness': 0.1,
    'contrast': 0.1,
    'flip_prob': 0.5,
}

ROTATION_MODES = ['none', 'rot180']

# ============================================================
# GENERATOR CONFIG — 128x128
# Architecture:
#   Mamba: 8x8 → 16x16 → 32x32 → 64x64
#   StyleGAN2Block: 128x128
#   CNN refine → toRGB 128x128
# ============================================================
G_CONFIG = {
    'resolution': RESOLUTION,
    'z_dim': Z_DIM,
    'z_tok_dim': LATENT_BASE,
    'z_dir_dim': LATENT_DIR * NUM_DIR,
    'num_classes': NUM_CLASSES,
    'y_embed_dim': 64,
    'temperature': 1.0,
    'noise_strength': 0.01,
    'use_vit_residual': True,
    'token_residual_weight': 0.3,
    'spatial_residual_weight': 0.3,
    'rotation_modes': ROTATION_MODES,

    'use_class_gating': False,
    'film_enabled': {k: v for k, v in FILM_ENABLED.items() if k != '128x128'},
    'use_multiscale_skips': False,
    'skip_8_to_64': 0.0,
    'skip_16_to_128': 0.0,
    'use_blur_upsample': False,
    'blur_kernel': [1, 3, 3, 1],
    'use_dense_connections': False,
    'use_lightweight_attention': False,
    'use_grouped_conv': True,

    # Channel dims
    'ch_8x8':   148,
    'ch_16x16': 148,
    'ch_32x32': 148,
    'ch_64x64': 148,
    # 128x128: StyleGAN2Block
    'ch_128x128': 168,

    'use_pixel_shuffle_32': False,
    'use_pixel_shuffle_64': False,

    'z_mlp_hidden': {
        '8x8':   64,
        '16x16': 128,
        '32x32': 256,
        '64x64': 256,
    },

    'mamba_tokens': {
        'depth': 2,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
    },

    'mamba_8x8': {
        'depth': 2,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['8x8'],
        'residual_routing': True,
    },

    'mamba_16x16': {
        'depth': 1,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['16x16'],
        'residual_routing': True,
    },

    'mamba_32x32': {
        'depth': 1,
        'd_state': 64,
        'd_conv': 4,
        'expand': 2,
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['32x32'],
        'residual_routing': True,
    },

    'mamba_64x64': {
        'depth': 1,
        'd_state': 64,
        'd_conv': 3,
        'expand': 1.5,
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['64x64'],
        'residual_routing': True,
    },

    'output': {
        'final_ch': 64,
        'refine_channels': [128],
    },
}

# ============================================================
# DISCRIMINATOR CONFIG — 256x256
# ============================================================
D_CONFIG = {
    'resolution': RESOLUTION,
    'num_classes': NUM_CLASSES,
    'base_ch': 96,
    'channel_max': 512,
    'mbstd_group_size': 8,
}

SAVE_EVERY = 3

os.makedirs(OUT_DIR, exist_ok=True)

def save_config():
    config_dict = {
        'variant': {
            'name': VARIANT_NAME,
            'film_enabled': FILM_ENABLED,
            'routing_clip': ROUTING_CLIP,
        },
        'architecture': {
            'type': 'MAD-GAN 128x128',
            'output_resolution': '128x128',
            'generator': 'Mamba(8→16→32→64) + StyleGAN2(128)',
            'discriminator': 'StyleGAN2-ADA 128→4',
            'note': 'Mamba pipeline to 64x64, StyleGAN2Block on 128x128'
        },
        'generator': G_CONFIG,
        'discriminator': D_CONFIG,
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'seed': SEED,
            'lr_g': LR_G,
            'lr_d': LR_D,
            'betas': BETAS,
            'd_steps': D_STEPS,
            'g_steps': G_STEPS,
            'clipg': GRADIENT_CLIP_G,
            'clipd': GRADIENT_CLIP_D,
            'scheduler': LR_SCHEDULER,
        },
        'regularization': {
            'r1_gamma': R1_GAMMA,
            'r1_interval': R1_INTERVAL,
            'pl_weight': PL_WEIGHT,
            'pl_interval': PL_INTERVAL,
            'pl_start_epoch': PL_START_EPOCH,
        },
        'ema': {
            'decay': EMA_DECAY_1,
            'decay2': EMA_DECAY_2,
        },
        'saving': {
            'save_every': SAVE_EVERY,
            'out_dir': OUT_DIR,
        }
    }
    config_path = os.path.join(OUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config saved to {config_path}")

save_config()

print(f"\n{'='*60}")
print("128x128 ARCHITECTURE SUMMARY:")
print(f"{'='*60}")
print(f"G: Mamba 8→16→32→64 | StyleGAN2Block 128 | CNN refine | toRGB")
print(f"D: StyleGAN2-ADA 128→4 | mbstd | fc")
print(f"Batch: {BATCH_SIZE}")
print(f"Z_DIM: {Z_DIM}")
print(f"ch: 8/16/32/64=148 | 128=168")
print(f"{'='*60}\n")