import os
import json

CLASSES = sorted(["cat", "dog", "wild"])

FID_REAL_IMG = "AFHQ_resize_128"
IMG_SIZE = 128
RESOLUTION = IMG_SIZE

DATASET = "afhq128"

NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 96
EPOCHS = 500
SEED = 42

FID_EVERY = 10
FID_SAMPLES = 20000
FID_BATCH = 32

# ============================================================
# EVALUATION SETTINGS
# ============================================================
FID_SAMPLES_GLOBAL = 20000
FID_SAMPLES_PERCLASS = 10000

IS_SAMPLES_GLOBAL = 20000
IS_SAMPLES_PERCLASS = 5000

PR_SAMPLES_GLOBAL = 10000
PR_SAMPLES_PERCLASS = 10000

FID_BATCH = 32
FID_SAMPLES = FID_SAMPLES_GLOBAL

# ============================================================
# FiLM CONTROL
# ============================================================
FILM_ENABLED = {
    '8x8':   False,
    '16x16': False,
    '32x32': False,
    '64x64': False,
    '128x128': False,  # NEW for Mamba 128
}

if all(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.3, '16x16': 0.3, '32x32': 0.15, '64x64': 0.1, '128x128': 0.05}
    VARIANT_NAME = "A_FiLM_all"
elif not any(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.8, '16x16': 0.8, '32x32': 0.5, '64x64': 0.3, '128x128': 0.2}
    VARIANT_NAME = "C_NoFiLM_Mamba128"
else:
    ROUTING_CLIP = {'8x8': 0.5, '16x16': 0.5, '32x32': 0.3, '64x64': 0.2, '128x128': 0.2}
    VARIANT_NAME = "Custom"

print(f"\n{'='*60}")
print(f"🧪 VARIANT: {VARIANT_NAME}  |  RESOLUTION: 256x256")
print(f"   Architecture: Mamba→128 + StyleGAN2→256")
print(f"   FiLM: 8={FILM_ENABLED['8x8']}, 16={FILM_ENABLED['16x16']}, 32={FILM_ENABLED['32x32']}, 64={FILM_ENABLED['64x64']}, 128={FILM_ENABLED['128x128']}")
print(f"   Routing clip: {ROUTING_CLIP}")
print(f"{'='*60}\n")

OUT_DIR = f"AFHQ_256_class_routing"

LATENT_BASE = 88
LATENT_DIR = 28
SCAN_DIRECTIONS = ['row_fwd', 'col_fwd','diag_left']

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
# GENERATOR CONFIG — 256x256 with Mamba on 128x128
# Architecture:
#   Mamba: 8x8 → 16x16 → 32x32 → 64x64 → 128x128 [NEW]
#   StyleGAN2Block: 256x256
#   toRGB: 256x256
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
    'film_enabled': FILM_ENABLED,
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
    # 128x128: Mamba [NEW in this variant]
    'ch_128x128': 168,
    # 256x256: StyleGAN2Block
    'ch_256x256': 196,

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
        'expand': 1.5,
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
        'expand': 1.5,
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
        'expand': 1.5,
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
        'expand': 1,
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['64x64'],
        'residual_routing': True,
    },

    # NEW: Mamba on 128x128 (16384 tokens!)
    'mamba_128x128': {
        'depth': 1,          # Only 1 block — memory intensive
        'd_state': 48,
        'd_conv':3,
        'expand': 1,         # Reduced from 2 — lower memory footprint
        'scan_directions': SCAN_DIRECTIONS,
        'use_z_routing': True,
        'random_corner': False,
        'noise': False,
        'routing_clip': ROUTING_CLIP['128x128'],
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
    'base_ch': 64,
    'channel_max': 384,
    'mbstd_group_size': 8,
}

SAVE_EVERY = 3
FID_CACHE = "FID/afhq256_real.npz"

os.makedirs(OUT_DIR, exist_ok=True)

def save_config():
    config_dict = {
        'variant': {
            'name': VARIANT_NAME,
            'film_enabled': FILM_ENABLED,
            'routing_clip': ROUTING_CLIP,
        },
        'architecture': {
            'type': 'MAD-GAN 256x256 Mamba128 Variant',
            'output_resolution': '256x256',
            'generator': 'Mamba(8→128) + StyleGAN2(256)',
            'discriminator': 'StyleGAN2-ADA + extra block 256→128',
            'note': 'Mamba extends to 128x128 (16384 tokens), StyleGAN2 on 256 only'
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
        'evaluation': {
            'fid_every': FID_EVERY,
            'fid_samples': FID_SAMPLES,
            'fid_batch': FID_BATCH,
        },
        'saving': {
            'save_every': SAVE_EVERY,
            'out_dir': OUT_DIR,
            'fid_cache': FID_CACHE,
        }
    }
    config_path = os.path.join(OUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✅ Config saved to {config_path}")

save_config()

print(f"\n{'='*60}")
print("256x256 ARCHITECTURE SUMMARY (MAMBA128 VARIANT):")
print(f"{'='*60}")
print(f"G: Mamba 8→16→32→64→128 | StyleGAN2 256 | toRGB")
print(f"D: extra conv 256→128 | StyleGAN2-ADA 128→4 | mbstd | fc")
print(f"Batch: {BATCH_SIZE}")
print(f"Z_DIM: {Z_DIM}")
print(f"ch: 8/16/32/64=148 | 128=168 | 256=196")
print(f"⚠️  WARNING: Mamba on 128x128 = 16384 tokens, very memory intensive!")
print(f"{'='*60}\n")