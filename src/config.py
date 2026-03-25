import os
import json
from dataset import get_dataset_cfg

DATASET = "afhq"
IMG_SIZE = 256

_cfg        = get_dataset_cfg(DATASET)
CLASSES     = _cfg["classes"]
DATA_ROOT   = _cfg["root"]
FID_REAL_IMG = DATA_ROOT
NUM_CLASSES  = len(CLASSES)

BATCH_SIZE = 96
EPOCHS     = 500
SEED       = 42

FILM_ENABLED = {
    '8x8':    False,
    '16x16':  False,
    '32x32':  False,
    '64x64':  False,
    '128x128': False,
}

if all(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.3, '16x16': 0.3, '32x32': 0.15, '64x64': 0.1, '128x128': 0.05}
    VARIANT_NAME = "A_FiLM_all"
elif not any(FILM_ENABLED.values()):
    ROUTING_CLIP = {'8x8': 0.8, '16x16': 0.8, '32x32': 0.5, '64x64': 0.3, '128x128': 0.2}
    VARIANT_NAME = "C_NoFiLM"
else:
    ROUTING_CLIP = {'8x8': 0.5, '16x16': 0.5, '32x32': 0.3, '64x64': 0.2, '128x128': 0.2}
    VARIANT_NAME = "Custom"

OUT_DIR = "AFHQ_256_class_routing_refactor_2_1row"

LATENT_BASE = 88
LATENT_DIR  = 28
SCAN_DIRECTIONS = ['row_fwd' ]
NUM_DIR = len(SCAN_DIRECTIONS)
Z_DIM   = LATENT_BASE + LATENT_DIR * NUM_DIR

LR_G   = 9e-5
LR_D   = 3e-5
BETAS  = (0.0, 0.99)
LR_SCHEDULER = {
    'type': 'None',
    'warmup_epochs': 20,
    'min_lr_g': 1e-6,
    'min_lr_d': 5e-7,
    'T_max': EPOCHS,
}

D_STEPS = 1
G_STEPS = 1
GRADIENT_CLIP_G = 10
GRADIENT_CLIP_D = 15

R1_GAMMA       = 5
R1_INTERVAL    = 4
PL_WEIGHT      = 0.0
PL_INTERVAL    = 4
PL_START_EPOCH = 10000

EMA_DECAY_1      = 0.999
EMA_DECAY_2      = 0.9995
EMA_SWITCH_IMAGES = 1_000_000

AUG_CONFIG   = {'brightness': 0.1, 'contrast': 0.1, 'flip_prob': 0.5}
AUG_CONFIG_G = {'brightness': 0.1, 'contrast': 0.1, 'flip_prob': 0.5}
ROTATION_MODES = ['none', 'rot180']

G_CONFIG = {
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
    'ch_8x8':    148,
    'ch_16x16':  148,
    'ch_32x32':  148,
    'ch_64x64':  148,
    'ch_128x128': 168,
    'ch_256x256': 196,
    'use_pixel_shuffle_32': False,
    'use_pixel_shuffle_64': False,
    'z_mlp_hidden': {
        '8x8':   64,
        '16x16': 128,
        '32x32': 256,
        '64x64': 256,
    },
    'mamba_tokens': {'depth': 2, 'd_state': 64, 'd_conv': 4, 'expand': 2},
    'mamba_8x8': {
        'depth': 2, 'd_state': 64, 'd_conv': 4, 'expand': 1.5,
        'scan_directions': SCAN_DIRECTIONS, 'use_z_routing': True,
        'random_corner': False, 'noise': False,
        'routing_clip': ROUTING_CLIP['8x8'], 'residual_routing': True,
    },
    'mamba_16x16': {
        'depth': 1, 'd_state': 64, 'd_conv': 4, 'expand': 1.5,
        'scan_directions': SCAN_DIRECTIONS, 'use_z_routing': True,
        'random_corner': False, 'noise': False,
        'routing_clip': ROUTING_CLIP['16x16'], 'residual_routing': True,
    },
    'mamba_32x32': {
        'depth': 1, 'd_state': 64, 'd_conv': 4, 'expand': 1.5,
        'scan_directions': SCAN_DIRECTIONS, 'use_z_routing': True,
        'random_corner': False, 'noise': False,
        'routing_clip': ROUTING_CLIP['32x32'], 'residual_routing': True,
    },
    'mamba_64x64': {
        'depth': 1, 'd_state': 64, 'd_conv': 3, 'expand': 1,
        'scan_directions': SCAN_DIRECTIONS, 'use_z_routing': True,
        'random_corner': False, 'noise': False,
        'routing_clip': ROUTING_CLIP['64x64'], 'residual_routing': True,
    },
    'mamba_128x128': {
        'depth': 1, 'd_state': 48, 'd_conv': 3, 'expand': 1,
        'scan_directions': SCAN_DIRECTIONS, 'use_z_routing': True,
        'random_corner': False, 'noise': False,
        'routing_clip': ROUTING_CLIP['128x128'], 'residual_routing': True,
    },
    'output': {'final_ch': 64, 'refine_channels': [128]},
}

D_CONFIG = {
    'num_classes': NUM_CLASSES,
    'base_ch': 64,
    'channel_max': 384,
    'mbstd_group_size': 8,
}

SAVE_EVERY = 3

os.makedirs(OUT_DIR, exist_ok=True)