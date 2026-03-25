import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# ── grid ──────────────────────────────────────────────────────────────────────

def generate_grid(G, device, Z_DIM, NUM_CLASSES, N_PER_CLASS, N_SHOW,
                  CLASS_NAMES, psi=0.8, seed=None, save=False):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        z_single = torch.randn(N_PER_CLASS, Z_DIM, device=device) * psi
        z = z_single.repeat(NUM_CLASSES, 1)
        y = torch.tensor([c for c in range(NUM_CLASSES) for _ in range(N_PER_CLASS)], device=device)
        imgs = G(z, y)
        imgs_np = (imgs.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(N_SHOW, NUM_CLASSES,
                             figsize=(NUM_CLASSES * 2.2, N_SHOW * 2.2),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    for cls_id in range(NUM_CLASSES):
        for j in range(N_SHOW):
            ax = axes[j, cls_id]
            img = imgs_np[cls_id * N_PER_CLASS + j].permute(1, 2, 0).numpy()
            ax.imshow(img, interpolation='lanczos')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_title(CLASS_NAMES[cls_id], fontsize=10, pad=4)
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.1)
    plt.show()


# ── z_tok scaling ─────────────────────────────────────────────────────────────

def z_tok_scaling_plot(G, device, Z_DIM, CLASS_NAMES, seed, gammas):
    NUM_CLASSES = len(CLASS_NAMES)
    torch.manual_seed(seed)
    z_base = torch.randn(1, Z_DIM, device=device)
    z_tok_base, z_dir_base = z_base.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    fig, axes = plt.subplots(
        NUM_CLASSES, len(gammas),
        figsize=(len(gammas) * 2.0, NUM_CLASSES * 2.2),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.08}
    )
    for cls_id in range(NUM_CLASSES):
        for col, gamma in enumerate(gammas):
            z_noisy = torch.cat([z_tok_base * gamma, z_dir_base], dim=1)
            with torch.no_grad():
                img = G(z_noisy, torch.tensor([cls_id], device=device))
            img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[cls_id, col]
            ax.imshow(img_np, interpolation='lanczos')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if cls_id == 0:
                ax.set_title(f"$\\gamma={gamma:.2f}$", fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls_id], fontsize=9, rotation=90, labelpad=4)
    plt.suptitle(f"z_tok scaling | seed={seed}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def z_tok_scaling_gif(G, device, Z_DIM, CLASS_NAMES, seed, n_frames,
                      gamma_range, img_size, save_path, fps):
    NUM_CLASSES = len(CLASS_NAMES)
    torch.manual_seed(seed)
    z_base = torch.randn(1, Z_DIM, device=device)
    z_tok_base, z_dir_base = z_base.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_frames)
    frames = []
    for gamma in gammas:
        row_imgs = []
        for cls_id in range(NUM_CLASSES):
            z_noisy = torch.cat([z_tok_base * gamma, z_dir_base], dim=1)
            with torch.no_grad():
                img = G(z_noisy, torch.tensor([cls_id], device=device))
            img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            row_imgs.append(Image.fromarray((img_np * 255).astype(np.uint8)))
        frame = Image.new("RGB", (img_size * NUM_CLASSES, img_size))
        for j, im in enumerate(row_imgs):
            frame.paste(im.resize((img_size, img_size), Image.LANCZOS), (j * img_size, 0))
        ImageDraw.Draw(frame).text((4, 4), f"γ={gamma:.3f}", fill=(255, 255, 255))
        frames.append(frame)
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"Saved: {save_path}  ({n_frames} frames, {fps} fps)")



def z_dir_diag_scaling_plot(G, device, Z_DIM, CLASS_NAMES, SCAN_DIRECTIONS, seed, gammas):
    NUM_CLASSES = len(CLASS_NAMES)
    torch.manual_seed(seed)
    z_base = torch.randn(1, Z_DIM, device=device)
    z_tok_base, z_dir_base = z_base.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    chunk = G.z_dir_dim // len(SCAN_DIRECTIONS)
    diag_idx = SCAN_DIRECTIONS.index('diag_left')
    fig, axes = plt.subplots(
        NUM_CLASSES, len(gammas),
        figsize=(len(gammas) * 2.0, NUM_CLASSES * 2.2),
        gridspec_kw={'wspace': 0.03, 'hspace': 0.08}
    )
    for cls_id in range(NUM_CLASSES):
        for col, gamma in enumerate(gammas):
            z_dir_scaled = z_dir_base.clone()
            z_dir_scaled[:, diag_idx * chunk:(diag_idx + 1) * chunk] *= gamma
            z_noisy = torch.cat([z_tok_base, z_dir_scaled], dim=1)
            with torch.no_grad():
                img = G(z_noisy, torch.tensor([cls_id], device=device))
            img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            ax = axes[cls_id, col]
            ax.imshow(img_np, interpolation='lanczos')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if cls_id == 0:
                ax.set_title(f"$\\gamma={gamma:.2f}$", fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls_id], fontsize=9, rotation=90, labelpad=4)
    plt.suptitle(f"z_dir[diag] scaling | seed={seed}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def z_dir_diag_scaling_gif(G, device, Z_DIM, CLASS_NAMES, SCAN_DIRECTIONS, seed,
                            n_frames, gamma_range, img_size, save_path, fps):
    NUM_CLASSES = len(CLASS_NAMES)
    torch.manual_seed(seed)
    z_base = torch.randn(1, Z_DIM, device=device)
    z_tok_base, z_dir_base = z_base.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    chunk = G.z_dir_dim // len(SCAN_DIRECTIONS)
    diag_idx = SCAN_DIRECTIONS.index('diag_left')
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_frames)
    frames = []
    for gamma in gammas:
        row_imgs = []
        for cls_id in range(NUM_CLASSES):
            z_dir_scaled = z_dir_base.clone()
            z_dir_scaled[:, diag_idx * chunk:(diag_idx + 1) * chunk] *= gamma
            z_noisy = torch.cat([z_tok_base, z_dir_scaled], dim=1)
            with torch.no_grad():
                img = G(z_noisy, torch.tensor([cls_id], device=device))
            img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            row_imgs.append(Image.fromarray((img_np * 255).astype(np.uint8)))
        frame = Image.new("RGB", (img_size * NUM_CLASSES, img_size))
        for j, im in enumerate(row_imgs):
            frame.paste(im.resize((img_size, img_size), Image.LANCZOS), (j * img_size, 0))
        ImageDraw.Draw(frame).text((4, 4), f"γ={gamma:.3f}", fill=(255, 255, 255))
        frames.append(frame)
    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"Saved: {save_path}  ({n_frames} frames, {fps} fps)")




def _mixed_latent(G, device, Z_DIM, SCAN_DIRECTIONS, z_base, z_override,
                  dir_indices, swap_tok, stage_filter, cls_id):
    z_tok_base, z_dir_base = z_base.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    z_tok_over, z_dir_over = z_override.split([G.z_tok_dim, G.z_dir_dim], dim=1)
    chunk = G.z_dir_dim // len(SCAN_DIRECTIONS)
    z_dir_mixed = z_dir_base.clone()
    if stage_filter is None and dir_indices:
        for d in dir_indices:
            z_dir_mixed[:, d*chunk:(d+1)*chunk] = z_dir_over[:, d*chunk:(d+1)*chunk]
    z_tok_mixed = z_tok_over if swap_tok else z_tok_base
    z_mixed = torch.cat([z_tok_mixed, z_dir_mixed], dim=1)
    hooks = []
    if stage_filter is not None and dir_indices:
        z_dir_stage = z_dir_base.clone()
        for d in dir_indices:
            z_dir_stage[:, d*chunk:(d+1)*chunk] = z_dir_over[:, d*chunk:(d+1)*chunk]
        stage_map = {
            '8x8':     G.mamba_8x8,
            '16x16':   G.mamba_16x16,
            '32x32':   G.mamba_32x32,
            '64x64':   G.mamba_64x64,
            '128x128': G.mamba_128x128,
        }
        for stage_name in stage_filter:
            for blk in stage_map[stage_name]:
                def make_hook(z_d):
                    def hook(module, args, kwargs):
                        return args, {**kwargs, 'z_dir': z_d}
                    return hook
                hooks.append(blk.register_forward_pre_hook(make_hook(z_dir_stage), with_kwargs=True))
    with torch.no_grad():
        img = G(z_mixed, torch.tensor([cls_id], device=device))
    for h in hooks:
        h.remove()
    return (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def latent_dir_swap_plot(G, device, Z_DIM, SCAN_DIRECTIONS, seed, class_names,
                         CLASS_NAMES, columns, save_path=None, psi=1.0):
    torch.manual_seed(seed)
    z_base     = torch.randn(1, Z_DIM, device=device) * psi
    z_override = torch.randn(1, Z_DIM, device=device) * psi
    n_classes  = len(class_names)
    fig, axes = plt.subplots(
        n_classes, len(columns),
        figsize=(len(columns) * 2.0, n_classes * 2.2),
        gridspec_kw={'wspace': 0.04, 'hspace': 0.08}
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]
    for row, cls_name in enumerate(class_names):
        cls_id = CLASS_NAMES.index(cls_name)
        for col_idx, (label, dir_indices, swap_tok, stage_filter) in enumerate(columns):
            ax = axes[row, col_idx]
            if col_idx == 0:
                with torch.no_grad():
                    img = G(z_base, torch.tensor([cls_id], device=device))
                img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
            else:
                img_np = _mixed_latent(G, device, Z_DIM, SCAN_DIRECTIONS, z_base,
                                       z_override, dir_indices, swap_tok, stage_filter, cls_id)
            ax.imshow(img_np, interpolation='lanczos')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=7, pad=3)
            if col_idx == 0:
                ax.set_ylabel(cls_name, fontsize=9, rotation=90, labelpad=4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()





def generate_mixed_class(model, device, SCAN_DIRECTIONS, z, base_cls,
                         override_dir_indices, override_cls,
                         stage_filter=None, patch_cnn=False):
    stage_map = {
        '8x8':     model.mamba_8x8,
        '16x16':   model.mamba_16x16,
        '32x32':   model.mamba_32x32,
        '64x64':   model.mamba_64x64,
        '128x128': model.mamba_128x128,
    }
    if stage_filter is None:
        all_blocks = [(name, blk) for name, blocks in stage_map.items() for blk in blocks]
    else:
        all_blocks = [(name, blk) for name in stage_filter for blk in stage_map[name]]
    override_directions = [SCAN_DIRECTIONS[i] for i in override_dir_indices]
    saved = {}
    for i, (name, blk) in enumerate(all_blocks):
        if hasattr(blk, 'class_embeds'):
            saved[('dlr', i)] = {}
            for direction in override_directions:
                if direction in blk.class_embeds:
                    emb = blk.class_embeds[direction]
                    saved[('dlr', i)][direction] = emb.weight.data[base_cls].clone()
                    emb.weight.data[base_cls] = emb.weight.data[override_cls].clone()
    if patch_cnn:
        cnn = model.style_refine_256
        saved['cnn'] = cnn.style.weight.data[base_cls].clone()
        cnn.style.weight.data[base_cls] = cnn.style.weight.data[override_cls].clone()
    with torch.no_grad():
        img = model(z, torch.tensor([base_cls], device=z.device))
    for i, (name, blk) in enumerate(all_blocks):
        if ('dlr', i) in saved and hasattr(blk, 'class_embeds'):
            for direction, orig in saved[('dlr', i)].items():
                blk.class_embeds[direction].weight.data[base_cls] = orig
    if patch_cnn and 'cnn' in saved:
        model.style_refine_256.style.weight.data[base_cls] = saved['cnn']
    return (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_override(G, device, SCAN_DIRECTIONS, z, base_cls, override_cls,
                  columns, save_path=None):
    fig, axes = plt.subplots(
        1, len(columns),
        figsize=(len(columns) * 2.0, 2.6),
        gridspec_kw={'wspace': 0.04}
    )
    for col_idx, (label, dir_indices, stage_filter, patch_cnn) in enumerate(columns):
        ax = axes[col_idx]
        if col_idx == 0:
            with torch.no_grad():
                img = G(z, torch.tensor([base_cls], device=device))
            img_np = (img[0].cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        else:
            img_np = generate_mixed_class(G, device, SCAN_DIRECTIONS, z,
                                          base_cls, dir_indices, override_cls,
                                          stage_filter, patch_cnn)
        ax.imshow(img_np, interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(label, fontsize=7, labelpad=3)
    plt.tight_layout()
    if save_path:
        ext = save_path.rsplit('.', 1)[-1].lower()
        plt.savefig(save_path, bbox_inches='tight', dpi=(None if ext == 'pdf' else 300))
        print(f"Saved: {save_path}")
    plt.show()