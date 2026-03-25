import os
import json
import time
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
from torchvision.utils import save_image

from config import *
from utils import set_seed, weights_init, EMA, DiffAug
from model.generator import Generator
from model.discriminator import Discriminator
from model.losses import d_logistic, g_logistic, r1_regularizer, path_length_regularizer
from dataset import sample_class_grid, get_dataloader, make_class_rows


def save_source_files(output_dir):
    for f in ['generator.py', 'discriminator.py', 'config.py', 'train.py']:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(output_dir, f))


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr):
    base_lr = optimizer.param_groups[0]['lr']
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def d_binary_stats(d_real, d_fake, thresh=0.0):
    r = d_real.detach().flatten()
    f = d_fake.detach().flatten()
    real_acc = (r > thresh).float().mean().item()
    fake_acc = (f < thresh).float().mean().item()
    return {
        "d_real_mean": r.mean().item(),
        "d_fake_mean": f.mean().item(),
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "acc": 0.5 * (real_acc + fake_acc),
    }


def append_jsonl(out_dir, record, filename="gan_metrics_log.json"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "a") as f:
        f.write(json.dumps(record) + "\n")


def train():
    set_seed()
    save_source_files(OUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    diff_aug   = DiffAug(**AUG_CONFIG)
    diff_aug_g = DiffAug(**AUG_CONFIG_G)

    G = Generator(G_CONFIG).to(device)
    D = Discriminator(**D_CONFIG).to(device)

    g_params = sum(p.numel() for p in G.parameters()) / 1e6
    d_params = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"G: {g_params:.2f}M  D: {d_params:.2f}M")

    def init_non_mamba(m):
        if "mamba" not in m.__class__.__name__.lower():
            weights_init(m)

    G.apply(init_non_mamba)
    D.apply(weights_init)

    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    if LR_SCHEDULER['type'] == 'cosine_warmup':
        schedG = get_cosine_schedule_with_warmup(optG, LR_SCHEDULER['warmup_epochs'], LR_SCHEDULER['T_max'], LR_SCHEDULER['min_lr_g'])
        schedD = get_cosine_schedule_with_warmup(optD, LR_SCHEDULER['warmup_epochs'], LR_SCHEDULER['T_max'], LR_SCHEDULER['min_lr_d'])
    else:
        schedG = schedD = None

    ema       = EMA(G, decay=EMA_DECAY_1)
    G_ema_eval = Generator(G_CONFIG).to(device)
    G_ema_eval.eval()

    z_dbg = torch.randn(16, Z_DIM, device=device)
    y_dbg = torch.randint(0, NUM_CLASSES, (16,), device=device)

    loader = get_dataloader(DATA_ROOT, CLASSES, BATCH_SIZE)
    dataset_size    = len(loader.dataset)
    images_per_epoch = dataset_size
    EMA_SWITCH_EPOCH = int(EMA_SWITCH_IMAGES / images_per_epoch)

    print(f"Dataset: {dataset_size:,}  EMA switch: epoch {EMA_SWITCH_EPOCH}")

    step = 0
    pl_mean = 0.0
    last_r1_val = last_pl_val = last_pl_mean_val = None
    current_d_steps = current_g_steps = 1
    epoch_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        used_d_steps = current_d_steps
        used_g_steps = current_g_steps

        if epoch == EMA_SWITCH_EPOCH:
            ema.decay = EMA_DECAY_2
            print(f"EMA decay → {EMA_DECAY_2} at epoch {epoch}")

        last_r1_val = last_pl_val = last_pl_mean_val = None
        losses_G, losses_D, d_stats_epoch = [], [], []

        G.train(); D.train()
        data_time = d_time = g_time = 0.0
        iter_start = time.time()

        for imgs, labels in loader:
            data_time += time.time() - iter_start
            imgs, labels = imgs.to(device), labels.to(device)
            b = imgs.size(0)
            step += 1

            t_d = time.time()
            for _ in range(used_d_steps):
                z      = torch.randn(b, Z_DIM, device=device)
                y_fake = torch.randint(0, NUM_CLASSES, (b,), device=device)

                with torch.no_grad():
                    x_fake = G(z, y_fake)
                x_fake_aug = diff_aug_g(x_fake)

                do_r1 = (step % R1_INTERVAL == 0)
                if do_r1:
                    imgs_r1 = diff_aug(imgs)
                    imgs_r1.requires_grad_(True)
                    with autocast(enabled=False):
                        d_real_r1 = D(imgs_r1.float(), labels)
                        r1 = r1_regularizer(d_real_r1, imgs_r1)
                    last_r1_val = float(r1.item())

                imgs_aug = diff_aug(imgs)
                with autocast(dtype=torch.bfloat16):
                    d_real = D(imgs_aug, labels)
                    d_fake = D(x_fake_aug, y_fake)
                    d_stats_epoch.append(d_binary_stats(d_real, d_fake))
                    lossD = d_logistic(d_real, d_fake)

                lossD = lossD.float()
                if do_r1:
                    lossD = lossD + 0.5 * R1_GAMMA * r1

                optD.zero_grad(set_to_none=True)
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), GRADIENT_CLIP_D)
                optD.step()
                losses_D.append(lossD.item())
            d_time += time.time() - t_d

            t_g = time.time()
            for _ in range(used_g_steps):
                z      = torch.randn(b, Z_DIM, device=device)
                y_fake = torch.randint(0, NUM_CLASSES, (b,), device=device)

                with autocast(dtype=torch.bfloat16):
                    x_fake     = G(z, y_fake)
                    x_fake_aug = diff_aug_g(x_fake)
                    d_fake     = D(x_fake_aug, y_fake)
                    lossG      = g_logistic(d_fake)

                if epoch >= PL_START_EPOCH and step % PL_INTERVAL == 0:
                    z_pl = torch.randn(b, Z_DIM, device=device).requires_grad_(True)
                    y_pl = torch.randint(0, NUM_CLASSES, (b,), device=device)
                    with autocast(enabled=False):
                        pl_penalty, pl_mean = path_length_regularizer(G, z_pl.float(), y_pl, pl_mean)
                    last_pl_val      = float(pl_penalty.item())
                    last_pl_mean_val = float(pl_mean)
                    lossG = lossG.float() + PL_WEIGHT * pl_penalty

                optG.zero_grad(set_to_none=True)
                lossG.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), GRADIENT_CLIP_G)
                optG.step()
                losses_G.append(lossG.item())

            ema.update(G)
            g_time += time.time() - t_g

            if torch.isnan(lossG).any() or torch.isnan(lossD).any():
                print(f"NaN at step {step}, stopping.")
                return

            iter_start = time.time()

        avg = {k: np.mean([s[k] for s in d_stats_epoch]) for k in d_stats_epoch[0]}
        real_acc, fake_acc = avg["real_acc"], avg["fake_acc"]

        next_d_steps = next_g_steps = 1
        if real_acc < 0.55 or fake_acc < 0.55:
            next_d_steps = 2
        elif real_acc > 0.75 or fake_acc > 0.75:
            next_g_steps = 2
        current_d_steps, current_g_steps = next_d_steps, next_g_steps

        if epoch % SAVE_EVERY == 0 or epoch == 1:
            ema.copy_to(G_ema_eval)
            sample_path = os.path.join(OUT_DIR, f"grid_epoch_{epoch:03d}.png")
            sample_class_grid(G_ema_eval, device, sample_path, CLASSES)

            with torch.no_grad():
                x_now = G(z_dbg, y_dbg)
                x_ema = G_ema_eval(z_dbg, y_dbg)
                ema_l1 = float((x_now - x_ema).abs().mean().item())

                dir_stats = []
                for w in getattr(G, "directions_w", []):
                    p = (w.mean(0) if w.dim() == 2 else w).clamp_min(1e-12)
                    dir_stats.append({
                        "mean": p.tolist(),
                        "entropy": float(-(p * p.log()).sum().item()),
                        "pmax": float(p.max().item()),
                    })

                dlr = [
                    {"block": i, "gamma_abs_mean": ga, "beta_abs_mean": ba}
                    for i, (ga, ba) in enumerate(zip(getattr(G, "dlr_gamma_abs", []), getattr(G, "dlr_beta_abs", [])))
                ]

                routing_alpha_log = []
                for i, alpha in enumerate(getattr(G, "routing_alphas", [])):
                    routing_alpha_log.append({
                        "block": i, "alpha": alpha.tolist(),
                        "alpha_mean": float(alpha.mean().item()),
                    })

                append_jsonl(OUT_DIR, {
                    "epoch": epoch, "step": step, "ema_l1": ema_l1,
                    "r1_last": last_r1_val, "pl_penalty_last": last_pl_val,
                    "directional": dir_stats, "dlr": dlr, "routing_alpha": routing_alpha_log,
                    "img": {"min": float(x_now.min()), "max": float(x_now.max()), "std": float(x_now.std())},
                    "accuracy": {"real_acc": real_acc, "fake_acc": fake_acc, "total_acc": avg["acc"]},
                    "d_steps_used": used_d_steps, "g_steps_used": used_g_steps,
                })

            G_eval = Generator(G_CONFIG).to(device)
            ema.copy_to(G_eval)
            torch.save(G,      f"{OUT_DIR}/G_full_epoch_{epoch:03d}.pt")
            torch.save(G_eval, f"{OUT_DIR}/G_full_ema_epoch_{epoch:03d}.pt")

        epoch_time = time.time() - epoch_start
        avg_d = sum(losses_D) / max(1, len(losses_D))
        avg_g = sum(losses_G) / max(1, len(losses_G))
        print(f"Epoch {epoch}/{EPOCHS}  D={avg_d:.4f}  G={avg_g:.4f}  "
              f"real={real_acc*100:.1f}%  fake={fake_acc*100:.1f}%  "
              f"D_steps={used_d_steps}  G_steps={used_g_steps}  "
              f"t={epoch_time:.1f}s")

        if schedG: schedG.step()
        if schedD: schedD.step()
        epoch_start = time.time()

    print("Training complete.")


if __name__ == "__main__":
    train()
