import os
from typing import List, Optional, Tuple

import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# registry — single source of truth for dataset roots and classes
DATASET_CONFIGS = {
    "afhq128":    {"root": "AFHQ_resize_128",       "classes": sorted(["cat", "dog", "wild"])},
    "afhq":       {"root": "AFHQ_resize_256",       "classes": sorted(["cat", "dog", "wild"])},
    "afhq512":    {"root": "AFHQ_resize_512",       "classes": sorted(["cat", "dog", "wild"])},
    "celeba":     {"root": "CelebA_subset_256",     "classes": sorted([
        "Bald", "Black_Hair", "Blond_Hair", "Eyeglasses", "Goatee",
        "Male", "Mustache", "Smiling", "Wearing_Hat", "Young",
    ])},
    "lsun":       {"root": "LSUN_resize_256",       "classes": sorted(["bridge", "church_outdoor", "tower"])},
    "ffhq":       {"root": "FFHQ_resize_256",       "classes": sorted(["ffhq"])},
    "lsun_rooms": {"root": "LSUN_rooms_resize_128", "classes": sorted(["bedroom", "kitchen"])},
}


def get_dataset_cfg(dataset: str) -> dict:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: '{dataset}'. Available: {list(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[dataset]


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in _IMG_EXTS


class UniversalDataset(Dataset):
    def __init__(self, root: str, classes: List[str], transform=None):
        super().__init__()
        self.root      = root
        self.transform = transform
        self.classes   = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Data root not found: {root}")

        for c in self.classes:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            idx = self.class_to_idx[c]
            for fname in sorted(os.listdir(cdir)):
                if _is_image(fname):
                    self.samples.append((os.path.join(cdir, fname), idx))

        if not self.samples:
            raise RuntimeError(f"No images found in {root} for classes {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataloader(root: str, classes: List[str], bs: int, num_workers: int = 4) -> DataLoader:
    tfm = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = UniversalDataset(root=root, classes=classes, transform=tfm)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                      drop_last=True, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def make_class_rows(imgs, labels, num_classes: int, n_per_class: int = 10):
    out = []
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        count = min(len(idx), n_per_class)
        if count > 0:
            out.append(imgs[idx[:count]])
    return torch.cat(out, dim=0) if out else imgs[:n_per_class]


@torch.no_grad()
def sample_class_grid(G, device, path: str, classes: List[str], n_per_class: int = 10):
    if not classes:
        raise ValueError("classes is empty")
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    G.eval()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    class_indices = list(range(len(classes)))
    z = torch.randn(len(class_indices) * n_per_class, G.z_dim, device=device)
    y = torch.tensor([c for c in class_indices for _ in range(n_per_class)], device=device).long()
    imgs = G(z, y).clamp(-1, 1)
    grid = tv.utils.make_grid((imgs + 1) / 2, nrow=n_per_class)
    tv.utils.save_image(grid, path)
