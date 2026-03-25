import os
import random
from typing import List
from PIL import Image
import cv2
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

DATASET = "afhq"   # afhq | afhq512 | ffhq | celeba | lsun

OUT_SIZE = 256
OVERWRITE = False
JPEG_QUALITY = 100

AFHQ_CONFIG = {
    "input_root":  "afhq/train",
    "output_root": "AFHQ_resize_256",
    "classes":     ["cat", "dog", "wild"],
}

AFHQ512_CONFIG = {
    "input_root":  "AFHQ/train",
    "output_root": "AFHQ_resize_512",
    "classes":     ["cat", "dog", "wild"],
}

FFHQ_CONFIG = {
    "input_root":  "FFHQ/images1024x1024",
    "output_root": "FFHQ_resize_256",
    "class_name":  "ffhq",
}

CELEBA_CONFIG = {
    "input_images": "img_align_celeba",
    "attr_file":    "img_align_celeba/list_attr_celeba.txt",
    "output_root":  "CelebA_subset_256",
    "classes": [
        "Male", "Smiling", "Eyeglasses", "Goatee", "Mustache",
        "Wearing_Hat", "Bald", "Young", "Blond_Hair", "Black_Hair",
    ],
    "max_per_class": 10000,
    "crop_size":     178,
}

LSUN_CONFIG = {
    "input_root":        "data/lsun",
    "output_root":       "LSUN_resize_256",
    "classes":           ["tower", "church_outdoor", "bridge"],
    "split":             "train",
    "crop_size":         256,
    "samples_per_class": 20000,
    "seed":              123,
}

# ============================================================
# HELPERS
# ============================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in IMG_EXTS

def center_crop_cv2(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]

def center_crop_pil(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    return img.crop(((w - size) // 2, (h - size) // 2,
                     (w - size) // 2 + size, (h - size) // 2 + size))

def collect_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, fnames in os.walk(root):
        for f in sorted(fnames):
            if is_image(f):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)

# ============================================================
# DATASETS
# ============================================================

def prep_afhq(config: dict, out_size: int):
    in_base  = config["input_root"]
    out_base = config["output_root"]

    if not os.path.isdir(in_base):
        raise FileNotFoundError(f"Input not found: {in_base}")

    for cls in config["classes"]:
        in_dir  = os.path.join(in_base, cls)
        out_dir = os.path.join(out_base, cls)

        if not os.path.isdir(in_dir):
            print(f"Skipping missing class dir: {in_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in sorted(os.listdir(in_dir)) if is_image(f)]

        for fname in tqdm(files, desc=f"{cls}"):
            out_path = os.path.join(out_dir, fname)
            if not OVERWRITE and os.path.isfile(out_path):
                continue
            try:
                img = cv2.imread(os.path.join(in_dir, fname), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("imread returned None")
                img = center_crop_cv2(img)
                img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path, img)
            except Exception as e:
                print(f"Failed {fname}: {e}")


def prep_ffhq(config: dict, out_size: int):
    in_root  = config["input_root"]
    out_dir  = os.path.join(config["output_root"], config["class_name"])

    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"Input not found: {in_root}")

    os.makedirs(out_dir, exist_ok=True)
    paths = collect_images(in_root)

    for src in tqdm(paths, desc="ffhq"):
        out_path = os.path.join(out_dir, os.path.basename(src))
        if not OVERWRITE and os.path.isfile(out_path):
            continue
        try:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("imread returned None")
            img = center_crop_cv2(img)
            img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(out_path, img)
        except Exception as e:
            print(f"Failed {src}: {e}")


def prep_celeba(config: dict, out_size: int):
    raw_dir       = config["input_images"]
    attr_file     = config["attr_file"]
    out_root      = config["output_root"]
    classes       = config["classes"]
    max_per_class = config["max_per_class"]
    crop_size     = config["crop_size"]

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Input not found: {raw_dir}")
    if not os.path.isfile(attr_file):
        raise FileNotFoundError(f"Attr file not found: {attr_file}")

    for cls in classes:
        os.makedirs(os.path.join(out_root, cls), exist_ok=True)

    with open(attr_file) as f:
        lines = f.readlines()

    attr_idx = {name: i for i, name in enumerate(lines[1].split())}
    for attr in classes:
        if attr not in attr_idx:
            raise ValueError(f"Attribute '{attr}' not found in {attr_file}")

    counts = {cls: 0 for cls in classes}

    for line in tqdm(lines[2:], desc="celeba"):
        if all(v >= max_per_class for v in counts.values()):
            break

        parts    = line.split()
        filename = parts[0]
        values   = parts[1:]

        targets = [
            attr for attr in classes
            if values[attr_idx[attr]] == '1' and counts[attr] < max_per_class
        ]
        if not targets:
            continue

        try:
            with Image.open(os.path.join(raw_dir, filename)) as img:
                img = img.convert("RGB")
                img = center_crop_pil(img, crop_size)
                img = img.resize((out_size, out_size), Image.BICUBIC)
                for attr in targets:
                    dst = os.path.join(out_root, attr, filename)
                    if not OVERWRITE and os.path.isfile(dst):
                        continue
                    img.save(dst, quality=JPEG_QUALITY)
                    counts[attr] += 1
        except Exception as e:
            print(f"Failed {filename}: {e}")


def prep_lsun(config: dict, out_size: int):
    from torchvision.datasets import LSUN

    lsun_root         = config["input_root"]
    out_root          = config["output_root"]
    classes           = config["classes"]
    split             = config["split"]
    crop_size         = config["crop_size"]
    samples_per_class = config["samples_per_class"]
    seed              = config["seed"]

    if not os.path.isdir(lsun_root):
        raise FileNotFoundError(f"Input not found: {lsun_root}")

    for cls in classes:
        out_dir = os.path.join(out_root, cls)
        os.makedirs(out_dir, exist_ok=True)

        ds   = LSUN(root=lsun_root, classes=[f"{cls}_{split}"])
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)

        done = 0
        for i in tqdm(idxs, desc=cls, total=samples_per_class):
            if done >= samples_per_class:
                break

            out_path = os.path.join(out_dir, f"{cls}_{done:06d}.jpg")
            if not OVERWRITE and os.path.isfile(out_path):
                done += 1
                continue

            try:
                img, _ = ds[i]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if w < crop_size or h < crop_size:
                    continue
                img = center_crop_pil(img, crop_size)
                img = img.resize((out_size, out_size), Image.BICUBIC)
                img.save(out_path, format="JPEG", quality=JPEG_QUALITY)
                done += 1
            except Exception:
                pass

# ============================================================
# MAIN
# ============================================================

def main():
    size = 512 if DATASET == "afhq512" else OUT_SIZE

    if DATASET == "afhq":
        prep_afhq(AFHQ_CONFIG, size)
    elif DATASET == "afhq512":
        prep_afhq(AFHQ512_CONFIG, size)
    elif DATASET == "ffhq":
        prep_ffhq(FFHQ_CONFIG, size)
    elif DATASET == "celeba":
        prep_celeba(CELEBA_CONFIG, size)
    elif DATASET == "lsun":
        prep_lsun(LSUN_CONFIG, size)
    else:
        raise ValueError(f"Unknown dataset: '{DATASET}'")


if __name__ == "__main__":
    main()
