import argparse
import os
from pathlib import Path
from PIL import Image, ImageEnhance
import random
from tqdm import tqdm

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
def open_image(path: Path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Cannot open image {path}: {e}")
def resize_image(img: Image.Image, size):
    return img.resize(size, Image.BILINEAR)
def normalize_image(img: Image.Image):
    return img
def augment_image(img: Image.Image, seed=None):
    """Return an augmented PIL.Image instance. Random but deterministic if seed provided."""
    if seed is not None:
        random.seed(seed)
    im = img
    if random.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    angle = random.uniform(-15, 15)
    im = im.rotate(angle, resample=Image.BILINEAR, expand=False)
    if random.random() < 0.5:
        enh = ImageEnhance.Brightness(im)
        im = enh.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        enh = ImageEnhance.Contrast(im)
        im = enh.enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        enh = ImageEnhance.Color(im)
        im = enh.enhance(random.uniform(0.8, 1.3))

    return im

def process_one(path: Path, out_dir: Path, size=(224, 224), augment=0, fmt="jpg", seed_base=0):
    img = open_image(path)
    img_r = resize_image(img, size)
    img_r = normalize_image(img_r)

    stem = path.stem
    out_path = out_dir / f"{stem}.{fmt}"
    img_r.save(out_path, quality=95)
    for i in range(augment):
        aug = augment_image(img_r, seed=seed_base + i)
        aug_path = out_dir / f"{stem}_aug{i}.{fmt}"
        aug.save(aug_path, quality=95)
def walk_images(raw_dir: Path):
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS:
            yield p

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"), help="Raw image folder")
    p.add_argument("--out-dir", type=Path, default=Path("dataset/processed/images"), help="Processed images output")
    p.add_argument("--size", nargs=2, type=int, default=[224, 224], help="Output size: width height")
    p.add_argument("--augment", type=int, default=0, help="Number of augmented copies per original image")
    p.add_argument("--format", type=str, default="jpg", help="Output image format (jpg, png)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for augmentation")
    args = p.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    size = (args.size[0], args.size[1])
    augment = max(0, args.augment)
    fmt = args.format.lower()
    seed = args.seed

    if not raw_dir.exists():
        raise SystemExit(f"Raw directory {raw_dir} does not exist. Put your images in that folder.")

    ensure_dir(out_dir)

    files = list(walk_images(raw_dir))
    if not files:
        raise SystemExit(f"No images found in {raw_dir}. Check supported formats: {SUPPORTED_FORMATS}")

    print(f"Processing {len(files)} images -> {out_dir} (size={size}, augment={augment})")
    for idx, f in enumerate(tqdm(files)):
        try:
            process_one(f, out_dir, size=size, augment=augment, fmt=fmt, seed_base=seed + idx)
        except Exception as e:
            print(f"Warning: failed to process {f}: {e}")


if __name__ == "__main__":
    main()