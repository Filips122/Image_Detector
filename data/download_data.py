# =========================
# Imports
# =========================
import io
import os
import shutil
import secrets
import string
from pathlib import Path

import kagglehub
from datasets import load_dataset
from PIL import Image, ImageFile, UnidentifiedImageError


# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parent

REAL_OUT = ROOT / "REAL_224"
FAKE_OUT = ROOT / "FAKE_224"
REAL_OUT.mkdir(parents=True, exist_ok=True)
FAKE_OUT.mkdir(parents=True, exist_ok=True)

SIZE = 224
ImageFile.LOAD_TRUNCATED_IMAGES = False


# =========================
# Utils: unique filenames
# =========================
_ALNUM = string.ascii_letters + string.digits

def random_alnum(n: int = 16) -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(n))

def unique_path(dst_dir: Path, filename: str, rand_len: int = 16) -> Path:
    p = dst_dir / filename
    if not p.exists():
        return p

    ext = p.suffix
    while True:
        candidate = dst_dir / f"{random_alnum(rand_len)}{ext}"
        if not candidate.exists():
            return candidate


# =========================
# Utils: resize + validation
# =========================
def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return False


def resize_inplace(folder: Path, size: int = 224):
    saved = 0
    skipped = 0

    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        if not is_valid_image(img_path):
            skipped += 1
            print(f"Skipped (invalid): {img_path.name}")
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_224 = resize_and_center_crop(img, size)
                img_224.save(img_path, format="JPEG", quality=95, optimize=True)

            saved += 1
            if saved % 500 == 0:
                print(f"{folder.name}: {saved} images processed")

        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            skipped += 1
            print(f"Error with {img_path.name}: {e}")

    print(f"{folder.name}: {saved} saved | {skipped} skipped")


def copy_images(src_dirs, dst_dir: Path):
    count = 0
    for src in src_dirs:
        if not src.exists():
            print(f"Does not exist: {src}")
            continue

        for img in src.iterdir():
            if not img.is_file():
                continue

            dst_file = unique_path(dst_dir, img.name)
            shutil.copy2(img, dst_file)
            count += 1

    return count


# ============================================================
# IMG4: Kaggle dataset
# ============================================================
def run_img4():
    print("\n=== IMG4: download -> merge -> resize ===")

    dataset_path = Path(
        kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
    )

    local_dataset = ROOT / "ai-generated-images-vs-real-images"
    if not local_dataset.exists():
        shutil.copytree(dataset_path, local_dataset)
        print(f"Dataset copied to: {local_dataset}")
    else:
        print(f"Dataset already exists at: {local_dataset}")

    real_dirs = [
        local_dataset / "train" / "real",
        local_dataset / "test" / "real",
    ]
    fake_dirs = [
        local_dataset / "train" / "fake",
        local_dataset / "test" / "fake",
    ]

    real_count = copy_images(real_dirs, REAL_OUT)
    fake_count = copy_images(fake_dirs, FAKE_OUT)

    print(f"IMG4: REAL images copied: {real_count}")
    print(f"IMG4: FAKE images copied: {fake_count}")

    print("IMG4: Resizing REAL_224...")
    resize_inplace(REAL_OUT, SIZE)
    print("IMG4: Resizing FAKE_224...")
    resize_inplace(FAKE_OUT, SIZE)

    try:
        shutil.rmtree(local_dataset)
        print(f"Folder removed: {local_dataset}")
    except Exception as e:
        print(f"Could not remove {local_dataset}: {e}")


# ============================================================
# IMG3: Hugging Face FAKE
# ============================================================
def run_img3(n: int = 2000):
    print("\n=== IMG3: download -> resize (FAKE) ===")
    print("1) Loading dataset (no streaming)...")
    ds = load_dataset("ostris/sdxl_10_reg", split="train")
    print("2) Dataset loaded.")
    print(f"3) Columns: {ds.column_names}")
    print(f"4) Number of samples: {len(ds)}")
    print(f"5) Output directory: {FAKE_OUT.resolve()}")

    n = min(n, len(ds))
    print(f"6) Saving {n} images...")

    for i in range(n):
        img = ds[i]["image"]
        base_name = f"img3_{i:06}.png"
        out_path = unique_path(FAKE_OUT, base_name)
        img.save(out_path)

        if i % 50 == 0:
            print(f"   saved {i+1}/{n}")

    print("7) Resizing FAKE_224 in-place...")
    resize_inplace(FAKE_OUT, SIZE)


# ============================================================
# IMG1: Hugging Face REAL (streaming)
# ============================================================
def run_img1(n: int = 6000):
    print("\n=== IMG1: download (REAL) streaming + resize ===")
    print(f"Output directory: {REAL_OUT.resolve()}")

    ds = load_dataset("pixparse/cc12m-wds", split="train", streaming=True)

    saved = 0
    for row in ds:
        if saved >= n:
            break

        img = None
        if "image" in row and row["image"] is not None:
            if isinstance(row["image"], (bytes, bytearray)):
                img = Image.open(io.BytesIO(row["image"]))
            else:
                img = row["image"]
        elif "jpg" in row and row["jpg"] is not None:
            img = row["jpg"]

        if img is None:
            continue

        try:
            img = img.convert("RGB")
            img_224 = resize_and_center_crop(img, SIZE)

            base_name = f"img1_{saved:06}.jpg"
            out_path = unique_path(REAL_OUT, base_name)
            img_224.save(out_path, format="JPEG", quality=95, optimize=True)

            saved += 1
            if saved % 200 == 0:
                print(f"Saved: {saved}/{n}")
        except Exception:
            continue

    print(f"IMG1: Done: {saved} real images saved to {REAL_OUT.resolve()}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    IMG3_N = 2000
    IMG1_N = 6000

    run_img4()
    run_img3(n=IMG3_N)
    run_img1(n=IMG1_N)

    print("\nFinished.")
    print(f"REAL -> {REAL_OUT}")
    print(f"FAKE -> {FAKE_OUT}")
