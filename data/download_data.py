# =========================
# Imports
# =========================
import io
import os
import shutil
import secrets
import string
from pathlib import Path

# kagglehub: download Kaggle datasets programmatically
import kagglehub

# Hugging Face datasets
from datasets import load_dataset

# PIL image handling + safety helpers
from PIL import Image, ImageFile, UnidentifiedImageError


# =========================
# Config
# =========================
# ROOT is the folder where this script lives.
ROOT = Path(__file__).resolve().parent

# Output folders for the final dataset (already resized to 224x224).
REAL_OUT = ROOT / "REAL_224"
FAKE_OUT = ROOT / "FAKE_224"
REAL_OUT.mkdir(parents=True, exist_ok=True)
FAKE_OUT.mkdir(parents=True, exist_ok=True)

# Target size for model input.
SIZE = 224

# If False, PIL will raise an error on truncated images instead of silently loading them.
# For "clean" datasets it's better to keep this strict.
ImageFile.LOAD_TRUNCATED_IMAGES = False


# =========================
# Utils: unique filenames
# =========================
_ALNUM = string.ascii_letters + string.digits

def random_alnum(n: int = 16) -> str:
    """
    Generate a random alphanumeric string.

    Uses `secrets` instead of `random`:
      - designed for generating unique tokens
      - lower chance of collisions in practice
      - deterministic reproducibility is NOT the goal here; uniqueness is.
    """
    return "".join(secrets.choice(_ALNUM) for _ in range(n))

def unique_path(dst_dir: Path, filename: str, rand_len: int = 16) -> Path:
    """
    Return a destination path that does not collide.

    If dst_dir/filename doesn't exist -> use it.
    If it exists -> generate a random filename but keep the same extension.

    Example:
      filename="a.jpg" exists -> produce something like "k8F1aZ...Q.jpg"
    """
    p = dst_dir / filename
    if not p.exists():
        return p

    ext = p.suffix  # includes the dot, e.g. ".jpg"
    # If the original has no extension, ext = ""
    while True:
        candidate = dst_dir / f"{random_alnum(rand_len)}{ext}"
        if not candidate.exists():
            return candidate


# =========================
# Utils: resize + validation
# =========================
def resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    """
    Resize the image so that the smaller side becomes `size`,
    then center-crop a square of size x size.

    This preserves aspect ratio (no stretching), but may crop content.
    Typical for classification pipelines.
    """
    w, h = img.size

    # Scale factor that makes min(w,h) -> size
    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # LANCZOS is high-quality down/up-sampling
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def is_valid_image(path: Path) -> bool:
    """
    Check if an image file can be opened and verified by PIL.

    `.verify()` does a lightweight integrity check without fully decoding pixels.
    It can catch truncated/corrupted files.

    Returns:
      True if it seems valid, False otherwise.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return False


def resize_inplace(folder: Path, size: int = 224):
    """
    Iterate over images in `folder`, validate them, resize+center-crop,
    and overwrite them as high-quality JPEG.

    Side effects:
      - Converts everything to RGB and JPEG
      - Overwrites the original file (same path)
      - Skips invalid images
    """
    saved = 0
    skipped = 0

    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue

        # Only process common image extensions
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        # Skip corrupted / invalid images
        if not is_valid_image(img_path):
            skipped += 1
            print(f"⚠️ Saltada (inválida): {img_path.name}")
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_224 = resize_and_center_crop(img, size)

                # Overwrite same file. Even if input was PNG, we save as JPEG.
                img_224.save(img_path, format="JPEG", quality=95, optimize=True)

            saved += 1
            if saved % 500 == 0:
                print(f"{folder.name}: {saved} imágenes procesadas")

        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            skipped += 1
            print(f"❌ Error con {img_path.name}: {e}")

    print(f"✅ {folder.name}: {saved} guardadas | ⚠️ {skipped} saltadas")


def copy_images(src_dirs, dst_dir: Path):
    """
    Copy images from multiple source directories into a single destination directory.

    Key behavior:
      - If a filename collision happens in dst_dir, it generates a unique random name.
      - Uses shutil.copy2 to preserve basic file metadata when possible.
    """
    count = 0
    for src in src_dirs:
        if not src.exists():
            print(f"⚠️ No existe: {src}")
            continue

        for img in src.iterdir():
            if not img.is_file():
                continue

            # If filename already exists, generate a unique path
            dst_file = unique_path(dst_dir, img.name)
            shutil.copy2(img, dst_file)
            count += 1

    return count


# ============================================================
# IMG4: Kaggle dataset -> merge train/test -> resize -> cleanup
# ============================================================
def run_img4():
    """
    Download a Kaggle dataset that already contains REAL and FAKE images,
    merge train+test folders into our REAL_OUT/FAKE_OUT, then resize everything to 224.

    Cleanup:
      - deletes the copied local dataset folder afterwards to save disk space.
    """
    print("\n=== IMG4: download -> juntar -> resize ===")

    dataset_path = Path(
        kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
    )

    # Copy dataset into the script directory (so we can access predictable paths)
    local_dataset = ROOT / "ai-generated-images-vs-real-images"
    if not local_dataset.exists():
        shutil.copytree(dataset_path, local_dataset)
        print(f"📥 Dataset copiado a: {local_dataset}")
    else:
        print(f"📦 Dataset ya existe en: {local_dataset}")

    # Kaggle dataset has separate train/test folders
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

    print(f"✅ IMG4: Imágenes REAL copiadas: {real_count}")
    print(f"✅ IMG4: Imágenes FAKE copiadas: {fake_count}")

    # Resize everything to SIZE and overwrite files
    print("🚀 IMG4: Resize REAL_224...")
    resize_inplace(REAL_OUT, SIZE)
    print("🚀 IMG4: Resize FAKE_224...")
    resize_inplace(FAKE_OUT, SIZE)

    # -------------------------
    # Final cleanup
    # -------------------------
    # Remove the intermediate dataset folder to avoid keeping duplicated data.
    try:
        shutil.rmtree(local_dataset)
        print(f"🧹 Carpeta eliminada: {local_dataset}")
    except Exception as e:
        print(f"⚠️ No se pudo borrar {local_dataset}: {e}")


# ============================================================
# IMG3: Hugging Face dataset (FAKE) -> download -> resize
# ============================================================
def run_img3(n: int = 2000):
    """
    Download FAKE images from the Hugging Face dataset "ostris/sdxl_10_reg".

    Notes:
      - Loads the dataset fully (no streaming).
      - Saves the first n images to FAKE_OUT.
      - Then resizes everything in-place to 224.
    """
    print("\n=== IMG3: download -> resize (FAKE) ===")
    print("1) Cargando dataset (sin streaming)...")
    ds = load_dataset("ostris/sdxl_10_reg", split="train")
    print("2) Dataset cargado.")
    print(f"3) Columnas: {ds.column_names}")
    print(f"4) Nº ejemplos: {len(ds)}")
    print(f"5) Carpeta destino: {FAKE_OUT.resolve()}")

    n = min(n, len(ds))
    print(f"6) Guardando {n} imágenes...")

    for i in range(n):
        img = ds[i]["image"]

        # Human-readable base filename. If it collides, unique_path() randomizes it.
        base_name = f"img3_{i:06}.png"
        out_path = unique_path(FAKE_OUT, base_name)

        img.save(out_path)
        if i % 50 == 0:
            print(f"   guardadas {i+1}/{n}")

    print("7) Resize inplace FAKE_224...")
    resize_inplace(FAKE_OUT, SIZE)


# ============================================================
# IMG1: Hugging Face dataset (REAL) streaming -> resize while saving
# ============================================================
def run_img1(n: int = 6000):
    """
    Download REAL images from a large dataset using streaming:
      "pixparse/cc12m-wds"

    Streaming means:
      - we don't download the entire dataset metadata/items first
      - we iterate over samples until we collect `n` valid images

    Important detail:
      - The dataset can store images in different fields ("image" vs "jpg")
      - Images might be stored as raw bytes or already decoded objects
    """
    print("\n=== IMG1: download (REAL) streaming + resize ===")
    print(f"Destino: {REAL_OUT.resolve()}")

    ds = load_dataset("pixparse/cc12m-wds", split="train", streaming=True)

    saved = 0
    for row in ds:
        if saved >= n:
            break

        img = None

        # Some HF datasets store images in "image"; it may be bytes or a PIL-like object.
        if "image" in row and row["image"] is not None:
            if isinstance(row["image"], (bytes, bytearray)):
                img = Image.open(io.BytesIO(row["image"]))
            else:
                img = row["image"]

        # Other variants store image bytes under "jpg"
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
                print(f"Guardadas: {saved}/{n}")
        except Exception:
            # If conversion/resize/save fails for any reason, skip the sample
            continue

    print(f"✅ IMG1: Listo: {saved} imágenes reales guardadas en {REAL_OUT.resolve()}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Number of images to download from each source
    IMG3_N = 2000
    IMG1_N = 6000

    # 1) Kaggle dataset provides both REAL and FAKE
    run_img4()

    # 2) Add more FAKE images from HF
    run_img3(n=IMG3_N)

    # 3) Add more REAL images from HF (streaming)
    run_img1(n=IMG1_N)

    # Summary
    print("\n🎉 Terminado.")
    print(f"REAL -> {REAL_OUT}")
    print(f"FAKE -> {FAKE_OUT}")
