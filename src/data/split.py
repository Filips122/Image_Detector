# =========================
# Imports / Constants
# =========================
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Allowed image file extensions (case-insensitive via .lower()).
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# Helper: recursively list images in a folder
# =========================
def _list_images(folder: Path) -> List[Path]:
    """
    Recursively scan `folder` and return a list of image file Paths.

    Notes:
      - Uses rglob("*") to traverse all subdirectories.
      - Filters only files with extensions in IMG_EXTS.
    """
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


# =========================
# Build a balanced list of (filepath, label)
# =========================
def build_file_list(
    root_dir: str,
    real_dir: str,
    fake_dir: str,
    max_total: int = 10000,
    seed: int = 42
) -> List[Tuple[str, int]]:
    """
    Build a *balanced* dataset list of (filepath, label).

    Label convention:
      0 -> REAL
      1 -> FAKE

    Balancing strategy:
      - Collect all REAL files and all FAKE files.
      - Choose the same number from each class:
          n = min(#real, #fake, max_total // 2)
      - Sample/shuffle deterministically using `seed`.
      - Return a combined list shuffled across classes.

    Why max_total // 2?
      Because the final dataset includes both classes, so the total size is ~2*n.
      This caps total samples at max_total while maintaining balance.
    """
    root = Path(root_dir)
    real_path = root / real_dir
    fake_path = root / fake_dir

    # Basic sanity checks: fail early if directories are missing.
    assert real_path.exists(), f"REAL dir not found: {real_path}"
    assert fake_path.exists(), f"FAKE dir not found: {fake_path}"

    # Collect all image files under each folder (including subfolders).
    real_files = _list_images(real_path)
    fake_files = _list_images(fake_path)

    # -------------------------
    # Balance classes
    # -------------------------
    # n is the number of samples per class.
    # We cannot exceed the smaller class size.
    # Also, we cap the total to max_total => per class cap is max_total//2.
    n = min(len(real_files), len(fake_files), max_total // 2)

    # Use a local RNG with a fixed seed for reproducible shuffling.
    rnd = random.Random(seed)
    rnd.shuffle(real_files)
    rnd.shuffle(fake_files)

    # Keep only first n from each class after shuffling (random subset).
    real_files = real_files[:n]
    fake_files = fake_files[:n]

    # Convert Path -> str and attach labels.
    pairs = [(str(p), 0) for p in real_files] + [(str(p), 1) for p in fake_files]

    # Shuffle again so the combined list isn't "all real then all fake".
    rnd.shuffle(pairs)

    return pairs


# =========================
# Split pairs into train/val/test
# =========================
def split_pairs(
    pairs: List[Tuple[str, int]],
    train: float,
    val: float,
    test: float,
    seed: int = 42
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Split a list of (filepath, label) into train/val/test partitions.

    Args:
      pairs: full dataset list
      train/val/test: fractions that must sum to 1.0 (within tolerance)
      seed: ensures deterministic shuffling

    Returns:
      {
        "train": [...],
        "val": [...],
        "test": [...]
      }

    Notes about splitting:
      - This is a simple random split (no stratification beyond earlier balancing).
      - It shuffles the list first, then slices contiguous chunks.
    """
    # Ensure the fractions sum to 1.0 (allow tiny floating-point error).
    assert abs((train + val + test) - 1.0) < 1e-6

    # Deterministic shuffle before slicing
    rnd = random.Random(seed)
    rnd.shuffle(pairs)

    n = len(pairs)

    # Compute split sizes (int truncation means test gets the remainder).
    n_train = int(n * train)
    n_val = int(n * val)

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }
