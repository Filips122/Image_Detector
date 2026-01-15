# =========================
# Imports / Dependencies
# =========================
import os
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image
import numpy as np

# External deps:
#   - imagehash: perceptual hashing (pHash) for near-duplicate detection
#   - opencv (cv2): image processing utilities for heuristics
#   - tqdm: progress bars
import imagehash
import cv2
from tqdm import tqdm


# =========================
# Constants
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# File listing utilities
# =========================
def list_images(folder: Path):
    """
    Recursively list image files inside `folder`.
    Returns a list of Path objects.
    """
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def safe_open_rgb(path: Path):
    """
    Safely open an image and convert it to RGB.
    Returns:
      - PIL.Image in RGB if successful
      - None if the file can't be opened/decoded
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


# =========================
# Perceptual hashing (pHash) helpers
# =========================
def compute_phash(img: Image.Image, hash_size=16):
    """
    Compute a perceptual hash (pHash) for an image.

    hash_size=16 -> 16x16 DCT hash => 256-bit hash.
    Larger hash sizes are more discriminative than the common hash_size=8 (64-bit).
    """
    return imagehash.phash(img, hash_size=hash_size)


def hamming(h1, h2):
    """
    Hamming distance between two ImageHash objects.

    In `imagehash`, subtraction returns the Hamming distance (#bit differences).
    """
    return (h1 - h2)


# ============================================================
# Duplicate clustering using pHash + Union-Find (Disjoint Sets)
# ============================================================
def connected_components_from_hashes(paths, hashes, max_dist=8):
    """
    Group near-duplicate images using perceptual hash similarity.

    Approach:
      - We want connected components where edges exist if hamming(pHash) <= max_dist.
      - Naively comparing all pairs is O(n^2), too expensive for large datasets.
      - This implementation reduces comparisons by bucketing hashes by a prefix.

    Parameters:
      paths: list of paths (Path or str)
      hashes: list of corresponding pHash objects
      max_dist: max Hamming distance considered "near-duplicate"
                Typical values for hash_size=16: ~6..10

    Complexity:
      - Still O(k^2) inside each bucket, but buckets are usually much smaller than the full set.
    """
    # -------------------------
    # 1) Bucket items by a hash prefix to reduce comparisons
    # -------------------------
    buckets = defaultdict(list)
    for p, h in zip(paths, hashes):
        # Prefix of the hash as a heuristic grouping key.
        # Using first 16 hex chars is a compromise: fewer comparisons, still catches many near matches.
        key = str(h)[:16]
        buckets[key].append((p, h))

    # -------------------------
    # 2) Union-Find (Disjoint Set Union) to build connected components
    # -------------------------
    parent = {}
    rank = {}

    def find(x):
        """
        Find with path compression.
        Ensures amortized almost-O(1) find operations.
        """
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        """
        Union by rank.
        Joins two sets efficiently.
        """
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        rank.setdefault(ra, 0)
        rank.setdefault(rb, 0)
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # -------------------------
    # 3) Compare pairs within each bucket
    # -------------------------
    for key, items in tqdm(buckets.items(), desc="Clustering (bucketed pHash)"):
        if len(items) <= 1:
            continue

        # All-vs-all inside bucket (expected to be small)
        for i in range(len(items)):
            pi, hi = items[i]
            for j in range(i + 1, len(items)):
                pj, hj = items[j]

                # If hashes are close enough, consider them connected duplicates
                if hamming(hi, hj) <= max_dist:
                    union(str(pi), str(pj))

    # -------------------------
    # 4) Build components from union-find structure
    # -------------------------
    groups = defaultdict(list)
    for p in paths:
        root = find(str(p))
        groups[root].append(str(p))

    # Output: list of groups, each group is a list of file paths (strings)
    return list(groups.values())


# ============================================================
# Image quality / style heuristics (used for audit)
# ============================================================
def laplacian_var(img_rgb: np.ndarray):
    """
    Sharpness proxy: variance of Laplacian (higher = sharper / more edges).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def saturation_mean(img_rgb: np.ndarray):
    """
    Average saturation in HSV space (normalized to 0..1).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) / 255.0


def edges_density(img_rgb: np.ndarray):
    """
    Edge density: fraction of pixels flagged by Canny edge detector.
    Returns a value roughly in [0..1].
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(gray, 80, 160)
    return float((e > 0).mean())


def jpeg_blockiness_score(img_rgb: np.ndarray):
    """
    Simple heuristic to detect JPEG "blockiness":
      - JPEG compression works in 8x8 blocks.
      - Strong compression often creates visible discontinuities at 8x8 boundaries.

    Strategy:
      - Measure average absolute difference across vertical boundaries at columns 8,16,24,...
      - Measure average absolute difference across horizontal boundaries at rows 8,16,24,...
      - Average the two.

    Notes:
      - Works even when width/height not multiples of 8 (it just uses available boundaries).
      - This is a heuristic, not a definitive detector.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape[:2]

    if w < 9 or h < 9:
        return 0.0

    # Vertical boundaries between columns 7|8, 15|16, 23|24, ...
    cols = np.arange(8, w, 8)
    if len(cols) > 0:
        v = np.abs(gray[:, cols] - gray[:, cols - 1]).mean()
    else:
        v = 0.0

    # Horizontal boundaries between rows 7|8, 15|16, 23|24, ...
    rows = np.arange(8, h, 8)
    if len(rows) > 0:
        hh = np.abs(gray[rows, :] - gray[rows - 1, :]).mean()
    else:
        hh = 0.0

    return float((v + hh) / 2.0)


def renderish_heuristic(img_rgb: np.ndarray):
    """
    VERY rough heuristic to estimate "render/illustration-like" vs "photo-like".

    Intuition (not guaranteed!):
      - Renders/illustrations may have:
          * cleaner / more structured edges (higher edge density)
          * different sharpness statistics
          * more stylized color saturation

    Returns:
      score in [0..1], where higher means "more renderish".

    IMPORTANT:
      This is not a classifier. It's an audit hint for dataset characterization.
    """
    ed = edges_density(img_rgb)          # 0..1
    sat = saturation_mean(img_rgb)       # 0..1
    sharp = laplacian_var(img_rgb)       # >0

    # Normalize sharpness into ~[0..1] using a log-like compression:
    #   log1p(sharp) reduces sensitivity to huge values,
    #   then map to [0..1] with an exponential saturation curve.
    sharp_n = 1.0 - math.exp(-math.log1p(sharp) / 6.0)

    # Weighted combination (tunable)
    score = 0.45 * ed + 0.25 * sharp_n + 0.30 * sat
    return float(max(0.0, min(1.0, score)))


# ============================================================
# Group-aware, stratified splitting
# ============================================================
def stratified_group_split(groups, group_labels, train=0.8, val=0.1, test=0.1, seed=42):
    """
    Split data into train/val/test *by groups* (duplicate-safe),
    while trying to preserve class balance at the group level.

    Inputs:
      groups: list of groups, each group is a list of image paths (strings)
              (each group represents a connected component of near-duplicates)
      group_labels: list[int], one label per group (0 real, 1 fake)

    Key idea:
      - Instead of splitting individual images, split entire groups.
      - This prevents near-duplicates from leaking across train/val/test.
      - Stratification is done by label: groups of label 0 and label 1 are split separately.
    """
    assert abs(train + val + test - 1.0) < 1e-6
    rnd = random.Random(seed)

    # Collect indices of groups by their label
    idx_by_label = {0: [], 1: []}
    for i, y in enumerate(group_labels):
        idx_by_label[y].append(i)

    # Shuffle group indices per label for randomization
    for y in (0, 1):
        rnd.shuffle(idx_by_label[y])

    def take_split(idxs, frac):
        """
        Take first round(n*frac) indices for a split, return (taken, remaining).
        Using round() keeps proportions closer for small counts.
        """
        n = len(idxs)
        k = int(round(n * frac))
        return idxs[:k], idxs[k:]

    # Split label 0 groups into train / (val+test), then split the remainder into val/test
    train_idx, rest0 = take_split(idx_by_label[0], train)
    val_idx0, test_idx0 = take_split(rest0, val / (val + test))

    # Same for label 1 groups
    train_idx_f, rest1 = take_split(idx_by_label[1], train)
    val_idx1, test_idx1 = take_split(rest1, val / (val + test))

    # Merge labels back together per split
    train_idx = train_idx + train_idx_f
    val_idx = val_idx0 + val_idx1
    test_idx = test_idx0 + test_idx1

    # Shuffle within each split
    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    rnd.shuffle(test_idx)

    def expand(idxs):
        """
        Expand group indices into a flat list of (path, label) pairs.
        Every image in a group inherits the group's label.
        """
        out = []
        for gi in idxs:
            y = group_labels[gi]
            for p in groups[gi]:
                out.append((p, y))
        return out

    return {"train": expand(train_idx), "val": expand(val_idx), "test": expand(test_idx)}


# ============================================================
# CLI / Main script
# ============================================================
def main():
    """
    End-to-end pipeline:
      1) List REAL and FAKE images
      2) Compute pHash for each image
      3) Cluster near-duplicates across the *entire* dataset (detect cross-class leakage too)
      4) Build group labels (majority label per group)
      5) Produce duplicate stats + a "renderish" estimate for FAKE subset
      6) Perform a stratified group split (duplicate-safe)
      7) Save split + audit report + preview file
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="carpeta data/")
    ap.add_argument("--real", default="REAL_224")
    ap.add_argument("--fake", default="FAKE_224")
    ap.add_argument("--out", default="artifacts/audit_split", help="salida")
    ap.add_argument("--hash_size", type=int, default=16)
    ap.add_argument("--phash_max_dist", type=int, default=8)
    ap.add_argument("--max_images_per_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--renderish_samples", type=int, default=800, help="muestras para estimación renderish")
    args = ap.parse_args()

    # -------------------------
    # Resolve paths
    # -------------------------
    root = Path(args.root)
    real_dir = root / args.real
    fake_dir = root / args.fake
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(args.seed)

    # -------------------------
    # Load file lists (optionally subsample)
    # -------------------------
    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)
    rnd.shuffle(real_files)
    rnd.shuffle(fake_files)

    if args.max_images_per_class and args.max_images_per_class > 0:
        real_files = real_files[:args.max_images_per_class]
        fake_files = fake_files[:args.max_images_per_class]

    print(f"REAL files: {len(real_files)}  FAKE files: {len(fake_files)}")

    # -------------------------
    # Compute pHash for all images (REAL + FAKE)
    # -------------------------
    all_paths = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    hashes = []
    ok_paths = []
    ok_labels = []

    for p, y in tqdm(list(zip(all_paths, all_labels)), total=len(all_paths), desc="Computing pHash"):
        img = safe_open_rgb(p)
        if img is None:
            continue
        h = compute_phash(img, hash_size=args.hash_size)
        hashes.append(h)
        ok_paths.append(p)
        ok_labels.append(y)

    # -------------------------
    # Cluster near-duplicates across the entire set
    # (detect duplicates inside a class AND across classes)
    # -------------------------
    groups = connected_components_from_hashes(ok_paths, hashes, max_dist=args.phash_max_dist)

    # Map each path -> original label
    path_to_label = {str(p): y for p, y in zip(ok_paths, ok_labels)}

    # -------------------------
    # Assign a label per group via majority vote
    # -------------------------
    group_labels = []
    group_sizes = []
    cross_class_groups = 0

    for g in groups:
        ys = [path_to_label[p] for p in g]
        c = Counter(ys)

        # Majority vote:
        # If tie or more fakes than reals -> label as fake
        y = 1 if c[1] >= c[0] else 0

        group_labels.append(y)
        group_sizes.append(len(g))

        # If a group contains both labels, it suggests potential leakage / mislabeled duplicates
        if c[0] > 0 and c[1] > 0:
            cross_class_groups += 1

    # -------------------------
    # Duplicate stats summary
    # -------------------------
    dup_groups = [sz for sz in group_sizes if sz > 1]
    stats = {
        "n_images_used": len(ok_paths),
        "n_groups": len(groups),
        "n_duplicate_groups": int(len(dup_groups)),
        "dup_group_size_summary": {
            "min": int(min(dup_groups)) if dup_groups else 1,
            "max": int(max(dup_groups)) if dup_groups else 1,
            "mean": float(np.mean(dup_groups)) if dup_groups else 1.0,
        },
        "cross_class_groups": int(cross_class_groups),
        "phash": {"hash_size": args.hash_size, "max_dist": args.phash_max_dist},
    }

    # -------------------------
    # "Renderish" audit on a sample of FAKE images
    # -------------------------
    fake_ok = [Path(p) for p in ok_paths if path_to_label[str(p)] == 1]
    rnd.shuffle(fake_ok)
    sample_fake = fake_ok[: min(args.renderish_samples, len(fake_ok))]

    render_scores = []
    jpeg_block = []
    sharpness = []

    for p in tqdm(sample_fake, desc="Renderish estimate (FAKE sample)"):
        img = safe_open_rgb(p)
        if img is None:
            continue
        arr = np.array(img)

        # Heuristics computed per image
        render_scores.append(renderish_heuristic(arr))
        jpeg_block.append(jpeg_blockiness_score(arr))
        sharpness.append(laplacian_var(arr))

    stats["fake_renderish_estimate"] = {
        "n_samples": int(len(render_scores)),
        "renderish_score_mean": float(np.mean(render_scores)) if render_scores else None,
        "renderish_score_p50": float(np.percentile(render_scores, 50)) if render_scores else None,
        "renderish_score_p90": float(np.percentile(render_scores, 90)) if render_scores else None,
        "jpeg_blockiness_mean": float(np.mean(jpeg_block)) if jpeg_block else None,
        "laplacian_var_mean": float(np.mean(sharpness)) if sharpness else None,
        "note": "Score alto sugiere más 'ilustración/render' (heurístico, no concluyente).",
    }

    # -------------------------
    # Stratified group split (duplicate-safe)
    # -------------------------
    split = stratified_group_split(
        groups=groups,
        group_labels=group_labels,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )

    # -------------------------
    # Save artifacts
    # -------------------------
    split_path = out_dir / "split_groups.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    report_path = out_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Quick preview file (first 20 items of each split)
    txt_path = out_dir / "split_groups_preview.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for k in ("train", "val", "test"):
            f.write(f"== {k} ==\n")
            for p, y in split[k][:20]:
                f.write(f"{y}\t{p}\n")
            f.write("\n")

    # -------------------------
    # Print summary
    # -------------------------
    print("\nDONE")
    print(f"Report: {report_path}")
    print(f"Split:  {split_path}")
    print(f"Preview:{txt_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
