# =========================
# Imports
# =========================
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import torch
from PIL import Image

# Transforms: spatial (RGB) + frequency (FFT representations)
from src.data.transforms import build_spatial_transform, FFTMultiScale, FFTPatchGrid

# Dual-stream model (spatial + frequency)
from src.models.dual_stream import DualStreamNet


# ============================================================
# Helper: build the frequency transform + compute its channel count
# ============================================================
def build_freq_transform(cfg: dict):
    """
    Build the frequency-domain transform specified by the config, and
    return both:
      - the transform object
      - the number of output channels (needed to construct the freq CNN)

    This mirrors the logic used in train/eval:
      - If patch_fft is enabled: FFTPatchGrid
      - Else: FFTMultiScale

    Channel counting:
      - c_per = 3 if using only magnitude (RGB => 3 channels)
      - c_per = 6 if including phase (magnitude + phase per RGB channel)
      - Multi-scale => c_per * num_scales
      - Patch grid  => c_per * (1 + grid*grid)  (global + each patch)
    """
    image_size = int(cfg["data"]["image_size"])
    use_phase = bool(cfg["model"].get("use_phase", False))
    c_per = 6 if use_phase else 3

    freq_cfg = cfg["data"].get("freq", {})
    if freq_cfg.get("patch_fft", False):
        grid = int(freq_cfg.get("patch_grid", 2))
        t = FFTPatchGrid(image_size=image_size, grid=grid, use_phase=use_phase)
        ch = c_per * (1 + grid * grid)
    else:
        scales = freq_cfg.get("scales", [1.0, 0.5, 0.25])
        t = FFTMultiScale(image_size=image_size, scales=scales, use_phase=use_phase)
        ch = c_per * len(scales)

    return t, ch


# ============================================================
# Main script
# ============================================================
@torch.no_grad()
def main():
    """
    Purpose:
      Find REAL images that the trained model incorrectly predicts as FAKE
      (i.e., false positives on REAL class), and save them for inspection.

    What it does:
      1) Load a checkpoint (must contain cfg in ckpt["config"])
      2) Load split_groups.json (group-aware splits)
      3) Select val or test split
      4) Filter only REAL samples (label=0)
      5) Run inference and compute P(FAKE)
      6) Keep items with P(FAKE) >= threshold
      7) Sort by probability (worst mistakes first)
      8) Save top-N images into an output folder + write summary.json
    """
    # -------------------------
    # CLI arguments
    # -------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="si no se pasa, guarda en data/REAL_PROB_(timestamp)")
    ap.add_argument("--split_json", default="artifacts/audit_split/split_groups.json")
    ap.add_argument("--which", choices=["val", "test"], default="val")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--thr", type=float, default=None, help="si no se pasa, usa threshold_safe del ckpt")
    args = ap.parse_args()

    # -------------------------
    # Resolve project root
    # -------------------------
    # project_root is assumed to be the parent folder that contains /scripts, /src, /data, etc.
    project_root = Path(__file__).resolve().parents[1]

    # -------------------------
    # Resolve output directory
    # -------------------------
    # If user doesn't provide --out, create:
    #   data/REAL_PROB_YYYYMMDD_HHMMSS
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = project_root / "data" / f"REAL_PROB_{ts}"
    else:
        # Allow passing a custom path (relative to root or absolute)
        outdir = Path(args.out)
        if not outdir.is_absolute():
            outdir = project_root / outdir

    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load checkpoint + config
    # -------------------------
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # This script assumes the training saved the full YAML config inside the checkpoint.
    cfg = ckpt.get("config", None)
    if not isinstance(cfg, dict):
        raise RuntimeError("El ckpt no contiene 'config'. Reentrena guardando cfg en el ckpt.")

    # -------------------------
    # Load the split file (group-aware split)
    # -------------------------
    split_path = Path(args.split_json)
    if not split_path.is_absolute():
        split_path = project_root / split_path

    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # pairs is a list[(path, label)] for the chosen split
    pairs = [(p, int(y)) for (p, y) in splits[args.which]]

    # -------------------------
    # Build transforms
    # -------------------------
    image_size = int(cfg["data"]["image_size"])

    # Spatial eval transform: no augmentation, just resize + tensor + normalize
    spatial_eval = build_spatial_transform(
        image_size,
        train=False,
        aug_cfg=cfg.get("augment", {})
    )

    # Frequency transform + number of channels
    freq_t, freq_ch = build_freq_transform(cfg)

    # -------------------------
    # Build model + load weights
    # -------------------------
    model = DualStreamNet(
        spatial_backbone=cfg["model"]["backbone"],
        freq_in_ch=freq_ch,
        embed_dim=int(cfg["model"].get("fusion_dim", 256)),
        num_classes=2,
        pretrained_spatial=False,  # important: we load trained weights, no need ImageNet init here
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------
    # Choose threshold for "bad real" detection
    # -------------------------
    thr = args.thr

    # If user didn't pass --thr, try to use the "safe threshold" stored in ckpt.
    if (
        thr is None
        and isinstance(ckpt.get("thresholds"), dict)
        and ckpt["thresholds"].get("threshold_safe") is not None
    ):
        thr = float(ckpt["thresholds"]["threshold_safe"])

    # Fall back to 0.5 if nothing else is available
    if thr is None:
        thr = 0.5

    # -------------------------
    # Filter only REAL items (label == 0)
    # -------------------------
    reals = [(p, y) for (p, y) in pairs if y == 0]

    # -------------------------
    # Inference loop: collect false positives on REAL
    # -------------------------
    bad: List[Tuple[str, float]] = []
    for path, _y in reals:
        img = Image.open(path).convert("RGB")

        # Transforms produce tensors of shape [C, H, W].
        # The model expects a batch dimension, so we unsqueeze(0) to get [1, C, H, W].
        xs = spatial_eval(img).unsqueeze(0)
        xf = freq_t(img).unsqueeze(0)

        logits, _, _ = model(xs, xf)

        # Probability of class 1 ("FAKE") from logits:
        # softmax returns [P(REAL), P(FAKE)]
        prob = torch.softmax(logits, dim=1)[0, 1].item()

        # If the model is too confident it's FAKE, count it as a problematic REAL example.
        if prob >= thr:
            bad.append((path, float(prob)))

    # -------------------------
    # Sort mistakes by "most fake" probability and keep top-N
    # -------------------------
    bad.sort(key=lambda x: x[1], reverse=True)
    bad = bad[: args.max]

    # -------------------------
    # Save the problematic REAL images to disk
    # -------------------------
    # Each file name includes:
    #   - rank (0000, 0001, ...)
    #   - probability (prob0.987)
    #   - original base filename
    for i, (path, prob) in enumerate(bad):
        try:
            img = Image.open(path).convert("RGB")
            dst = outdir / f"{i:04d}_prob{prob:.3f}_{Path(path).name}"
            img.save(str(dst))
        except Exception:
            # If an image fails to save, skip it and continue.
            pass

    # -------------------------
    # Save a JSON summary for reproducibility / review
    # -------------------------
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"thr": thr, "n_bad": len(bad), "items": bad}, f, indent=2)

    print(f"[done] guardados {len(bad)} REAL mal clasificados en: {outdir}")


# Standard "run as script" entrypoint
if __name__ == "__main__":
    main()
