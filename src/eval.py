# =========================
# Imports / Dependencies
# =========================
import os
import json
import yaml
import time
import torch
from torch.utils.data import DataLoader

# Data split + dataset + transforms
from src.data.split import build_file_list, split_pairs
from src.data.dataset import RealFakeDataset
from src.data.transforms import build_spatial_transform, FFTMultiScale, FFTPatchGrid

# Models for each mode
from src.models.dual_stream import DualStreamNet
from src.models.spatial import SpatialNet
from src.models.frequency import SimpleFreqCNN

# Metrics + reporting utilities
from src.metrics import compute_metrics
from src.reporting import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_reliability_diagram,
    find_threshold_max_recall_at_precision,
)


# =========================
# String helper (safe folder/file names)
# =========================
def _safe_name(s: str) -> str:
    """
    Normalize names for filesystem usage:
      - strip whitespace
      - lowercase
      - replace spaces and '-' with '_'
    """
    s = (s or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


# =========================
# "Latest checkpoint" resolver
# =========================
def _resolve_latest_ckpt(ckpt_root: str, mode: str) -> str:
    """
    Resolve the checkpoint path from a "latest" pointer JSON.

    Expected file:
      ckpt_root/<mode>_latest.json

    Expected JSON payload format:
      {
        "best_checkpoint": "/path/to/ckpt.pt",
        ...
      }

    This allows the eval script to run without passing --ckpt explicitly.
    """
    latest_json = os.path.join(ckpt_root, f"{mode}_latest.json")
    if not os.path.exists(latest_json):
        raise FileNotFoundError(f"No existe {latest_json}. Entrena primero o pasa --ckpt explícito.")

    with open(latest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    path = payload.get("best_checkpoint")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint del latest no válido: {path}")

    return path


# =========================
# Report directory creator
# =========================
def _make_report_dir(cfg: dict, mode: str, backbone: str) -> str:
    """
    Create an evaluation output directory under:
      report_root/evals/<mode>/<backbone>/eval_YYYYMMDD_HHMMSS
    """
    report_root = cfg["output"].get("report_dir", "artifacts/reports")
    out = os.path.join(
        report_root,
        "evals",
        _safe_name(mode),
        _safe_name(backbone),
        f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out, exist_ok=True)
    return out


# =========================
# Load dataset splits (group split preferred)
# =========================
def _load_splits(cfg: dict) -> dict:
    """
    Use split_groups.json if available (group/duplicate-safe splitting).
    Otherwise fall back to the original random split.

    Output:
      splits["train"], splits["val"], splits["test"] as lists of (path, label) pairs.
    """
    split_cfg = cfg["data"]["split"]

    split_json = cfg.get("data", {}).get("group_split_json", "artifacts/audit_split/split_groups.json")
    split_json = str(split_json)

    # Preferred: deterministic group split from disk
    if os.path.exists(split_json):
        print(f"[eval] Using group split from: {split_json}")
        with open(split_json, "r", encoding="utf-8") as f:
            splits = json.load(f)

        # Enforce label type
        for k in ("train", "val", "test"):
            splits[k] = [(p, int(y)) for (p, y) in splits[k]]
        return splits

    # Fallback: build file list and random split
    print("[eval] Group split not found -> using random split")
    pairs = build_file_list(
        root_dir=cfg["data"]["root_dir"],
        real_dir=cfg["data"]["real_dir"],
        fake_dir=cfg["data"]["fake_dir"],
        max_total=int(cfg["data"]["max_images_total"]),
        seed=int(cfg.get("seed", 42)),
    )
    splits = split_pairs(
        pairs,
        float(split_cfg["train"]),
        float(split_cfg["val"]),
        float(split_cfg["test"]),
        seed=int(cfg.get("seed", 42)),
    )
    return splits


# =========================
# Main evaluation routine
# =========================
@torch.no_grad()
def run_eval(mode: str, cfg: dict, ckpt_path: str):
    """
    Evaluate a trained checkpoint on the TEST split.

    Key points:
      - Runs on CPU (device fixed to cpu for reproducibility/portability)
      - Loads checkpoint and *prefers* checkpoint's own config (avoids mismatch)
      - Applies temperature scaling (if stored in ckpt) before computing probabilities
      - Computes metrics at:
          a) threshold = 0.5
          b) threshold = "safe" threshold (either from ckpt or recomputed on test)
      - Saves plots + a JSON report into a new eval report directory
    """
    device = torch.device("cpu")

    # -------------------------
    # Load checkpoint (and possibly override config)
    # -------------------------
    # ✅ Load ckpt first and use its config if present.
    # This prevents config mismatch (e.g., different backbone or FFT settings).
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_ckpt = ckpt.get("config", None)
    if isinstance(cfg_ckpt, dict):
        cfg = cfg_ckpt

    backbone = _safe_name(cfg.get("model", {}).get("backbone", "unknown"))
    image_size = int(cfg["data"]["image_size"])

    # -------------------------
    # Load splits and build evaluation transforms
    # -------------------------
    splits = _load_splits(cfg)
    print(f"[eval] split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # Spatial transform for evaluation: no augmentation
    spatial_eval = build_spatial_transform(image_size, train=False, aug_cfg=cfg.get("augment", {}))

    # -------------------------
    # Build frequency transform + expected channels
    # -------------------------
    use_phase = bool(cfg["model"].get("use_phase", False))
    c_per = 6 if use_phase else 3

    freq_cfg = cfg["data"].get("freq", {})
    if freq_cfg.get("patch_fft", False):
        grid = int(freq_cfg.get("patch_grid", 2))
        freq_t = FFTPatchGrid(image_size=image_size, grid=grid, use_phase=use_phase)

        # Channel calculation: base FFT + per-patch FFTs
        freq_ch = c_per * (1 + grid * grid)
    else:
        scales = freq_cfg.get("scales", [1.0, 0.5, 0.25])
        freq_t = FFTMultiScale(image_size=image_size, scales=scales, use_phase=use_phase)

        # Channel calculation: concatenation across scales
        freq_ch = c_per * len(scales)

    # -------------------------
    # Build dataset + model depending on mode
    # -------------------------
    if mode == "dual":
        ds_test = RealFakeDataset(
            splits["test"],
            spatial_transform=spatial_eval,
            freq_transform=freq_t,
            return_both=True
        )
        model = DualStreamNet(
            spatial_backbone=cfg["model"]["backbone"],
            freq_in_ch=freq_ch,
            embed_dim=int(cfg["model"]["fusion_dim"]),
            num_classes=2,
        )

    elif mode == "spatial":
        ds_test = RealFakeDataset(
            splits["test"],
            spatial_transform=spatial_eval,
            freq_transform=None,
            return_both=False
        )
        # Sequential: feature extractor then linear classifier
        model = torch.nn.Sequential(
            SpatialNet(cfg["model"]["backbone"], out_dim=256, pretrained=False),
            torch.nn.Linear(256, 2),
        )

    elif mode == "frequency":
        ds_test = RealFakeDataset(
            splits["test"],
            spatial_transform=None,
            freq_transform=freq_t,
            return_both=False
        )
        model = torch.nn.Sequential(
            SimpleFreqCNN(in_ch=freq_ch, out_dim=256),
            torch.nn.Linear(256, 2),
        )

    else:
        raise ValueError("mode must be spatial|frequency|dual")

    # -------------------------
    # DataLoader
    # -------------------------
    loader = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0)

    # -------------------------
    # Load model weights and set eval mode
    # -------------------------
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # -------------------------
    # Read temperature scaling from checkpoint
    # -------------------------
    # ckpt["temperature"] may be:
    #   - a float (legacy format)
    #   - a dict: {"main": ..., "spatial": ..., "frequency": ...}
    temp = ckpt.get("temperature", 1.0)
    if isinstance(temp, dict):
        T_main = float(temp.get("main", 1.0))
    else:
        T_main = float(temp)

    # -------------------------
    # Precision constraint and optional "safe threshold" from checkpoint
    # -------------------------
    precision_min = float(cfg["train"].get("precision_min_for_safe_threshold", 0.90))

    # If checkpoint includes thresholds info, prefer it (more consistent with training selection).
    thr_safe_ckpt = None
    if isinstance(ckpt.get("thresholds"), dict):
        precision_min = float(ckpt["thresholds"].get("precision_min", precision_min))
        thr_safe_ckpt = ckpt["thresholds"].get("threshold_safe", None)

    # -------------------------
    # Run inference over the test set
    # -------------------------
    y_true, y_prob = [], []
    for batch in loader:
        labels = batch["label"].to(device)

        if mode == "dual":
            xs = batch["spatial"].to(device)
            xf = batch["frequency"].to(device)
            fused_logits, _, _ = model(xs, xf)
            logits = fused_logits
        else:
            x = batch["x"].to(device)
            logits = model(x)

        # Convert logits to probability of class 1 ("FAKE"), applying temperature scaling:
        #   softmax(logits / T_main)
        # Temperature scaling changes confidence calibration without changing argmax ordering.
        probs = torch.softmax(logits / T_main, dim=1)[:, 1]

        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    # -------------------------
    # Compute safe threshold on TEST (for analysis), but prefer ckpt threshold if available
    # -------------------------
    thr_safe_test, rec_at_pmin, prec_at_safe = find_threshold_max_recall_at_precision(
        y_true, y_prob, precision_min=precision_min
    )

    # If checkpoint stored a "safe threshold", use it; otherwise use the test-derived one.
    # Note: using test-derived threshold can be optimistic (it uses test labels).
    thr_safe_used = float(thr_safe_ckpt) if thr_safe_ckpt is not None else float(thr_safe_test)

    # Metrics at default 0.5 and at safe threshold
    metrics_05 = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics_safe = compute_metrics(y_true, y_prob, threshold=thr_safe_used)

    # -------------------------
    # Reporting (plots + JSON)
    # -------------------------
    report_dir = _make_report_dir(cfg, mode, backbone)

    plot_confusion_matrix(
        y_true, y_prob,
        os.path.join(report_dir, "test_confusion_matrix.png"),
        title="Test Confusion Matrix (safe threshold)",
        threshold=thr_safe_used
    )
    plot_pr_curve(
        y_true, y_prob,
        os.path.join(report_dir, "test_pr_curve.png"),
        title="Test Precision-Recall Curve"
    )

    # Reliability diagram returns ECE (Expected Calibration Error)
    ece = plot_reliability_diagram(
        y_true, y_prob,
        os.path.join(report_dir, "test_reliability.png"),
        title="Test Reliability Diagram"
    )

    # Collect all outputs into one JSON-friendly dict
    out = {
        "mode": mode,
        "backbone": backbone,
        "ckpt_used": ckpt_path,
        "run_id": ckpt.get("run_id"),
        "best_epoch": ckpt.get("best_epoch"),
        "temperature_main": T_main,
        "ece": ece,

        # Threshold selection metadata
        "precision_min": precision_min,
        "threshold_safe_ckpt": thr_safe_ckpt,
        "threshold_safe_test": thr_safe_test,
        "threshold_safe_used": thr_safe_used,
        "recall_at_pmin_test": rec_at_pmin,
        "precision_at_threshold_safe_test": prec_at_safe,

        # Metrics at different operating points
        "metrics_at_0.5": metrics_05,
        "metrics_at_safe": metrics_safe,

        # Where artifacts were saved
        "report_dir": report_dir,
    }

    # Write final evaluation summary
    with open(os.path.join(report_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


# =========================
# CLI Entrypoint
# =========================
def main():
    """
    Command-line interface:
      --config configs/config.yaml
      --mode spatial|frequency|dual
      --ckpt /path/to/checkpoint.pt  (optional)

    If --ckpt is omitted, the script reads ckpt_root/<mode>_latest.json
    and uses the checkpoint pointed by "best_checkpoint".
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mode", choices=["spatial", "frequency", "dual"], default="dual")
    ap.add_argument("--ckpt", default=None, help="Si no se pasa, usa el último entrenado (latest).")
    args = ap.parse_args()

    # Load the YAML config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_root = cfg["output"]["checkpoint_dir"]

    # If no checkpoint provided, resolve "latest"
    ckpt_path = args.ckpt if args.ckpt else _resolve_latest_ckpt(ckpt_root, args.mode)

    # Run evaluation and print a short summary
    m = run_eval(args.mode, cfg, ckpt_path)
    print(f"[TEST] {args.mode}: metrics_at_safe={m['metrics_at_safe']}")
    print(f"[TEST] reports saved in: {m.get('report_dir')}")


# Standard Python "main script" pattern
if __name__ == "__main__":
    main()
