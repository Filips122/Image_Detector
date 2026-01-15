# =========================
# Imports / Dependencies
# =========================
import os
import json
import yaml
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Matplotlib in "Agg" mode means "no GUI backend".
# This is important for training on servers/containers where no display is available.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project-specific utilities: data splitting, datasets, transforms, models, metrics, utils, and reporting.
from src.data.split import build_file_list, split_pairs
from src.data.dataset import RealFakeDataset
from src.data.transforms import build_spatial_transform, FFTMultiScale, FFTPatchGrid
from src.models.dual_stream import DualStreamNet
from src.models.spatial import SpatialNet
from src.models.frequency import SimpleFreqCNN
from src.metrics import compute_metrics
from src.utils import (
    set_seed,
    collect_logits_and_labels,
    collect_dual_logits_and_labels,
    fit_temperature,
    AverageMeter,
    clip_gradients,
    build_scheduler,
    EarlyStopping,
    EarlyStoppingConfig,
)
from src.reporting import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_reliability_diagram,
    find_threshold_max_recall_at_precision,
)


# =========================
# Device / Run naming helpers
# =========================
def _resolve_device(cfg: dict) -> torch.device:
    """
    Decide whether to run on CPU or GPU based on config.
    - cfg["device"] can be "cpu", "cuda", or "gpu"
    - if GPU is requested but not available, falls back to CPU
    """
    want = str(cfg.get("device", "cpu")).lower()
    if want in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _safe_name(s: str) -> str:
    """
    Normalize a string so it is safe for folder/file names:
    - lowercased
    - spaces and '-' replaced by '_'
    """
    s = (s or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def _now_run_id(mode: str, backbone: str, tag: str | None = None) -> str:
    """
    Create a unique run identifier based on:
      - mode (spatial/frequency/dual)
      - backbone (e.g., resnet50)
      - optional tag
      - current timestamp YYYYMMDD_HHMMSS

    NOTE: It formats the string twice: if tag exists, it injects it into the run id.
    """
    base = f"{mode}_{backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    return f"{mode}_{backbone}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}" if tag else base


# =========================
# JSON utilities for saving/loading metadata
# =========================
def _save_json(obj: dict, path: str):
    """
    Save dict -> JSON file. Ensures output directory exists.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str) -> dict | None:
    """
    Safe JSON reader:
    - returns None if file doesn't exist
    - returns None if JSON parsing fails
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================
# Pointer checkpoint helpers (latest/best indirection)
# =========================
def _read_pointer_checkpoint(ptr_json: str) -> str | None:
    """
    Pointer JSON convention:
      { "best_checkpoint": "/path/to/checkpoint.pt", ... }

    This reads the pointer file and returns the checkpoint path if it exists.
    """
    d = _read_json(ptr_json)
    if not d:
        return None
    p = d.get("best_checkpoint")
    if not p or not os.path.exists(p):
        return None
    return p


def _resolve_ckpt_pointer(ckpt_root: str, kind: str, mode: str, backbone: str) -> str | None:
    """
    Resolve a checkpoint pointer path, for example:
      ckpt_root/best/dual/resnet50.json
      ckpt_root/latest/spatial/efficientnet_b0.json

    kind: "best" or "latest"
    """
    mode = _safe_name(mode)
    backbone = _safe_name(backbone)
    ptr = os.path.join(ckpt_root, kind, mode, f"{backbone}.json")
    return _read_pointer_checkpoint(ptr)


def _write_latest_pointer(ckpt_root: str, mode: str, backbone: str, best_path: str, meta: dict):
    """
    Write a "latest" pointer JSON that tells other scripts:
      - where the most recent best checkpoint is
      - extra metadata (AP, thresholds, etc.)

    It writes:
      1) ckpt_root/latest/<mode>/<backbone>.json    (new style)
      2) ckpt_root/<mode>_latest.json              (legacy global pointer)
    """
    mode = _safe_name(mode)
    backbone = _safe_name(backbone)

    latest_dir = os.path.join(ckpt_root, "latest", mode)
    os.makedirs(latest_dir, exist_ok=True)
    latest_path = os.path.join(latest_dir, f"{backbone}.json")

    payload = {
        "best_checkpoint": best_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "backbone": backbone,
        **meta,  # merge extra info (run_id, best_ap, etc.)
    }
    _save_json(payload, latest_path)

    # Legacy pointer (kept for backward compatibility)
    legacy_path = os.path.join(ckpt_root, f"{mode}_latest.json")
    _save_json(payload, legacy_path)


def _maybe_update_best_pointer(ckpt_root: str, mode: str, backbone: str, best_path: str, meta: dict):
    """
    Update the "best" pointer only if the new run is better than the previous best.

    Comparison logic:
      - prefer higher AP
      - if AP ties (within ~1e-12), prefer higher recall_at_pmin
    """
    mode = _safe_name(mode)
    backbone = _safe_name(backbone)

    best_dir = os.path.join(ckpt_root, "best", mode)
    os.makedirs(best_dir, exist_ok=True)
    best_ptr_path = os.path.join(best_dir, f"{backbone}.json")

    new_ap = float(meta.get("best_ap", -1.0))
    new_rec = float(meta.get("best_recall_at_pmin", -1.0))

    old = _read_json(best_ptr_path)
    if old:
        old_ap = float(old.get("best_ap", -1.0))
        old_rec = float(old.get("best_recall_at_pmin", -1.0))

        # If new AP is worse -> do not overwrite.
        # If AP is effectively equal and recall is not better -> do not overwrite.
        if (new_ap < old_ap) or (abs(new_ap - old_ap) < 1e-12 and new_rec <= old_rec):
            return

    payload = {
        "best_checkpoint": best_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "backbone": backbone,
        **meta,
    }
    _save_json(payload, best_ptr_path)


# =========================
# Plotting utilities (loss curves)
# =========================
def _plot_losses(history: dict, out_path: str):
    """
    Plot train/val loss curves and save to disk (no display).
    - train_loss_aug: loss on augmented training batches
    - train_eval_loss: loss on a train subset evaluated WITHOUT augmentation (more comparable to val)
    - val_loss: validation loss
    """
    epochs = history.get("epoch", [])
    train_loss_aug = history.get("train_loss_aug", [])
    train_eval_loss = history.get("train_eval_loss", [])
    val_loss = history.get("val_loss", [])
    if not epochs or not val_loss:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    if train_loss_aug:
        plt.plot(epochs, train_loss_aug, label="train_loss_aug")
    if train_eval_loss:
        plt.plot(epochs, train_eval_loss, label="train_eval_loss (no-aug)")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curves (coherent train/val via train_eval_loss)")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# Data split loading
# =========================
def _load_splits(cfg: dict) -> dict:
    """
    Load dataset splits. Two possible paths:
      1) If group_split_json exists -> use it (reproducible audit split by groups)
      2) Else -> build_file_list + random split_pairs (seeded)

    Output format:
      splits["train"], splits["val"], splits["test"] are lists of (path, label) pairs.
    """
    split_cfg = cfg["data"]["split"]
    split_json = cfg.get("data", {}).get("group_split_json", "artifacts/audit_split/split_groups.json")
    split_json = str(split_json)

    # Preferred: stable split loaded from disk
    if os.path.exists(split_json):
        print(f"[train] Using group split from: {split_json}")
        with open(split_json, "r", encoding="utf-8") as f:
            splits = json.load(f)

        # Ensure label is int (JSON may store as number but we enforce int here)
        for k in ("train", "val", "test"):
            splits[k] = [(p, int(y)) for (p, y) in splits[k]]
        return splits

    # Fallback: create list of (filepath, label) and split randomly
    print("[train] Group split not found -> using random split")
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
# Evaluation: loss + metrics + probabilities
# =========================
@torch.no_grad()
def eval_loss_and_metrics_and_probs(
    model, loader, device, mode: str, criterion: nn.Module, temperature_for_metrics: float = 1.0
):
    """
    Evaluate a model on a loader and return:
      - mean loss
      - computed metrics (accuracy, precision, recall, AUC, AP, etc.)
      - y_true list
      - y_prob list (probability of class "FAKE")

    temperature_for_metrics:
      - used ONLY for probability computation (softmax(logits / T))
      - does not affect loss (loss uses raw logits here)
    """
    model.eval()
    losses = []
    y_true, y_prob = [], []

    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)

        if mode == "dual":
            xs = batch["spatial"].to(device, non_blocking=True)
            xf = batch["frequency"].to(device, non_blocking=True)
            fused_logits, _, _ = model(xs, xf)
            logits = fused_logits
        else:
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)

        # Loss is computed on logits vs labels (CrossEntropy expects raw logits)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        # Convert logits -> probabilities; take [:, 1] as the "fake" class probability.
        # The "/ temperature_for_metrics" allows calibrated probabilities if T != 1.
        probs_fake = torch.softmax(logits / float(temperature_for_metrics), dim=1)[:, 1]
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(probs_fake.detach().cpu().numpy().tolist())

    val_loss = float(np.mean(losses)) if losses else None
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)  # default threshold for metric computation
    return val_loss, metrics, y_true, y_prob


# =========================
# Dual-model specific helpers
# =========================
def _maybe_freeze_spatial_backbone(model, mode: str, freeze: bool):
    """
    For the dual-stream model, optionally freeze the spatial backbone parameters.
    Freezing can stabilize early training or allow the frequency branch to catch up.

    NOTE: This assumes model has: model.spatial.backbone.parameters().
    """
    if mode != "dual":
        return
    for p in model.spatial.backbone.parameters():
        p.requires_grad = (not freeze)


def _init_dual_spatial_from_spatial_ckpt(model_dual: DualStreamNet, spatial_ckpt_path: str, expected_backbone: str):
    """
    Initialize the dual model's spatial branch from a pretrained spatial-only checkpoint.

    Key idea:
      - spatial-only model was likely: nn.Sequential(SpatialNet(...), Linear(...))
      - so state_dict keys may be prefixed (e.g. "0.<...>") for the first module in Sequential

    Steps:
      1) load checkpoint on CPU
      2) sanity check backbone match (avoid mixing incompatible architectures)
      3) extract only keys belonging to the SpatialNet ("0.*")
      4) load those weights into model_dual.spatial
    """
    ckpt = torch.load(spatial_ckpt_path, map_location="cpu")

    cfg_ckpt = ckpt.get("config", {})
    bb_ckpt = _safe_name(cfg_ckpt.get("model", {}).get("backbone", ""))

    # If the checkpoint says it used a backbone, enforce it matches expected_backbone.
    if bb_ckpt and bb_ckpt != _safe_name(expected_backbone):
        raise RuntimeError(f"[train] spatial ckpt backbone mismatch: ckpt={bb_ckpt} expected={expected_backbone}")

    state = ckpt.get("model_state", None)
    if not isinstance(state, dict):
        raise RuntimeError("[train] spatial ckpt inválido: falta model_state")

    # Extract weights for the first module in nn.Sequential (index "0.")
    spatial_state = {}
    for k, v in state.items():
        if k.startswith("0."):
            spatial_state[k[len("0."):]] = v  # drop "0." prefix so it matches model_dual.spatial keys

    # strict=False allows partial load (e.g., classifier head differs)
    missing, unexpected = model_dual.spatial.load_state_dict(spatial_state, strict=False)
    print(f"[train] init dual.spatial from spatial ckpt: {os.path.basename(spatial_ckpt_path)}")
    if missing:
        print(f"[train]   missing_keys={len(missing)}")
    if unexpected:
        print(f"[train]   unexpected_keys={len(unexpected)}")


# =========================
# Frequency transform + channels helper
# =========================
def _build_freq_transform_and_channels(cfg: dict):
    """
    Build the frequency-domain transform and compute expected input channels.

    Logic:
      - use_phase determines whether FFT output includes phase (more channels)
      - choose between:
          a) patch FFT grid (FFTPatchGrid)
          b) multi-scale FFT (FFTMultiScale)

    Returns:
      (freq_transform, freq_input_channels)
    """
    image_size = int(cfg["data"]["image_size"])
    use_phase = bool(cfg["model"].get("use_phase", False))

    # c_per: channels contributed per "FFT representation block"
    # - if use_phase: likely (magnitude + phase) => 6 channels
    # - else magnitude only => 3 channels
    c_per = 6 if use_phase else 3

    freq_cfg = cfg["data"].get("freq", {})
    if freq_cfg.get("patch_fft", False):
        # Patch-grid FFT: represent FFT over patches arranged in grid x grid
        grid = int(freq_cfg.get("patch_grid", 2))
        freq_t = FFTPatchGrid(image_size=image_size, grid=grid, use_phase=use_phase)

        # freq_ch: base + per-patch channels
        # (1 + grid*grid) is likely: global FFT + each patch FFT
        freq_ch = c_per * (1 + grid * grid)
    else:
        # Multi-scale FFT: compute FFT at multiple downsample scales
        scales = freq_cfg.get("scales", [1.0, 0.5, 0.25])
        freq_t = FFTMultiScale(image_size=image_size, scales=scales, use_phase=use_phase)
        freq_ch = c_per * len(scales)

    return freq_t, freq_ch


# =========================
# Model factory
# =========================
def _make_model(mode: str, cfg: dict, freq_ch: int | None):
    """
    Create the model depending on training mode:
      - "dual": DualStreamNet(spatial + frequency fusion)
      - "spatial": SpatialNet backbone + linear classifier
      - "frequency": SimpleFreqCNN + linear classifier
    """
    if mode == "dual":
        return DualStreamNet(
            spatial_backbone=cfg["model"]["backbone"],
            freq_in_ch=int(freq_ch),
            embed_dim=int(cfg["model"]["fusion_dim"]),
            num_classes=2,
            pretrained_spatial=bool(cfg["model"].get("pretrained_spatial", True)),
        )

    if mode == "spatial":
        # nn.Sequential means:
        #  - module 0: feature extractor (SpatialNet)
        #  - module 1: final Linear classifier
        return nn.Sequential(
            SpatialNet(
                backbone=cfg["model"]["backbone"],
                out_dim=256,
                pretrained=bool(cfg["model"].get("pretrained_spatial", True))
            ),
            nn.Linear(256, 2)
        )

    if mode == "frequency":
        return nn.Sequential(
            SimpleFreqCNN(in_ch=int(freq_ch), out_dim=256),
            nn.Linear(256, 2)
        )

    raise ValueError("mode must be spatial|frequency|dual")


# =========================
# Dataset factory
# =========================
def _make_dataset(mode: str, pairs, spatial_t, freq_t):
    """
    Build dataset depending on mode:
      - dual: returns both spatial and frequency tensors
      - spatial: returns only spatial tensor
      - frequency: returns only frequency tensor
    """
    if mode == "dual":
        return RealFakeDataset(pairs, spatial_transform=spatial_t, freq_transform=freq_t, return_both=True)
    if mode == "spatial":
        return RealFakeDataset(pairs, spatial_transform=spatial_t, freq_transform=None, return_both=False)
    if mode == "frequency":
        return RealFakeDataset(pairs, spatial_transform=None, freq_transform=freq_t, return_both=False)
    raise ValueError("mode must be spatial|frequency|dual")


# =========================
# Main training routine (single run)
# =========================
def train_one(mode: str, cfg: dict):
    """
    Train a model for one mode (spatial/frequency/dual) using the config.

    High-level flow:
      1) setup seeds + device + directories
      2) load dataset splits + transforms + dataloaders
      3) build model + optimizer + loss
      4) training loop:
          - train on augmented data
          - evaluate on coherent train subset (no aug)
          - evaluate on val
          - compute selection score
          - early stopping + checkpointing
    """
    # Reproducibility first
    set_seed(int(cfg.get("seed", 42)))

    # Choose device
    device = _resolve_device(cfg)
    print(f"[train] device={device} (cuda_available={torch.cuda.is_available()})")

    # cuDNN benchmark can speed up convs when input sizes are constant
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Naming and output directories
    backbone = _safe_name(cfg.get("model", {}).get("backbone", "unknown"))
    ckpt_root = cfg["output"]["checkpoint_dir"]
    report_root = cfg["output"].get("report_dir", "artifacts/reports")

    run_id = _now_run_id(mode, backbone, cfg.get("output", {}).get("run_tag", None))
    ckpt_run_dir = os.path.join(ckpt_root, "runs", mode, backbone, run_id)
    report_run_dir = os.path.join(report_root, "runs", mode, backbone, run_id)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(report_run_dir, exist_ok=True)

    # Where we store the best model for THIS run
    best_ckpt_path = os.path.join(ckpt_run_dir, "best.pt")

    # Reporting artifacts
    history_path = os.path.join(report_run_dir, "history.json")
    loss_plot_path = os.path.join(report_run_dir, "training_curves.png")

    # "Safe threshold" is defined as threshold achieving max recall while maintaining precision >= precision_min
    precision_min = float(cfg["train"].get("precision_min_for_safe_threshold", 0.90))

    # -------------------------
    # Early stopping setup
    # -------------------------
    es_cfg = cfg["train"].get("early_stopping", {}) or {}
    es = EarlyStopping(
        EarlyStoppingConfig(
            enabled=bool(es_cfg.get("enabled", True)),
            patience=int(es_cfg.get("patience", 6)),
            min_epochs=int(es_cfg.get("min_epochs", 10)),
            # min_delta_score is stored in cfg as "min_delta_score" and mapped here:
            min_delta=float(es_cfg.get("min_delta_score", 0.0)),
        )
    )

    # -------------------------
    # Selection score weights
    # -------------------------
    # This training chooses the "best epoch" using a composite score:
    #   selection_score = AP + score_rec_w * recall_at_pmin - w_loss * val_loss_pen - w_gap * gap_pen
    sel_cfg = cfg["train"].get("selection", {}) or {}
    w_gap = float(sel_cfg.get("gap_weight", 0.35))
    w_loss = float(sel_cfg.get("val_loss_weight", 0.20))
    gap_target = float(sel_cfg.get("gap_target", 0.0))
    score_rec_w = float(sel_cfg.get("recall_weight", 0.10))

    print(f"[train] run_id={run_id}")
    print(f"[train] mode={mode} backbone={backbone}")
    print(f"[train] ckpt_run_dir={ckpt_run_dir}")
    print(f"[train] report_run_dir={report_run_dir}")
    print(f"[train] selection=AP (+{score_rec_w}*rec@pmin) - {w_loss}*val_loss_pen - {w_gap}*gap_pen")

    # -------------------------
    # Load splits
    # -------------------------
    splits = _load_splits(cfg)
    print(f"[train] split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # -------------------------
    # Transforms (spatial + frequency)
    # -------------------------
    image_size = int(cfg["data"]["image_size"])

    # Augmentations can be provided globally or per class (real vs fake)
    aug_cfg = cfg.get("augment", {}) or {}
    aug_real = aug_cfg.get("real", aug_cfg)
    aug_fake = aug_cfg.get("fake", aug_cfg)

    # Training spatial transforms: class-conditional augmentation
    spatial_train_real = build_spatial_transform(image_size, train=True, aug_cfg=aug_real)
    spatial_train_fake = build_spatial_transform(image_size, train=True, aug_cfg=aug_fake)

    # Eval transform: no augmentation (consistent evaluation)
    spatial_eval = build_spatial_transform(image_size, train=False, aug_cfg=aug_cfg)

    # Frequency transform used only for dual/frequency modes
    freq_t, freq_ch = _build_freq_transform_and_channels(cfg) if mode in ("dual", "frequency") else (None, None)

    # -------------------------
    # Datasets
    # -------------------------
    if mode == "dual":
        # For train: use per-class spatial transforms + frequency transform
        ds_train = RealFakeDataset(
            splits["train"],
            spatial_transform=None,
            spatial_transform_real=spatial_train_real,
            spatial_transform_fake=spatial_train_fake,
            freq_transform=freq_t,
            return_both=True
        )
        # For val: use coherent transforms (no aug)
        ds_val = _make_dataset(mode, splits["val"], spatial_eval, freq_t)

    elif mode == "spatial":
        ds_train = RealFakeDataset(
            splits["train"],
            spatial_transform=None,
            spatial_transform_real=spatial_train_real,
            spatial_transform_fake=spatial_train_fake,
            freq_transform=None,
            return_both=False
        )
        ds_val = _make_dataset(mode, splits["val"], spatial_eval, None)

    elif mode == "frequency":
        # Frequency-only mode uses only frequency transform (no spatial)
        ds_train = _make_dataset(mode, splits["train"], None, freq_t)
        ds_val = _make_dataset(mode, splits["val"], None, freq_t)

    else:
        raise ValueError("mode must be spatial|frequency|dual")

    # -------------------------
    # Train-eval subset (coherent train evaluation)
    # -------------------------
    # This subset is used to compute train_eval_loss WITHOUT augmentation,
    # giving a more meaningful "generalization gap" against val_loss.
    train_eval_subset = int(cfg["train"].get("train_eval_subset", 4000))
    if train_eval_subset > 0 and train_eval_subset < len(splits["train"]):
        train_eval_pairs = splits["train"][:train_eval_subset]
    else:
        train_eval_pairs = splits["train"]

    ds_train_eval = _make_dataset(mode, train_eval_pairs, spatial_eval, freq_t)

    # -------------------------
    # DataLoaders
    # -------------------------
    num_workers = int(cfg["train"].get("num_workers", 0))

    # Small usability tweak: if using GPU and num_workers wasn't set, use 2 by default.
    if device.type == "cuda" and num_workers == 0:
        num_workers = 2

    pin_memory = (device.type == "cuda")  # speeds up host->GPU transfers in many cases

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    train_eval_loader = DataLoader(ds_train_eval, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    # -------------------------
    # Model creation (+ optional init from spatial checkpoint)
    # -------------------------
    model = _make_model(mode, cfg, freq_ch)

    # Optional: initialize dual model's spatial branch from a previously trained spatial checkpoint.
    if mode == "dual" and bool(cfg.get("model", {}).get("init_spatial_from_latest", False)):
        spatial_best = _resolve_ckpt_pointer(ckpt_root, kind="best", mode="spatial", backbone=backbone)
        spatial_latest = _resolve_ckpt_pointer(ckpt_root, kind="latest", mode="spatial", backbone=backbone)

        # Prefer "best" if available, otherwise "latest"
        chosen = spatial_best or spatial_latest
        if chosen:
            _init_dual_spatial_from_spatial_ckpt(model, chosen, expected_backbone=backbone)
        else:
            print(f"[train] init_spatial_from_latest=true but no best/latest pointer for spatial/{backbone}. Using ImageNet init.")

    model.to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    # -------------------------
    # Class weighting (optional)
    # -------------------------
    # Useful if dataset is imbalanced: gives different penalties to misclassifying each class.
    class_weights = cfg["train"].get("class_weights", None)
    weight_tensor = None
    if isinstance(class_weights, (list, tuple)) and len(class_weights) == 2:
        try:
            w0 = float(class_weights[0])  # REAL class weight
            w1 = float(class_weights[1])  # FAKE class weight
            weight_tensor = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            print(f"[train] Using class_weights: REAL={w0} FAKE={w1}")
        except Exception:
            weight_tensor = None

    # -------------------------
    # Loss function
    # -------------------------
    # label_smoothing can make training more stable / reduce overconfidence.
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    # Dual mode uses auxiliary losses from each branch (spatial + frequency)
    aux_w = float(cfg["train"].get("aux_loss_weight", 0.3))

    # Temperature scaling (calibration) is done only on improved epochs
    do_temp_scaling = bool(cfg["train"].get("temperature_scaling", True))

    # Save a checkpoint each epoch (debugging/recovery convenience)
    save_every_epoch = bool(cfg["train"].get("save_every_epoch", True))

    # Gradient clipping for stability (<=0 disables)
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))

    # -------------------------
    # Freeze spatial backbone (dual only, optional)
    # -------------------------
    freeze_epochs = int(cfg["train"].get("freeze_spatial_epochs", 0))
    if mode == "dual" and freeze_epochs > 0:
        _maybe_freeze_spatial_backbone(model, mode, freeze=True)
        print(f"[train] Spatial backbone frozen for first {freeze_epochs} epochs")

    # -------------------------
    # Scheduler (optional)
    # -------------------------
    scheduler = build_scheduler(opt, cfg["train"], steps_per_epoch=len(train_loader))

    # -------------------------
    # History container (for logging + plots)
    # -------------------------
    history = {
        "run_id": run_id,
        "mode": mode,
        "backbone": backbone,
        "precision_min_for_safe_threshold": precision_min,
        "ckpt_run_dir": ckpt_run_dir,
        "report_run_dir": report_run_dir,

        # Curves / stats tracked per epoch
        "epoch": [],
        "train_loss_aug": [],
        "train_eval_loss": [],
        "val_loss": [],
        "gap_eval": [],
        "lr": [],

        # Validation metrics
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "val_ap": [],
        "val_ece": [],

        # Threshold selection outputs
        "val_threshold_safe": [],
        "val_recall_at_pmin": [],
        "val_precision_at_threshold_safe": [],

        # Model selection objective
        "selection_score": [],
    }

    # Best tracking variables
    best_epoch = -1
    best_T_main, best_T_spatial, best_T_freq = 1.0, 1.0, 1.0
    best_threshold_safe = 0.5

    # For val_loss_pen calculation (track best val loss seen so far)
    best_val_loss_seen = 1e18

    # =========================
    # Epoch loop
    # =========================
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):

        # Unfreeze spatial backbone after freeze_epochs (dual mode)
        if mode == "dual" and freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _maybe_freeze_spatial_backbone(model, mode, freeze=False)
            print(f"[train] Spatial backbone UNFROZEN at epoch {epoch}")

        # -------------------------
        # Training phase
        # -------------------------
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(
            train_loader,
            desc=f"[{mode}/{backbone}] epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5
        )

        for batch in pbar:
            labels = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if mode == "dual":
                xs = batch["spatial"].to(device, non_blocking=True)
                xf = batch["frequency"].to(device, non_blocking=True)

                # Dual model outputs three logits:
                # - fused_logits: from fusion head (main prediction)
                # - spatial_logits: from spatial branch head
                # - freq_logits: from frequency branch head
                fused_logits, spatial_logits, freq_logits = model(xs, xf)

                # Main loss
                loss_main = criterion(fused_logits, labels)

                # Auxiliary losses encourage each branch to be predictive on its own
                loss_aux_s = criterion(spatial_logits, labels)
                loss_aux_f = criterion(freq_logits, labels)

                # Combine:
                # - main loss always counts fully
                # - auxiliary losses are averaged
                # - aux_w scales the aux contribution
                loss = loss_main + aux_w * 0.5 * (loss_aux_s + loss_aux_f)

            else:
                # Single stream: spatial-only or frequency-only model
                x = batch["x"].to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, labels)

            # Backprop + optional gradient clipping + optimizer step
            loss.backward()
            clip_gradients(model, grad_clip)
            opt.step()

            # Track average loss weighted by batch size
            loss_meter.update(loss.item(), n=int(labels.shape[0]))
            pbar.set_postfix(loss=float(loss_meter.avg))

        train_loss_aug = float(loss_meter.avg)

        # -------------------------
        # Save per-epoch checkpoint (optional)
        # -------------------------
        if save_every_epoch:
            epoch_path = os.path.join(ckpt_run_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "mode": mode,
                    "epoch": epoch,
                    "run_id": run_id,
                    "model_state": model.state_dict(),
                    "config": cfg,
                    # temperatures are placeholders here; best temps saved in best.pt
                    "temperature": {"main": 1.0, "spatial": 1.0, "frequency": 1.0},
                    "report_dir": report_run_dir,
                },
                epoch_path,
            )

        # -------------------------
        # 1) "Coherent" train evaluation loss (no augmentation)
        # -------------------------
        train_eval_loss, _, _, _ = eval_loss_and_metrics_and_probs(
            model=model,
            loader=train_eval_loader,
            device=device,
            mode="dual" if mode == "dual" else "single",
            criterion=criterion,
            temperature_for_metrics=1.0,
        )

        # -------------------------
        # 2) Validation loss + metrics
        # -------------------------
        val_loss, val_metrics, y_val_true, y_val_prob = eval_loss_and_metrics_and_probs(
            model=model,
            loader=val_loader,
            device=device,
            mode="dual" if mode == "dual" else "single",
            criterion=criterion,
            temperature_for_metrics=1.0,
        )

        # -------------------------
        # Threshold selection for "safe" operation
        # -------------------------
        # Find threshold that maximizes recall while enforcing precision >= precision_min
        thr_safe, rec_at_pmin, prec_at_safe = find_threshold_max_recall_at_precision(
            y_val_true, y_val_prob, precision_min=precision_min
        )

        # Generate validation plots (written to report folder)
        plot_confusion_matrix(
            y_val_true, y_val_prob,
            os.path.join(report_run_dir, "val_confusion_matrix.png"),
            title="Validation Confusion Matrix",
            threshold=thr_safe
        )
        plot_pr_curve(
            y_val_true, y_val_prob,
            os.path.join(report_run_dir, "val_pr_curve.png"),
            title="Validation Precision-Recall Curve"
        )

        # Reliability diagram also returns ECE (expected calibration error)
        val_ece = plot_reliability_diagram(
            y_val_true, y_val_prob,
            os.path.join(report_run_dir, "val_reliability.png"),
            title="Validation Reliability Diagram"
        )

        # -------------------------
        # Compute selection score (model selection objective)
        # -------------------------
        ap = float(val_metrics.get("ap") or -1.0)

        # gap: val_loss - train_eval_loss, a proxy for generalization gap (same transforms)
        gap = None
        if (train_eval_loss is not None) and (val_loss is not None):
            gap = float(val_loss - train_eval_loss)

        # Track best val loss ever seen, used for a normalized penalty term
        if val_loss is not None:
            best_val_loss_seen = min(best_val_loss_seen, float(val_loss))

        # val_loss_pen: penalize worsening vs best val loss (relative scale)
        # Example: if best_val_loss_seen=0.2 and current val_loss=0.25 => penalty = (0.25-0.2)/0.2 = 0.25
        val_loss_pen = 0.0
        if val_loss is not None and best_val_loss_seen > 0:
            val_loss_pen = max(0.0, (float(val_loss) - float(best_val_loss_seen)) / float(best_val_loss_seen))

        # gap_pen: penalize if gap exceeds gap_target
        gap_pen = 0.0
        if gap is not None:
            gap_pen = max(0.0, float(gap) - float(gap_target))

        # Final selection score (higher is better)
        selection_score = ap + score_rec_w * float(rec_at_pmin) - w_loss * val_loss_pen - w_gap * gap_pen

        # -------------------------
        # Scheduler step (per epoch)
        # -------------------------
        if scheduler is not None:
            scheduler.step_epoch()

        # Log current LR (first param group)
        lr_now = float(opt.param_groups[0]["lr"])

        # -------------------------
        # Update history + write artifacts
        # -------------------------
        history["epoch"].append(epoch)
        history["train_loss_aug"].append(train_loss_aug)
        history["train_eval_loss"].append(float(train_eval_loss) if train_eval_loss is not None else None)
        history["val_loss"].append(float(val_loss) if val_loss is not None else None)
        history["gap_eval"].append(gap)
        history["lr"].append(lr_now)

        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_ap"].append(ap)
        history["val_ece"].append(val_ece)
        history["val_threshold_safe"].append(thr_safe)
        history["val_recall_at_pmin"].append(rec_at_pmin)
        history["val_precision_at_threshold_safe"].append(prec_at_safe)
        history["selection_score"].append(float(selection_score))

        _save_json(history, history_path)
        _plot_losses(history, loss_plot_path)

        # -------------------------
        # Early stopping decision
        # -------------------------
        improved, should_stop = es.step(epoch, float(selection_score))

        # -------------------------
        # If improved: (re)calibrate temperature + save best checkpoint + update pointers
        # -------------------------
        if improved:
            best_epoch = epoch
            best_threshold_safe = float(thr_safe)

            # Temperature scaling only when the selected score improves (saves time)
            T_main, T_spatial, T_freq = 1.0, 1.0, 1.0
            if do_temp_scaling:
                if mode == "dual":
                    # Collect logits per head on validation set, then fit separate temperatures
                    fused_logits_cpu, sp_logits_cpu, fr_logits_cpu, y_cpu = collect_dual_logits_and_labels(
                        model, val_loader, device=device
                    )
                    T_main = fit_temperature(fused_logits_cpu, y_cpu, max_iter=50)
                    T_spatial = fit_temperature(sp_logits_cpu, y_cpu, max_iter=50)
                    T_freq = fit_temperature(fr_logits_cpu, y_cpu, max_iter=50)
                else:
                    # Single head temperature calibration
                    val_logits, val_labels = collect_logits_and_labels(
                        model, val_loader, device=device, mode="single"
                    )
                    T_main = fit_temperature(val_logits, val_labels, max_iter=50)

            best_T_main, best_T_spatial, best_T_freq = float(T_main), float(T_spatial), float(T_freq)

            # Save best checkpoint for this run
            torch.save(
                {
                    "mode": mode,
                    "run_id": run_id,
                    "best_epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "config": cfg,

                    # Store learned temperatures for calibration at inference time
                    "temperature": {"main": best_T_main, "spatial": best_T_spatial, "frequency": best_T_freq},

                    # Keep raw val metrics for convenience
                    "val_metrics_raw": val_metrics,
                    "report_dir": report_run_dir,

                    # Store how selection was computed (for audit/debug)
                    "selection": {
                        "primary": "selection_score",
                        "details": {
                            "ap": ap,
                            "recall_at_pmin": rec_at_pmin,
                            "val_loss_pen": val_loss_pen,
                            "gap_pen": gap_pen,
                            "weights": {"recall_weight": score_rec_w, "val_loss_weight": w_loss, "gap_weight": w_gap},
                        },
                    },

                    # Store thresholding info for "safe" decision boundary
                    "thresholds": {
                        "precision_min": precision_min,
                        "threshold_safe": best_threshold_safe,
                        "recall_at_pmin": rec_at_pmin,
                        "precision_at_threshold_safe": prec_at_safe,
                    },
                },
                best_ckpt_path,
            )

            # Metadata written into pointer JSON files
            meta = {
                "run_id": run_id,
                "best_epoch": best_epoch,
                "best_ap": ap,
                "best_recall_at_pmin": rec_at_pmin,
                "precision_min": precision_min,
                "threshold_safe": best_threshold_safe,
                "selection_score": float(selection_score),
                "report_dir": report_run_dir,
            }

            # Update "latest" pointer (always) and "best" pointer (only if globally best)
            _write_latest_pointer(ckpt_root, mode, backbone, best_ckpt_path, meta)
            _maybe_update_best_pointer(ckpt_root, mode, backbone, best_ckpt_path, meta)

        # -------------------------
        # Stop if early stopping triggers
        # -------------------------
        if should_stop:
            print(f"[train] EARLY STOPPING at epoch={epoch} (best_epoch={es.best_epoch}, best_score={es.best_score:.6f})")
            break


# =========================
# CLI entrypoint
# =========================
def main():
    """
    Parse command line arguments:
      --config path/to/config.yaml
      --mode spatial|frequency|dual

    Then load YAML config and run training.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--mode", type=str, choices=["spatial", "frequency", "dual"], default="dual")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_one(args.mode, cfg)


# Standard Python "main script" pattern
if __name__ == "__main__":
    main()
