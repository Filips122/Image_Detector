# =========================
# Imports / Dependencies
# =========================
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================
# Reproducibility utilities
# =========================
def set_seed(seed: int):
    """
    Set random seeds across common libraries so experiments are repeatable.
    This helps ensure that randomness in Python, NumPy, and PyTorch is consistent
    between runs (as much as the environment allows).
    """
    random.seed(seed)         # Python standard RNG
    np.random.seed(seed)      # NumPy RNG
    torch.manual_seed(seed)   # PyTorch RNG (CPU; also affects some GPU ops)


# ==========================================
# Temperature Scaling (logits calibration)
# ==========================================
class TemperatureScaler(nn.Module):
    """
    Learn a positive scalar temperature T > 0 by minimizing NLL (CrossEntropy)
    on a validation set.

    After training:
        calibrated_logits = logits / T

    Intuition:
      - If the model is over-confident, T tends to become > 1, softening logits.
      - If the model is under-confident, T may become < 1, sharpening logits.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()

        # We store log(T) as a parameter instead of T directly:
        # - guarantees positivity when we exponentiate
        # - provides smoother unconstrained optimization
        self.log_t = nn.Parameter(torch.tensor([float(init_temp)]).log())

    def temperature(self) -> torch.Tensor:
        """
        Convert log(T) -> T, enforce a reasonable numeric range.
        - exp() ensures T is positive
        - clamp prevents extreme values that could destabilize division
        """
        return self.log_t.exp().clamp(1e-3, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        Dividing logits by T changes confidence without changing argmax class
        (for T > 0) in a monotonic way.
        """
        return logits / self.temperature()


# ==========================================================
# Collect logits and labels for calibration (generic version)
# ==========================================================
@torch.no_grad()
def collect_logits_and_labels(model, loader, device: torch.device, mode: str):
    """
    Collect logits and labels from a dataloader, typically on validation data,
    so we can fit temperature scaling afterwards.

    mode:
      - "dual": batch contains "spatial" and "frequency" inputs
      - "single": batch contains a single "x" input

    Returns:
      (logits, labels) concatenated across all batches.
    """
    model.eval()  # set evaluation mode (disables dropout, uses running stats in BN)

    all_logits = []
    all_labels = []

    for batch in loader:
        # Always read the label and move it to the target device
        y = batch["label"].to(device)

        if mode == "dual":
            # Dual-input case: the model expects two tensors (spatial + frequency)
            xs = batch["spatial"].to(device)
            xf = batch["frequency"].to(device)

            # Model returns: fused_logits, spatial_logits, freq_logits
            # Here we only keep the fused logits for calibration
            fused_logits, _, _ = model(xs, xf)
            logits = fused_logits
        else:
            # Single-input case: the model expects one tensor "x"
            x = batch["x"].to(device)
            logits = model(x)

        # Detach to break gradient graph (extra safe) and move to CPU for storage.
        # Using .cpu() avoids keeping GPU memory occupied by collected outputs.
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    # Concatenate all batches along dimension 0 (the batch dimension).
    # Shapes typically:
    #   logits: [N, num_classes]
    #   labels: [N]
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ==========================================================
# Collect logits and labels (dual version with all heads)
# ==========================================================
@torch.no_grad()
def collect_dual_logits_and_labels(model, loader, device: torch.device):
    """
    Collect (fused_logits, spatial_logits, freq_logits, labels) from validation.

    Useful if you want to calibrate different temperatures for each head:
      - fused head temperature
      - spatial head temperature
      - frequency head temperature

    Returns:
      fused_logits_cpu, spatial_logits_cpu, freq_logits_cpu, labels_cpu
    """
    model.eval()

    fused_all, sp_all, fr_all, y_all = [], [], [], []

    for batch in loader:
        y = batch["label"].to(device)
        xs = batch["spatial"].to(device)
        xf = batch["frequency"].to(device)

        # Forward pass produces three different logits outputs
        fused_logits, spatial_logits, freq_logits = model(xs, xf)

        fused_all.append(fused_logits.detach().cpu())
        sp_all.append(spatial_logits.detach().cpu())
        fr_all.append(freq_logits.detach().cpu())
        y_all.append(y.detach().cpu())

    # Return CPU tensors concatenated across all validation batches
    return (
        torch.cat(fused_all, dim=0),
        torch.cat(sp_all, dim=0),
        torch.cat(fr_all, dim=0),
        torch.cat(y_all, dim=0),
    )


# ==========================================
# Fit temperature with LBFGS optimization
# ==========================================
def fit_temperature(logits_cpu: torch.Tensor, labels_cpu: torch.Tensor, max_iter: int = 50) -> float:
    """
    Fit temperature scaling on CPU using LBFGS (a second-order optimizer).

    Why CPU?
      - This step is usually small (one scalar parameter) and stable on CPU.
      - Keeps GPU free and avoids device juggling if validation outputs are on CPU.

    Returns:
      temperature (float)
    """
    device = torch.device("cpu")

    # Ensure both are on CPU (inputs are already named logits_cpu/labels_cpu, but enforced)
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)

    scaler = TemperatureScaler(init_temp=1.0).to(device)
    nll = nn.CrossEntropyLoss()

    # LBFGS in PyTorch requires a "closure" function that:
    # - recomputes the loss
    # - runs backward()
    # because LBFGS may evaluate the objective multiple times per step.
    optimizer = torch.optim.LBFGS([scaler.log_t], lr=0.5, max_iter=max_iter)

    def closure():
        optimizer.zero_grad(set_to_none=True)  # reset gradients (set_to_none is slightly more efficient)
        loss = nll(scaler(logits), labels)     # CrossEntropy on calibrated logits
        loss.backward()                        # compute gradient wrt log_t
        return loss

    optimizer.step(closure)

    # Extract the learned temperature scalar as a Python float
    return float(scaler.temperature().item())


# ==============================
# NEW: training helper utilities
# ==============================

# ------------------------------------
# Running average tracker (loss/metric)
# ------------------------------------
class AverageMeter:
    """
    Stable incremental average tracker.
    Typical usage: update(loss_value, batch_size), then read meter.avg.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Keep sum and count to compute average accurately
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1):
        # Multiply by n so you can weight by batch size (recommended)
        self.sum += float(v) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        # Guard against division by zero if nothing has been recorded
        if self.count <= 0:
            return 0.0
        return float(self.sum / self.count)


# ------------------------------------
# Gradient clipping (optional)
# ------------------------------------
def clip_gradients(model: nn.Module, max_norm: Optional[float]) -> Optional[float]:
    """
    Clip gradients by global norm to improve training stability.
    If max_norm is None or <= 0, do nothing.

    Returns:
      The total norm (as float) returned by PyTorch, or None if not applied.
    """
    if max_norm is None or max_norm <= 0:
        return None

    # clip_grad_norm_ modifies gradients in-place.
    # It returns the total norm *before* clipping (useful for logging).
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm))


# ------------------------------------
# Learning-rate scheduler builder
# ------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer, cfg_train: dict, steps_per_epoch: int):
    """
    Optional scheduler factory. Default is None so existing configs won't break.

    Supported config format:
      train.scheduler:
        name: "cosine"
        warmup_epochs: 2
        min_lr: 1.0e-6

    Notes:
      - This implementation steps "per epoch" (not per batch).
      - steps_per_epoch is currently unused here, but kept for future extensions.
    """
    sch = cfg_train.get("scheduler", None)
    if not isinstance(sch, dict):
        return None

    name = str(sch.get("name", "")).strip().lower()
    if not name or name == "none":
        return None

    if name == "cosine":
        # Cosine annealing over epochs, with a simple manual warmup.
        warmup_epochs = int(sch.get("warmup_epochs", 0))
        min_lr = float(sch.get("min_lr", 1e-6))

        # Store each param group's initial/base LR, so we can scale from it.
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        def set_lr(mult: float):
            """
            Multiply each base LR by 'mult', but never go below min_lr.
            This is used during warmup.
            """
            for pg, blr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = float(max(min_lr, blr * mult))

        class _CosineWithWarmup:
            """
            Minimal scheduler-like object:
              - keeps track of epoch count
              - updates optimizer LRs when step_epoch() is called

            This is a lightweight alternative to using torch schedulers directly.
            """
            def __init__(self):
                self.epoch = 0
                self.total_epochs = int(cfg_train.get("epochs", 1))

            def step_epoch(self):
                """
                Call this once per epoch.

                Warmup phase (if enabled):
                  - linearly increases LR multiplier from ~0 -> 1 across warmup_epochs

                Cosine phase:
                  - cosine decay from base_lr down toward min_lr across remaining epochs
                """
                self.epoch += 1
                e = self.epoch

                # ---- Warmup (linear) ----
                if warmup_epochs > 0 and e <= warmup_epochs:
                    mult = e / float(max(1, warmup_epochs))
                    set_lr(mult)
                    return

                # ---- Cosine annealing after warmup ----
                # We shift the cosine schedule to start after warmup.
                t0 = warmup_epochs
                t = max(0, e - t0)                      # time since warmup ended
                T = max(1, self.total_epochs - t0)      # duration of cosine phase

                # Standard cosine factor in [0..1]:
                # cos=1 at start, cos~0 at middle, cos=0 at end (depending on exact indexing)
                cos = 0.5 * (1.0 + np.cos(np.pi * (t / T)))

                # Interpolate each LR between min_lr and base_lr using cosine factor
                for pg, blr in zip(optimizer.param_groups, base_lrs):
                    lr = min_lr + (blr - min_lr) * cos
                    pg["lr"] = float(lr)

        return _CosineWithWarmup()

    # Unknown scheduler name => no scheduler
    return None


# ------------------------------------
# Early stopping config (dataclass)
# ------------------------------------
@dataclass
class EarlyStoppingConfig:
    """
    Configuration container for early stopping.
    """
    enabled: bool = True     # enable/disable early stopping
    patience: int = 6        # stop after this many non-improving epochs
    min_epochs: int = 10     # do not allow stopping before this epoch
    min_delta: float = 0.0   # required improvement margin to count as "better"


# ------------------------------------
# Early stopping logic
# ------------------------------------
class EarlyStopping:
    """
    Generic early stopping based on a "score" where higher is better
    (e.g., accuracy, F1, AUROC).

    It tracks:
      - best_score observed so far
      - best_epoch
      - how many consecutive epochs have NOT improved (num_bad)
    """
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best_score = -1e18  # initialize very low so first score improves
        self.best_epoch = -1
        self.num_bad = 0

    def step(self, epoch: int, score: float) -> Tuple[bool, bool]:
        """
        Update state given the current epoch and validation score.

        Returns:
          improved: True if score beat best_score by at least min_delta
          should_stop: True if early stopping criteria are met
        """
        improved = False

        # Improvement check includes min_delta margin.
        # This avoids counting tiny fluctuations as meaningful improvements.
        if score > self.best_score + float(self.cfg.min_delta):
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.num_bad = 0
            improved = True
        else:
            # No improvement => increment "bad epochs" counter
            self.num_bad += 1

        # Stop only if:
        # - early stopping is enabled
        # - we've reached at least min_epochs
        # - we've had >= patience consecutive non-improving epochs
        should_stop = (
            bool(self.cfg.enabled)
            and epoch >= int(self.cfg.min_epochs)
            and self.num_bad >= int(self.cfg.patience)
        )

        return improved, should_stop
