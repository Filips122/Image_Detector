# =========================
# Imports / Setup
# =========================
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

# Use a non-interactive backend so plots can be saved on servers/CI without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn helpers for common evaluation curves/metrics.
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


# =========================
# Filesystem helper
# =========================
def _ensure_dir(path: str):
    """
    Create a directory if it doesn't exist.
    Used before saving plots to disk.
    """
    os.makedirs(path, exist_ok=True)


# =========================
# Calibration metric: ECE (Expected Calibration Error)
# =========================
def compute_ece(y_true: List[int], y_prob: List[float], n_bins: int = 15):
    """
    Compute Expected Calibration Error (ECE) for a binary classifier.

    Inputs:
      y_true: ground-truth labels (0/1)
      y_prob: predicted probability for the positive class (here: "FAKE")
      n_bins: number of confidence bins

    Key idea:
      - Convert probability into "confidence" = max(p, 1-p)
        (how sure the model is about its chosen class)
      - Compare confidence vs actual accuracy within bins
      - ECE = sum over bins: (bin_fraction) * |bin_accuracy - bin_confidence|

    Returns:
      ece (float),
      bin_conf (mean confidence per bin),
      bin_acc  (mean accuracy per bin),
      bin_count (samples per bin)
    """
    # Convert to NumPy arrays with consistent dtypes.
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # Confidence is the probability assigned to the predicted class:
    # - if p >= 0.5 predicted class is 1, confidence is p
    # - if p < 0.5 predicted class is 0, confidence is 1-p
    conf = np.maximum(y_prob, 1.0 - y_prob)

    # Predicted class using threshold 0.5 (binary decision rule)
    pred = (y_prob >= 0.5).astype(np.int64)

    # correct[i] = 1 if prediction matches truth else 0
    correct = (pred == y_true).astype(np.float32)

    # Build bin edges uniformly in [0, 1].
    # Example with n_bins=15 -> 16 edges: 0.0, 0.066..., ..., 1.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # digitize returns bin index in [1..n_bins+?], so we shift by -1 to get [0..n_bins-1].
    # Then clip for safety.
    bin_ids = np.digitize(conf, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    # Prepare arrays to store per-bin stats.
    bin_conf = np.zeros(n_bins, dtype=np.float32)   # average confidence in bin
    bin_acc = np.zeros(n_bins, dtype=np.float32)    # average accuracy in bin
    bin_count = np.zeros(n_bins, dtype=np.int64)    # number of samples in bin

    # Compute per-bin averages.
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        bin_count[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(correct[mask].mean())

    # Weighted average of calibration gaps across bins.
    total = float(len(y_true))
    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] > 0:
            w = bin_count[b] / total  # fraction of samples in this bin
            ece += w * abs(float(bin_acc[b]) - float(bin_conf[b]))

    return float(ece), bin_conf, bin_acc, bin_count


# =========================
# Threshold selection helper (maximize recall subject to precision constraint)
# =========================
def find_threshold_max_recall_at_precision(
    y_true: List[int],
    y_prob: List[float],
    precision_min: float = 0.90,
) -> Tuple[float, float, float]:
    """
    Return (threshold_safe, recall_at_safe, precision_at_safe).

    Criterion:
      - Choose a threshold that maximizes recall
      - Subject to precision >= precision_min

    This is useful for "safe" operation:
      "Only predict FAKE when we can keep precision very high,
       but within that constraint, try to catch as many fakes as possible (recall)."
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # precision_recall_curve returns arrays of precision/recall for many thresholds.
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Important sklearn detail:
    # - precision and recall have length = len(thresholds) + 1
    # - thresholds[i] corresponds to precision[i+1] and recall[i+1]
    if len(thresholds) == 0:
        # Degenerate case: cannot compute thresholds (e.g., all labels same).
        # Fall back to evaluating at 0.5 with manual precision/recall.
        return (
            0.5,
            float(recall_score_fallback(y_true, y_prob, 0.5)),
            float(precision_score_fallback(y_true, y_prob, 0.5)),
        )

    # Align arrays so that each threshold has a matching precision/recall value.
    prec_t = precision[1:]
    rec_t = recall[1:]
    thr_t = thresholds

    # Boolean mask of thresholds that satisfy the precision constraint.
    ok = prec_t >= precision_min

    if not np.any(ok):
        # If no threshold meets precision_min:
        # choose the threshold that yields the maximum precision possible
        # (even if it's below precision_min), and return its recall too.
        j = int(np.argmax(prec_t))
        return float(thr_t[j]), float(rec_t[j]), float(prec_t[j])

    idxs = np.where(ok)[0]

    # Among allowed thresholds, maximize recall.
    best_rec = np.max(rec_t[idxs])
    cand = idxs[rec_t[idxs] == best_rec]

    # Tie-break #1: among those, choose the highest precision.
    best_prec = np.max(prec_t[cand])
    cand2 = cand[prec_t[cand] == best_prec]

    # Tie-break #2: choose the lowest threshold (most permissive),
    # which typically yields more positives if multiple solutions tie.
    j = int(cand2[np.argmin(thr_t[cand2])])

    return float(thr_t[j]), float(rec_t[j]), float(prec_t[j])


# =========================
# Fallback precision/recall implementations (no sklearn dependency at runtime)
# =========================
def recall_score_fallback(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    """
    Compute recall at a fixed threshold:
      recall = TP / (TP + FN)

    Adds a small epsilon to avoid division by zero.
    """
    pred = (y_prob >= thr).astype(np.int64)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-12)


def precision_score_fallback(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    """
    Compute precision at a fixed threshold:
      precision = TP / (TP + FP)

    Adds a small epsilon to avoid division by zero.
    """
    pred = (y_prob >= thr).astype(np.int64)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp + 1e-12)


# =========================
# Plot: Confusion Matrix
# =========================
def plot_confusion_matrix(
    y_true: List[int],
    y_prob: List[float],
    out_path: str,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
    class_names: Tuple[str, str] = ("REAL", "FAKE"),
):
    """
    Save a confusion matrix plot given probabilities.

    Steps:
      1) Convert probabilities to predicted labels using threshold
      2) Compute confusion matrix counts (TN/FP/FN/TP)
      3) Render a heatmap-like image and annotate each cell with its count
    """
    _ensure_dir(os.path.dirname(out_path))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # Convert probabilities to hard predictions with a chosen threshold.
    y_pred = (y_prob >= threshold).astype(np.int64)

    # confusion_matrix with explicit label order ensures consistent layout:
    # label 0 -> REAL, label 1 -> FAKE
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    # Axis labels show the class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    # Add numeric counts inside each cell for readability.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# Plot: Precision-Recall curve
# =========================
def plot_pr_curve(
    y_true: List[int],
    y_prob: List[float],
    out_path: str,
    title: str = "Precision-Recall Curve",
):
    """
    Save a Precision-Recall curve plot.

    Also computes Average Precision (AP) and shows it in the legend.
    """
    _ensure_dir(os.path.dirname(out_path))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # Returns precision/recall points across thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # AP is the area under the PR curve (average precision)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# Plot: Reliability diagram (calibration curve) + return ECE
# =========================
def plot_reliability_diagram(
    y_true: List[int],
    y_prob: List[float],
    out_path: str,
    n_bins: int = 15,
    title: str = "Reliability Diagram (Calibration)",
) -> float:
    """
    Plot a reliability diagram and return the ECE value.

    Reliability diagram:
      - x-axis: predicted confidence (mean confidence per bin)
      - y-axis: empirical accuracy (mean correctness per bin)

    Perfect calibration lies on the diagonal y = x.
    """
    _ensure_dir(os.path.dirname(out_path))

    # Compute ECE and per-bin confidence/accuracy
    ece, bin_conf, bin_acc, bin_count = compute_ece(y_true, y_prob, n_bins=n_bins)

    # Only plot bins that contain at least one sample
    mask = bin_count > 0
    conf = bin_conf[mask]
    acc = bin_acc[mask]

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.plot(conf, acc, marker="o", label=f"ECE={ece:.4f}")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return float(ece)
