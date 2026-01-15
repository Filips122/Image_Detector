# =========================
# Imports
# =========================
from typing import Dict, List, Optional

import numpy as np

# Standard classification metrics from scikit-learn.
# Note: AUC/AP operate on probabilities (scores), not hard labels.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# =========================
# Metrics computation helper
# =========================
def compute_metrics(
    y_true: List[int],
    y_prob_fake: List[float],
    threshold: float = 0.5
) -> Dict[str, Optional[float]]:
    """
    Compute common binary classification metrics for the task:

      - y_true: 0 = REAL, 1 = FAKE
      - y_prob_fake: predicted probability of class FAKE (class 1)
      - threshold: probability threshold used to convert probabilities into hard predictions

    Returns a dictionary containing:
      - accuracy, precision, recall, f1 (always floats)
      - auc, ap (may be None if they cannot be computed)
    """

    # -------------------------
    # Convert inputs to NumPy arrays
    # -------------------------
    # Using NumPy makes downstream operations consistent and fast.
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob_fake, dtype=np.float32)

    # -------------------------
    # Convert probabilities -> predicted class labels
    # -------------------------
    # Binary decision rule:
    #   if P(FAKE) >= threshold -> predict FAKE (1)
    #   else -> predict REAL (0)
    y_pred = (y_prob >= threshold).astype(np.int64)

    # -------------------------
    # Metrics based on hard predictions (y_pred)
    # -------------------------
    # accuracy: fraction of correct predictions
    acc = float(accuracy_score(y_true, y_pred))

    # precision: TP / (TP + FP)
    # zero_division=0 prevents exceptions when the model predicts no positives at all.
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    # recall: TP / (TP + FN)
    # zero_division=0 similarly avoids edge-case crashes.
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # f1: harmonic mean of precision and recall
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # -------------------------
    # Metrics based on probabilities (y_prob)
    # -------------------------
    # ROC AUC measures ranking quality across all possible thresholds.
    # It can fail if y_true contains only one class (all 0s or all 1s).
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    # Average Precision (AP) is essentially PR-AUC style summary.
    # It can also fail in degenerate cases (e.g., only one class present).
    try:
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        ap = None

    # -------------------------
    # Return a structured result
    # -------------------------
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "ap": ap,  # PR-AUC / Average Precision
    }
