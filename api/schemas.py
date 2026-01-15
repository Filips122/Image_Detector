# =========================
# Imports
# =========================
from typing import Any, Dict, Optional
from pydantic import BaseModel


# =========================
# Notes schema
# =========================
class Notes(BaseModel):
    """
    Additional, structured metadata returned together with a detection result.

    This object is meant to be *explanatory* rather than strictly required
    for the main prediction. It helps with:
      - debugging
      - auditing decisions
      - exposing calibration / threshold logic to the user
    """

    # -------------------------
    # Per-branch scores (optional)
    # -------------------------
    # These may come from auxiliary heads in a dual-stream model.
    # They are optional because:
    #   - not all models have spatial/frequency branches
    #   - not all inference pipelines compute them
    spatial_score: Optional[float] = None
    frequency_score: Optional[float] = None

    # -------------------------
    # Temperature scaling info
    # -------------------------
    # Dictionary because there may be multiple temperatures:
    #   e.g. {"main": T, "spatial": Ts, "frequency": Tf}
    # or a single {"main": T}.
    temperature: Dict[str, Any]

    # -------------------------
    # Decision thresholds
    # -------------------------
    # decision_threshold:
    #   The threshold actually used to make the decision.
    #   This is often the "safe threshold" optimized for high precision.
    decision_threshold: Optional[float] = None

    # decision_threshold_default:
    #   Reference threshold (usually 0.5) shown for comparison.
    decision_threshold_default: Optional[float] = None

    # -------------------------
    # Decision outputs
    # -------------------------
    # decision:
    #   Human-readable decision string, e.g. "REAL" / "FAKE".
    decision: Optional[str] = None

    # precision_min:
    #   Minimum precision constraint used when computing a "safe" threshold.
    #   Example: 0.90 means "only decide FAKE if precision >= 90%".
    precision_min: Optional[float] = None

    # decision_binary:
    #   String version of the binary decision (often redundant with `decision`),
    #   kept for clarity or backward compatibility.
    decision_binary: Optional[str] = None

    # -------------------------
    # Extra metrics from checkpoint (optional)
    # -------------------------
    # This can include validation-time metrics saved in the checkpoint, such as:
    #   {
    #     "val_metrics_raw": {...},
    #     "thresholds": {...}
    #   }
    #
    # It is optional because:
    #   - not all checkpoints include it
    #   - inference may not want to expose it in all contexts
    metrics_extra: Optional[Dict[str, Any]] = None


# =========================
# Main detection response schema
# =========================
class DetectResponse(BaseModel):
    """
    Top-level response schema for an AI-vs-REAL detection API.

    This is typically what the inference service returns to a client.
    """

    # Probability that the image is AI-generated (FAKE).
    # Expected range: [0.0, 1.0]
    is_ai_probability: float

    # Probability that the image is REAL.
    # Often computed as: 1 - is_ai_probability
    is_real_probability: float

    # Identifier of the model/checkpoint used for inference.
    # Useful for versioning, reproducibility, and debugging.
    model_version: str

    # Rich, optional metadata explaining how the decision was made.
    notes: Notes
