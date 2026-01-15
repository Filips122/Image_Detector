# =========================
# Imports
# =========================
from typing import List, Tuple, Dict, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset


# =========================
# Dataset: REAL vs FAKE images
# =========================
class RealFakeDataset(Dataset):
    """
    A PyTorch Dataset that loads images and returns tensors for training/evaluation.

    Key features:
      1) Supports *class-specific* spatial transforms:
           - spatial_transform_real (used when label == 0)
           - spatial_transform_fake (used when label == 1)

         If those are not provided, it falls back to `spatial_transform`
         (the older / simpler behavior).

      2) Can return:
           - both spatial + frequency inputs (dual-stream training), or
           - a single input tensor under key "x" (single-stream training)

    Label convention:
      - 0 = REAL
      - 1 = FAKE
    """

    def __init__(
        self,
        pairs: List[Tuple[str, int]],
        spatial_transform=None,
        freq_transform=None,
        return_both: bool = True,
        spatial_transform_real=None,
        spatial_transform_fake=None,
    ):
        """
        Args:
          pairs: list of (image_path, label) tuples.
          spatial_transform: generic spatial transform (used if class-specific not provided).
          freq_transform: frequency transform (FFT, etc.), used in frequency/dual modes.
          return_both: if True -> output contains both "spatial" and "frequency" tensors.
                       if False -> output contains only "x".
          spatial_transform_real: optional transform for REAL images only (label=0).
          spatial_transform_fake: optional transform for FAKE images only (label=1).
        """
        self.pairs = pairs

        # Generic spatial transform (legacy behavior)
        self.spatial_transform = spatial_transform

        # Optional class-specific transforms (override generic when provided)
        self.spatial_transform_real = spatial_transform_real
        self.spatial_transform_fake = spatial_transform_fake

        self.freq_transform = freq_transform
        self.return_both = return_both

    def __len__(self):
        """
        Standard Dataset length: number of samples.
        """
        return len(self.pairs)

    # =========================
    # Helper: choose correct spatial transform based on class
    # =========================
    def _pick_spatial_transform(self, label: int):
        """
        Select which spatial transform to use:
          - if label==0 and spatial_transform_real exists -> use it
          - if label==1 and spatial_transform_fake exists -> use it
          - else -> use the generic spatial_transform
        """
        # Prefer class-specific transforms if provided
        if label == 0 and self.spatial_transform_real is not None:
            return self.spatial_transform_real
        if label == 1 and self.spatial_transform_fake is not None:
            return self.spatial_transform_fake
        return self.spatial_transform

    # =========================
    # Core: load one item
    # =========================
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one image and return a dictionary with:
          Always:
            - "label": torch.long scalar (0 or 1)
            - "path": original file path (string)

          If return_both=True (dual-stream):
            - "spatial": spatial tensor
            - "frequency": frequency tensor

          If return_both=False (single-stream):
            - "x": either spatial tensor or frequency tensor (whichever is available)
        """
        path, label = self.pairs[idx]

        # Load image and force RGB to ensure consistent 3-channel input.
        # (Some images could be grayscale or RGBA otherwise.)
        img = Image.open(path).convert("RGB")

        # Create label tensor as long (required by CrossEntropyLoss)
        y = torch.tensor(int(label), dtype=torch.long)

        # Output always includes label and path for debugging/reporting.
        out = {"label": y, "path": path}

        # -------------------------
        # Dual-stream output (spatial + frequency)
        # -------------------------
        if self.return_both:
            # In dual-stream mode, BOTH transforms are required
            assert self.freq_transform is not None, "freq_transform required when return_both=True"

            st = self._pick_spatial_transform(int(label))
            assert st is not None, "spatial_transform required when return_both=True"

            # Apply transforms to the same PIL image:
            # - st(img) returns spatial tensor (normalized, resized, etc.)
            # - freq_transform(img) returns frequency representation tensor
            out["spatial"] = st(img)
            out["frequency"] = self.freq_transform(img)

        # -------------------------
        # Single-stream output ("x" only)
        # -------------------------
        else:
            # Decide which transform to apply:
            # - prefer spatial if available
            # - otherwise use frequency transform if available
            st = self._pick_spatial_transform(int(label))

            if st is not None:
                out["x"] = st(img)
            elif self.freq_transform is not None:
                out["x"] = self.freq_transform(img)
            else:
                # If neither is provided, the dataset cannot produce input tensors.
                raise ValueError("Need spatial_transform or freq_transform.")

        return out
