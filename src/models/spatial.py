# =========================
# Imports
# =========================
import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# Pretrained-weights resolver
# =========================
def _get_weights(backbone: str, pretrained: bool):
    """
    Return the torchvision "Weights enum" for a given backbone, or None.

    Why this exists:
      - torchvision model constructors accept a `weights=` argument.
      - Each architecture has its own enum, e.g. ResNet50_Weights.DEFAULT.
      - If pretrained=False, we explicitly return None (random init).

    NOTE:
      This function currently supports only the backbones listed below.
      For any other backbone name, it raises a ValueError.
    """
    if not pretrained:
        return None

    b = backbone.lower()

    # -------- ResNet family --------
    if b == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    if b == "resnet50":
        return models.ResNet50_Weights.DEFAULT

    # -------- ConvNeXt --------
    if b == "convnext_tiny":
        return models.ConvNeXt_Tiny_Weights.DEFAULT

    # -------- EfficientNet --------
    if b == "efficientnet_b2":
        return models.EfficientNet_B2_Weights.DEFAULT

    # -------- Swin Transformer --------
    if b == "swin_t":
        return models.Swin_T_Weights.DEFAULT
    if b == "swin_s":
        return models.Swin_S_Weights.DEFAULT
    if b == "swin_b":
        return models.Swin_B_Weights.DEFAULT

    # If we get here, pretrained=True was requested but we don't know which weights to use.
    raise ValueError(f"pretrained weights no definidos para backbone={backbone}")


# =========================
# Backbone factory (remove classification head)
# =========================
def _build_backbone(backbone: str, pretrained: bool):
    """
    Build a torchvision backbone WITHOUT its classification head.

    Returns:
      (feature_extractor_module, in_features)

    Meaning:
      - feature_extractor_module: outputs a feature vector per image
      - in_features: size of that feature vector, needed to build the projection layer

    Implementation detail:
      Different backbones expose their "final classifier" differently:
        - ResNet: `m.fc`
        - ConvNeXt: `m.classifier[...]`
        - EfficientNet: `m.classifier[...]`
        - Swin: `m.head`
      So we need architecture-specific logic to:
        1) read in_features
        2) replace the head with nn.Identity()
    """
    b = backbone.lower()
    w = _get_weights(b, pretrained)

    # -------- ResNets --------
    if b == "resnet18":
        m = models.resnet18(weights=w)

        # ResNet final classifier: m.fc is a Linear layer.
        # We keep its input dimension as the feature dimension.
        in_features = m.fc.in_features

        # Replace classifier with Identity so forward() returns features instead of logits.
        m.fc = nn.Identity()
        return m, in_features

    if b == "resnet50":
        m = models.resnet50(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_features

    # -------- ConvNeXt --------
    if b == "convnext_tiny":
        m = models.convnext_tiny(weights=w)

        # convnext_tiny classifier is typically:
        #   classifier = [LayerNorm2d, Flatten, Linear]
        # The final Linear is at index 2, so its in_features defines the feature size.
        in_features = m.classifier[2].in_features

        # Remove whole classifier stack -> return pooled features
        m.classifier = nn.Identity()
        return m, in_features

    # -------- EfficientNet --------
    if b == "efficientnet_b2":
        m = models.efficientnet_b2(weights=w)

        # EfficientNet classifier is typically:
        #   classifier = [Dropout, Linear]
        # The Linear is at index 1.
        in_features = m.classifier[1].in_features

        # Remove classifier so the model outputs features
        m.classifier = nn.Identity()
        return m, in_features

    # -------- Swin Transformer --------
    if b in ("swin_t", "swin_s", "swin_b"):
        # Choose correct Swin variant
        if b == "swin_t":
            m = models.swin_t(weights=w)
        elif b == "swin_s":
            m = models.swin_s(weights=w)
        else:
            m = models.swin_b(weights=w)

        # Swin classification head is `m.head` (Linear).
        in_features = m.head.in_features

        # Replace head with Identity so forward() returns a feature vector
        m.head = nn.Identity()
        return m, in_features

    raise ValueError(f"Unknown backbone: {backbone}")


# =========================
# Spatial feature extractor module
# =========================
class SpatialNet(nn.Module):
    """
    Spatial branch network:

        image -> backbone (optionally pretrained) -> projection -> embedding

    Output:
      A dense embedding vector of size `out_dim` per image.
      (This is not logits; a separate classifier can be added on top.)

    Why a projection layer?
      - Different backbones produce different feature sizes.
      - Projection normalizes everything into a common embedding size (out_dim),
        which is convenient for fusion (dual-stream) or a consistent head.
    """
    def __init__(self, backbone: str = "resnet18", out_dim: int = 256, pretrained: bool = False):
        super().__init__()

        self.backbone_name = backbone

        # Build the backbone feature extractor and get feature dimension
        self.backbone, in_features = _build_backbone(backbone, pretrained)

        # Projection block:
        # - Linear: map backbone features -> out_dim
        # - ReLU: introduce non-linearity
        # - Dropout: regularization to reduce overfitting
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1) backbone extracts features
          2) ensure features are flat [B, C]
          3) projection produces final embedding [B, out_dim]
        """
        f = self.backbone(x)

        # Most torchvision backbones (with head removed) return [B, C].
        # If a backbone returns a tensor with extra dimensions (e.g., [B, C, H, W]),
        # flatten it so we can feed it into a Linear layer.
        if f.ndim > 2:
            f = torch.flatten(f, start_dim=1)

        return self.proj(f)
