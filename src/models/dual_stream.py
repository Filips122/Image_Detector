# =========================
# Imports
# =========================
import torch
import torch.nn as nn

# Local modules: one network for the spatial (RGB/image) branch
# and another network for the frequency (FFT-based) branch.
from .spatial import SpatialNet
from .frequency import SimpleFreqCNN


# =========================
# Dual-stream model definition
# =========================
class DualStreamNet(nn.Module):
    """
    Dual-stream classifier that combines two complementary representations:

      1) Spatial branch (image-domain features):
           x_spatial -> SpatialNet -> embedding fs  (shape: [B, D])

      2) Frequency branch (frequency-domain features):
           x_freq -> SimpleFreqCNN -> embedding ff  (shape: [B, D])

    It produces THREE outputs:
      - fused_logits: final prediction after fusing both embeddings
      - spatial_logits: auxiliary prediction using only the spatial embedding
      - freq_logits: auxiliary prediction using only the frequency embedding

    Why auxiliary heads?
      - They encourage each branch to be independently predictive.
      - This can stabilize training and prevent the fusion from relying on only one branch.
      - They allow diagnostics: you can see whether spatial or frequency alone performs well.
    """

    def __init__(
        self,
        spatial_backbone: str = "resnet18",
        freq_in_ch: int = 3,
        embed_dim: int = 256,
        num_classes: int = 2,
        pretrained_spatial: bool = True,
    ):
        """
        Args:
          spatial_backbone: torchvision backbone name used inside SpatialNet.
          freq_in_ch: number of channels in the frequency input representation.
          embed_dim: embedding size produced by BOTH branches (kept same for easy fusion).
          num_classes: number of output classes (2 for REAL/FAKE).
          pretrained_spatial: whether to initialize spatial backbone with torchvision pretrained weights.
        """
        super().__init__()

        # -------------------------
        # Branch networks
        # -------------------------
        # SpatialNet returns a vector embedding fs of shape [B, embed_dim]
        self.spatial = SpatialNet(
            backbone=spatial_backbone,
            out_dim=embed_dim,
            pretrained=pretrained_spatial
        )

        # SimpleFreqCNN returns a vector embedding ff of shape [B, embed_dim]
        self.freq = SimpleFreqCNN(
            in_ch=freq_in_ch,
            out_dim=embed_dim
        )

        # -------------------------
        # Auxiliary per-branch heads
        # -------------------------
        # Each branch produces its own logits, helpful for auxiliary losses.
        # Shape: [B, num_classes]
        self.spatial_head = nn.Linear(embed_dim, num_classes)
        self.freq_head = nn.Linear(embed_dim, num_classes)

        # -------------------------
        # Fusion classifier
        # -------------------------
        # Fusion is simple concatenation:
        #   fused = [fs, ff] -> shape [B, 2*embed_dim]
        #
        # Then a small MLP produces final logits.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # combine both embeddings into a hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                # regularize fusion head to reduce overfitting
            nn.Linear(256, num_classes),    # final logits
        )

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor):
        """
        Forward pass.

        Inputs:
          x_spatial: spatial-domain tensor (e.g., image) shaped like [B, C, H, W]
          x_freq: frequency-domain tensor (e.g., FFT maps) shaped like [B, C', H', W']

        Returns:
          fused_logits, spatial_logits, freq_logits
        """
        # -------------------------
        # 1) Compute embeddings for each branch
        # -------------------------
        fs = self.spatial(x_spatial)  # [B, D] embedding from spatial network
        ff = self.freq(x_freq)        # [B, D] embedding from frequency network

        # -------------------------
        # 2) Compute auxiliary logits (per-branch predictions)
        # -------------------------
        spatial_logits = self.spatial_head(fs)  # [B, num_classes]
        freq_logits = self.freq_head(ff)        # [B, num_classes]

        # -------------------------
        # 3) Fuse embeddings + compute final logits
        # -------------------------
        # Concatenate along feature dimension (dim=1):
        #   fs: [B, D]
        #   ff: [B, D]
        # -> fused: [B, 2D]
        fused = torch.cat([fs, ff], dim=1)

        # Final classifier produces fused logits for REAL/FAKE
        fused_logits = self.classifier(fused)   # [B, num_classes]

        return fused_logits, spatial_logits, freq_logits
