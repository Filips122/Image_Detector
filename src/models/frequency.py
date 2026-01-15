# =========================
# Imports
# =========================
import torch
import torch.nn as nn


# =========================
# Frequency-domain CNN
# =========================
class SimpleFreqCNN(nn.Module):
    """
    A compact but robust CNN designed for frequency-domain inputs (e.g. FFT maps).

    Design choices explained:
      - Strided convolutions instead of MaxPool:
          * Downsampling is learned, not fixed.
          * Typically introduces less aliasing artifacts in frequency maps.
      - GroupNorm instead of BatchNorm:
          * More stable when batch size is small or varies.
          * Independent of batch statistics.
      - Dropout2d:
          * Randomly drops entire feature channels.
          * Helps reduce spurious high-energy spikes common in frequency data.
    """
    def __init__(self, in_ch: int = 3, out_dim: int = 256):
        """
        Args:
          in_ch: number of input channels (depends on FFT representation).
          out_dim: size of the final embedding vector.
        """
        super().__init__()

        # -------------------------
        # Convolutional building block
        # -------------------------
        def block(cin, cout, stride):
            """
            One downsampling block:
              Conv2d (stride > 1 for downsampling)
              -> GroupNorm
              -> ReLU
              -> Dropout2d
            """
            return nn.Sequential(
                # 3x3 convolution with padding keeps spatial alignment.
                # stride=2 performs learned downsampling.
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False
                ),

                # GroupNorm is batch-size independent.
                # num_groups=8 is a common compromise between instance- and layer-level normalization.
                nn.GroupNorm(num_groups=8, num_channels=cout),

                # In-place ReLU saves memory.
                nn.ReLU(inplace=True),

                # Drop entire feature maps with probability 0.10.
                nn.Dropout2d(p=0.10),
            )

        # -------------------------
        # Feature extractor
        # -------------------------
        # Input resolution example (for comments): 224 x 224
        self.features = nn.Sequential(
            block(in_ch, 32, stride=2),     # 224 -> 112
            block(32, 64, stride=2),        # 112 -> 56
            block(64, 128, stride=2),       # 56 -> 28
            block(128, 256, stride=2),      # 28 -> 14

            # Extra conv without downsampling:
            # increases receptive field and feature mixing at the final scale.
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(inplace=True),

            # Global average pooling:
            # collapses spatial dimensions -> output shape [B, 256, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # -------------------------
        # Projection head
        # -------------------------
        self.proj = nn.Sequential(
            # Flatten [B, 256, 1, 1] -> [B, 256]
            nn.Flatten(),

            # Linear projection to a fixed-size embedding
            nn.Linear(256, out_dim),

            # Non-linearity
            nn.ReLU(inplace=True),

            # Dropout for regularization of the embedding
            nn.Dropout(0.25),
        )

    def forward(self, x):
        """
        Forward pass:
          1) Extract hierarchical frequency features
          2) Project to a compact embedding vector
        """
        f = self.features(x)
        return self.proj(f)
