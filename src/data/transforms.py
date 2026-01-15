# =========================
# Imports / Dependencies
# =========================
import io
import random
import numpy as np
import torch
import torch.nn.functional as F

import cv2
from PIL import Image, ImageFilter
from torchvision import transforms

# skimage is used here to extract connected components / regions from a binary mask.
from skimage.measure import label, regionprops


# ============================================================
# Anti-overfit augmentations (simulate real-world degradations)
# ============================================================

class RandomJPEGCompression:
    """
    Randomly re-encode an image as JPEG at a random quality.
    This simulates compression artifacts common on social platforms.

    Parameters:
      p: probability of applying the transform
      qmin/qmax: JPEG quality range (lower = more artifacts)
    """
    def __init__(self, p=0.5, qmin=35, qmax=95):
        self.p = p
        self.qmin = qmin
        self.qmax = qmax

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        q = random.randint(self.qmin, self.qmax)

        # Use an in-memory buffer so we don't write temporary files.
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)

        # Reload as PIL Image (forces JPEG artifacts to exist in pixels)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlurPIL:
    """
    Random Gaussian blur to simulate defocus / motion blur / post-processing.
    """
    def __init__(self, p=0.3, rmin=0.2, rmax=1.2):
        self.p = p
        self.rmin = rmin
        self.rmax = rmax

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        r = random.uniform(self.rmin, self.rmax)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomResizeRecompress:
    """
    Simulate "upload then download" degradation:
      - downscale by a random factor
      - upscale back to original size
      - then JPEG recompress

    This approximates what happens when images are resized/compressed by platforms.
    """
    def __init__(self, p=0.5, smin=0.6, smax=1.0, jpeg_qmin=50, jpeg_qmax=95):
        self.p = p
        self.smin = smin
        self.smax = smax
        self.jpeg = RandomJPEGCompression(p=1.0, qmin=jpeg_qmin, qmax=jpeg_qmax)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(self.smin, self.smax)

        # Prevent tiny images (guardrail)
        nw, nh = max(32, int(w * scale)), max(32, int(h * scale))

        # Downscale then upscale (bicubic keeps it smooth but still loses detail)
        img2 = img.resize((nw, nh), resample=Image.BICUBIC)
        img2 = img2.resize((w, h), resample=Image.BICUBIC)

        # Apply JPEG compression (always, because p=1.0 inside self.jpeg)
        img2 = self.jpeg(img2)
        return img2


class RandomAdditiveGaussianNoise:
    """
    Add sensor-like Gaussian noise.

    sigma is defined in the 0..255 pixel scale (because PIL image arrays are 0..255).
    """
    def __init__(self, p=0.2, sigma_min=0.0, sigma_max=8.0):
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        if sigma <= 0:
            return img

        # Convert to float array, add noise, clip back to valid uint8 range.
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


class RandomSharpenPIL:
    """
    Mild sharpening (unsharp mask style).
    Useful to simulate fake post-processing "enhancements" that sometimes appear in generated content.
    """
    def __init__(self, p=0.1, factor_min=0.3, factor_max=1.2):
        self.p = p
        self.factor_min = factor_min
        self.factor_max = factor_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        factor = random.uniform(self.factor_min, self.factor_max)

        # Unsharp mask idea:
        #   out = original + factor * (original - blurred)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        arr = np.array(img).astype(np.float32)
        arr_blur = np.array(blurred).astype(np.float32)
        out = arr + factor * (arr - arr_blur)

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out).convert("RGB")


class RandomDenoiseCV2:
    """
    Mild denoising using OpenCV.
    This can simulate "beautification" or smoothing applied to fake images.

    Warning:
      Strong denoising can remove artifacts that are actually useful for detection,
      so this implementation keeps it soft (small h).
    """
    def __init__(self, p=0.2, h_min=2, h_max=8):
        self.p = p
        self.h_min = h_min
        self.h_max = h_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        h = int(random.uniform(self.h_min, self.h_max))
        arr = np.array(img.convert("RGB"))

        # fastNlMeansDenoisingColored can be somewhat expensive but works well for small strengths.
        try:
            den = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
        except Exception:
            # Fallback: bilateral filter is cheaper and more widely supported.
            den = cv2.bilateralFilter(arr, d=5, sigmaColor=30, sigmaSpace=30)

        return Image.fromarray(den).convert("RGB")


# ============================================================
# Residual ROI crops (focus on "AI leftovers" / residual artifacts)
# ============================================================

class ResidualROICrop:
    """
    Crop a region-of-interest (ROI) based on a residual/high-pass map:

      residual = |gray - GaussianBlur(gray)|

    Intuition:
      - High-pass residual highlights edges/textures.
      - Some fake generation artifacts can appear as localized high-frequency anomalies.
      - This augmentation encourages the model to see "zoomed in" residual-heavy regions.

    Output:
      Returns a PIL Image crop (not resized here).
      In the pipeline, a Resize(...) happens afterwards to standardize shape.
    """
    def __init__(
        self,
        image_size: int = 224,
        p: float = 0.4,
        min_area: int = 200,
        margin: float = 0.15,
        topk: int = 3,
        percentile: float = 92.0,
        blur_ksize: int = 9
    ):
        self.image_size = image_size
        self.p = p
        self.min_area = min_area
        self.margin = margin
        self.topk = topk
        self.percentile = percentile

        # OpenCV GaussianBlur requires an odd kernel size.
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # Work on a standardized base resolution so thresholds/areas are comparable.
        img_rgb = img.convert("RGB")
        base = img_rgb.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        arr = np.array(base)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Residual/high-pass:
        # - blur removes fine details
        # - subtracting highlights fine details / edges / anomalies
        blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        residual = np.abs(gray - blur)

        # Threshold the residual map at a high percentile (keep only strongest residual areas)
        thr = np.percentile(residual, self.percentile)
        mask = (residual >= thr).astype(np.uint8)

        # Clean mask with morphology:
        # - OPEN removes small noise blobs
        # - CLOSE fills small holes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Connected components labeling (skimage)
        lab = label(mask, connectivity=2)
        props = regionprops(lab, intensity_image=residual)

        # Filter out very small regions
        props = [p for p in props if p.area >= self.min_area]
        if not props:
            return img

        # Some skimage versions expose mean intensity under different attribute names.
        def _region_intensity_mean(r):
            if hasattr(r, "intensity_mean"):
                return r.intensity_mean
            if hasattr(r, "mean_intensity"):
                return r.mean_intensity
            return 0.0

        # Sort regions by a combined "importance" score:
        #   score = area * mean_residual_intensity
        # This prefers large + strong residual regions.
        props_sorted = sorted(
            props,
            key=lambda r: (r.area * (_region_intensity_mean(r) + 1e-6)),
            reverse=True
        )

        # Consider top-k best regions (currently picks first one anyway, but keeps option open).
        candidates = props_sorted[:max(1, self.topk)]
        best = candidates[0]

        # regionprops bbox format: (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = best.bbox

        # Expand bbox by a margin to include some context.
        h = self.image_size
        w = self.image_size
        mr = int((maxr - minr) * self.margin)
        mc = int((maxc - minc) * self.margin)

        r0 = max(0, minr - mr)
        c0 = max(0, minc - mc)
        r1 = min(h, maxr + mr)
        c1 = min(w, maxc + mc)

        # Guardrail: avoid producing an extremely tiny crop
        if (r1 - r0) < 16 or (c1 - c0) < 16:
            return img

        # PIL crop uses box=(left, upper, right, lower) => (c0, r0, c1, r1)
        crop = base.crop((c0, r0, c1, r1))
        return crop


# ============================================================
# Spatial transform pipeline builder
# ============================================================

def build_spatial_transform(image_size: int = 224, train: bool = True, aug_cfg: dict | None = None):
    """
    Build the torchvision Compose() transform for spatial images.

    Behavior:
      - If train=True: includes augmentations (JPEG, blur, noise, etc.) + normalization
      - If train=False: NO augmentations (only resize + tensor + normalize)

    Supported extra augmentation params (from aug_cfg):
      - color_jitter_prob, color_jitter_strength
      - noise_prob, noise_sigma_min, noise_sigma_max
      - sharpen_prob, sharpen_factor_min, sharpen_factor_max
      - denoise_prob, denoise_h_min, denoise_h_max
      - recompress_prob, recompress_scale_min/max, jpeg_quality_min/max
      - residual_roi_* settings
    """
    aug_cfg = aug_cfg or {}
    t = []

    if train:
        # -------------------------
        # 1) Realistic degradations
        # -------------------------
        t.extend([
            RandomResizeRecompress(
                p=aug_cfg.get("recompress_prob", 0.5),
                smin=aug_cfg.get("recompress_scale_min", 0.6),
                smax=aug_cfg.get("recompress_scale_max", 1.0),
                jpeg_qmin=aug_cfg.get("jpeg_quality_min", 35),
                jpeg_qmax=aug_cfg.get("jpeg_quality_max", 95),
            ),
            RandomJPEGCompression(
                p=aug_cfg.get("jpeg_prob", 0.5),
                qmin=aug_cfg.get("jpeg_quality_min", 35),
                qmax=aug_cfg.get("jpeg_quality_max", 95),
            ),
            RandomGaussianBlurPIL(
                p=aug_cfg.get("blur_prob", 0.3),
                rmin=aug_cfg.get("blur_radius_min", 0.2),
                rmax=aug_cfg.get("blur_radius_max", 1.2),
            ),
            RandomAdditiveGaussianNoise(
                p=aug_cfg.get("noise_prob", 0.0),
                sigma_min=aug_cfg.get("noise_sigma_min", 0.0),
                sigma_max=aug_cfg.get("noise_sigma_max", 8.0),
            ),
            RandomSharpenPIL(
                p=aug_cfg.get("sharpen_prob", 0.0),
                factor_min=aug_cfg.get("sharpen_factor_min", 0.3),
                factor_max=aug_cfg.get("sharpen_factor_max", 1.2),
            ),
            RandomDenoiseCV2(
                p=aug_cfg.get("denoise_prob", 0.0),
                h_min=aug_cfg.get("denoise_h_min", 2),
                h_max=aug_cfg.get("denoise_h_max", 8),
            ),

            # Horizontal flip is a safe default for many image tasks (unless semantics break).
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        # Optional color jitter (wrapped in RandomApply to make it probabilistic)
        cj_p = float(aug_cfg.get("color_jitter_prob", 0.5))
        cj_s = float(aug_cfg.get("color_jitter_strength", 0.2))
        if cj_p > 0:
            t.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(cj_s, cj_s, cj_s, 0.05)],
                    p=cj_p
                )
            )

        # -------------------------
        # 2) Residual ROI crop (AI leftovers)
        # -------------------------
        # This is appended near the end so it operates after degradations,
        # then Resize() later standardizes output shape.
        t.append(
            ResidualROICrop(
                image_size=image_size,
                p=aug_cfg.get("residual_roi_prob", 0.4),
                min_area=int(aug_cfg.get("residual_roi_min_area", 200)),
                margin=float(aug_cfg.get("residual_roi_margin", 0.15)),
                topk=int(aug_cfg.get("residual_roi_topk", 3)),
            )
        )

    # -------------------------
    # Standardization steps (always applied)
    # -------------------------
    # Resize ensures consistent input dimensions regardless of crop size.
    t.append(transforms.Resize((image_size, image_size), antialias=True))

    # Convert PIL -> torch.FloatTensor in [0, 1] and permute to [C, H, W]
    t.append(transforms.ToTensor())

    # Normalize with ImageNet mean/std (common when using ImageNet-pretrained backbones)
    t.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    return transforms.Compose(t)


# ============================================================
# Frequency representations (FFT feature extraction)
# ============================================================

def _fft_mag(img_arr: np.ndarray, use_phase: bool):
    """
    Compute an FFT-based spectral representation for an RGB image array.

    Input:
      img_arr: float array shaped [H, W, 3], typically in [0, 1]
      use_phase: if True, also include phase channels

    For each RGB channel:
      - compute 2D FFT
      - shift so DC is centered (fftshift)
      - magnitude: log(1 + |F|) to compress dynamic range

    If use_phase:
      - phase = angle(F) in [-pi, pi]
      - normalize phase to [0, 1] for stable learning

    Output:
      spec: stacked channels shaped [Cspec, H, W]
            Cspec = 3 if magnitude only
            Cspec = 6 if magnitude + phase for each RGB channel

    Final step:
      - normalize spec to zero mean / unit std (global normalization).
    """
    chans = []
    for c in range(3):
        x = img_arr[:, :, c]

        # 2D FFT of one channel
        F2 = np.fft.fft2(x)

        # Move the zero-frequency component to the center of the spectrum
        F2 = np.fft.fftshift(F2)

        # Log magnitude spectrum (stabilizes scale)
        mag = np.log1p(np.abs(F2)).astype(np.float32)
        chans.append(mag)

        if use_phase:
            # Phase in [-pi, pi]
            phase = np.angle(F2).astype(np.float32)

            # Normalize to [0, 1] for easier learning
            phase = (phase + np.pi) / (2 * np.pi)
            chans.append(phase)

    # Stack into [C, H, W]
    spec = np.stack(chans, axis=0)

    # Standardize (global) to make scale consistent across images
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec


# ============================================================
# FFT Multi-scale transform
# ============================================================

class FFTMultiScale:
    """
    Compute FFT representations at multiple image scales.

    Steps:
      1) Resize original to image_size x image_size (base)
      2) For each scale s in scales:
           - downsample base to (s*image_size, s*image_size) (min 16x16)
           - compute FFT representation
           - upsample back to image_size x image_size (bilinear)
      3) Concatenate representations along channel dimension

    Output shape:
      [C_total, image_size, image_size]
      where C_total = C_per_scale * num_scales
    """
    def __init__(self, image_size: int = 224, scales=(1.0, 0.5, 0.25), use_phase: bool = False):
        self.image_size = image_size
        self.scales = list(scales)
        self.use_phase = use_phase

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        base = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        specs = []
        for s in self.scales:
            if s == 1.0:
                im_s = base
            else:
                # Guardrail: avoid very small scale leading to near-empty FFT
                sz = max(16, int(self.image_size * s))
                im_s = base.resize((sz, sz), resample=Image.BICUBIC)

            # Convert PIL -> float array in [0, 1]
            arr = np.array(im_s).astype(np.float32) / 255.0

            # Compute spectral representation [Cspec, H, W]
            spec = _fft_mag(arr, self.use_phase)

            # Convert to torch and upsample back to (image_size, image_size)
            # Note: interpolate expects a 4D tensor [N, C, H, W], so we add batch dim.
            spec_t = torch.from_numpy(spec).unsqueeze(0)
            spec_t = F.interpolate(
                spec_t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )
            specs.append(spec_t.squeeze(0))

        # Concatenate per-scale spectra along channel dimension
        out = torch.cat(specs, dim=0)
        return out.float()


# ============================================================
# FFT Patch-grid transform
# ============================================================

class FFTPatchGrid:
    """
    Compute FFT representations for:
      - full image (global spectrum)
      - plus each patch in a grid (local spectra)

    Steps:
      1) Resize image to (image_size, image_size)
      2) Compute FFT for full image -> one spectrum block
      3) Split image into grid x grid patches
      4) Compute FFT for each patch
      5) Upsample each patch spectrum to (image_size, image_size)
      6) Concatenate all spectrum blocks along channel dimension

    Output channels:
      C_total = C_per_block * (1 + grid*grid)
    """
    def __init__(self, image_size: int = 224, grid: int = 2, use_phase: bool = False):
        self.image_size = image_size
        self.grid = grid
        self.use_phase = use_phase

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Ensure consistent size and RGB
        img = img.convert("RGB").resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0

        specs = []

        # Global FFT spectrum (whole image)
        specs.append(torch.from_numpy(_fft_mag(arr, self.use_phase)).float())

        g = self.grid
        h = self.image_size // g
        w = self.image_size // g

        # Local FFT spectra for each patch
        for i in range(g):
            for j in range(g):
                patch = arr[i*h:(i+1)*h, j*w:(j+1)*w, :]

                spec = _fft_mag(patch, self.use_phase)

                # Upsample patch spectrum to match full image resolution for CNN compatibility
                spec_t = torch.from_numpy(spec).unsqueeze(0)
                spec_t = F.interpolate(
                    spec_t,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False
                )
                specs.append(spec_t.squeeze(0).float())

        return torch.cat(specs, dim=0)
