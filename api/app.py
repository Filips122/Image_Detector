# =========================
# Imports
# =========================
# api/app.py
import os
import io
import json
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# FastAPI primitives for API + file uploads + HTML UI.
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi import HTTPException

# Pydantic response schema (typed API response)
from api.schemas import DetectResponse

# Preprocessing transforms (spatial RGB + frequency FFT)
from src.data.transforms import build_spatial_transform, FFTMultiScale, FFTPatchGrid

# Models
from src.models.dual_stream import DualStreamNet
from src.models.spatial import SpatialNet
from src.models.frequency import SimpleFreqCNN


# =========================
# FastAPI app
# =========================
app = FastAPI(title="AI Image Detector", version="1.0")

# -----------------------------
# Globals / runtime cache
# -----------------------------
# Default device for runtime. Startup will update it based on CUDA availability.
DEVICE = torch.device("cpu")

# Registry of available models (ONLY trained ones discovered from checkpoints pointers).
MODELS_INDEX: List["ModelSpec"] = []
MODELS_BY_KEY: Dict[str, "ModelSpec"] = {}

# Lazy-loaded models cache: key -> LoadedModel
LOADED: Dict[str, "LoadedModel"] = {}


# =========================
# Data structures
# =========================
@dataclass
class ModelSpec:
    """
    Metadata about a model "option" that exists on disk.

    key: stable identifier for API/UI selection
    mode: dual | spatial | frequency
    backbone: resnet50, convnext_tiny, etc.
    ptr_json: pointer JSON in artifacts/checkpoints/latest/<mode>/<backbone>.json
    ckpt_path: resolved actual checkpoint path (best.pt inside a run dir)
    """
    key: str          # e.g. "dual|convnext_tiny"
    mode: str         # dual/spatial/frequency
    backbone: str     # convnext_tiny/resnet50/swin_t...
    ptr_json: str     # pointer json path
    ckpt_path: str    # resolved best checkpoint path


@dataclass
class LoadedModel:
    """
    Runtime object containing:
      - the loaded torch model
      - the preprocessing transforms
      - calibration temps and thresholds
      - the config stored in the checkpoint (for consistent preprocessing)
    """
    spec: ModelSpec
    cfg: dict
    model: torch.nn.Module
    spatial_t: Any
    freq_t: Any
    temps: Dict[str, float]
    thresholds: Dict[str, Any]
    val_metrics_raw: Dict[str, Any]


# =========================
# Helpers
# =========================
def _safe_name(s: str) -> str:
    """
    Normalize a string to be filesystem/URL-friendly:
      - lowercase
      - spaces and '-' become '_'
    """
    s = (s or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def _resolve_device(cfg: dict) -> torch.device:
    """
    Decide whether to run on CPU or GPU based on config + availability.

    cfg["device"] can be:
      - "cpu"
      - "cuda" / "gpu"
    If user requests cuda but it's not available, fall back to CPU.
    """
    want = str(cfg.get("device", "cpu")).lower()
    if want in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _parse_temps_from_ckpt(ckpt: dict) -> Dict[str, float]:
    """
    Parse temperature scaling values from checkpoint.

    Supported formats:
      - float: single temperature for main logits
      - dict: separate temperatures for {"main", "spatial", "frequency"}

    Returned dict always contains these keys to simplify downstream logic.
    """
    temp = ckpt.get("temperature", 1.0)
    if isinstance(temp, dict):
        return {
            "main": float(temp.get("main", 1.0)),
            "spatial": float(temp.get("spatial", 1.0)),
            "frequency": float(temp.get("frequency", 1.0)),
        }
    t = float(temp)
    return {"main": t, "spatial": 1.0, "frequency": 1.0}


def _extract_thresholds_and_metrics(ckpt: dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pull 'thresholds' and 'val_metrics_raw' from a checkpoint if present.
    These fields are optional, so we normalize to empty dicts when missing.
    """
    thresholds = ckpt.get("thresholds", {}) if isinstance(ckpt.get("thresholds", None), dict) else {}
    val_metrics_raw = ckpt.get("val_metrics_raw", {}) if isinstance(ckpt.get("val_metrics_raw", None), dict) else {}
    return thresholds, val_metrics_raw


def _build_freq_transform_and_channels(cfg: dict):
    """
    Build the frequency-domain transform (FFT-based) and compute how many channels it outputs.

    Why channel counting matters:
      - The frequency CNN (SimpleFreqCNN) expects a fixed number of input channels.
      - FFTMultiScale concatenates multiple spectra => channels scale with number of scales.
      - FFTPatchGrid concatenates global + per-patch spectra => channels scale with grid size.

    use_phase:
      - If True, each RGB channel contributes magnitude + phase => 2 channels per color.
      - If False, only magnitude => 1 channel per color.
    """
    image_size = int(cfg["data"]["image_size"])
    use_phase = bool(cfg["model"].get("use_phase", False))
    c_per = 6 if use_phase else 3  # 3*(mag) or 3*(mag+phase)

    freq_cfg = cfg["data"].get("freq", {})
    if freq_cfg.get("patch_fft", False):
        grid = int(freq_cfg.get("patch_grid", 2))
        freq_t = FFTPatchGrid(image_size=image_size, grid=grid, use_phase=use_phase)
        freq_ch = c_per * (1 + grid * grid)  # global + each grid patch
    else:
        scales = freq_cfg.get("scales", [1.0, 0.5, 0.25])
        freq_t = FFTMultiScale(image_size=image_size, scales=scales, use_phase=use_phase)
        freq_ch = c_per * len(scales)        # one block per scale

    return freq_t, freq_ch


def _read_latest_pointer_json(ptr_json: str) -> str:
    """
    Read a pointer JSON file produced during training:
      artifacts/checkpoints/latest/<mode>/<backbone>.json

    Expected structure:
      {
        "best_checkpoint": "/abs/or/rel/path/to/best.pt",
        ...
      }

    Returns:
      path to checkpoint file (must exist)
    """
    with open(ptr_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    p = payload.get("best_checkpoint")
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Pointer inválido: {ptr_json} -> best_checkpoint={p}")
    return p


def _discover_available_models(ckpt_root: str) -> List[ModelSpec]:
    """
    Discover ONLY models that have an entry in:
      artifacts/checkpoints/latest/<mode>/<backbone>.json

    This guarantees:
      - only trained models appear in the UI
      - you don't accidentally select an untrained mode/backbone
    """
    out: List[ModelSpec] = []
    latest_dir = os.path.join(ckpt_root, "latest")
    if not os.path.isdir(latest_dir):
        return out

    for mode in ("dual", "spatial", "frequency"):
        mode_dir = os.path.join(latest_dir, mode)
        if not os.path.isdir(mode_dir):
            continue

        for fn in os.listdir(mode_dir):
            if not fn.endswith(".json"):
                continue

            backbone = _safe_name(fn[:-5])  # strip .json
            ptr_json = os.path.join(mode_dir, fn)
            try:
                ckpt_path = _read_latest_pointer_json(ptr_json)
            except Exception:
                # If pointer is broken, skip it (won't show in UI)
                continue

            key = f"{mode}|{backbone}"
            out.append(ModelSpec(key=key, mode=mode, backbone=backbone, ptr_json=ptr_json, ckpt_path=ckpt_path))

    # Stable ordering so UI is predictable.
    # convnext is boosted to appear first within each mode.
    def _sort_key(s: ModelSpec):
        convnext_boost = 0 if "convnext" in s.backbone else 1
        return (s.mode, convnext_boost, s.backbone)

    out.sort(key=_sort_key)
    return out


def _ensure_loaded(spec: ModelSpec) -> LoadedModel:
    """
    Lazy-load a model (and its transforms) on first use.

    Why lazy-load?
      - Faster API startup
      - Avoid loading large models that might never be used
      - Cache makes repeated requests fast

    It loads:
      - checkpoint (weights + config)
      - transforms (built from checkpoint config to avoid mismatches)
      - model architecture matching mode/backbone
      - temperature scaling + thresholds + saved validation metrics
    """
    if spec.key in LOADED:
        return LOADED[spec.key]

    ckpt = torch.load(spec.ckpt_path, map_location="cpu")
    cfg_ckpt = ckpt.get("config", None)
    if not isinstance(cfg_ckpt, dict):
        raise RuntimeError(f"Checkpoint sin config válida: {spec.ckpt_path}")

    cfg = cfg_ckpt
    device = _resolve_device(cfg)

    # Build eval transforms from config to match training/eval preprocessing.
    image_size = int(cfg["data"]["image_size"])
    spatial_t = build_spatial_transform(image_size, train=False, aug_cfg=cfg.get("augment", {}))

    freq_t, freq_ch = (None, None)
    if spec.mode in ("dual", "frequency"):
        freq_t, freq_ch = _build_freq_transform_and_channels(cfg)

    # Build correct architecture depending on mode
    if spec.mode == "dual":
        model = DualStreamNet(
            spatial_backbone=cfg["model"]["backbone"],
            freq_in_ch=freq_ch,
            embed_dim=int(cfg["model"]["fusion_dim"]),
            num_classes=2,
            pretrained_spatial=False,  # weights come from checkpoint
        )
    elif spec.mode == "spatial":
        model = torch.nn.Sequential(
            SpatialNet(backbone=cfg["model"]["backbone"], out_dim=256, pretrained=False),
            torch.nn.Linear(256, 2),
        )
    elif spec.mode == "frequency":
        model = torch.nn.Sequential(
            SimpleFreqCNN(in_ch=freq_ch, out_dim=256),
            torch.nn.Linear(256, 2),
        )
    else:
        raise ValueError(f"Modo no soportado: {spec.mode}")

    # Load trained weights, move to device, and switch to inference mode.
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    temps = _parse_temps_from_ckpt(ckpt)
    thresholds, val_metrics_raw = _extract_thresholds_and_metrics(ckpt)

    lm = LoadedModel(
        spec=spec,
        cfg=cfg,
        model=model,
        spatial_t=spatial_t,
        freq_t=freq_t,
        temps=temps,
        thresholds=thresholds,
        val_metrics_raw=val_metrics_raw,
    )
    LOADED[spec.key] = lm
    return lm


def _decision_band_from_pai(p_ai: float) -> str:
    """
    Convert p(AI) into a human-friendly band.

    Current logic:
      - >= 0.8  => "FAKE - High"
      - >= 0.5  => "FAKE - Low"
      - <= 0.2  => "REAL - High"
      - else    => "REAL - Low"

    This is purely presentation logic; it's not the actual thresholded decision.
    """
    if p_ai >= 0.8:
        return "FAKE - High"
    if p_ai >= 0.5:
        return "FAKE - Low"
    if p_ai <= 0.2:
        return "REAL - High"
    return "REAL - Low"


def _thr_safe_or_default(thresholds: Dict[str, Any], default: float = 0.5) -> float:
    """
    Get the 'safe threshold' stored in checkpoint if available, otherwise default to 0.5.

    The safe threshold usually comes from validation:
      maximize recall subject to precision >= precision_min.

    This function is defensive:
      - ensures conversion to float
      - falls back to default on any parsing issue
    """
    thr = thresholds.get("threshold_safe", None) if isinstance(thresholds, dict) else None
    try:
        return float(thr) if thr is not None else float(default)
    except Exception:
        return float(default)


# ============================================================
# Inference for a single model
# ============================================================
@torch.no_grad()
def _predict_one(lm: LoadedModel, img: Image.Image) -> Dict[str, Any]:
    """
    Run inference on a single image with one loaded model.

    Returns a dict containing:
      - p_ai, p_real
      - spatial_score / frequency_score (if available)
      - decision_binary (REAL/FAKE) using safe threshold if present
      - thr_used (threshold actually used)
      - temps (temperature scaling values)

    Key detail:
      logits are divided by temperature BEFORE softmax:
        probs = softmax(logits / T)
    """
    device = _resolve_device(lm.cfg)
    img = img.convert("RGB")

    # -------------------------
    # Dual model: fused + per-branch auxiliary outputs
    # -------------------------
    if lm.spec.mode == "dual":
        xs = lm.spatial_t(img).unsqueeze(0).to(device)
        xf = lm.freq_t(img).unsqueeze(0).to(device)

        fused_logits, spatial_logits, freq_logits = lm.model(xs, xf)

        fused_probs = torch.softmax(fused_logits / float(lm.temps["main"]), dim=1)[0]
        sp_probs = torch.softmax(spatial_logits / float(lm.temps.get("spatial", 1.0)), dim=1)[0]
        fr_probs = torch.softmax(freq_logits / float(lm.temps.get("frequency", 1.0)), dim=1)[0]

        # Convention: class 0 = REAL, class 1 = FAKE (AI)
        p_real = float(fused_probs[0].cpu().item())
        p_ai = float(fused_probs[1].cpu().item())

        # Auxiliary "scores" are the per-stream probability of FAKE
        spatial_score = float(sp_probs[1].cpu().item())
        freq_score = float(fr_probs[1].cpu().item())

    # -------------------------
    # Spatial-only model
    # -------------------------
    elif lm.spec.mode == "spatial":
        xs = lm.spatial_t(img).unsqueeze(0).to(device)
        logits = lm.model(xs)
        probs = torch.softmax(logits / float(lm.temps["main"]), dim=1)[0]

        p_real = float(probs[0].cpu().item())
        p_ai = float(probs[1].cpu().item())

        spatial_score = p_ai
        freq_score = None

    # -------------------------
    # Frequency-only model
    # -------------------------
    elif lm.spec.mode == "frequency":
        xf = lm.freq_t(img).unsqueeze(0).to(device)
        logits = lm.model(xf)
        probs = torch.softmax(logits / float(lm.temps["main"]), dim=1)[0]

        p_real = float(probs[0].cpu().item())
        p_ai = float(probs[1].cpu().item())

        spatial_score = None
        freq_score = p_ai

    else:
        raise RuntimeError("modo no soportado")

    # Use "safe threshold" (from validation) if available; otherwise 0.5
    thr = _thr_safe_or_default(lm.thresholds, default=0.5)
    decision_binary = "FAKE" if p_ai >= thr else "REAL"

    return {
        "key": lm.spec.key,
        "mode": lm.spec.mode,
        "backbone": lm.spec.backbone,
        "p_ai": p_ai,
        "p_real": p_real,
        "spatial_score": spatial_score,
        "frequency_score": freq_score,
        "thr_used": thr,
        "decision_binary": decision_binary,
        "temps": lm.temps,
    }


# ============================================================
# Ensemble logic (combine multiple model predictions)
# ============================================================
def _combine_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple predictions.

    Combine strategy:
      - Probabilities: arithmetic mean across models.
      - Decision: majority vote using each model's decision_binary
                  (which itself used that model's safe threshold).

    Tie-breaking:
      - If tied, prefer a convnext model decision if present.
      - If multiple convnext models, pick the one with largest margin from threshold
        (abs(p_ai - thr_used)).
      - If no convnext, fall back to mean p_ai >= 0.5.
    """
    if not preds:
        raise RuntimeError("No hay predicciones para combinar.")

    # Mean probabilities
    p_ai_mean = float(sum(p["p_ai"] for p in preds) / len(preds))
    p_real_mean = float(sum(p["p_real"] for p in preds) / len(preds))

    # Majority vote on binary decision
    fake_votes = sum(1 for p in preds if p["decision_binary"] == "FAKE")
    real_votes = len(preds) - fake_votes

    if fake_votes > real_votes:
        decision_majority = "FAKE"
        tie = False
    elif real_votes > fake_votes:
        decision_majority = "REAL"
        tie = False
    else:
        tie = True

        # Tie-break 1: use any convnext model(s)
        convnext = [p for p in preds if "convnext" in (p.get("backbone") or "")]
        if convnext:
            # Choose the convnext prediction with highest confidence margin
            def _margin(pp):
                return abs(pp["p_ai"] - pp["thr_used"])

            convnext.sort(key=_margin, reverse=True)
            decision_majority = convnext[0]["decision_binary"]
        else:
            # Tie-break 2: mean at default threshold 0.5
            decision_majority = "FAKE" if p_ai_mean >= 0.5 else "REAL"

    # Average threshold used (informational only; ensemble decision is majority vote)
    thr_mean = float(sum(float(p.get("thr_used", 0.5)) for p in preds) / len(preds))

    return {
        "p_ai": p_ai_mean,
        "p_real": p_real_mean,
        "decision_majority": decision_majority,
        "decision_band": _decision_band_from_pai(p_ai_mean),
        "votes": {"fake": fake_votes, "real": real_votes, "tie": tie, "tie_break": "convnext"},
        "thr_mean": thr_mean,
    }


# -----------------------------
# Startup: discover models
# -----------------------------
@app.on_event("startup")
def _startup():
    """
    On startup:
      - choose DEVICE based on CUDA availability
      - discover all trained models via checkpoint pointers in latest/
      - build lookup dict

    Note:
      This does NOT load actual torch models yet (lazy load happens on demand).
    """
    global DEVICE, MODELS_INDEX, MODELS_BY_KEY

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allow overriding checkpoint directory via env var
    ckpt_root = os.getenv("AI_CHECKPOINT_DIR", "artifacts/checkpoints")

    MODELS_INDEX = _discover_available_models(ckpt_root)
    MODELS_BY_KEY = {m.key: m for m in MODELS_INDEX}

    print(f"[api] startup device={DEVICE} models_index={len(MODELS_INDEX)} (use /ui)")


# ============================================================
# UI routes (fix 404 on / and favicon)
# ============================================================
@app.get("/", include_in_schema=False)
def root():
    # Redirect root to the HTML UI page
    return RedirectResponse(url="/ui")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Avoid favicon 404 spam in logs
    return Response(status_code=204)


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """
    Serve a minimal single-page HTML UI that:
      - lets you choose a model (or ensemble of all)
      - upload an image
      - calls /ui/detect
      - renders a summary and raw JSON
    """
    # Build <option> list dynamically based on discovered models
    options_html = ['<option value="__ALL__">Promedio de TODOS (solo entrenados)</option>']
    for m in MODELS_INDEX:
        label = f"{m.mode} / {m.backbone}"
        options_html.append(f'<option value="{m.key}">{label}</option>')

    # HTML includes a small JS fetch() POST to /ui/detect with multipart/form-data
    html = f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI Image Detector</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    .card {{ max-width: 880px; border: 1px solid #ddd; border-radius: 12px; padding: 18px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    label {{ display:block; font-weight: 600; margin-bottom: 6px; }}
    input, select, button {{ width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; }}
    button {{ cursor: pointer; font-weight: 700; }}
    pre {{ background: #0b1020; color: #e7e7e7; padding: 12px; border-radius: 12px; overflow:auto; }}
    .muted {{ color:#666; font-size: 13px; }}
    .warn {{ color:#b45309; font-weight:700; }}
    .ok {{ color:#047857; font-weight:700; }}
    .top {{ display:flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="top">
      <h2>AI Image Detector (Local UI)</h2>
      <div class="muted">Modelos disponibles: {len(MODELS_INDEX)}</div>
    </div>

    <p class="muted">
      Esto usa SOLO modelos entrenados (detectados en <code>artifacts/checkpoints/latest</code>).
      Si un modo/backbone no lo entrenaste, no aparece y no influye.
    </p>

    <div class="row">
      <div>
        <label>Elegir modelo</label>
        <select id="modelKey">
          {''.join(options_html)}
        </select>
      </div>
      <div>
        <label>Imagen</label>
        <input id="file" type="file" accept="image/*"/>
      </div>
    </div>

    <div style="margin-top: 14px;">
      <button id="btn">Detectar</button>
    </div>

    <div id="status" class="muted" style="margin-top: 12px;"></div>
    <div id="summary" style="margin-top: 12px;"></div>
    <pre id="out" style="margin-top: 12px; display:none;"></pre>
  </div>

<script>
const btn = document.getElementById("btn");
const out = document.getElementById("out");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");

function pct(x) {{
  if (x === null || x === undefined) return "-";
  return (100.0 * x).toFixed(2) + "%";
}}

btn.onclick = async () => {{
  const f = document.getElementById("file").files[0];
  const key = document.getElementById("modelKey").value;

  if (!f) {{
    statusEl.innerHTML = '<span class="warn">Sube una imagen primero.</span>';
    return;
  }}

  statusEl.textContent = "Procesando...";
  summaryEl.innerHTML = "";
  out.style.display = "none";
  out.textContent = "";

  const form = new FormData();
  form.append("file", f);
  form.append("model_key", key);

  try {{
    const res = await fetch("/ui/detect", {{
      method: "POST",
      body: form
    }});
    const data = await res.json();

    if (!res.ok) {{
      statusEl.innerHTML = '<span class="warn">Error:</span> ' + (data.detail || JSON.stringify(data));
      return;
    }}

    statusEl.innerHTML = '<span class="ok">OK</span>';

    const p_ai = data.is_ai_probability;
    const p_real = data.is_real_probability;
    const notes = data.notes || {{}};

    const vote = (notes.metrics_extra && notes.metrics_extra.combine && notes.metrics_extra.combine.votes)
      ? notes.metrics_extra.combine.votes
      : null;

    const decisionBinary = notes.decision_binary || "-";
    const decisionBand = notes.decision || "-";

    let extra = "";
    if (vote) {{
      extra = `<div class="muted">Votos: FAKE=${{vote.fake}}, REAL=${{vote.real}} (tie=${{vote.tie}}; tie-break=${{vote.tie_break}})</div>`;
    }}

    summaryEl.innerHTML = `
      <div><b>p(AI):</b> ${{pct(p_ai)}} &nbsp; <b>p(REAL):</b> ${{pct(p_real)}}</div>
      <div><b>Decisión:</b> ${{decisionBinary}} &nbsp; <span class="muted">${{decisionBand}}</span></div>
      ${{extra}}
      <div class="muted">Threshold usado: ${{notes.decision_threshold}}</div>
    `;

    out.style.display = "block";
    out.textContent = JSON.stringify(data, null, 2);

  }} catch (e) {{
    statusEl.innerHTML = '<span class="warn">Error:</span> ' + e;
  }}
}};
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ============================================================
# UI endpoint: /ui/detect
# ============================================================
@app.post("/ui/detect", response_model=DetectResponse, include_in_schema=False)
async def ui_detect(
    file: UploadFile = File(...),
    model_key: str = Form("__ALL__"),
):
    """
    Handle a multipart form submission from the UI:
      - file: uploaded image
      - model_key: "__ALL__" or a specific "mode|backbone"

    Returns a DetectResponse (Pydantic schema).
    """
    if not MODELS_INDEX:
        raise HTTPException(
            status_code=400,
            detail="No hay modelos disponibles en artifacts/checkpoints/latest. Entrena al menos 1."
        )

    # Read uploaded file bytes and decode as PIL image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # Choose models to run:
    #  - "__ALL__" => all discovered models (ensemble)
    #  - else => single selected model
    if model_key == "__ALL__":
        specs = MODELS_INDEX[:]  # all available
    else:
        if model_key not in MODELS_BY_KEY:
            raise HTTPException(status_code=400, detail=f"model_key inválido: {model_key}")
        specs = [MODELS_BY_KEY[model_key]]

    # Run inference per model (lazy-loading each model if needed)
    per_model_preds: List[Dict[str, Any]] = []
    for s in specs:
        lm = _ensure_loaded(s)
        per_model_preds.append(_predict_one(lm, img))

    # -------------------------
    # Single-model response
    # -------------------------
    if len(per_model_preds) == 1:
        p0 = per_model_preds[0]
        p_ai = float(p0["p_ai"])
        p_real = float(p0["p_real"])
        spatial_score = p0.get("spatial_score", None)
        freq_score = p0.get("frequency_score", None)

        thr_used = float(p0.get("thr_used", 0.5))
        decision_binary = p0.get("decision_binary", "FAKE" if p_ai >= thr_used else "REAL")
        band = _decision_band_from_pai(p_ai)

        return DetectResponse(
            is_ai_probability=p_ai,
            is_real_probability=p_real,
            model_version=p0["key"],
            notes={
                "spatial_score": spatial_score,
                "frequency_score": freq_score,
                "temperature": p0.get("temps", {}),
                "decision_threshold": thr_used,
                "decision_threshold_default": 0.5,
                "decision": band,
                "precision_min": None,
                "decision_binary": decision_binary,
                "metrics_extra": {
                    "selected": "single",
                    "per_model": per_model_preds,
                },
            },
        )

    # -------------------------
    # Ensemble response
    # -------------------------
    combined = _combine_predictions(per_model_preds)
    p_ai = float(combined["p_ai"])
    p_real = float(combined["p_real"])

    # Optional averaged branch scores across models that provide them
    sp_vals = [p["spatial_score"] for p in per_model_preds if p.get("spatial_score", None) is not None]
    fr_vals = [p["frequency_score"] for p in per_model_preds if p.get("frequency_score", None) is not None]
    spatial_score = float(sum(sp_vals) / len(sp_vals)) if sp_vals else None
    freq_score = float(sum(fr_vals) / len(fr_vals)) if fr_vals else None

    thr_mean = float(combined["thr_mean"])
    decision_binary = combined["decision_majority"]
    band = combined["decision_band"]

    return DetectResponse(
        is_ai_probability=p_ai,
        is_real_probability=p_real,
        model_version="ensemble_all_available" if model_key == "__ALL__" else "ensemble_selected",
        notes={
            "spatial_score": spatial_score,
            "frequency_score": freq_score,
            "temperature": {"ensemble": True},
            "decision_threshold": thr_mean,
            "decision_threshold_default": 0.5,
            "decision": band,
            "precision_min": None,
            "decision_binary": decision_binary,
            "metrics_extra": {
                "selected": "all" if model_key == "__ALL__" else "subset",
                "combine": combined,
                "per_model": per_model_preds,
            },
        },
    )


# ============================================================
# Backward-compatible API endpoint: /detect-image
# ============================================================
@app.post("/detect-image", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Original API endpoint retained for backward compatibility.

    Behavior:
      - always uses ALL available models (ensemble) by default,
        so external callers benefit from improvements without changes.
    """
    if not MODELS_INDEX:
        raise HTTPException(
            status_code=400,
            detail="No hay modelos disponibles en artifacts/checkpoints/latest. Entrena al menos 1."
        )

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    per_model_preds: List[Dict[str, Any]] = []
    for s in MODELS_INDEX:
        lm = _ensure_loaded(s)
        per_model_preds.append(_predict_one(lm, img))

    # Single-model case (only one trained model exists)
    if len(per_model_preds) == 1:
        p0 = per_model_preds[0]
        p_ai = float(p0["p_ai"])
        p_real = float(p0["p_real"])
        thr_used = float(p0.get("thr_used", 0.5))
        decision_binary = p0.get("decision_binary", "FAKE" if p_ai >= thr_used else "REAL")
        band = _decision_band_from_pai(p_ai)

        return DetectResponse(
            is_ai_probability=p_ai,
            is_real_probability=p_real,
            model_version=p0["key"],
            notes={
                "spatial_score": p0.get("spatial_score", None),
                "frequency_score": p0.get("frequency_score", None),
                "temperature": p0.get("temps", {}),
                "decision_threshold": thr_used,
                "decision_threshold_default": 0.5,
                "decision": band,
                "precision_min": None,
                "decision_binary": decision_binary,
                "metrics_extra": {"selected": "single", "per_model": per_model_preds},
            },
        )

    # Ensemble case
    combined = _combine_predictions(per_model_preds)
    p_ai = float(combined["p_ai"])
    p_real = float(combined["p_real"])
    band = combined["decision_band"]

    sp_vals = [p["spatial_score"] for p in per_model_preds if p.get("spatial_score", None) is not None]
    fr_vals = [p["frequency_score"] for p in per_model_preds if p.get("frequency_score", None) is not None]
    spatial_score = float(sum(sp_vals) / len(sp_vals)) if sp_vals else None
    freq_score = float(sum(fr_vals) / len(fr_vals)) if fr_vals else None

    return DetectResponse(
        is_ai_probability=p_ai,
        is_real_probability=p_real,
        model_version="ensemble_all_available",
        notes={
            "spatial_score": spatial_score,
            "frequency_score": freq_score,
            "temperature": {"ensemble": True},
            "decision_threshold": float(combined["thr_mean"]),
            "decision_threshold_default": 0.5,
            "decision": band,
            "precision_min": None,
            "decision_binary": combined["decision_majority"],
            "metrics_extra": {"selected": "all", "combine": combined, "per_model": per_model_preds},
        },
    )
