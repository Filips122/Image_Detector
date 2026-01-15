# 🚀 How Everything Works
**This section describes the complete workflow to train, evaluate, and deploy the real vs AI-generated image detector, using a dual-stream model with probability calibration.**

0️⃣ Create and activate a virtual environment (Required)
**From the project root:**
**Recommended (Python 3.11):**
```bash
py -3.11 -m venv .venv
```

**Commands (Linux / macOS):**
```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

**Commands (Windows – PowerShell):**
```bash
python -m venv .venv
```

```bash
powershell -ExecutionPolicy Bypass
```

```bash
.venv\Scripts\Activate.ps1
```

**Important notes:**
🔹 **Once activated, you will see something like (.venv) in the terminal**
🔹 **All subsequent commands must be run with the environment activated**



1️⃣ Install dependencies (Required)
**From the project root:**
**Commands:**
```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```



2️⃣ Train models (baselines + dual-stream)

**Three variants are trained for experimental comparison:**

🔹 **Spatial model (RGB only)**
🔹 **Frequency model (frequency domain only, multi-scale FFT)**
🔹 **Dual-stream model (RGB + frequency, with auxiliary heads and temperature scaling)**

**During training:**
🔹 **Anti-overfitting augmentations are used (blur, JPEG artifacts, resize/recompress)**
🔹 **Temperatures are automatically adjusted:**
  🔹 **T_main (fused model)**
  🔹 **T_spatial (spatial branch)**
  🔹 **T_frequency (frequency branch)**
🔹 **Everything is saved in the corresponding checkpoint**



3️⃣ Evaluate models on test set (with calibration)
**Each model is evaluated on the test set using calibrated probabilities (temperature scaling).**
**The output includes:**
🔹 **Accuracy**
🔹 **F1-score**
🔹 **AUC**
🔹 **Etc.**



4️⃣ Serve the model as a REST API
**Launches a FastAPI-based REST API, automatically loading:**
🔹 **The dual-stream model**
🔹 **The calibrated temperatures (T_main, T_spatial, T_frequency)**
🔹 **Spatial and frequency transforms**


**Command:**
```bash
🔹 uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Once started, the API will be available at:**
**http://localhost:8000**



5️⃣ Quick test of the /detect-image endpoint
**Send a JPG/PNG image and receive calibrated probabilities:**
**Command:**
```bash
curl.exe -X POST "http://localhost:8000/detect-image" `
  -F "file=@{path_img}"
```



ℹ️ Important Notes
🔹 **is_ai_probability is calibrated, so values like 0.87 are interpretable as real probabilities (post–temperature scaling).**
🔹 **spatial_score and frequency_score come from explicitly trained heads.**
🔹 **The pipeline is fully reproducible via configs/.**
🔹 **This project uses a local dataset.**


**Recommended Commands**
```bash
🔹 python scripts/audit_and_split.py --root data --real REAL_224 --fake FAKE_224 --out artifacts/audit_split
```
```bash
🔹 .\scripts\run_full_pretrain_and_dual.ps1
```


**Future work**
```bash
🔹 python scripts/analyze_real_errors.py --ckpt artifacts/checkpoints/runs/dual/swin_t/dual_swin_t_"lastest"/best.pt --which val
```