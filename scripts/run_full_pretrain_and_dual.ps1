# scripts/run_full_pretrain_and_dual.ps1
$ErrorActionPreference = "Stop"

# Ir a la raíz del proyecto (padre de /scripts)
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "=============================================="
Write-Host " AI IMAGE DETECTOR - FULL TRAIN + EVAL PIPELINE"
Write-Host "=============================================="

Write-Host "`n== Activando venv (si existe) =="
$venvActivate = Join-Path (Get-Location) ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "Venv activado: $($env:VIRTUAL_ENV)"
} else {
    Write-Host "WARNING: No se encontró .venv. Continuo sin venv."
}

function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Title,
        [Parameter(Mandatory=$true)][string]$Command
    )

    Write-Host "`n------------------------------------------------"
    Write-Host $Title
    Write-Host "CMD: $Command"
    Write-Host "------------------------------------------------"

    Invoke-Expression $Command

    if ($LASTEXITCODE -ne 0) {
        throw "Falló: $Title"
    }
}

Write-Host "`n========== STAGE 1: SPATIAL PRETRAIN =========="

Invoke-Step -Title "[SPATIAL 1/3] convnext_tiny" `
    -Command "python -m src.train --config configs/config_gpu_spatial_convnext_tiny.yaml --mode spatial"

Invoke-Step -Title "[SPATIAL 2/3] efficientnet_b2" `
    -Command "python -m src.train --config configs/config_gpu_spatial_efficientnet_b2.yaml --mode spatial"

# Si quieres incluir swin también en spatial pretrain (recomendado si lo usas luego)
Invoke-Step -Title "[SPATIAL 3/3] swin_t" `
    -Command "python -m src.train --config configs/config_gpu_spatial_swin_t.yaml --mode spatial"


Write-Host "`n========== STAGE 2: SPATIAL EVAL =========="

Invoke-Step -Title "[EVAL SPATIAL 1/4] convnext_tiny" `
    -Command "python -m src.eval --config configs/config_gpu_spatial_convnext_tiny.yaml --mode spatial"

Invoke-Step -Title "[EVAL SPATIAL 2/3] efficientnet_b2" `
    -Command "python -m src.eval --config configs/config_gpu_spatial_efficientnet_b2.yaml --mode spatial"

Invoke-Step -Title "[EVAL SPATIAL 3/3] swin_t" `
    -Command "python -m src.eval --config configs/config_gpu_spatial_swin_t.yaml --mode spatial"


Write-Host "`n========== STAGE 3: DUAL TRAIN =========="

Invoke-Step -Title "[DUAL 1/3] convnext_tiny" `
    -Command "python -m src.train --config configs/config_gpu_dual_convnext_tiny.yaml --mode dual"

Invoke-Step -Title "[DUAL 2/3] efficientnet_b2" `
    -Command "python -m src.train --config configs/config_gpu_dual_efficientnet_b2.yaml --mode dual"

Invoke-Step -Title "[DUAL 3/3] swin_t" `
    -Command "python -m src.train --config configs/config_gpu_dual_swin_t.yaml --mode dual"


Write-Host "`n========== STAGE 4: DUAL EVAL =========="

Invoke-Step -Title "[EVAL DUAL 1/3] convnext_tiny" `
    -Command "python -m src.eval --config configs/config_gpu_dual_convnext_tiny.yaml --mode dual"

Invoke-Step -Title "[EVAL DUAL 2/3] efficientnet_b2" `
    -Command "python -m src.eval --config configs/config_gpu_dual_efficientnet_b2.yaml --mode dual"

Invoke-Step -Title "[EVAL DUAL 3/3] swin_t" `
    -Command "python -m src.eval --config configs/config_gpu_dual_swin_t.yaml --mode dual"


Write-Host "`n=============================================="
Write-Host " DONE: SPATIAL + DUAL TRAINING + EVAL COMPLETED"
Write-Host "=============================================="
