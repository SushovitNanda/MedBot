# Reinstall PyTorch with CUDA (fixes CPU-only torch from PyPI)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$pip = Join-Path $Root ".venv\Scripts\pip.exe"
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $pip)) {
    Write-Host "Create .venv first: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Removing existing PyTorch packages..."
& $pip uninstall -y torch torchvision torchaudio 2>$null

Write-Host "Installing CUDA 12.4 PyTorch from requirements.txt (~2.5 GB download)..."
& $pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

& $py -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
