# MedRAG one-click bootstrap - venv, deps, Qdrant, ingest, models, servers
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# Load repo .env into this session (single source of truth)
$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    throw "Missing .env at $envFile"
}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line -match '^([^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $val = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($name, $val, "Process")
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MedRAG Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Virtual environment ───────────────────────────────────────────────────
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[1/7] Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv - install Python 3.11+" }
} else {
    Write-Host "[1/7] Virtual environment found." -ForegroundColor Green
}

$pip = Join-Path $Root ".venv\Scripts\pip.exe"
$python = $venvPython

# ── 2. Python dependencies (PyTorch CUDA via requirements.txt) ─────────────
Write-Host "[2/7] Installing Python packages from requirements.txt..." -ForegroundColor Yellow
& $pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed" }
& $python -c "import torch; print('      PyTorch CUDA:', torch.cuda.is_available())"
Write-Host "      Python packages OK." -ForegroundColor Green

# ── 3. Qdrant (Docker or embedded) ───────────────────────────────────────────
Write-Host "[3/7] Starting Qdrant vector database..." -ForegroundColor Yellow
$env:QDRANT_MODE = "auto"
$dockerOk = $false

# Check if docker and docker compose commands exist
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
$composeCmd = Get-Command docker-compose -ErrorAction SilentlyContinue

if ($dockerCmd) {
    try {
        # Try docker compose first (more reliable)
        $out = docker compose version 2>&1
        if ($out -match "Docker Compose") { $dockerOk = $true }
    } catch {}
    
    # Fallback to docker info if compose check fails
    if (-not $dockerOk) {
        try {
            $out = docker info 2>&1
            if ($out -match "Containers|Images") { $dockerOk = $true }
        } catch {}
    }
}

if ($dockerOk) {
    Write-Host "      Docker detected - starting Qdrant container..."
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    docker compose up qdrant -d 2>&1 | Out-Null
    $ErrorActionPreference = $oldErrorAction
    
    $retries = 30
    $ready = $false
    for ($i = 0; $i -lt $retries; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if ($ready) {
        $env:QDRANT_MODE = "server"
        Write-Host "      Qdrant server ready on :6333" -ForegroundColor Green
    } else {
        Write-Host "      Docker Qdrant not ready - using embedded local storage." -ForegroundColor Yellow
        $env:QDRANT_MODE = "local"
    }
} else {
    Write-Host "      Docker not available - using embedded Qdrant (no Docker required)." -ForegroundColor Yellow
    $env:QDRANT_MODE = "local"
}

# ── 4. Smart ingestion ───────────────────────────────────────────────────────
Write-Host "[4/7] Indexing knowledge base (DHA / MDS / ORE Data)..." -ForegroundColor Yellow
& $python scripts\smart_ingest.py
if ($LASTEXITCODE -ne 0) { throw "smart_ingest.py failed" }

# ── 5. Model warmup ───────────────────────────────────────────────────────────
Write-Host "[5/7] Loading ML models (first run downloads to models_cache/)..." -ForegroundColor Yellow
& $python scripts\warmup_models.py
if ($LASTEXITCODE -ne 0) { throw "warmup_models.py failed" }

# ── 6. Frontend setup (sync NEXT_PUBLIC_* from root .env) ───────────────────
Write-Host "[6/7] Preparing frontend..." -ForegroundColor Yellow
$apiUrl = $env:NEXT_PUBLIC_API_URL
if (-not $apiUrl) { $apiUrl = "http://localhost:8000" }
$feToken = $env:NEXT_PUBLIC_API_TOKEN
if (-not $feToken) { $feToken = $env:API_SECRET_TOKEN }
@"
NEXT_PUBLIC_API_URL=$apiUrl
NEXT_PUBLIC_API_TOKEN=$feToken
"@ | Set-Content (Join-Path $Root "frontend\.env.local") -Encoding UTF8

$frontendDir = Join-Path $Root "frontend"
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "      Running npm install (first time)..."
    Push-Location $frontendDir
    npm install
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm install failed" }
    Pop-Location
}
Write-Host "      Frontend ready." -ForegroundColor Green

# ── 7. Launch servers in new terminals ───────────────────────────────────────
Write-Host "[7/7] Starting API and frontend servers..." -ForegroundColor Yellow

$apiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }

# Kill any orphaned Node.js or Python processes to free ports
Write-Host "      Cleaning up any orphaned processes..." -ForegroundColor Yellow
$oldErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
taskkill /F /IM node.exe 2>$null | Out-Null
taskkill /F /IM python.exe 2>$null | Out-Null
$ErrorActionPreference = $oldErrorAction
Start-Sleep -Seconds 1

$apiCmd = "cd /d `"$Root`" && .venv\Scripts\activate.bat && title MedRAG API && python run.py"
$feCmd  = "cd /d `"$frontendDir`" && title MedRAG Frontend && npm run dev"

Start-Process cmd.exe -ArgumentList "/k", $apiCmd
Start-Sleep -Seconds 3
Start-Process cmd.exe -ArgumentList "/k", $feCmd

# Wait for services
Write-Host "      Waiting for backend and frontend to be ready..."
$apiReady = $false
$feReady = $false
$maxAttempts = 90
for ($i = 0; $i -lt $maxAttempts; $i++) {
  if (-not $apiReady) {
    try {
      $h = Invoke-RestMethod -Uri "http://localhost:$apiPort/health" -TimeoutSec 2
      if ($h) { $apiReady = $true; Write-Host "      API ready" -ForegroundColor Green }
    } catch {}
  }
  if (-not $feReady) {
    try {
      $null = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 2
      $feReady = $true; Write-Host "      Frontend ready" -ForegroundColor Green
    } catch {}
  }
  if ($apiReady -and $feReady) { break }
  Start-Sleep -Seconds 2
  if (($i + 1) % 15 -eq 0) { Write-Host "      Still waiting... ($((($i + 1) * 2))s elapsed)" -ForegroundColor Gray }
}

if (-not $feReady) { Write-Host "      Frontend did not respond in time" -ForegroundColor Yellow }
if (-not $apiReady) { Write-Host "      API did not respond in time" -ForegroundColor Yellow }

# Open Chrome
$chromePaths = @(
    "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$url = "http://localhost:3000"
$opened = $false

Write-Host "      Opening browser at $url..." -ForegroundColor Yellow
foreach ($cp in $chromePaths) {
    if (Test-Path $cp) {
        try {
            Start-Process -FilePath $cp -ArgumentList $url -ErrorAction Stop
            $opened = $true
            Write-Host "      Browser opened" -ForegroundColor Green
            break
        } catch {
            Write-Host "      Failed to open Chrome at $cp" -ForegroundColor Yellow
        }
    }
}

if (-not $opened) {
    try {
        Start-Process $url -ErrorAction Stop
        Write-Host "      Browser opened (default)" -ForegroundColor Green
    } catch {
        Write-Host "      Could not open browser automatically" -ForegroundColor Yellow
        Write-Host "      Please open http://localhost:3000 manually in your browser" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  MedRAG is running!" -ForegroundColor Green
Write-Host "  Chat UI:  http://localhost:3000" -ForegroundColor Green
Write-Host "  API:      http://localhost:8000" -ForegroundColor Green
Write-Host "  Terminals: MedRAG API + MedRAG Frontend" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Keep the API and Frontend terminal windows open." -ForegroundColor Yellow
Write-Host "Re-run start_medrag.bat to restart (ingestion skips if PDFs unchanged)." -ForegroundColor Yellow
Write-Host ""
