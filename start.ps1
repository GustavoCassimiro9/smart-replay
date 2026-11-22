# Sobe o Smart Replay completo: webcam + banco de dados + app.
# Uso: start.bat  (ou: powershell -ExecutionPolicy Bypass -File start.ps1 [-NoWebcam])
param([switch]$NoWebcam)

Set-Location $PSScriptRoot

function Test-Docker { cmd /c "docker info >nul 2>&1"; return $LASTEXITCODE -eq 0 }

# 1. Docker Desktop
if (-not (Test-Docker)) {
    Write-Host "Abrindo o Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    $tries = 0
    while (-not (Test-Docker)) {
        Start-Sleep -Seconds 2
        $tries++
        if ($tries -gt 90) { Write-Host "O Docker nao iniciou. Abra o Docker Desktop e tente de novo."; exit 1 }
    }
}

# 2. Webcam (pulada se o .env apontar para outra camera, como DroidCam ou camera IP)
if (Test-Path .env) {
    $line = Get-Content .env | Where-Object { $_ -match '^\s*CAMERA_SOURCE=' } | Select-Object -Last 1
    if ($line -and $line -notmatch 'host\.docker\.internal') {
        Write-Host "Usando a camera do .env: $line"
        $NoWebcam = $true
    }
}

if (-not $NoWebcam) {
    $py = Join-Path $PSScriptRoot ".venv-webcam\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Write-Host "Preparando o ambiente da webcam (so na primeira vez)..."
        python -m venv .venv-webcam
        & $py -m pip install -q opencv-python
    }
    if (Test-Path .webcam.pid) {
        $old = Get-Process -Id (Get-Content .webcam.pid) -ErrorAction SilentlyContinue
        if ($old -and $old.ProcessName -like "python*") { Stop-Process -Id $old.Id -Force }
        Remove-Item .webcam.pid
    }
    $proc = Start-Process $py -ArgumentList "tools\webcam_server.py" -WindowStyle Minimized -PassThru
    $proc.Id | Out-File .webcam.pid -Encoding ascii
    Write-Host "Webcam ligada (janela minimizada)."
}

# 3. Banco + app (reconstroi a imagem so se o codigo mudou)
Write-Host "Subindo o banco e o app..."
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Write-Host "Falha no docker compose."; exit 1 }

# 4. Espera a API responder e abre o preview
Write-Host "Aguardando o app iniciar..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-WebRequest -Uri "http://localhost:5000/partida/ping" -UseBasicParsing -TimeoutSec 2 | Out-Null
        $ready = $true
    } catch {
        if ($_.Exception.Response) { $ready = $true }  # any HTTP answer means the API is up
    }
    if ($ready) { break }
    Start-Sleep -Seconds 1
}
if (-not $ready) { Write-Host "O app nao respondeu. Veja os logs: docker compose logs app"; exit 1 }

$codigo = docker compose logs app 2>$null | Select-String "Partida iniciada" | Select-Object -Last 1
Write-Host ""
Write-Host "Smart Replay rodando!"
if ($codigo) { Write-Host ($codigo.Line -replace '^.*\|\s*', '') }
Write-Host "Preview da camera: http://localhost:5000/preview"
Write-Host "Logs ao vivo:      docker compose logs -f app"
Write-Host "Para parar:        stop.bat"
Start-Process "http://localhost:5000/preview"
