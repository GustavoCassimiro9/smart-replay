# Para o Smart Replay: encerra a partida, desliga o app, o banco e a webcam.
# Os dados do banco e os clipes salvos sao mantidos.
Set-Location $PSScriptRoot

Write-Host "Encerrando a partida e parando os containers..."
docker compose down

if (Test-Path .webcam.pid) {
    $proc = Get-Process -Id (Get-Content .webcam.pid) -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -like "python*") { Stop-Process -Id $proc.Id -Force }
    Remove-Item .webcam.pid
    Write-Host "Webcam desligada."
}
Write-Host "Pronto."
