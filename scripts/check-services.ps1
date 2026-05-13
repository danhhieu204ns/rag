$ErrorActionPreference = "Continue"

function Test-ServiceUrl {
  param([string]$Name, [string]$Url)
  try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $Url | Out-Null
    "{0,-18} OK      {1}" -f $Name, $Url
  } catch {
    "{0,-18} FAIL    {1}" -f $Name, $Url
  }
}

Write-Host "== Processes =="
Get-Process | Where-Object {
  $_.ProcessName -match "python|uvicorn|ollama|qdrant|node|npm"
} | Select-Object ProcessName, Id, Path | Format-Table -AutoSize

Write-Host "== Ports =="
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in 5173,8000,8100,8200,8300,6333,11434 } |
  Select-Object LocalAddress, LocalPort, OwningProcess |
  Format-Table -AutoSize

Write-Host "== Health checks =="
Test-ServiceUrl "frontend" "http://localhost:5173"
Test-ServiceUrl "backend" "http://localhost:8000/ready"
Test-ServiceUrl "ingestion" "http://localhost:8100/ready"
Test-ServiceUrl "ollama-service" "http://localhost:8200/ready"
Test-ServiceUrl "retrieval" "http://localhost:8300/ready"
Test-ServiceUrl "retrieval-health" "http://localhost:8300/health"
Test-ServiceUrl "ollama" "http://localhost:11434/api/tags"
