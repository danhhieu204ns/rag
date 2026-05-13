$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonServices = @(
  @{ Name = "ollama-service"; Dir = "ollama_service"; Port = 8200 },
  @{ Name = "ingestion-service"; Dir = "ingestion_service"; Port = 8100 },
  @{ Name = "retrieval-service"; Dir = "retrieval_service"; Port = 8300 },
  @{ Name = "backend"; Dir = "backend"; Port = 8000 }
)

function Start-TerminalCommand {
  param([string]$Title, [string]$Command)
  Start-Process powershell -ArgumentList @("-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='$Title'; $Command") | Out-Null
}

function Test-Url {
  param([string]$Url)
  try {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 $Url | Out-Null
    return $true
  } catch {
    return $false
  }
}

if (-not (Test-Url "http://127.0.0.1:11434/api/tags")) {
  Start-TerminalCommand "ollama" "ollama serve"
}

if (-not (Test-Url "http://127.0.0.1:6333/readyz")) {
  $QdrantPath = Join-Path $Root ".qdrant"
  New-Item -ItemType Directory -Force $QdrantPath | Out-Null
  Start-TerminalCommand "qdrant" "cd '$Root'; `$env:QDRANT__SERVICE__HTTP_PORT='6333'; `$env:QDRANT__STORAGE__STORAGE_PATH='$QdrantPath'; qdrant"
}

foreach ($Svc in $PythonServices) {
  $Dir = Join-Path $Root $Svc.Dir
  $Cmd = "cd '$Dir'; if (!(Test-Path .venv)) { python -m venv .venv }; . .\.venv\Scripts\Activate.ps1; python -m pip install -q -U pip wheel setuptools; python -m pip install -q -r requirements.txt; python -m uvicorn app.main:app --host 0.0.0.0 --port $($Svc.Port)"
  Start-TerminalCommand $Svc.Name $Cmd
}

$FrontendDir = Join-Path $Root "frontend"
$FrontendCmd = "cd '$FrontendDir'; if (!(Test-Path node_modules)) { npm install }; npm run dev"
Start-TerminalCommand "frontend" $FrontendCmd

Write-Host "Started RAG app in separate PowerShell windows."
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend:  http://localhost:8000"
