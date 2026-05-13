param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("ollama", "qdrant", "ollama-service", "ingestion-service", "retrieval-service", "backend", "frontend")]
  [string]$Service
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Start-PythonService {
  param([string]$DirName, [int]$Port)
  $Dir = Join-Path $Root $DirName
  Set-Location $Dir
  if (!(Test-Path .venv)) {
    python -m venv .venv
  }
  . .\.venv\Scripts\Activate.ps1
  python -m pip install -q -U pip wheel setuptools
  python -m pip install -q -r requirements.txt
  python -m uvicorn app.main:app --host 0.0.0.0 --port $Port
}

switch ($Service) {
  "ollama" {
    ollama serve
  }
  "qdrant" {
    $QdrantPath = Join-Path $Root ".qdrant"
    New-Item -ItemType Directory -Force $QdrantPath | Out-Null
    $env:QDRANT__SERVICE__HTTP_PORT = "6333"
    $env:QDRANT__STORAGE__STORAGE_PATH = $QdrantPath
    qdrant
  }
  "ollama-service" {
    Start-PythonService "ollama_service" 8200
  }
  "ingestion-service" {
    Start-PythonService "ingestion_service" 8100
  }
  "retrieval-service" {
    Start-PythonService "retrieval_service" 8300
  }
  "backend" {
    Start-PythonService "backend" 8000
  }
  "frontend" {
    Set-Location (Join-Path $Root "frontend")
    if (!(Test-Path node_modules)) {
      npm install
    }
    npm run dev
  }
}
