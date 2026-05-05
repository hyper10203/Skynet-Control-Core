#!/bin/bash
# PowerShell script to download NeuroGolf 2026 competition data
# Usage: .\download_competition_data.ps1 [-Force]

param(
    [switch]$Force
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuroGolf 2026 Competition Data Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python and kaggle CLI are available
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    Write-Error "Python not found. Please install Python and the kaggle package."
    exit 1
}

# Check kaggle CLI
$kagglePath = (Get-Command kaggle -ErrorAction SilentlyContinue).Source
if (-not $kagglePath) {
    Write-Host "Installing kaggle package..." -ForegroundColor Yellow
    & python -m pip install kaggle -q
}

# Check for kaggle.json credentials
$kaggleConfigDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleConfig = Join-Path $kaggleConfigDir "kaggle.json"

if (-not (Test-Path $kaggleConfig)) {
    Write-Warning "Kaggle credentials not found at $kaggleConfig"
    Write-Host "Please set up your Kaggle API credentials:"
    Write-Host "  1. Go to https://www.kaggle.com/account"
    Write-Host "  2. Click 'Create New API Token'"
    Write-Host "  3. Place kaggle.json in $kaggleConfigDir"
    exit 1
}

Write-Host "Kaggle credentials found ✓" -ForegroundColor Green

# Create data directory
$dataDir = Join-Path $PSScriptRoot "data" "neurogolf-2026"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
Write-Host "Data directory: $dataDir" -ForegroundColor Gray

# Download competition data
Write-Host ""
Write-Host "Downloading competition data..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Gray

$args = @("competitions", "download", "-c", "neurogolf-2026", "-p", $dataDir)
if ($Force) {
    $args += "--force"
}

$process = Start-Process -FilePath "kaggle" -ArgumentList $args -Wait -PassThru -NoNewWindow

if ($process.ExitCode -ne 0) {
    Write-Error "Failed to download competition data"
    exit 1
}

Write-Host "Download complete ✓" -ForegroundColor Green

# Extract zip files
Write-Host ""
Write-Host "Extracting data files..." -ForegroundColor Cyan

$zipFiles = Get-ChildItem -Path $dataDir -Filter "*.zip"
foreach ($zip in $zipFiles) {
    Write-Host "  Extracting $($zip.Name)..." -ForegroundColor Gray
    Expand-Archive -Path $zip.FullName -DestinationPath $dataDir -Force
}

# List downloaded files
Write-Host ""
Write-Host "Downloaded files:" -ForegroundColor Cyan
$files = Get-ChildItem -Path $dataDir -File | Where-Object { $_.Extension -ne ".zip" }
foreach ($file in $files) {
    $size = "{0:N1} KB" -f ($file.Length / 1KB)
    Write-Host "  $($file.Name) ($size)" -ForegroundColor Gray
}

# Save metadata
$metadata = @{
    competition = "neurogolf-2026"
    downloaded_at = (Get-Date -Format "o")
    downloaded_by = $env:USERNAME
    files = $files.Name
} | ConvertTo-Json

$metadataPath = Join-Path $dataDir "metadata.json"
$metadata | Out-File -FilePath $metadataPath

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Competition data is ready in: $dataDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run the dashboard: .\run_dashboard.bat" -ForegroundColor Gray
Write-Host "  2. Enable competition data in Control tab" -ForegroundColor Gray
Write-Host "  3. Start the autonomy daemon" -ForegroundColor Gray
