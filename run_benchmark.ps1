# run_benchmark.ps1 -- Full CEC 2017 benchmark for CARS
#
# Usage:
#   .\run_benchmark.ps1              # Auto-detect cores
#   .\run_benchmark.ps1 -Workers 190 # Use 190 workers
#   .\run_benchmark.ps1 -Mode quick  # Quick 3-run test

param(
    [int]$Workers = 0,
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".venv"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $ScriptDir "results\cec2017_$Timestamp"

if ($Workers -eq 0) {
    $Workers = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    if (-not $Workers) { $Workers = 4 }
}

Write-Host "=============================================="
Write-Host "  CARS CEC 2017 Benchmark"
Write-Host "  Workers: $Workers"
Write-Host "  Mode:    $Mode"
Write-Host "  Output:  $OutputDir"
Write-Host "=============================================="

# --- Virtual environment setup ---
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VenvDir
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript

Write-Host "Installing dependencies..."
pip install --quiet --upgrade pip setuptools
pip install --quiet numpy scipy "opfunu>=1.0"

# --- Run benchmark ---
Write-Host ""
Write-Host "Starting benchmark at $(Get-Date)..."

$RunnerScript = Join-Path $ScriptDir "benchmarks\run_cec2017.py"

if ($Mode -eq "quick") {
    python $RunnerScript `
        --func 21 22 23 24 25 26 27 28 29 30 `
        --dims 10 30 `
        --runs 3 `
        --workers $Workers `
        --output $OutputDir `
        2>&1 | Tee-Object -FilePath "$OutputDir.log"
} else {
    python $RunnerScript `
        --full `
        --workers $Workers `
        --output $OutputDir `
        2>&1 | Tee-Object -FilePath "$OutputDir.log"
}

Write-Host ""
Write-Host "Benchmark completed at $(Get-Date)"
Write-Host "Results: $OutputDir"
Write-Host "Summary: $OutputDir\summary.csv"
Write-Host "Stats:   $OutputDir\statistics.txt"
Write-Host "Log:     $OutputDir.log"
