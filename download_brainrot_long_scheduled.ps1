param(
    [string]$RunId = "long-horizon-60s-scheduled",
    [string]$OutputDir = "output\brainrot-v-long-horizon-60s"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (-not (Test-Path $LocalPython)) {
    throw "Modal client environment is missing: $LocalPython"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $LocalPython -m modal volume get brainrot-v-outputs `
    "$RunId-baseline/baseline.mp4" `
    (Join-Path $OutputDir "baseline-causal-forcing-60s-scheduled.mp4") --force
if ($LASTEXITCODE -ne 0) { throw "Baseline download failed" }

& $LocalPython -m modal volume get brainrot-v-outputs `
    "$RunId-brainrot-v/brainrot_v.mp4" `
    (Join-Path $OutputDir "proposed-brainrot-v-60s-scheduled.mp4") --force
if ($LASTEXITCODE -ne 0) { throw "Brainrot-V download failed" }
