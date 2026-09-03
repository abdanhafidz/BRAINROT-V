param(
    [string]$RunId = "long-horizon-60s-seed0-v2",
    [string]$OutputDir = "output\brainrot-v-long-horizon-60s"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (-not (Test-Path $LocalPython)) {
    throw "Modal client environment is missing: $LocalPython"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
& $LocalPython -m modal volume get brainrot-v-outputs "$RunId/baseline.mp4" `
    (Join-Path $OutputDir "baseline-causal-forcing-60s.mp4") --force
& $LocalPython -m modal volume get brainrot-v-outputs "$RunId/brainrot_v.mp4" `
    (Join-Path $OutputDir "proposed-brainrot-v-60s.mp4") --force
