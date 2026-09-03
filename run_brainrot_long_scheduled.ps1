param(
    [string]$PromptPath = "prompts\brainrot_v_long_horizon_60s.txt",
    [string]$RunId = "long-horizon-60s-scheduled",
    [double]$DurationSeconds = 60.0,
    [int]$SamplingSteps = 50,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (-not (Test-Path $LocalPython)) {
    throw "Modal client environment is missing: $LocalPython"
}

$Common = @(
    "-m", "modal", "run", "--detach", "--quiet",
    "modal_brainrot.py::app.submit_one_long",
    "--prompt-path", $PromptPath,
    "--duration-seconds", $DurationSeconds,
    "--sampling-steps", $SamplingSteps,
    "--seed", $Seed
)

& $LocalPython @Common --mode baseline --run-id "$RunId-baseline"
if ($LASTEXITCODE -ne 0) { throw "Baseline submission failed" }

& $LocalPython @Common --mode brainrot_v --run-id "$RunId-brainrot-v"
if ($LASTEXITCODE -ne 0) { throw "Brainrot-V submission failed" }

Write-Host "Submitted independent persistent workers."
Write-Host "Baseline volume path: $RunId-baseline/baseline.mp4"
Write-Host "Brainrot-V volume path: $RunId-brainrot-v/brainrot_v.mp4"
