param(
    [string]$Prompt = "A lone astronaut walking through a neon market in the rain",
    [string]$Output = "output/brainrot-v.mp4",
    [int]$NumFrames = 21,
    [int]$SamplingSteps = 50,
    [ValidateSet("brainrot_v", "baseline")][string]$Mode = "brainrot_v"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (Test-Path $LocalPython) {
    & $LocalPython -m modal run modal_brainrot.py::app.main --prompt $Prompt --output $Output `
        --num-frames $NumFrames --sampling-steps $SamplingSteps --mode $Mode
} elseif (Get-Command modal -ErrorAction SilentlyContinue) {
    & modal run modal_brainrot.py::app.main --prompt $Prompt --output $Output `
        --num-frames $NumFrames --sampling-steps $SamplingSteps --mode $Mode
} else {
    throw "Modal CLI is missing. Install/authenticate once with: pip install modal; modal setup"
}
