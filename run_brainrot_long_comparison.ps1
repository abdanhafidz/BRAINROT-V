param(
    [string]$PromptPath = "prompts\brainrot_v_long_horizon_60s.txt",
    [string]$OutputDir = "output\brainrot-v-long-horizon-60s",
    [double]$DurationSeconds = 60.0,
    [int]$SamplingSteps = 50,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (Test-Path $LocalPython) {
    & $LocalPython -m modal run modal_brainrot.py::app.long_compare `
        --prompt-path $PromptPath --output-dir $OutputDir `
        --duration-seconds $DurationSeconds --sampling-steps $SamplingSteps --seed $Seed
} elseif (Get-Command modal -ErrorAction SilentlyContinue) {
    & modal run modal_brainrot.py::app.long_compare `
        --prompt-path $PromptPath --output-dir $OutputDir `
        --duration-seconds $DurationSeconds --sampling-steps $SamplingSteps --seed $Seed
} else {
    throw "Modal CLI is missing. Install/authenticate once with: pip install modal; modal setup"
}
