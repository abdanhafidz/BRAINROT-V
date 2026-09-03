param(
    [string]$PromptPath = "prompts\brainrot_v_quality.txt",
    [string]$OutputDir = "output\brainrot-v-comparison",
    [double]$DurationSeconds = 5.0,
    [int]$SamplingSteps = 50,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (Test-Path $LocalPython) {
    & $LocalPython -m modal run modal_brainrot.py::app.compare `
        --prompt-path $PromptPath --output-dir $OutputDir `
        --duration-seconds $DurationSeconds --sampling-steps $SamplingSteps --seed $Seed
} elseif (Get-Command modal -ErrorAction SilentlyContinue) {
    & modal run modal_brainrot.py::app.compare `
        --prompt-path $PromptPath --output-dir $OutputDir `
        --duration-seconds $DurationSeconds --sampling-steps $SamplingSteps --seed $Seed
} else {
    throw "Modal CLI is missing. Install/authenticate once with: pip install modal; modal setup"
}
