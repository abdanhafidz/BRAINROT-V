param(
    [ValidateSet("baseline", "brainrot_v")][string]$Mode,
    [string]$PromptPath = "prompts\brainrot_v_long_horizon_60s.txt",
    [string]$RunId = "long-horizon-60s-seed0-v2",
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

& $LocalPython -m modal run --detach --quiet modal_brainrot.py::app.submit_one_long `
    --mode $Mode --prompt-path $PromptPath --run-id $RunId `
    --duration-seconds $DurationSeconds --sampling-steps $SamplingSteps --seed $Seed
