param(
    [string]$Prompt = "A cinematic tracking shot of a red fox running through a misty pine forest at sunrise",
    [string]$Output = "output/modal-causal-forcing.mp4"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot

$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (Test-Path $LocalPython) {
    & $LocalPython -m modal run modal_app.py --prompt $Prompt --output $Output
} elseif (Get-Command modal -ErrorAction SilentlyContinue) {
    & modal run modal_app.py --prompt $Prompt --output $Output
} else {
    throw "Modal CLI is missing. Install/authenticate once with: pip install modal; modal setup"
}
