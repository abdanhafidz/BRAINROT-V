param(
    [string]$RunId = "long-horizon-60s-scheduled",
    [string]$OutputDir = "output\brainrot-v-long-horizon-60s",
    [string]$Suffix = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
$LocalPython = Join-Path $PSScriptRoot ".modal-client\Scripts\python.exe"
if (-not (Test-Path $LocalPython)) {
    throw "Modal client environment is missing: $LocalPython"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Video, log and manifest are all part of the record. A run whose MP4 arrives
# without its log and manifest cannot be audited later.
$Artifacts = @(
    @{ Mode = "baseline";   Remote = "baseline.mp4";               Local = "baseline-causal-forcing-60s$Suffix.mp4" },
    @{ Mode = "baseline";   Remote = "baseline.log";               Local = "baseline-causal-forcing-60s$Suffix.log" },
    @{ Mode = "baseline";   Remote = "baseline.manifest.json";     Local = "baseline-causal-forcing-60s$Suffix.manifest.json" },
    @{ Mode = "baseline";   Remote = "baseline.schedule.json";     Local = "baseline-causal-forcing-60s$Suffix.schedule.json" },
    @{ Mode = "brainrot-v"; Remote = "brainrot_v.mp4";             Local = "proposed-brainrot-v-60s$Suffix.mp4" },
    @{ Mode = "brainrot-v"; Remote = "brainrot_v.log";             Local = "proposed-brainrot-v-60s$Suffix.log" },
    @{ Mode = "brainrot-v"; Remote = "brainrot_v.manifest.json";   Local = "proposed-brainrot-v-60s$Suffix.manifest.json" },
    @{ Mode = "brainrot-v"; Remote = "brainrot_v.schedule.json";   Local = "proposed-brainrot-v-60s$Suffix.schedule.json" }
)

$Failed = @()
foreach ($Artifact in $Artifacts) {
    $Remote = "$RunId-$($Artifact.Mode)/$($Artifact.Remote)"
    $Local = Join-Path $OutputDir $Artifact.Local
    & $LocalPython -m modal volume get brainrot-v-outputs $Remote $Local --force
    if ($LASTEXITCODE -ne 0) { $Failed += $Remote }
}

if ($Failed.Count -gt 0) {
    Write-Warning "Could not download: $($Failed -join ', ')"
    # A missing MP4 is fatal; a missing log only degrades the record.
    if ($Failed -match '\.mp4$') { throw "Required video artifact missing" }
}

Write-Host "Downloaded run '$RunId' into $OutputDir"
