<#
.SYNOPSIS
    One-shot setup for running Brainrot-V on a local NVIDIA GPU (RTX 3090 class).

.DESCRIPTION
    Checks the machine can do the job before changing anything, then builds the
    environment, fetches the weights, and proves the pipeline works with a short
    run. Safe to re-run: every stage detects work already done and skips it.

    Everything is written to a transcript next to this script, so a failure can
    be read afterwards rather than reproduced.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_lab.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_lab.ps1 -Full60s
#>

[CmdletBinding()]
param(
    # Where the repository lives (or will be cloned to).
    [string]$RepoDir = $PSScriptRoot,
    [string]$RepoUrl = "https://github.com/abdanhafidz/Causal-Forcing.git",
    # Skip the ~23 GB download, e.g. when the weights are already in place.
    [switch]$SkipWeights,
    # Skip the 3-second proof run.
    [switch]$SkipSmoke,
    # Also run a full 60-second generation (1.5-2.5 h on a 3090).
    [switch]$Full60s,
    # Minimum driver for the CUDA 12.4 wheels this project pins.
    [int]$MinDriverMajor = 550,
    [int]$RequiredFreeGB = 60
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = Join-Path $RepoDir "setup_lab-$stamp.log"
if (-not (Test-Path $RepoDir)) { New-Item -ItemType Directory -Force -Path $RepoDir | Out-Null }
Start-Transcript -Path $logFile -Force | Out-Null

$script:Stage = 0
function Write-Stage($text) {
    $script:Stage++
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ("  STAGE $($script:Stage): $text") -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
}
function Write-Good($text) { Write-Host "  [ok]   $text" -ForegroundColor Green }
function Write-Note($text) { Write-Host "  [note] $text" -ForegroundColor Yellow }
function Fail($text, $fix) {
    Write-Host ""
    Write-Host "  [STOP] $text" -ForegroundColor Red
    if ($fix) { Write-Host "         $fix" -ForegroundColor Red }
    Write-Host ""
    Write-Host "  Transcript: $logFile"
    Stop-Transcript | Out-Null
    exit 1
}

try {

# ---------------------------------------------------------------- preflight
Write-Stage "Preflight - can this machine do the job?"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail "nvidia-smi not found, so there is no usable NVIDIA driver." `
         "Install the NVIDIA driver (needs admin), then re-run this script."
}

$driver = (nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim()
$gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader).Trim()
$mem = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits).Trim()
$used, $total = $mem -split ",\s*"
$driverMajor = [int]($driver -split "\.")[0]

Write-Host "  GPU            : $gpuName"
Write-Host "  Driver         : $driver"
Write-Host "  GPU memory     : $used MiB used of $total MiB"

if ($driverMajor -lt $MinDriverMajor) {
    Fail "Driver $driver is older than $MinDriverMajor, which the CUDA 12.4 wheels need." `
         "Update the NVIDIA driver (needs admin). Nothing else here will work until then."
}
Write-Good "Driver is new enough for CUDA 12.4"

if ([int]$total -lt 20000) {
    Write-Note "This GPU has under 20 GB. A 60-second run needs roughly 18-20 GB; expect OOM."
}
if ([int]$used -gt 2000) {
    Write-Note "$used MiB is already in use - someone else may be on this GPU. You are sharing 24 GB."
}

$drive = (Resolve-Path $RepoDir).Path.Substring(0, 1)
$freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
Write-Host "  Free on ${drive}: : $freeGB GB"
if ($freeGB -lt $RequiredFreeGB) {
    Fail "Only $freeGB GB free on drive $drive; about $RequiredFreeGB GB is needed (23 GB of weights plus outputs)." `
         "Free space, or re-run with -RepoDir pointing at a larger drive."
}
Write-Good "Enough disk space"

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Note "Not running as admin. Fine for this script; you will need admin later for OpenSSH and power settings."
}

foreach ($tool in "git", "python") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "$tool is not on PATH." "Install $tool, reopen PowerShell, and re-run."
    }
}
Write-Good "git and python are present"

# ------------------------------------------------------------------- repo
Write-Stage "Repository"

if (Test-Path (Join-Path $RepoDir "brainrot_inference.py")) {
    Write-Good "Repository already present at $RepoDir"
    Push-Location $RepoDir
    try { git pull --ff-only; Write-Good "Pulled latest" }
    catch { Write-Note "git pull skipped: $($_.Exception.Message)" }
    Pop-Location
} else {
    Write-Host "  Cloning $RepoUrl ..."
    git clone $RepoUrl $RepoDir
    if (-not (Test-Path (Join-Path $RepoDir "brainrot_inference.py"))) {
        Fail "Clone finished but brainrot_inference.py is missing." "Check the repository URL."
    }
    Write-Good "Cloned"
}
Set-Location $RepoDir

# -------------------------------------------------------------- environment
Write-Stage "Python environment"

$venvPython = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $launcher = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
    if ($launcher -eq "py") { & py -3.11 -m venv .venv } else { & python -m venv .venv }
    if (-not (Test-Path $venvPython)) { Fail "Could not create the virtual environment." "Check your Python install." }
    Write-Good "Created .venv"
} else {
    Write-Good ".venv already exists"
}

$pyVer = (& $venvPython -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
Write-Host "  venv Python    : $pyVer"

& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Fail "pip failed to upgrade itself." "Check network access from this machine." }

# Does a CUDA-enabled torch already work? Re-running must not reinstall 2.5 GB.
$cudaOk = $false
try {
    $probe = (& $venvPython -c "import torch;print(torch.__version__, torch.cuda.is_available())" 2>$null)
    if ($probe -and $probe -match "True") { $cudaOk = $true; Write-Good "torch already CUDA-enabled: $probe" }
} catch { }

if (-not $cudaOk) {
    Write-Host "  Installing torch 2.6.0 + torchvision 0.21.0 (cu124). This is ~2.5 GB ..."
    # Order matters: requirements.txt pins torch==2.6.0, and installing it from
    # PyPI silently yields the CPU build. Install from the CUDA index first so
    # the later requirements pass sees the pin already satisfied.
    & $venvPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { Fail "torch install failed." "Usually network or proxy. See $logFile." }
}

Write-Host "  Installing the remaining requirements ..."
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Note "requirements.txt did not install cleanly; continuing to the CUDA check."
    Write-Note "Some entries (wandb, flask, onnx) are unused by inference and may fail harmlessly."
}

$check = (& $venvPython -c "import torch;print(torch.__version__);print(torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')")
$lines = $check -split "`r?`n"
Write-Host "  torch          : $($lines[0])"
Write-Host "  CUDA available : $($lines[1])"
Write-Host "  Device         : $($lines[2])"
if ($lines[1] -ne "True") {
    Fail "torch cannot see the GPU." `
         "Almost always the driver. Confirm nvidia-smi works, then reinstall torch from the cu124 index."
}
Write-Good "torch can see the GPU"
Write-Note "flash-attn is intentionally not installed; the code falls back to scaled_dot_product_attention."

# ----------------------------------------------------------------- weights
Write-Stage "Model weights (~23 GB)"

$weights = @{
    "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" = 11.0
    "wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" = 5.0
    "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" = 0.4
    "checkpoints/framewise/ar_diffusion.pt" = 5.0
}

function Test-Weights {
    foreach ($path in $weights.Keys) {
        if (-not (Test-Path $path)) { return $false }
        if ((Get-Item $path).Length / 1GB -lt $weights[$path]) { return $false }
    }
    return (Test-Path "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl") -and
           (Test-Path "wan_models/Wan2.1-T2V-1.3B/config.json")
}

if ($SkipWeights) {
    Write-Note "-SkipWeights given; not downloading."
} elseif (Test-Weights) {
    Write-Good "All weights already present and the right size"
} else {
    $hf = Join-Path $RepoDir ".venv\Scripts\hf.exe"
    if (-not (Test-Path $hf)) { $hf = Join-Path $RepoDir ".venv\Scripts\huggingface-cli.exe" }
    if (-not (Test-Path $hf)) {
        & $venvPython -m pip install "huggingface_hub[cli]" --quiet
        $hf = Join-Path $RepoDir ".venv\Scripts\hf.exe"
        if (-not (Test-Path $hf)) { $hf = Join-Path $RepoDir ".venv\Scripts\huggingface-cli.exe" }
    }
    if (-not (Test-Path $hf)) { Fail "The HuggingFace CLI is missing." "pip install `"huggingface_hub[cli]`" inside .venv" }

    Write-Host "  Downloading Wan2.1-T2V-1.3B (this is the long part) ..."
    & $hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B `
        --include "*.json" "*.pth" "*.safetensors" "google/*" "*.txt"
    if ($LASTEXITCODE -ne 0) { Fail "Wan2.1 download failed." "Re-run the script; the download resumes." }

    Write-Host "  Downloading the Causal-Forcing checkpoint ..."
    & $hf download zhuhz22/Causal-Forcing framewise/ar_diffusion.pt --local-dir checkpoints
    if ($LASTEXITCODE -ne 0) { Fail "Checkpoint download failed." "Re-run the script; the download resumes." }

    if (-not (Test-Weights)) {
        Fail "Downloads finished but files are missing or truncated." "Re-run; see $logFile for which."
    }
    Write-Good "Weights downloaded and verified"
}

if (-not $SkipWeights -or (Test-Weights)) {
    Get-ChildItem -Recurse wan_models, checkpoints -Include *.pth, *.safetensors, *.pt -ErrorAction SilentlyContinue |
        ForEach-Object { "  {0,-46} {1,6:N2} GB" -f $_.Name, ($_.Length / 1GB) }
}

# -------------------------------------------------------------- smoke test
Write-Stage "Proof run"

function Invoke-Run([string]$runId, [double]$seconds, [int]$steps) {
    if (Test-Path (Join-Path $RepoDir "run_local.py")) {
        & $venvPython run_local.py --run-id $runId --seed 0 `
            --duration-seconds $seconds --sampling-steps $steps
        return (Join-Path $RepoDir "output\local-runs\$runId\brainrot_v.mp4")
    }
    # run_local.py adds the log and manifest; without it, call the CLI directly
    # so the proof run still works on a fresh clone.
    Write-Note "run_local.py not found - using brainrot_inference.py directly (no manifest)."
    $dir = Join-Path $RepoDir "output\local-runs\$runId"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $mp4 = Join-Path $dir "brainrot_v.mp4"
    $frames = [math]::Ceiling(([math]::Ceiling($seconds * 16) - 1) / 4) + 1
    & $venvPython -u brainrot_inference.py `
        --prompt_file prompts/brainrot_v_long_horizon_60s.txt `
        --config_path configs/brainrot_v_framewise.yaml `
        --checkpoint_path checkpoints/framewise/ar_diffusion.pt `
        --output_file $mp4 --num_output_frames $frames `
        --sampling_steps $steps --seed 0 --mode brainrot_v
    return $mp4
}

if ($SkipSmoke) {
    Write-Note "-SkipSmoke given; not running."
} else {
    Write-Host "  3 seconds at 2 denoising steps - exercises load, memory, compaction and decode."
    Write-Host "  Watch VRAM in another window with:  nvidia-smi -l 2"
    Write-Host ""
    $started = Get-Date
    $mp4 = Invoke-Run -runId "smoke-3s" -seconds 3 -steps 2
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if (-not (Test-Path $mp4)) {
        Fail "The proof run produced no video after $elapsed s." "Read the end of $logFile for the Python error."
    }
    $kb = [math]::Round((Get-Item $mp4).Length / 1KB, 0)
    Write-Good "Produced $kb KB in $elapsed s -> $mp4"
    if ($kb -lt 50) { Write-Note "That file looks small; open it and confirm it is not blank." }
}

if ($Full60s) {
    Write-Stage "Full 60-second run"
    Write-Host "  Expect 1.5-2.5 hours on a 3090, peaking near 18-20 GB."
    $started = Get-Date
    $mp4 = Invoke-Run -runId "first-60s" -seconds 60 -steps 50
    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if (Test-Path $mp4) {
        Write-Good "60-second run finished in $mins minutes -> $mp4"
    } else {
        Write-Note "The 60-second run did not produce a video after $mins minutes."
        Write-Note "If it died at the very end it is the decode: enable use_cache=True in decode_to_pixel."
    }
}

# ------------------------------------------------------------------ summary
Write-Stage "Done"
Write-Host "  Repository : $RepoDir"
Write-Host "  Python     : $venvPython"
Write-Host "  Transcript : $logFile"
Write-Host ""
Write-Host "  Next, and only now that generation works:" -ForegroundColor Cyan
Write-Host "    1. Install Tailscale, sign in - gives this box a stable address from home."
Write-Host "    2. Enable OpenSSH Server (needs admin):"
Write-Host "         Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0"
Write-Host "         Set-Service sshd -StartupType Automatic; Start-Service sshd"
Write-Host "    3. Stop the machine sleeping mid-run (needs admin):"
Write-Host "         powercfg /change standby-timeout-ac 0"
Write-Host "         powercfg /change hibernate-timeout-ac 0"
Write-Host "    4. Then ask for the overnight queue worker."
Write-Host ""

} catch {
    Write-Host ""
    Write-Host "  [STOP] Unhandled error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "         $($_.ScriptStackTrace)" -ForegroundColor DarkRed
    Write-Host "         Transcript: $logFile"
    Stop-Transcript | Out-Null
    exit 1
}

Stop-Transcript | Out-Null
