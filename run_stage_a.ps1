<#
.SYNOPSIS
    Stage A of the token-eviction plan: finish the 10 October screen.

.DESCRIPTION
    Five runs remain from a design that is already half paid for: m3 seeds 1-2
    and m4 seeds 0-2, at 30 s with the configs the completed arms used.

    Submits ONE run at a time and waits for it to finish before starting the
    next. The 10 October batch of fifteen was fired at once and five were lost
    when credits ran out mid-flight; serial submission means an interruption
    costs one run, and the spend is visible as it accrues.

    Nothing about the generation path is changed for this stage. Peak-VRAM
    logging, which Stage B needs, is deliberately NOT added here: the point of
    Stage A is to complete a comparison, so m3 and m4 must run exactly as
    m0-m2 did.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File run_stage_a.ps1 -WhatIf
    powershell -ExecutionPolicy Bypass -File run_stage_a.ps1
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$RepoDir = $PSScriptRoot,
    [double]$DurationSeconds = 30,
    [int]$SamplingSteps = 50,
    # Roughly 25 min of A100 time each, from the ten completed runs.
    [int]$ExpectedMinutesPerRun = 25
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $RepoDir

$python = Join-Path $RepoDir ".modal-client\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Modal client not found: $python" }

# arm, seed -- the five runs missing from the 10 October screen.
$runs = @(
    @{ Arm = "m3-scissor-decay";  Seed = 1 },
    @{ Arm = "m3-scissor-decay";  Seed = 2 },
    @{ Arm = "m4-adakv-entropy";  Seed = 0 },
    @{ Arm = "m4-adakv-entropy";  Seed = 1 },
    @{ Arm = "m4-adakv-entropy";  Seed = 2 }
)

$totalMin = $runs.Count * $ExpectedMinutesPerRun
Write-Host ""
Write-Host "Stage A - completing the 10 October screen" -ForegroundColor Cyan
Write-Host "  runs            : $($runs.Count), submitted one at a time"
Write-Host "  duration        : $DurationSeconds s at $SamplingSteps steps"
Write-Host "  expected GPU    : ~$([math]::Round($totalMin/60,1)) A100-hours (~`$$([math]::Round($totalMin/60*2.5))-$([math]::Round($totalMin/60*3)))"
Write-Host ""

$completed = 0
foreach ($run in $runs) {
    $runId = "llm-$($run.Arm)-s$($run.Seed)-30s"
    $config = "configs/llm_methods/brainrot_v_$($run.Arm).yaml"
    if (-not (Test-Path $config)) { throw "Missing config: $config" }

    if (-not $PSCmdlet.ShouldProcess($runId, "submit and wait")) { continue }

    Write-Host ("-" * 60)
    Write-Host "  $runId  (~$ExpectedMinutesPerRun min)" -ForegroundColor Cyan
    $started = Get-Date
    $startedUtc = (Get-Date).ToUniversalTime().AddMinutes(-2)

    # --detach so the GPU work survives a dropped terminal; the loop then polls
    # the run's own manifest, which keeps submission serial without tying the
    # run's life to this shell.
    & $python -m modal run --detach --quiet modal_brainrot.py::app.submit_one_long `
        --mode brainrot_v `
        --prompt-path prompts/brainrot_v_long_horizon_60s.txt `
        --run-id $runId `
        --duration-seconds $DurationSeconds `
        --sampling-steps $SamplingSteps `
        --seed $run.Seed `
        --config-path $config

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  submission failed for $runId - stopping." -ForegroundColor Red
        Write-Host "  $completed of $($runs.Count) completed before this." -ForegroundColor Red
        exit 1
    }

    # A previous attempt at this run id may have left a manifest on the volume.
    # Polling for "succeeded" or "failed" alone would read that stale record and
    # report a result before this run has written anything, so the manifest is
    # only believed once its started_at is newer than this submission.
    $tmp = Join-Path $env:TEMP "$runId.manifest.json"
    $status = ""
    while ($status -ne "succeeded" -and $status -ne "failed") {
        Start-Sleep -Seconds 60
        & $python -m modal volume get brainrot-v-outputs `
            "$runId/brainrot_v.manifest.json" $tmp --force 2>$null | Out-Null
        $status = ""
        if (Test-Path $tmp) {
            try {
                $m = Get-Content $tmp -Raw | ConvertFrom-Json
                if ([datetime]$m.started_at -gt $startedUtc) { $status = $m.status }
            } catch { }
        }
        $waited = [math]::Round(((Get-Date) - $started).TotalMinutes, 0)
        Write-Host "    ... $waited min, status=$(if($status){$status}else{'pending'})"
    }

    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    if ($status -eq "failed") {
        Write-Host "  $runId FAILED after $mins min - stopping." -ForegroundColor Red
        Write-Host "  Check the log: modal volume get brainrot-v-outputs $runId/brainrot_v.log ." -ForegroundColor Red
        exit 1
    }
    $completed++
    Write-Host "  done in $mins min  ($completed/$($runs.Count))" -ForegroundColor Green
}

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "  Stage A submitted: $completed of $($runs.Count)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Then collect and apply the gate:"
Write-Host "    python ..\..\..\scratchpad\collect_llm_screen.py   # or the saved copy"
Write-Host ""
Write-Host "  GATE: proceed to Stage B only if an arm wins >= 4 of 6 paired"
Write-Host "        comparisons against m0. Otherwise report a negative result."
Write-Host ""
