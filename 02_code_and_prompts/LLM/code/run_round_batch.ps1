param(
    [ValidateRange(1, 25)]
    [int]$PerShard = 5
)

$ErrorActionPreference = 'Stop'
$mutex = [System.Threading.Mutex]::new($false, 'Local\MPR90_Codex_Batched_Rounds')
if (-not $mutex.WaitOne(0)) {
    Write-Output 'A cohort-batched inference run is already active; no overlapping batch was started.'
    exit 0
}

try {
    $workspace = 'D:\gpt'
    $python = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    $runner = Join-Path $workspace 'codex_round_inference_epoch03.py'
    $archiveGuard = Join-Path $workspace 'archive_failed_attempts.py'
    $auditor = Join-Path $workspace 'audit_round_inference.py'
    $schema = Join-Path $workspace 'llm_round_prediction_schema.json'
    $runDir = Join-Path $workspace 'publication_run_20260829_v3_batched'
    $logDir = Join-Path $runDir 'predictions\batch_orchestrator_logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null

    & $python $archiveGuard '--run-dir' $runDir
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed-attempt archival guard did not pass; no inference was started.'
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $processes = @()
    foreach ($shard in 0..3) {
        $arguments = @(
            $runner,
            '--run-dir', $runDir,
            '--model', 'gpt-5.6-sol',
            '--reasoning-effort', 'high',
            '--schema', $schema,
            '--max-rounds', $PerShard,
            '--max-attempts', '2',
            '--timeout-seconds', '1800',
            '--confirm-training-disabled',
            '--shard-count', '4',
            '--shard-index', $shard
        )
        $stdout = Join-Path $logDir ("batch_{0}_shard{1}.stdout.txt" -f $stamp, $shard)
        $stderr = Join-Path $logDir ("batch_{0}_shard{1}.stderr.txt" -f $stamp, $shard)
        $processes += Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    }

    $processes | Wait-Process
    & $python $archiveGuard '--run-dir' $runDir
    if ($LASTEXITCODE -ne 0) {
        throw 'Post-batch failed-attempt archival guard did not pass.'
    }
    $failed = @($processes | Where-Object { $_.ExitCode -ne 0 })
    if ($failed.Count -gt 0) {
        throw "$($failed.Count) cohort-batched shard process(es) failed; all artifacts were preserved."
    }
    & $python $auditor '--run-dir' $runDir '--expected-model' 'gpt-5.6-sol' '--expected-effort' 'high'
    if ($LASTEXITCODE -ne 0) {
        throw 'Outcome-blind post-batch audit failed; no further inference is allowed.'
    }
    Write-Output "Completed and audited up to $PerShard rounds per shard."
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
