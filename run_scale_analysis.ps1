$ErrorActionPreference = "Continue"
$outDir = "recordings\split_watcher_batch"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$checkpoints = @()

# PPO_127 checkpoints
foreach ($step in @("5000000", "10000000", "15000000", "20000000", "25000000")) {
    $path = "models/PPO_127/checkpoint/latest_checkpoint_$($step)_steps.zip"
    if (Test-Path $path) {
        $checkpoints += @{Name="PPO_127"; Label="$($step/1000000)M"; Path=$path}
    }
}
if (Test-Path "models/PPO_127/best_model.zip") {
    $checkpoints += @{Name="PPO_127"; Label="best"; Path="models/PPO_127/best_model.zip"}
}
if (Test-Path "models/PPO_127/final_model.zip") {
    $checkpoints += @{Name="PPO_127"; Label="25M_final"; Path="models/PPO_127/final_model.zip"}
}

# PPO_128 checkpoints
foreach ($step in @("5000000", "10000000", "15000000", "20000000", "25000000")) {
    $path = "models/PPO_128/checkpoint/latest_checkpoint_$($step)_steps.zip"
    if (Test-Path $path) {
        $checkpoints += @{Name="PPO_128"; Label="$($step/1000000)M"; Path=$path}
    }
}
if (Test-Path "models/PPO_128/best_model.zip") {
    $checkpoints += @{Name="PPO_128"; Label="best"; Path="models/PPO_128/best_model.zip"}
}
if (Test-Path "models/PPO_128/final_model.zip") {
    $checkpoints += @{Name="PPO_128"; Label="25M_final"; Path="models/PPO_128/final_model.zip"}
}

$total = $checkpoints.Count
Write-Output "Running split-watcher on $total checkpoints (10 games/layout each)"
Write-Output ""

for ($idx = 0; $idx -lt $total; $idx++) {
    $ckpt = $checkpoints[$idx]
    $num = $idx + 1
    $logFile = Join-Path $outDir "$($ckpt.Name)_$($ckpt.Label).log"

    Write-Output "[$num/$total] RUNNING $($ckpt.Name) $($ckpt.Label)"
    $startTime = Get-Date

    $output = & python -u verify_split_watcher_notiming.py --model $ckpt.Path --games 10 2>&1
    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $startTime
    Write-Output "  -> Exit: $exitCode, Elapsed: $($elapsed.TotalMinutes.ToString('F1')) min"

    $output | Out-File -FilePath $logFile -Encoding UTF8

    $verdictLine = ($output | Select-String "VERDICT:" | Select-Object -First 1).Line
    Write-Output "  $verdictLine"
    $perfectLine = ($output | Select-String "Games with perfect transfer" | Select-Object -First 1).Line
    Write-Output "  $perfectLine"
    $retentionLine = ($output | Select-String "Avg ALT score retention" | Select-Object -First 1).Line
    Write-Output "  $retentionLine"
    $divLine = ($output | Select-String "Avg action divergence" | Select-Object -First 1).Line
    Write-Output "  $divLine"
    Write-Output ""
}

Write-Output "Done. Logs in $outDir/"
