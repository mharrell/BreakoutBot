$ErrorActionPreference = "Continue"
$outDir = "recordings\split_watcher_batch"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$names = @(
    "PPO_124", "PPO_124", "PPO_124", "PPO_124", "PPO_124", "PPO_124",
    "PPO_126", "PPO_126", "PPO_126", "PPO_126", "PPO_126", "PPO_126"
)
$labels = @(
    "5M", "10M", "15M", "19.2M_best", "20M", "25M_final",
    "30M", "35M", "40M", "45M", "47.4M_best", "50M_final"
)
$paths = @(
    "models/PPO_124/checkpoint/latest_checkpoint_5000000_steps.zip",
    "models/PPO_124/checkpoint/latest_checkpoint_10000000_steps.zip",
    "models/PPO_124/checkpoint/latest_checkpoint_15000000_steps.zip",
    "models/PPO_124/best_model.zip",
    "models/PPO_124/checkpoint/latest_checkpoint_20000000_steps.zip",
    "models/PPO_124/final_model.zip",
    "models/PPO_126/checkpoint/latest_checkpoint_30001984_steps.zip",
    "models/PPO_126/checkpoint/latest_checkpoint_35001984_steps.zip",
    "models/PPO_126/checkpoint/latest_checkpoint_40001984_steps.zip",
    "models/PPO_126/checkpoint/latest_checkpoint_45001984_steps.zip",
    "models/PPO_126/best_model.zip",
    "models/PPO_126/final_model.zip"
)

$total = $paths.Count

for ($idx = 0; $idx -lt $total; $idx++) {
    $name = $names[$idx]
    $label = $labels[$idx]
    $path = $paths[$idx]
    $num = $idx + 1

    $logFile = Join-Path $outDir "$($name)_$label.log"

    if (-not (Test-Path $path)) {
        Write-Output "[$num/$total] SKIP $name $label - file not found"
        continue
    }

    Write-Output "[$num/$total] RUNNING $name $label"
    $startTime = Get-Date

    $output = & python -u verify_split_watcher_notiming.py --model $path --games 20 2>&1
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
