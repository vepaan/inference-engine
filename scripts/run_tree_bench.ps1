param(
    [string] $BuildDir = "build/Release",
    [string] $Model135 = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    [string] $Model1B = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$executable = Join-Path $BuildDir "ie_tree_bench.exe"
if (-not (Test-Path $executable)) {
    throw "Missing $executable. Configure with -DIE_WITH_LLAMA=ON and build ie_tree_bench first."
}

New-Item -ItemType Directory -Force results | Out-Null
$outputPath = "results/tree_bench.csv"
$header = "model,arm,b,P,S,n_ctx_min,n_ctx_pred,kappa_bytes,kv_bytes_min,rho_meas,rho_pred,prefill_ms,fork_us_p50,fork_us_p99,step_us_p50,step_us_p90,step_us_p99,jitter_us,tok_per_s,private_bytes,peak_ws_bytes,max_abs_logit_diff,token_match_rate"
Set-Content -Path $outputPath -Value $header

function Invoke-Case {
    param(
        [string] $Model,
        [string] $Arm,
        [int] $Branches,
        [int] $Prefix,
        [int] $Suffix
    )

    Write-Host "Running $Model $Arm b=$Branches P=$Prefix S=$Suffix"
    $arguments = @(
        "--model", $Model,
        "--arm", $Arm,
        "--b", $Branches,
        "--prefix", $Prefix,
        "--suffix", $Suffix,
        "--single"
    )
    $logPath = [System.IO.Path]::GetTempFileName()
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $executable @arguments 1> $logPath 2>&1
        $exitCode = $LASTEXITCODE
        $output = Get-Content -Path $logPath
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Remove-Item -Force $logPath -ErrorAction SilentlyContinue
    }
    if ($exitCode -ne 0) {
        $output | ForEach-Object { Write-Host $_ }
        throw "Benchmark failed with exit code $exitCode"
    }
    $output | ForEach-Object { Write-Host $_ }
    $csvLine = $output | Where-Object { "$($_)" -like "CSV,*" } | Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($csvLine)) {
        throw "Benchmark produced no CSV row"
    }
    Add-Content -Path $outputPath -Value $csvLine
}

foreach ($arm in @("naive", "tree")) {
    foreach ($branches in @(2, 4, 8, 16)) {
        foreach ($prefix in @(1024, 2048)) {
            Invoke-Case $Model135 $arm $branches $prefix 200
        }
    }
}

foreach ($arm in @("naive", "tree")) {
    Invoke-Case $Model1B $arm 8 2048 200
}

$rows = Import-Csv -Path $outputPath
$groups = $rows | Group-Object model, b, P, S
foreach ($group in $groups) {
    $naive = $group.Group | Where-Object { $_.arm -eq "naive" } | Select-Object -First 1
    $tree = $group.Group | Where-Object { $_.arm -eq "tree" } | Select-Object -First 1
    if ($null -ne $naive -and $null -ne $tree) {
        $rhoMeasured = [double]$naive.kv_bytes_min / [double]$tree.kv_bytes_min
        foreach ($row in $group.Group) {
            $row.rho_meas = "{0:F6}" -f $rhoMeasured
        }
    }
}

$rows | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $outputPath
Write-Host "Wrote $outputPath"
