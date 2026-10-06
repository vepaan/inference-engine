param(
    [string] $BuildDir = "build/Release",
    [string] $Model135 = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    [string] $Model1B = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    [string] $OutputPath = "results/tree_bench_remaining.csv",
    [string] $ExistingPath = "results/tree_bench.csv"
)

$ErrorActionPreference = "Stop"
$executable = Join-Path $BuildDir "ie_tree_bench.exe"
if (-not (Test-Path $executable)) {
    throw "Missing $executable. Configure with -DIE_WITH_LLAMA=ON and build ie_tree_bench first."
}

New-Item -ItemType Directory -Force results | Out-Null
$header = "model,arm,b,P,S,n_ctx_min,n_ctx_pred,kappa_bytes,kv_bytes_min,rho_meas,rho_pred,prefill_ms,fork_us_p50,fork_us_p99,step_us_p50,step_us_p90,step_us_p99,jitter_us,tok_per_s,private_bytes,peak_ws_bytes,max_abs_logit_diff,token_match_rate"
$columns = $header -split ","
$existingKeys = @{}

function Get-CaseKey {
    param($Row)
    return "{0}|{1}|{2}|{3}|{4}" -f $Row.model, $Row.arm, $Row.b, $Row.P, $Row.S
}

$existingRows = @()
function Read-CsvRows {
    param([string] $Path)
    $readRows = @()
    if (Test-Path $Path) {
        foreach ($line in Get-Content -Path $Path) {
            if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("model,")) {
                continue
            }
            if ($line.StartsWith("CSV,")) {
                $line = $line.Substring(4)
            }
            $readRows += $line | ConvertFrom-Csv -Header $columns
        }
    }
    return $readRows
}

foreach ($sourcePath in @($ExistingPath, $OutputPath)) {
    foreach ($row in Read-CsvRows $sourcePath) {
        $existingRows += $row
        $existingKeys[(Get-CaseKey $row)] = $true
    }
}

Set-Content -Path $OutputPath -Value $header

function Invoke-Case {
    param(
        [string] $Model,
        [string] $Arm,
        [int] $Branches,
        [int] $Prefix,
        [int] $Suffix
    )

    $caseKey = "{0}|{1}|{2}|{3}|{4}" -f (Split-Path $Model -Leaf), $Arm, $Branches, $Prefix, $Suffix
    if ($existingKeys.ContainsKey($caseKey)) {
        Write-Host "Skipping existing $Model $Arm b=$Branches P=$Prefix S=$Suffix"
        return
    }

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
    $normalizedCsvLine = "$csvLine"
    if ($normalizedCsvLine.StartsWith("CSV,")) {
        $normalizedCsvLine = $normalizedCsvLine.Substring(4)
    }
    Add-Content -Path $OutputPath -Value $normalizedCsvLine
    $existingKeys[$caseKey] = $true
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

$rows = Read-CsvRows $OutputPath
$groups = @($existingRows + $rows) | Group-Object model, b, P, S
$rhoByCase = @{}
foreach ($group in $groups) {
    $naive = $group.Group | Where-Object { $_.arm -eq "naive" } | Select-Object -First 1
    $tree = $group.Group | Where-Object { $_.arm -eq "tree" } | Select-Object -First 1
    if ($null -ne $naive -and $null -ne $tree) {
        $rhoMeasured = [double]$naive.kv_bytes_min / [double]$tree.kv_bytes_min
        foreach ($row in $group.Group) {
            $rhoByCase[(Get-CaseKey $row)] = "{0:F6}" -f $rhoMeasured
        }
    }
}

$rows | ForEach-Object {
    $key = Get-CaseKey $_
    if ($rhoByCase.ContainsKey($key)) {
        $_.rho_meas = $rhoByCase[$key]
    }
}
$rows | ConvertTo-Csv -NoTypeInformation | Set-Content -Path $OutputPath
Write-Host "Wrote $OutputPath"
