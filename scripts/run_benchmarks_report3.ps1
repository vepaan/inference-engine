param(
    [string] $BuildDir = "build/Release",
    [Alias("ModelPath")][string] $Model135 = "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    [string] $OutputDir = "results/report3/continuation",
    [Alias("Repetitions")][ValidateRange(1,100)][int] $Trials = 5,
    [ValidateRange(1,100)][int] $Warmup = 1,
    [ValidateSet("All","Matrix","Threads","Attention","Correctness","Bandwidth")][string] $Experiment = "All",
    [int[]] $Branches = @(2,4,8,16),
    [int[]] $Prefixes = @(1024,2048),
    [ValidateSet("tree","naive")][string] $VerificationArm = "tree",
    [ValidateSet(0,1)][int[]] $AttentionModes = @(1,0),
    [int] $ProfileBranches = 8,
    [int] $ProfilePrefix = 1024,
    [ValidateRange(1,256)][int] $Threads = 2,
    [ValidateRange(0,10000)][int] $MaxTrials = 0
)
$ErrorActionPreference = "Stop"
# Native stderr is backend diagnostics, not a PowerShell terminating error.
$PSNativeCommandUseErrorActionPreference = $false
$treeBenchExe = (Resolve-Path (Join-Path $BuildDir "ie_tree_bench.exe")).Path
$bwBenchExe = (Resolve-Path (Join-Path $BuildDir "ie_bandwidth_bench.exe")).Path
$Model135 = (Resolve-Path $Model135).Path
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$logDir = Join-Path $OutputDir "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$rawPath = Join-Path $OutputDir "raw_trials.csv"
$corrPath = Join-Path $OutputDir "correctness.csv"
$ctxPath = Join-Path $OutputDir "contexts.csv"
$envPath = Join-Path $OutputDir "environment.json"
# Registry and .NET avoid a dependency on a responsive WMI service.
$cpuName = (Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0').ProcessorNameString.Trim()
$logicalCores = [Environment]::ProcessorCount
$hardwareJson = python -c "import ctypes,json; from ctypes import wintypes; x=(ctypes.c_ulonglong*8)(); x[0]=64; ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(x)); print(json.dumps({'memory_bytes':x[1]}))"
if ($LASTEXITCODE -ne 0) { throw "Could not collect memory metadata using Python" }
$hardware = $hardwareJson | ConvertFrom-Json
$sourceFiles = @(git ls-files --cached --others --exclude-standard core src apps tests scripts CMakeLists.txt)
$sourceHashes = [ordered]@{}
foreach ($file in ($sourceFiles | Sort-Object -Unique)) {
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        $sourceHashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
    }
}
$identity = [ordered]@{
    git_commit = (git rev-parse HEAD).Trim()
    llama_cpp_commit = (git -C third_party/llama.cpp rev-parse HEAD).Trim()
    source_sha256 = $sourceHashes
    executable_sha256 = (Get-FileHash $treeBenchExe -Algorithm SHA256).Hash
    bandwidth_executable_sha256 = (Get-FileHash $bwBenchExe -Algorithm SHA256).Hash
    model_sha256 = (Get-FileHash $Model135 -Algorithm SHA256).Hash
    model_path = $Model135
    threads = $Threads
    warmup_processes_per_trial = $Warmup
    flash_attn = 1
    n_batch = 128
    sampling = "greedy; synthetic token IDs 1 + index modulo (vocabulary size - 1); no EOS stopping"
    kv_type = "f16 K and V; CPU only"
}
$identityJson = $identity | ConvertTo-Json -Depth 8 -Compress
$cacheFile = (Resolve-Path (Join-Path (Split-Path $BuildDir) "CMakeCache.txt")).Path
$cacheMetadata = @([IO.File]::ReadAllLines($cacheFile) -match "^(CMAKE_CXX_COMPILER|CMAKE_GENERATOR|CMAKE_BUILD_TYPE|IE_WITH_LLAMA)")
if (Test-Path $envPath) {
    $old = Get-Content $envPath -Raw | ConvertFrom-Json
    if ($old.identity_json -cne $identityJson) {
        throw "Source, binary, model or settings changed. Use a new OutputDir to preserve comparable trials."
    }
} else {
    if (Test-Path $rawPath) { throw "Existing raw trials lack this runner's provenance. Use a new OutputDir." }
    [ordered]@{
        identity_json = $identityJson
        identity = $identity
        timestamp = (Get-Date -Format o)
        cpu = $cpuName
        physical_cores = $null
        logical_cores = $logicalCores
        os = [Environment]::OSVersion.VersionString
        memory_bytes = $hardware.memory_bytes
        build_type = "Release"
        cmake_cache = $cacheMetadata
    } | ConvertTo-Json -Depth 10 | Set-Content $envPath
    git diff --binary | Set-Content (Join-Path $OutputDir "source.patch")
    foreach ($file in $sourceHashes.Keys) {
        $target = Join-Path (Join-Path $OutputDir "source_snapshot") $file
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Copy-Item -LiteralPath $file -Destination $target
    }
}
function Initialize-Csv($Path, $Header) {
    if (-not (Test-Path $Path)) { Set-Content $Path $Header }
}
Initialize-Csv $rawPath "trial,model,arm,b,P,S,threads,flash_attn,n_ctx,n_ctx_pred,kappa_bytes,kv_bytes,rho_pred,prefill_ms,fork_us_p50,fork_us_p99,step_us_p50,step_us_p90,step_us_p99,jitter_us,tok_per_s,private_bytes,peak_ws_bytes"
Initialize-Csv $ctxPath "model,arm,b,P,S,threads,flash_attn,n_ctx"
$existing = @{}
foreach ($r in @(Import-Csv $rawPath)) {
    $existing["$($r.trial)|$($r.arm)|$($r.b)|$($r.P)|$($r.S)|$($r.threads)|$($r.flash_attn)"] = $true
}
$contexts = @{}
foreach ($r in @(Import-Csv $ctxPath)) {
    $contexts["$($r.arm)|$($r.b)|$($r.P)|$($r.S)|$($r.threads)|$($r.flash_attn)"] = [int]$r.n_ctx
}
function Invoke-Native($Arguments, $LogName) {
    $stdout = Join-Path $logDir "$LogName.stdout.log"
    $stderr = Join-Path $logDir "$LogName.stderr.log"
    # Start-Process separates stderr reliably under Windows PowerShell 5.1.
    $quoted = @($Arguments | ForEach-Object { '"' + ([string]$_).Replace('"','\"') + '"' })
    $p = Start-Process -FilePath $treeBenchExe -ArgumentList $quoted -NoNewWindow -PassThru -Wait -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $lines = @(Get-Content $stdout)
    if ($p.ExitCode -ne 0) {
        $v = $lines | Where-Object { $_ -like "CSV_VERIFY,*" } | Select-Object -Last 1
        if ($v -and (Test-Path $corrPath)) { Add-Content $corrPath $v.Substring(11) }
        throw "Benchmark exited $($p.ExitCode); see $stdout and $stderr"
    }
    return $lines
}
$script:completed = 0
function Invoke-Trial($Arm, $B, $P, $S, $T, $FA, $Trial) {
    $key = "$Trial|$Arm|$B|$P|$S|$T|$FA"
    if ($existing.ContainsKey($key)) { return }
    if ($MaxTrials -gt 0 -and $script:completed -ge $MaxTrials) { return }
    $ctxKey = "$Arm|$B|$P|$S|$T|$FA"
    $label = "$Arm-b$B-p$P-s$S-t$T-fa$FA"
    $faText = if ($FA -eq 1) { "on" } else { "off" }
    $base = @("--model",$Model135,"--arm",$Arm,"--b",$B,"--prefix",$P,"--suffix",$S,"--threads",$T,"--flash-attn",$faText)
    if (-not $contexts.ContainsKey($ctxKey)) {
        Write-Host "Discovering capacity: $label"
        $out = Invoke-Native ($base + @("--discovery-only")) "$label-discovery"
        $min = $out | Where-Object { $_ -match '^MIN_N_CTX,\d+$' } | Select-Object -Last 1
        if (-not $min) { throw "Missing capacity discovery result for $label" }
        $contexts[$ctxKey] = [int]$min.Split(',')[1]
        Add-Content $ctxPath "$(Split-Path $Model135 -Leaf),$Arm,$B,$P,$S,$T,$FA,$($contexts[$ctxKey])"
    }
    $argsFixed = $base + @("--n-ctx",$contexts[$ctxKey],"--single")
    for ($w = 1; $w -le $Warmup; $w++) {
        $null = Invoke-Native $argsFixed "$label-trial$Trial-warmup$w"
    }
    $out = Invoke-Native $argsFixed "$label-trial$Trial"
    $line = $out | Where-Object { $_ -like "CSV_EXTENDED,*" } | Select-Object -Last 1
    if (-not $line) { throw "Missing CSV_EXTENDED for $label" }
    $parts = $line.Substring(13).Split(',')
    if ($parts.Count -ne 26) { throw "Unexpected CSV field count: $($parts.Count)" }
    Add-Content $rawPath ("$Trial," + ($parts[0..21] -join ','))
    $existing[$key] = $true
    $script:completed++
    Write-Host "Recorded $label trial $Trial; tokens/s=$($parts[19])"
}
if ($Experiment -in @("All","Correctness")) {
    Initialize-Csv $corrPath "model,arm,b,P,S,threads,flash_attn,token_match_rate,max_abs_logit_diff,mean_abs_logit_diff,status"
    foreach ($fa in $AttentionModes) {
        foreach ($b in $Branches) {
            foreach ($p in $Prefixes) {
                $mode = if ($fa -eq 1) { "on" } else { "off" }
                $out = Invoke-Native @("--model",$Model135,"--arm",$VerificationArm,"--b",$b,"--prefix",$p,"--suffix",64,"--threads",$Threads,"--flash-attn",$mode,"--verify") "verify-$VerificationArm-b$b-p$p-fa$fa"
                $line = $out | Where-Object { $_ -like "CSV_VERIFY,*" } | Select-Object -Last 1
                if (-not $line) { throw "Missing verification result" }
                Add-Content $corrPath $line.Substring(11)
                Write-Host $line
            }
        }
    }
}
if ($Experiment -in @("All","Matrix")) {
    foreach ($b in $Branches) { foreach ($p in $Prefixes) { for ($r=1; $r -le $Trials; $r++) {
        foreach ($arm in @("naive","tree")) { Invoke-Trial $arm $b $p 200 $Threads 1 $r }
    } } }
}
if ($Experiment -in @("All","Threads")) {
    foreach ($t in @(1,2,4,8)) {
        if ($t -gt $logicalCores) { continue }
        for ($r=1; $r -le $Trials; $r++) { foreach ($arm in @("naive","tree")) {
            Invoke-Trial $arm $ProfileBranches $ProfilePrefix 200 $t 1 $r
        } }
    }
}
if ($Experiment -in @("All","Attention")) {
    foreach ($fa in @(0,1)) { for ($r=1; $r -le $Trials; $r++) { foreach ($arm in @("naive","tree")) {
        Invoke-Trial $arm $ProfileBranches $ProfilePrefix 200 $Threads $fa $r
    } } }
}
if ($Experiment -in @("All","Bandwidth")) {
    # Preserve prior calibrations and individual pass measurements.
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = & $bwBenchExe --size-mb 256 --sweep --trials $Trials --warmup $Warmup
    if ($LASTEXITCODE -ne 0) { throw "Bandwidth calibration failed" }
    $out | Set-Content (Join-Path $logDir "bandwidth-$stamp.log")
    $bwPath = Join-Path $OutputDir "bandwidth-$stamp.csv"
    Set-Content $bwPath "size_mb,threads,median_gb_s,min_gb_s,max_gb_s"
    $out | Where-Object { $_ -like "CSV,*" } | ForEach-Object { Add-Content $bwPath $_.Substring(4) }
}
Write-Host "Finished chunk: $script:completed measured trials added. Re-run to resume."
