param(
    [Parameter(Mandatory = $true)]
    [string] $Executable
)

$actual = (& $Executable | Out-String).TrimEnd()
$expected = @"
Concurrent Radix-Tree KV Cache MVP
  branches: 8
  prefix tokens: 2048
  suffix tokens per branch: 200
  bytes per token: 32768
  naive KV allocation: 562.00 MiB
  prefix-paged KV allocation: 114.00 MiB
  peak allocation reduction: 79.72%
  radix-tree nodes: 10
  shared prefix ref_count: 8
"@.TrimEnd()

if ($actual -cne $expected) {
    Write-Error "memory_bench output changed.`nExpected:`n$expected`nActual:`n$actual"
    exit 1
}
