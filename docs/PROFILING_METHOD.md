# Flat CPU sampling: method and limits

The Windows x64 sampler launches its own benchmark child. On the sampler thread
it briefly suspends each child thread, reads its instruction pointer, and resumes
it in a finally block. It records process CPU time, RSS and per-thread CPU deltas.
All DbgHelp calls occur on that same sampler thread. Other applications are not
suspended or profiled.

Symbol initialization does not enumerate a child still starting its loader.
The module list is refreshed after an application phase marker. Addresses are
resolved while the process is alive. The search uses local PDBs and normal image/
symbol matching; LOAD_ANYTHING is never enabled. DLL export fallback is allowed
and appears explicitly as `export::name+offset`. These are nearest exported-symbol
labels and displacements, not private PDB function/line information. Unresolved
addresses are retained. The earlier EXACT_SYMBOLS flag also disabled exports;
Microsoft documents both effects in [SymSetOptions](https://learn.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symsetoptions).

The requested interval is 20 ms, but suspension and symbol lookup add overhead.
The saved timeline contains actual observation times. The profiled latency is
excluded from raw performance trials. The CPU-delta-weighted flat distribution is
an approximation: the delta describes work since the previous observation, while
the IP is captured afterward. A worker can finish computing and enter a wait
before it is sampled. Therefore a large share labeled NtWaitForSingleObject or
another waiting primitive does not measure CPU time consumed by that wait, wall
time blocked, or the cause of the throughput tails. Nearest exported labels must
not be promoted into exact kernel attributions.

The matched collection has three runs per arm, one binary, one model and identical
b=2/P1024/S200/t8/f16/flash settings. It is useful for identifying candidate compute
symbols and for exposing limits of this sampler. It has no stacks, on-CPU event
sampling or hardware counters; it cannot establish DRAM saturation, a precise
synchronization cost, or the root cause of observed tail delays. ETW/on-CPU stack
sampling would be a separate follow-up if a causal decode diagnosis is required.
