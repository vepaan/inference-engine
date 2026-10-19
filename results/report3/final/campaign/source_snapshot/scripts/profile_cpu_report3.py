#!/usr/bin/env python3
"""Flat Windows x64 instruction-pointer sampling of a benchmark child we launch.

Uses local PDBs via DbgHelp; no administrator or system-wide trace is required.
Sampling perturbs the child. CPU-delta weights approximate attribution; they are
not hardware-event samples, stack-inclusive costs, or measurements of DRAM traffic.
"""
import argparse
import collections
import ctypes as ct
from ctypes import wintypes as wt
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import psutil

class SymbolInfo(ct.Structure):
    _fields_ = [("SizeOfStruct",wt.ULONG),("TypeIndex",wt.ULONG),("Reserved",ct.c_ulonglong*2),
        ("Index",wt.ULONG),("Size",wt.ULONG),("ModBase",ct.c_ulonglong),("Flags",wt.ULONG),
        ("Value",ct.c_ulonglong),("Address",ct.c_ulonglong),("Register",wt.ULONG),
        ("Scope",wt.ULONG),("Tag",wt.ULONG),("NameLen",wt.ULONG),("MaxNameLen",wt.ULONG),("Name",ct.c_char*1)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",required=True)
    ap.add_argument("--interval-ms",type=float,default=20)
    ap.add_argument("command",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if os.name!="nt" or ct.sizeof(ct.c_void_p)!=8:
        raise SystemExit("Windows x64 is required")
    cmd=args.command[1:] if args.command[:1]==["--"] else args.command
    if not cmd or args.interval_ms<1:
        raise SystemExit("Provide a benchmark command after -- and a positive interval")
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    kernel=ct.WinDLL("kernel32",use_last_error=True)
    dbg=ct.WinDLL("dbghelp",use_last_error=True)
    kernel.OpenThread.argtypes=[wt.DWORD,wt.BOOL,wt.DWORD]; kernel.OpenThread.restype=wt.HANDLE
    kernel.SuspendThread.argtypes=[wt.HANDLE]; kernel.SuspendThread.restype=wt.DWORD
    kernel.ResumeThread.argtypes=[wt.HANDLE]; kernel.ResumeThread.restype=wt.DWORD
    kernel.GetThreadContext.argtypes=[wt.HANDLE,ct.c_void_p]; kernel.GetThreadContext.restype=wt.BOOL
    kernel.CloseHandle.argtypes=[wt.HANDLE]; kernel.CloseHandle.restype=wt.BOOL
    dbg.SymInitialize.argtypes=[wt.HANDLE,ct.c_char_p,wt.BOOL]; dbg.SymInitialize.restype=wt.BOOL
    dbg.SymRefreshModuleList.argtypes=[wt.HANDLE]; dbg.SymRefreshModuleList.restype=wt.BOOL
    dbg.SymFromAddr.argtypes=[wt.HANDLE,ct.c_ulonglong,ct.POINTER(ct.c_ulonglong),ct.POINTER(SymbolInfo)]; dbg.SymFromAddr.restype=wt.BOOL
    dbg.SymCleanup.argtypes=[wt.HANDLE]; dbg.SymCleanup.restype=wt.BOOL
    dbg.SymSetOptions.argtypes=[wt.DWORD]; dbg.SymSetOptions.restype=wt.DWORD
    dbg.SymSetOptions(0x4|0x2|0x400)
    phases=queue.SimpleQueue()
    stdout_file=(out/"stdout.log").open("w")
    stderr_file=(out/"stderr.log").open("w")
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=stderr_file,text=True,bufsize=1)
    child=psutil.Process(proc.pid)
    handle=wt.HANDLE(int(proc._handle))
    def read_stdout():
        for line in proc.stdout:
            stdout_file.write(line)
            if line.startswith("PROFILE_PHASE,"):
                phases.put(line.strip().split(",",1)[1])
    reader=threading.Thread(target=read_stdout,daemon=True); reader.start()
    symbols=bool(dbg.SymInitialize(handle,str(Path(cmd[0]).resolve().parent).encode(),True))
    phase="load"; refreshed=False; samples=[]; timeline=[]; previous_threads={}; last_cpu=0.; last_wall=time.perf_counter(); start=last_wall
    failures=collections.Counter()
    try:
        while proc.poll() is None:
            tick=time.perf_counter()
            while not phases.empty(): phase=phases.get()
            if phase=="prefill" and symbols and not refreshed:
                dbg.SymRefreshModuleList(handle); refreshed=True
            try:
                times=child.cpu_times(); cpu=times.user+times.system
                wall=tick-last_wall
                memory=child.memory_info()
                timeline.append(dict(elapsed_s=tick-start,phase=phase,cpu_core_equivalents=(cpu-last_cpu)/wall if timeline and wall>0 else 0,
                    rss_bytes=memory.rss,threads=child.num_threads()))
                last_cpu=cpu; last_wall=tick
                for th in child.threads():
                    cpu_time=th.user_time+th.system_time
                    prior=previous_threads.get(th.id,cpu_time)
                    previous_threads[th.id]=cpu_time
                    delta=max(0.,cpu_time-prior)
                    h=kernel.OpenThread(0x8|0x2|0x40,False,th.id)
                    if not h: failures["open_thread"]+=1; continue
                    suspended=False
                    try:
                        if kernel.SuspendThread(h)==0xffffffff:
                            failures["suspend"]+=1; continue
                        suspended=True
                        context=ct.create_string_buffer(1232+16)
                        aligned=(ct.addressof(context)+15)&~15
                        ct.c_uint32.from_address(aligned+48).value=0x100001
                        if kernel.GetThreadContext(h,aligned):
                            rip=ct.c_uint64.from_address(aligned+248).value
                            samples.append(dict(elapsed_s=time.perf_counter()-start,phase=phase,thread=th.id,rip=rip,cpu_delta_s=delta))
                        else: failures["context"]+=1
                    finally:
                        if suspended: kernel.ResumeThread(h)
                        kernel.CloseHandle(h)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                failures["process_query"]+=1
            time.sleep(max(0.,args.interval_ms/1000-(time.perf_counter()-tick)))
        code=proc.wait(); reader.join(timeout=5)
        cache={}
        for rip in {r["rip"] for r in samples}:
            storage=ct.create_string_buffer(ct.sizeof(SymbolInfo)+1024)
            info=ct.cast(storage,ct.POINTER(SymbolInfo)); info.contents.SizeOfStruct=ct.sizeof(SymbolInfo); info.contents.MaxNameLen=1024
            displacement=ct.c_ulonglong()
            if symbols and dbg.SymFromAddr(handle,rip,ct.byref(displacement),info):
                cache[rip]=ct.string_at(ct.addressof(storage)+SymbolInfo.Name.offset,info.contents.NameLen).decode(errors="replace")
            else: cache[rip]="unresolved"
        groups=collections.defaultdict(lambda:[0,0,0.])
        for row in samples:
            row["symbol"]=cache[row["rip"]]; row["rip"]=hex(row["rip"])
            value=groups[(row["phase"],row["symbol"])]; value[0]+=1; value[1]+=row["cpu_delta_s"]>0; value[2]+=row["cpu_delta_s"]
        totals=collections.Counter()
        for (phase_name,_),v in groups.items(): totals[phase_name]+=v[2]
        summary=[dict(phase=k[0],symbol=k[1],samples=v[0],active_samples=v[1],cpu_delta_s=v[2],estimated_cpu_share=v[2]/totals[k[0]] if totals[k[0]] else 0.) for k,v in groups.items()]
        summary.sort(key=lambda r:(r["phase"],-r["cpu_delta_s"]))
        for name,rows in [("ip_samples.csv",samples),("cpu_timeline.csv",timeline),("symbols.csv",summary)]:
            if rows:
                with (out/name).open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        metadata=dict(command=cmd,exit_code=code,interval_ms=args.interval_ms,symbol_handler_initialized=symbols,samples=len(samples),failures=dict(failures),
            method="Flat instruction pointers of this launched child's threads; CPU-time deltas weight approximate attribution; instrumentation perturbs execution; no stacks or hardware counters",
            executable_sha256=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest(),
            profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
        (out/"profile.json").write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))
        for row in [r for r in summary if r["phase"]=="decode"][:12]: print(row)
        if code: raise SystemExit(code)
    finally:
        if symbols: dbg.SymCleanup(handle)
        if proc.poll() is None: proc.terminate(); proc.wait()
        stdout_file.close(); stderr_file.close()

if __name__=="__main__": main()
