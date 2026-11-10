#!/usr/bin/env python3
"""Retain same-input numerical controls, long parity checks and paired CPU profiles.

Numerical failures are diagnostic observations, not passing correctness tests.
Only named positive controls are required to pass. Runs are never timing trials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validation", "profiles"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = root / "build/Release/ie_tree_bench.exe"
    model = root / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
    large = root / "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    sources = subprocess.check_output(["git", "ls-files", "--cached", "--others",
        "--exclude-standard", "core", "src", "apps", "bench", "tests", "scripts", "CMakeLists.txt"], cwd=root, text=True).splitlines()
    hashes = {}
    for name in sorted(set(sources)):
        src = root/name
        if src.is_file():
            hashes[name] = digest(src)
            target = out/"source_snapshot"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src,target)
    metadata = dict(base_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        dependency_commit=subprocess.check_output(["git","-c",f"safe.directory={(root/'third_party/llama.cpp').as_posix()}","-C",str(root/'third_party/llama.cpp'),"rev-parse","HEAD"],text=True).strip(),
        executable_sha256=digest(exe), models={p.name:digest(p) for p in [model,large]},
        source_sha256=hashes, purpose=args.mode, results=[])
    (out/"source.patch").write_bytes(subprocess.check_output(["git","diff","--binary"],cwd=root))
    def save():
        (out/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    save()
    if args.mode == "profiles":
        for repetition in range(1,4):
            for arm,capacity in [("naive",2560),("tree",1536)]:
                command=[sys.executable,str(root/'scripts/profile_cpu_report3.py'),"--output",str(out/f"{arm}-{repetition}"),"--",str(exe),"--model",str(model),"--arm",arm,"--b","2","--prefix","1024","--suffix","200","--threads","8","--flash-attn","on","--kv-type","f16","--n-ctx",str(capacity),"--phase-markers"]
                subprocess.run(command,cwd=root,check=True)
        subprocess.run([sys.executable,str(root/'scripts/analyze_profiles_report3.py'),str(out)],cwd=root,check=True)
        return
    cases=[]
    for arm in ["tree","naive"]:
        for threads in [1,8]:
            for forced in [False,True]:
                cases.append((f"{arm}-t{threads}-f16-on-{'replay' if forced else 'greedy'}",model,arm,2,1024,64,threads,"f16","on",forced,threads==1))
    for threads in [1,8]:
        cases.append((f"tree-t{threads}-f32-off-replay",model,"tree",2,1024,64,threads,"f32","off",True,False))
    cases.append(("tree-t8-f16-off-replay",model,"tree",2,1024,64,8,"f16","off",True,False))
    for forced in [False,True]:
        cases.append((f"llama-1b-{'replay' if forced else 'greedy'}",large,"tree",2,128,16,1,"f16","on",forced,False))
    for branches in [2,8]:
        cases.append((f"batched-b{branches}-200",model,"tree",branches,1024,200,8,"f16","on",False,True))
    failures=[]
    for name,path,arm,branches,prefix,suffix,threads,precision,attention,forced,must_pass in cases:
        capacity = ((prefix+branches*suffix+255)//256)*256 if arm=="tree" else branches*((prefix+suffix+255)//256)*256
        # Match the known long-prefix serial control and keep useful headroom.
        if arm=="tree": capacity=max(capacity,1536 if prefix==1024 and branches==2 else capacity)
        command=[str(exe),"--model",str(path),"--arm",arm,"--b",str(branches),"--prefix",str(prefix),"--suffix",str(suffix),"--threads",str(threads),"--kv-type",precision,"--flash-attn",attention,"--n-ctx",str(capacity),"--verify-batched" if name.startswith("batched-") else "--verify"]
        if forced: command.append("--teacher-forced")
        with (out/f"{name}.stdout.log").open('w') as stdout, (out/f"{name}.stderr.log").open('w') as stderr:
            result=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr)
        text=(out/f"{name}.stdout.log").read_text()
        metrics=[line for line in text.splitlines() if line.startswith(("CSV_VERIFY,","CSV_BATCH_VERIFY,"))]
        valid=result.returncode in (0,2) and bool(metrics)
        if not name.startswith("batched-"): valid=valid and any(line.startswith("CSV_VERIFY,") for line in metrics)
        record=dict(name=name,command=command,exit_code=result.returncode,teacher_forced=forced,required_positive_control=must_pass,metrics=metrics,completed=valid)
        metadata["results"].append(record); save()
        print(json.dumps(record),flush=True)
        if not valid or (must_pass and result.returncode!=0): failures.append(name)
    metadata["required_control_failures"]=failures
    save()
    if failures: raise SystemExit("Required controls failed: "+", ".join(failures))

if __name__ == "__main__":
    main()
