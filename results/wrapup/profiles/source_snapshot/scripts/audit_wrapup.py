#!/usr/bin/env python3
"""Check the bounded Pillar A acceptance evidence without promoting failed controls."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/wrapup"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest().lower()
def load(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def text(path):
    data=path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig")
def snapshot(folder,metadata):
    for name,digest in metadata["source_sha256"].items():
        assert sha(folder/"source_snapshot"/name)==digest.lower(), str(folder/name)

def main():
    numerical=load(OUT/"numerical-final/manifest.json")
    assert len(numerical["results"])==15
    assert numerical["required_control_failures"]==[]
    snapshot(OUT/"numerical-final",numerical)
    failed=[r["name"] for r in numerical["results"] if r["exit_code"]!=0]
    assert all(r["completed"] for r in numerical["results"])
    identity=load(OUT/"measurements/environment.json")["identity"]
    snapshot(OUT/"measurements",identity)
    assert numerical["executable_sha256"].lower()==identity["executable_sha256"].lower()
    assert sha(ROOT/"build/Release/ie_tree_bench.exe")==identity["executable_sha256"].lower()
    data=list(csv.DictReader((OUT/"measurements/raw_trials.csv").open()))
    assert len(data)==20
    expected={(arm,str(b),str(t)) for arm in ("naive","tree") for b in (2,8) for t in range(1,6)}
    assert {(r["arm"],r["b"],r["trial"]) for r in data}==expected
    assert all((r["P"],r["S"],r["threads"],r["flash_attn"],r["kv_type"])==("1024","200","8","1","f16") for r in data)
    profiles=list((OUT/"profiles").glob("*/profile.json"))
    assert len(profiles)==6
    profile_summary=[]
    for p in profiles:
        m=load(p); assert m["exit_code"]==0
        assert m["executable_sha256"].lower()==identity["executable_sha256"].lower()
        assert m["model_sha256"].lower()==identity["model_sha256"].lower()
        command=m["command"]
        for flag,value in [("--b","2"),("--prefix","1024"),("--suffix","200"),("--threads","8"),("--kv-type","f16"),("--flash-attn","on")]:
            assert command[command.index(flag)+1]==value
        syms=list(csv.DictReader((p.parent/"symbols.csv").open()))
        decode=[s for s in syms if s["phase"]=="decode"]
        total=sum(float(s["cpu_delta_s"]) for s in decode)
        resolved=sum(float(s["cpu_delta_s"]) for s in decode if s["symbol"]!="unresolved")
        assert total>0
        profile_summary.append(dict(run=p.parent.name,samples=m["samples"],resolved_decode_cpu_fraction=resolved/total,api_failures=m["failures"]))
    profile_manifest=load(OUT/"profiles/manifest.json")
    snapshot(OUT/"profiles",profile_manifest)
    assert "100% tests passed, 0 tests failed out of 5" in text(OUT/"validation/ctest-real.log")
    assert "100% tests passed, 0 tests failed out of 2" in text(OUT/"validation/ctest-mock.log")
    comparisons=[]
    for branches in (2,8):
        for metric in ("kv_bytes","private_bytes","peak_ws_bytes","prefill_ms","tok_per_s","step_us_p50","step_us_p99"):
            values={a:[float(r[metric]) for r in data if r["arm"]==a and r["b"]==str(branches)] for a in ("naive","tree")}
            n,t=(statistics.median(values[a]) for a in ("naive","tree"))
            comparisons.append(dict(b=branches,metric=metric,naive_median=n,tree_median=t,tree_change_percent=(t/n-1)*100))
    with (OUT/"metric_comparisons.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    audit=dict(acceptance="bounded Pillar A prototype evidence complete",tree_regressions=21,real_ctest_targets=5,mock_ctest_targets=2,lifecycle_reuse_cycles=12,diagnostic_cases=15,diagnostic_failures_retained=failed,high_priority_measured_trials=20,original_primary_matrix_target=80,original_primary_matrix_unmeasured=60,profile_runs=profile_summary,scope="CPU, deterministic synthetic workloads, explicit single-writer lifetimes; not general backend numerical invariance",all_source_snapshots_verified=True,comparison=comparisons)
    (OUT/"ACCEPTANCE_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
