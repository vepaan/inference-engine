#!/usr/bin/env python3
"""Analyze recorded independent trials; never pool different controlled settings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["model", "arm", "b", "P", "S", "threads", "flash_attn", "kv_type"]
METRICS = ["n_ctx", "kv_bytes", "prefill_ms", "fork_us_p50", "fork_us_p99",
           "step_us_p50", "step_us_p90", "step_us_p99", "tok_per_s",
           "private_bytes", "peak_ws_bytes"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/report3/final/campaign")
    args = parser.parse_args()
    root = Path(args.directory)
    df = pd.read_csv(root / "raw_trials.csv")
    if "kv_type" not in df.columns:
        df["kv_type"] = "f16"
    if df.empty:
        raise ValueError("No measured trials available")
    required = KEYS + METRICS + ["trial", "kappa_bytes"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    numeric = [c for c in required if c not in ("model", "arm", "kv_type")]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Non-finite or missing numeric measurements")
    if df.duplicated(KEYS + ["trial"]).any():
        raise ValueError("Duplicate trial identifiers within a configuration")
    if not df.arm.isin(["naive", "tree"]).all():
        raise ValueError("Unknown arm")
    for c in ["n_ctx", "kv_bytes", "prefill_ms", "step_us_p50", "tok_per_s"]:
        if (df[c] <= 0).any():
            raise ValueError(f"Non-positive {c}")
    if not ((df.step_us_p50 <= df.step_us_p90) & (df.step_us_p90 <= df.step_us_p99)).all():
        raise ValueError("Invalid latency percentiles")
    if not np.allclose(df.kv_bytes, df.n_ctx * df.kappa_bytes):
        raise ValueError("Inconsistent configured KV byte units")
    grouped = df.groupby(KEYS, dropna=False)
    summary = grouped[METRICS].agg(["median", "std", "min", "max",
        lambda x: x.quantile(.75) - x.quantile(.25)])
    summary.columns = [f"{c}_{'iqr' if f == '<lambda_0>' else f}" for c, f in summary.columns]
    summary["repetitions"] = grouped.size()
    summary = summary.reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    # Each figure fixes model, prefix, suffix, threads, and attention mode.
    controls = ["model", "P", "S", "threads", "flash_attn", "kv_type"]
    for setting, subset in summary.groupby(controls):
        model, p, s, threads, fa, kv = setting
        label = f"{Path(model).stem}-P{p}-S{s}-t{threads}-fa{fa}-{kv}"
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for arm, data in subset.groupby("arm"):
            data = data.sort_values("b")
            for ax, metric, scale, ylabel in zip(axes,
                ["prefill_ms", "step_us_p50", "tok_per_s"], [1, .001, 1],
                ["Prefill (ms)", "Decode step p50 (ms)", "Aggregate tokens/s"]):
                ax.errorbar(data.b, data[f"{metric}_median"] * scale,
                    yerr=data[f"{metric}_iqr"] * scale / 2, marker="o", label=arm,
                    capsize=3)
                ax.set(xlabel="Branches", ylabel=ylabel)
                ax.set_xticks(sorted(data.b.unique()))
                ax.legend()
        fig.suptitle(f"P={p}, S={s}, threads={threads}, FA={fa}, KV={kv}; median +/- half IQR across trials")
        fig.tight_layout()
        fig.savefig(figures / f"performance-{label}.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for arm, data in subset.groupby("arm"):
            data = data.sort_values("b")
            for ax, metric, title in zip(axes, ["kv_bytes", "private_bytes", "peak_ws_bytes"],
                ["Estimated configured KV", "Measured private memory", "Measured peak working set"]):
                ax.plot(data.b, data[f"{metric}_median"] / 1e6, marker="o", label=arm)
                ax.set(xlabel="Branches", ylabel="MB (decimal)", title=title)
                ax.set_xticks(sorted(data.b.unique()))
                ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"memory-{label}.png", dpi=160)
        plt.close(fig)
    for setting, subset in summary.groupby(["model", "b", "P", "S", "flash_attn", "kv_type"]):
        if subset.threads.nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for arm, data in subset.groupby("arm"):
            data = data.sort_values("threads")
            for ax, metric, scale, ylabel in zip(axes, ["step_us_p50", "tok_per_s"],
                [.001, 1], ["Decode step p50 (ms)", "Aggregate tokens/s"]):
                ax.errorbar(data.threads, data[f"{metric}_median"] * scale,
                    yerr=data[f"{metric}_iqr"] * scale / 2, marker="o", label=arm)
                ax.set(xlabel="CPU threads", ylabel=ylabel)
                ax.set_xticks(sorted(data.threads.unique()))
                ax.legend()
        name = "-".join(map(str, setting))
        fig.suptitle(f"b={setting[1]}, P={setting[2]}, S={setting[3]}, FA={setting[4]}, KV={setting[5]}; median +/- half IQR")
        fig.tight_layout()
        fig.savefig(figures / f"threads-{name}.png", dpi=160)
        plt.close(fig)
    # Compare arms only at identical controlled settings.
    paired = summary.pivot(index=["model", "b", "P", "S", "threads", "flash_attn", "kv_type"],
                           columns="arm", values=["prefill_ms_median", "step_us_p50_median", "kv_bytes_median"])
    paired.columns = ["_".join(c) for c in paired.columns]
    paired.to_csv(root / "comparisons.csv")
    bandwidth_files = sorted(root.glob("bandwidth-*.csv")) or ([root / "bandwidth.csv"] if (root / "bandwidth.csv").exists() else [])
    if bandwidth_files:
        bw = pd.read_csv(bandwidth_files[-1])
        if not np.isfinite(bw.select_dtypes(include="number").to_numpy()).all():
            raise ValueError("Invalid bandwidth measurements")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bw.threads, bw.median_gb_s, marker="o")
        ax.fill_between(bw.threads, bw.min_gb_s, bw.max_gb_s, alpha=.2)
        ax.set(xlabel="CPU threads", ylabel="Effective sequential read GB/s",
               title="Calibration median and min/max across passes")
        fig.tight_layout()
        fig.savefig(figures / "bandwidth.png", dpi=160)
        plt.close(fig)
        # This is an illustrative logical read model, not measured llama.cpp traffic.
        traffic = df.groupby(KEYS, as_index=False).kappa_bytes.median()
        traffic["modeled_mean_kv_read_bytes_per_step"] = (
            traffic.b * (traffic.P + (traffic.S + 1) / 2) * traffic.kappa_bytes)
        traffic = traffic.merge(bw[["threads", "median_gb_s"]], on="threads", how="inner")
        traffic["modeled_transfer_time_ms"] = (
            traffic.modeled_mean_kv_read_bytes_per_step / (traffic.median_gb_s * 1e9) * 1000)
        traffic.to_csv(root / "modeled_kv_transfer.csv", index=False)
    complete = sum(1 for _, g in df.groupby(["model", "b", "P", "S", "threads", "flash_attn", "kv_type"])
                   if set(g.arm) == {"naive", "tree"})
    expected = {(a,b,p,200) for a in ["naive", "tree"] for b in [2,4,8,16] for p in [1024,2048]}
    observed = {(r.arm,r.b,r.P,r.S) for r in df.itertuples()}
    audit = dict(trials=len(df), configurations=len(summary), paired_configurations=complete,
                 missing_primary_configurations=sorted(expected-observed),
                 variability_note="IQR and sample std use independent process trials; std undefined for n=1. Error bars are descriptive, not confidence intervals.")
    environment_path = root / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        identity = environment.get("identity", {})
        baseline_threads = identity.get("threads", 2)
        baseline_fa = identity.get("default_flash_attn", identity.get("flash_attn", 1))
        baseline_kv = identity.get("kv_type", "f16").split()[0]
        baseline = df[(df.threads == baseline_threads) & (df.flash_attn == baseline_fa) & (df.kv_type == baseline_kv)]
        coverage = []
        for arm in ["naive", "tree"]:
            for branches in [2, 4, 8, 16]:
                for prefix in [1024, 2048]:
                    count = len(baseline[(baseline.arm == arm) & (baseline.b == branches) & (baseline.P == prefix) & (baseline.S == 200)])
                    coverage.append(dict(arm=arm,b=branches,P=prefix,S=200,threads=baseline_threads,flash_attn=baseline_fa,kv_type=baseline_kv,repetitions=count,target=5,remaining=max(0,5-count)))
        pd.DataFrame(coverage).to_csv(root / "matrix_coverage.csv", index=False)
        audit["primary_measured_trials"] = sum(row["repetitions"] for row in coverage)
        audit["primary_remaining_trials"] = sum(row["remaining"] for row in coverage)
        audit["primary_target_trials"] = 80
    (root / "validation.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"Wrote summary, comparisons, and figures to {root}")

if __name__ == "__main__":
    main()
