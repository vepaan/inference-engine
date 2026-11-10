#!/usr/bin/env python3
"""Summarize isolated flat CPU profiles; these runs are excluded from latency statistics."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("directory"); args=ap.parse_args()
    root=Path(args.directory); summaries=[]; symbols=[]
    for metadata_path in sorted(root.glob("*/profile.json")):
        folder=metadata_path.parent; meta=json.loads(metadata_path.read_text()); cmd=meta["command"]
        arm=cmd[cmd.index("--arm")+1]
        timeline=pd.read_csv(folder/"cpu_timeline.csv")
        timeline["interval_s"]=timeline.elapsed_s.diff().fillna(0)
        for phase,data in timeline.groupby("phase"):
            seconds=data.interval_s.sum()
            if seconds>0:
                summaries.append(dict(run=folder.name,arm=arm,phase=phase,observed_wall_s=seconds,
                    mean_cpu_core_equivalents=(data.cpu_core_equivalents*data.interval_s).sum()/seconds,peak_rss_bytes=data.rss_bytes.max()))
        sym=pd.read_csv(folder/"symbols.csv"); sym["arm"]=arm; sym["run"]=folder.name; symbols.append(sym)
    if not summaries: raise ValueError("No completed profiles found")
    stats=pd.DataFrame(summaries); stats.to_csv(root/"cpu_summary.csv",index=False)
    flat=pd.concat(symbols,ignore_index=True)
    flat.to_csv(root/"combined_symbols.csv",index=False)
    decode=flat[flat.phase=="decode"].groupby(["arm","symbol"],as_index=False).cpu_delta_s.sum()
    decode["estimated_share"]=decode.cpu_delta_s/decode.groupby("arm").cpu_delta_s.transform("sum")
    decode.sort_values(["arm","estimated_share"],ascending=[True,False]).to_csv(root/"decode_hotspots.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,(arm,data) in zip(axes,decode.groupby("arm")):
        data=data.nlargest(8,"estimated_share").sort_values("estimated_share")
        ax.barh(data.symbol,data.estimated_share*100)
        ax.set(xlabel="Approximate CPU-weighted IP share (%)",title=arm)
    fig.suptitle("Decode flat profile: sampled instruction pointers, not stack-inclusive costs")
    fig.tight_layout(); fig.savefig(root/"decode_hotspots.png",dpi=170); plt.close(fig)
    print(stats.to_string(index=False))
    print(decode.sort_values("estimated_share",ascending=False).head(12).to_string(index=False))

if __name__=="__main__": main()
