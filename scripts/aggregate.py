#!/usr/bin/env python3
"""Build the summary table from results/perf_*.json and results/mmlu_*.summary.json."""
import glob, json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
rows = {}
for p in glob.glob(f"{R}/perf_*.json"):
    d = json.load(open(p)); rows.setdefault(d["config"], {}).update(d)
for p in glob.glob(f"{R}/mmlu_*.summary.json"):
    d = json.load(open(p)); rows.setdefault(d["config"], {}).update({"mmlu_acc": d["accuracy"], "mmlu_n": d["n"]})
hdr = ["Configuration", "MMLU Acc", "Init (ms)", "Peak RAM (MB)", "TTFT 128tok (ms)", "Decode tok/s", "Prefill tok/s"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
def f(v, nd=1): return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
for name in sorted(rows):
    r = rows[name]
    acc = f"{r['mmlu_acc']*100:.1f}% (n={r['mmlu_n']})" if "mmlu_acc" in r else ""
    print(f"| {name} | {acc} | {f(r.get('init_ms'),0)} | {f(r.get('peak_rss_mb'),0)} | {f(r.get('ttft_ms'),0)} | {f(r.get('decode_tps'))} | {f(r.get('prefill_tps'))} |")
