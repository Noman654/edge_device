#!/usr/bin/env python3
"""Perf harness for one config on the device. Produces results/perf_<config>.json.

Measures:
  init_ms       : model load time (llama_perf 'load time' from llama-completion)
  ttft_ms       : prompt eval time for a fixed 128-token prompt (llama-completion)
  decode_tps    : llama-bench tg64 tokens/sec (mean +- std)
  prefill_tps   : llama-bench pp128 tokens/sec
  peak_rss_mb   : max VmHWM of the llama-completion process, polled at 100ms
"""
import argparse, json, os, re, subprocess, sys, time

DEV_DIR = "/data/local/tmp/llama.cpp"
DEV_MODELS = "/data/local/tmp/gguf"

def adb(cmd, timeout=1800):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr

def thermal():
    """Max temp (C) across thermal zones plus a few named zones, for throttle detection."""
    out = adb("for z in /sys/class/thermal/thermal_zone*; do echo $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null); done")
    zones = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip('-').isdigit():
            v = int(parts[-1]); v = v / 1000 if abs(v) > 1000 else v
            zones[" ".join(parts[:-1])] = v
    cpu = [v for k, v in zones.items() if "cpu" in k.lower()]
    npu = [v for k, v in zones.items() if "nsp" in k.lower() or "npu" in k.lower() or "cdsp" in k.lower()]
    return {"max_c": max(zones.values()) if zones else None, "cpu_max_c": max(cpu) if cpu else None,
            "npu_max_c": max(npu) if npu else None, "n_zones": len(zones)}

def cooldown(limit_c=55, max_wait=300):
    t0 = time.time()
    while True:
        t = thermal()
        if t["max_c"] is None or t["max_c"] <= limit_c or time.time() - t0 > max_wait:
            return t
        time.sleep(10)

def env_prefix(extra_env):
    # NOTE: no `&&` chains before the binary: `A && B &` backgrounds a subshell and $! is then the wrong pid.
    return f"cd {DEV_DIR}; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib {extra_env};"

def run_bench(model, dev_args, extra_env, p=128, n=64, reps=3):
    cmd = f"{env_prefix(extra_env)} ./bin/llama-bench -m {DEV_MODELS}/{model} -p {p} -n {n} -r {reps} {dev_args} -o json 2>/dev/null"
    out = adb(cmd)
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        print("llama-bench raw output:\n", out, file=sys.stderr); return {}
    res = {}
    for row in json.loads(m.group(0)):
        name = row.get("test") or (f"pp{row['n_prompt']}" if row.get("n_prompt") else f"tg{row['n_gen']}")
        res[name] = {"tps": row["avg_ts"], "std": row["stddev_ts"]}
    return res

def wallclock_init_ms(model, dev_args, extra_env):
    """Process launch -> first token, measured on device with date +%s%N (includes FastRPC session + HTP power-up)."""
    out = adb(f"t0=$(date +%s%N); {env_prefix(extra_env)} ./bin/llama-completion -m {DEV_MODELS}/{model} -p hi -n 1 -no-cnv {dev_args} >/dev/null 2>&1; t1=$(date +%s%N); echo WALL $(( (t1-t0)/1000000 ))")
    m = re.search(r"WALL (\d+)", out); return int(m.group(1)) if m else None

def run_completion_with_rss(model, dev_args, extra_env, prompt_file, n_predict=32):
    """Run llama-completion on device under a local sampler loop: max VmHWM / Pss / Rss / Shared every 100 ms."""
    log = "/data/local/tmp/_bench_completion.log"
    mem = "/data/local/tmp/_bench_mem.log"
    cmd = (f"{env_prefix(extra_env)} ./bin/llama-completion -m {DEV_MODELS}/{model} -f {prompt_file} "
           f"-n {n_predict} -no-cnv --temp 0 --ignore-eos {dev_args}")
    # No sh -c wrapper: adb() passes cmd as one arg to the device shell, so the grep's single quotes are safe.
    sampler = (f"{cmd} >{log} 2>&1 & p=$!; : >{mem}; "
               f"while kill -0 $p 2>/dev/null; do cat /proc/$p/status /proc/$p/smaps_rollup 2>/dev/null | "
               f"grep -E '^(VmHWM|Rss|Pss|Shared_Clean|Shared_Dirty):' >>{mem}; echo -- >>{mem}; sleep 0.1; done; wait $p")
    adb(sampler, timeout=900)
    memlog = adb(f"cat {mem}")
    def pk(key):
        vals = [int(m) for m in re.findall(key + r":\s+(\d+)", memlog)]
        return max(vals) // 1024 if vals else 0
    peak = pk("VmHWM"); peak_pss = pk("Pss"); peak_rss_rollup = pk("Rss")
    shared = [int(a) + int(b) for a, b in zip(re.findall(r"Shared_Clean:\s+(\d+)", memlog), re.findall(r"Shared_Dirty:\s+(\d+)", memlog))]
    peak_shared = max(shared) // 1024 if shared else 0
    out = adb(f"cat {log}")
    def grab(pat):
        m = re.search(pat, out); return float(m.group(1)) if m else None
    return {
        "load_ms": grab(r"load time\s*=\s*([\d.]+) ms"),
        "prompt_ms": grab(r"prompt eval time\s*=\s*([\d.]+) ms"),
        "prompt_n": grab(r"prompt eval time\s*=\s*[\d.]+ ms /\s*(\d+) tokens"),
        "eval_ms_per_tok": grab(r"eval time\s*=\s*[\d.]+ ms /\s*\d+ runs\s*\(\s*([\d.]+) ms per token"),
        "peak_rss_mb": peak, "peak_pss_mb": peak_pss, "peak_rss_rollup_mb": peak_rss_rollup, "peak_shared_mb": peak_shared,
        "offloaded": (re.search(r"offloaded (\d+/\d+) layers", out) or [None, None])[1],
        "draft_acceptance": grab(r"draft acceptance\s*=\s*([\d.]+)"),
        "raw_tail": out[-3000:],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="gguf filename under /data/local/tmp/gguf")
    ap.add_argument("--dev-args", default="", help="shared args, e.g. '-dev HTP0 -ngl 99 -t 6 -fa on'")
    ap.add_argument("--completion-args", default="", help="extra args only for llama-completion (e.g. MTP: '-md X --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-device none')")
    ap.add_argument("--skip-bench", action="store_true", help="skip llama-bench (use when completion-args include a drafter)")
    ap.add_argument("--env", default="", help="e.g. 'GGML_HEXAGON_MM_SELECT=3'")
    ap.add_argument("--prompt-file", default="/data/local/tmp/prompt128.txt")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    therm_start = cooldown()
    print(f"[{a.config}] thermal at start: {therm_start}", file=sys.stderr)
    wall_init = wallclock_init_ms(a.model, (a.dev_args + " " + a.completion_args).strip(), a.env)
    print(f"[{a.config}] llama-bench ...", file=sys.stderr)
    bench = {} if a.skip_bench else run_bench(a.model, a.dev_args, a.env)
    print(f"[{a.config}] completion + rss ...", file=sys.stderr)
    comp = run_completion_with_rss(a.model, (a.dev_args + " " + a.completion_args).strip(), a.env, a.prompt_file, n_predict=128)
    therm_end = thermal()
    res = {"config": a.config, "model": a.model, "dev_args": a.dev_args, "env": a.env,
           "init_ms": comp["load_ms"], "init_wallclock_ms": wall_init,
           "thermal_start": therm_start, "thermal_end": therm_end, "ttft_ms": comp["prompt_ms"], "ttft_prompt_tokens": comp["prompt_n"],
           "decode_tps": bench.get("tg64", {}).get("tps"), "decode_tps_std": bench.get("tg64", {}).get("std"),
           "prefill_tps": bench.get("pp128", {}).get("tps"),
           "decode_ms_per_tok_completion": comp["eval_ms_per_tok"],
           "peak_rss_mb": comp["peak_rss_mb"], "peak_pss_mb": comp["peak_pss_mb"], "peak_rss_rollup_mb": comp["peak_rss_rollup_mb"], "peak_shared_mb": comp["peak_shared_mb"],
           "decode_tps_completion": (1000.0 / comp["eval_ms_per_tok"]) if comp["eval_ms_per_tok"] else None,
           "draft_acceptance": comp.get("draft_acceptance"), "offloaded": comp["offloaded"],
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(a.out_dir, f"perf_{a.config}.json")
    json.dump({**res, "raw_tail": comp["raw_tail"]}, open(path, "w"), indent=2)
    # keep every run: the per-config file above is overwritten, results/history/ is not
    hist = os.path.join(os.path.dirname(path), "history"); os.makedirs(hist, exist_ok=True)
    json.dump({**res, "raw_tail": comp["raw_tail"]}, open(os.path.join(hist, f"perf_{a.config}_{time.strftime('%Y%m%d-%H%M%S')}.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
