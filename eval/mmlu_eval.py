#!/usr/bin/env python3
"""MMLU scorer against a running llama-server.

Scoring: apply the model's chat template, ask for exactly one token with top-N
logprobs at temperature 0, and pick the highest-probability letter among A/B/C/D.
Every config sees byte-identical prompts (fixed subset, fixed order, fixed template).
Also records per-request server timings (prompt_ms ~ TTFT, prompt_n tokens).
"""
import argparse, csv, json, os, subprocess, sys, time
import requests

def post(url, path, payload, timeout=600):
    """POST JSON. url may be http://host:port or adb://<port> (runs curl on the device over adb shell)."""
    if url.startswith("adb://"):
        port = url[len("adb://"):] or "8080"
        r = subprocess.run(["adb", "shell", "curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
                            "--data-binary", "@-", f"http://127.0.0.1:{port}{path}"],
                           input=json.dumps(payload).encode(), capture_output=True, timeout=timeout)
        class R:  # minimal requests.Response look-alike
            status_code = 200 if r.returncode == 0 and r.stdout.strip().startswith(b"{") else 599
            text = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
            def json(self): return json.loads(r.stdout)
        return R()
    return requests.post(f"{url}{path}", json=payload, timeout=timeout)

LETTERS = ["A", "B", "C", "D"]

def build_messages(row):
    subj = row["subject"].replace("_", " ")
    q = row["question"].strip()
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"]))
    user = (f"The following is a multiple choice question about {subj}. "
            f"Reply with only the letter of the correct answer.\n\n{q}\n{opts}")
    return [{"role": "user", "content": user}]

GEMMA4_TEMPLATE = "<|turn>user\n{content}<turn|>\n<|turn>model\n"   # == server /apply-template with enable_thinking=false (verified)

def batch_post_adb(port, path, payloads, timeout=600):
    """Run many POSTs in ONE adb shell call: push bodies (one JSON per line), loop curl on device, return list of dicts."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False); tmp.write("\n".join(json.dumps(p) for p in payloads) + "\n"); tmp.close()
    subprocess.run(["adb", "push", tmp.name, "/data/local/tmp/_req.jsonl"], capture_output=True, check=True)
    os.unlink(tmp.name)
    script = (f"while IFS= read -r line; do printf '%s' \"$line\" | curl -s -X POST -H 'Content-Type: application/json' --data-binary @- "
              f"http://127.0.0.1:{port}{path}; echo; done < /data/local/tmp/_req.jsonl")
    r = subprocess.run(["adb", "shell", script], capture_output=True, timeout=timeout)
    out = []
    for line in r.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: out.append({"error": line[:200]})
    return out

def apply_template(url, messages, thinking=False):
    # Gemma 4 opens a <|channel>thought block by default; MMLU scoring needs the letter as token 0.
    body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    r = post(url, "/apply-template", body, timeout=60)
    if r.status_code != 200: raise RuntimeError(f"apply-template failed: {r.text[:200]}")
    return r.json()["prompt"]

def pick_letter(resp):
    """Return (letter, method). Prefer argmax over A-D in top_logprobs, else generated token."""
    probs = resp.get("completion_probabilities") or []
    best, best_lp = None, -1e30
    if probs:
        for t in probs[0].get("top_logprobs", []):
            tok = t["token"].strip().strip("*").upper()
            if tok in LETTERS and t["logprob"] > best_lp:
                best, best_lp = tok, t["logprob"]
    if best is not None:
        return best, "logprob"
    gen = (resp.get("content") or "").strip().strip("*").upper()[:1]
    if gen in LETTERS:
        return gen, "generated"
    return None, "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "mmlu_1000_seed42.jsonl"))
    ap.add_argument("--config", required=True, help="config name, used for output file names")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-probs", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--thinking", action="store_true", help="leave thinking enabled (default off)")
    ap.add_argument("--batch", type=int, default=50, help="adb:// transport: questions per adb call")
    ap.add_argument("--local-template", action="store_true", help="build the Gemma 4 prompt locally instead of /apply-template")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    if a.limit: rows = rows[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    per_path = os.path.join(a.out_dir, f"mmlu_{a.config}.csv")
    sum_path = os.path.join(a.out_dir, f"mmlu_{a.config}.summary.json")

    done = {}
    if a.resume and os.path.exists(per_path):
        for r in csv.DictReader(open(per_path)):
            done[int(r["id"])] = r
    mode = "a" if done else "w"
    f = open(per_path, mode, newline="")
    w = csv.writer(f)
    if mode == "w":
        w.writerow(["id","subject","gold","pred","correct","method","prompt_n","prompt_ms","predicted_ms","wall_ms"])

    correct = sum(int(r["correct"]) for r in done.values())
    n = len(done)
    t_start = time.time()
    todo = [row for row in rows if row["id"] not in done]
    def make_payload(row):
        if a.local_template or a.url.startswith("adb://"):
            prompt = GEMMA4_TEMPLATE.format(content=build_messages(row)[0]["content"])
        else:
            prompt = apply_template(a.url, build_messages(row), a.thinking)
        return {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": a.n_probs, "cache_prompt": False, "samplers": []}
    def record(row, resp, wall_ms):
        nonlocal n, correct
        if not resp or "error" in resp or "completion_probabilities" not in resp:
            print(f"[{a.config}] id={row['id']} error: {str(resp)[:200]}", file=sys.stderr, flush=True)
            pred, method = None, "error"; resp = resp or {}
        else:
            pred, method = pick_letter(resp)
        gold = LETTERS[int(row["answer"])]
        ok = int(pred == gold)
        tm = resp.get("timings", {})
        w.writerow([row["id"], row["subject"], gold, pred, ok, method,
                    tm.get("prompt_n"), f"{tm.get('prompt_ms',0):.1f}", f"{tm.get('predicted_ms',0):.1f}", f"{wall_ms:.1f}"])
        f.flush(); n += 1; correct += ok
        if n % 50 == 0:
            print(f"[{a.config}] {n}/{len(rows)} acc={correct/n:.4f} elapsed={time.time()-t_start:.0f}s", file=sys.stderr, flush=True)
    if a.url.startswith("adb://"):
        port = a.url[len("adb://"):] or "8080"
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            t0 = time.time()
            resps = batch_post_adb(port, "/completion", [make_payload(r) for r in chunk])
            per_ms = (time.time() - t0) * 1000 / max(1, len(chunk))
            if len(resps) != len(chunk):
                print(f"[{a.config}] batch {i}: got {len(resps)} responses for {len(chunk)} requests", file=sys.stderr, flush=True)
            for row, resp in zip(chunk, resps + [{}] * (len(chunk) - len(resps))):
                record(row, resp, per_ms)
    else:
        for row in todo:
            t0 = time.time()
            r = post(a.url, "/completion", make_payload(row), timeout=600)
            wall_ms = (time.time() - t0) * 1000
            record(row, r.json() if r.status_code == 200 else {"error": r.text[:200]}, wall_ms)
    f.close()

    recs = list(csv.DictReader(open(per_path)))
    acc = sum(int(r["correct"]) for r in recs) / len(recs)
    pm = [float(r["prompt_ms"]) for r in recs if r["prompt_ms"]]
    pn = [float(r["prompt_n"]) for r in recs if r["prompt_n"]]
    summary = {"config": a.config, "n": len(recs), "accuracy": round(acc, 4),
               "mean_prompt_tokens": round(sum(pn)/len(pn), 1) if pn else None,
               "mean_prompt_ms": round(sum(pm)/len(pm), 1) if pm else None,
               "p50_prompt_ms": sorted(pm)[len(pm)//2] if pm else None,
               "thinking": a.thinking, "methods": {m: sum(1 for r in recs if r["method"] == m) for m in ("logprob","generated","none","error")}}
    json.dump(summary, open(sum_path, "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
