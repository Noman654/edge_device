# REPRODUCE — every number in `report/`, in order, with the checkpoint you can go back to

Everything below was run from this directory on a Mac (host) against a Qualcomm Device Cloud QRD board
(SM8750 / Snapdragon 8 Elite, Hexagon v79). One device, one job at a time. Expected numbers are the ones in
`report/REPORT.md`; run-to-run decode noise is about ±0.5%, MMLU on the frozen 1000 is exact for a given binary
(prefill-only scoring, temperature 0).

## 0. Pinned inputs
| What | Pin | Where |
|---|---|---|
| llama.cpp | commit `6a1a922d26` | `VERSIONS.md`, `scripts/build_android.sh` |
| Toolchain | `ghcr.io/snapdragon-toolchain/arm64-android:v0.7` (Hexagon SDK 6.6.0.0, NDK r28b) | `llama.cpp/scripts/snapdragon/build.py` |
| Models | `ggml-org/gemma-4-E2B-it-GGUF`: Q4_0, Q8_0, BF16, MTP drafter, sha256 in `models/SHA256SUMS` | `scripts/get_models.sh` |
| Our backend patch | `patches/0001-ggml-hexagon-fuse-gelu-mul-into-geglu.patch` (99 lines, host lib only) | applied by `scripts/build_android.sh` |
| MMLU workload | `eval/mmlu_1000_seed42.jsonl` (1000 questions, seed 42, all 57 subjects), thinking off | `eval/mmlu_eval.py` |
| TTFT prompt | `eval/prompt128.txt` (142 tokens) | `scripts/device_bench.py` |
| Fidelity text | `eval/ppl_text.txt` (6 x 512 tokens) | `scripts/kl_fidelity.sh` |

Host prerequisites: Docker, `hf` CLI, `adb`, Python venv at `.venv` (`pip install -r` nothing exotic: requests only).

## 1. Models (host)
```bash
scripts/get_models.sh q4 mtp          # what the device runs (2.8 GB + 59 MB), sha256-verified
scripts/get_models.sh q8              # 8-bit quality ceiling (5.0 GB)
scripts/get_models.sh bf16 && scripts/make_quants.sh   # only to re-derive the 4-bit-head variants (host MMLU only)
```
Checkpoint: `shasum -a 256 -c models/SHA256SUMS` passes.

## 2. Host-side quantization study (Mac, Metal build; the numbers in FINDINGS §6)
```bash
cmake -B llama.cpp/build-mac llama.cpp && cmake --build llama.cpp/build-mac -j        # Metal reference build
scripts/make_quants.sh                                                                # Q4_0-embq4, IQ4_NL, IQ4_NL-embq4, Q4_0-embiq4
# for each variant: start llama-server locally on :8080 then
.venv/bin/python eval/mmlu_eval.py --url http://127.0.0.1:8080 --config mac_<variant>_reference
.venv/bin/python scripts/paired_test.py results/mmlu_mac_q4_0_reference.csv results/mmlu_mac_q8_0_reference.csv
```
Expected (n=1000): Q8_0 58.1, Q4_0 56.8, IQ4_NL+Q4 head 55.7, IQ4_NL 55.6, Q4_0+Q4 head 52.1, Q4_0+IQ4_NL head 51.1.
Only the Q4_0-head drop is significant (p=0.0002). Result files: `results/mmlu_mac_*`.
Decision this fixes: **head stays Q8_0; body Q4_0**. Everything on-device uses the stock ggml-org Q4_0 file.

## 3. Build (host, Docker)
```bash
scripts/build_android.sh              # patched (the shipped binary; fusion on by default)
scripts/build_android.sh --no-patch   # the pre-change binary, to reproduce the baseline rows bit-for-bit
```
Checkpoint: `llama.cpp/pkg-adb/llama.cpp/lib/libggml-hexagon.so` exists; `git -C llama.cpp diff --stat` shows
99 insertions in `ggml-hexagon.cpp` (patched) or nothing (unpatched).

## 4. Device session (QDC)
```bash
adb kill-server; ssh -i <KEY.pem> -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 5037:<sa-host>:5037 -N sshtunnel@ssh.qdc.qualcomm.com &
scripts/device_deploy.sh                       # platform check (SM8750), push pkg + Q4_0 + MTP + prompt, HTP0 smoke test
adb push models/gemma-4-E2B-it-Q8_0.gguf /data/local/tmp/gguf/   # only for step 7
scripts/device_bw_anchor.sh | tee results/bw_anchor.txt          # roofline anchor (FINDINGS §3)
```
Every new QDC reservation wipes `/data/local/tmp`; re-run deploy. Never run `adb start-server` locally (port 5037 is the tunnel).

## 5. Baselines (perf x2 warm, thermally gated, then full MMLU) — REPORT rows 1-2
```bash
scripts/run_matrix.sh 'cpu_q4_0_baseline'     # ~45 min -> results/perf_cpu_q4_0_baseline.json, results/mmlu_cpu_q4_0_baseline.*
scripts/run_matrix.sh 'htp_q4_0_default'      # ~45 min -> results/perf_htp_q4_0_default.json, results/mmlu_htp_q4_0_default.*
scripts/device_profile.sh htp_q4_0_default    # verbose + per-op profile -> results/profile/ (FINDINGS §4)
```
Expected: CPU 56.9% / 37.3 t/s decode / 725 ms TTFT / 5.1 GB. NPU default 56.9% / 25.4 (session 1) or 26.9-27.1 (session 2 re-run) / 190 ms / 1.85 GB.
The NPU-default decode drifts ~6% between QDC sessions; measure baseline and OPPOLL in the same session for the delta.

## 6. Execution knob sweep — REPORT row 3 and the "ruled out" list
```bash
scripts/run_sweep.sh                          # scripts/sweep.tsv: oppoll, nhvx2/4, mm_tiled, fa_hvx, kvq8, lmhead_cpu, vmem512 -> results/perf_htp_<name>.json
scripts/run_matrix.sh 'htp_oppoll'            # perf x2 + full MMLU for the winner -> results/perf_htp_oppoll.json, results/mmlu_htp_oppoll.*
```
Expected: OPPOLL=1 -> 31.2 t/s (+15% to +23% vs the same-session default), MMLU 56.9%; every other knob neutral or worse (FINDINGS §5 table).

## 7. 8-bit quality ceiling — REPORT row "Quality ceiling"
```bash
scripts/run_matrix.sh 'htp_q8_0_oppoll'       # -> results/perf_htp_q8_0_oppoll.json, results/mmlu_htp_q8_0_oppoll.*
.venv/bin/python scripts/paired_test.py results/mmlu_htp_oppoll.csv results/mmlu_htp_q8_0_oppoll.csv
```
Expected: 58.3% (+1.4 pp, p=0.19, not significant), 20.2 t/s, 2.97 GB, init 4.6 s.

## 8. MTP speculative decoding — REPORT "Best" row
```bash
scripts/mtp_server_bench.sh 1      # depth 1: 35-40 t/s at 48-66% acceptance (content dependent)
scripts/mtp_server_bench.sh 2      # 30.0 t/s   (worse)
scripts/mtp_server_bench.sh 3      # 25.2 t/s   (worse)
```
Draft must run on HTP0 with the target (the script does this). `--spec-draft-device none` aborts with "ctx_other".
MMLU is unchanged by construction (target verifies every token); `results/mtp_server_bench.csv`.

## 9. The code change: GELU+MUL -> GEGLU fusion — REPORT row "fusion", FINDINGS §16
```bash
scripts/fusion_firecount.sh                                  # expect 34 of 35 GELU ops fused per graph
scripts/fusion_firecount.sh GGML_HEXAGON_OPFUSION=62         # expect 0 (same binary, our bit off)
scripts/kl_fidelity.sh base                                  # CPU logits, once per session (~30 s)
scripts/kl_fidelity.sh unfused "-dev HTP0 -ngl 99 -t 6 -fa on" "GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPFUSION=62"
scripts/kl_fidelity.sh fused   "-dev HTP0 -ngl 99 -t 6 -fa on" "GGML_HEXAGON_OPPOLL=1"
scripts/run_matrix.sh --perf-only 'htp_oppoll_(fuse|nofuse)' # interleave by running it twice for a clean A/B
scripts/run_matrix.sh 'htp_oppoll_fuse'                      # full MMLU on the fused binary
.venv/bin/python scripts/paired_test.py results/mmlu_htp_oppoll.csv results/mmlu_htp_oppoll_fuse.csv
```
Expected: mean KL to CPU 0.00161 (unfused) -> 0.00151 (fused), top-1 agreement 98.17 -> 98.37%; decode +0.3%
(speed-neutral); MMLU 56.8 vs 56.9, p=1.0.

## 10. Context, batching, latency, GPU — FINDINGS §7, §15, §21
```bash
scripts/longctx_bench.sh                      # tg64 @ d512/2048/8192: 30.7 / 29.8 / 28.1 t/s (re-run: 28.7±2.6 / 29.7 / 28.0; run it on a cool device)
scripts/batched_bench.sh                      # B=1/8/16 aggregate decode with OPPOLL=1: 28.3 / 80.2 / 144.4 (report's 122 @ B16 was without polling)
scripts/latency_sweep.sh                      # CPU vs NPU end-to-end at n=8..256 -> results/latency_sweep.csv (crossover ~103 tokens)
scripts/run_matrix.sh --perf-only 'gpu_q4_0'  # Adreno OpenCL: 319 prefill / 25.9 decode (dominated)
```

## 10b. Accuracy-budget variants (2026-09-07) — FINDINGS §23
```bash
# MXFP4 LM head from the STOCK file (host or on-device; only token_embd changes, 540/541 tensors byte-identical). The anchored
# --tensor-type is required: --token-embedding-type also matches per_layer_token_embd, and the default rule sends it to Q6_K.
llama.cpp/build-mac/bin/llama-quantize --allow-requantize --tensor-type '^token_embd\.weight$=mxfp4' --tensor-type '^per_layer_token_embd\.weight$=q4_0' models/gemma-4-E2B-it-Q4_0.gguf models/gemma-4-E2B-it-Q4_0-head_mxfp4.gguf Q4_0
# (q4_0 / iq4_nl / q4_1 heads: same command with the format swapped)
.venv/bin/python scripts/prune_layers.py models/gemma-4-E2B-it-Q4_0.gguf models/gemma-4-E2B-it-Q4_0-prune4.gguf 4   # drop last N KV-shared layers
scripts/run_matrix.sh 'htp_head4'            # perf + full MMLU on device;  TARGET=gemma-4-E2B-it-Q4_0-head4.gguf scripts/mtp_server_bench.sh 1
```
Expected: MXFP4 head 57.0% host / 56.7% device / 35.0 t/s / 1642 MB / MTP 37-40; Q4_0 head 56.3% / 34.6 t/s / 1656 MB / MTP 37-41; prune4 54.9% / 34.9 t/s but MTP acceptance 5-10%; prune8 10.6%.
Do NOT quantize from the Hugging Face BF16: it is not the source of the stock Q4_0 (norm tensors differ), everything made from it scores ~5 points low.

## 11. Init time and energy (2026-09-07 additions)
```bash
scripts/init_bench.sh        # wall-clock init, 6 variants x 3: NPU 3.01 s default -> 2.06 s with --fit off; CPU 2.58 -> 2.12 s
scripts/energy_bench.sh      # NOT measurable on the QDC board (external supply); ready for a phone with battery counters
scripts/mtp_server_bench.sh 1  # add SPEC_EXTRA/DRAFT/SPEC_TYPE env for the tuning sweep in results/mtp_tuning_sweep.csv
```
`GGML_HEXAGON_HMX_MIN_NROWS=<n>` (patched lib) selects HVX vs HMX for M<=n; results in `results/hmx_threshold_experiment.md`.
Recommended model file: `gemma-4-E2B-it-Q4_0-head_mxfp4.gguf` (built as above).
Recommended launch flags for the best config: `-dev HTP0 -ngl 99 -t 6 -fa on --fit off` + `GGML_HEXAGON_OPPOLL=1` + the MTP drafter (depth 1).

## 12. Tables
```bash
.venv/bin/python scripts/aggregate.py         # results/final_table.md from results/perf_*.json + results/mmlu_*.summary.json
```

## Going back to a checkpoint
- "Undo the code change": `scripts/build_android.sh --no-patch`, push `lib/libggml-hexagon.so`, or just set
  `GGML_HEXAGON_OPFUSION=62` at runtime (identical op stream to the pre-patch binary).
- "Undo polling": drop `GGML_HEXAGON_OPPOLL=1` (backend default is interrupt mode).
- "Undo MTP": don't pass `--model-draft`.
- "Back to 8-bit": step 7 config; "back to CPU": `-dev none -ngl 0`.
- Any MMLU pair: `scripts/paired_test.py A.csv B.csv` (McNemar exact) — the per-question CSVs are all in `results/`.
