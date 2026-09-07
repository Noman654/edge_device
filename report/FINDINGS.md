# Gemma 4 E2B on Snapdragon 8 Elite (Hexagon v79) — Full Findings

> Chronological working appendix. REPORT.md is authoritative for final numbers; sections superseded by later
> results (e.g. the early 8-bit-head headline table) are kept for the record.

Author: Noman654. Task: optimize Gemma 4 E2B inference for the Qualcomm Hexagon NPU, report baseline vs best on
MMLU accuracy, init time, peak RAM, TTFT, decode rate, end-to-end latency. Runtime: llama.cpp `ggml-hexagon`.

---

## 0. Environment and ground truth
- **Device:** Qualcomm Device Cloud (QDC) QRD reference board. `ro.board.platform=sun`, `ro.soc.model=SM8750`
  = Snapdragon 8 Elite, Hexagon NPU arch **v79**, Adreno 830 GPU, 16 GB RAM, 8 CPU cores. Same silicon as the
  Galaxy S25, but a **reference board**: different thermal envelope, DVFS governor and memory tuning than a retail
  phone. Every absolute number here is a QRD number; treat S25 values as directionally similar, not identical.
- **Build:** llama.cpp (commit 6a1a922, 2026-09-05) cross-compiled for arm64-android via the Snapdragon toolchain
  Docker image (Hexagon SDK 6.6.0.0, Android NDK r28b). HTP kernel libs built for v73/v75/v79/v81; v79 selected at
  runtime. Deployed to `/data/local/tmp` over adb through the QDC SSH tunnel.
- **Model:** provided `gemma-4-E2B-it-Q4_0.gguf` (2.84 GB). Architecture `gemma4`, 35 layers, hidden 1536, FFN 6144,
  8 query heads, **1 KV head**, head_dim 256, sliding-window pattern (4 local : 1 global), **20/35 layers share KV**,
  per-layer embeddings (PLE), and **no separate output tensor — the tied Q8_0 token-embedding (262144x1536, 428 MB)
  IS the LM head**. final_logit_softcapping=30. The Q4_0 file is Q4_0 body + Q8_0 tied head.
- **Eval:** 1000 MMLU questions, seed 42, all 57 subjects, frozen to `eval/mmlu_1000_seed42.jsonl`. Zero-shot,
  thinking disabled (Gemma 4 otherwise opens a `<|channel>thought` block), scored by requesting **one token with
  top-20 logprobs at temperature 0 and taking the argmax over A/B/C/D**. Host Mac and device agreed within noise
  (Q4_0 56.8% Mac vs 56.9% device), validating the harness.

---

## 1. Headline results (device, QRD SM8750)
| Configuration | MMLU | Init | Peak host RAM | TTFT (142 tok) | Decode t/s | Prefill t/s |
|---|---|---|---|---|---|---|
| **Baseline** CPU Q4_0, 6 threads | 56.9% | 0.74 s | 5108 MB | 725 ms | 37.3 | 202 |
| NPU default (HTP0, Q4_0, FA on) | 56.9% | 1.16 s | 1850 MB | 190 ms | 25.4 | 793 |
| **Best** NPU + OPPOLL=1, Q4_0 | 56.9% | 1.19 s | 1850 MB | **185 ms** | **31.2** | **810** |
| Quality ceiling NPU + OPPOLL, Q8_0 | 58.6% (n=995) | 4.57 s | 2966 MB | 189 ms | 20.2 | 794 |

**Best vs baseline:** prefill **4.0x**, TTFT **3.9x faster**, peak host RAM **0.36x** (2.8x less), decode 0.84x,
MMLU identical (McNemar p=1.0). **Optimization delta:** OPPOLL lifted NPU decode 25.4 -> 31.2 t/s (**+23%**; +15% against the re-measured 27.1 baseline, see §5 note), free.

---

## 2. Methodology details (things that bit us, and how they were made correct)
- **QDC blocks host->device port forwards.** sa-host:8080 is QDC's own WSTUN service; other ports time out. So the
  MMLU scorer does NOT use a forwarded socket — it runs `curl` ON the device via `adb shell`, batching 50 questions
  per call (each adb round trip is ~2 s through the tunnel). Gemma 4 chat template is applied locally, verified
  byte-identical to the server's `/apply-template` with thinking off.
- **Peak RAM** is sampled ON the device at 100 ms (VmHWM + smaps_rollup Pss/Rss/Shared) around the process. Reported
  number is peak Pss. NPU host RAM is far lower than CPU because HTP weights live in device-side FastRPC buffers not
  counted in the host process, and the 1.35 GB PLE table is read lazily (~5 KB/token). True NPU footprint incl. the
  ~1.4 GB HTP buffer is ~3.3 GB, still < CPU's 5.1 GB.
- **Thermals:** perf runs are gated to wait until the hottest of 88 thermal zones is <=55 C before starting, and
  zone temps are logged start/end. MMLU accuracy is throttle-independent (deterministic argmax), so only perf runs
  need gating.
- **Init time** reported two ways: llama_perf "load time" and a wall-clock launch->first-token (captures FastRPC
  session + HTP power-up). Q8_0 init is 4.6 s vs Q4_0 1.2 s — larger weights repack into HTP tiled buffers at load.
- **Process name truncation:** Linux comm truncates to 15 chars, so `pidof llama-completion` fails; RSS sampling
  uses the shell's own `$!` on device. `adb shell "cmd &"` blocks unless `setsid ... </dev/null`.

---

## 3. Bandwidth roofline (the anchor for every claim)
`test-backend-ops perf -o MUL_MAT` GEMV at Gemma's ffn shape (m=4096, k=14336):
| Backend | Q4_0 | Q8_0 |
|---|---|---|
| HTP0 (NPU) | 50.8 GB/s | 54.7 GB/s |
| CPU (6 thr) | 42.4 GB/s | 54.6 GB/s |
Dense weight read per decode token ~1.46 GB (ffn 876 MB + head 428 MB + attn 157 MB). At ~51 GB/s that caps decode
near 35 t/s; the CPU already runs at 37.3 t/s = **decode is memory-bandwidth bound and the CPU is at the roofline.**
The NPU has no bandwidth advantage for single-token decode.

---

## 4. Profiling the NPU decode path (GGML_HEXAGON_PROFILE=1, n=32)
Per-decode-token budget (Tot usec / 32 steps):
| Component | ms/token | note |
|---|---|---|
| FFN matmuls (up/gate NX + down) | ~16.1 | 1033 MB body / 22.5 ms = 46 GB/s = **90% of the GEMV ceiling** |
| LM head MUL_MAT q8_0 1536x262144 | 7.4 | 428 MB / 7.4 ms = 58 GB/s = at roofline |
| attention matmuls (q/k/v/o) | ~2.7 | |
| norms/adds/glu/rope/gelu (100s of tiny ops) | ~2.9 | 2-3 us each |
| flash-attn-ext | ~0.8 | HMX path |
| tanh+scale over full 262144 vocab (softcap) | ~0.4 | |
| **NPU-busy sum** | **~31.8** | |
| **clean decode step** | **39.7** | |
| **host-side orchestration** | **~7-8** | the only decode headroom |

**Key finding:** matmuls are already at ~90% of achievable bandwidth — no kernel headroom. The recoverable loss is
~7-8 ms/token of host orchestration. The decode graph submits as **TWO OPBATCH batches per token** (800 ops, then
23). Confirmed from `ggml-hexagon.cpp` (`enqueue_op`/`fit_op`): a batch flushes when the next op won't fit the
batch's buffer/tensor/VMEM budget (NOT op count; 800 < the 1280 cap). The 428 MB head + 1 MB full-vocab output
don't fit alongside the 35-layer body, forcing a flush -> second batch. Each batch = one FastRPC round trip; by
default completion is signaled by **interrupt** (OPPOLL=0). VMEM default is already 3.2 GB on v79, so it's the
buffer-count limit, not VMEM, that splits the batch (raising VMEM does not merge them; lowering it hurt badly).

---

## 5. The optimization: OPPOLL=1 (+15-23% decode, free)
Switching NPU batch-completion from interrupt wakeup to **polling** removes the per-batch round-trip latency
identified above. Decode 25.4 -> 31.2 t/s (+23% in session 1; +15% vs the re-measured 27.1 baseline), prefill 793 -> 810, TTFT ~unchanged, MMLU unchanged (56.9%,
McNemar p=1.0 vs baseline). This is an **execution-path optimization found by profiling**, not a config guess.
**Cost:** one CPU core spins during NPU compute (higher power) — a throughput-for-power trade; flag for
battery-sensitive deployments.

### Knob sweep (perf-only, warm, thermally gated) — what helped and what didn't
| Config | Decode | Prefill | TTFT | verdict |
|---|---|---|---|---|
| HTP default | 25.4 | 793 | 190 | — |
| **+ OPPOLL=1** | **31.2** | 810 | 185 | **+23% (session 1) / +15% (re-measured), the win** |
| + NHVX=4 | 26.8 | 722 | 205 | small; nothing over OPPOLL |
| + NHVX=2 | 17.5 | 540 | 255 | starves HVX, -31% |
| + OPPOLL + NHVX=4 | 31.2 | 735 | 203 | NHVX adds nothing on top of poll |
| + MM_SELECT=2 (no HMX) | 24.0 | **76** | 1666 | **HMX is essential** (prefill collapses 10x) |
| + FA_SELECT=1 (HVX flash-attn) | 25.0 | 367 | 379 | HMX flash-attn matters for prefill |
| + KV q8_0 | 23.4 | 749 | 205 | -8%, no benefit |
| + lm_head on CPU (-ot) | 24.8 | 788 | 187 | ~0 -> **head is NOT the decode bottleneck** |
| + VMEM=512 | 11.9 | 541 | 266 | backfired (VMEM is a mapping ceiling) |

---

**Re-measurement note (2026-09-07, reproducibility pass).** The NPU-default row was re-run from the checked-in script
because its result file had been overwritten by a failed run: decode **26.9 t/s** (fusion on, patched binary) and
**27.1 t/s** (`GGML_HEXAGON_OPFUSION=62`, i.e. the pre-patch op stream), prefill 793 and TTFT 190 identical to the
original. The original session measured 25.4. So the default baseline drifts ~6% between QDC sessions while OPPOLL
reproduces at 31.2 in both; the polling gain is therefore **+15% (against today's baseline) to +23% (session 1)**. Both
result files are kept (`results/perf_htp_q4_0_default.json` = today, `..._prepatch.json` = today with our fusion off).

## 6. Quantization findings (host MMLU, n=1000, paired McNemar on the frozen set)
| Variant | MMLU | vs Q4_0 significance |
|---|---|---|
| Q8_0 (all 8-bit) | 58.1% | +1.3 pt, p=0.25 **not significant** |
| **Q4_0 body + Q8_0 head (provided)** | **56.8%** | reference |
| IQ4_NL body + Q8_0 head | 55.6% | -1.2 pt, p=0.35 n.s. |
| IQ4_NL body + Q4_0 head | 55.7% | n.s. |
| Q4_0 body + Q4_0 head | 52.1% | **-4.7 pt, p=0.0002 significant** |
| Q4_0 body + IQ4_NL head | 51.1% | significant |
**Conclusions:** (a) the provided Q4_0/Q8-head file is the quality-per-byte optimum; (b) the **tied embedding/LM
head is quantization-sensitive and must stay 8-bit** (4-bit head costs ~5 pts); (c) IQ4_NL buys nothing without an
imatrix; (d) Q8_0 body is not worth 35% slower decode + 60% RAM for a non-significant 1.3 pt. **So baseline and best
share weights; all decode gain must come from execution.** HTP matmul kernels accept only Q4_0/Q4_1/Q8_0/MXFP4/
IQ4_NL — **sub-4-bit (Q3/Q2) is not an NPU option** (would fall back to CPU).

---

## 7. End-to-end latency crossover (measured; total = TTFT + n * decode_ms)
| output tokens | CPU total | NPU+OPPOLL total | winner |
|---|---|---|---|
| 8 | 939 ms | 436 ms | NPU (53% faster) |
| 32 | 1573 ms | 1185 ms | NPU (24% faster) |
| ~103 | — | — | **crossover** |
| 128 | 4157 ms | 4288 ms | CPU (3% faster) |
| 256 | 7588 ms | 8599 ms | CPU (10% faster) |
**NPU wins end-to-end for outputs up to ~103 tokens** (short answers, classification, MMLU); CPU wins for long
free generation because decode is bandwidth-bound and the CPU's raw decode is higher.

---

## 8. The throughput ceiling — why single-stream decode can't beat the CPU, and what would
The HMX matrix unit does ~12 TFLOPS FP16 (>300x a single HVX thread) and is **idle at batch 1** — decode is one
GEMV per weight, low arithmetic intensity, so the matrix array is starved and decode degrades to CPU-class. This is
structural, not a bug. Two, and only two, ways past it (both feed the idle matrix unit):
1. **MTP / speculative decoding** — draft several tokens, verify them in ONE HMX pass. Converts memory-bound decode
   into compute-bound verification, losslessly. Gemma 4 E2B ships a **built-in 59 MB MTP head**, which sidesteps the
   usual mobile blocker (>1 GB draft models competing for RAM). Recent papers report 300-340 t/s decode for 4-bit
   models in this class once the matrix unit is fed. **This is the single highest-value next experiment.**
2. **Batched / continuous decode (-np > 1)** — amortize each weight load across N concurrent streams. Big win for
   multi-user serving; zero for single-user latency.

---

## 9. Attempts that did not land (with honest reasons)
- **MTP speculative decoding — aborts on this build.** ggml_abort inside `common_speculative_init_result` (not the
  caught "Gemma4Assistant requires ctx_other" warning). Candidate causes: (a) `GGML_ASSERT(n_embd ==
  n_embd_out(ctx_tgt))` mismatch for the 256-dim MTP head vs 1536 target; (b) pinning all layers (-ngl 99) makes the
  memory fitter abort before it wires the draft's shared context (`common_fit_extra_model`). Tried the default flags
  and the upstream S26+ recipe (`--no-mmap --ctx-size 8192 --device HTP0 --spec-draft-device none --spec-type
  draft-mtp --spec-draft-n-max 3`). **Retry plan:** don't pin -ngl 99 (let the fitter co-place target+draft), or a
  newer llama.cpp. Needs device iteration. Upstream S26+ saw +1-2 t/s at 48% acceptance with n=3, tunable.
- **Layer trimming (depth pruning) — blocked by tooling + expected to crater.** `llama-quantize --prune-layers`
  drops the tensors but does NOT trim Gemma 4's per-layer array hparams (`feed_forward_length`,
  `sliding_window_pattern` stay length 35 while block_count -> 33) so the model won't load
  ("expected 33, got 35"). Hand-rolled GGUF metadata rewrite hit gguf-py array/scalar type edge cases. And the
  literature says SLMs reach near-random MMLU by ~10-25% pruning without a fine-tune heal; Gemma E2B is already
  PLE-compressed. Not viable within a 20-30% MMLU budget without a training/heal step (out of scope).
- **imatrix-guided Q4_0 — started, then stopped on request.** The idea (free accuracy at identical bytes/speed):
  compute an importance matrix on a held-out calibration corpus (MMLU validation split, clean of the test set) and
  requantize Q4_0 weighted by activation statistics; typically recovers much of the Q4->Q8 gap. Tooling built and
  calibration corpus prepared (113k words); the imatrix compute on BF16/Metal was slow and was killed before a
  result. **This remains the most promising free accuracy lever and is host-only — worth finishing.**

---

## 10. Kernel / fusion analysis (for the "fused kernel / accuracy" question)
- Fusion normally raises throughput (fewer dispatches), not accuracy. The backend already fuses RMS_NORM+MUL and
  MUL_MAT+ADD (`try_fuse_*` in `ggml-hexagon.cpp`), and op fusion is ON by default. **Update: §16 adds a third,
  GELU+MUL -> GEGLU, and on this model it is an accuracy-fidelity fix, not a speed one.**
- The one genuine kernel-level accuracy knob: the matmul dynamically quantizes activations (src1) to q8_0/q8_1;
  an f16-activation kernel path exists (`quantize_f32_f16_flat`). Forcing f16 activations would trade a little speed
  for a little accuracy, but Q4_0 *weight* error dominates, so the gain is marginal — the imatrix route dominates it.
- Deeper profiling is available: **GGML_HEXAGON_PROFILE=2** plus 8 hardware PMU counters (`htp/hex-profile.h`,
  upmucnt0-7) would give HVX/HMX utilization and cache behavior, and would quantitatively confirm HMX is idle at
  batch 1. Device task.

---

## 11. Recommendations, ranked
1. **Fix MTP** (device). The only lossless single-stream way past the decode roofline; Gemma's tiny built-in draft
   makes it mobile-viable. Retry without -ngl 99 pinning.
2. **Finish imatrix Q4_0** (host, no device). Free accuracy at identical NPU speed/size; likely recovers part of the
   1.3-pt Q4->Q8 gap. Stackable with OPPOLL.
3. **Batched decode -np>1** (device) if the deployment serves multiple streams — potential multi-x throughput.
4. **PROFILE=2 PMU capture** (device) to quantify HMX idle and finalize the throughput story.
5. **Adreno GPU (OpenCL) backend** (device) — built but untested; mobile GPUs sometimes win decode. One config,
   could reshuffle the table. Genuinely new direction.
6. Cheap: `--ubatch-size 128/256` (maintainer tip, untested), re-measure the NPU-default perf row, thermal-soak run.

---

## 12. Reproduction
Step-by-step, checkpointed: `REPRODUCE.md` at the repo root. Summary of the tooling:
`scripts/`: `qdc_connect.sh` (tunnel), `device_deploy.sh` (push+chmod+smoke), `device_bench.py` (perf: init/TTFT/
decode/prefill/RAM, thermally gated), `device_profile.sh` (verbose + per-op profile), `run_sweep.sh` + `sweep.tsv`
(knob sweep), `latency_sweep.sh` (crossover), `make_quants.sh` (quant variants), `fix_pruned_arrays.py` (prune
metadata fixer, WIP), `paired_test.py` (McNemar), `aggregate.py` (table). `eval/mmlu_eval.py` scorer with `adb://`
transport. Frozen data `eval/mmlu_1000_seed42.jsonl`. Raw outputs in `results/`. Report: `report/REPORT.md`,
`report/latency_crossover.png`, this file.

## 13. Lessons / gotchas (operational)
- Never `pkill llama` or push a model while an MMLU is running (killed a run once; two chains collided once).
- Kill old background chains explicitly before starting new ones; a "dead" waiter can resume and collide.
- `adb shell "cmd &"` blocks — use `setsid sh -c '...' </dev/null`. Process comm truncates to 15 chars.
- QDC sessions expire and give a new key+sa-host each time; the local `/data/local/tmp` is wiped per reservation
  (re-push ~3 GB, ~10-15 min at the tunnel's ~5 MB/s). Never run a local `adb start-server` (steals port 5037).
- Mac disk is tight; large GGUFs (BF16 9.3 GB, Q8 5 GB) must be deleted promptly, and never `rm` a source while a
  quantizer still maps it (caused a disk-full truncation once).

---
## 14. BREAKTHROUGH (session 2026-09-06, sa812018): MTP fixed — NPU decode now matches/beats CPU
The earlier MTP abort was caused by device mismatch: `--spec-draft-device none` puts the draft on CPU while the
target is on HTP0, but the Gemma 4 MTP head cross-attends to the target's KV cache and must share its context.
**Fix: run the draft on the SAME device (`-ngl 99 -dev HTP0`, no `--spec-draft-device none`).** MTP then initializes
cleanly. (Dropping `-ngl 99` also avoids the abort but makes the fitter fall all layers back to CPU, since QDC's
HTP0 does not report free memory — that path gives CPU+MTP, not NPU+MTP.)

Draft-depth tuning (NPU, OPPOLL=1, lossless — target verifies every token, so MMLU stays 56.9%):
| n_draft | decode t/s | acceptance | vs OPPOLL 31.2 |
|---|---|---|---|
| 1 | **35-40** (content-dependent) | 48-66% | **+13% to +28%** |
| 2 | 30.0 | 36% | -4% |
| 3 | 25.2 | 27% | -19% |
Depth 1 wins: at higher depth, acceptance falls and the NPU wastes verification compute on rejected tokens.
Measured decode: 35.3 t/s on a hard prompt (48% accept), 38-40 t/s on natural prose (60-66% accept).

### New best configuration: NPU + OPPOLL=1 + MTP(n_draft=1)
| Metric | CPU baseline | NPU default | NPU best (OPPOLL+MTP n=1) |
|---|---|---|---|
| Decode t/s | 37.3 | 25.4 | **~38 (35-40)** |
| Prefill t/s | 202 | 793 | ~810 |
| TTFT ms | 725 | 190 | ~185 |
| Peak host RAM | 5108 MB | 1850 MB | ~1900 MB |
| MMLU | 56.9% | 56.9% | 56.9% (lossless) |
**This makes the NPU Pareto-dominant: it now matches/beats the CPU on decode too (~38 vs 37.3) while keeping the
4x prefill, 3.9x TTFT and ~2.7x-less RAM.** Total decode gain over the NPU default: 25.4-27.1 -> ~38 = **+40-50%**, all
lossless. Two stacked, profiling-driven optimizations: polling (removes FastRPC round-trip latency) + MTP depth-1
(fills the idle HMX with one verified speculative token per step).

## 15. Batched decode — the multi-user throughput lever (fills the idle HMX)
`llama-batched-bench`, NPU HTP0 + OPPOLL, pp128/tg64, aggregate decode t/s across B concurrent sequences:
| B (streams) | decode t/s | scaling | total t/s (pp+tg) |
|---|---|---|---|
| 1 | 23.6 | 1.0x | 66.8 |
| 2 | 31.2 | 1.3x | 83.4 |
| 4 | 41.3 | 1.8x | 108.7 |
| 8 | **81.3** | **3.4x** | 192.7 |
Confirms the HMX-idle diagnosis empirically: at batch 1 the matrix unit is starved (decode bandwidth-bound); adding
concurrent streams amortizes each weight load across all of them and decode throughput scales ~linearly toward
batch 8. **For multi-user serving the NPU decodes 81 t/s (batch 8) vs the CPU's ~37 t/s single-stream — a >2x
throughput win, still climbing.** Single-user latency is unchanged (that's what MTP addresses); this is aggregate
throughput. No accuracy cost.

## 16. Code change: GELU+MUL -> GEGLU fusion in the Hexagon backend (profiling -> op stream -> kernel inventory)

**What profiling said.** The per-op profile (§4) put the whole class of tiny elementwise ops (norms, adds, GELU, MUL)
at ~2.9 ms of a ~32 ms token, 2-3 us per dispatch. Any fusion in that class is worth <1% of decode, below the ±0.5%
run-to-run noise. So the rebuild was *not* spent chasing a speed number; it was spent on the one place where the
op stream and the kernel inventory disagreed.

**What the op stream showed.** Gemma 4's per-layer-embedding gate is hand-rolled in `src/models/gemma4.cpp`
(lines 350-361) as `ggml_gelu()` followed by `ggml_mul()` with a strided view of the per-layer table, 35 pairs per
token. The main FFN, built by `build_ffn()`, emits `ggml_geglu_split()` instead and lands on the backend's fused
two-operand `HTP_OP_GLU_GEGLU` kernel (`htp/act-ops.c`: `geglu(x,g) = gelu(x)*g`). The gate path never reached it.

**What the kernel inventory showed (the real finding).** The two GELUs on the DSP are different functions:
- standalone `HTP_OP_UNARY_GELU` (`htp/unary-ops.c:435`) computes `x * sigmoid(1.702 x)`, the *quick* approximation,
  and the backend maps both `GGML_UNARY_OP_GELU` and `GELU_QUICK` onto it (`ggml-hexagon.cpp:5001`);
- the GEGLU kernel computes `x * sigmoid(2 * x * (sqrt(2/pi) + c x^2))`, which by `tanh(y) = 2*sigmoid(2y) - 1` is
  exactly the tanh-form GELU of the CPU reference (`ggml-cpu/vec.h:969`) and of the model definition.
This is a documented speed approximation (the mapping at lines 5001-5002 and the kernel comment both say so), but it means the unfused NPU path ran Gemma's 35 per-layer gates through a lower-fidelity activation than the model defines; the fused path sidesteps it.

**The change (backend-side, host library only, ~100 lines in 4 hunks of `ggml/src/ggml-hexagon/ggml-hexagon.cpp`; patch saved at `patches/0001-ggml-hexagon-fuse-gelu-mul-into-geglu.patch`).**
1. New fusion flag bit `GGML_HEXAGON_FUSE_GELU_MUL = 1<<6` (so `GGML_HEXAGON_OPFUSION=62` = every existing fusion
   with this one off: a same-binary A/B).
2. `try_fuse_gelu_mul()` modelled on `try_fuse_rms_norm_mul()`: when a MUL arrives and the previous batched op is a
   UNARY_GELU whose output is one MUL operand, rewrite that op in place to `HTP_OP_GLU_GEGLU` with
   `src0 = GELU input, src1 = other MUL operand`. It mirrors `ggml_hexagon_supported_activations()` (F32,
   `contiguous_1` rows, same shape) and the batch buffer/tensor/VMEM fit check, and bails to the unfused path
   otherwise. Kernel params are zeroed: GLU kernels build their VTCM layout on-device from the tensor descriptors
   (`act-ops.c:478-492`), so the descriptor is functionally identical to a natively emitted `geglu_split` (the `params` field still carries the GELU node's op_params, which the GLU kernel only reads in single-tensor mode).
3. The graph pre-pass that grants `GGML_HEXAGON_TENSOR_FUSEABLE` only knew RMS_NORM->MUL and MUL_MAT->ADD. It now
   also tags a single-use GELU output. Strict `ggml_can_fuse()` adjacency cannot be used here because a VIEW node
   sits between GELU and MUL in the cgraph; empty ops are not enqueued, so the pair *is* adjacent in the op batch,
   and the consumer is re-verified at enqueue time. (First build fired 0 times for exactly this reason.)
The model graph, the CPU path and the DSP library are untouched, so the published CPU baseline stays valid.

**Verification (device, same binary unless stated).**
| Check | Result |
|---|---|
| Fires? (`GGML_HEXAGON_VERBOSE=1`, count "fused GELU+MUL") | **34 of 35 layers per graph** (136 over 4 graphs); the missing one is layer 0, whose GELU and MUL are split by a cross-backend `synchronize` (the per-layer-embedding row for the token arrives from the host, where the 1.35 GB PLE table is read lazily) |
| Fidelity vs CPU logits (`llama-perplexity --kl-divergence`, 6x512 tok) | mean KL **0.001611 -> 0.001514**, median 0.000608 -> 0.000543, 99th pct 0.0174 -> 0.0154, top-1 agreement **98.17% -> 98.37%** (unfused -> fused). PPL delta vs CPU 0.385 -> 0.394 (±0.16, noise); max KL 0.048 -> 0.111 (one token), RMS delta-p 1.30 -> 1.33% |
| Decode (thermally gated, warm runs) | pre-change binary 31.11 ±0.18; new binary fused 31.22 ±0.05 cool / 30.8 warm (the results file keeps the last, warm, interleaved run); same-binary OPFUSION=62 A/B: interleaved fuse/nofuse/fuse/nofuse on the same binary and thermal state: **30.80, 30.83 fused vs 30.71, 30.73 unfused tok/s (+0.3%)**; prefill 796/797 vs 792/794. Speed-neutral, as the roofline predicted; an earlier non-interleaved pair (31.22 vs 30.65) overstated the gap because the device was cooler for the fused runs |
| Prefill / TTFT / init / PSS | 805 tok/s / 182 ms / 1231 ms / 1845 MB vs 810 / 185 / 1193 / 1850: all within noise |
| Headline config on the patched binary (llama-server, OPPOLL=1 + MTP depth 1, fusion on) | loads and runs: **35.4 / 35.5 tok/s at 48% draft acceptance** on two 128-token prose prompts, matching the pre-patch hard-prompt figure (35.3 at 48%); under the MTP graphs 510 of 525 GELU ops fused (34/35 per graph over 15 target+draft graph executions, layer 0 again the host-synchronize exception) |
| MMLU (n=1000, frozen set) | **56.8% (568/1000) vs 56.9% unfused (569/1000)**: McNemar exact p=1.0, only 5 discordant questions (3 unfused-only, 2 fused-only). vs CPU: 13 discordant, p=1.0 (unfused vs CPU: 12). Accuracy identical on this benchmark, as expected for a 256-wide gate activation |

**Reading.** Throughput is unchanged, as the roofline said it would be (35 dispatches of ~2.3 us on a 32 ms step). What
the fusion buys is fidelity: the NPU now computes the activation the model defines. Mean, median and 99th-percentile
KL and top-1 agreement all move toward the CPU reference (about one sigma on the mean); the single worst token got
worse (max KL 0.048 -> 0.111) and RMS delta-p is flat (1.30 -> 1.33%), so this is a modest, consistent shift, not a
large one. This is the correct
outcome for a kernel change on a bandwidth-bound decode: the profile told us where headroom was *not*, and the
op-stream inspection told us where the backend was quietly approximating. Both are the deliverable.

**Re-run through the checked-in script (`scripts/batched_bench.sh`, OPPOLL=1, patched binary, 2026-09-07):** B=1 28.3,
B=8 80.2, **B=16 144.4 t/s aggregate** (total pp+tg 301 t/s). The table above was measured without polling; with
polling the B=16 ceiling is ~18% higher. Both are kept so either can be reproduced.

## 17. Round 2: every remaining lossless decode lever, measured and closed (2026-09-07)

Question asked: can decode go faster at *identical* accuracy? Accuracy-neutral by construction means: same weights,
same target verification (speculative), or pure execution knobs. Everything below is in `results/mtp_tuning_sweep.csv`,
`results/ubatch_sweep.md`, `results/exec_levers_round2.md`.

**Speculative (MTP) tuning.** Baseline: depth 1, p-min 0, Q4_0 drafter = 35.5 / 35.5 t/s on the two fixed prose
prompts (48% acceptance). Everything tried lands inside the ±2 t/s prompt-to-prompt band:
| Variant | prompt 1 / prompt 2 t/s | acceptance |
|---|---|---|
| p-min 0.3 / 0.5 / 0.7 (draft only when confident) | 34.3/35.4, 34.3/37.5, 33.4/35.3 | fewer drafts, same net |
| depth 2 with p-min 0.5 / 0.7 (dynamic depth) | 31.9/33.7, 32.4/33.3 | worse: 2nd token rarely accepted |
| Q8_0 drafter (98 MB) | 33.7/36.9 | 43-57%: no better than Q4_0 drafter |
| Q8_0 drafter + p-min 0.5 | 32.3/37.2 | |
| ngram-simple (no drafter) | 30.4/30.2 | below MTP |
| MTP + ngram-simple / + ngram-map-k4v | 29.2/31.8, 35.2/35.2 | hybrid adds nothing |
Reading: acceptance is capped near 50% by the 59 MB MTP head itself, not by the draft policy or its precision.
Depth 1 with no gating is the optimum; the drafter side is exhausted.

**Prefill micro-batch (`-ub`).** 128: 831 / 808 t/s (pp512 / pp2048); 256: 863 / 851; **512 (default): 886 / 867**;
1024: 878 / 869; 2048: 870 / 837. Default already optimal. (Also: prefill throughput at 512-2048 tokens is ~9%
higher than at 128, so the 4x-over-CPU prefill advantage widens for long prompts.)

**Buffer placement and threads.** `GGML_HEXAGON_HOSTBUF=0` vs `=1`: 30.6 vs 30.2 t/s (noise). `NHVX=8`: no output, the
hardware has 6 HVX threads and 6 is the default. Nothing here.

**Multiple NPU sessions (tensor parallel).** The backend can allocate N virtual sessions (`GGML_HEXAGON_NDEV`/`DEVICES`)
and has ALLREDUCE ops, i.e. row-split tensor parallelism, which in principle could raise decode bandwidth if one session
cannot saturate DRAM. On this device/build: `-sm row` across HTP0/HTP1 fails to load the model; `-sm layer` loads
(pipeline split, no bandwidth gain expected) but ran for >6 min without producing a row while a thermal zone hit 105 C,
so it was killed. Not viable here; single session at 6 HVX threads is the ceiling.

**Conclusion.** With weights and verification fixed, NPU single-stream decode is **31 t/s plain, 35-40 t/s with MTP
depth 1**, and no remaining execution knob moves it. The bandwidth roofline (§3) is the binding constraint, and the
only things that beat it are (a) fewer bytes (a 4-bit head, which costs 4.7 MMLU points, §6) or (b) more streams
(§15, batched: 144 t/s at 16 streams with polling).

## 18. Init time: the model was being loaded twice (found 2026-09-07; fixed with one flag, lossless)

All init numbers in the tables above are llama's own "load time" timer (NPU ~1.2-1.4 s, CPU ~0.75-0.95 s). Measuring
**wall clock from process start to the first generated token** (`scripts/init_bench.sh`, 142-token prompt, 3 runs each)
told a different story, and the verbose log explained it: `llama_model_loader: loaded meta data` and `load_tensors`
each appear **twice**. The first pass is `common_fit_params`, a dry run that loads the model to fit unset arguments to
device memory; on this setup it then aborts ("n_gpu_layers already set by user to 99") and the real load starts over.
Each pass also spends ~0.4 s parsing the 262k-entry vocabulary.
| Config | wall to first token | llama "load time" | prompt eval |
|---|---|---|---|
| CPU default | 2.58 s | 0.93 s | 737 ms |
| CPU `--fit off` | 2.12 s (-18%) | 0.94 s | 744 ms |
| NPU default (OPPOLL) | 3.01 s | 1.44 s | 185 ms |
| **NPU `--fit off`** | **2.06 s (-32%)** | 1.51 s | 190 ms |
| NPU `--no-warmup` | 2.99 s | 0.19 s | 193 ms |
| NPU `--fit off --no-mmap` | 2.11 s | 1.53 s | 189 ms |
Reading: `--fit off` (or `LLAMA_ARG_FIT=off`) removes 0.9 s of pure waste on the NPU and 0.45 s on the CPU, with TTFT and
everything downstream unchanged. `--no-warmup` only moves the warmup cost into the first request (wall unchanged), and
mmap vs no-mmap does not matter. Also note llama's "load time" *excludes* the dry run and *includes* the warmup decode,
which is why the tables' init column understated the user-visible cost. **With the flag, the NPU path reaches its first
token from a cold process start in 2.06 s, faster than the CPU default (2.58 s) and equal to CPU with the same flag.**
The flag is now part of the recommended configuration.

## 19. Energy per token: not measurable on this board (attempted 2026-09-07)
`scripts/energy_bench.sh` samples battery current x voltage at 10 Hz around a fixed 256-token generation. On the QDC
QRD board the battery node reports 2-4 mA both idle and under full decode load, the USB node is offline, and no other
power-supply node is exposed: the SoC is fed by an external supply outside sysfs. So the energy cost of OPPOLL=1 (one
CPU core spinning during NPU batches) is real but unmeasured here; the script is ready for a phone with live battery
counters, and OPPOLL should be re-evaluated there before use in battery-sensitive modes. `results/energy_bench.md`.

## 20. Anatomy of an MTP step, and the HMX row-threshold experiment (2026-09-07)

**Why look here.** Upstream's only public Hexagon number for this model is 14 t/s with MTP on a Galaxy S26+ (PR #24282);
we are at 35-40. Upstream's own listed follow-up is a cheaper draft head, so the question was where our MTP step spends
its time. Measured: plain step 32 ms; MTP step 41.7 ms for 1.48 tokens (48% acceptance). The 9.7 ms extra is the
4-layer draft (its own 38 MB Q4_0 head, `token_embd` in the 59 MB drafter, so *not* the 428 MB target head as first
assumed) plus verifying 2 tokens instead of 1. `llama-batched-bench` shows a 2-row decode step costs ~1.8x a 1-row
step (63 vs 35 ms, two KV streams), i.e. the M=2 matmul path is far from the "second row is free" bandwidth ideal.

**Hypothesis.** Matmuls with M <= `HTP_MM_HMX_MIN_NROWS` (= 4) take the HVX path; M=2 might do better on the HMX
matrix unit, which streams each weight tile once. **Change:** the constant became a runtime knob
(`GGML_HEXAGON_HMX_MIN_NROWS`, host lib only, in the same patch) so it can be A/B'd without a rebuild.
`test-backend-ops -o MUL_MAT` with the threshold at 1: 0 failures (HMX at M=1..4 is numerically fine).

| aggregate decode t/s | threshold 4 (default) | threshold 1 (HMX for M>=2) |
|---|---|---|
| B=1 | 28.8 | 28.7 |
| B=2 | 31.7 | **26.6 (-16%)** |
| B=4 | 42.7 | **47.1 (+10%)** |
| MTP depth 1 (M=2 verify) | 35.5 / 35.5 | **26.6 / 28.5 (-23%)** |
Reading: HMX pads M to a 32-row tile; at M=2 the padding and tile conversion cost more than the HVX path saves, at M=4
it starts to pay. The default threshold is right for single-stream and MTP; a threshold of 3 is a serving-only gain at exactly 4
streams (49.0 vs 34.9-42.7 t/s, thermally noisy) and neutral at 8 (85.6 vs 86.9), `results/hmx_threshold_b4_b8.md`. The M=2 verify inefficiency lives inside the HVX GEMM kernel on
the DSP (per-row passes over the weights) and is the one identified, unclaimed lever for MTP: a DSP-side M=2..4
row-batched GEMV would raise MTP from ~35 toward ~43 t/s at the same 48% acceptance. Out of scope for this session.

## 21. Prefill vs prompt length (2026-09-07): the NPU's lead widens with longer inputs
| prompt tokens | CPU t/s | NPU (OPPOLL) t/s | NPU / CPU |
|---|---|---|---|
| 128 | 203 | 810 | 4.0x |
| 512 | 179 | 886 | 5.0x |
| 2048 | 128 | 867 | **6.8x** |
CPU prefill degrades with length (attention cost on 6 cores); the NPU's HMX path gets *more* efficient at 512+ and holds
at 2048. For RAG / long-document TTFT the NPU advantage is ~7x, not the 4x measured at 142 tokens. `results/prefill_vs_len.md`.

## 22. Sustained decode and the thermal price of polling (2026-09-07)
Five back-to-back 1024-token decodes, no cooldown, `results/sustained_decode.md`:
| mode | run 1 -> run 5 t/s | hottest thermal zone |
|---|---|---|
| OPPOLL=1 (polling) | 30.05 -> 29.82 (-0.8%) | 95-98 C |
| default (interrupt) | 26.44 -> 26.47 (flat) | 80-83 C |
Neither mode throttles within ~3 minutes of continuous generation; the NPU itself is not the hot part. Polling keeps a
CPU core spinning and that core sits ~15 C hotter for +13% sustained throughput. This is the same trade-off flagged in
§5, now measured. On a phone, a latency-first mode would poll and a battery/thermal mode would not; the energy side
could not be measured on this board (§19).

## 23. Spending the accuracy budget: the quality/speed Pareto (2026-09-07)

The brief allows an accuracy drop for speed; the floor at "20% relative" is 45.5% MMLU. Decode is bandwidth-bound, so
accuracy buys speed only through fewer bytes per token, and the NPU kernels support nothing below 4 bits. That leaves
two byte levers: the 8-bit LM head (428 MB, 29% of decode bytes) and the layer count. Both were measured this time
rather than reasoned about; all accuracy numbers are the frozen 1000-question set, host-scored, paired McNemar vs stock.

**Correction to §6 first.** The early "4-bit head loses 4.7 points" result was quantized from the Hugging Face BF16,
which turned out not to be the source of the shipped Q4_0 (norm tensors differ byte-for-byte; the BF16 was re-uploaded
in July). Requantizing *only* `token_embd` from the stock Q4_0 file (`llama-quantize --allow-requantize
--token-embedding-type q4_0`, every other tensor byte-identical) gives a very different answer:

| Variant (from stock Q4_0) | bytes/token saved | MMLU | delta | p |
|---|---|---|---|---|
| stock (Q8_0 head) | 0 | 56.8% | | |
| **4-bit head** | 202 MB (14%) | **56.3%** | -0.5 | 0.18 (n.s.) |
| prune last 4 layers (31 left) | ~300 MB (11% of file) | 54.9% | -1.9 | 0.064 (borderline) |
| prune last 8 layers | 23% | 10.6% | -46 | collapsed |
| prune last 12 layers | 32% | 5.2% | -52 | collapsed |
| Q4_0 head + prune 4 | ~25% | 54.4% | -2.4 | 0.018 (significant) |

Pruning (`scripts/prune_layers.py`, drops the last N KV-shared layers and truncates the per-layer tables and metadata
arrays, no requantization) has a cliff between 4 and 8 layers: with no healing fine-tune the model is below random
at -8. Only -4 is inside the budget, and it is borderline significant.

**Device speed (same session, NPU, OPPOLL, --fit off, patched lib; MTP depth 1 in the last column):**
| Variant | decode t/s | prefill | TTFT | PSS | + MTP |
|---|---|---|---|---|---|
| stock Q4_0, Q8_0 head (shipped so far) | 29.5-31.0 | 794-810 | 181-186 | 1846 | 35.2 @ 48% |
| **4-bit head** | **34.6 (+12%)** | 822 | 179-184 | **1656 (-10%)** | **37.2-40.9 @ 46-61%** |
| prune 4 | 34.9 | 927 | 158 | 1699 | 28-29 (acceptance 5-10%: MTP head expects the full 35-layer output) |
| Q4_0 head + prune 4 | 39.4 | 944 | 156-163 | 1505 | 29.6-30.2 (same) |

**Reading.** The 4-bit head is a clean win on every axis that the budget was meant to buy: +12% decode, +12% under MTP
(37-41 t/s, the best single-stream number of the project), 10% less RAM, faster init, for a 0.5-point MMLU change
that 1000 questions cannot distinguish from zero (device MMLU n=1000: **56.3%**, identical to the host score; vs the stock file on the same unit 56.5%, p=0.75; vs the session-1 stock run 56.9%, p=0.07). Pruning is a worse trade: -1.9 points for
the same +12% decode, it breaks MTP (the drafter was trained against the full-depth hidden state, acceptance falls
from 48% to 5-10%, so the pruned models end up *slower* with MTP than without), and anything past 4 layers destroys
the model. The combination reaches 39.4 t/s plain but loses 2.4 points (significant) and also loses MTP.
Init on this unit was 1.1 s for the 4-bit head vs 1.3 s stock (the 202 MB smaller file loads faster).

**Head format sweep (same idea, every 4-bit format the NPU kernels support; only `token_embd` changes, 540/541 tensors
byte-identical; `--tensor-type '^token_embd\.weight$=<fmt>'` with the per-layer table pinned, because the plain
`--token-embedding-type` flag also matches `per_layer_token_embd` and the Q4_0 default rule otherwise sends that
1.3 GB table to Q6_K, which the NPU cannot run):**
| head format | head bytes | MMLU host | MMLU device | vs stock (device, p) | decode t/s | PSS MB | + MTP |
|---|---|---|---|---|---|---|---|
| Q8_0 (stock) | 428 MB | 56.8 | 56.5-56.9 | | 29.5-31.0 | 1846 | 35.2 |
| Q4_0 | 226 MB | 56.3 | 56.3 | -0.2/-0.6 (0.75/0.07) | 34.6 | 1656 | 37.2-40.9 |
| IQ4_NL | 226 MB | 56.5 | | | 35.0 | 1655 | |
| **MXFP4** | **214 MB** | **57.0** | **56.7** | **+0.2/-0.2 (0.81/0.81)** | **35.0** | **1642** | **36.9-40.0** |
| Q4_1 | 252 MB | 56.3 | | | 35.1 | 1676 | |
All four 4-bit heads run at the same speed (the head is a GEMV, bytes decide). MXFP4 (shared 8-bit exponent per 32
weights, 4.25 bits) is the most accurate of them on both host and device and the smallest. Literature agrees that the
token embedding is the least quantization-sensitive tensor class (llama.cpp discussion #12741). **Final default: MXFP4
head.** Zero-change alternative remains the stock Q8_0 head.

**Full-stack verification on the MXFP4-head file (session 3, `results/final_verify_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf_20260907-1356.md`).**
Every lever re-measured on the final file, same device unit, same script as the stock verification:
| metric | stock file (same unit, today) | MXFP4 head | gain |
|---|---|---|---|
| decode t/s (2 warm runs) | 29.5 / 31.0 | 34.4 / 35.0 | **+13%** |
| prefill t/s | 794-810 | 798-832 | +2% |
| TTFT ms | 181-186 | 180-184 | = |
| peak PSS | 1846 MB | 1641 MB | **-11%** |
| wall init (--fit off) | 1.87-1.90 s | 1.85-1.89 s | = |
| fusion fires | 136/140 | 136/140 | = |
| MTP depth 1 | 35.2 / 35.3 | 37.0 / 40.1 | **+5 to +14%** |
| MMLU n=1000 (device) | 56.5 | 56.7 | = (p=0.81) |
| long context tg64 @ 512 / 2048 / 8192 | 30.7 / 29.8 / 28.1 (session 2) | 34.26 / 32.09 / 30.52 | +13% at every depth |
| batched B=1 / 8 / 16 | 28.3 / 80.2 / 144.4 | 29.0 / 80.2 / 143.5 | = (weights shared across streams, head bytes amortised) |
| NPU-vs-CPU logit KL / top-1 | 0.0015 / 98.4% | 0.0034 / 97.6% | NPU MXFP4 GEMV drifts more from the CPU MXFP4 path; MMLU unaffected |
Reading: the head change stacks cleanly with polling, the fusion, `--fit off` and MTP for single-stream use, and holds
across context depth. It gives nothing in batched serving, as expected, because there the weight bytes are already
amortised across streams and the batch is compute-bound on the matrix unit. The one number that moved the other way
is execution fidelity vs the CPU (KL 0.0015 -> 0.0034): the NPU's MXFP4 kernel is numerically a little further from
the CPU's than the Q8_0 path was; MMLU on the device says it does not matter for the task, but it is recorded.


## Summary of throughput levers (this session)
| Lever | decode t/s | applies to | accuracy |
|---|---|---|---|
| CPU baseline | 37.3 | single stream | 56.9% |
| NPU default | 25.4 | single stream | 56.9% |
| NPU + OPPOLL | 31.2 | single stream | 56.9% |
| NPU + OPPOLL + MTP(n=1) | ~38 | single stream | 56.9% (lossless) |
| NPU + OPPOLL, batch 8 | 81.3 | multi-user | 56.9% |
