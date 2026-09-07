# NPU decode bottleneck analysis (Q4_0 default, HTP0, per-op profile n=32+1 prefill)

Clean bench: NPU decode 25.2 tok/s (39.7 ms/token) vs CPU 37.3 tok/s (26.8 ms/token). NPU is SLOWER at decode.
MMLU bit-comparable: CPU 56.9% vs HTP 56.9%, McNemar p=1.0, 6/6 discordant -> NPU numerics correct, 36/36 layers offloaded.
TTFT/prefill: HTP 195 ms / 775 tok/s vs CPU 725 ms / 202 tok/s -> NPU wins prefill 3.7x.

## Per-token decode budget (Tot usec / 32 decode steps)
| Component | per-token ms | note |
|---|---|---|
| FFN matmuls (up/gate NX + down) | ~16.1 | 1033 MB body / 22.5 ms = 46 GB/s = 90% of the 51 GB/s GEMV anchor |
| LM head MUL_MAT q8_0 1536x262144 | 7.4 | 428 MB / 7.4 ms = 58 GB/s = at roofline |
| attention matmuls (q/k/v/o) | ~2.7 | |
| norms/adds/glu/rope/gelu (100s of small ops) | ~2.9 | 2-3 us each |
| flash-attn-ext | ~0.8 | |
| tanh+scale over full 262144 vocab (logit softcap) | ~0.4 | |
| **sum of NPU-busy time** | **~31.8** | both OPBATCH submissions |
| **clean step** | **39.7** | |
| **host-side orchestration (step - busy)** | **~7-8** | **the only NPU decode headroom** |

## Diagnosis
Matmuls already run at 90% of achievable bandwidth; there is NO body-kernel headroom. The ~7-8 ms/token gap vs
NPU-busy time is host-side: the decode graph submits as **two OPBATCH batches per token** (800 ops, then 23),
i.e. two FastRPC round trips + interrupt-mode completion wakeup (OPPOLL=0 default) + 800-op descriptor build +
logits readback. CONFIRMED cause of the 2-batch split (read from ggml-hexagon.cpp enqueue_op/fit_op): a batch flushes when the
next op would exceed the batch buffer/tensor/VMEM budget (NOT op count; 800 < the 1280 cap). The LM head
(428 MB Q8_0 weight + 1 MB full-vocab f32 output) does not fit alongside the 35-layer body, forcing flush ->
second batch (23 ops). So the split is a VMEM-budget effect. (Note: GET_ROWS IS implemented on HTP now, and at
decode n_tokens==n_outputs so the inp_out_ids get_rows is skipped entirely - earlier get_rows theory was wrong.)

## Levers, by expected payoff (decode)
1. OPPOLL=1 (poll instead of interrupt wakeup) - directly attacks the round-trip latency, no code. Tested in sweep.
2. Merge the two batches: raise GGML_HEXAGON_VMEM so the head fits with the body (config, no code) -> fewer FastRPC round trips.
3. NHVX sweep: small ops may pay sync cost across 6 HVX threads.
Ceiling reality: even at zero host overhead the NPU decode ties the CPU (both bandwidth-bound ~35-37 tok/s).
The NPU's real, large win is TTFT/prefill (3.7x). Honest "best": NPU for prefill-heavy/short-output (MMLU is
exactly this); report the decode crossover length where CPU overtakes for long generations.

## Peak RAM (from verbose): CPU buffer 1694 MiB + HTP buffer 1409 MiB = 3103 MiB (file is 2826 MB).
The ~430 MB excess is the tied token_embd allocated on BOTH sides (CPU for the embedding lookup, HTP repacked for
the head matmul). Expect NPU peak PSS >= CPU's 5.1 GB. Memory-path finding, not a harness bug.

## Knob sweep results (warm, thermally gated, QRD) — decode tok/s
| Config | decode | prefill | TTFT ms | vs default decode |
|---|---|---|---|---|
| HTP default | 25.4 | 793 | 190 | - |
| **HTP + OPPOLL=1** | **31.2** | 810 | 185 | **+23%** |
| HTP + NHVX=4 | 26.8 | 722 | 205 | +6% |
| HTP + NHVX=2 | 17.5 | 540 | 255 | -31% (starves HVX) |
| HTP + OPPOLL=1 + NHVX=4 | 31.2 | 735 | 203 | +23% (NHVX adds nothing over poll) |
| HTP + MM_SELECT=2 (HVX tiled, no HMX) | 24.0 | 76 | 1666 | prefill collapses 10x -> HMX is essential |
| HTP + FA_SELECT=1 (HVX flash-attn) | 25.0 | 367 | 379 | prefill halves -> HMX flash-attn matters |
| HTP + KV q8_0 | 23.4 | 749 | 205 | -8% |
| HTP + lm_head on CPU | 24.8 | 788 | 187 | ~0 (head is NOT the decode bottleneck) |

## Verdict
**OPPOLL=1 is the win: +23% decode (25.4 -> 31.2 tok/s), for free, no code, no quality change.** It replaces
interrupt-mode completion wakeup with polling, directly removing the per-batch FastRPC round-trip latency that the
profile attributed to host orchestration. This confirms the bottleneck diagnosis empirically. Best config = HTP +
OPPOLL=1: decode 31.2 (closes the CPU gap from 12 to 6 tok/s), prefill 810 (4x CPU), TTFT 185 ms (3.9x CPU),
host RAM 1.85 GB (vs CPU 5.1 GB). HMX must stay on (MM_SELECT>=3, default) - disabling it destroys prefill.
Cost of polling: one CPU core spins during NPU compute (higher power); acceptable for throughput, note in report.
