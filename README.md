# Gemma 4 E2B on the Snapdragon 8 Elite Hexagon NPU

Profiling-driven inference optimization of Gemma 4 E2B on the Hexagon NPU (Snapdragon 8 Elite / SM8750, the Galaxy S25
silicon) with llama.cpp's `ggml-hexagon` backend. On the same 1000 MMLU questions, NPU decode went from 25 to 37-40
tokens/s at unchanged accuracy, with prefill 4.1x, time-to-first-token 4.0x and peak RAM 3.1x better than the CPU
baseline. The work includes a bandwidth roofline and per-op profile, a fusion patch in the backend, speculative decoding
with the model's multi-token head, and a measured quality-versus-speed study across quantization formats and pruning.

## Start here
- **[`report/REPORT.md`](report/REPORT.md)**: results table, how the bottlenecks were found, each fix with its number,
  the quality/speed trade-off, two figures. **[`report/FINDINGS.md`](report/FINDINGS.md)** is the full-depth appendix.
- **[`REPRODUCE.md`](REPRODUCE.md)** rebuilds every number from pinned inputs (commit, model hashes, patch, scripts with
  expected outputs).

The final model is one command on the ggml-org Q4_0 file:
```bash
llama-quantize --allow-requantize --tensor-type '^token_embd\.weight$=mxfp4' --tensor-type '^per_layer_token_embd\.weight$=q4_0' \
  gemma-4-E2B-it-Q4_0.gguf gemma-4-E2B-it-Q4_0-head_mxfp4.gguf Q4_0
```
Final launch: `-dev HTP0 -ngl 99 -t 6 -fa on --fit off` with `GGML_HEXAGON_OPPOLL=1` and the MTP drafter at depth 1,
on the patched backend (`patches/`).

## Layout
- `report/` — REPORT.md, FINDINGS.md, figures.
- `REPRODUCE.md`, `VERSIONS.md`, `patches/` — pinned inputs, the backend patch, and the ordered procedure.
- `eval/` — the frozen 1000-question MMLU set (`mmlu_1000_seed42.jsonl`), the scorer, the TTFT prompt, the fidelity text.
- `scripts/` — device deploy/bench/profile harness, sweeps, MTP/batched/long-context/latency/init/energy benches,
  `paired_test.py` (McNemar), `prune_layers.py`, `imatrix_quant.sh`, `final_verify.sh`.
- `results/` — every run's JSON/CSV/MD output (per-config files plus `history/`), MMLU per-question CSVs.
- `artifacts/` — the exact pre-patch and patched `libggml-hexagon.so` measured on the device.
- `models/` — only `SHA256SUMS`; model files are fetched by `scripts/get_models.sh`.
