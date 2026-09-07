#!/usr/bin/env bash
# Decode rate vs context depth (does Gemma's KV-sharing keep decode flat?). tg64 at depth 512/2048/8192.
# Usage: scripts/longctx_bench.sh -> results/longctx_bench.md
set -uo pipefail
cd "$(dirname "$0")/."; cd ..; source scripts/device_env.sh
adb shell "$DEV_ENV GGML_HEXAGON_OPPOLL=1 ./bin/llama-bench -m $DEV_MODELS/${MODEL:-gemma-4-E2B-it-Q4_0.gguf} -dev HTP0 -ngl 99 -fa on -t 6 -p 0 -n 64 -d 512,2048,8192 -r 3 -o md 2>/dev/null" </dev/null | tee results/longctx_bench.md
