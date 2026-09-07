#!/usr/bin/env bash
# Multi-sequence decode throughput on the NPU (fills the HMX that is idle at batch 1). pp128 / tg64 per sequence.
# Usage: scripts/batched_bench.sh [batch sizes...]   (default 1 2 4 8 12 16) -> results/batched_bench.md
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
OUT=results/batched_bench.md; : > $OUT
for b in ${@:-1 2 4 8 12 16}; do
  echo "== B=$b" | tee -a $OUT
  adb shell "$DEV_ENV GGML_HEXAGON_OPPOLL=1 ./bin/llama-batched-bench -m $DEV_MODELS/${MODEL:-gemma-4-E2B-it-Q4_0.gguf} -dev HTP0 -ngl 99 -fa on -t 6 -c 8192 -npp 128 -ntg 64 -npl $b 2>&1 | grep -vE 'ggml-hex|^$' | tail -3" </dev/null | tee -a $OUT
done
