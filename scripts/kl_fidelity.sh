#!/usr/bin/env bash
# Numerical fidelity of an NPU config vs the CPU reference: perplexity + KL divergence of logits on a fixed text
# (eval/ppl_text.txt, 6 chunks x 512 tokens). Step 1 saves CPU logits once; each later call scores one config.
# Usage: scripts/kl_fidelity.sh base                      -> /data/local/tmp/cpu_logits.bin + results/kl_cpu_base.txt
#        scripts/kl_fidelity.sh <label> "<dev args>" "<env>"   e.g.  fused "-dev HTP0 -ngl 99 -t 6 -fa on" "GGML_HEXAGON_OPPOLL=1"
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
adb push eval/ppl_text.txt /data/local/tmp/ppl_text.txt >/dev/null
COMMON="-m $DEV_MODELS/${MODEL:-gemma-4-E2B-it-Q4_0.gguf} -f /data/local/tmp/ppl_text.txt -c 512 -b 512 --chunks 6"
if [ "${1:-}" = base ]; then
  adb shell "$DEV_ENV ./bin/llama-perplexity $COMMON --kl-divergence-base /data/local/tmp/cpu_logits.bin -dev none -ngl 0 -t 6 2>&1 | grep -E 'Final estimate'" </dev/null | tee results/kl_cpu_base.txt
else
  L=$1; DA=$2; ENV=${3:-}
  adb shell "$DEV_ENV $ENV ./bin/llama-perplexity $COMMON --kl-divergence-base /data/local/tmp/cpu_logits.bin --kl-divergence $DA 2>&1 | grep -E 'Mean +KLD|Median +KLD|99.0%   KLD|Maximum KLD|RMS|Same top|PPL\(Q\)-'" </dev/null | tee results/kl_$L.txt
fi
