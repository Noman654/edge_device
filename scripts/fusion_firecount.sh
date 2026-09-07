#!/usr/bin/env bash
# Count how many GELU+MUL pairs the backend fuses per graph (proof the patch is active on the device).
# Usage: scripts/fusion_firecount.sh [extra env]   e.g. GGML_HEXAGON_OPFUSION=62 to prove it can be switched off
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
adb shell "$DEV_ENV GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_VERBOSE=1 ${1:-} ./bin/llama-completion -m $DEV_MODELS/${MODEL:-gemma-4-E2B-it-Q4_0.gguf} -p 'The capital of France is' -n 4 -no-cnv --temp 0 -dev HTP0 -ngl 99 -t 6 -fa on --verbose > /data/local/tmp/fire.log 2>&1; echo fused_gelu_mul=\$(grep -c 'fused GELU+MUL' /data/local/tmp/fire.log) gelu_ops_added=\$(grep -c 'execute-op GELU|' /data/local/tmp/fire.log) rms_norm_mul_fused=\$(grep -c 'fused RMS_NORM+MUL' /data/local/tmp/fire.log)" </dev/null
echo "expect fused = 34/35 of gelu ops (layer 0 is split by a host synchronize); 0 with GGML_HEXAGON_OPFUSION=62"
