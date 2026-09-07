#!/usr/bin/env bash
# Profile the HTP path: (1) verbose log -> offload/fallback/buffer placement, (2) per-op profile for decode and prefill.
# Usage: scripts/device_profile.sh [tag] [model.gguf] [extra env]
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
TAG=${1:-htp_q4_0_default}; MODEL=${2:-gemma-4-E2B-it-Q4_0.gguf}; ENV=${3:-}
ARGS="-m $DEV_MODELS/$MODEL -f /data/local/tmp/prompt128.txt -no-cnv --temp 0 --ignore-eos -v -dev HTP0 -ngl 99 -fa on -t 6"
mkdir -p results/profile
echo "== verbose (n=4)"; adb shell "$DEV_ENV $ENV GGML_HEXAGON_VERBOSE=1 ./bin/llama-completion $ARGS -n 4" > results/profile/${TAG}_verbose.log 2>&1
grep -E "offloaded|model buffer size|KV buffer|falling back|not supported|unsupported|CPU buffer" results/profile/${TAG}_verbose.log | head -12
echo "-- ops per graph-compute (decode steps):"; grep -c "graph-compute" results/profile/${TAG}_verbose.log
grep -oE "HTP0 [a-z_]+ :" results/profile/${TAG}_verbose.log | sort | uniq -c | sort -rn | head -15
echo "== per-op profile (n=32)"; adb shell "$DEV_ENV $ENV GGML_HEXAGON_PROFILE=1 ./bin/llama-completion $ARGS -n 32" > results/profile/${TAG}_profile_raw.log 2>&1
./llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py results/profile/${TAG}_profile_raw.log > results/profile/${TAG}_profile.txt 2>&1 || ./llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py - < results/profile/${TAG}_profile_raw.log > results/profile/${TAG}_profile.txt 2>&1
head -60 results/profile/${TAG}_profile.txt
grep -E "prompt eval time|eval time|load time" results/profile/${TAG}_profile_raw.log
