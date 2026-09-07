#!/usr/bin/env bash
# Start llama-server on device with a config. QDC blocks host->device port forwards, so the scorer talks to it via `adb shell curl` (adb://8080).
# Usage: scripts/device_server.sh <config-name> <model.gguf> [extra llama-server args...]
#   ENV: HEX_ENV="GGML_HEXAGON_MM_SELECT=3 ..." to pass backend env vars.
set -euo pipefail
source "$(dirname "$0")/device_env.sh"
CFG="$1"; MODEL="$2"; shift 2
adb shell "pkill -f llama-server" || true
# setsid + stdin from /dev/null: otherwise `adb shell` blocks for the lifetime of the backgrounded server
adb shell "setsid sh -c 'cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib ${HEX_ENV:-}; ./bin/llama-server -m $DEV_MODELS/$MODEL --host 127.0.0.1 --port 8080 $*' </dev/null >/data/local/tmp/server_$CFG.log 2>&1 &"
for i in $(seq 1 120); do
  adb shell "curl -sf -m 5 http://127.0.0.1:8080/health" 2>/dev/null | grep -q ok && { echo "server up ($CFG)"; exit 0; }
  sleep 1
done
echo "server failed to start; log:"; adb shell "tail -50 /data/local/tmp/server_$CFG.log"; exit 1
