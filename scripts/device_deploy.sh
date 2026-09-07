#!/usr/bin/env bash
# One-shot device setup after the QDC tunnel is up. Idempotent (adb push skips unchanged files).
# Usage: scripts/device_deploy.sh [model.gguf ...]   (default: Q4_0 + MTP draft)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/device_env.sh
echo "== platform"; adb shell getprop ro.board.platform; adb shell getprop ro.product.model; adb shell getprop ro.build.version.release
echo "== hexagon arch"; adb shell "cat /sys/devices/soc0/soc_id 2>/dev/null; getprop ro.soc.model"
echo "== mem/cpu"; adb shell "grep MemTotal /proc/meminfo; nproc"
echo "== push package"; adb shell "mkdir -p $DEV_DIR" && adb push llama.cpp/pkg-adb/llama.cpp/bin llama.cpp/pkg-adb/llama.cpp/lib $DEV_DIR/ >/dev/null && adb shell "chmod -R 755 $DEV_DIR/bin $DEV_DIR/lib" && echo ok
adb shell "mkdir -p $DEV_MODELS"
for m in "${@:-models/gemma-4-E2B-it-Q4_0.gguf models/mtp-gemma-4-E2B-it-Q4_0.gguf}"; do
  for f in $m; do echo "== push $f"; adb push "$f" $DEV_MODELS/ | tail -1; done
done
adb push eval/prompt128.txt /data/local/tmp/prompt128.txt >/dev/null
echo "== smoke: backend registry"; adb shell "$DEV_ENV ./bin/llama-bench --list-devices 2>&1 | tail -8"
echo "== smoke: HTP op test (MUL_MAT q4_0)"; adb shell "$DEV_ENV GGML_HEXAGON_HOSTBUF=0 ./bin/test-backend-ops -b HTP0 -o MUL_MAT 2>&1 | grep -E 'q4_0.*n=1,|OK|FAIL|Backend' | head -6"
