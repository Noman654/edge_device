#!/usr/bin/env bash
# Bandwidth anchor: GEMV (n=1) throughput on HTP0 vs CPU via test-backend-ops perf, converted to GB/s.
# Q4_0 = 0.5625 B/param, Q8_0 = 1.0625 B/param. Shape in the built-in perf set: m=4096 k=14336.
set -uo pipefail
source "$(dirname "$0")/device_env.sh"
for b in HTP0 CPU; do
  echo "== $b"
  adb shell "$DEV_ENV GGML_HEXAGON_HOSTBUF=0 timeout 300 ./bin/test-backend-ops perf -b $b -o MUL_MAT 2>&1" | grep -E "type_a=(q4_0|q8_0),type_b=f32,m=4096,n=1,k=14336" | sed 's/\x1b\[[0-9;]*m//g' | \
  awk '{ match($0,/type_a=[a-z0-9_]+/); t=substr($0,RSTART+7,RLENGTH-7); match($0,/[0-9.]+ us\/run/); us=substr($0,RSTART,RLENGTH-7)+0; bpp=(t=="q4_0")?0.5625:1.0625; mb=4096*14336*bpp/1e6; printf "  %-5s %8.1f us/run  %6.1f MB  %6.1f GB/s\n", t, us, mb, mb/us*1000 }'
done
