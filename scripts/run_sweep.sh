#!/usr/bin/env bash
# Perf-only sweep from scripts/sweep.tsv (2 runs each, warm kept, thermal-gated inside device_bench.py).
set -uo pipefail
cd "$(dirname "$0")/.."
grep -v '^#' scripts/sweep.tsv | grep -v '^\s*$' | while IFS=$'\t' read -r name model dev env; do
  [ "$env" = "-" ] && env=""
  echo "=================== $name  ($(date +%H:%M))"
  for rep in 1 2; do
    .venv/bin/python scripts/device_bench.py --config "$name" --model "$model" --dev-args "$dev" --env "$env" </dev/null 2>&1 | grep -vE "raw_tail" | grep -E "thermal at start|\"decode_tps\"|\"prefill_tps\"|\"ttft_ms\"|\"init_ms\"|\"peak_pss_mb\"" || echo "FAIL $name"
  done
done
echo "SWEEP_EXIT $(date)"
