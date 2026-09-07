#!/usr/bin/env bash
# Run perf (+ MMLU per the config's mmlu column) for each config in scripts/configs.tsv. One device, one job at a time.
# Usage: scripts/run_matrix.sh [--perf-only] [name-regex]
set -uo pipefail
cd "$(dirname "$0")/.."
PERF_ONLY=0; PAT=".*"
while [ $# -gt 0 ]; do case "$1" in --perf-only) PERF_ONLY=1;; *) PAT="$1";; esac; shift; done
grep -v '^#' scripts/configs.tsv | grep -v '^\s*$' | while IFS=$'\t' read -r name model dev env comp mmlu extra; do
  [[ "$name" =~ $PAT ]] || continue
  [ "$env" = "-" ] && env=""; [ "$comp" = "-" ] && comp=""
  skip=""; [ -n "$comp" ] && skip="--skip-bench"
  echo "=================== $name  ($(date +%H:%M))"
  for rep in 1 2; do   # run twice, keep the warm (second) run
    .venv/bin/python scripts/device_bench.py --config "$name" --model "$model" --dev-args "$dev" --env "$env" --completion-args "$comp" $skip </dev/null || { echo "PERF FAILED: $name"; break; }
  done
  # cooldown is enforced inside device_bench.py (waits until max thermal zone <= 55C, up to 5 min)
  if [ "$PERF_ONLY" = 0 ] && [ "$mmlu" != "no" ]; then
    lim=""; [ "$mmlu" = "sanity" ] && lim="--limit 100"
    HEX_ENV="$env" scripts/device_server.sh "$name" "$model" $extra || { echo "SERVER FAILED: $name"; continue; }
    .venv/bin/python eval/mmlu_eval.py --url adb://8080 --config "$name" --resume $lim </dev/null || echo "MMLU FAILED: $name"
    adb shell "pkill -f llama-server" || true
  fi
done
