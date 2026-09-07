#!/usr/bin/env bash
# End-to-end latency: TTFT + decode for a fixed 142-token prompt at several output lengths, CPU vs NPU+OPPOLL.
# Answers "end-to-end latency where relevant" and locates the decode crossover.
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
OUT=results/latency_sweep.csv; echo "config,n_predict,prompt_ms,eval_ms,total_ms" > $OUT
adb shell "pkill -f llama" 2>/dev/null; sleep 1
for cfg in "cpu:-dev none -ngl 0 -t 6:" "npu_oppoll:-dev HTP0 -ngl 99 -t 6 -fa on:GGML_HEXAGON_OPPOLL=1"; do
  name=${cfg%%:*}; rest=${cfg#*:}; args=${rest%%:*}; env=${rest#*:}
  for n in 8 32 128 256; do
    log=$(adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib $env; ./bin/llama-completion -m $DEV_MODELS/gemma-4-E2B-it-Q4_0.gguf -f /data/local/tmp/prompt128.txt -n $n -no-cnv --temp 0 --ignore-eos $args 2>&1 | grep -E 'prompt eval time|eval time|total time'")
    p=$(echo "$log" | grep 'prompt eval' | grep -oE '= *[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
    e=$(echo "$log" | grep -E '^\S+ +I +.*eval time' | grep -v prompt | grep -oE '= *[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
    t=$(echo "$log" | grep 'total time' | grep -oE '= *[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')
    echo "$name,$n,$p,$e,$t" | tee -a $OUT
  done
done
