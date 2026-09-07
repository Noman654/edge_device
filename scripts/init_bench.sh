#!/usr/bin/env bash
# Model-init time: wall clock from process start to first generated token (142-token prompt, n=1), plus llama's own
# "load time" and prompt-eval time, 3 runs each. Variants: default, --fit off (skips the fit-params dry run that loads
# the model twice), --no-warmup, --no-mmap. Output: results/init_bench.csv
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
OUT=results/init_bench.csv; echo "config,run,wall_ms,load_ms,prompt_eval_ms" > $OUT
run() { # name, dev-args
  for r in 1 2 3; do
    adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_OPPOLL=1; t0=\$EPOCHREALTIME; ./bin/llama-completion -m $DEV_MODELS/gemma-4-E2B-it-Q4_0.gguf -f /data/local/tmp/prompt128.txt -n 1 -no-cnv $2 > /data/local/tmp/i.log 2>&1; t1=\$EPOCHREALTIME; echo \"\$(echo \"\$t1 \$t0\" | awk '{printf \"%.0f\", (\$1-\$2)*1000}') \$(grep -o 'load time = *[0-9.]*' /data/local/tmp/i.log | grep -o '[0-9.]*$') \$(grep -o 'prompt eval time = *[0-9.]*' /data/local/tmp/i.log | grep -o '[0-9.]*$')\"" </dev/null | tr -d '\r' | awk -v n=$1 -v r=$r '{printf "%s,%d,%s,%s,%s\n", n, r, $1, $2, $3}' | tee -a $OUT
  done
}
run cpu_default        "-dev none -ngl 0 -t 6"
run cpu_fit_off        "-dev none -ngl 0 -t 6 --fit off"
run npu_default        "-dev HTP0 -ngl 99 -t 6 -fa on"
run npu_fit_off        "-dev HTP0 -ngl 99 -t 6 -fa on --fit off"
run npu_no_warmup      "-dev HTP0 -ngl 99 -t 6 -fa on --no-warmup"
run npu_fit_off_nommap "-dev HTP0 -ngl 99 -t 6 -fa on --fit off --no-mmap"
