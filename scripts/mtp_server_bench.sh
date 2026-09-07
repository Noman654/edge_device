#!/usr/bin/env bash
# MTP (speculative) decode on the NPU, measured through llama-server timings (llama-bench cannot load a drafter).
# The draft MUST run on the same device as the target (no --spec-draft-device none): Gemma's MTP head shares the KV context.
# Usage: scripts/mtp_server_bench.sh [n_draft=1] [extra env, e.g. GGML_HEXAGON_OPFUSION=62]
#   optional env: TARGET=<target gguf name>  SPEC_EXTRA="--spec-draft-p-min 0.5"   DRAFT=mtp-gemma-4-E2B-it-Q8_0.gguf
# Appends to results/mtp_server_bench.csv: n_draft, env, prompt#, predicted_n, decode_tps, draft_n, draft_accepted
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
N=${1:-1}; EXTRA=${2:-}; OUT=results/mtp_server_bench.csv
[ -f $OUT ] || echo "n_draft,env,prompt,predicted_n,decode_tps,draft_n,draft_accepted" > $OUT
TARGET=${TARGET:-gemma-4-E2B-it-Q4_0.gguf}; DRAFT=${DRAFT:-mtp-gemma-4-E2B-it-Q4_0.gguf}; SPEC_EXTRA=${SPEC_EXTRA:-}; SPEC_TYPE=${SPEC_TYPE:-draft-mtp}
MTP="--model-draft $DEV_MODELS/$DRAFT --spec-type $SPEC_TYPE --spec-draft-n-max $N $SPEC_EXTRA"
adb shell "pkill -f bin/llama-serve[r]" </dev/null; sleep 1
adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_OPPOLL=1 $EXTRA; setsid sh -c './bin/llama-server --no-mmap -m $DEV_MODELS/$TARGET $MTP -ngl 99 -dev HTP0 -fa on -t 6 --ctx-size 4096 --host 127.0.0.1 --port 8080 > /data/local/tmp/srv_mtp.log 2>&1' </dev/null >/dev/null 2>&1 &" </dev/null
for i in $(seq 1 40); do adb shell "curl -s -m 5 http://127.0.0.1:8080/health" </dev/null 2>/dev/null | grep -q ok && break; sleep 3; done
i=0
for P in "Write a short story about a lighthouse keeper who discovers a message in a bottle." "Explain how photosynthesis works in plants, step by step."; do
  i=$((i+1))
  adb shell "curl -s -m 180 http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d '{\"prompt\":\"$P\",\"n_predict\":128,\"temperature\":0,\"cache_prompt\":false}'" </dev/null \
   | python3 -c "import sys,json; t=json.load(sys.stdin)['timings']; print(f\"$N,{'$TARGET $EXTRA $DRAFT $SPEC_TYPE $SPEC_EXTRA'.strip()},$i,{t['predicted_n']},{t['predicted_per_second']:.2f},{t.get('draft_n')},{t.get('draft_n_accepted')}\")" | tee -a $OUT
done
adb shell "pkill -f bin/llama-serve[r]" </dev/null
