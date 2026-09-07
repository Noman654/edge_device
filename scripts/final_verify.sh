#!/usr/bin/env bash
# End-of-session re-verification of the shipped configuration on the device currently attached:
#   NPU HTP0, Q4_0 (Q8_0 head), OPPOLL=1, GELU+MUL fusion (patched lib, default), --fit off, MTP depth 1.
# MODEL=<gguf name> selects the target file (default stock Q4_0); SKIP_MMLU=1 skips the 1000-question run.
# Runs, in order: perf x2 (thermally gated), fusion fire count, KL fidelity vs CPU, MTP decode, init, full MMLU (n=1000),
# then prints the recorded-vs-now table. ~25 min. Output: results/final_verify_<date>.md
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
OUT=results/final_verify_${MODEL:-stock}_$(date +%Y%m%d-%H%M).md; exec > >(tee $OUT) 2>&1
echo "# Final verification $(date)   lib sha256: $(adb shell 'sha256sum /data/local/tmp/llama.cpp/lib/libggml-hexagon.so' </dev/null | cut -c1-16)"
for rep in 1 2; do .venv/bin/python scripts/device_bench.py --config htp_final_${MODEL:-stock} --model ${MODEL:-gemma-4-E2B-it-Q4_0.gguf} --dev-args "-dev HTP0 -ngl 99 -t 6 -fa on" --completion-args "--fit off" --env "GGML_HEXAGON_OPPOLL=1" </dev/null | grep -E '"(decode_tps|prefill_tps|ttft_ms|init_ms|peak_pss_mb)"'; done
echo "## fusion"; scripts/fusion_firecount.sh | head -1
echo "## KL vs CPU logits"; scripts/kl_fidelity.sh base >/dev/null; scripts/kl_fidelity.sh final "-dev HTP0 -ngl 99 -t 6 -fa on" "GGML_HEXAGON_OPPOLL=1" | grep -E "Mean +KLD|Same top"
echo "## MTP depth 1"; TARGET=${MODEL:-gemma-4-E2B-it-Q4_0.gguf} scripts/mtp_server_bench.sh 1 | grep -E '^[0-9]'
echo "## init (wall ms, --fit off)"; for r in 1 2 3; do adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_OPPOLL=1; t0=\$EPOCHREALTIME; ./bin/llama-completion -m $DEV_MODELS/${MODEL:-gemma-4-E2B-it-Q4_0.gguf} -f /data/local/tmp/prompt128.txt -n 1 -no-cnv -dev HTP0 -ngl 99 -t 6 -fa on --fit off >/dev/null 2>&1; t1=\$EPOCHREALTIME; echo \"\$t1 \$t0\" | awk '{printf \"%.0f ms\\n\", (\$1-\$2)*1000}'" </dev/null; done
[ -n "${SKIP_MMLU:-}" ] && { echo "## MMLU skipped (SKIP_MMLU set)"; exit 0; }
echo "## MMLU n=1000"; adb shell "pkill -f bin/llama-serve[r]" </dev/null; HEX_ENV="GGML_HEXAGON_OPPOLL=1" scripts/device_server.sh htp_final gemma-4-E2B-it-Q4_0.gguf -c 2048 -np 1 -t 6 -ngl 99 -fa on --fit off </dev/null && .venv/bin/python eval/mmlu_eval.py --url adb://8080 --config htp_final --resume </dev/null | grep -E "1000/1000|acc=" | tail -1; adb shell "pkill -f bin/llama-serve[r]" </dev/null
.venv/bin/python scripts/paired_test.py results/mmlu_htp_oppoll.csv results/mmlu_htp_final.csv
echo "## recorded (report): decode 31.2 | prefill 805-810 | TTFT 182-185 | PSS 1845 | fusion 136/140 | KL 0.001514 / top-1 98.37% | MTP 35.5 @ 48% | init 2.06 s | MMLU 56.8-56.9%"
