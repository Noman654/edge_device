#!/usr/bin/env bash
# Energy per generated token: samples battery current*voltage at 10 Hz on the device during a fixed 256-token greedy
# generation, for CPU, NPU default, NPU polling, and NPU polling + MTP(1). Output: results/energy_bench.csv
# Columns: config, tokens, gen_s, decode_tps, mean_W, idle_W, net_mJ_per_token
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/device_env.sh
OUT=results/energy_bench.csv; echo "config,tokens,gen_s,decode_tps,mean_W,idle_W,net_mJ_per_token" > $OUT
adb shell "cat > /data/local/tmp/psample.sh" </dev/null <<'EOS'
#!/system/bin/sh
# usage: psample.sh <outfile>   (runs until killed) ; columns: t_ns current_uA voltage_uV
while true; do echo "$(date +%s%N) $(cat /sys/class/power_supply/battery/current_now) $(cat /sys/class/power_supply/battery/voltage_now)"; sleep 0.1; done > $1
EOS
adb shell "chmod +x /data/local/tmp/psample.sh" </dev/null
meanW() { adb shell "cat $1" </dev/null | awk '{p=($2<0?-$2:$2)/1e6*$3/1e6; s+=p; n++} END{printf "%.3f", (n?s/n:0)}'; }
# idle
adb shell "setsid /data/local/tmp/psample.sh /data/local/tmp/p_idle.txt </dev/null >/dev/null 2>&1 & sleep 6; pkill -f psample.s[h]" </dev/null
IDLE=$(meanW /data/local/tmp/p_idle.txt); echo "idle_W=$IDLE"
run_completion() { # name, env, dev-args
  adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib $2; setsid /data/local/tmp/psample.sh /data/local/tmp/p_$1.txt </dev/null >/dev/null 2>&1 & ./bin/llama-completion -m $DEV_MODELS/gemma-4-E2B-it-Q4_0.gguf -f /data/local/tmp/prompt128.txt -n 256 --ignore-eos -no-cnv --temp 0 $3 > /data/local/tmp/gen_$1.log 2>&1; pkill -f psample.s[h]" </dev/null
  adb shell "grep 'eval time' /data/local/tmp/gen_$1.log | grep -v prompt" </dev/null | tr -d '\r' | python3 -c "
import sys,re; l=sys.stdin.read(); m=re.search(r'=\s*([0-9.]+) ms /\s*([0-9]+) runs.*?([0-9.]+) tokens per second', l)
ms,toks,tps=float(m.group(1)),int(m.group(2)),float(m.group(3)); W=float('$(meanW /data/local/tmp/p_$1.txt)'); I=float('$IDLE'); s=ms/1000
print(f'$1,{toks},{s:.2f},{tps:.2f},{W:.3f},{I:.3f},{(W-I)*s/toks*1000:.1f}')" | tee -a $OUT
}
run_completion cpu "" "-dev none -ngl 0 -t 6"
run_completion npu_default "" "-dev HTP0 -ngl 99 -t 6 -fa on"
run_completion npu_oppoll "GGML_HEXAGON_OPPOLL=1" "-dev HTP0 -ngl 99 -t 6 -fa on"
# MTP via server
adb shell "pkill -f bin/llama-serve[r]" </dev/null; sleep 1
adb shell "cd $DEV_DIR; export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib GGML_HEXAGON_OPPOLL=1; setsid sh -c './bin/llama-server --no-mmap -m $DEV_MODELS/gemma-4-E2B-it-Q4_0.gguf --model-draft $DEV_MODELS/mtp-gemma-4-E2B-it-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 1 -ngl 99 -dev HTP0 -fa on -t 6 --ctx-size 4096 --host 127.0.0.1 --port 8080 > /data/local/tmp/srv_e.log 2>&1' </dev/null >/dev/null 2>&1 &" </dev/null
for i in $(seq 1 40); do adb shell "curl -s -m 5 http://127.0.0.1:8080/health" </dev/null 2>/dev/null | grep -q ok && break; sleep 3; done
P="Explain how photosynthesis works in plants, step by step, in detail."
adb shell "setsid /data/local/tmp/psample.sh /data/local/tmp/p_mtp.txt </dev/null >/dev/null 2>&1 & curl -s -m 300 http://127.0.0.1:8080/completion -H 'Content-Type: application/json' -d '{\"prompt\":\"$P\",\"n_predict\":256,\"temperature\":0,\"ignore_eos\":true}' > /data/local/tmp/gen_mtp.json; pkill -f psample.s[h]" </dev/null
adb shell "cat /data/local/tmp/gen_mtp.json" </dev/null | python3 -c "
import sys,json; t=json.load(sys.stdin)['timings']; W=float('$(meanW /data/local/tmp/p_mtp.txt)'); I=float('$IDLE'); s=t['predicted_ms']/1000; n=t['predicted_n']
print(f\"npu_oppoll_mtp1,{n},{s:.2f},{t['predicted_per_second']:.2f},{W:.3f},{I:.3f},{(W-I)*s/n*1000:.1f}\")" | tee -a $OUT
adb shell "pkill -f bin/llama-serve[r]" </dev/null
