#!/usr/bin/env bash
# Quantization variants from the BF16 source. HTP-native types only (Q4_0 / Q8_0 / IQ4_NL).
# token_embd.weight is tied to the LM head in Gemma 4 (no output.weight): --token-embedding-type sets the head type too.
# Result so far (Mac MMLU n=1000): Q4_0 body + Q8_0 head 56.8% | Q4_0 body + Q4_0 head 52.1% (too lossy) -> try IQ4_NL head.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=./llama.cpp/build-mac/bin/llama-quantize; SRC=models/gemma-4-E2B-it-BF16.gguf
build() { [ -s "models/gemma-4-E2B-it-$2.gguf" ] && { echo "have $2"; return; }; $Q $1 "$SRC" "models/gemma-4-E2B-it-$2.gguf" $3 2>&1 | tail -2; sync; }
build "--token-embedding-type q4_0"   "Q4_0-embq4"    Q4_0
build ""                              "IQ4_NL"        IQ4_NL
build "--token-embedding-type q4_0"   "IQ4_NL-embq4"  IQ4_NL
build "--token-embedding-type iq4_nl" "Q4_0-embiq4"   Q4_0
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0,"llama.cpp/gguf-py")
from gguf import GGUFReader
bad=0
for f in ["Q4_0","Q4_0-embq4","Q4_0-embiq4","IQ4_NL","IQ4_NL-embq4","Q8_0"]:
    try:
        r=GGUFReader(f"models/gemma-4-E2B-it-{f}.gguf"); t={x.name:x for x in r.tensors}
        print(f"{f:14s} token_embd={t['token_embd.weight'].tensor_type.name:7s} ffn_up={t['blk.0.ffn_up.weight'].tensor_type.name:7s} total={sum(int(x.n_bytes) for x in r.tensors)/1e6:.0f}MB")
    except Exception as e: print(f"{f:14s} BROKEN: {e}"); bad+=1
sys.exit(bad)
PY
