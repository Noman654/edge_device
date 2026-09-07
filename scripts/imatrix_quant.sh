#!/usr/bin/env bash
# imatrix-guided Q4_0 (host / Metal): same layout as the stock ggml-org Q4_0 (Q4_0 body incl. per-layer table, Q8_0
# token_embd/LM head), but with importance-weighted block scales from wikitext-2 train activations. Speed-identical on
# device (same tensor types/bytes); the only question is accuracy, answered by MMLU on the frozen set.
# Usage: scripts/imatrix_quant.sh [chunks=100]   ->  models/gemma-4-E2B-it-Q4_0-imatrix.gguf
set -euo pipefail
cd "$(dirname "$0")/.."
B=llama.cpp/build-mac/bin; SRC=models/gemma-4-E2B-it-BF16.gguf; CAL=eval/calib/wikitext-2-raw/wiki.train.raw
IM=models/imatrix-gemma4-e2b-wikitext2-c512-${1:-100}.gguf; OUT=models/gemma-4-E2B-it-Q4_0-imatrix.gguf
[ -s "$CAL" ] || (cd eval/calib && bash ../../llama.cpp/scripts/get-wikitext-2.sh)
[ -s "$IM" ] || $B/llama-imatrix -m "$SRC" -f "$CAL" -o "$IM" -c 512 --chunks "${1:-100}" -ngl 99 -t 8 2>&1 | grep -E "save|chunk|error|compute_imatrix" | tail -5
[ -s "$OUT" ] || $B/llama-quantize --imatrix "$IM" --token-embedding-type q8_0 "$SRC" "$OUT" Q4_0 2>&1 | tail -3
.venv/bin/python - "$OUT" <<'PY'
import sys; sys.path.insert(0,"llama.cpp/gguf-py")
from gguf import GGUFReader
from collections import Counter
for f in ["models/gemma-4-E2B-it-Q4_0.gguf", sys.argv[1]]:
    r=GGUFReader(f); c=Counter(t.tensor_type.name for t in r.tensors); tot=sum(int(t.n_bytes) for t in r.tensors)
    print(f"{f}: {dict(c)} total={tot/1e9:.3f} GB")
PY
