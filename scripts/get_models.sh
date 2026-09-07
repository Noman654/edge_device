#!/usr/bin/env bash
# Download the exact model files used in the report from ggml-org/gemma-4-E2B-it-GGUF (pinned revision) and verify sha256.
# Usage: scripts/get_models.sh [q4|q8|bf16|mtp|all ...]   (default: q4 mtp = what the device runs). bash 3.2 compatible.
# BF16 (9.3 GB) is only needed to re-derive the quant variants with scripts/make_quants.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=ggml-org/gemma-4-E2B-it-GGUF
REV=b4243c156154b6dca9324415f8c7ccc098b4aed1   # HF repo commit at the time of the study (2026-09-07)
name() { case "$1" in q4) echo gemma-4-E2B-it-Q4_0.gguf;; q8) echo gemma-4-E2B-it-Q8_0.gguf;; bf16) echo gemma-4-E2B-it-BF16.gguf;; mtp) echo mtp-gemma-4-E2B-it-Q4_0.gguf;; *) echo "unknown: $1" >&2; exit 2;; esac; }
sel="$*"; [ -z "$sel" ] && sel="q4 mtp"; [ "$sel" = all ] && sel="q4 q8 bf16 mtp"
for k in $sel; do
  f=$(name "$k"); [ -s "models/$f" ] && { echo "have $f"; continue; }
  hf download "$REPO" "$f" --revision "$REV" --local-dir models
done
cd models && for k in $sel; do grep "$(name "$k")" SHA256SUMS; done | shasum -a 256 -c -
