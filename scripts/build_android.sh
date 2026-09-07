#!/usr/bin/env bash
# Reproducible Android/HTP build: pin llama.cpp, apply our backend patch, build in the Snapdragon toolchain container.
# Usage: scripts/build_android.sh [--no-patch]      (needs Docker; ~25 min first time incl. image pull, ~3 min incremental)
# Output: llama.cpp/pkg-adb/llama.cpp/{bin,lib}. Only lib/libggml-hexagon.so differs between patched and unpatched.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=6a1a922d269908a29cbd4b49c27e6a8e7fd10fae
PATCH=patches/0001-ggml-hexagon-fuse-gelu-mul-into-geglu.patch
[ -d llama.cpp/.git ] || git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
[ "$(git rev-parse HEAD)" = "$COMMIT" ] || git checkout -q "$COMMIT"
if [ "${1:-}" = "--no-patch" ]; then
  git apply --check --reverse "../$PATCH" 2>/dev/null && git apply --reverse "../$PATCH" && echo "patch reverted"
else
  if git apply --check --reverse "../$PATCH" 2>/dev/null; then echo "patch already applied"; else git apply "../$PATCH" && echo "patch applied"; fi
fi
python3 scripts/snapdragon/build.py --target adb
ls -la pkg-adb/llama.cpp/lib/libggml-hexagon.so
