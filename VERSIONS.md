llama.cpp commit: 6a1a922d269908a29cbd4b49c27e6a8e7fd10fae
toolchain image: ghcr.io/snapdragon-toolchain/arm64-android:v0.7 (Hexagon SDK 6.6.0.0, NDK r28b)
frozen: Sat Sep  5 22:21:20 IST 2026
android package: llama.cpp/pkg-adb (arm64-android-snapdragon-release preset, GGML_HEXAGON=ON, GGML_OPENCL=ON, HTP kernels v73/v75/v79/v81)
mac reference build: llama.cpp/build-mac (Metal) — used only to validate the eval harness

models (ggml-org/gemma-4-E2B-it-GGUF, sha256 in models/SHA256SUMS):
  gemma-4-E2B-it-Q4_0.gguf  8e30dff3ac4c8434...  (device: all NPU/CPU/GPU rows)
  gemma-4-E2B-it-Q8_0.gguf  996d08777aadc6bf...  (device: quality ceiling row)
  gemma-4-E2B-it-BF16.gguf  246c5fb19e64b941...  (host only: source for make_quants.sh variants)
  mtp-gemma-4-E2B-it-Q4_0.gguf 718d3a44057924d5... (MTP drafter)
backend patch: patches/0001-ggml-hexagon-fuse-gelu-mul-into-geglu.patch on top of 6a1a922 (applies clean; 99 insertions)
patched package built: Mon Sep 7 00:07 IST 2026 (only lib/libggml-hexagon.so differs from the 2026-09-05 package)
HF repo revision pinned in get_models.sh: b4243c156154b6dca9324415f8c7ccc098b4aed1
artifacts/libggml-hexagon.so.prepatch  sha256 9f0069c0e6b4253a...  (exact device binary behind every pre-fusion number; pulled from device 2026-09-07)
artifacts/libggml-hexagon.so.patched   sha256 89ca21e8b73a3c0e...  (shipped binary; GELU+MUL fusion on by default + GGML_HEXAGON_HMX_MIN_NROWS knob, default 4 = unchanged behaviour)
