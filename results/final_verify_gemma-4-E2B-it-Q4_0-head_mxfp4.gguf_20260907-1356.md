# Final verification Mon Sep  7 13:56:23 IST 2026   lib sha256: 89ca21e8b73a3c0e
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] thermal at start: {'max_c': 41.0, 'cpu_max_c': 40.9, 'npu_max_c': 39.3, 'n_zones': 88}
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] llama-bench ...
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] completion + rss ...
  "init_ms": 1288.97,
  "ttft_ms": 180.32,
  "decode_tps": 34.358188,
  "prefill_tps": 797.595119,
  "peak_pss_mb": 1641,
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] thermal at start: {'max_c': 49.0, 'cpu_max_c': 49.0, 'npu_max_c': 44.8, 'n_zones': 88}
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] llama-bench ...
[htp_final_gemma-4-E2B-it-Q4_0-head_mxfp4.gguf] completion + rss ...
  "init_ms": 1308.57,
  "ttft_ms": 183.77,
  "decode_tps": 34.9622,
  "prefill_tps": 831.999658,
  "peak_pss_mb": 1641,
## fusion
fused_gelu_mul=136 gelu_ops_added=140 rms_norm_mul_fused=908
## KL vs CPU logits
eval/ppl_text.txt: 1 file pushed, 0 skipped. 20.2 MB/s (30000 bytes in 0.001s)
eval/ppl_text.txt: 1 file pushed, 0 skipped. 12.2 MB/s (30000 bytes in 0.002s)
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
Mean    KLD:   0.003376 ±   0.000891
Same top p: 97.647 ± 0.388 %
## MTP depth 1
1,gemma-4-E2B-it-Q4_0-head_mxfp4.gguf  mtp-gemma-4-E2B-it-Q4_0.gguf draft-mtp,1,128,36.98,89,38
1,gemma-4-E2B-it-Q4_0-head_mxfp4.gguf  mtp-gemma-4-E2B-it-Q4_0.gguf draft-mtp,2,128,40.08,81,45
## init (wall ms, --fit off)
1889 ms
1852 ms
1856 ms
## MMLU skipped (SKIP_MMLU set)
