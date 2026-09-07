# Execution levers, round 2 (NPU, OPPOLL=1, patched binary, 2026-09-07). llama-bench pp128/tg64, r=3.
== ndev2 (GGML_HEXAGON_NDEV=2 -dev HTP0,HTP1)
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | pp128 | 775.83 ± 6.23 |
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | tg64 | 28.96 ± 2.11 |
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP1 | pp128 | 776.54 ± 2.43 |
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP1 | tg64 | 30.63 ± 0.09 |
== hostbuf0 (GGML_HEXAGON_HOSTBUF=0 -dev HTP0)
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | pp128 | 778.56 ± 1.71 |
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | tg64 | 30.58 ± 0.20 |
== hostbuf1 (GGML_HEXAGON_HOSTBUF=1 -dev HTP0)
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | pp128 | 776.52 ± 3.56 |
| gemma4 E2B Q4_0 | 2.63 GiB | 4.63 B | OpenCL,HTP | 99 | 6 | 1 | HTP0 | tg64 | 30.22 ± 0.14 |
== nhvx8 (GGML_HEXAGON_NHVX=8 -dev HTP0)
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
Terminated 
EXEC2_DONE

== NDEV=2 -dev HTP0/HTP1 -sm row
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
llama_bench: error: failed to load model '/data/local/tmp/gguf/gemma-4-E2B-it-Q4_0.gguf'
== NDEV=2 -dev HTP0/HTP1 -sm layer
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
Terminated 
== NDEV=2 -dev HTP0/HTP1 -sm row
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
llama_bench: error: failed to load model '/data/local/tmp/gguf/gemma-4-E2B-it-Q4_0.gguf'
== NDEV=2 -dev HTP0/HTP1 -sm layer
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
Terminated 

[exited with code 0]
