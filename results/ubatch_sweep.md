# prefill t/s vs -ub (NPU HTP0, OPPOLL=1, patched binary, 2026-09-07). Default 512 is optimal.
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      128 |   1 | HTP0         |           pp512 |        830.82 ± 3.45 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      128 |   1 | HTP0         |          pp2048 |        807.65 ± 2.82 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      256 |   1 | HTP0         |           pp512 |        863.18 ± 1.19 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      256 |   1 | HTP0         |          pp2048 |        851.08 ± 1.69 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      512 |   1 | HTP0         |           pp512 |        885.69 ± 3.89 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |      512 |   1 | HTP0         |          pp2048 |        867.24 ± 0.60 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |     1024 |   1 | HTP0         |           pp512 |        877.66 ± 1.66 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |     1024 |   1 | HTP0         |          pp2048 |        868.53 ± 1.86 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |     2048 |   1 | HTP0         |           pp512 |        870.44 ± 0.24 |
| gemma4 E2B Q4_0                |   2.63 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |     2048 |   1 | HTP0         |          pp2048 |        836.55 ± 2.45 |
