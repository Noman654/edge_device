| model                          |       size |     params | backend    | ngl | threads |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | ------------ | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |   1 | HTP0         |     tg64 @ d512 |         34.26 ± 0.10 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |   1 | HTP0         |    tg64 @ d2048 |         32.09 ± 0.07 |
| gemma4 E2B Q4_0                |   2.43 GiB |     4.63 B | OpenCL,HTP |  99 |       6 |   1 | HTP0         |    tg64 @ d8192 |         30.52 ± 0.35 |

build: 6a1a922 (1)
