| Configuration | MMLU Acc | Init (ms) | Peak RAM (MB) | TTFT 128tok (ms) | Decode tok/s | Prefill tok/s |
|---|---|---|---|---|---|---|
| cpu_q4_0_baseline | 56.9% (n=1000) | 744 | 5123 | 724 | 37.3 | 202.3 |
| htp_fa_hvx |  | 1186 | 1861 | 379 | 25.0 | 366.6 |
| htp_kvq8 |  | 1186 | 1864 | 205 | 23.4 | 739.0 |
| htp_lmhead_cpu |  | 1243 | 2272 | 186 | 24.8 | 787.1 |
| htp_mm_tiled |  | 1298 | 2165 | 1734 | 24.0 | 76.4 |
| htp_mtp3 |  |  | 4 |  |  |  |
| htp_nhvx2 |  | 1211 | 1860 | 257 | 17.5 | 537.5 |
| htp_nhvx4 |  | 1158 | 1861 | 204 | 26.8 | 722.6 |
| htp_oppoll | 56.9% (n=1000) | 1193 | 1860 | 185 | 31.1 | 810.2 |
| htp_oppoll_fuse | 56.8% (n=1000) | 1289 | 1856 | 186 | 30.8 | 797.0 |
| htp_oppoll_nhvx4 |  | 1177 | 1863 | 204 | 31.2 | 734.8 |
| htp_oppoll_nofuse |  | 1285 | 1855 | 193 | 30.7 | 793.5 |
| htp_q4_0_default | 56.9% (n=1000) | 1318 | 1855 | 190 | 26.9 | 792.7 |
| htp_q4_0_default_prepatch |  | 1372 | 1861 | 187 | 27.1 | 785.1 |
| htp_q8_0_oppoll | 58.3% (n=1000) | 4565 | 2973 | 189 | 20.2 | 794.4 |
| htp_vmem512 |  | 1502 | 1862 | 266 | 11.9 | 541.0 |
| mac_iq4_nl_embq4_reference | 55.7% (n=1000) |  |  |  |  |  |
| mac_iq4_nl_reference | 55.6% (n=1000) |  |  |  |  |  |
| mac_q4_0_embiq4_reference | 51.1% (n=1000) |  |  |  |  |  |
| mac_q4_0_embq4_reference | 52.1% (n=1000) |  |  |  |  |  |
| mac_q4_0_reference | 56.8% (n=1000) |  |  |  |  |  |
| mac_q8_0_reference | 58.1% (n=1000) |  |  |  |  |  |
