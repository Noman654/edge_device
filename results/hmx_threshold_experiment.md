== correctness: test-backend-ops MUL_MAT q4_0/q8_0 with HMX_MIN_NROWS=1
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=3,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=4,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=5,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=2,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=3,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=4,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=5,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=8,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=8,k=4096,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=32,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;32
 MUL_MAT(type_a=q8_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): [1;3
0
== batched-bench HMX_MIN_NROWS=4
| 128 | 64 | 1 | 192 | 0.163 | 783.67 | 2.220 | 28.83 | 2.384 | 80.55 |
| 128 | 64 | 2 | 384 | 0.522 | 490.14 | 4.032 | 31.74 | 4.555 | 84.31 |
| 128 | 64 | 4 | 768 | 0.869 | 588.92 | 6.000 | 42.66 | 6.870 | 111.79 |
== batched-bench HMX_MIN_NROWS=1
| 128 | 64 | 1 | 192 | 0.166 | 770.30 | 2.231 | 28.69 | 2.397 | 80.11 |
| 128 | 64 | 2 | 384 | 0.506 | 505.80 | 4.814 | 26.59 | 5.321 | 72.17 |
| 128 | 64 | 4 | 768 | 0.892 | 573.96 | 5.433 | 47.12 | 6.325 | 121.42 |
== MTP n=1 with HMX_MIN_NROWS=1
1,GGML_HEXAGON_HMX_MIN_NROWS=1 mtp-gemma-4-E2B-it-Q4_0.gguf draft-mtp,1,128,26.59,87,39
1,GGML_HEXAGON_HMX_MIN_NROWS=1 mtp-gemma-4-E2B-it-Q4_0.gguf draft-mtp,2,128,28.47,81,45
HMX_CHAIN_DONE
