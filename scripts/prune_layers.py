#!/usr/bin/env python3
"""Drop the LAST n transformer layers of a Gemma-4 GGUF (they are all KV-shared layers when n <= shared_kv_layers),
keeping the file loadable: truncates block_count, shared_kv_layers, the per-layer arrays (feed_forward_length,
sliding_window_pattern), the per-layer embedding table (per_layer_token_embd, [256*n_layer, vocab], Q4_0 blocks)
and its projection (per_layer_model_proj, [n_embd, 256*n_layer]). Tensor bytes are copied verbatim (no requant).
Usage: prune_layers.py in.gguf out.gguf n_drop"""
import sys; sys.path.insert(0, "llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
src, dst, n_drop = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = GGUFReader(src); kv = {f.name: f for f in r.fields.values()}
arch = kv["general.architecture"].parts[-1].tobytes().decode()
n_layer = int(kv[f"{arch}.block_count"].parts[-1][0]); n_shared = int(kv[f"{arch}.attention.shared_kv_layers"].parts[-1][0])
n_ple = int(kv[f"{arch}.embedding_length_per_layer_input"].parts[-1][0])
assert n_drop <= n_shared, "only KV-shared (last) layers can be dropped without touching the KV layout"
n_keep = n_layer - n_drop
w = GGUFWriter(dst, arch)
for name, f in kv.items():
    if name.startswith("GGUF.") or name == "general.architecture": continue
    val = f.contents()
    if name == f"{arch}.block_count": val = n_keep
    elif name == f"{arch}.attention.shared_kv_layers": val = n_shared - n_drop
    elif isinstance(val, list) and len(val) == n_layer: val = val[:n_keep]
    if isinstance(val, list): w.add_array(name, val)
    else: w.add_key_value(name, val, f.types[0])
kept = 0
for t in r.tensors:
    n = t.name
    if n.startswith("blk."):
        if int(n.split(".")[1]) >= n_keep: continue
    raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8); shape = [int(x) for x in t.shape]  # shape = [ne0, ne1, ...]
    if n == "per_layer_token_embd.weight":            # [n_ple*n_layer, vocab], Q4_0: 18 bytes per 32 elems along ne0
        assert t.tensor_type == T.Q4_0 and shape[0] == n_ple * n_layer
        row_b = shape[0] // 32 * 18; keep_b = n_ple * n_keep // 32 * 18
        raw = raw.reshape(shape[1], row_b)[:, :keep_b].copy().reshape(-1); shape[0] = n_ple * n_keep
    elif n == "per_layer_model_proj.weight":          # [n_embd, n_ple*n_layer] BF16: keep first n_ple*n_keep rows of ne1
        assert shape[1] == n_ple * n_layer
        row_b = shape[0] * 2; raw = raw.reshape(shape[1], row_b)[: n_ple * n_keep].copy().reshape(-1); shape[1] = n_ple * n_keep
    # GGUFWriter derives the element shape from a uint8 array's byte shape: pass [rows, bytes_per_row] (1-D stays 1-D)
    assert len(shape) <= 2, (n, shape)
    arr = raw if len(shape) == 1 else raw.reshape(shape[1], len(raw) // shape[1])
    w.add_tensor(n, arr, raw_dtype=t.tensor_type); kept += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{dst}: kept {kept} tensors, layers {n_layer}->{n_keep}, shared_kv {n_shared}->{n_shared-n_drop}")
