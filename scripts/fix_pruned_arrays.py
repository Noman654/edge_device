#!/usr/bin/env python3
"""Fix per-layer array hparams after llama-quantize --prune-layers on Gemma 4 (which leaves them full length).
Usage: fix_pruned_arrays.py <in.gguf> <out.gguf> <pruned_idx_csv e.g. 26,27>"""
import sys, os; sys.path.insert(0, "llama.cpp/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
inp, outp, pruned = sys.argv[1], sys.argv[2], set(int(x) for x in sys.argv[3].split(","))
r = GGUFReader(inp)
arch = r.fields["general.architecture"]; arch_str = str(bytes(arch.parts[arch.data[0]]), "utf-8")
per_layer_keys = {f"{arch_str}.feed_forward_length", f"{arch_str}.attention.sliding_window_pattern"}
w = GGUFWriter(outp, arch_str, use_temp_file=False)
for f in r.fields.values():
    name = f.name
    if name in ("general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"): continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        elem_t = f.types[1]
        vals = [f.parts[di].tolist() if hasattr(f.parts[di],'tolist') else f.parts[di] for di in f.data]
        vals = [v[0] if isinstance(v, list) and len(v)==1 else v for v in vals]
        if name in per_layer_keys:  # drop pruned indices
            vals = [v for i, v in enumerate(vals) if i not in pruned]
        if elem_t == GGUFValueType.STRING:
            vals = [str(bytes(v), "utf-8") if isinstance(v,(bytes,bytearray,np.ndarray)) else v for v in vals]
        elif elem_t in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
            vals = [float(v) for v in vals]
        else:
            vals = [int(v) for v in vals]
        w.add_array(name, vals)
    elif vt == GGUFValueType.STRING:
        w.add_string(name, str(bytes(f.parts[f.data[0]]), "utf-8"))
    else:
        import numpy as _np
        val = f.parts[f.data[0]]
        if hasattr(val, 'tolist'): val = val.tolist()
        if isinstance(val, list): val = val[0]
        if vt in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64): val = float(val)
        elif vt == GGUFValueType.BOOL: val = bool(val)
        elif vt != GGUFValueType.STRING: val = int(val)
        w.add_key_value(name, val, vt)
for t in r.tensors:
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {outp}: dropped layer idx {sorted(pruned)} from {per_layer_keys}")
