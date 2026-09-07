#!/usr/bin/env python3
"""Paired McNemar test between two MMLU result CSVs scored on the same frozen subset.
Usage: scripts/paired_test.py results/mmlu_A.csv results/mmlu_B.csv"""
import csv, sys, math
def load(p): return {int(r["id"]): int(r["correct"]) for r in csv.DictReader(open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2]); ids = sorted(set(a) & set(b))
n01 = sum(1 for i in ids if a[i] == 0 and b[i] == 1); n10 = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
acc_a = sum(a[i] for i in ids) / len(ids); acc_b = sum(b[i] for i in ids) / len(ids)
# exact binomial McNemar (two-sided)
k, n = min(n01, n10), n01 + n10
p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n) if n else 1.0
print(f"n={len(ids)}  A={acc_a:.3f}  B={acc_b:.3f}  delta={acc_b-acc_a:+.3f}  discordant: A-only-correct={n10} B-only-correct={n01}  McNemar exact p={p:.4f}")
