#!/usr/bin/env python3
"""Fit the heterogeneous-set all-reduce term from the probe_het sweep: per-call excess of per-tensor over the best
equal-operand bundle (bucket6) at training scope, as a function of mean operand bytes and call count.
Model: excess_per_call_us = (bytes / BW) * n ** e  (A fixed at 0 so tiny operands pay ~nothing). usage: fit_het.py <train dirs...>"""
import json, glob, os, sys, math, itertools, statistics
sys.path.insert(0, "/tmp/g2"); 
K = 1024
import random
def mixed_sizes(n, choices, seed=11):
    rng = random.Random(seed); return [rng.choice(choices) for _ in range(n)]
CH = {"small": [64*K, 128*K, 256*K, 512*K], "mid": [K*K, 2*K*K, 4*K*K, 8*K*K], "big": [4*K*K, 8*K*K, 16*K*K, 22*K*K]}
rows = []
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "train", "newprob_train_probe_het_*.json")):
        j = json.load(open(f)); rows.append((j["problem"], j["cand"], j["measured_ms"]))
import collections; T = collections.defaultdict(dict)
for p, c, v in rows: T[p].setdefault(c, []).append(v)
pts = []
for p, cs in sorted(T.items()):
    _, _, cnt, lab = p.split("_"); cnt = int(cnt); sizes = mixed_sizes(cnt, CH[lab]); mean_bytes = 2 * sum(sizes) / cnt
    pt = statistics.mean(cs.get("per_tensor_ar", [float("nan")])); bu = statistics.mean(cs.get("bucket6", [float("nan")])); fl = statistics.mean(cs.get("flat_cat_ar", [float("nan")]))
    best = min(x for x in (bu, fl) if x == x) if any(x == x for x in (bu, fl)) else float("nan")
    excess = (pt - best) * 1000.0 / cnt   # us per call (fwd+bwd = 2 collectives per call)
    pts.append((cnt, mean_bytes, excess, pt, bu, fl)); print(f"{p:22s} n={cnt:3d} mean_bytes={mean_bytes/1e6:6.1f}MB  per_tensor={pt:7.1f} bucket6={bu:7.1f} flat={fl:7.1f}  excess/call={excess:8.1f} us")
# grid fit of BW (bytes/us) and count exponent e minimizing log-space error on positive excesses
best = None
for e in [i / 10 for i in range(0, 11)]:
    for bw in [200 * 1.15 ** i for i in range(60)]:
        err = 0.0; n = 0
        for cnt, mb, ex, *_ in pts:
            if ex > 0:
                pred = (mb / bw) * cnt ** e / 2.0   # per call per direction
                err += (math.log(pred) - math.log(ex / 2.0)) ** 2; n += 1
        if n and (best is None or err < best[0]): best = (err, bw, e)
print(f"\nfit: HET_BW_BPUS={best[1]:.0f} bytes/us ({best[1]/1000:.2f} GB/s), HET_COUNT_EXP={best[2]}, logrms={math.sqrt(best[0]/max(1,len(pts))):.2f}")
