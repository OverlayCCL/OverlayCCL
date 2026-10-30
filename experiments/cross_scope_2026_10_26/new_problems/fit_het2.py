#!/usr/bin/env python3
"""Fit the heterogeneous-set all-reduce term from every per-tensor-vs-bundle problem measured at training scope.
Point = (problem, n tensors, operand byte list, measured excess of per-tensor over the best bundle per call, in us).
Candidate models (excess per call, per direction): (mean_bytes / BW) * n ** e * F, with F = 1 or a spread factor
log(max/min bytes)/log(8) or a 2-D flag. usage: fit_het2.py <train result dirs...>"""
import json, glob, os, sys, math, statistics, collections
sys.path.insert(0, "/tmp/g2")
import torch; ns = {}; exec(open("/tmp/g2/g_newprobs.py").read(), ns); PROBLEMS = ns["PROBLEMS"]; flatten = ns["flatten"]
PT = {"per_tensor_ar", "per_tensor", "per_tensor_grouped", "per_tensor_grouped_ar"}
BUNDLES = {"bucket6", "flat_cat_ar", "per_layer_bucket", "bucket_4mb", "bucket_32mb", "equal_group_stacks", "cast_flat_bf16", "cast_flat_fp32", "two_dtype_groups",
           "per_layer_grouped", "flat_grouped", "per_layer", "flat", "flat_cat_2d", "bucket_1mb", "bucket_16mb", "bucket_64mb", "bucket_256mb", "stack_ar"}
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "train", "newprob_train_*.json")) + glob.glob(os.path.join(d, "newprob_train_*.json")):
        j = json.load(open(f)); rows[j["problem"]][j["cand"]].append(j["measured_ms"])
pts = []
for p, cs in sorted(rows.items()):
    if p not in PROBLEMS: continue
    pt_c = [c for c in cs if c in PT]; bu_c = [c for c in cs if c in BUNDLES]
    if not pt_c or not bu_c: continue
    tensors = flatten(PROBLEMS[p]["make"](64, 0, "cpu")[0]); nb = [t.numel() * t.element_size() for t in tensors]; n = len(nb)
    pt = statistics.mean(cs[pt_c[0]]); best_c = min(bu_c, key=lambda c: statistics.mean(cs[c])); best = statistics.mean(cs[best_c])
    excess_us = (pt - best) * 1000.0 / n / 2.0   # per call, per direction (fwd + bwd each issue the set)
    spread = math.log(max(nb) / max(1, min(nb))) / math.log(8); twod = 1.0 if any(t.dim() > 1 for t in tensors) else 0.0
    pts.append(dict(p=p, n=n, mean_b=sum(nb) / n, max_b=max(nb), spread=spread, twod=twod, ex=excess_us, pt=pt, best=best, best_c=best_c))
    print(f"{p:28s} n={n:3d} mean={sum(nb)/n/1e6:6.2f}MB max={max(nb)/1e6:6.1f}MB spread={spread:4.1f} 2d={int(twod)} per_tensor={pt:7.1f} best_bundle={best:7.1f} ({best_c:16s}) excess/call/dir={excess_us:8.1f} us")
def fit(feature):
    best = None
    for e in [i / 10 for i in range(0, 13)]:
        for bw in [100 * 1.12 ** i for i in range(90)]:
            err = 0.0; cnt = 0
            for q in pts:
                pred = (q["mean_b"] / bw) * q["n"] ** e * feature(q); meas = max(q["ex"], 20.0)   # clamp non-positive excess at 20 us
                err += (math.log(max(pred, 1.0)) - math.log(meas)) ** 2; cnt += 1
            if best is None or err < best[0]: best = (err, bw, e)
    return best
for name, feat in (("plain", lambda q: 1.0), ("spread", lambda q: max(q["spread"], 0.3)), ("2d x spread", lambda q: max(q["spread"], 0.3) * (1.0 if q["twod"] else 0.25))):
    err, bw, e = fit(feat)
    print(f"\nmodel {name:12s}: BW={bw:8.0f} B/us ({bw/1000:.1f} GB/s) e={e} logrms={math.sqrt(err/len(pts)):.2f}")
    for q in pts:
        pred = (q["mean_b"] / bw) * q["n"] ** e * feat(q); print(f"   {q['p']:28s} meas={q['ex']:8.1f} pred={pred:8.1f} -> per_tensor pred {q['best'] + 2*q['n']*pred/1000:7.1f} vs meas {q['pt']:7.1f}")
