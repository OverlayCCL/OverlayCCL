#!/usr/bin/env python3
"""Fit the unequal-concat per-piece term from the catk sweep (training scope): excess of flat_cat_ar over bucket8 per
direction as a function of the piece count k. Model: excess_us = UNEQ_PIECE_US * max(0, k - knee). usage: fit_catk.py <dirs...>"""
import json, glob, os, sys, statistics, collections, math
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "train", "newprob_train_catk_*.json")) + glob.glob(os.path.join(d, "train", "newprob_train_hs_*.json")) + glob.glob(os.path.join(d, "train", "newprob_train_probe_het_*.json")):
        j = json.load(open(f)); rows[j["problem"]][j["cand"]].append(j["measured_ms"])
pts = []
for p, cs in sorted(rows.items()):
    fl = cs.get("flat_cat_ar"); bu = cs.get("bucket8") or cs.get("bucket_4mb") or cs.get("bucket6")
    if not fl or not bu: continue
    if p.startswith("catk_"): k = int(p.split("_")[1])
    elif p.startswith("hs_a"): k = 64
    elif p.startswith("hs_c"): k = 32
    elif p.startswith("hs_b"): k = 128
    elif p.startswith("probe_het_"): k = int(p.split("_")[2])
    else: continue
    ex = (statistics.mean(fl) - statistics.mean(bu)) * 1000.0 / 2.0   # us per direction per step
    pts.append((p, k, ex, statistics.mean(fl), statistics.mean(bu))); print(f"{p:22s} k={k:4d} flat={statistics.mean(fl):7.1f} bucket={statistics.mean(bu):7.1f} excess/dir={ex:9.1f} us")
best = None
for knee in range(0, 72, 4):
    for per in [10 * 1.15 ** i for i in range(50)]:
        err = sum((math.log(max(per * max(0, k - knee), 1.0)) - math.log(max(ex, 20.0))) ** 2 for _, k, ex, *_ in pts)
        if best is None or err < best[0]: best = (err, knee, per)
print(f"\nfit: UNEQ_KNEE={best[1]} UNEQ_PIECE_US={best[2]:.0f} logrms={math.sqrt(best[0]/max(1,len(pts))):.2f}")
for p, k, ex, fl, bu in pts: print(f"   {p:22s} meas={ex:8.0f} pred={best[2]*max(0,k-best[1]):8.0f}")
