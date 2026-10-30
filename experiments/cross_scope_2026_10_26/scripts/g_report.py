#!/usr/bin/env python3
"""(g) cross-scope report: per problem/candidate, 1-node bench (32 ranks) and 64-rank training-scope step time,
each as absolute ms and ratio vs the production baseline (baseline/candidate, >1 = candidate faster), with the
simulator's bench/training predictions; repeats are averaged (mean, n). usage: g_report.py <hw dir> <sim dir>"""
import json, glob, os, sys, statistics, collections
hw, simd = sys.argv[1], sys.argv[2]
def load_dir(pattern):
    vals = collections.defaultdict(list)
    for f in glob.glob(pattern):
        if f.endswith("_percall.json"): continue
        j = json.load(open(f)); v = j.get("measured_ms", j.get("med_ms"))
        if v is not None: vals[f"{j['problem']}__{j['cand']}"].append(float(v))
    return vals
bench = load_dir(os.path.join(hw, "bench_1node", "*__*.json"))
if "pp_send_recv__harness_baseline" not in bench and "pp_send_recv__ar_masked_per_mb" in bench:
    bench["pp_send_recv__harness_baseline"] = bench["pp_send_recv__ar_masked_per_mb"]   # sandbox per_mb builtin allocates on CPU; same masked-AR loop
train = collections.defaultdict(list)
for d in sorted(glob.glob(os.path.join(hw, "sweep_2node*"))):
    for k, v in load_dir(os.path.join(d, "*__*.json")).items(): train[k] += v
sim = {}
for tag in ("bench_32", "train_64", "train_224"):
    p = os.path.join(simd, f"sim_{tag}.json"); sim[tag] = json.load(open(p)) if os.path.exists(p) else {}
def ratio(vals, p, c):
    b = vals.get(f"{p}__harness_baseline"); v = vals.get(f"{p}__{c}")
    if not b or not v: return None
    return statistics.mean(b) / statistics.mean(v)
def sratio(tag, p, c):
    b = sim[tag].get(f"{p}__harness_baseline"); v = sim[tag].get(f"{p}__{c}")
    return (b / v) if (b and v) else None
keys = sorted(set(bench) | set(train)); byp = collections.defaultdict(set)
for k in keys: byp[k.split("__")[0]].add(k.split("__", 1)[1])
f = lambda r: f"{r:.2f}x" if r is not None else "—"
ms = lambda vals, k: (f"{statistics.mean(vals[k]):.1f}" + (f"±{statistics.stdev(vals[k]):.1f}" if len(vals[k]) > 1 else "")) if vals.get(k) else "fail"
print("| problem | candidate | bench 32r ms | bench ratio | train 64r ms (n) | train ratio | sim bench | sim train64 | sim train224 | note |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(byp):
    for c in sorted(byp[p]):
        if c == "harness_baseline": continue
        k = f"{p}__{c}"; rb, rt = ratio(bench, p, c), ratio(train, p, c)
        note = ""
        if rb is not None and rt is not None and abs(rb - 1) > 0.03 and abs(rt - 1) > 0.03 and (rb > 1) != (rt > 1): note = "INVERSION"
        elif rt is not None and rb is not None and rt > 1.05 and rb <= 1.03: note = "bench-tie, training win"
        elif rt is not None and rb is not None and rb > 1.05 and rt <= 1.03: note = "bench win, training tie"
        n = len(train.get(k, []))
        print(f"| {p} | {c} | {ms(bench, k)} | {f(rb)} | {ms(train, k)} ({n}) | {f(rt)} | {f(sratio('bench_32', p, c))} | {f(sratio('train_64', p, c))} | {f(sratio('train_224', p, c))} | {note} |")
    kb = f"{p}__harness_baseline"
    print(f"| {p} | harness_baseline | {ms(bench, kb)} | 1.00x | {ms(train, kb)} ({len(train.get(kb, []))}) | 1.00x | 1.00x | 1.00x | 1.00x | |")
