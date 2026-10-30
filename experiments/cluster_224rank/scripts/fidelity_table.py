#!/usr/bin/env python3
"""Simulator-fidelity table for the paper: per problem, n candidates, Spearman(sim, measured step), top-1 hit, regret,
ratio-MAPE (speedup-vs-baseline ratios), and median relative step-time error after a per-problem affine rescaling.
With --arms, also the per-problem Spearman of every simulator arm (constant swaps) against the measured pool and the
rank agreement of each arm with the reference arm. usage: fidelity_table.py <measured root> <sim json> [--arms dir glob]"""
import glob, json, os, statistics, sys, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
        for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(x, y):
    n = len(x)
    if n < 3: return float("nan")
    rx, ry = rank(x), rank(y); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else float("nan")
root, simf = sys.argv[1], sys.argv[2]; arms = sys.argv[sys.argv.index("--arms") + 1] if "--arms" in sys.argv else None
runs = defaultdict(list)
extra = [d for d in os.environ.get("EXTRA_DIRS", "").split(",") if d and os.path.isdir(d)]
for rep in sorted(glob.glob(os.path.join(root, "rep*"))) + extra:
    for f in glob.glob(os.path.join(rep, "*__*.json")):
        if f.endswith("_percall.json"): continue
        j = json.load(open(f)); v = j.get("measured_ms")
        if v is not None: runs[f"{j['problem']}__{j['cand']}"].append(float(v))
sim = json.load(open(simf)); byp = defaultdict(list)
for k in runs: byp[k.split("__")[0]].append(k)
NAMES = {"alltoallv": "AllToAllV", "dxe": "Distributed CE (dxe)", "fsdp_prefetch": "FSDP prefetch", "llama_block_ar": "Layer-block AR", "pp_send_recv": "PP send/recv", "ring_kv": "Ring KV", "tp_mlp": "TP MLP", "uniform_a2a": "Uniform A2A"}
print("| Problem | n | Spearman ρ | top-1 | regret | ratio-MAPE | rescaled step error (median) |\n|---|---|---|---|---|---|---|")
rows = {}
for p, ks in sorted(byp.items()):
    ks = [k for k in ks if sim.get(k) is not None]
    if len(ks) < 3: continue
    m = {k: statistics.mean(runs[k]) for k in ks}; s_ = {k: sim[k] for k in ks}
    rho = spearman([s_[k] for k in ks], [m[k] for k in ks]); sp = min(ks, key=lambda k: s_[k]); mb = min(ks, key=lambda k: m[k]); reg = (m[sp] - m[mb]) / m[mb] * 100
    base = f"{p}__harness_baseline"; mape = float("nan")
    if base in m:
        errs = [abs((s_[base] / s_[k]) / (m[base] / m[k]) - 1) for k in ks if k != base]; mape = statistics.mean(errs) * 100 if errs else float("nan")
    # affine rescale sim -> measured (least squares), median relative error
    xs = [s_[k] for k in ks]; ys = [m[k] for k in ks]; mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0; a = my - b * mx
    rel = statistics.median(abs((a + b * x) - y) / y for x, y in zip(xs, ys)) * 100
    rows[p] = dict(n=len(ks), rho=rho, top1=(sp == mb), regret=reg, mape=mape, resc=rel)
    print(f"| {NAMES.get(p, p)} | {len(ks)} | {rho:.2f} | {'yes' if sp == mb else f'no ({reg:.1f}% regret)'} | {reg:.1f}% | {mape:.0f}% | {rel:.1f}% |")
if arms:
    print("\n| arm | " + " | ".join(NAMES.get(p, p) for p in sorted(rows)) + " | pooled agreement with reference (ρ of candidate ranks) |\n|---|" + "---|" * (len(rows) + 1))
    ref = sim
    for af in sorted(glob.glob(arms)):
        a = json.load(open(af)); tag = os.path.basename(af).replace("sim_v4_", "").replace(".json", ""); cells = []; agree = []
        for p in sorted(rows):
            ks = [k for k in byp[p] if a.get(k) is not None and ref.get(k) is not None]
            if len(ks) < 3: cells.append("–"); continue
            cells.append(f"{spearman([a[k] for k in ks], [statistics.mean(runs[k]) for k in ks]):.2f}"); agree.append(spearman([a[k] for k in ks], [ref[k] for k in ks]))
        print(f"| {tag} | " + " | ".join(cells) + f" | {statistics.mean(agree):.2f} |")
