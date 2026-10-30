#!/usr/bin/env python3
"""Cross-scope table: per problem, candidate ratio vs the production baseline at each scope (sim or measured).
Usage: analyze_scopes.py <label=json> [<label=json> ...]   (json: {"<problem>__<cand>": value_or_null})
A value is a time; ratio = baseline / cand (>1 = candidate faster). Flags INVERSION when the sign of
(cand - baseline) differs between the first (bench) column and any later (training) column."""
import json, sys, collections
cols = []
for a in sys.argv[1:]:
    lab, path = a.split("=", 1)
    try: d = json.load(open(path))
    except Exception: d = {}
    cols.append((lab, d))
keys = sorted({k for _, d in cols for k in d})
byp = collections.defaultdict(list)
for k in keys:
    p, c = k.split("__", 1); byp[p].append(c)
def ratio(d, p, c):
    b = d.get(f"{p}__harness_baseline"); v = d.get(f"{p}__{c}")
    if b is None or v is None or v == 0: return None
    return b / v
hdr = f"{'problem':15s} {'candidate':26s} " + " ".join(f"{lab:>12s}" for lab, _ in cols) + "   flag"
print(hdr); print("-" * len(hdr))
for p in sorted(byp):
    for c in sorted(byp[p]):
        if c == "harness_baseline": continue
        rs = [ratio(d, p, c) for _, d in cols]
        flag = ""
        if rs[0] is not None:
            for r in rs[1:]:
                if r is not None and ((rs[0] > 1.0) != (r > 1.0)) and abs(rs[0] - 1) > 0.03 and abs(r - 1) > 0.03:
                    flag = "INVERSION"
        print(f"{p:15s} {c:26s} " + " ".join(f"{(f'{r:.2f}x' if r is not None else 'fail'):>12s}" for r in rs) + f"   {flag}")
    print()
