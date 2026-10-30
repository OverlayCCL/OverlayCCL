#!/usr/bin/env python3
"""Collect sweep_2node/*.json (one per candidate, from g_sweep.sh) into {problem__cand: measured_ms} + a status list.
usage: collect_sweep.py <dir> <out.json>"""
import json, glob, sys, os
d, out = sys.argv[1], sys.argv[2]
res, status = {}, {}
for f in sorted(glob.glob(os.path.join(d, "*__*.json"))):
    if f.endswith("_percall.json"): continue
    j = json.load(open(f)); k = f"{j['problem']}__{j['cand']}"
    res[k] = j.get("measured_ms"); status[k] = j.get("error") or f"rc={j.get('rc')} src={j.get('measured_src')}"
json.dump(res, open(out, "w"), indent=1)
for k in sorted(res): print(f"{k:48s} {res[k]!s:>10s}   {status[k]}")
