#!/usr/bin/env python3
"""Bench (32 ranks, 1 node) vs training-scope (64 ranks, 2 nodes) table for the new problems.
usage: newprobs_table.py <dir with newprob_bench_*.json and train/newprob_train_*.json>"""
import json, glob, os, sys, collections
import statistics
dirs = sys.argv[1:]
_b = collections.defaultdict(lambda: collections.defaultdict(list)); _t = collections.defaultdict(lambda: collections.defaultdict(list))
_ok = collections.defaultdict(dict); _b64 = collections.defaultdict(lambda: collections.defaultdict(list))
for d in dirs:
    for f in glob.glob(os.path.join(d, "newprob_bench_*.json")):
        j = json.load(open(f)); tgt = _b64 if f.endswith("_ws64.json") else _b
        for c, r in j["results"].items():
            if r.get("med_ms") is not None: tgt[j["problem"]][c].append(r["med_ms"])
            if tgt is _b: _ok[j["problem"]][c] = r.get("ok")
    for f in glob.glob(os.path.join(d, "train", "newprob_train_*.json")):
        j = json.load(open(f))
        if j.get("measured_ms") is not None: _t[j["problem"]][j["cand"]].append(j["measured_ms"])
bench = {p: {c: {"med_ms": statistics.mean(v), "n": len(v), "ok": _ok[p].get(c)} for c, v in cs.items()} for p, cs in _b.items()}
train = {p: {c: statistics.mean(v) for c, v in cs.items()} for p, cs in _t.items()}
bench64 = {p: {c: statistics.mean(v) for c, v in cs.items()} for p, cs in _b64.items()}
ntrain = {p: {c: len(v) for c, v in cs.items()} for p, cs in _t.items()}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from g_newprobs import PROBLEMS
except Exception:
    PROBLEMS = {}
print(f"{'problem':22s} {'candidate':24s} {'bench32':>8s} {'b32 x':>6s} {'bench64':>8s} {'b64 x':>6s} {'train64':>8s} {'t64 x':>6s}  note")
for p in sorted(set(bench) | set(train)):
    base = PROBLEMS.get(p, {}).get("baseline")
    b = bench.get(p, {}); t = train.get(p, {})
    bb = b.get(base, {}).get("med_ms") if base else None; tb = t.get(base) if base else None
    cands = sorted(set(b) | set(t), key=lambda c: (c != base, c))
    for c in cands:
        bm = b.get(c, {}).get("med_ms"); tm = t.get(c)
        rb = bb / bm if (bb and bm) else None; rt = tb / tm if (tb and tm) else None
        b64 = bench64.get(p, {}); b64b = b64.get(base) if base else None; bm64 = b64.get(c); rb64 = b64b / bm64 if (b64b and bm64) else None
        rbench = rb64 if rb64 is not None else rb
        ok = b.get(c, {}).get("ok"); note = "" if ok in (True, None) else "CHECK-FAIL"
        for lab, r_ in (("b32", rb), ("b64", rb64)):
            if r_ is not None and rt is not None and abs(r_ - 1) > 0.10 and abs(rt - 1) > 0.10 and (r_ > 1) != (rt > 1): note += f" INVERSION({lab})"
        if "INVERSION" not in note and rbench is not None and rt is not None and abs(rbench - 1) <= 0.05 and abs(rt - 1) > 0.10: note += " bench-tie, training " + ("win" if rt > 1 else "loss")
        f = lambda v, w: (f"{v:{w}.2f}" if v is not None else f"{'—':>{w}s}")
        print(f"{p:22s} {c:24s} {f(bm,8)} {f(rb,6)} {f(bm64,8)} {f(rb64,6)} {f(tm,8)} {f(rt,6)}  n={b.get(c, {}).get('n', 0)}/{ntrain.get(p, {}).get(c, 0)} {note.strip()}")
    print()
