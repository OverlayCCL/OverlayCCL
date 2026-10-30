#!/usr/bin/env python3
"""Loop output under the paper's rule (Phase 5 ships the lowest-simulated-time candidate among those that pass the
correctness/TV gates), with the fitted v11 simulator. The training-scope hardware gate of 2026-10-28 is demoted to a
validation column. Inputs: the gate candidate sets (sim-ranked top-k of Phase 3 + builtins) with their 64-rank training
measurements (/tmp/g2/hw/gate_final_v12/*.json), the simulator scores recorded next to each gate candidate in
g_newprobs_gates_all.py, and the 224-rank repeats (sy7_train*)."""
import json, glob, re, statistics, collections, os
G = "/tmp/g2/hw"
sim = collections.defaultdict(dict)   # hw problem -> cand -> sim us (from the gates_all comments)
for line in open("/tmp/g2/g_newprobs_gates_all.py"):
    m = re.match(r'PROBLEMS\["([^"]+)"\]\["cands"\]\["(gate_\d)"\] = \S+\s+# sim (\d+) us, (.*)', line.strip())
    if m: sim[m.group(1)][m.group(2)] = (int(m.group(3)), m.group(4).strip())
t224 = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{G}/sy7_train*/train/*.json"):
    j = json.load(open(f)); t224[j["problem"]][j["cand"]].append(j["measured_ms"])
rows = ["| problem | paper-rule pick (lowest sim among gate survivors) | 64r: pick / per-tensor / best measured (ms) | 64r speedup | 224r pick vs per-tensor | training gate (validation) |", "|---|---|---|---|---|---|"]
summary = {}
for f in sorted(glob.glob(f"{G}/gate_final_v12/*.json")):
    if ".v11." in f: continue
    d = json.load(open(f)); p = d["hw_problem"]; meas = d["measured_ms"]; eff = d.get("effective", {})
    cands = sorted(sim[p].items(), key=lambda kv: kv[1][0]) if sim[p] else [(c, (t, n)) for c, n, t in d.get("gate_candidates", [])]
    pick = None
    for c, (t, n) in cands:
        if c in meas and eff.get(c, True) is not False: pick = (c, t, n); break      # correctness gate: a candidate that produced no result or an ineffective collective is removed
    base = next((c for c in meas if c.startswith("per_tensor")), None)
    best = min(meas, key=meas.get)
    if pick is None: rows.append(f"| {p} | (no finalist survives) | — / {meas[base]:.1f} / {meas[best]:.1f} ({best}) | — | — | {d['gate_choice']} |"); continue
    c, t, n = pick; sp64 = meas[base] / meas[c]
    a, b = t224[p].get(c), t224[p].get(base)
    s224 = f"{statistics.mean(b)/statistics.mean(a):.2f}x (n={min(len(a), len(b))})" if a and b else "—"
    rows.append(f"| {p} | {c}: {n[:45]} (sim {t} us) | {meas[c]:.1f} / {meas[base]:.1f} / {meas[best]:.1f} ({best}) | {sp64:.2f}x | {s224} | {d['gate_choice']} ({meas[d['gate_choice']]:.1f}) |")
    summary[p] = dict(pick=c, pick_name=n, sim_us=t, ms64=meas[c], base64=meas[base], best64=meas[best], best_name=best, speedup64=sp64, speedup224=s224, gate_choice=d["gate_choice"])
hdr = ("# Loop output under the paper's selection rule (simulator v11 + correctness/TV gates; training-scope gate = validation only)\n\n"
       "Rule: Phase 5 ships the candidate with the lowest simulated time among the candidates that pass the correctness gates "
       "(compiles, runs at training shape, output effective). The 'training gate (validation)' column is the fastest measured "
       "candidate of 2026-10-28, kept for reference; it is NOT the selector. 64r = 2-node training step (day-5 pair), 224r = 7-node.\n\n")
out = hdr + "\n".join(rows) + "\n"
os.makedirs(f"{G}/loop_simdecides", exist_ok=True); open(f"{G}/loop_simdecides/convergence_simdecides.md", "w").write(out)
json.dump(summary, open(f"{G}/loop_simdecides/summary.json", "w"), indent=1); print(out)
