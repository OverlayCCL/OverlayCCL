#!/usr/bin/env python3
"""Rank check of a sim_newprobs_v5 output against hardware training medians: for each problem, does the simulator put the
measured training winner first, and does it order per-tensor vs the best bundle the same way as hardware?
usage: rank_eval.py <sim json> <train result dirs...>"""
import json, sys, glob, os, statistics, collections
sim = json.load(open(sys.argv[1])); hw = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sys.argv[2:]:
    for f in glob.glob(os.path.join(d, "train", "newprob_train_*.json")) + glob.glob(os.path.join(d, "newprob_train_*.json")):
        j = json.load(open(f)); hw[j["problem"]][j["cand"]].append(j["measured_ms"])
probs = sorted({k.split("__")[0] for k in sim})
ok_first = 0; ok_pair = 0; n_first = 0; n_pair = 0
for p in probs:
    cands = {k.split("__")[1]: v for k, v in sim.items() if k.startswith(p + "__") and k.endswith("__train") and v is not None}
    meas = {c: statistics.mean(v) for c, v in hw.get(p, {}).items() if c in cands}
    if len(meas) < 2: print(f"{p:28s} (no hw pairs)"); continue
    hw_best = min(meas, key=meas.get); sim_best = min(cands, key=cands.get)
    n_first += 1; ok_first += (hw_best == sim_best)
    pt = next((c for c in meas if c.startswith("per_tensor")), None)
    msg = ""
    if pt:
        others = [c for c in meas if c != pt]; hb = min(others, key=meas.get)
        hw_pt_wins = meas[pt] <= meas[hb] * 1.02; sim_pt_wins = cands[pt] <= cands[hb] * 1.02
        n_pair += 1; ok_pair += (hw_pt_wins == sim_pt_wins)
        msg = f"per_tensor vs {hb}: hw {meas[pt]:.1f} vs {meas[hb]:.1f}  sim {cands[pt]/1000:.1f} vs {cands[hb]/1000:.1f} ms {'OK' if hw_pt_wins == sim_pt_wins else 'WRONG'}"
    print(f"{p:28s} hw_best={hw_best:18s} sim_best={sim_best:18s} {'OK' if hw_best == sim_best else 'MISS'}  {msg}")
print(f"\nwinner-first {ok_first}/{n_first}; per-tensor-vs-bundle direction {ok_pair}/{n_pair}")
