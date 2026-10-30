#!/usr/bin/env python3
"""Simulator-fidelity (tab:simfidelity) and constant-swap (tab:constswap) rows for the new problems: simulator = ORIGINAL constants
(arms from run_fid_arms.sh), hardware = mirror-harness 224-rank training medians (sy7_v10*, sy7_v10_nosim, sy7_v11ag*)."""
import json, glob, collections, statistics, os
G = "/tmp/g2/hw"
hw = collections.defaultdict(lambda: collections.defaultdict(list))
for d in ("sy7_v10", "sy7_v10_r2", "sy7_v10_r3", "sy7_v10_r4", "sy7_v10_r5", "sy7_v10_nosim", "sy7_v11ag", "sy7_v11ag_r2", "sy7_v11ag_r3", "sy7_v11ag_r4", "sy7_v11ag_r5"):
    for f in glob.glob(f"{G}/{d}/train/*.json"):
        j = json.load(open(f)); hw[j["problem"]][j["cand"]].append(j["measured_ms"])
HW = {p: {c: statistics.median(v) for c, v in cs.items()} for p, cs in hw.items()}
def load(tag):
    f = f"{G}/sim_newprobs_v5_fid_{tag}.json"
    if not os.path.exists(f): return {}
    d = json.load(open(f)); out = collections.defaultdict(dict)
    for k, v in d.items():
        p, c, scope = k.split("__"); out[p][c] = v
    return out
def spearman(xs, ys):
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i]); r = [0] * len(a)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    if n < 3: return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")
calib = load("calib"); calib.update(load("ag_calib"))
base_of = lambda p: "ag_pinned" if p == "fsdp_ag_layout" else "per_tensor_ar"
rows = ["| Problem | n | rho | top-1 | regret | ratio-MAPE |", "|---|---|---|---|---|---|"]; picks = {}
for p in sorted(calib):
    cands = [c for c in calib[p] if c in HW.get(p, {}) and isinstance(calib[p][c], (int, float))]
    if len(cands) < 2: continue
    sim = [calib[p][c] for c in cands]; meas = [HW[p][c] for c in cands]
    rho = spearman(sim, meas); best_m = min(cands, key=lambda c: HW[p][c]); pick = min(cands, key=lambda c: calib[p][c]); picks[p] = pick
    regret = HW[p][pick] / HW[p][best_m] - 1
    b = base_of(p)
    if b in cands:
        mape = statistics.mean(abs((calib[p][b] / calib[p][c]) / (HW[p][b] / HW[p][c]) - 1) for c in cands if c != b) if len(cands) > 1 else float("nan")
    else: mape = float("nan")
    rows.append(f"| {p} | {len(cands)} | {rho:.2f} | {'yes' if pick == best_m else 'no (' + pick + ')'} | {100*regret:.1f}% | {100*mape:.0f}% |")
arms = [("calib", "calibrated (original constants)"), ("net_half", "network x1/2"), ("net_quarter", "network x1/4"), ("net_x2", "network x2"), ("net_x4", "network x4"), ("local_snapshot", "local terms = snapshot"), ("opc_isolated", "op costs = isolated"), ("reset_graph", "run reset per graph"), ("knee_64k", "amort. knee 64 KB"), ("knee_1m", "amort. knee 1 MB"), ("fwdbwd_0", "fwd+bwd factor 0"), ("fwdbwd_2", "fwd+bwd factor 2")]
probs = [p for p in sorted(calib) if p != "fsdp_ag_layout" and len([c for c in calib[p] if c in HW.get(p, {})]) >= 2]
cs = ["| Arm | " + " | ".join(probs) + " | agree. |", "|---|" + "---|" * (len(probs) + 1)]
for tag, lab in arms:
    S = load(tag); cells = []; agree = 0; tot = 0
    for p in probs:
        cands = [c for c in S.get(p, {}) if c in HW.get(p, {}) and isinstance(S[p][c], (int, float))]
        if len(cands) < 2: cells.append("—"); continue
        rho = spearman([S[p][c] for c in cands], [HW[p][c] for c in cands]); pick = min(cands, key=lambda c: S[p][c])
        tot += 1; agree += (pick == picks.get(p)); cells.append(f"{rho:.2f}" + ("" if pick == picks.get(p) else "*"))
    cs.append(f"| {lab} | " + " | ".join(cells) + f" | {agree}/{tot} |")
out = "# Simulator fidelity (original constants) vs mirror-harness 224-rank training, new problems\n\n" + "\n".join(rows) + "\n\n# Constant-swap arms (* = the swap changes the simulator's pick)\n\n" + "\n".join(cs) + "\n"
open(f"{G}/fidelity_new.md", "w").write(out); print(out)
