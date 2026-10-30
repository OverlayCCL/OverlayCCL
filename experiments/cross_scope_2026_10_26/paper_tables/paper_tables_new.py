#!/usr/bin/env python3
"""Regenerate, for every table of the paper, the rows the new heterogeneous-gradient problems add (2-node data; 7-node
columns are marked pending). Reads the gate outputs, the 32/64-rank benches, the simulator scores and the Phase-3 logs;
writes paper_tables_new.md next to this file. Nothing here touches the paper sources."""
import json, glob, os, re, statistics, math, collections
G = "/tmp/g2/hw"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tables_new.md")
SHORT = {"layer_grads_hetero_ar": "Layer grads (6 hetero)", "model_grads_4layers": "Model grads (24 hetero)", "layer_grads_mixed_dtype": "Mixed-dtype grads",
         "hs_a_64x64k_512k": "Bucket count A (64 x 64K-512K)", "hs_c_32x256k_2m": "Bucket count C (32 x 256K-2M)", "probe_het_24_mid": "Probe 24 x 1-8M",
         "probe_het_48_mid": "Probe 48 x 1-8M", "probe_het_48_small": "Probe 48 x 64-512K", "hetero_grad_ar_many": "Hetero-many (48 2-D)", "zero2_hetero_rs": "ZeRO-2 hetero RS",
         "model_grads_bucket_sweep": "Model grads bucket sweep"}
ORDER = ["layer_grads_hetero_ar", "model_grads_4layers", "layer_grads_mixed_dtype", "hs_a_64x64k_512k", "hs_c_32x256k_2m", "probe_het_24_mid", "probe_het_48_mid", "probe_het_48_small", "hetero_grad_ar_many", "zero2_hetero_rs"]
SEARCH = {"hs_a_64x64k_512k": "dp_grad_bucketing_64", "hs_c_32x256k_2m": "hs_c_32mixed"}
def load_json(p):
    try: return json.load(open(p))
    except Exception: return None
def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")
def _key(f):
    k = os.path.basename(f)[:-5]
    return k[:-6] if k.endswith("_dtype_dtype") else k
gate2 = {_key(f): load_json(f) for f in glob.glob(f"{G}/gate_final_v12/*.json") if not f.endswith(".v11.json")}
gate1 = {_key(f): load_json(f) for f in glob.glob(f"{G}/gate_round1/*.json") if not f.endswith(".v11.json")}
b32 = {}
for f in glob.glob(f"{G}/gate_bench32/newprob_bench_*.json"):
    j = load_json(f); b32[j["problem"]] = {c: v.get("med_ms") for c, v in j["results"].items()}
b64 = {}
for f in glob.glob(f"{G}/bench64_gates/newprob_bench_*_ws64.json"):
    j = load_json(f); b64[j["problem"]] = {c: v.get("med_ms") for c, v in j["results"].items()}
sim_all = load_json(f"{G}/sim_newprobs_v5_v11gates.json") or load_json(f"{G}/sim_newprobs_v5_v11all.json") or {}
ONE_SCALE = os.path.exists(f"{G}/sim_newprobs_v5_v11gates.json")   # gate finalists scored on the same scale as the hand-written candidates
md = ["# New-problem rows for every paper table (2-node data of 2026-10-27/28; 7-node columns pending)\n"]
# ---- Table: problems
md.append("## tab:problems — rows to add (\"Gradient-sync side\")\n\n| Problem | Role in the training step |\n|---|---|")
md.extend(["| Layer grads AR | DP all-reduce of one layer's heterogeneous gradients (q/k/v/o, up+gate, down, norms) |",
       "| Model grads AR | DP all-reduce of all layers' gradients; bucket-count choice |",
       "| Mixed-dtype grads AR | bf16 weight gradients with fp32 norm gradients |",
       "| Hetero-many grads AR | 48 odd-width 2-D gradients; grouping choice |",
       "| ZeRO-2 hetero RS | reduce-scatter of heterogeneous gradient shards |",
       "| Bucket-count probes (hs_a, hs_c, 24/48-piece) | mixed-size 1-D gradient sets; bucket-count calibration instances |", ""])
# ---- Table: setup
md.append("## tab:setup — column to add\n\n| | Grad-sync harness |\n|---|---|\n| Scale, steps | 2n·64r, 300 (7n·224r pending) |\n| Model shell | matmul → collective → loss → backward (collective re-issued in backward) |\n| Shapes | Llama-7B-class layer (DM 2048, HID 5376), 1-D mixed sets 64K-8M |\n| Collective(s) | one per script (gradient all-reduce / reduce-scatter) |\n")
# ---- Table: perproblem (3 scopes) + inversion data
md.append("## tab:perproblem — rows to add (ms; 1n bench = 32 ranks, 2n bench = 64 ranks, 2n train = 64-rank training step; 7n columns pending)\n")
md.append("| Problem | 1n bench base | 1n bench Overlay | 2n bench base | 2n bench Overlay | 2n train base | 2n train Overlay | train speedup | Overlay strategy |\n|---|---|---|---|---|---|---|---|---|")
rows_ms = {}
for p in ORDER:
    g = gate2.get(p) or gate1.get(p)
    if not g: continue
    m = g["measured_ms"]; base = next(c for c in m if c.startswith("per_tensor")); ch = g["gate_choice"]; name = next((n for gi, n, s in g["gate_candidates"] if gi == ch), ch)
    pb = b32.get(p, {}); pb64 = b64.get(p, {})
    f = lambda d, c: (f"{d[c]:.2f}" if d.get(c) else "—")
    rows_ms[p] = (m[base], m[ch])
    bold = lambda a, b: (f"**{a}**", b) if (a != "—" and b != "—" and float(a) < float(b)) else ((a, f"**{b}**") if (a != "—" and b != "—") else (a, b))
    a1, b1 = bold(f(pb, base), f(pb, ch)); a2, b2 = bold(f(pb64, base), f(pb64, ch)); a3, b3 = bold(f"{m[base]:.1f}", f"{m[ch]:.1f}")
    md.append(f"| {SHORT[p]} | {a1} | {b1} | {a2} | {b2} | {a3} | {b3} | {m[base]/m[ch]:.2f}x | {name.replace('strat:', '').replace('baseline:', '')[:40]} |")
md.append("\nInversion figure data (fig:disagree): bars = base/Overlay at the three scopes above.\n")
# ---- Table: multiseed (n=2 gate rounds on the day-5 pair)
md.append("## tab:multiseed — rows to add (2-node pair, n = gate rounds; 7-node n>=5 pending)\n\n| Primitive | n | baseline (ms/step) | Overlay (ms/step) | Speedup |\n|---|---|---|---|---|")
for p in ORDER:
    g1, g2 = gate1.get(p), gate2.get(p)
    if not (g1 and g2): continue
    base = next(c for c in g2["measured_ms"] if c.startswith("per_tensor")); ch = g2["gate_choice"]
    ov = [g["measured_ms"][ch] for g in (g1, g2) if ch in g["measured_ms"]]; bs = [g["measured_ms"][base] for g in (g1, g2) if base in g["measured_ms"]]
    bs = sorted(set(round(x, 3) for x in bs)); ov_m, ov_s = statistics.mean(ov), (statistics.stdev(ov) if len(ov) > 1 else 0.0)
    md.append(f"| {SHORT[p]} | {len(ov)} | {statistics.mean(bs):.2f} ± {(statistics.stdev(bs) if len(bs)>1 else 0):.2f} | {ov_m:.2f} ± {ov_s:.2f} | {statistics.mean(bs)/ov_m:.2f}x |")
# ---- Table: cost calculus (search cost on the new problems)
md.append("\n## tab:cost-calculus — new-problem search cost (strategy-enumerate, simulator v11, + training-scope gate)\n")
md.append("| Problem | Phase-3 wall (s) | LLM calls | gate finalists + baseline (HW runs) | gate cluster time (min) |\n|---|---|---|---|---|")
gate_times = {}
for logf in (f"{G}/gates_v11.out", f"{G}/gates_v12.out"):
    try: lines = open(logf).read().splitlines()
    except Exception: continue
    stamps = [(re.search(r"\[(\d\d:\d\d:\d\d)\] gate (\S+) (\S+)", l), i) for i, l in enumerate(lines)]
    stamps = [(m.group(1), m.group(3), i) for m, i in stamps if m]
    for k, (t, hw, i) in enumerate(stamps):
        t2 = stamps[k + 1][0] if k + 1 < len(stamps) else None
        if t2:
            h1, m1, s1 = map(int, t.split(":")); h2, m2, s2 = map(int, t2.split(":")); gate_times[hw] = ((h2 * 3600 + m2 * 60 + s2) - (h1 * 3600 + m1 * 60 + s1)) / 60.0
tot_wall = tot_calls = tot_gate = 0
for p in ORDER:
    sp = SEARCH.get(p, p); wall = calls = None
    for tag in ("v12", "v11"):
        lf = f"{G}/phase3_{tag}_{sp}.log"
        if os.path.exists(lf):
            s = open(lf).read(); mw = re.search(r"wall (\d+)s", s); mc = re.search(r"budget ~(\d+) LLM calls", s)
            wall = int(mw.group(1)) if mw else None; calls = int(mc.group(1)) if mc else None; break
    g = gate2.get(p) or gate1.get(p); nf = (len(g["gate_candidates"]) + 1) if g else None; gt = gate_times.get(p)
    md.append(f"| {SHORT[p]} | {wall if wall else '—'} | {calls if calls else '—'} | {nf if nf else '—'} | {f'{gt:.0f}' if gt else '—'} |")
    tot_wall += wall or 0; tot_calls += calls or 0; tot_gate += gt or 0
md.append(f"| **total (10 problems)** | {tot_wall} s = {tot_wall/60:.0f} min | {tot_calls} | | {tot_gate:.0f} min of 2-node cluster |")
md.append("\nLLM cost: at the paper's observed Sonnet 4.5 rate (~$19 for ~80 calls over 8 problems, i.e. ~$0.24/call) the new problems cost about $" + f"{tot_calls*0.24:.0f}" + " in LLM calls (estimate; token counts not logged by this driver).\n")
# ---- Table: simulator fidelity (new problems, 64-rank training measurements of the day-5 pair)
md.append("## tab:simfidelity — rows to add (simulator v11 at training scope vs 64-rank training step; day-5 pair measurements)\n\n| Problem | n | rho | top-1 | regret | ratio-MAPE |\n|---|---|---|---|---|---|")
pooled = []
for p in ORDER:
    g = gate2.get(p) or gate1.get(p)
    if not g: continue
    meas = dict(g["measured_ms"]); sims = {} if ONE_SCALE else {gi: s / 1000.0 for gi, n, s in g["gate_candidates"]}
    for c, v in sim_all.items():
        pp, cand, scope = c.split("__")
        if pp == p and scope == "train" and v is not None and cand in meas: sims[cand] = v / 1000.0
    cands = [c for c in meas if c in sims]
    if len(cands) < 3: md.append(f"| {SHORT[p]} | {len(cands)} | — | — | — | — |"); continue
    x = [sims[c] for c in cands]; y = [meas[c] for c in cands]; rho = spearman(x, y)
    base = next(c for c in cands if c.startswith("per_tensor")); best = min(cands, key=lambda c: meas[c]); pick = min(cands, key=lambda c: sims[c])
    regret = (meas[pick] - meas[best]) / meas[best] * 100
    mape = statistics.mean(abs((sims[base] / sims[c]) - (meas[base] / meas[c])) / (meas[base] / meas[c]) for c in cands if c != base) * 100
    pooled += [((sims[base] / sims[c]), (meas[base] / meas[c])) for c in cands if c != base]
    md.append(f"| {SHORT[p]} | {len(cands)} | {rho:.2f} | {'yes' if pick == best else 'no'} | {regret:.1f}% | {mape:.0f}% |")
if pooled:
    md.append(f"| pooled rho on speedup ratios | {len(pooled)} | {spearman([a for a, b in pooled], [b for a, b in pooled]):.2f} | | | direction agreement {sum(1 for a, b in pooled if (a > 1) == (b > 1))}/{len(pooled)} |")
# ---- Table: constswap arms (new problems)
md.append("\n## tab:constswap — new-problem columns (per-problem Spearman of each arm against the 64-rank training step; hand-written candidates measured on the day-5 pair)\n")
arms = [("calibrated (reference)", "v11gates" if ONE_SCALE else "v11all")] + [(l, ("arm2_" if ONE_SCALE else "arm_") + t) for l, t in (("network x1/2", "net_x0p5"), ("network x1/4", "net_x0p25"), ("network x2", "net_x2"), ("network x4", "net_x4"), ("op costs = isolated", "opc_isolated"), ("run reset per graph", "reset_graph"), ("amort. knee 64 KB", "knee_64k"), ("amort. knee 1 MB", "knee_1m"), ("fwd+bwd factor 0", "fwdbwd_0"), ("fwd+bwd factor 2", "fwdbwd_2"))]
hdr = "| Arm | " + " | ".join(SHORT[p].split(" (")[0] for p in ORDER) + " |"; md += [hdr, "|---|" + "---|" * len(ORDER)]
for label, tag in arms:
    sj = load_json(f"{G}/sim_newprobs_v5_{tag}.json")
    if not sj: md.append(f"| {label} | " + " | ".join("pending" for _ in ORDER) + " |"); continue
    cells = []
    for p in ORDER:
        g = gate2.get(p) or gate1.get(p); meas = g["measured_ms"] if g else {}
        sims = {c.split("__")[1]: v / 1000.0 for c, v in sj.items() if c.split("__")[0] == p and c.split("__")[2] == "train" and v is not None}
        cands = [c for c in meas if c in sims]
        cells.append(f"{spearman([sims[c] for c in cands], [meas[c] for c in cands]):.2f} (n={len(cands)})" if len(cands) >= 3 else "—")
    md.append(f"| {label} | " + " | ".join(cells) + " |")
# ---- Table: ablation-cost (per-style search cost on the new problems; LLM-only wall, strategy-enumerate = v11/v12 runs)
md.append("\n## tab:ablation-cost — new-problem rows (wall = LLM search time on the laptop, 10 problems; cluster gate time separate)\n\n| Phase-3 style | Wall (min) | problems with a result | note |\n|---|---|---|---|")
for label, tags in (("strategy-enumerate (default)", ["v12", "v11"]), ("cc-react", ["style_ccreact"]), ("multi-island", ["style_multiisland"])):
    tot = 0; n = 0
    for pr in ORDER:
        sp = SEARCH.get(pr, pr)
        for tag in tags:
            lf = f"{G}/phase3_{tag}_{sp}.log"
            if os.path.exists(lf):
                m = re.search(r"wall (\d+)s", open(lf).read())
                if m: tot += int(m.group(1)); n += 1; break
    md.append(f"| {label} | {tot/60:.0f} | {n}/10 | {'picks gated on hardware: results/gate_final' if label.startswith('strategy') else 'picks gated on hardware: results/gate_' + tags[0] + ' (pending)'} |")
md.append("\nLLM cost per style: the driver does not log tokens; at the paper's observed rate the three styles cost roughly in proportion to their wall time.\n")
# ---- Table: ablation-nonllm (four non-LLM searches, 400 simulator evaluations each, same v11 simulator at training scope)
md.append("## tab:ablation-nonllm — new-problem row\n\n| Problem | Best non-LLM find | Sim vs base | HW (ms) | OverlayCCL (ms) |\n|---|---|---|---|---|")
found = []
for f in sorted(glob.glob(f"{G}/nonllm/*/generic_search_default_s0.json")):
    d = load_json(f) or {}
    for prob, r in d.items():
        if not isinstance(r, dict): continue
        vals = [(r[m]["best_sim"] if isinstance(r.get(m), dict) else r.get(m)) for m in ("random", "ga", "beam", "rules") if isinstance(r.get(m), (int, float, dict)) and (not isinstance(r.get(m), dict) or isinstance(r[m].get("best_sim"), (int, float)))]
        if not vals or not isinstance(r.get("start_sim"), (int, float)): continue
        found.append((prob, r["start_sim"], min(vals)))
imp = [(p, s0, b) for p, s0, b in found if b < s0 * 0.999]
if imp:
    for p, s0, b in imp: md.append(f"| {p} | edit sequence (see nonllm/{p}) | {b/s0:.2f}x | pending | see gate table |")
md.append(f"| all {len(found)} new problems{' (others)' if imp else ''} | none (every edit sequence scores >= baseline) | 1.00x | --- | see tab:perproblem rows |")
md.append("\n(random walk, GA, beam, rule-based; 400 simulator evaluations per problem from the per-tensor baseline; the generic edit space never produces concatenation, bucketing, dtype casts or shape-class grouping, as in the paper.)\n")
# ---- Table: e2e preview (2-node amp5 matrix: amp3's four sites + the gradient-sync site)
md.append("## tab:e2e / tab:llama-sweep — 2-node preview from train_llama_e2e_amp5.py (64 ranks, 2 layers/stage, 4 microbatches, 300 steps; 7-node numbers pending)\n")
md.append("| backend | gradient sync | mixed dtype | sync x | step (ms) | loss (per-mb normalized) |\n|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{G}/amp5/llama_e2e_amp5_*_L2_seed0.json")):
    d = load_json(f); nmb = d.get("N_MB", 4); loss = d["final_loss"] / (nmb if d["backend"] == "per_mb" else 1)
    md.append(f"| {d['backend']} | {d['grad_sync']} | {d['mixed']} | {d['sync_repeat']} | {d['steady_median_ms']:.2f} | {loss:.3f} |")
md.append("\nReading: the four amp3 sites give 2.2x (38.6 -> 17.5 ms); on this replicated set (8 equal-size attention matrices + 4 norm gains + embedding + head shard) the per-tensor sync is already fast and every bundle adds 2-4 ms of concatenation, i.e. the heterogeneous-set inversion does not appear when most members are equal-sized. A grouped-query (GQA) variant with a heterogeneous replicated set is being measured (results_amp5_2node/gqa_*).\n")
# ---- Table: e2e at 7 nodes (224 ranks), both sync-graph forms (Sydney block 2026-10-29); numbers from chain_sy7.out bench lines
md.append("## tab:e2e (Llama) — 7-node rows, train_llama_e2e_amp5.py (224 ranks, GQA=4, 2 layers/stage, 4 microbatches, 300 steps, loss identical 3.466 per-mb-normalized)\n")
md.append("| sync graph | arm | step (ms) | speedup |\n|---|---|---|---|")
E2E7 = [("fused", "baseline: per-microbatch, per-tensor sync", 39.49), ("fused", "OverlayCCL: 4 sites, per-tensor sync (gate choice)", 14.98), ("fused", "4 sites + 8-count buckets", 17.89), ("fused", "4 sites + per-layer bundle", 70.17), ("fused", "4 sites + flat / bf16 flat", None),
        ("isolated", "baseline", 38.55), ("isolated", "OverlayCCL: 4 sites, per-tensor sync (gate choice)", 12.76), ("isolated", "4 sites + per-layer bundle", 13.98), ("isolated", "4 sites + 8-count buckets", 15.16), ("isolated", "4 sites + flat", 16.29), ("isolated", "4 sites + bf16 flat (mixed dtype)", 16.27)]
for g, arm, ms in E2E7:
    base = 39.49 if g == "fused" else 38.55
    md.append(f"| {g} | {arm} | {ms if ms else 'no result in 20 min'} | {f'{base/ms:.2f}x' if ms else '—'} |")
md.append("\nReading: at 224 ranks the gradient-sync site keeps per-tensor as the best strategy in BOTH graph forms; every bundle of the replicated set costs 1.2-4.7x the per-tensor step in the fused graph and 1.1-1.3x in the isolated graph. The loop's gate therefore selects per-tensor and the Llama headline is the four-site speedup (2.64x fused, 3.01x isolated).\n")
# ---- Table: perproblem 7-node columns (224-rank bench + 224-rank training, Sydney block 2026-10-29)
b224 = load_json(f"{G}/sy7_bench224/bench224_from_log.json") or {}
t224 = collections.defaultdict(dict)
for f in glob.glob(f"{G}/sy7_train*/train/*.json"):
    j = load_json(f); t224[j["problem"]].setdefault(j["cand"], []).append(j["measured_ms"])
md.append("\n## tab:perproblem — 7-node columns (224-rank bench, 224-rank training step; OverlayCCL = gate winner; n = training rounds)\n")
md.append("| Problem | 7n bench base | 7n bench Overlay | 7n train base | 7n train Overlay | train speedup | n |\n|---|---|---|---|---|---|---|")
for p in ORDER:
    g = gate2.get(p) or gate1.get(p)
    if not g: continue
    m = g["measured_ms"]; base = next(c for c in m if c.startswith("per_tensor")); ch = g["gate_choice"]
    bb, bo = (b224.get(p, {}).get(base), b224.get(p, {}).get(ch)); tb, to = t224.get(p, {}).get(base), t224.get(p, {}).get(ch)
    f = lambda v: f"{v:.2f}" if v else "—"; fm = lambda l: f"{statistics.mean(l):.1f}" + (f" ± {statistics.stdev(l):.1f}" if l and len(l) > 1 else "") if l else "—"
    sp = f"{statistics.mean(tb)/statistics.mean(to):.2f}x" if tb and to else "—"
    md.append(f"| {SHORT[p]} | {f(bb)} | {f(bo)} | {fm(tb)} | {fm(to)} | {sp} | {min(len(tb or []), len(to or []))} |")
md.append("\nzero2_hetero_rs: no 224-rank training result (shard sizes not divisible by 224); probe 48-mid buckets: see sy7_train_48mid if the longer-timeout rerun finished.\n")
# ---- e2e: cluster-size subsets and second seed (from chain_sy7.out bench lines)
md.append("## tab:3node — Llama rows (amp5, fused sync graph, HID = 24 x ranks, VOCAB = 32 x ranks)\n\n| Harness | base (ms) | OverlayCCL (ms) | Speedup |\n|---|---|---|---|")
for lab, b, o in (("Llama amp5 7n (224r)", 39.49, 14.98), ("Llama amp5 5n (160r)", 38.29, 14.58), ("Llama amp5 3n (96r)", 36.34, 14.36)):
    md.append(f"| {lab} | {b:.2f} | {o:.2f} | {b/o:.2f}x |")
md.append("\nSecond seed at 7n (fused): baseline 39.89, OverlayCCL 15.11 (2.64x); isolated second seed at 7n: baseline 38.04, OverlayCCL 13.07 (2.91x; seed 0: 38.55 / 12.76 = 3.02x).\n")
# ---- e2e footprint of the new problems in real 10B models (REPORT.md section 8)
md.append("\n## e2e footprint — gradient-sync share of real 10B steps (224 ranks)\n")
md.append("OLMoE-10B (paper configuration, 7-node phase timers): per-tensor replicated-gradient sync = 38 ms of a 4257 ms baseline step (0.9%), 37 ms of a 3035 ms OverlayCCL step (1.2%); both arms per-tensor.\n")
md.append("Real 10B Llama TP=32 DP gradient set (291 pieces, 312M elements bf16) in the 224-rank training harness, 60 steps (results/sy7_10b):\n\n| strategy | step (ms) |\n|---|---|")
for f in sorted(glob.glob(f"{G}/sy7_10b/train/newprob_train_llama10b_dp_grads__*.json")):
    j = load_json(f); md.append(f"| {j['cand']} | {j['measured_ms']:.1f} |")
md.append("\nAll strategies within +-3%: bandwidth-bound, no inversion; the paper's original simulator predicts a tie within 2%, the v11 heterogeneous-set term wrongly predicts a 12x per-tensor penalty (see REPORT.md section 8).\n")
# ---- The two problems carried into the Llama e2e (paper selection rule: lowest sim among correctness-gate survivors)
sd = load_json(f"{G}/loop_simdecides/summary.json") or {}
md.append("\n## chosen problems (paper rule = simulator-decides with the fitted v11 terms; training gate = validation only)\n")
md.append("| Problem | ingredient | paper-rule pick | 64r speedup | 224r speedup (n) | no-sim pick | no-sim 224r |\n|---|---|---|---|---|---|---|")
NOSIM = {"layer_grads_mixed_dtype": "3 in-order pairs, dtype preserved", "probe_het_48_small": "single flat concat", "hs_a_64x64k_512k": "two-phase hierarchical buckets"}
ING = {"layer_grads_mixed_dtype": "bf16 wire cast", "probe_het_48_small": "pathology-aware grouping (8-piece in-order)", "hs_a_64x64k_512k": "pathology-aware grouping (2 alternating buckets)"}
ns = collections.defaultdict(dict)
for f in glob.glob(f"{G}/sy7_nosim*/train/*.json"):
    j = load_json(f); ns[j["problem"]][j["cand"]] = j["measured_ms"]
for p in ("layer_grads_mixed_dtype", "probe_het_48_small", "hs_a_64x64k_512k"):
    d = sd.get(p, {}); base = t224.get(p, {}).get("per_tensor_ar"); pick = t224.get(p, {}).get(d.get("pick"))
    s224 = f"{statistics.mean(base)/statistics.mean(pick):.2f}x ({min(len(base), len(pick))})" if base and pick else "—"
    nsv = ns.get(p, {}); nsr = f"{statistics.mean(base)/nsv['nosim_1']:.2f}x" if base and nsv.get("nosim_1") else "pending"
    md.append(f"| {SHORT[p]} | {ING[p]} | {d.get('pick_name', '—')[:40]} | {d.get('speedup64', float('nan')):.2f}x | {s224} | {NOSIM[p]} | {nsr} |")
# ---- amp6 e2e rows
md.append("\n## tab:e2e (Llama) — amp6: four amp3 sites + the two per-problem gradient-sync sites (224 ranks unless noted)\n")
md.append("| arm | sync graph | nodes | step (ms) | loss | tag |\n|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{G}/amp6/json/llama_e2e_amp6_*.json")):
    j = load_json(f)
    if not j or "steady_median_ms" not in j: continue
    arm = f"{j['backend']} / {j['grad_sync'] if j['backend'] != 'per_mb' else 'base sync ' + j.get('base_sync', '')}"
    md.append(f"| {arm} | {j.get('sync_graph')} | {j.get('ws', 0)//32} | {j['steady_median_ms']:.2f} | {j.get('final_loss') if j.get('final_loss') is None else round(j['final_loss'], 3)} | {j.get('tag', '')} |")
# ---- hunt: all-gather layout problem (three scopes) and the clean-shell e2e summary
md.append("\n## tab:perproblem — hunt row: fsdp_ag_layout (FSDP prefetch gathers; strategy type = collective lowering/layout)\n")
hb = {}; ht = collections.defaultdict(list)
for f in glob.glob(f"{G}/sy7_hunt/train/newprob_train_fsdp_ag_layout__*.json"):
    j = load_json(f); ht[j["cand"]].append(j["measured_ms"])
md.append("| candidate | 1n bench (32r) | 7n bench (224r lone call) | 7n training step | sim v12 (us) |\n|---|---|---|---|---|")
simv = {"ag_pinned": 52358, "ag_dim0_t": 52856, "ag_dim0_t_native": 30552}
b32 = {"ag_pinned": "fails to compile (transposed variants)", }
for c, lab in (("ag_pinned", "pinned default (developer baseline)"), ("ag_dim0_t", "leading-dim gather + transposes, pinned"), ("ag_dim0_t_native", "leading-dim gather + transposes, native lowering")):
    t = ht.get(c); tt = f"{statistics.mean(t):.1f}" if t else "—"
    b32v = "ok" if c == "ag_pinned" else "does not compile"; b224 = "16.1" if c == "ag_pinned" else "does not compile"
    md.append(f"| {lab} | {b32v} | {b224} | {tt} | {simv[c]} |")
md.append("\nSpeedup (training step): native 1.50x over pinned; the lone-call scopes cannot rank the winner (compile failure outside the training graph).\n")
md.append("\n## tab:e2e / tab:3node — clean amp3-equivalent shell (amp6 SITES=0), isolated sync graph, 300 steps\n")
md.append("| nodes | baseline (per-mb) | OverlayCCL (bundled, pinned AG) | OverlayCCL + native AG | speedup pinned / native |\n|---|---|---|---|---|")
rows = {}   # parsed from the chain log (the result-file tags do not carry the node count, so subsets overwrite each other)
import re as _re
_lines = open(f"{G}/amp6/chain_clean.out").read().splitlines() if os.path.exists(f"{G}/amp6/chain_clean.out") else []
_tag = None
for ln in _lines:
    m = _re.match(r"=== \[[0-9:]+\] (c[357]_\w+)", ln)
    if m: _tag = m.group(1); continue
    m = _re.search(r"steady_median=([0-9.]+)ms", ln)
    if m and _tag: rows[_tag] = float(m.group(1)); _tag = None
def _r(n, kind, sd=""):
    return rows.get(f"c{n}_{kind}{sd}")
for n in (7, 5, 3):
    for sd in ("", "_s1"):
        b = _r(n, "base_pinned", sd); o = _r(n, "bund_pinned", sd); v = _r(n, "bund_native", sd)
        if b and o: md.append(f"| {n}{' (seed 1)' if sd else ''} | {b:.2f} | {o:.2f} | {v:.2f} | {b/o:.2f}x / {b/v:.2f}x |" if v else f"| {n} | {b:.2f} | {o:.2f} | — | {b/o:.2f}x / — |")
bn = _r(7, "base_native")
if bn: md.append(f"\n7-node baseline with the native gather: {bn:.2f} ms (the developer baseline keeps the pinned default by convention).\n")
# ---- tab:perproblem / tab:multiseed with the re-labeled scopes: microbench = forward-collective harness (g_train), training = mirror harness (v10 / v11ag)
MB_WIN = {"layer_grads_hetero_ar": "flat_cat_ar", "model_grads_4layers": "flat_cat_ar", "probe_het_24_mid": "bucket6", "probe_het_48_mid": "bucket6", "hetero_grad_ar_many": "p3_pick3",
          "hs_c_32x256k_2m": "bucket_4mb", "layer_grads_mixed_dtype": "cast_flat_bf16", "hs_a_64x64k_512k": "bucket_4mb", "probe_het_48_small": "bucket6", "fsdp_ag_layout": "ag_pinned"}
TOOL = {k: "per_tensor_ar" for k in MB_WIN}; TOOL["fsdp_ag_layout"] = "ag_dim0_t_native"
mir = collections.defaultdict(lambda: collections.defaultdict(list))
for d in ("sy7_v10", "sy7_v10_r2", "sy7_v10_r3", "sy7_v10_r4", "sy7_v10_r5", "sy7_v11ag", "sy7_v11ag_r2", "sy7_v11ag_r3", "sy7_v11ag_r4", "sy7_v11ag_r5"):
    for f in glob.glob(f"{G}/{d}/train/*.json"):
        j = load_json(f); mir[j["problem"]][j["cand"]].append(j["measured_ms"])
mir1n = collections.defaultdict(dict)
for d in ("sy7_v10_1n", "sy7_v11ag_1n"):
    for f in glob.glob(f"{G}/{d}/*.json"):
        j = load_json(f); mir1n[j["problem"]][j["cand"]] = j["measured_ms"]
md.append("\n## tab:perproblem (re-labeled scopes) — developer baseline = microbench winner, OverlayCCL = per-tensor (native gather for FSDP AG)\n")
md.append("| Problem | 1n bench base/tool | 7n bench base/tool | microbench (fwd-collective harness, 224r) base/tool | 1n TRAINING base/tool | 7n TRAINING base/tool | 7n train speedup | n |\n|---|---|---|---|---|---|---|---|")
for pn in list(ORDER) + ["fsdp_ag_layout"]:
    if pn not in MB_WIN: continue
    b, o = MB_WIN[pn], TOOL[pn]
    def f1(dct, c):
        v = dct.get(c) if isinstance(dct, dict) else None
        if isinstance(v, dict): v = v.get("med_ms")
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
    _b224 = b224.get(pn, {}) if isinstance(b224, dict) else {}
    bb32 = "see tab:perproblem rows above"
    bb224 = f"{f1(_b224, b)}/{f1(_b224, o)}"
    mb = t224.get(pn, {}); mbs = f"{statistics.mean(mb[b]):.1f}/{statistics.mean(mb[o]):.1f}" if mb.get(b) and mb.get(o) else "—"
    tr = mir.get(pn, {}); tb, to = tr.get(b), tr.get(o)
    trs = f"{statistics.mean(tb):.1f}±{statistics.stdev(tb) if len(tb) > 1 else 0:.1f}/{statistics.mean(to):.1f}±{statistics.stdev(to) if len(to) > 1 else 0:.1f}" if tb and to else "pending"
    sp = f"{statistics.mean(tb)/statistics.mean(to):.2f}x" if tb and to else "—"
    n1 = mir1n.get(pn, {}); one = f"{f1(n1, b)}/{f1(n1, o)}" if n1 else "pending"
    md.append(f"| {SHORT.get(pn, pn)} | {bb32} | {bb224} | {mbs} | {one} | {trs} | {sp} | {min(len(tb or []), len(to or []))} |")
md.append("\n(1n/7n bench = lone call; 'microbench' = the former per-problem harness (collective in the forward on flat-buffer views); TRAINING = mirror harness: the set as parameters of a small model, N_MB=4 accumulate, sync after the backward.)\n")
# ---- Tables needing 7 nodes or other runs
md.append("\n## pending\n\n* tab:e2e (OLMoE / Llama end-to-end) — needs the 7-node runs; the 2-node amp5 matrix (results/amp5) gives a preview of the gradient-sync site.\n* tab:llama-sweep, tab:3node, tab:ablation-nosim, appendix nosim per-problem — 7-node.\n* appendix per-problem style cells — cc-react and multi-island picks are being gated on hardware (results/gate_style_*).\n* tab:probe-tools / tab:cost-calculus text — add the training-scope gate as a tool/row (text change, later).\n")
open(OUT, "w").write("\n".join(md) + "\n"); print("wrote", OUT, len(md), "lines")
