#!/usr/bin/env python3
"""Llama-7B end-to-end rows of Tables 4, 5 and 9 and the Llama-7B shell row of Table 6, from the per-arm JSON files.
usage: python3 reproduce/e2e_tables.py [json dir]   (default: experiments/cross_scope_2026_10_26/new_problems/results/amp7_json)
Each JSON is one 300-step run; steady_median_ms is the median over steps 100-300 (the paper's Llama numbers), steady_mean_ms
the mean. The warm 224-rank seed-0 pair of the paper (600.1 / 79.1 ms) was overwritten by the cold-cache repeat (595.4 / 72.3,
same tag); both are in results/chain_amp7.out ([bench] lines)."""
import glob, json, os, re, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "experiments/cross_scope_2026_10_26/new_problems/results/amp7_json")
runs = []
for f in sorted(glob.glob(f"{D}/*.json")):
    j = json.load(open(f)); t = j["tag"]
    m = re.search(r"_M(\d+)_S(\d+)_.*_ws(\d+)_seed(\d+)", t); M, S, ws, seed = (int(x) for x in m.groups()) if m else (None,) * 4
    if m is None:
        m2 = re.search(r"_M(\d+)_.*_ws(\d+)_seed(\d+)", t); M, ws, seed = (int(x) for x in m2.groups()); S = 1024
    arm = ("baseline" if "per_mb_base_per_mb_microbench" in t else "nosim" if "bundled_nosim" in t else "control" if "bundled_microbench" in t
           else "overlayccl" if ("bundled_per_tensor" in t and "agdim0_t_native" in t) else t)
    runs.append(dict(arm=arm, M=M, S=S, ws=ws, seed=seed, med=j["steady_median_ms"], mean=j["steady_mean_ms"], loss=j["final_loss"], tag=t))
def pick(arm, **kw): return [r for r in runs if r["arm"] == arm and all(r[k] == v for k, v in kw.items())]
def row(label, b, o):
    if b and o: print(f"{label:34s} base {b[0]['med']:7.1f}  OverlayCCL {o[0]['med']:6.1f}  speedup {b[0]['med']/o[0]['med']:.2f}x  loss {b[0]['loss']:.3f}/{o[0]['loss']:.3f}")
    else: print(f"{label:34s} missing ({'base' if not b else ''}{' tool' if not o else ''})")
print("# Table 4 / Table 5 Llama-7B rows (knobs and cluster sizes) (steady median ms/step)")
row("7 nodes, S=1024, M=4, seed 0", pick("baseline", ws=224, S=1024, M=4, seed=0), pick("overlayccl", ws=224, S=1024, M=4, seed=0))
row("knob S=2048, M=4", pick("baseline", ws=224, S=2048, M=4, seed=0), pick("overlayccl", ws=224, S=2048, M=4, seed=0))
row("knob S=1024, M=8", pick("baseline", ws=224, S=1024, M=8, seed=0), pick("overlayccl", ws=224, S=1024, M=8, seed=0))
row("5 nodes (160 ranks)", pick("baseline", ws=160, S=1024, M=4), pick("overlayccl", ws=160, S=1024, M=4))
row("3 nodes (96 ranks)", pick("baseline", ws=96, S=1024, M=4), pick("overlayccl", ws=96, S=1024, M=4))
print("\n# Table 9 Llama-7B row (no-sim picks at every gradient site) and the control arm (OverlayCCL schedule, microbench strategies)")
for a in ("nosim", "control"):
    r = pick(a, ws=224, S=1024, M=4); b = pick("baseline", ws=224, S=1024, M=4, seed=0)
    if r and b: print(f"{a:10s} {r[0]['med']:7.1f} ms  vs baseline {b[0]['med']:.1f}: {b[0]['med']/r[0]['med']:.2f}x")
print("\n# Table 6 Llama-7B shell row (224 ranks, per-seed medians)")
b = sorted(pick("baseline", ws=224, S=1024, M=4), key=lambda r: r["seed"]); o = sorted(pick("overlayccl", ws=224, S=1024, M=4), key=lambda r: r["seed"])
print("seeds:", [r["seed"] for r in b], [f"{r['med']:.1f}" for r in b], "|", [f"{r['med']:.1f}" for r in o])
if len(b) > 1 and len(o) > 1:
    bm, om = [r["med"] for r in b], [r["med"] for r in o]
    print(f"n={min(len(bm), len(om))}  base {st.mean(bm):.1f} +- {st.stdev(bm):.1f}  OverlayCCL {st.mean(om):.1f} +- {st.stdev(om):.1f}  speedup {st.mean(bm)/st.mean(om):.2f}x")
