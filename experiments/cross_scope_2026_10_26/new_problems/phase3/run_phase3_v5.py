#!/usr/bin/env python3
"""Run OverlayCCL's Phase 2 (seed baselines) + Phase 3 (strategy-enumerate, Sonnet 4.5 via Bedrock) on a registered new
problem with the v5 simulator at training scope (2 nodes, 64 ranks) patched in. Prints the sim ranking of everything the
loop produced and saves the candidates' code for hardware verification. usage: run_phase3_v5.py <problem> [max_rounds]"""
import os, sys, json, importlib.util, time
REPO = os.path.expanduser("~/work/paper/overlayccl_public"); sys.path.insert(0, REPO); os.chdir(REPO)
import search.problems_newprobs
from search.problems import get_problem
import search.correctness_test as CT
from experiments.run_search import phase2_generic_baseline, make_send_counts, phase3_generic_evolution_dispatch
from simulator.topology import MultiNodeTopology
from simulator.cost_model import CostModel
R = os.path.expanduser("~/work/paper/overlayccl_public/experiments/cluster_224rank")
spec = importlib.util.spec_from_file_location("ct5", f"{REPO}/experiments/cluster_224rank/sim/correctness_test_v5.py"); ct5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct5)
P = json.load(open(f"{R}/results/hwprobe/hw_probe_measured.json")); DT = P["derived_table"]; SP = P["sim_params"]
SCALE = float(os.environ.get("SCALE", "64"))
import importlib.util as _ilu
_hp = _ilu.spec_from_file_location('het_penalty', f'{REPO}/experiments/cluster_224rank/sim/het_penalty.py'); _hpm = _ilu.module_from_spec(_hp); _hp.loader.exec_module(_hpm); het = _hpm.make_het_penalty()
V5KW = dict(op_costs=dict(DT["xla_op_overhead_us_marginal"]), dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"],
    compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000,
    pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
    memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"], hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"] * 1000.0,
    coll_curves=None, amort_knee_bytes=262144, local_fwd_bwd_factor=1.0, ablate="", index_select_bytes_per_us=0.0,
    microbatch_loops_induce_graphs=False, coll_run_reset="fusion", training_scale_bytes_multiplier=SCALE,
    het_set_penalty=het, unequal_concat_factor=1.0, unequal_piece_penalty=_hpm.make_unequal_piece_penalty())
def patched_bench(problem, fn, topo, send, world, num_nodes=1, **kw):
    kw2 = dict(kw); kw2.update(V5KW); kw2["unsupported_primitives"] = ["all_to_all"]
    return ct5.benchmark_xla_candidate_generic(problem, fn, topo, send, world, num_nodes=num_nodes, **kw2)
CT.benchmark_xla_candidate_generic = patched_bench
name = sys.argv[1]; max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
problem = get_problem(name); WORLD, NODES = 64, 2
topo = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5, num_devices_per_node=16, cores_per_device=2,
                         device_adjacency=None, efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
send = make_send_counts("moe", world=WORLD)
t0 = time.time()
base = phase2_generic_baseline(problem, topo, SP["dispatch_overhead_us"], NODES, unsupported_primitives=["all_to_all"], training_scale_bytes_multiplier=SCALE)
cm = CostModel(topo, send, dispatch_overhead_us=SP["dispatch_overhead_us"])
res = phase3_generic_evolution_dispatch(problem, topo, send, cm, base, True, os.environ.get("LLM", "sonnet"), NODES, max_rounds, verbose=True, phase3_style=os.environ.get("P3_STYLE", "strategy-enumerate"))
out = []
for r in res:
    if isinstance(r, dict):
        d = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in r.items()}
    else:
        d = {"raw": [(x if isinstance(x, (str, int, float, bool, type(None))) else str(x)) for x in r]}
        for x in r:
            if isinstance(x, str) and "def evolved_" in x: d["code"] = x
            elif isinstance(x, (int, float)) and "sim_us" not in d and x > 0: d["sim_us"] = x
            elif isinstance(x, str) and "name" not in d and "def " not in x: d["name"] = x
    out.append(d)
def _key(r):
    for k in ("sim_us", "sim_time_us", "sim_time"):
        if isinstance(r.get(k), (int, float)): return r[k]
    return 1e18
out.sort(key=_key)
od = os.path.expanduser("~/work/g2_backup/phase3_runs"); os.makedirs(od, exist_ok=True)
json.dump(out, open(f"{od}/{name}_{os.environ.get('TAGP3', 'v5')}.json", "w"), indent=1)
print(f"\n=== Phase 3 result for {name} (wall {time.time()-t0:.0f}s): {len(out)} candidates, sim ranking ===")
for r in out: print(f"  {str(_key(r)):>14}  {str(r.get('name'))[:70]}  code={'yes' if r.get('code') else 'no'}")
