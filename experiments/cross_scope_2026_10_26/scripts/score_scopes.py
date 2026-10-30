#!/usr/bin/env python3
"""Two-scope simulator scoring for the (g) candidate pool. Same cost model and probe table as score_v4.py
(the paper's calibrated simulator), parameterized by world/scope:
  SCOPE=bench  : WORLD=32, 1 node, isolated op costs, dispatch run reset per graph  (the 1-node microbench)
  SCOPE=train  : WORLD=224 (or env WORLD) over WORLD/32 nodes, marginal op costs, fusion-aware runs (training scope)
Env: MANIFEST, OUTDIR, TAG, WORLD, TABLE, CT_V4, REPO, MULTS (json), PROBLEMS (comma list).
"""
import json, os, re, sys, importlib.util
REPO = os.environ.get("REPO", "/home/ubuntu/OverlayCCL"); sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("ct_v4", os.environ.get("CT_V4", "/home/ubuntu/ap7/correctness_test_v4.py"))
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
from search.problems import get_problem
import search.problems_modext
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts, _problem_train_scale_multiplier
import torch

SCOPE = os.environ.get("SCOPE", "train")
WORLD = int(os.environ.get("WORLD", "32" if SCOPE == "bench" else "224")); NODES = max(1, WORLD // 32)
OPC = os.environ.get("OPC", "isolated" if SCOPE == "bench" else "marginal")
RESET = os.environ.get("RESET", "graph" if SCOPE == "bench" else "fusion")
P = json.load(open(os.environ["TABLE"])); DT = P["derived_table"]; SP = P["sim_params"]
op_costs = dict(DT["xla_op_overhead_us_marginal"] if OPC == "marginal" else DT["xla_op_overhead_us"])
KW = dict(op_costs=op_costs, dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"],
          compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000,
          pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
          memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"],
          hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"] * 1000.0,
          coll_curves=None, amort_knee_bytes=262144, local_fwd_bwd_factor=1.0, ablate="",
          index_select_bytes_per_us=(1_000_000 * 4) / float(DT["index_select_scaling"]["large_index_1M"]) if "large_index_1M" in DT.get("index_select_scaling", {}) else 0.0,
          microbatch_loops_induce_graphs=False, coll_run_reset=RESET)
TOPO = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5, num_devices_per_node=16, cores_per_device=2,
                         device_adjacency=None, efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
send = make_send_counts("moe", world=WORLD)
man = json.load(open(os.environ["MANIFEST"]))

def fit_curve(table, bytes_key_factor):
    pts = []
    for k, v in table.items():
        if not isinstance(v, (int, float)): continue
        nb = int(k.split("bytes_")[1].split("_")[0]); pts.append((nb * bytes_key_factor, float(v)))
    if len(pts) < 2: return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0; L = max(my - slope * mx, min(ys) * 0.5)
    return (round(L, 1), round((1.0 / slope) if slope > 0 else 1e12, 1))
cws = P["world_size"]
CURVES = {"all_gather": fit_curve(DT["all_gather_latency_us"], cws), "all_reduce": fit_curve(DT["all_reduce_latency_us"], 2.0),
          "reduce_scatter": fit_curve({k.replace("_in", ""): v for k, v in DT["reduce_scatter_latency_us"].items()}, 1.0)}
KW["coll_curves"] = {k: v for k, v in CURVES.items() if v}

# real per-problem-harness shapes (same as score_v4.py; shapes that depend on world size use WORLD)
def dxe_real(ws, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed); T, V = 256, 32256; V_local = V // ws
    targets = torch.randint(0, V_local * ws, (T,), dtype=torch.long)
    args = [{"logits_local": torch.randn(T, V_local), "targets": targets, "V_local": V_local} for r in range(ws)]
    try:
        from search.problems import _dxe_reference; expected = _dxe_reference(args, ws)
    except Exception:
        expected = None
    return {"per_rank_args": args, "shared_args": {}, "expected": expected}
def _wrap(g, **kw):
    return lambda ws, pattern="uniform", *a, **k: g(ws, pattern, **kw)
REAL = {"dxe": lambda prob: setattr(prob, "generate_test_case", dxe_real),
        "tp_mlp":         lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, N_LAYERS=1, B=1, S=1024, DM=2048, seed=0)),
        "fsdp_prefetch":  lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, N_LAYERS=1, shard_size=5376 // WORLD, DM=2048, seed=0)),
        "pp_send_recv":   lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, M=4, B=1, S=512, DM=2048, seed=0)),
        "llama_block_ar": lambda prob: setattr(prob, "generate_test_case", _wrap(prob.generate_test_case, B=1, S=1024, DM=2048, seed=0))}
DTYPE_MULT = {"tp_mlp": 0.5, "fsdp_prefetch": 0.5, "pp_send_recv": 0.5, "llama_block_ar": 0.5, "dxe": 1.0}
MULTS = json.loads(os.environ.get("MULTS", "{}"))
BASE = {"alltoallv": "allgather_reduce_scatter", "uniform_a2a": "allgather_reduce_scatter", "ring_kv": "per_chunk_per_part_allgather",
        "dxe": "full_vocab_ag", "pp_send_recv": "per_mb", "tp_mlp": "per_mb", "fsdp_prefetch": "per_mb", "llama_block_ar": "sequential_2ar"}
def code_for(c, prob):
    if c["kind"] == "builtin": return prob.builtin_templates[c["name"]]
    if c["kind"] == "code": return open(c["code"]).read()
    return prob.builtin_templates[BASE[c["problem"]]]
TAG = os.environ.get("TAG", f"{SCOPE}_{WORLD}")
ONLY = [x for x in os.environ.get("PROBLEMS", "").split(",") if x]
print(f"scope={SCOPE} world={WORLD} nodes={NODES} opc={OPC} reset={RESET} mults={MULTS}", flush=True)
out = {}
for c in man:
    if ONLY and c["problem"] not in ONLY: continue
    key = f"{c['problem']}__{c['cand']}"
    try:
        prob = get_problem(c["problem"]); code = code_for(c, prob)
        if c["problem"] in REAL: REAL[c["problem"]](prob)
        ns = {}; exec(code, ns); fn = ns[prob.evolved_fn_name]; fn.__candidate_source__ = code
        kw = dict(KW); kw["training_scale_bytes_multiplier"] = MULTS.get(c["problem"], DTYPE_MULT.get(c["problem"], _problem_train_scale_multiplier(prob)))
        b = ct.benchmark_xla_candidate_generic(prob, fn, TOPO, send, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **kw)
        v = None if "error" in b else round(b["sim_time_us"], 1)
        extra = (f"coll={b.get('total_bytes')} local={b.get('local_cost_us', 0):.0f} steps={b.get('steps')} neff={b.get('n_neff_graphs')}" if v is not None else b.get("error", "")[:120])
    except Exception as e:
        v = None; extra = f"{type(e).__name__}: {str(e)[:120]}"
    out[key] = v; print(f"{key:52s} V4={v}  {extra}", flush=True)
OUTDIR = os.environ.get("OUTDIR", "/tmp/g2/sim"); os.makedirs(OUTDIR, exist_ok=True)
json.dump(out, open(f"{OUTDIR}/sim_{TAG}.json", "w"), indent=1); print("wrote", f"{OUTDIR}/sim_{TAG}.json")
