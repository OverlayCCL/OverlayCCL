"""Score the day-2 inverting problems' candidates with the paper's calibrated simulator at bench scope (32 ranks, one node,
isolated op costs, run reset per graph) and training scope (64 ranks, two nodes, marginal op costs, fusion-aware runs).
Candidates come from g_newprobs.py (hardware harness) and are adapted to the sandbox signature; sizes are the hardware sizes
scaled by 1/64 (the simulator scales bytes back up with training_scale_bytes_multiplier=64)."""
import os, sys, json, inspect, importlib.util, re, textwrap
REPO = os.path.expanduser("~/work/paper/overlayccl_public"); sys.path.insert(0, REPO); sys.path.insert(0, "/tmp/g2")
R = os.path.expanduser("~/work/paper/overlayccl_public/experiments/cluster_224rank")
spec = importlib.util.spec_from_file_location("ct_v4", f"{R}/sim/correctness_test_v4.py"); ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
from search.problems import CollectiveProblem
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts
import torch
import g_newprobs as G
P = json.load(open(f"{R}/results/hwprobe/hw_probe_measured.json")); DT = P["derived_table"]; SP = P["sim_params"]
SCALE = 64  # hardware tensors / sandbox tensors (bytes); multiplier handed to the simulator
def kw(scope):
    opc = dict(DT["xla_op_overhead_us"] if scope == "bench" else DT["xla_op_overhead_us_marginal"])
    return dict(op_costs=opc, dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"],
        compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000,
        pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
        memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"], hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"] * 1000.0,
        coll_curves=None, amort_knee_bytes=262144, local_fwd_bwd_factor=1.0, ablate="", index_select_bytes_per_us=0.0,
        microbatch_loops_induce_graphs=False, coll_run_reset=("graph" if scope == "bench" else "fusion"), training_scale_bytes_multiplier=float(SCALE))
# sandbox-sized inputs: shrink each hardware tensor's element count by SCALE (keep >= 4 elements), bf16 -> float32 (bytes x2, so halve again)
def shrink(t):
    n = max(4, t.numel() // SCALE // 2)
    return torch.randn(n) * 0.01
def make_problem(name, bench_make, base_name, cands, reduce_kind):
    def generate(world_size, pattern="uniform", shard_size=64, seed=0):
        torch.manual_seed(seed)
        hw = bench_make(32, 0, "cpu")[0]
        per = [[shrink(t) for t in hw] for _ in range(world_size)]
        if reduce_kind == "ar":
            expected = [[sum(per[r][i] for r in range(world_size)) for i in range(len(hw))] for _ in range(world_size)]
        else:  # reduce-scatter: rank r gets the r-th slice of each summed tensor
            exp0 = [sum(per[r][i] for r in range(world_size)) for i in range(len(hw))]
            expected = [[e[r * (e.numel() // world_size):(r + 1) * (e.numel() // world_size)] for e in exp0] for r in range(world_size)]
        return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": expected}
    def call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
        return fn(ra["grads"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
    return CollectiveProblem(name=name, display_name=name, evolved_fn_name="evolved_fn", signature="", signature_doc="",
        reference_fn=lambda *a: None, generate_test_case=generate, call_candidate=call, builtin_templates={})
def adapt(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    src = re.sub(r"\.view_as\((\w+)\)", r".view(\1.shape)", src)
    src = src.replace("torch.nn.functional.pad(", "_pad(")
    helper = "def _pad(t, p):\n    return torch.cat([t, torch.zeros(p[1], dtype=t.dtype, device=t.device)]) if p[1] > 0 else t\n"
    # helpers the candidates reference
    extra = ""
    for h in ("_p7_flat_pad", "_c_bucket", "_c_bucket_mb", "groups_same_slot", "tp_of"):
        if h in src and hasattr(G, h): extra += textwrap.dedent(inspect.getsource(getattr(G, h))) + "\n"
    # make sizes scale-free: hardware code uses absolute element counts only in bucket limits (kept) -> shrink them too
    extra = extra.replace("1024 * 1024 // 2", f"1024 * 1024 // 2 // {SCALE}").replace("mb * 1024 * 1024 // 2", f"mb * 1024 * 1024 // 2 // {SCALE}")
    src = src.replace("1024 * 1024 // 2", f"1024 * 1024 // 2 // {SCALE}")
    name = fn.__name__
    # closures from _c_bucket(mb)/_c_bucket_mb(mb): bind the captured bucket size
    if fn.__closure__ and fn.__code__.co_freevars:
        for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            if isinstance(cell.cell_contents, (int, float)): extra = f"{var} = {cell.cell_contents}\n" + extra
    code = helper + extra + src + f"\ndef evolved_fn(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):\n    return {name}(grads, rank, world_size, xm, torch)\n"
    code = re.sub(r"\.view_as\((\w+)\)", r".view(\1.shape)", code)
    code = code.replace("torch.bfloat16", "torch.float32").replace(".float()", "").replace(".to(torch.float32)", "")
    return code
SETS = {"model_grads_4layers": "ar", "layer_grads_hetero_ar": "ar", "layer_grads_mixed_dtype": "ar", "hs_a_64x64k_512k": "ar", "hs_c_32x256k_2m": "ar",
        "hetero_grad_ar_many": "ar", "zero2_hetero_rs": "rs", "model_grads_bucket_sweep": "ar"}
out = {}
for pname, kind in SETS.items():
    PP = G.PROBLEMS[pname]
    for scope, WORLD in (("bench", 32), ("train", 64)):
        NODES = WORLD // 32
        TOPO = MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5, num_devices_per_node=16, cores_per_device=2,
                                 device_adjacency=None, efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
        prob = make_problem(pname, PP["make"], PP["baseline"], PP["cands"], kind)
        for cname, fn in PP["cands"].items():
            if cname in ("per_tensor_marked", "flat_copy_into_buffer", "pad_equal_cat"): continue
            if "_c_bucket(" in cname: pass
            try:
                code = adapt(fn) if not cname.startswith("bucket_") or pname != "model_grads_bucket_sweep" else adapt(fn)
                ns = {"torch": torch}; exec(code, ns); f = ns["evolved_fn"]; f.__candidate_source__ = code
                b = ct.benchmark_xla_candidate_generic(prob, f, TOPO, make_send_counts("moe", world=WORLD), WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **kw(scope))
                v = None if "error" in b else round(b["sim_time_us"], 1); err = b.get("error", "")[:90]
            except Exception as e:
                v = None; err = f"{type(e).__name__}: {str(e)[:90]}"
            out[f"{pname}__{cname}__{scope}"] = v
            print(f"{pname:26s} {cname:24s} {scope:5s} sim_us={v} {err}", flush=True)
json.dump(out, open("/tmp/g2/hw/sim_newprobs.json", "w"), indent=1)
