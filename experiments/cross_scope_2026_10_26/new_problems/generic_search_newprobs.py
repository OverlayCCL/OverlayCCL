#!/usr/bin/env python3
"""Non-LLM searches (random walk, GA, beam, rules; experiments/cluster_224rank/scripts/generic_search.py) on the new
problems, scored by the simulator (constants from the environment: HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0 = original paper constants) at training scope on 2 nodes / 64 ranks, same as the Phase-3 runs.
usage: PROBLEMS=a,b BUDGET=400 python generic_search_newprobs.py"""
import os, sys, importlib.util, json
REPO = os.path.expanduser("~/work/paper/overlayccl_public"); os.environ["REPO"] = REPO; sys.path.insert(0, REPO)
SCR = f"{REPO}/experiments/cluster_224rank/scripts"; sys.path.insert(0, SCR)
import search.problems_newprobs
import search.correctness_test as CT
R = os.path.expanduser("~/work/paper/overlayccl_public/experiments/cluster_224rank")
spec = importlib.util.spec_from_file_location("ct5", f"{REPO}/experiments/cluster_224rank/sim/correctness_test_v5.py"); ct5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct5)
hp = importlib.util.spec_from_file_location("het_penalty", f"{REPO}/experiments/cluster_224rank/sim/het_penalty.py"); hpm = importlib.util.module_from_spec(hp); hp.loader.exec_module(hpm)
P = json.load(open(f"{R}/results/hwprobe/hw_probe_measured.json")); DT = P["derived_table"]; SP = P["sim_params"]
SCALE = float(os.environ.get("SCALE", "64"))
V5KW = dict(op_costs=dict(DT["xla_op_overhead_us_marginal"]), dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"],
    compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000,
    pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"],
    memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"], hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"] * 1000.0,
    coll_curves=None, amort_knee_bytes=262144, local_fwd_bwd_factor=1.0, ablate="", index_select_bytes_per_us=0.0,
    microbatch_loops_induce_graphs=False, coll_run_reset="fusion", training_scale_bytes_multiplier=SCALE,
    het_set_penalty=hpm.make_het_penalty(), unequal_concat_factor=1.0, unequal_piece_penalty=hpm.make_unequal_piece_penalty())
def patched(problem, fn, topo, send, world, num_nodes=1, **kw):
    kw2 = dict(kw); kw2.update(V5KW); kw2["unsupported_primitives"] = ["all_to_all"]
    return ct5.benchmark_xla_candidate_generic(problem, fn, topo, send, world, num_nodes=num_nodes, **kw2)
CT.benchmark_xla_candidate_generic = patched
import rewriter_search as RW   # the search iterates RW.BASELINE (problem -> developer-baseline builtin); register the new problems
for _p in ("layer_grads_hetero_ar", "model_grads_4layers", "layer_grads_mixed_dtype", "dp_grad_bucketing_64", "hs_c_32mixed", "hetero_grad_ar_many", "probe_het_24_mid", "probe_het_48_mid", "probe_het_48_small"):
    RW.BASELINE[_p] = "per_tensor_ar"
RW.BASELINE["zero2_hetero_rs"] = "per_tensor_rs"
# STARTS="problem:builtin,..." overrides the start of the search (the paper's developer baseline for the gradient problems is the
# microbench-winning bundle, OverlayCCL's pick is per-tensor; the default start above is per-tensor)
for _kv in [x for x in os.environ.get("STARTS", "").split(",") if x]:
    _p, _b = _kv.split(":"); RW.BASELINE[_p] = _b
src = open(f"{SCR}/generic_search.py").read()
src = src.replace("WORLD, NODES = 224, 7", "WORLD, NODES = 64, 2").replace("/home/ubuntu/sweep_results", "/tmp/g2/hw/nonllm").replace('sys.path.insert(0, "/home/ubuntu/ap7")', f'sys.path.insert(0, "{SCR}")')
src = src.replace("from search.correctness_test import benchmark_xla_candidate_generic,", "from search.correctness_test import benchmark_xla_candidate_generic,")
g = {"__name__": "__main__", "__file__": f"{SCR}/generic_search.py"}
exec(compile(src, f"{SCR}/generic_search.py", "exec"), g)
