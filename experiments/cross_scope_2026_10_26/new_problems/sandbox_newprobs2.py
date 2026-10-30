# score the new problems' builtins in the paper's simulator sandbox (checks the problem registrations work end to end)
import json, os, sys, importlib.util
REPO=os.path.expanduser("~/work/paper/overlayccl_public"); sys.path.insert(0, REPO)
R=os.path.expanduser("~/work/paper/overlayccl_public/experiments/cluster_224rank")
spec=importlib.util.spec_from_file_location("ct_v4", f"{R}/sim/correctness_test_v4.py"); ct=importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
from search.problems import get_problem
import search.problems_modext, search.problems_newprobs
from simulator.topology import MultiNodeTopology
from experiments.run_search import make_send_counts
P=json.load(open(f"{R}/results/hwprobe/hw_probe_measured.json")); DT=P["derived_table"]; SP=P["sim_params"]
for WORLD in (64,):
    NODES=WORLD//32
    TOPO=MultiNodeTopology(num_nodes=NODES, neuronlink_bandwidth_GBps=192.0, neuronlink_latency_us=0.5, num_devices_per_node=16, cores_per_device=2, device_adjacency=None, efa_bandwidth_GBps=12.5, efa_latency_us=5.0, efa_adapters_per_node=8)
    KW=dict(op_costs=dict(DT["xla_op_overhead_us_marginal"]), dispatch_overhead_us=SP["dispatch_overhead_us"], graph_launch_overhead_us=SP["graph_launch_overhead_us"], compilation_cost_samples=SP["compilation_cost_samples"], compilation_load_events_per_run=2, compilation_amortize_steps=5000, pipeline_amort_alpha1=SP["pipeline_amort_alpha1"], pipeline_amort_alpha2=SP["pipeline_amort_alpha2"], pipeline_amort_alpha3=SP["pipeline_amort_alpha3"], memcpy_bytes_per_us=SP["memcpy_bytes_per_us"], memcpy_seq_bytes_per_us=SP["memcpy_seq_bytes_per_us"], hbm_bytes_per_us=DT["memcpy_throughput"]["clone_gbps"]*1000.0, coll_curves=None, amort_knee_bytes=262144, local_fwd_bwd_factor=1.0, ablate="", index_select_bytes_per_us=0.0, microbatch_loops_induce_graphs=False, coll_run_reset="fusion", training_scale_bytes_multiplier=1.0)
    send=make_send_counts("moe", world=WORLD)
    for n in ("hs_c_32mixed","hetero_grad_ar_many","model_grads_ep_groups"):
        prob=get_problem(n)
        for bname, code in prob.builtin_templates.items():
            ns={}; exec(code, ns); fn=ns[prob.evolved_fn_name]; fn.__candidate_source__=code
            try:
                b=ct.benchmark_xla_candidate_generic(prob, fn, TOPO, send, WORLD, num_nodes=NODES, unsupported_primitives=["all_to_all"], **KW)
                print(WORLD, n, bname, "sim_us=", round(b.get("sim_time_us", -1), 1), b.get("error", "")[:100], flush=True)
            except Exception as e:
                print(WORLD, n, bname, "EXC", type(e).__name__, str(e)[:120], flush=True)
