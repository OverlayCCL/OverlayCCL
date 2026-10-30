#!/usr/bin/env python3
"""1-node bench (32 ranks) and on-device correctness for the new problems in g_newprobs.py: one torchrun per problem,
all candidates in one process. Writes $OUT/newprob_bench_<problem>.json {cand: {med_ms, mean_ms, max_abs_diff, ok}}."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/ubuntu/g2")
import torch, torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_newprobs import PROBLEMS, flatten

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--problem", required=True); ap.add_argument("--iters", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
    a, _unknown = ap.parse_known_args()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); assert ws == xr.world_size() and ws > 1
    P = PROBLEMS[a.problem]; args = P["make"](ws, rank, dev); xm.mark_step()
    res = {}; ref = None
    order = [P["baseline"]] + [c for c in P["cands"] if c != P["baseline"]]
    for name in order:
        fn = P["cands"][name]; r = {}
        try:
            for _ in range(a.warmup):
                y = fn(*args, rank, ws, xm, torch); xm.mark_step(); _ = sum(o.float().sum() for o in flatten(y)).item()
            out = torch.cat([o.reshape(-1).float() for o in flatten(y)]).cpu()
            if ref is None: ref = out
            r["max_abs_diff"] = float((out - ref).abs().max()) if out.numel() == ref.numel() else None
            r["ok"] = r["max_abs_diff"] is not None and r["max_abs_diff"] <= 2e-2 * max(1.0, float(ref.abs().max()))
            ts = []
            for _ in range(a.iters):
                xm.mark_step(); t0 = time.time(); y = fn(*args, rank, ws, xm, torch); xm.mark_step(); _ = sum(o.float().sum() for o in flatten(y)).item()
                ts.append((time.time() - t0) * 1e3)
            r["med_ms"] = statistics.median(ts); r["mean_ms"] = statistics.mean(ts); r["status"] = "ok"
        except Exception as e:
            r["status"] = f"{type(e).__name__}: {str(e)[:200]}"; r["med_ms"] = None
            if rank == 0: import traceback; traceback.print_exc()
        res[name] = r
        if rank == 0: print(f"[bench2] {a.problem} {name}: {r}", flush=True)
    if rank == 0:
        out = os.environ.get("OUT") or os.environ.get("RESULTS_DIR", "/home/ubuntu/g2/newprobs"); os.makedirs(out, exist_ok=True)
        json.dump({"problem": a.problem, "ws": ws, "results": res}, open(f"{out}/newprob_bench_{a.problem}{'_ws'+str(ws) if ws != 32 else ''}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
