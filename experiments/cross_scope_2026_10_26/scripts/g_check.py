#!/usr/bin/env python3
"""Correctness check of every manifest candidate of one problem against the production baseline, on device, at the
harness shapes (same inputs as g_bench.py). One torchrun per problem. Writes $OUT/check_<problem>.json:
{cand: {"max_abs_diff": x, "ref_scale": y, "ok": bool, "status": ...}}."""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/g2"); sys.path.insert(0, os.environ.get("REPO", "/home/ubuntu/OverlayCCL"))
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_bench import BASE, FN, make_args, flatten

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--problem", required=True); ap.add_argument("--manifest", default="/home/ubuntu/g2/manifest_g_node.json")
    a = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    assert ws == xr.world_size() and ws > 1, f"world size mismatch env={ws} xr={xr.world_size()}"
    from search.problems import get_problem
    import search.problems_modext
    base_code = get_problem(a.problem).builtin_templates[BASE[a.problem]]
    ns = {}; exec(base_code, ns); base_fn = ns[FN[a.problem]]
    args, desc = make_args(a.problem, ws, rank, dev)
    def run(fn):
        y = fn(*args, rank, ws, 16, 2, xm, torch, num_nodes=1); xm.mark_step()
        return torch.cat([o.reshape(-1).float() for o in flatten(y)]).cpu()
    ref = run(base_fn); scale = float(ref.abs().max())
    res = {}
    for c in json.load(open(a.manifest)):
        if c["problem"] != a.problem or c["kind"] != "code": continue
        try:
            ns = {}; exec(open(c["code"]).read(), ns); fn = ns[FN[a.problem]]
            out = run(fn)
            if out.numel() != ref.numel():
                res[c["cand"]] = {"status": f"shape mismatch {out.numel()} vs {ref.numel()}", "ok": False}
            else:
                d = float((out - ref).abs().max())
                res[c["cand"]] = {"max_abs_diff": d, "ref_scale": scale, "ok": d <= 2e-2 * max(1.0, scale), "status": "ok"}
        except Exception as e:
            res[c["cand"]] = {"status": f"{type(e).__name__}: {str(e)[:200]}", "ok": False}
        if rank == 0: print(f"[check] {a.problem} {c['cand']}: {res[c['cand']]}", flush=True)
    if rank == 0:
        out = os.environ.get("OUT", "/home/ubuntu/g2/bench_1node"); os.makedirs(out, exist_ok=True)
        json.dump(res, open(f"{out}/check_{a.problem}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
