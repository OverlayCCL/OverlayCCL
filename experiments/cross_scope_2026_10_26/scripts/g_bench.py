#!/usr/bin/env python3
"""1-node microbench of one candidate at the per-problem harness shapes (bench scope of the cross-scope study).
Run under torchrun --nproc_per_node=32 --nnodes=1. Writes JSON {problem, cand, med_ms, mean_ms, all} to $OUT.
  python g_bench.py --problem P --cand NAME [--code FILE | --baseline] [--iters 30 --warmup 5]
"""
import argparse, json, os, statistics, sys, time, math
sys.path.insert(0, os.environ.get("REPO", "/home/ubuntu/OverlayCCL"))
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

BASE = {"alltoallv": "allgather_reduce_scatter", "uniform_a2a": "allgather_reduce_scatter", "ring_kv": "per_chunk_per_part_allgather",
        "dxe": "full_vocab_ag", "pp_send_recv": "per_mb", "tp_mlp": "per_mb", "fsdp_prefetch": "per_mb", "llama_block_ar": "sequential_2ar"}
FN = {"uniform_a2a": "evolved_uniform_a2a", "ring_kv": "evolved_ring_kv", "dxe": "evolved_dxe", "pp_send_recv": "evolved_pp_send_recv",
      "tp_mlp": "evolved_tp_mlp", "fsdp_prefetch": "evolved_fsdp_prefetch", "llama_block_ar": "evolved_llama_block"}

def make_args(problem, ws, rank, dev):
    """Harness shapes (training/train_<p>_7node.py) at this world size, bf16 like the harnesses."""
    bf = torch.bfloat16
    torch.manual_seed(42 + rank)
    if problem == "alltoallv":
        mc = 1024
        x = (torch.randn(ws * mc, dtype=bf) * 0.01).to(dev)
        return (x, ws, mc), f"mc={mc}"
    if problem == "uniform_a2a":
        T, TOPK, DM = 128, 4, 2048
        cap = (T * TOPK + ws - 1) // ws; chunk = cap * DM
        x = (torch.randn(ws * chunk, dtype=bf) * 0.01).to(dev)
        return (x, chunk), f"chunk={chunk}"
    if problem == "ring_kv":
        KV, HEADS, SEQ, HD = 2, 8, 16, 128
        kv = (torch.randn(KV, HEADS, SEQ, HD, dtype=bf) * 0.01).to(dev)
        chunks = [kv[i].reshape(-1) for i in range(KV)]
        return (chunks,), f"slot={HEADS*SEQ*HD}"
    if problem == "dxe":
        T, V = 256, 32256; vl = V // ws
        logits = (torch.randn(T, vl, dtype=bf) * 0.01).to(dev)
        tgt = torch.randint(0, V, (T,), dtype=torch.long).to(dev)
        return (logits, tgt, vl), f"T={T} vl={vl}"
    if problem == "pp_send_recv":
        M, B, S, DM = 4, 1, 512, 2048; half = ws // 2
        acts = [((torch.randn(B, S, DM, dtype=bf) * 0.01) if rank < half else torch.zeros(B, S, DM, dtype=bf)).to(dev) for _ in range(M)]
        return (acts, 0, half, M), f"M={M} S={S} DM={DM}"
    if problem == "tp_mlp":
        M, L, B, S, DM = 4, 1, 1, 1024, 2048
        parts = [[(torch.randn(B, S, DM, dtype=bf) * 0.01).to(dev) for _ in range(L)] for _ in range(M)]
        return (parts, M, L), f"M={M} L={L} S={S} DM={DM}"
    if problem == "fsdp_prefetch":
        M, L, DM, HID = 4, 1, 2048, 5376; sh = HID // ws
        shards = [[(torch.randn(DM, sh, dtype=bf) * 0.01).to(dev) for _ in range(L)] for _ in range(M)]
        return (shards, M, L), f"M={M} L={L} DM={DM} shard={sh}"
    if problem == "llama_block_ar":
        B, S, DM = 1, 1024, 2048
        return ((torch.randn(B, S, DM, dtype=bf) * 0.01).to(dev), (torch.randn(B, S, DM, dtype=bf) * 0.01).to(dev)), f"S={S} DM={DM}"
    raise SystemExit(f"unknown problem {problem}")

def flatten(out):
    if torch.is_tensor(out): return [out]
    res = []
    for o in out: res += flatten(o)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True); ap.add_argument("--cand", required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--code"); g.add_argument("--baseline", action="store_true")
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ.get("WORLD_SIZE", xr.world_size()))
    assert ws == xr.world_size() and ws > 1, f"world size mismatch env={ws} xr={xr.world_size()}"
    if a.problem == "alltoallv":
        # file-kind candidates: load the runtime module and time its alltoallv(x, ws, mc) helper; baseline = AG+T+RS inline
        import importlib.util
        if a.baseline:
            def fn(x, ws_, mc, *rest, **kw):
                gathered = xm.all_gather(x.unsqueeze(0), dim=0)
                transposed = gathered.view(ws_, ws_, mc).permute(1, 0, 2).contiguous().view(-1)
                return xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0 / ws_, scatter_dim=0, shard_count=ws_)
        else:
            spec = importlib.util.spec_from_file_location("a2av_rt", a.code); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            mod.init_alltoallv(); fn = lambda x, ws_, mc, *rest, **kw: mod.alltoallv(x, ws_, mc)
    elif a.baseline:
        from search.problems import get_problem
        import search.problems_modext
        code = get_problem(a.problem).builtin_templates[BASE[a.problem]]
    else:
        code = open(a.code).read()
    if a.problem != "alltoallv":
        ns = {}; exec(code, ns); fn = ns[FN[a.problem]]
    args, desc = make_args(a.problem, ws, rank, dev)
    xm.mark_step(); _ = flatten(args[0])[0].float().sum().item() if torch.is_tensor(flatten(args[0])[0]) else None
    run = lambda: fn(*args, rank, ws, 16, 2, xm, torch, num_nodes=1)
    if rank == 0: print(f"[init] {a.problem} {a.cand} ws={ws} {desc}", flush=True)
    status = "ok"; ts = []
    try:
        for _ in range(a.warmup):
            y = run(); xm.mark_step(); _ = sum(o.float().sum() for o in flatten(y)).item()
        for _ in range(a.iters):
            xm.mark_step(); t0 = time.time()
            y = run(); xm.mark_step(); _ = sum(o.float().sum() for o in flatten(y)).item()
            ts.append((time.time() - t0) * 1000.0)
    except Exception as e:
        status = f"{type(e).__name__}: {str(e)[:300]}"
        if rank == 0:
            import traceback; traceback.print_exc()
    if rank == 0:
        res = {"problem": a.problem, "cand": a.cand, "ws": ws, "desc": desc, "status": status,
               "med_ms": statistics.median(ts) if ts else None, "mean_ms": statistics.mean(ts) if ts else None, "all": ts}
        out = os.environ.get("OUT", "/home/ubuntu/g2/bench_1node"); os.makedirs(out, exist_ok=True)
        json.dump(res, open(f"{out}/{a.problem}__{a.cand}.json", "w"), indent=1)
        print(f"[bench] {a.problem} {a.cand} med={res['med_ms']} mean={res['mean_ms']} status={status}", flush=True)

if __name__ == "__main__":
    main()
