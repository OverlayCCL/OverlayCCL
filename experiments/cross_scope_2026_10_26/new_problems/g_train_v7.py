#!/usr/bin/env python3
"""Training-scope micro-runner for the new problems (run under torchrun on both nodes via run_ap7.sh):
a small model step surrounds the collective (matmul producing the communicated tensors, loss, backward through an
autograd Function whose backward re-issues the same collective on the gradient, SGD update, one mark_step per step),
mirroring training/train_<p>_7node.py. Args (after torchrun): --backend <cand> --steps N --warmup W --problem P.
Writes $RESULTS_DIR/newprob_train_<problem>__<cand>.json with step_times_ms (rank 0)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/ubuntu/g2")
import torch, torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_newprobs import PROBLEMS, flatten

class _Coll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat_in, fn, unflat, rank, ws):
        ctx.fn, ctx.unflat, ctx.rank, ctx.ws = fn, unflat, rank, ws; ctx.n_in = flat_in.numel()
        # v7: G_TRAIN_FUSED=1 keeps the collective inside the step graph (no graph cut around it), like a real training step
        if os.environ.get("G_TRAIN_FUSED") != "1": xm.mark_step()
        outs = flatten(fn(*unflat(flat_in), rank, ws, xm, torch))
        ctx.shapes = [o.shape for o in outs]
        if os.environ.get("G_TRAIN_FUSED") != "1": xm.mark_step()
        out = torch.cat([o.reshape(-1) for o in outs])
        # v4 effectiveness check: a collective that silently did not execute (e.g. NCCL comm init failure for replica
        # groups at 64 ranks) returns its input or zeros; record whether the output differs from the input.
        _Coll.last_effective = bool((out[:min(out.numel(), ctx.n_in)] != flat_in[:min(out.numel(), ctx.n_in)]).any().item()) if out.numel() and flat_in.numel() else None
        return out
    @staticmethod
    def backward(ctx, g):
        # re-issue the same strategy on a gradient-sized buffer so each step carries the collective in forward and backward
        if os.environ.get("G_TRAIN_FUSED") != "1": xm.mark_step()
        outs = flatten(ctx.fn(*ctx.unflat(g.new_zeros(ctx.n_in) + g.float().mean().to(g.dtype)), ctx.rank, ctx.ws, xm, torch))
        gin = torch.cat([o.reshape(-1) for o in outs])
        if os.environ.get("G_TRAIN_FUSED") != "1": xm.mark_step()
        return gin[:ctx.n_in] if gin.numel() >= ctx.n_in else torch.cat([gin, gin.new_zeros(ctx.n_in - gin.numel())]), None, None, None, None

def main():
    p = argparse.ArgumentParser(); p.add_argument("--problem", required=True); p.add_argument("--backend", required=True)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--warmup", type=int, default=5); p.add_argument("--fused", action="store_true"); a = p.parse_args()
    if a.fused: os.environ["G_TRAIN_FUSED"] = "1"   # the launcher forwards argv, not the environment
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); assert ws == xr.world_size()
    P = PROBLEMS[a.problem]; fn = P["cands"][a.backend]
    args = P["make"](ws, rank, dev)
    tensors = flatten(args[0]); shapes = [t.shape for t in tensors]; numels = [t.numel() for t in tensors]
    rest = args[1:]
    def rebuild(template, it):
        # rebuild the (possibly nested) list structure of args[0] from an iterator of tensors
        if torch.is_tensor(template): return next(it)
        return [rebuild(x, it) for x in template]
    def unflat(flat):
        outs = []; off = 0
        for s, n in zip(shapes, numels): outs.append(flat[off:off + n].view(s)); off += n
        return (rebuild(args[0], iter(outs)),) + tuple(rest)
    torch.manual_seed(42 + rank)
    N = sum(numels)
    # the communicated tensors are produced by a matmul of a parameter so the collective sits inside a real fwd/bwd graph
    K = 256
    while K > 1 and N % K: K //= 2   # largest power of two <= 256 dividing N (K=1 falls back to a (N,1)x(1,1) matmul)
    w = torch.nn.Parameter((torch.randn(K, K, dtype=torch.bfloat16) * 0.01).to(dev))
    x_in = (torch.randn(N // K, K, dtype=torch.bfloat16)).to(dev)
    opt = torch.optim.SGD([w], lr=1e-4)
    def step(s):
        opt.zero_grad()
        flat = torch.matmul(x_in, w).reshape(-1)
        out = _Coll.apply(flat, fn, unflat, rank, ws)
        loss = out.float().pow(2).mean()
        loss.backward(); opt.step(); xm.mark_step()
        return loss
    if rank == 0: print(f"[init] {a.problem} backend={a.backend} ws={ws} N={N} steps={a.steps}", flush=True)
    for _ in range(a.warmup): step(0)
    effective = getattr(_Coll, "last_effective", None)
    if rank == 0: print(f"[check] {a.problem} backend={a.backend} collective_effective={effective}", flush=True)
    xm.rendezvous("pre_measure")
    times = []
    for s in range(a.steps):
        t0 = time.time(); loss = step(s); xm.wait_device_ops(); times.append((time.time() - t0) * 1e3)
    if rank == 0:
        import statistics
        out = os.environ.get("RESULTS_DIR", "/tmp"); os.makedirs(out, exist_ok=True)
        med = statistics.median(times[len(times) // 2:int(len(times) * 0.93)])
        json.dump(dict(problem=a.problem, cand=a.backend, ws=ws, steps=a.steps, measured_ms=med, collective_effective=effective, step_times_ms=[round(t, 2) for t in times]),
                  open(f"{out}/newprob_train_{a.problem}__{a.backend}.json", "w"), indent=1)
        print(f"[train] {a.problem} {a.backend} ws={ws} median_ms={med:.2f}", flush=True)

if __name__ == "__main__":
    main()
