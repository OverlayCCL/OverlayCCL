#!/usr/bin/env python3
"""Per-problem TRAINING harness v11 for FORWARD all-gather problems (FSDP parameter prefetch), e2e-like placement:
  * every per-rank shard of the problem is an nn.Parameter; each step the candidate strategy gathers the full weights
    (the collective under test) INSIDE the forward, the gathered weights feed a matmul on (S=1024, DM=2048) bf16 tokens,
    and the loss's backward runs through a custom autograd Function whose backward is the true one: the gradient of a
    gathered weight is REDUCE-SCATTERED back to the shard (xm.all_gather has no usable backward on Neuron);
  * N_MB microbatches accumulate shard gradients (one graph per microbatch), then an SGD update on the shards.
The candidate functions are the problem's g_newprobs candidates: fn(shards, rank, ws, xm, torch) -> gathered full weights.
Args (after torchrun): --backend <cand> --steps N --warmup W --problem P. Writes $RESULTS_DIR/newprob_train_<problem>__<cand>.json."""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/ubuntu/g2")
import torch, torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_newprobs import PROBLEMS, flatten

DM = 2048
S = int(os.environ.get("S", "1024"))
N_MB = int(os.environ.get("N_MB", "4"))
LR = 1e-4


class _Gather(torch.autograd.Function):
    """forward: the candidate's gathers on detached shards; backward: reduce-scatter of each full-weight gradient to the shard"""
    @staticmethod
    def forward(ctx, fn, rank, ws, *shards):
        ctx.rank, ctx.ws = rank, ws; ctx.shapes = [s.shape for s in shards]
        outs = flatten(fn([s.detach() for s in shards], rank, ws, xm, torch))
        ctx.dims = []
        for s, o in zip(shards, outs):   # which dim was gathered: the one that grew by ws
            d = next(i for i in range(s.dim()) if o.shape[i] == s.shape[i] * ws); ctx.dims.append(d)
        return tuple(outs)
    @staticmethod
    def backward(ctx, *gouts):
        gins = []
        for g, d, sh in zip(gouts, ctx.dims, ctx.shapes):
            gc = g.contiguous()
            if d != 0: gc = gc.transpose(0, d).contiguous()
            r = xm.reduce_scatter(xm.REDUCE_SUM, gc, 1.0, 0, ctx.ws)
            if d != 0: r = r.transpose(0, d).contiguous()
            gins.append(r.reshape(sh))
        return (None, None, None) + tuple(gins)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--problem", required=True); p.add_argument("--backend", required=True)
    p.add_argument("--steps", type=int, default=100); p.add_argument("--warmup", type=int, default=3); a = p.parse_args()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); assert ws == xr.world_size()
    P = PROBLEMS[a.problem]; fn = P["cands"][a.backend]
    args = P["make"](ws, rank, dev); tensors = flatten(args[0])
    torch.manual_seed(42 + rank)
    shards = [torch.nn.Parameter(t.detach().clone()) for t in tensors]
    xs = [(torch.randn(S, DM, dtype=torch.bfloat16) * 0.1).to(dev) for _ in range(N_MB)]
    opt = torch.optim.SGD(shards, lr=LR)
    def fit(x, b):
        if b == DM: return x
        if b < DM: return x[:, :b]
        return x.repeat(1, math.ceil(b / DM))[:, :b]
    def forward(x):
        fulls = _Gather.apply(fn, rank, ws, *shards)
        for w in fulls:                                   # gathered weight feeds a matmul, nn.Linear-style
            o, i = w.shape[0], w.shape[1]
            y = torch.matmul(fit(x, i).to(w.dtype), w.t()).to(torch.bfloat16)
            x = x + 0.01 * (y[:, :DM] if o >= DM else y.repeat(1, math.ceil(DM / o))[:, :DM])
        return (x.float() ** 2).mean()
    def step():
        opt.zero_grad()
        for m in range(N_MB):
            (forward(xs[m]) / N_MB).backward(); xm.mark_step()
        opt.step(); xm.mark_step()
    if rank == 0: print(f"[init] v11ag {a.problem} backend={a.backend} ws={ws} shards={len(shards)} S={S} N_MB={N_MB} steps={a.steps}", flush=True)
    for _ in range(a.warmup): step()
    # effectiveness: the gathered weight must not equal a tiling of the local shard (the collective executed)
    fulls = _Gather.apply(fn, rank, ws, *shards); f0 = fulls[0].detach(); s0 = shards[0].detach()
    d0 = next(i for i in range(s0.dim()) if f0.shape[i] == s0.shape[i] * ws)
    tiled = torch.cat([s0] * ws, dim=d0); effective = bool((f0 != tiled).any().item()); xm.mark_step()
    if rank == 0: print(f"[check] {a.problem} backend={a.backend} collective_effective={effective}", flush=True)
    xm.rendezvous("pre_measure")
    times = []
    for s in range(a.steps):
        xm.mark_step(); t0 = time.time(); step(); _ = shards[0].data.float().sum().item(); times.append((time.time() - t0) * 1e3)
    if rank == 0:
        ts = sorted(times[a.steps // 4:]); med = ts[len(ts) // 2]
        print(f"[train] {a.problem} {a.backend}: median {med:.2f} ms (n={len(ts)})", flush=True)
        out = os.environ.get("RESULTS_DIR", "/home/ubuntu/g2/newprobs/train"); os.makedirs(out, exist_ok=True)
        json.dump({"problem": a.problem, "cand": a.backend, "ws": ws, "measured_ms": med, "step_times_ms": times, "collective_effective": effective,
                   "runner": "v11ag_forward_gather_true_backward", "S": S, "N_MB": N_MB}, open(f"{out}/newprob_train_{a.problem}__{a.backend}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
