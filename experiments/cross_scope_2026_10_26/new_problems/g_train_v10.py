#!/usr/bin/env python3
"""Per-problem TRAINING harness v10 ("mirror" harness, 2026-10-29): the problem's gradient set is produced the way the Llama e2e
shell produces it, and synchronized the way the shell synchronizes it.

  * every piece of the problem's set is an nn.Parameter of a small model that runs on (S=1024, DM=2048) bf16 tokens:
      2-D pieces act as nn.Linear-style weights (x @ W.t(), so the gradient carries the matmul backward's layout),
      DM-sized 1-D pieces act as RMSNorm gains, other 1-D pieces enter through the mean hidden state (dense gradients);
  * N_MB microbatches run forward+backward and ACCUMULATE into separate .grad buffers (textbook accumulate-then-sync);
  * one gradient synchronization per step with the candidate strategy, in its own graph after the backward
    (xm.mark_step() before it, like a DDP reducer flush), followed by the SGD update in the same graph.
Compared with g_train.py (collective issued inside the forward on views of one flat buffer and re-issued in the backward)
this is the placement the e2e uses; g_train.py's numbers are to be read as a microbench of the collective itself.
Args (after torchrun): --backend <cand> --steps N --warmup W --problem P [N_MB=4 S=1024]. Writes
$RESULTS_DIR/newprob_train_<problem>__<cand>.json with measured_ms (median step) and collective_effective."""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/ubuntu/g2")
import torch, torch.distributed as dist
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_newprobs import PROBLEMS, flatten

DM = 2048
S = int(os.environ.get("S", "1024"))
N_MB = int(os.environ.get("N_MB", "4"))
LR = 1e-4


def main():
    p = argparse.ArgumentParser(); p.add_argument("--problem", required=True); p.add_argument("--backend", required=True)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--warmup", type=int, default=5); a = p.parse_args()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); assert ws == xr.world_size()
    P = PROBLEMS[a.problem]; fn = P["cands"][a.backend]
    args = P["make"](ws, rank, dev); rest = args[1:]; template = args[0]; tensors = flatten(template)
    torch.manual_seed(1234)   # identical parameters on every rank (replicated set)
    params = [torch.nn.Parameter((torch.randn(*t.shape, dtype=t.dtype) * (1.0 if t.dim() == 1 and t.numel() == DM else 0.01)).to(dev)) for t in tensors]
    def rebuild(tmpl, it):
        if torch.is_tensor(tmpl): return next(it)
        return [rebuild(x, it) for x in tmpl]
    torch.manual_seed(42 + rank)
    xs = [(torch.randn(S, DM, dtype=torch.bfloat16) * 0.1).to(dev) for _ in range(N_MB)]   # one token block per microbatch

    def rmsnorm(x, g):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()).to(torch.bfloat16)
    def fit_width(x, b):   # (S, DM) -> (S, b) by slicing or tiling
        if b == DM: return x
        if b < DM: return x[:, :b]
        return x.repeat(1, math.ceil(b / DM))[:, :b]
    def model(x):
        """one pass over the set: 2-D pieces as nn.Linear weights (gradient layout from the matmul backward), DM-vectors as
        RMSNorm gains, other vectors through an elementwise term (vectors have one layout; the matmul coupling of a 64-vector
        set compiled to a million-instruction graph)"""
        xs_ = x.float().mean()
        extra = 0.0
        for w in params:
            if w.dim() == 2:
                o, i = w.shape
                h = fit_width(x, i).to(w.dtype)
                y = torch.matmul(h, w.t()).to(torch.bfloat16)      # nn.Linear convention: the gradient of w comes from the matmul backward
                x = x + 0.01 * (y[:, :DM] if o >= DM else y.repeat(1, math.ceil(DM / o))[:, :DM])
            elif w.numel() == DM:
                x = rmsnorm(x, w)
            else:
                extra = extra + ((w.float() * xs_) ** 2).mean()
        return x, extra
    def sync_and_update():
        xm.mark_step()                                       # the sync runs in its own graph after the backward (DDP reducer flush)
        grads = [prm.grad if prm.grad is not None else torch.zeros_like(prm) for prm in params]
        outs = flatten(fn(*((rebuild(template, iter(grads)),) + tuple(rest)), rank, ws, xm, torch))
        with torch.no_grad():
            for prm, g in zip(params, outs): prm.data = prm.data - LR * (g.to(prm.dtype).view_as(prm) / ws)
            for prm in params: prm.grad = None
        xm.mark_step()
        return grads[0], outs[0]

    def step():
        for m in range(N_MB):
            xo, extra = model(xs[m]); loss = ((xo.float() ** 2).mean() + extra) / N_MB
            loss.backward()
            xm.mark_step()                               # one graph per microbatch (as the e2e shell's per-microbatch graphs); gradients accumulate across graphs
        return sync_and_update()

    if rank == 0: print(f"[init] v10 {a.problem} backend={a.backend} ws={ws} pieces={len(params)} elems={sum(t.numel() for t in params)} S={S} N_MB={N_MB} steps={a.steps}", flush=True)
    for _ in range(a.warmup): step()
    # effectiveness: the synced gradient differs from the local one (the collective executed)
    for m in range(N_MB):
        xo, extra = model(xs[m]); (((xo.float() ** 2).mean() + extra) / N_MB).backward(); xm.mark_step()
    g0 = params[0].grad.detach().clone()
    grads = [prm.grad for prm in params]; outs = flatten(fn(*((rebuild(template, iter(grads)),) + tuple(rest)), rank, ws, xm, torch))
    effective = bool((outs[0].to(g0.dtype).view_as(g0) != g0).any().item())
    for prm in params: prm.grad = None
    xm.mark_step()
    if rank == 0: print(f"[check] {a.problem} backend={a.backend} collective_effective={effective}", flush=True)
    xm.rendezvous("pre_measure")
    times = []
    for s in range(a.steps):
        xm.mark_step(); t0 = time.time(); g, o = step(); _ = params[0].data.float().sum().item(); times.append((time.time() - t0) * 1e3)
    if rank == 0:
        ts = sorted(times[a.steps // 4:]); med = ts[len(ts) // 2]
        print(f"[train] {a.problem} {a.backend}: median {med:.2f} ms (n={len(ts)})", flush=True)
        out = os.environ.get("RESULTS_DIR", "/home/ubuntu/g2/newprobs/train"); os.makedirs(out, exist_ok=True)
        json.dump({"problem": a.problem, "cand": a.backend, "ws": ws, "measured_ms": med, "step_times_ms": times, "collective_effective": effective,
                   "runner": "v10_mirror", "S": S, "N_MB": N_MB}, open(f"{out}/newprob_train_{a.problem}__{a.backend}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
