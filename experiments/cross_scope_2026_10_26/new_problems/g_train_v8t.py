#!/usr/bin/env python3
"""Training-scope runner v8: DDP-style gradient-sync placement. Every piece of the problem is its OWN parameter (a separate
device buffer, as gradients are in a real model), a small forward/backward produces its gradient, and the candidate strategy
synchronizes the gradient list in a separate graph after the backward (mark_step before the sync, as the per-microbatch
DDP reducer flush does), followed by the SGD update. Compared with g_train.py (v4/v5: collective issued inside the forward
and re-issued in the backward on views of one flat buffer) this isolates whether the per-tensor penalty measured there is a
property of gradient synchronization in a training step or of that harness's placement.
Args (after torchrun): --backend <cand> --steps N --warmup W --problem P. Writes $RESULTS_DIR/newprob_train_<problem>__<cand>.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/home/ubuntu/g2")
import torch, torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from g_newprobs import PROBLEMS, flatten

def main():
    p = argparse.ArgumentParser(); p.add_argument("--problem", required=True); p.add_argument("--backend", required=True)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--warmup", type=int, default=5); a = p.parse_args()
    if not dist.is_initialized(): dist.init_process_group("xla", init_method="xla://")
    dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); assert ws == xr.world_size()
    P = PROBLEMS[a.problem]; fn = P["cands"][a.backend]
    args = P["make"](ws, rank, dev); rest = args[1:]
    template = args[0]; tensors = flatten(template)
    params = [torch.nn.Parameter(t.detach().clone()) for t in tensors]     # one parameter per piece = separate buffers
    def rebuild(tmpl, it):
        if torch.is_tensor(tmpl): return next(it)
        return [rebuild(x, it) for x in tmpl]
    torch.manual_seed(42 + rank)
    # a per-step input scalar per piece makes every gradient data-dependent; the loss touches every element once
    xs = [(torch.randn(1) * 0.1 + 1.0).to(dev) for _ in params]
    # v8t: every 2-D piece is applied as a linear layer (x @ W.t(), 64 rows), so its gradient is produced by the matmul backward
    xin = {prm.shape: (torch.randn(64, prm.shape[1], dtype=prm.dtype) * 0.1).to(dev) for prm in params if prm.dim() == 2}
    def piece_loss(prm, x):
        if prm.dim() == 2: return (torch.matmul(xin[prm.shape], prm.t()).float() ** 2).mean() * x
        return ((prm.float() * x) ** 2).mean()
    opt = torch.optim.SGD(params, lr=1e-4)
    def step(s):
        opt.zero_grad()
        loss = sum(piece_loss(prm, x) for prm, x in zip(params, xs))
        loss.backward()
        xm.mark_step()                                   # the sync runs in its own graph after the backward (DDP reducer flush)
        grads = [prm.grad for prm in params]
        outs = flatten(fn(*((rebuild(template, iter(grads)),) + tuple(rest)), rank, ws, xm, torch))
        with torch.no_grad():
            for prm, g in zip(params, outs): prm.grad = g.to(prm.dtype).view_as(prm) / ws
        opt.step(); xm.mark_step()
        return loss
    if rank == 0: print(f"[init] v8t {a.problem} backend={a.backend} ws={ws} pieces={len(params)} elems={sum(t.numel() for t in params)} steps={a.steps}", flush=True)
    for _ in range(a.warmup): step(0)
    # effectiveness: the synced gradient must differ from the local one (collective executed)
    opt.zero_grad(); loss = sum(piece_loss(prm, x) for prm, x in zip(params, xs)); loss.backward(); xm.mark_step()
    g0 = params[0].grad.detach().clone(); outs = flatten(fn(*((rebuild(template, iter([prm.grad for prm in params])),) + tuple(rest)), rank, ws, xm, torch))
    effective = bool((outs[0].to(g0.dtype).view_as(g0) != g0).any().item()); opt.zero_grad(); xm.mark_step()
    if rank == 0: print(f"[check] {a.problem} backend={a.backend} collective_effective={effective}", flush=True)
    xm.rendezvous("pre_measure")
    times = []
    for s in range(a.steps):
        xm.mark_step(); t0 = time.time(); loss = step(s); _ = loss.item(); times.append((time.time() - t0) * 1e3)
    if rank == 0:
        times_s = sorted(times[a.steps // 4:]); med = times_s[len(times_s) // 2]
        print(f"[train] {a.problem} {a.backend}: median {med:.2f} ms (n={len(times_s)})", flush=True)
        out = os.environ.get("RESULTS_DIR", "/home/ubuntu/g2/newprobs/train"); os.makedirs(out, exist_ok=True)
        json.dump({"problem": a.problem, "cand": a.backend, "ws": ws, "measured_ms": med, "step_times_ms": times, "collective_effective": effective, "runner": "v8t_ddp_placement_linear_layout"},
                  open(f"{out}/newprob_train_{a.problem}__{a.backend}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
