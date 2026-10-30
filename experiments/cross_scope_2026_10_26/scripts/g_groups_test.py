#!/usr/bin/env python3
"""Probe whether xm.all_gather / xm.all_reduce with replica `groups` (intra-node, inter-node) run at all on this
stack at >1 node, with pin_layout True/False. Run under torchrun on both nodes. Prints one line per variant."""
import os, sys, time, traceback, json
RES = {}
import torch, torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
dist.init_process_group("xla", init_method="xla://")
dev = xm.xla_device(); rank = xr.global_ordinal(); ws = int(os.environ["WORLD_SIZE"]); nn = ws // 32; rpn = 32
intra = [[n * rpn + i for i in range(rpn)] for n in range(nn)]
inter = [[n * rpn + i for n in range(nn)] for i in range(rpn)]
x = (torch.randn(8192, dtype=torch.bfloat16) * 0.01 + rank).to(dev)
def t(label, fn):
    try:
        for _ in range(3): y = fn(); xm.mark_step(); s = float(y.float().sum().item())
        ts = []
        for _ in range(10):
            xm.mark_step(); t0 = time.time(); y = fn(); xm.mark_step(); _ = y.float().sum().item(); ts.append((time.time() - t0) * 1e3)
        RES[label] = {"ok": True, "shape": list(y.shape), "med_ms": sorted(ts)[5]}
        if rank % 32 == 0: print(f"[groups] rank{rank} {label}: ok shape={tuple(y.shape)} med={sorted(ts)[5]:.3f}ms", file=sys.stderr, flush=True)
    except Exception as e:
        RES[label] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        if rank % 32 == 0: print(f"[groups] rank{rank} {label}: FAIL {type(e).__name__}: {str(e)[:160]}", file=sys.stderr, flush=True)
for pin in (True, False):
    t(f"ag_intra pin={pin}", lambda: xm.all_gather(x.unsqueeze(0), dim=0, groups=intra, pin_layout=pin))
    t(f"ag_inter pin={pin}", lambda: xm.all_gather(x.unsqueeze(0), dim=0, groups=inter, pin_layout=pin))
    t(f"ar_intra pin={pin}", lambda: xm.all_reduce(xm.REDUCE_SUM, x, groups=intra, pin_layout=pin))
    t(f"ar_inter pin={pin}", lambda: xm.all_reduce(xm.REDUCE_SUM, x, groups=inter, pin_layout=pin))
    t(f"ag_hier pin={pin}", lambda: xm.all_gather(xm.all_gather(x.unsqueeze(0), dim=0, groups=intra, pin_layout=pin).unsqueeze(0), dim=0, groups=inter, pin_layout=pin))
t("ag_flat", lambda: xm.all_gather(x.unsqueeze(0), dim=0))
t("ar_flat", lambda: xm.all_reduce(xm.REDUCE_SUM, x))

if rank == 0:
    os.makedirs("/home/ubuntu/g2/groups_test", exist_ok=True); json.dump(RES, open("/home/ubuntu/g2/groups_test/results.json", "w"), indent=1)
print(f"[groups] rank{rank} done {len(RES)} variants", file=sys.stderr, flush=True)
