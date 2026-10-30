"""New collective problems (outside the paper's eight) with cross-scope inversion potential, each with a production
baseline and alternative strategies. Shared by g_bench2.py (1-node bench at harness shapes), g_check2.py (on-device
correctness vs the baseline) and g_train.py (2-node training-scope micro-runner).

Every strategy has the signature fn(args..., rank, ws, xm, torch) and returns a tensor or list of tensors; `args` come
from make_args(problem, ws, rank, dev). Replica groups are derived from the world size (32 ranks per node).
"""
import torch
RPN = 32  # ranks per trn1.32xlarge node

def groups_same_slot(ws, tp):
    """DP groups for a TP degree `tp`: ranks with the same (rank % tp). On one node (ws=32) with tp=16 the groups have
    2 members inside the node; at ws=64 they have 4 members spanning both nodes."""
    return [[r for r in range(ws) if r % tp == s] for s in range(tp)]

# ---------------------------------------------------------------- P1: DP gradient all-reduce across TP groups
L1 = 8; G1 = (2048, 512)
def tp_of(ws): return ws // 4   # DP groups of 4 ranks: inside one node at ws=32 (bench), 2 ranks per node at ws=64 (training)
def p1_make(ws, rank, dev):
    return ([(torch.randn(*G1, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(L1)],)
def p1_per_tensor_grouped_ar(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws))
    return [xm.all_reduce(xm.REDUCE_SUM, t, groups=g) for t in grads]
def p1_flat_grouped_ar(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws))
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def p1_bucketed_grouped_ar(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); outs = []
    for b in range(0, len(grads), 2):
        chunk = grads[b:b + 2]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g)
        off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def p1_grouped_rs_ag(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); gs = 4
    flat = torch.cat([t.reshape(-1) for t in grads]); n = flat.numel(); pad = (-n) % gs
    if pad: flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=gs, groups=g)
    r = xm.all_gather(rs, dim=0, groups=g, pin_layout=False)[:n]
    outs = []; off = 0
    for t in grads: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    return outs
def p1_masked_world_ar(grads, rank, ws, xm, torch):
    """no replica groups: put the flat gradient in slot (rank % TP) of a (TP, N) zero buffer, all-reduce over the whole
    world (ranks with the same slot are exactly the DP group), read the slot back"""
    tp = tp_of(ws)
    flat = torch.cat([t.reshape(-1) for t in grads]); buf = torch.zeros(tp, flat.numel(), dtype=flat.dtype, device=flat.device)
    buf[rank % tp] = flat; r = xm.all_reduce(xm.REDUCE_SUM, buf)[rank % tp]
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs

# ---------------------------------------------------------------- P2: ZeRO-2 gradient reduce-scatter (whole world)
L2 = 8; G2 = (2048, 1024)
def p2_make(ws, rank, dev):
    return ([(torch.randn(*G2, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(L2)],)
def p2_per_tensor_rs(grads, rank, ws, xm, torch):
    return [xm.reduce_scatter(xm.REDUCE_SUM, t.reshape(-1), scale=1.0, scatter_dim=0, shard_count=ws) for t in grads]
def p2_flat_rs(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); per = [t.numel() // ws for t in grads]
    # scatter each tensor's shard contiguously: lay the flat buffer out as (ws, sum(per)) so shard r holds every tensor's r-th slice
    view = torch.cat([t.reshape(ws, -1) for t in grads], dim=1).reshape(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=ws)
    outs = []; off = 0
    for n in per: outs.append(r[off:off + n]); off += n
    return outs
def p2_bucketed_rs(grads, rank, ws, xm, torch):
    outs = []
    for b in range(0, len(grads), 2):
        chunk = grads[b:b + 2]; per = [t.numel() // ws for t in chunk]
        view = torch.cat([t.reshape(ws, -1) for t in chunk], dim=1).reshape(-1)
        r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=ws)
        off = 0
        for n in per: outs.append(r[off:off + n]); off += n
    return outs
def p2_ar_then_slice(grads, rank, ws, xm, torch):
    outs = []
    for t in grads:
        r = xm.all_reduce(xm.REDUCE_SUM, t.reshape(-1)); n = t.numel() // ws; outs.append(r[rank * n:(rank + 1) * n])
    return outs
def p2_flat_ar_then_slice(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: n = t.numel() // ws; outs.append(r[off + rank * n: off + (rank + 1) * n]); off += t.numel()
    return outs

# ---------------------------------------------------------------- P3: ZeRO-1 parameter all-gather after the optimizer step
L3 = 8; N3_PER_RANK = 2048 * 1024 // 32   # each rank's shard of each of the 8 flat parameter tensors (fixed per-rank bytes)
def p3_make(ws, rank, dev):
    return ([(torch.randn(N3_PER_RANK, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(L3)],)
def p3_per_tensor_ag(shards, rank, ws, xm, torch):
    return [xm.all_gather(s, dim=0) for s in shards]
def p3_flat_ag(shards, rank, ws, xm, torch):
    flat = torch.cat(shards); g = xm.all_gather(flat.unsqueeze(0), dim=0)  # (ws, L*n)
    n = shards[0].numel()
    return [g[:, i * n:(i + 1) * n].reshape(-1) for i in range(len(shards))]
def p3_bucketed_ag(shards, rank, ws, xm, torch):
    outs = []; n = shards[0].numel()
    for b in range(0, len(shards), 2):
        g = xm.all_gather(torch.cat(shards[b:b + 2]).unsqueeze(0), dim=0)
        outs += [g[:, i * n:(i + 1) * n].reshape(-1) for i in range(2)]
    return outs
def p3_per_tensor_ag_native(shards, rank, ws, xm, torch):
    return [xm.all_gather(s, dim=0, pin_layout=False) for s in shards]
def p3_flat_ag_native(shards, rank, ws, xm, torch):
    flat = torch.cat(shards); g = xm.all_gather(flat.unsqueeze(0), dim=0, pin_layout=False)
    n = shards[0].numel()
    return [g[:, i * n:(i + 1) * n].reshape(-1) for i in range(len(shards))]

# ---------------------------------------------------------------- P4: Ulysses sequence-parallel attention all-to-all
H4, D4 = 128, 128; S4_LOCAL = 32   # H must be a multiple of the world size (64 at training scope)   # input (S/ws, H, D) per rank -> output (S, H/ws, D): every rank's head slice for every sequence block
def p4_make(ws, rank, dev):
    return ((torch.randn(S4_LOCAL, H4, D4, dtype=torch.bfloat16) * 0.01).to(dev),)
def _p4_chunks(x, ws):
    # chunk for destination d = heads [d*H/ws, (d+1)*H/ws) of my sequence block: (ws, S_local, H/ws, D)
    return x.view(S4_LOCAL, ws, H4 // ws, D4).permute(1, 0, 2, 3).contiguous()
def _p4_assemble(recv, ws):
    # recv: (ws_src, S_local, H/ws, D) -> (S, H/ws, D) with source-major sequence order
    return recv.reshape(ws * S4_LOCAL, H4 // ws, D4)
def p4_ag_transpose_rs(x, rank, ws, xm, torch):
    """baseline: all-gather everything, transpose to destination-major, reduce-scatter (the paper's A2A baseline shape)"""
    c = _p4_chunks(x, ws); n = c.numel() // ws
    g = xm.all_gather(c.reshape(-1).unsqueeze(0), dim=0).view(ws, ws, n).permute(1, 0, 2).contiguous().view(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, g, scale=1.0 / ws, scatter_dim=0, shard_count=ws)
    return _p4_assemble(r.view(ws, S4_LOCAL, H4 // ws, D4), ws)
def p4_ag_slice(x, rank, ws, xm, torch):
    c = _p4_chunks(x, ws); n = c.numel() // ws
    g = xm.all_gather(c.reshape(-1).unsqueeze(0), dim=0).view(ws, ws, n)
    return _p4_assemble(g[:, rank, :].contiguous().view(ws, S4_LOCAL, H4 // ws, D4), ws)
def p4_ag_slice_native(x, rank, ws, xm, torch):
    c = _p4_chunks(x, ws); n = c.numel() // ws
    g = xm.all_gather(c.reshape(-1).unsqueeze(0), dim=0, pin_layout=False).view(ws, ws, n)
    return _p4_assemble(g[:, rank, :].contiguous().view(ws, S4_LOCAL, H4 // ws, D4), ws)
def p4_masked_rs(x, rank, ws, xm, torch):
    c = _p4_chunks(x, ws); n = c.numel() // ws
    buf = torch.zeros(ws, ws, n, dtype=x.dtype, device=x.device); buf[:, rank, :] = c.view(ws, n)
    r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=ws)
    return _p4_assemble(r.view(ws, S4_LOCAL, H4 // ws, D4), ws)
def p4_ag_full_then_transpose(x, rank, ws, xm, torch):
    """post-processing placement: gather the raw (S_local, H, D) blocks first, transpose/slice afterwards"""
    g = xm.all_gather(x.unsqueeze(0), dim=0)   # (ws_src, S_local, H, D)
    return g[:, :, rank * (H4 // ws):(rank + 1) * (H4 // ws), :].reshape(ws * S4_LOCAL, H4 // ws, D4)
def p4_ag_full_then_transpose_native(x, rank, ws, xm, torch):
    g = xm.all_gather(x.unsqueeze(0), dim=0, pin_layout=False)
    return g[:, :, rank * (H4 // ws):(rank + 1) * (H4 // ws), :].reshape(ws * S4_LOCAL, H4 // ws, D4)

PROBLEMS = {
    "dp_grad_ar_groups": dict(make=p1_make, baseline="per_tensor_grouped_ar", cands={
        "per_tensor_grouped_ar": p1_per_tensor_grouped_ar, "flat_grouped_ar": p1_flat_grouped_ar, "bucketed_grouped_ar": p1_bucketed_grouped_ar,
        "grouped_rs_ag": p1_grouped_rs_ag, "masked_world_ar": p1_masked_world_ar}),
    "zero2_grad_rs": dict(make=p2_make, baseline="per_tensor_rs", cands={
        "per_tensor_rs": p2_per_tensor_rs, "flat_rs": p2_flat_rs, "bucketed_rs": p2_bucketed_rs,
        "ar_then_slice": p2_ar_then_slice, "flat_ar_then_slice": p2_flat_ar_then_slice}),
    "zero1_param_ag": dict(make=p3_make, baseline="per_tensor_ag", cands={
        "per_tensor_ag": p3_per_tensor_ag, "flat_ag": p3_flat_ag, "bucketed_ag": p3_bucketed_ag,
        "per_tensor_ag_native": p3_per_tensor_ag_native, "flat_ag_native": p3_flat_ag_native}),
    "ulysses_seq_a2a": dict(make=p4_make, baseline="ag_transpose_rs", cands={
        "ag_transpose_rs": p4_ag_transpose_rs, "ag_slice": p4_ag_slice, "ag_slice_native": p4_ag_slice_native, "masked_rs": p4_masked_rs,
        "ag_full_then_transpose": p4_ag_full_then_transpose, "ag_full_then_transpose_native": p4_ag_full_then_transpose_native}),
}

# ---------------------------------------------------------------- P5: expert-parallel all-to-all inside DP-slot groups (EP < world)
# EP group = ranks with the same (rank % tp_of(ws)): 4 ranks inside one node at ws=32 (bench), 2 ranks per node at ws=64
C5 = 8192   # elements per (src, dst) pair
def p5_make(ws, rank, dev):
    return ((torch.randn(4 * C5, dtype=torch.bfloat16) * 0.01).to(dev),)
def _p5_group(rank, ws):
    g = groups_same_slot(ws, tp_of(ws)); mine = [grp for grp in g if rank in grp][0]; return g, mine, mine.index(rank)
def p5_grouped_ag_slice(x, rank, ws, xm, torch):
    g, mine, i = _p5_group(rank, ws)
    gat = xm.all_gather(x.unsqueeze(0), dim=0, groups=g).view(4, 4, C5)   # (src_in_group, dst_in_group, C)
    return gat[:, i, :].reshape(-1)
def p5_grouped_ag_slice_native(x, rank, ws, xm, torch):
    g, mine, i = _p5_group(rank, ws)
    gat = xm.all_gather(x.unsqueeze(0), dim=0, groups=g, pin_layout=False).view(4, 4, C5)
    return gat[:, i, :].reshape(-1)
def p5_grouped_masked_rs(x, rank, ws, xm, torch):
    g, mine, i = _p5_group(rank, ws)
    buf = torch.zeros(4, 4, C5, dtype=x.dtype, device=x.device); buf[:, i, :] = x.view(4, C5)
    return xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=4, groups=g).reshape(-1)
def p5_world_masked_rs(x, rank, ws, xm, torch):
    """no replica groups: a world-wide masked reduce-scatter whose destination axis is the full rank list"""
    g, mine, i = _p5_group(rank, ws)
    buf = torch.zeros(ws, 4, C5, dtype=x.dtype, device=x.device)
    for j, dst in enumerate(mine): buf[dst, i, :] = x.view(4, C5)[j]
    return xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=ws).reshape(-1)
def p5_world_ag_slice(x, rank, ws, xm, torch):
    """no replica groups: gather everyone's buffer with the default lowering, pick the group's rows"""
    g, mine, i = _p5_group(rank, ws)
    gat = xm.all_gather(x.unsqueeze(0), dim=0).view(ws, 4, C5)
    return torch.stack([gat[src, i, :] for src in mine]).reshape(-1)

# ---------------------------------------------------------------- P6: DP gradient all-reduce of a large replicated model (bucketing)
NT6 = 48; T6 = (4096, 2048)   # 48 tensors x 16 MB bf16 = 768 MB of replicated gradients per rank
def p6_make(ws, rank, dev):
    return ([(torch.randn(*T6, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(NT6)],)
def p6_per_tensor_ar(grads, rank, ws, xm, torch):
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def p6_flat_ar(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def _p6_bucketed(grads, xm, torch, per_bucket):
    outs = []
    for b in range(0, len(grads), per_bucket):
        chunk = grads[b:b + per_bucket]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
        off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def p6_bucket_32mb(grads, rank, ws, xm, torch): return _p6_bucketed(grads, xm, torch, 2)
def p6_bucket_128mb(grads, rank, ws, xm, torch): return _p6_bucketed(grads, xm, torch, 8)
def p6_rs_ag_flat(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); n = flat.numel(); pad = (-n) % ws
    if pad: flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=ws)
    r = xm.all_gather(rs, dim=0, pin_layout=False)[:n]
    outs = []; off = 0
    for t in grads: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    return outs

PROBLEMS.update({
    "ep_a2a_grouped": dict(make=p5_make, baseline="grouped_ag_slice", cands={
        "grouped_ag_slice": p5_grouped_ag_slice, "grouped_ag_slice_native": p5_grouped_ag_slice_native, "grouped_masked_rs": p5_grouped_masked_rs,
        "world_masked_rs": p5_world_masked_rs, "world_ag_slice": p5_world_ag_slice}),
    "dp_grad_ar_buckets": dict(make=p6_make, baseline="bucket_32mb", cands={
        "bucket_32mb": p6_bucket_32mb, "per_tensor_ar": p6_per_tensor_ar, "flat_ar": p6_flat_ar, "bucket_128mb": p6_bucket_128mb, "rs_ag_flat": p6_rs_ag_flat}),
})

# ---------------------------------------------------------------- P7: heterogeneous gradient all-reduce: pad-and-stack bundling vs exact
# 16 tensors whose sizes vary 1x..8x (total 288 MB bf16); padding every tensor to the largest one inflates the bytes 1.8x
S7 = [(4096, 1024 * k) for k in (1, 2, 1, 4, 1, 8, 2, 1, 4, 1, 2, 8, 1, 1, 4, 2)]
def p7_make(ws, rank, dev):
    return ([(torch.randn(*s, dtype=torch.bfloat16) * 0.01).to(dev) for s in S7],)
def p7_per_tensor_ar(grads, rank, ws, xm, torch):
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def p7_flat_cat_ar(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def p7_pad_stack_ar(grads, rank, ws, xm, torch):
    """one dispatch without a cat: pad every tensor to the largest and stack (1.8x the bytes)"""
    m = max(t.numel() for t in grads)
    st = torch.stack([torch.nn.functional.pad(t.reshape(-1), (0, m - t.numel())) for t in grads])
    r = xm.all_reduce(xm.REDUCE_SUM, st)
    return [r[i, :t.numel()].view_as(t) for i, t in enumerate(grads)]
def p7_bucket_by_size_ar(grads, rank, ws, xm, torch):
    """bundle only equal-sized tensors (stack without padding): 4 dispatches, exact bytes"""
    outs = [None] * len(grads)
    for sz in sorted(set(t.numel() for t in grads)):
        idx = [i for i, t in enumerate(grads) if t.numel() == sz]
        r = xm.all_reduce(xm.REDUCE_SUM, torch.stack([grads[i].reshape(-1) for i in idx]))
        for j, i in enumerate(idx): outs[i] = r[j].view_as(grads[i])
    return outs

PROBLEMS.update({
    "hetero_grad_ar": dict(make=p7_make, baseline="per_tensor_ar", cands={
        "per_tensor_ar": p7_per_tensor_ar, "flat_cat_ar": p7_flat_cat_ar, "pad_stack_ar": p7_pad_stack_ar, "bucket_by_size_ar": p7_bucket_by_size_ar}),
})

def _p7_flat_pad(grads, xm, torch, multiple):
    flat = torch.cat([t.reshape(-1) for t in grads]); n = flat.numel(); pad = (-n) % multiple
    if pad: flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    return outs
def p7_flat_cat_pad_16m(grads, rank, ws, xm, torch): return _p7_flat_pad(grads, xm, torch, 16 * 1024 * 1024)   # 180.4M -> 184.5M elements
def p7_flat_cat_pad_pow2(grads, rank, ws, xm, torch): return _p7_flat_pad(grads, xm, torch, 256 * 1024 * 1024)  # 180.4M -> 256M elements
def p7_flat_cat_split2(grads, rank, ws, xm, torch):
    """two all-reduces over the exact bytes, split at a tensor boundary (sizes 4M*21 and 4M*22)"""
    half = len(grads) // 2; outs = []
    for chunk in (grads[:half], grads[half:]):
        flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
PROBLEMS["hetero_grad_ar"]["cands"].update({"flat_cat_pad_16m": p7_flat_cat_pad_16m, "flat_cat_pad_pow2": p7_flat_cat_pad_pow2, "flat_cat_split2": p7_flat_cat_split2})

# ---------------------------------------------------------------- P8: many small heterogeneous gradients (dispatch-bound at the bench)
# 48 tensors, sizes 1M..7M elements (odd multiples of 1M), total 90.2M elements (180 MB): flat cat = 1 dispatch of an odd size
S8 = [(1024, 1024 * k) for k in (1, 3, 1, 5, 1, 7, 3, 1, 1, 2, 1, 3, 1, 1, 5, 1) * 3]
def p8_make(ws, rank, dev):
    return ([(torch.randn(*s, dtype=torch.bfloat16) * 0.01).to(dev) for s in S8],)
def p8_per_tensor_ar(grads, rank, ws, xm, torch): return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def p8_flat_cat_ar(grads, rank, ws, xm, torch): return _p7_flat_pad(grads, xm, torch, 1)
def p8_flat_cat_pad_pow2(grads, rank, ws, xm, torch): return _p7_flat_pad(grads, xm, torch, 128 * 1024 * 1024)   # 90.2M -> 128M
def p8_bucket_16mb(grads, rank, ws, xm, torch):
    """buckets of ~8M elements (16 MB) in order, exact bytes"""
    outs = []; chunk = []; acc = 0
    def flush(chunk):
        flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    for t in grads:
        chunk.append(t); acc += t.numel()
        if acc >= 8 * 1024 * 1024: flush(chunk); chunk = []; acc = 0
    if chunk: flush(chunk)
    return outs
def p8_bucket_16mb_pad(grads, rank, ws, xm, torch):
    """same buckets, each padded to a power-of-two element count"""
    outs = []; chunk = []; acc = 0
    def flush(chunk):
        flat = torch.cat([t.reshape(-1) for t in chunk]); n = flat.numel(); p2 = 1 << (n - 1).bit_length(); pad = p2 - n
        if pad: flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
        r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    for t in grads:
        chunk.append(t); acc += t.numel()
        if acc >= 8 * 1024 * 1024: flush(chunk); chunk = []; acc = 0
    if chunk: flush(chunk)
    return outs
PROBLEMS["hetero_grad_ar_many"] = dict(make=p8_make, baseline="per_tensor_ar", cands={
    "per_tensor_ar": p8_per_tensor_ar, "flat_cat_ar": p8_flat_cat_ar, "flat_cat_pad_pow2": p8_flat_cat_pad_pow2,
    "bucket_16mb": p8_bucket_16mb, "bucket_16mb_pad": p8_bucket_16mb_pad})

def p8_flat_copy_into_buffer(grads, rank, ws, xm, torch):
    """one dispatch, no torch.cat: write each tensor into a preallocated flat buffer (dynamic-update-slice lowering)"""
    n = sum(t.numel() for t in grads); flat = torch.zeros(n, dtype=grads[0].dtype, device=grads[0].device); off = 0
    for t in grads: flat[off:off + t.numel()] = t.reshape(-1); off += t.numel()
    r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    return outs
def p8_pad_equal_cat(grads, rank, ws, xm, torch):
    """pad every tensor to the largest size so the cat has equal-sized operands (like the fast equal-size case), 1 dispatch"""
    m = max(t.numel() for t in grads)
    flat = torch.cat([torch.nn.functional.pad(t.reshape(-1), (0, m - t.numel())) for t in grads])
    r = xm.all_reduce(xm.REDUCE_SUM, flat)
    return [r[i * m:i * m + t.numel()].view_as(t) for i, t in enumerate(grads)]
PROBLEMS["hetero_grad_ar_many"]["cands"].update({"flat_copy_into_buffer": p8_flat_copy_into_buffer, "pad_equal_cat": p8_pad_equal_cat})

# ================================================================ day-2 campaign: rigid cross-scope inversions
# Benches must be DISPATCH-bound (many small collectives) so one-dispatch bundles win there; training exposes bytes,
# placement and the unequal-operand concat pathology.
def _mixed_sizes(n, choices, seed=7):
    import random; rng = random.Random(seed); return [rng.choice(choices) for _ in range(n)]
def _mk_small(sizes):
    def make(ws, rank, dev):
        return ([(torch.randn(n, dtype=torch.bfloat16) * 0.01).to(dev) for n in sizes],)
    return make
def _c_per_tensor(grads, rank, ws, xm, torch): return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def _c_flat_cat(grads, rank, ws, xm, torch): return _p7_flat_pad(grads, xm, torch, 1)
def _c_bucket(mb):
    def f(grads, rank, ws, xm, torch):
        outs = []; chunk = []; acc = 0; lim = mb * 1024 * 1024 // 2
        def flush(chunk):
            flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
            for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
        for t in grads:
            chunk.append(t); acc += t.numel()
            if acc >= lim: flush(chunk); chunk = []; acc = 0
        if chunk: flush(chunk)
        return outs
    return f
def _c_pad_equal_stack(grads, rank, ws, xm, torch):
    """one dispatch with EQUAL operands: pad every tensor to the largest size and stack (byte inflation = max/mean)"""
    m = max(t.numel() for t in grads)
    st = torch.stack([torch.nn.functional.pad(t.reshape(-1), (0, m - t.numel())) for t in grads])
    r = xm.all_reduce(xm.REDUCE_SUM, st)
    return [r[i, :t.numel()].view_as(t) for i, t in enumerate(grads)]
def _c_equal_group_stacks(grads, rank, ws, xm, torch):
    """one dispatch per distinct size, each a stack of equal operands (exact bytes, few dispatches)"""
    outs = [None] * len(grads)
    for sz in sorted(set(t.numel() for t in grads)):
        idx = [i for i, t in enumerate(grads) if t.numel() == sz]
        r = xm.all_reduce(xm.REDUCE_SUM, torch.stack([grads[i].reshape(-1) for i in idx]))
        for j, i in enumerate(idx): outs[i] = r[j].view_as(grads[i])
    return outs
_SMALL_CANDS = {"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "bucket_4mb": _c_bucket(4), "pad_equal_stack": _c_pad_equal_stack, "equal_group_stacks": _c_equal_group_stacks}
K = 1024
PROBLEMS["hs_a_64x64k_512k"] = dict(make=_mk_small(_mixed_sizes(64, [64 * K, 128 * K, 256 * K, 512 * K])), baseline="per_tensor_ar", cands=dict(_SMALL_CANDS))
PROBLEMS["hs_b_128x16k_128k"] = dict(make=_mk_small(_mixed_sizes(128, [16 * K, 32 * K, 64 * K, 128 * K])), baseline="per_tensor_ar", cands=dict(_SMALL_CANDS))
PROBLEMS["hs_c_32x256k_2m"] = dict(make=_mk_small(_mixed_sizes(32, [256 * K, 512 * K, K * K, 2 * K * K])), baseline="per_tensor_ar", cands=dict(_SMALL_CANDS))
PROBLEMS["hs_d_64x256k_equal"] = dict(make=_mk_small([256 * K] * 64), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "stack_ar": _c_equal_group_stacks})

# grouped DP all-reduce with tiny tensors: masked world-wide AR (1 dispatch, tp x bytes) vs grouped per-tensor
def _mk_grp_small(ws, rank, dev):
    return ([(torch.randn(64 * K, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(8)],)
PROBLEMS["grp_small_dp_ar"] = dict(make=_mk_grp_small, baseline="per_tensor_grouped_ar", cands={
    "per_tensor_grouped_ar": p1_per_tensor_grouped_ar, "flat_grouped_ar": p1_flat_grouped_ar, "masked_world_ar": p1_masked_world_ar, "grouped_rs_ag": p1_grouped_rs_ag})

# PP cross-stage transfer with many small microbatches: stacked masked AR (1 dispatch, half x M buffer) vs per-mb
M9, A9 = 8, (1, 64, 512)
def _mk_pp_small(ws, rank, dev):
    half = ws // 2
    acts = [((torch.randn(*A9, dtype=torch.bfloat16) * 0.01) if rank < half else torch.zeros(*A9, dtype=torch.bfloat16)).to(dev) for _ in range(M9)]
    return (acts, 0, half, M9)
def pp_per_mb_masked_ar(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half; outs = []
    for m in range(M):
        buf = torch.zeros(half, *acts[0].shape, dtype=acts[0].dtype, device=acts[0].device)
        if stage == src_stage: buf[pid] = acts[m]
        outs.append(xm.all_reduce(xm.REDUCE_SUM, buf)[pid])
    return outs
def pp_stacked_masked_ar(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half
    buf = torch.zeros(half, M, *acts[0].shape, dtype=acts[0].dtype, device=acts[0].device)
    if stage == src_stage: buf[pid] = torch.stack(acts)
    r = xm.all_reduce(xm.REDUCE_SUM, buf); return [r[pid, m] for m in range(M)]
def pp_stacked_ag(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1; src = rank if stage == src_stage else (rank - half if src_stage == 0 else rank + half)
    g = xm.all_gather(torch.stack(acts).unsqueeze(0), dim=0); return [g[src, m] for m in range(M)]
def pp_stacked_ag_native(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1; src = rank if stage == src_stage else (rank - half if src_stage == 0 else rank + half)
    g = xm.all_gather(torch.stack(acts).unsqueeze(0), dim=0, pin_layout=False); return [g[src, m] for m in range(M)]
def pp_stacked_masked_rs(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1
    buf = torch.zeros(ws, M, *acts[0].shape, dtype=acts[0].dtype, device=acts[0].device)
    if stage == src_stage:
        st = torch.stack(acts); buf[rank] = st; buf[rank + half if src_stage == 0 else rank - half] = st
    r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=ws); return [r[0, m] for m in range(M)]
PROBLEMS["pp_small_transfer"] = dict(make=_mk_pp_small, baseline="per_mb_masked_ar", cands={
    "per_mb_masked_ar": pp_per_mb_masked_ar, "stacked_masked_ar": pp_stacked_masked_ar, "stacked_ag": pp_stacked_ag,
    "stacked_ag_native": pp_stacked_ag_native, "stacked_masked_rs": pp_stacked_masked_rs})

# ================================================================ day-2b: realistic heterogeneous bundles (AR / AG / RS) + diagnostics
# control result: flat cat of unequal 1-D pieces is FASTER at the forward-only bench (1.29x at 64 ranks) and 7-9x slower
# inside the training graph at both 32 and 64 ranks -> a graph-context inversion. Realistic workloads with unequal bundles:
DM_, HID_, V_ = 2048, 5376, 32000
def _mk_tensors(shapes):
    def make(ws, rank, dev):
        return ([(torch.randn(*sh, dtype=torch.bfloat16) * 0.01).to(dev) for sh in shapes],)
    return make
# (1) DP gradient all-reduce of one transformer layer's parameters: qkv (3DM x DM), out (DM x DM), up+gate (2HID x DM), down (DM x HID), 2 norms (DM)
LAYER_SHAPES = [(3 * DM_, DM_), (DM_, DM_), (2 * HID_, DM_), (DM_, HID_), (DM_,), (DM_,)]
def _c_flat_cat_clone(grads, rank, ws, xm, torch):
    """diagnostic: materialize each piece (clone) before the cat"""
    flat = torch.cat([t.reshape(-1).clone() for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def _c_flat_cat_2d(grads, rank, ws, xm, torch):
    """diagnostic: cat as (1, n) pieces along dim 1 instead of 1-D pieces"""
    flat = torch.cat([t.reshape(1, -1) for t in grads], dim=1); r = xm.all_reduce(xm.REDUCE_SUM, flat)[0]; outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
_AR_CANDS = {"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "bucket_32mb": _c_bucket(32), "equal_group_stacks": _c_equal_group_stacks,
             "flat_cat_clone": _c_flat_cat_clone, "flat_cat_2d": _c_flat_cat_2d}
PROBLEMS["layer_grads_hetero_ar"] = dict(make=_mk_tensors(LAYER_SHAPES), baseline="per_tensor_ar", cands=dict(_AR_CANDS))
# (2) embedding + LM head + norms: two big and several tiny gradients
EMB_SHAPES = [(V_ // 8, DM_), (V_ // 8, DM_), (DM_,), (DM_,), (DM_,), (DM_,)]
PROBLEMS["emb_ln_hetero_ar"] = dict(make=_mk_tensors(EMB_SHAPES), baseline="per_tensor_ar", cands=dict(_AR_CANDS))
# (3) ZeRO-1 parameter all-gather of heterogeneous shards (per-rank shard of each layer parameter)
def _mk_shards(shapes):
    def make(ws, rank, dev):
        return ([(torch.randn(max(1, (a * (b if len(sh) > 1 else 1)) // 64 // 32 * 32), dtype=torch.bfloat16) * 0.01).to(dev) for sh in shapes for a, b in [(sh[0], sh[1] if len(sh) > 1 else 1)]],)
    return make
def _ag_per_tensor(shards, rank, ws, xm, torch): return [xm.all_gather(s, dim=0) for s in shards]
def _ag_flat_cat(shards, rank, ws, xm, torch):
    flat = torch.cat(shards); g = xm.all_gather(flat.unsqueeze(0), dim=0); outs = []; off = 0
    for s in shards: n = s.numel(); outs.append(g[:, off:off + n].reshape(-1)); off += n
    return outs
def _ag_flat_cat_native(shards, rank, ws, xm, torch):
    flat = torch.cat(shards); g = xm.all_gather(flat.unsqueeze(0), dim=0, pin_layout=False); outs = []; off = 0
    for s in shards: n = s.numel(); outs.append(g[:, off:off + n].reshape(-1)); off += n
    return outs
def _ag_per_tensor_native(shards, rank, ws, xm, torch): return [xm.all_gather(s, dim=0, pin_layout=False) for s in shards]
def _ag_pad_equal_stack(shards, rank, ws, xm, torch):
    m = max(s.numel() for s in shards); st = torch.stack([torch.nn.functional.pad(s, (0, m - s.numel())) for s in shards])  # (L, m)
    g = xm.all_gather(st.unsqueeze(0), dim=0)   # (ws, L, m)
    return [g[:, i, :s.numel()].reshape(-1) for i, s in enumerate(shards)]
_AG_CANDS = {"per_tensor_ag": _ag_per_tensor, "flat_cat_ag": _ag_flat_cat, "pad_equal_stack_ag": _ag_pad_equal_stack,
             "per_tensor_ag_native": _ag_per_tensor_native, "flat_cat_ag_native": _ag_flat_cat_native}
PROBLEMS["zero1_hetero_ag"] = dict(make=_mk_shards(LAYER_SHAPES), baseline="per_tensor_ag", cands=dict(_AG_CANDS))
# (4) FSDP weight prefetch of heterogeneous layer weights: shards (rows, cols/ws) gathered along the last dim
def _mk_fsdp_shards(ws, rank, dev):
    return ([(torch.randn(a, b // 64, dtype=torch.bfloat16) * 0.01).to(dev) for a, b in [(3 * DM_, DM_), (DM_, DM_), (2 * HID_, DM_), (DM_, HID_ // 64 * 64)]],)
def _fsdp_per_tensor(shards, rank, ws, xm, torch): return [xm.all_gather(s, dim=-1) for s in shards]
def _fsdp_flat_cat(shards, rank, ws, xm, torch):
    flat = torch.cat([s.reshape(-1) for s in shards]); g = xm.all_gather(flat.unsqueeze(0), dim=0)  # (ws, N): rank-major
    outs = []; off = 0
    for s in shards:
        n = s.numel(); piece = g[:, off:off + n].reshape(ws, *s.shape)   # (ws, rows, cols/ws) -> (rows, ws*cols/ws)
        outs.append(piece.permute(1, 0, 2).reshape(s.shape[0], ws * s.shape[1])); off += n
    return outs
def _fsdp_flat_cat_native(shards, rank, ws, xm, torch):
    flat = torch.cat([s.reshape(-1) for s in shards]); g = xm.all_gather(flat.unsqueeze(0), dim=0, pin_layout=False)
    outs = []; off = 0
    for s in shards:
        n = s.numel(); piece = g[:, off:off + n].reshape(ws, *s.shape); outs.append(piece.permute(1, 0, 2).reshape(s.shape[0], ws * s.shape[1])); off += n
    return outs
PROBLEMS["fsdp_hetero_ag"] = dict(make=_mk_fsdp_shards, baseline="per_tensor_ag", cands={
    "per_tensor_ag": _fsdp_per_tensor, "flat_cat_ag": _fsdp_flat_cat, "flat_cat_ag_native": _fsdp_flat_cat_native})
# (5) ZeRO-2 reduce-scatter of heterogeneous gradients: per-tensor RS vs 2-D cat along dim 1 (equal row count ws) vs 1-D cat RS of the concatenation
def _mk_rs_grads(ws, rank, dev):
    return ([(torch.randn(a * b // 64 * 64, dtype=torch.bfloat16) * 0.01).to(dev) for a, b in [(3 * DM_, DM_), (DM_, DM_), (2 * HID_, DM_), (DM_, HID_), (DM_, 1), (DM_, 1)]],)
def _rs_per_tensor(grads, rank, ws, xm, torch): return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0, scatter_dim=0, shard_count=ws) for t in grads]
def _rs_cat_dim1(grads, rank, ws, xm, torch):
    per = [t.numel() // ws for t in grads]; view = torch.cat([t.reshape(ws, -1) for t in grads], dim=1).reshape(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=ws); outs = []; off = 0
    for n in per: outs.append(r[off:off + n]); off += n
    return outs
def _rs_bucket_dim1(grads, rank, ws, xm, torch):
    outs = []
    for b in range(0, len(grads), 2):
        chunk = grads[b:b + 2]; per = [t.numel() // ws for t in chunk]; view = torch.cat([t.reshape(ws, -1) for t in chunk], dim=1).reshape(-1)
        r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=ws); off = 0
        for n in per: outs.append(r[off:off + n]); off += n
    return outs
def _rs_ar_then_slice(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel() // ws; outs.append(r[off + rank * n: off + (rank + 1) * n]); off += t.numel()
    return outs
PROBLEMS["zero2_hetero_rs"] = dict(make=_mk_rs_grads, baseline="per_tensor_rs", cands={
    "per_tensor_rs": _rs_per_tensor, "cat_dim1_rs": _rs_cat_dim1, "bucket_dim1_rs": _rs_bucket_dim1, "flat_ar_then_slice": _rs_ar_then_slice})
# diagnostics on hs_a as well
PROBLEMS["hs_a_64x64k_512k"]["cands"].update({"flat_cat_clone": _c_flat_cat_clone, "flat_cat_2d": _c_flat_cat_2d})

# (6) extreme dispatch-bound bench: 256 tiny mixed-size gradients (4K..32K elements, 4.5M total)
PROBLEMS["hs_e_256x4k_32k"] = dict(make=_mk_small(_mixed_sizes(256, [4 * K, 8 * K, 16 * K, 32 * K])), baseline="per_tensor_ar",
    cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "bucket_4mb": _c_bucket(4), "pad_equal_stack": _c_pad_equal_stack, "equal_group_stacks": _c_equal_group_stacks})
# (7) mixed-dtype layer gradients (norm params kept in fp32): bundle by casting everything to fp32 vs per-tensor in native dtype
def _mk_mixed_dtype(ws, rank, dev):
    ts = [(torch.randn(*sh, dtype=torch.bfloat16) * 0.01).to(dev) for sh in LAYER_SHAPES[:4]]
    ts += [(torch.randn(*sh, dtype=torch.float32) * 0.01).to(dev) for sh in LAYER_SHAPES[4:]]
    return (ts,)
def _md_per_tensor(grads, rank, ws, xm, torch): return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def _md_cast_flat_fp32(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1).float() for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t).to(t.dtype)); off += n
    return outs
def _md_cast_flat_bf16(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1).to(torch.bfloat16) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t).to(t.dtype)); off += n
    return outs
def _md_two_dtype_groups(grads, rank, ws, xm, torch):
    """one dispatch per dtype (exact bytes, 2 dispatches)"""
    outs = [None] * len(grads)
    for dt in (torch.bfloat16, torch.float32):
        idx = [i for i, t in enumerate(grads) if t.dtype == dt]
        if not idx: continue
        flat = torch.cat([grads[i].reshape(-1) for i in idx]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for i in idx: n = grads[i].numel(); outs[i] = r[off:off + n].view_as(grads[i]); off += n
    return outs
PROBLEMS["layer_grads_mixed_dtype"] = dict(make=_mk_mixed_dtype, baseline="per_tensor_ar", cands={
    "per_tensor_ar": _md_per_tensor, "cast_flat_fp32": _md_cast_flat_fp32, "cast_flat_bf16": _md_cast_flat_bf16, "two_dtype_groups": _md_two_dtype_groups})

# ================================================================ day-2e: more realistic many-small-collective workloads + mechanism diagnostic
# (8) whole-model DP gradient all-reduce: 4 transformer layers x 6 heterogeneous tensors = 24 gradients (~230 MB)
MODEL4_SHAPES = LAYER_SHAPES * 4
def _c_per_layer_bucket(grads, rank, ws, xm, torch):
    """one dispatch per layer (6 tensors each), exact bytes"""
    outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def _c_per_tensor_marked(grads, rank, ws, xm, torch):
    """mechanism diagnostic: per-tensor all-reduce with a graph cut after each call (what a DDP hook per gradient does)"""
    outs = []
    for t in grads:
        outs.append(xm.all_reduce(xm.REDUCE_SUM, t)); xm.mark_step()
    return outs
PROBLEMS["model_grads_4layers"] = dict(make=_mk_tensors(MODEL4_SHAPES), baseline="per_tensor_ar", cands={
    "per_tensor_ar": _c_per_tensor, "per_layer_bucket": _c_per_layer_bucket, "bucket_32mb": _c_bucket(32), "flat_cat_ar": _c_flat_cat, "per_tensor_marked": _c_per_tensor_marked})
PROBLEMS["layer_grads_hetero_ar"]["cands"]["per_tensor_marked"] = _c_per_tensor_marked
# (9) MoE pipeline transfer with heterogeneous per-microbatch payloads: hidden (1,512,DM) + router logits (1,512,64), M=4
M10 = 4; PP_SHAPES = [(1, 512, DM_), (1, 512, 64)]
def _mk_pp_hetero(ws, rank, dev):
    half = ws // 2
    acts = [[((torch.randn(*sh, dtype=torch.bfloat16) * 0.01) if rank < half else torch.zeros(*sh, dtype=torch.bfloat16)).to(dev) for sh in PP_SHAPES] for _ in range(M10)]
    return (acts, 0, half, M10)
def _pp_masked_ar(buf_fill, half, src_stage, rank, xm, torch):
    pass
def pph_per_tensor(acts, src_stage, half, M, rank, ws, xm, torch):
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half; outs = []
    for m in range(M):
        row = []
        for a in acts[m]:
            buf = torch.zeros(half, *a.shape, dtype=a.dtype, device=a.device)
            if stage == src_stage: buf[pid] = a
            row.append(xm.all_reduce(xm.REDUCE_SUM, buf)[pid])
        outs.append(row)
    return outs
def pph_per_mb_cat(acts, src_stage, half, M, rank, ws, xm, torch):
    """one masked AR per microbatch over the concatenation of its (unequal) payloads"""
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half; outs = []
    for m in range(M):
        flat = torch.cat([a.reshape(-1) for a in acts[m]]); buf = torch.zeros(half, flat.numel(), dtype=flat.dtype, device=flat.device)
        if stage == src_stage: buf[pid] = flat
        r = xm.all_reduce(xm.REDUCE_SUM, buf)[pid]; row = []; off = 0
        for a in acts[m]: n = a.numel(); row.append(r[off:off + n].view_as(a)); off += n
        outs.append(row)
    return outs
def pph_all_cat(acts, src_stage, half, M, rank, ws, xm, torch):
    """one masked AR for every payload of every microbatch"""
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half
    flat = torch.cat([a.reshape(-1) for row in acts for a in row]); buf = torch.zeros(half, flat.numel(), dtype=flat.dtype, device=flat.device)
    if stage == src_stage: buf[pid] = flat
    r = xm.all_reduce(xm.REDUCE_SUM, buf)[pid]; outs = []; off = 0
    for row in acts:
        o = []
        for a in row: n = a.numel(); o.append(r[off:off + n].view_as(a)); off += n
        outs.append(o)
    return outs
def pph_per_kind_stack(acts, src_stage, half, M, rank, ws, xm, torch):
    """one masked AR per payload kind (hidden stacked over microbatches, logits stacked over microbatches): equal operands"""
    stage = 0 if rank < half else 1; pid = rank if stage == 0 else rank - half; outs = [[None] * len(PP_SHAPES) for _ in range(M)]
    for k in range(len(PP_SHAPES)):
        st = torch.stack([acts[m][k] for m in range(M)]); buf = torch.zeros(half, *st.shape, dtype=st.dtype, device=st.device)
        if stage == src_stage: buf[pid] = st
        r = xm.all_reduce(xm.REDUCE_SUM, buf)[pid]
        for m in range(M): outs[m][k] = r[m]
    return outs
PROBLEMS["pp_hetero_transfer"] = dict(make=_mk_pp_hetero, baseline="per_tensor", cands={
    "per_tensor": pph_per_tensor, "per_mb_cat": pph_per_mb_cat, "all_cat": pph_all_cat, "per_kind_stack": pph_per_kind_stack})

# ================================================================ day-2h: three more realistic many-small-collective workloads
# (10) MoE expert gradients inside EP-DP groups (4-rank groups; intra-node at ws=32, cross-node at ws=64): w1 (HID, DM), w2 (DM, HID), gate (E, DM) per layer x 2 layers
EXP_SHAPES = [(1024, DM_), (DM_, 1024), (64, DM_)] * 2
def _mk_exp(ws, rank, dev): return ([(torch.randn(*sh, dtype=torch.bfloat16) * 0.01).to(dev) for sh in EXP_SHAPES],)
def exp_per_tensor_grouped(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); return [xm.all_reduce(xm.REDUCE_SUM, t, groups=g) for t in grads]
def exp_flat_grouped(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def exp_per_layer_grouped(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); outs = []
    for b in range(0, len(grads), 3):
        chunk = grads[b:b + 3]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def exp_equal_stacks_grouped(grads, rank, ws, xm, torch):
    g = groups_same_slot(ws, tp_of(ws)); outs = [None] * len(grads)
    for sz in sorted(set(t.numel() for t in grads)):
        idx = [i for i, t in enumerate(grads) if t.numel() == sz]
        r = xm.all_reduce(xm.REDUCE_SUM, torch.stack([grads[i].reshape(-1) for i in idx]), groups=g)
        for j, i in enumerate(idx): outs[i] = r[j].view_as(grads[i])
    return outs
PROBLEMS["expert_grads_ep_groups"] = dict(make=_mk_exp, baseline="per_tensor_grouped", cands={
    "per_tensor_grouped": exp_per_tensor_grouped, "flat_grouped": exp_flat_grouped, "per_layer_grouped": exp_per_layer_grouped, "equal_stacks_grouped": exp_equal_stacks_grouped})
# (11) 4-stage pipeline transfers: ranks split into 4 stages; each boundary moves M=4 activations (1, 512, DM) from stage s to s+1 via masked AR
M11, A11 = 4, (1, 512, DM_)
def _mk_pp4(ws, rank, dev):
    ss = ws // 4; stage = rank // ss
    acts = [[((torch.randn(*A11, dtype=torch.bfloat16) * 0.01) if stage == b else torch.zeros(*A11, dtype=torch.bfloat16)).to(dev) for _ in range(M11)] for b in range(3)]
    return (acts, ws // 4)
def _pp4_mask(ws, rank, ss, b):
    stage = rank // ss; pid = rank % ss
    return stage == b, pid   # sender of boundary b is stage b; receiver (stage b+1) reads slot pid
def pp4_per_transfer(acts, ss, rank, ws, xm, torch):
    outs = []
    for b in range(3):
        send, pid = _pp4_mask(ws, rank, ss, b); row = []
        for m in range(M11):
            buf = torch.zeros(ss, *A11, dtype=torch.bfloat16, device=acts[b][m].device)
            if send: buf[pid] = acts[b][m]
            row.append(xm.all_reduce(xm.REDUCE_SUM, buf)[pid])
        outs.append(row)
    return outs
def pp4_per_boundary_stack(acts, ss, rank, ws, xm, torch):
    outs = []
    for b in range(3):
        send, pid = _pp4_mask(ws, rank, ss, b)
        buf = torch.zeros(ss, M11, *A11, dtype=torch.bfloat16, device=acts[b][0].device)
        if send: buf[pid] = torch.stack(acts[b])
        r = xm.all_reduce(xm.REDUCE_SUM, buf)[pid]; outs.append([r[m] for m in range(M11)])
    return outs
def pp4_all_stack(acts, ss, rank, ws, xm, torch):
    buf = torch.zeros(3, ss, M11, *A11, dtype=torch.bfloat16, device=acts[0][0].device)
    for b in range(3):
        send, pid = _pp4_mask(ws, rank, ss, b)
        if send: buf[b, pid] = torch.stack(acts[b])
    r = xm.all_reduce(xm.REDUCE_SUM, buf)
    return [[r[b, rank % ss, m] for m in range(M11)] for b in range(3)]
PROBLEMS["pp_4stage_transfer"] = dict(make=_mk_pp4, baseline="per_transfer", cands={
    "per_transfer": pp4_per_transfer, "per_boundary_stack": pp4_per_boundary_stack, "all_stack": pp4_all_stack})
# (12) global gradient-norm clipping statistic: sum over 24 tensors of ||g||^2 across DP ranks
def _mk_norm(ws, rank, dev): return ([(torch.randn(*sh, dtype=torch.bfloat16) * 0.01).to(dev) for sh in MODEL4_SHAPES],)
def norm_per_tensor_ar(grads, rank, ws, xm, torch):
    return [xm.all_reduce(xm.REDUCE_SUM, t.float().pow(2).sum().reshape(1)) for t in grads]
def norm_stack_ar(grads, rank, ws, xm, torch):
    r = xm.all_reduce(xm.REDUCE_SUM, torch.stack([t.float().pow(2).sum() for t in grads])); return [r[i].reshape(1) for i in range(len(grads))]
def norm_local_sum_then_ar(grads, rank, ws, xm, torch):
    """algebraic: reduce locally to one scalar first (a per-tensor breakdown is not needed for clipping)"""
    s = sum(t.float().pow(2).sum() for t in grads).reshape(1); r = xm.all_reduce(xm.REDUCE_SUM, s)
    return [r] + [torch.zeros_like(r) for _ in grads[1:]]
PROBLEMS["grad_norm_clip_stat"] = dict(make=_mk_norm, baseline="per_tensor_ar", cands={
    "per_tensor_ar": norm_per_tensor_ar, "stack_ar": norm_stack_ar, "local_sum_then_ar": norm_local_sum_then_ar})

# (13) DDP bucket-size sweep on the 4-layer model gradients (same tensors as model_grads_4layers)
def _c_bucket_mb(mb):
    def f(grads, rank, ws, xm, torch):
        outs = []; chunk = []; acc = 0; lim = mb * 1024 * 1024 // 2
        def flush(chunk):
            flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
            for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
        for t in grads:
            chunk.append(t); acc += t.numel()
            if acc >= lim: flush(chunk); chunk = []; acc = 0
        if chunk: flush(chunk)
        return outs
    return f
PROBLEMS["model_grads_bucket_sweep"] = dict(make=_mk_tensors(MODEL4_SHAPES), baseline="per_tensor_ar", cands={
    "per_tensor_ar": _c_per_tensor, "bucket_1mb": _c_bucket_mb(1), "bucket_4mb": _c_bucket_mb(4), "bucket_16mb": _c_bucket_mb(16),
    "bucket_64mb": _c_bucket_mb(64), "bucket_256mb": _c_bucket_mb(256), "flat_cat_ar": _c_flat_cat})

# ================================================================ day-3a: calibration probe for the simulator fix
# N independent all-reduces of equal size, lone graph (bench) vs training-shaped graph (g_train): per-collective cost curve
def _mk_eq(n_elems, count):
    def make(ws, rank, dev): return ([(torch.randn(n_elems, dtype=torch.bfloat16) * 0.01).to(dev) for _ in range(count)],)
    return make
def _ar_each(grads, rank, ws, xm, torch): return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
def _ar_stack(grads, rank, ws, xm, torch):
    r = xm.all_reduce(xm.REDUCE_SUM, torch.stack(grads)); return [r[i] for i in range(len(grads))]
def _rs_each(grads, rank, ws, xm, torch): return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0, scatter_dim=0, shard_count=ws) for t in grads]
def _ag_each(grads, rank, ws, xm, torch): return [xm.all_gather(t, dim=0) for t in grads]
def _ag_each_native(grads, rank, ws, xm, torch): return [xm.all_gather(t, dim=0, pin_layout=False) for t in grads]
for _n, _lab in ((64 * K, "64k"), (512 * K, "512k"), (4 * K * K, "4m"), (16 * K * K, "16m")):
    PROBLEMS[f"probe_eq8_{_lab}"] = dict(make=_mk_eq(_n, 8), baseline="ar_each", cands={"ar_each": _ar_each, "ar_stack": _ar_stack, "rs_each": _rs_each, "ag_each": _ag_each, "ag_each_native": _ag_each_native})
# ================================================================ day-3b: algebraic rewrites and API-parameter choices on the new workloads
def _two_stage_grouped_ar(grads, rank, ws, xm, torch):
    """two-stage all-reduce through replica groups: g1 groups of 8 consecutive ranks, then groups of ranks sharing (rank % 8)"""
    a = 8; b = ws // a
    g1 = [[i * a + j for j in range(a)] for i in range(b)]; g2 = [[i * a + j for i in range(b)] for j in range(a)]
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g1); r = xm.all_reduce(xm.REDUCE_SUM, r, groups=g2)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def _rs_ag_flat_native(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); n = flat.numel(); pad = (-n) % ws
    if pad: flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=ws); r = xm.all_gather(rs, dim=0, pin_layout=False)[:n]
    outs = []; off = 0
    for t in grads: m = t.numel(); outs.append(r[off:off + m].view_as(t)); off += m
    return outs
def _flat_ar_unpinned(grads, rank, ws, xm, torch):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, pin_layout=False); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def _per_tensor_unpinned(grads, rank, ws, xm, torch): return [xm.all_reduce(xm.REDUCE_SUM, t, pin_layout=False) for t in grads]
def _flat_ar_scaled(grads, rank, ws, xm, torch):
    """mean instead of sum via the collective's scale argument (post-multiplied back to a sum for the check)"""
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, scale=1.0 / ws) * ws; outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
PROBLEMS["layer_grads_api_variants"] = dict(make=_mk_tensors(LAYER_SHAPES), baseline="per_tensor_ar", cands={
    "per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "two_stage_grouped_ar": _two_stage_grouped_ar, "rs_ag_flat_native": _rs_ag_flat_native,
    "flat_ar_unpinned": _flat_ar_unpinned, "per_tensor_unpinned": _per_tensor_unpinned, "flat_ar_scaled": _flat_ar_scaled})
# Ulysses all-to-all through the all_to_all API (compiles on one node; multi-node compile is the open question) vs masked RS
def p4_all_to_all_api(x, rank, ws, xm, torch):
    c = _p4_chunks(x, ws)                      # (ws, S_local, H/ws, D), chunk d for destination d
    r = xm.all_to_all(c, split_dimension=0, concat_dimension=0, split_count=ws)
    return _p4_assemble(r, ws)
PROBLEMS["ulysses_api_variants"] = dict(make=p4_make, baseline="ag_transpose_rs", cands={
    "ag_transpose_rs": p4_ag_transpose_rs, "masked_rs": p4_masked_rs, "ag_slice_native": p4_ag_slice_native, "all_to_all_api": p4_all_to_all_api})
# PP transfer through collective_permute pairs (the API the paper's baselines avoid at 224 ranks) vs masked AR, small payload
def pp_permute_pairs(acts, src_stage, half, M, rank, ws, xm, torch):
    pairs = [(r, r + half) for r in range(half)]
    stage = 0 if rank < half else 1; outs = []
    for m in range(M):
        y = xm.collective_permute(acts[m], pairs)
        outs.append(acts[m] if stage == src_stage else y)
    return outs
def pp_permute_stacked(acts, src_stage, half, M, rank, ws, xm, torch):
    pairs = [(r, r + half) for r in range(half)]; stage = 0 if rank < half else 1
    y = xm.collective_permute(torch.stack(acts), pairs)
    return list(acts) if stage == src_stage else [y[m] for m in range(M)]
PROBLEMS["pp_api_variants"] = dict(make=_mk_pp_small, baseline="per_mb_masked_ar", cands={
    "per_mb_masked_ar": pp_per_mb_masked_ar, "stacked_masked_ar": pp_stacked_masked_ar, "permute_pairs": pp_permute_pairs, "permute_stacked": pp_permute_stacked})

# ================================================================ day-3b: heterogeneous-set probe for the simulator fix: count x size-class
# per-tensor AR vs one equal-operand bundle (per-layer-style buckets of 6) vs flat cat; sizes mixed within each class
def _mk_het(count, choices, seed):
    sizes = _mixed_sizes(count, choices, seed)
    def make(ws, rank, dev): return ([(torch.randn(n, dtype=torch.bfloat16) * 0.01).to(dev) for n in sizes],)
    return make
def _c_bucket6(grads, rank, ws, xm, torch):
    outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
for _cnt in (6, 24, 48):
    for _lab, _ch in (("small", [64 * K, 128 * K, 256 * K, 512 * K]), ("mid", [K * K, 2 * K * K, 4 * K * K, 8 * K * K]), ("big", [4 * K * K, 8 * K * K, 16 * K * K, 22 * K * K])):
        PROBLEMS[f"probe_het_{_cnt}_{_lab}"] = dict(make=_mk_het(_cnt, _ch, 11), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "bucket6": _c_bucket6, "flat_cat_ar": _c_flat_cat})

# ================================================================ Phase-3 picks (OverlayCCL with simulator v5), adapted to the hardware harness signature
def _p3_layer_grads_hetero_ar_inner(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """Single flat concatenation all-reduce: concatenate all 6 tensors, one all_reduce, slice back."""
    # Flatten and concatenate all gradients into a single 1-D buffer
    flat = torch.cat([t.reshape(-1) for t in grads])
    
    # Perform a single all-reduce on the concatenated buffer
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Slice and reshape back to original shapes
    outs = []
    offset = 0
    for t in grads:
        num_elements = t.numel()
        outs.append(reduced[offset:offset + num_elements].view(t.shape))
        offset += num_elements
    
    return outs
def p3_pick_layer_grads_hetero_ar(grads, rank, ws, xm, torch): return _p3_layer_grads_hetero_ar_inner(grads, rank, ws, 16, 2, xm, torch, num_nodes=max(1, ws // 32))
PROBLEMS["layer_grads_hetero_ar"]["cands"]["p3_pick"] = p3_pick_layer_grads_hetero_ar
def _p3_model_grads_4layers_inner(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """Per-layer bucketed all-reduce with optimized flatten/unflatten using cat/split"""
    outs = [None] * 24
    
    # Process 4 layers, each with 6 parameters
    for layer_idx in range(4):
        start_idx = layer_idx * 6
        end_idx = start_idx + 6
        chunk = grads[start_idx:end_idx]
        
        # Store shapes and sizes for unpacking
        shapes = [t.shape for t in chunk]
        sizes = [t.numel() for t in chunk]
        
        # Flatten all tensors and concatenate in one operation
        flat_tensors = [t.reshape(-1) for t in chunk]
        flat = torch.cat(flat_tensors)
        
        # Single all_reduce for the entire layer
        reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
        
        # Unpack using split (single operation) and reshape
        split_tensors = torch.split(reduced, sizes)
        for i, (split_t, shape) in enumerate(zip(split_tensors, shapes)):
            outs[start_idx + i] = split_t.view(shape)
    
    return outs
def p3_pick_model_grads_4layers(grads, rank, ws, xm, torch): return _p3_model_grads_4layers_inner(grads, rank, ws, 16, 2, xm, torch, num_nodes=max(1, ws // 32))
PROBLEMS["model_grads_4layers"]["cands"]["p3_pick"] = p3_pick_model_grads_4layers
def _p3_hs_a_64x64k_512k_inner(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    """Single flat concatenation all-reduce: concatenate all 64 tensors, one all_reduce, slice outputs."""
    # Concatenate all gradient tensors into a single flat buffer
    flat = torch.cat([t.reshape(-1) for t in grads])
    
    # Perform a single all-reduce operation
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Slice and reshape outputs back to original shapes
    outs = []
    offset = 0
    for t in grads:
        num_elements = t.numel()
        outs.append(reduced[offset:offset + num_elements].view(t.shape))
        offset += num_elements
    
    return outs
def p3_pick_hs_a_64x64k_512k(grads, rank, ws, xm, torch): return _p3_hs_a_64x64k_512k_inner(grads, rank, ws, 16, 2, xm, torch, num_nodes=max(1, ws // 32))
PROBLEMS["hs_a_64x64k_512k"]["cands"]["p3_pick"] = p3_pick_hs_a_64x64k_512k

# ================================================================ day-3e: the inverting workload under different setups
# (a) heterogeneous model gradients all-reduced inside 4-rank DP groups that span nodes at ws=64 (intra-node at ws=32)
def _g4(ws): return groups_same_slot(ws, tp_of(ws))
def mg_per_tensor_grouped(grads, rank, ws, xm, torch):
    g = _g4(ws); return [xm.all_reduce(xm.REDUCE_SUM, t, groups=g) for t in grads]
def mg_per_layer_grouped(grads, rank, ws, xm, torch):
    g = _g4(ws); outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def mg_flat_grouped(grads, rank, ws, xm, torch):
    g = _g4(ws); flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
PROBLEMS["model_grads_ep_groups"] = dict(make=_mk_tensors(MODEL4_SHAPES), baseline="per_tensor_grouped", cands={
    "per_tensor_grouped": mg_per_tensor_grouped, "per_layer_grouped": mg_per_layer_grouped, "flat_grouped": mg_flat_grouped})
# (b) same inside 32-rank intra-node groups (one group per node): placement identical at both scopes -> control
def _gnode(ws): return [[n * 32 + i for i in range(32)] for n in range(ws // 32)]
def mn_per_tensor(grads, rank, ws, xm, torch):
    g = _gnode(ws); return [xm.all_reduce(xm.REDUCE_SUM, t, groups=g) for t in grads]
def mn_per_layer(grads, rank, ws, xm, torch):
    g = _gnode(ws); outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
def mn_flat(grads, rank, ws, xm, torch):
    g = _gnode(ws); flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=g); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
    return outs
PROBLEMS["model_grads_intra_node_groups"] = dict(make=_mk_tensors(MODEL4_SHAPES), baseline="per_tensor", cands={
    "per_tensor": mn_per_tensor, "per_layer": mn_per_layer, "flat": mn_flat})
# (c) fp32 gradients (2x bytes per element)
def _mk_tensors_fp32(shapes):
    def make(ws, rank, dev): return ([(torch.randn(*sh, dtype=torch.float32) * 0.01).to(dev) for sh in shapes],)
    return make
PROBLEMS["model_grads_fp32"] = dict(make=_mk_tensors_fp32(MODEL4_SHAPES), baseline="per_tensor_ar", cands={
    "per_tensor_ar": _c_per_tensor, "per_layer_bucket": _c_per_layer_bucket, "flat_cat_ar": _c_flat_cat})

# ================================================================ day-3 diagnostics: what makes the layer-structured per-tensor sets slow?
_L1D = [(int(__import__("math").prod(sh)),) for sh in LAYER_SHAPES]
PROBLEMS["layer_grads_1d"] = dict(make=_mk_tensors(_L1D), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["layer_grads_no_norm"] = dict(make=_mk_tensors(LAYER_SHAPES[:4]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["model_grads_1d"] = dict(make=_mk_tensors(_L1D * 4), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "per_layer_bucket": _c_bucket6})
PROBLEMS["model_grads_no_norm"] = dict(make=_mk_tensors(LAYER_SHAPES[:4] * 4), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})

# ================================================================ day-4: (a) unequal-concat piece-count sweep, (b) pathology trigger diagnostics
def _c_bucket_pieces(npieces):
    def f(grads, rank, ws, xm, torch):
        outs = []
        for b in range(0, len(grads), npieces):
            chunk = grads[b:b + npieces]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
            for t in chunk: n = t.numel(); outs.append(r[off:off + n].view_as(t)); off += n
        return outs
    return f
for _k in (8, 16, 32, 64, 128):
    PROBLEMS[f"catk_{_k}_small"] = dict(make=_mk_het(_k, [64 * K, 128 * K, 256 * K, 512 * K], 11), baseline="per_tensor_ar",
        cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "bucket8": _c_bucket_pieces(8)})
for _k in (8, 16, 32, 48):
    PROBLEMS[f"catk_{_k}_mid"] = dict(make=_mk_het(_k, [K * K, 2 * K * K, 4 * K * K, 8 * K * K], 11), baseline="per_tensor_ar",
        cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "bucket8": _c_bucket_pieces(8)})
# pathology trigger: which change makes the flat concat fast / slow?
_BIG4 = LAYER_SHAPES[:4]
PROBLEMS["path_no_norm_plus_tiny"] = dict(make=_mk_tensors(_BIG4 + [(DM_,)]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["path_no_norm_1d"] = dict(make=_mk_tensors([(int(__import__("math").prod(sh)),) for sh in _BIG4]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["path_no_norm_clone"] = dict(make=_mk_tensors(_BIG4), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat, "flat_cat_clone": _c_flat_cat_clone})
PROBLEMS["path_many_1d"] = dict(make=_mk_tensors([(r * c,) for (r, c) in S8]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["path_many_plus_tiny"] = dict(make=_mk_tensors(list(S8) + [(DM_,)]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["path_many_clone"] = dict(make=p8_make, baseline="per_tensor_ar", cands={"per_tensor_ar": p8_per_tensor_ar, "flat_cat_ar": p8_flat_cat_ar, "flat_cat_clone": _c_flat_cat_clone})
PROBLEMS["path_many_half"] = dict(make=_mk_tensors(list(S8)[:24]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})

# ================================================================ Phase-3 picks (v5 + dtype sandbox, 2026-10-27) adapted to the harness signature
def p3pick_layer_grads_mixed_dtype(grads, rank, ws, xm, torch):
    """
    Parallel dtype-split with async dispatch: launches two independent all_reduce operations
    (one for bf16 tensors 0-3, one for fp32 tensors 4-5) to exploit concurrent execution.
    """
    # Split tensors by dtype
    bf16_grads = grads[:4]
    fp32_grads = grads[4:6]
    
    # Concatenate each dtype group
    bf16_flat = torch.cat([t.reshape(-1) for t in bf16_grads])
    fp32_flat = torch.cat([t.reshape(-1) for t in fp32_grads])
    
    # Launch both all_reduce operations (potentially in parallel/overlapped)
    bf16_reduced = xm.all_reduce(xm.REDUCE_SUM, bf16_flat)
    fp32_reduced = xm.all_reduce(xm.REDUCE_SUM, fp32_flat)
    
    # Unpack bf16 results
    bf16_results = []
    offset = 0
    for t in bf16_grads:
        n = t.numel()
        bf16_results.append(bf16_reduced[offset:offset + n].view(t.shape))
        offset += n
    
    # Unpack fp32 results
    fp32_results = []
    offset = 0
    for t in fp32_grads:
        n = t.numel()
        fp32_results.append(fp32_reduced[offset:offset + n].view(t.shape))
        offset += n
    
    # Combine results in original order
    return bf16_results + fp32_results
PROBLEMS["layer_grads_mixed_dtype"]["cands"]["p3_pick"] = p3pick_layer_grads_mixed_dtype
def p3pick_probe_het_24_mid(grads, rank, ws, xm, torch):
    """Single flat concatenation all-reduce: concatenate all 24 tensors, one all_reduce, then slice back."""
    # Concatenate all tensors into one flat buffer
    flat = torch.cat([t.reshape(-1) for t in grads])
    
    # Perform single all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Slice and reshape back to original tensor shapes
    outs = []
    offset = 0
    for t in grads:
        n = t.numel()
        outs.append(reduced[offset:offset + n].view(t.shape))
        offset += n
    
    return outs
PROBLEMS["probe_het_24_mid"]["cands"]["p3_pick"] = p3pick_probe_het_24_mid
def p3pick_probe_het_48_mid(grads, rank, ws, xm, torch):
    """Single bucket all-reduce: concatenate all tensors once, all-reduce, then split."""
    
    # Flatten all gradients and track sizes for splitting
    flat_grads = []
    sizes = []
    shapes = []
    
    for t in grads:
        flat_grads.append(t.reshape(-1))
        sizes.append(t.numel())
        shapes.append(t.shape)
    
    # Concatenate all flattened gradients
    all_flat = torch.cat(flat_grads)
    
    # Single all-reduce operation
    reduced_flat = xm.all_reduce(xm.REDUCE_SUM, all_flat)
    
    # Split back into original shapes using views where possible
    results = []
    offset = 0
    for size, shape in zip(sizes, shapes):
        # Use narrow to create a view (metadata-only operation)
        tensor_view = reduced_flat.narrow(0, offset, size)
        # Reshape to original shape
        results.append(tensor_view.view(shape))
        offset += size
    
    return results
PROBLEMS["probe_het_48_mid"]["cands"]["p3_pick"] = p3pick_probe_het_48_mid
def p3pick_probe_het_48_small(grads, rank, ws, xm, torch):
    """Single flat concatenation all-reduce with optimized view-based reconstruction."""
    # Calculate total size
    total_size = sum(t.numel() for t in grads)
    
    # Pre-allocate flat buffer and copy data in-place
    flat = torch.empty(total_size, dtype=grads[0].dtype, device=grads[0].device)
    offset = 0
    for t in grads:
        n = t.numel()
        flat[offset:offset + n] = t.reshape(-1)
        offset += n
    
    # Perform single all-reduce operation
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Return views directly from the reduced buffer (metadata-only)
    outs = []
    offset = 0
    for t in grads:
        n = t.numel()
        # Use narrow (metadata-only view) instead of slicing, then reshape
        outs.append(reduced.narrow(0, offset, n).view(t.shape))
        offset += n
    
    return outs
PROBLEMS["probe_het_48_small"]["cands"]["p3_pick"] = p3pick_probe_het_48_small

# ================================================================ day-4e: what cures / causes the pathological flat concat?
# E1-E3: EQUAL pieces whose total matches a pathological / benign unequal total (is it the total, not the inequality?)
PROBLEMS["eq95_512k"] = dict(make=_mk_small([512 * K] * 95), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})      # 2^19 x 95 (= layer no-norm total)
PROBLEMS["eq111_1m"] = dict(make=_mk_small([K * K] * 111), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})       # 2^20 x 111 (= hetero_many total)
PROBLEMS["eq4_unequal_total"] = dict(make=_mk_small([12 * K * K, 4 * K * K, 21 * K * K, 10 * K * K]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})  # 4 1-D pieces, 47M total, odd mix
# E4: size of the appended tiny piece that cures the layer no-norm concat
for _s in (1, 64, 1024, 65536):
    PROBLEMS[f"path_no_norm_plus_{_s}"] = dict(make=_mk_tensors(_BIG4 + [(_s,)]), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
# E5: tiny piece FIRST (shifts every offset) and E6: pieces sorted by size
PROBLEMS["path_many_tiny_first"] = dict(make=_mk_tensors([(DM_,)] + list(S8)), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
PROBLEMS["path_many_sorted"] = dict(make=_mk_tensors(sorted(S8, key=lambda sh: sh[0] * sh[1])), baseline="per_tensor_ar", cands={"per_tensor_ar": _c_per_tensor, "flat_cat_ar": _c_flat_cat})
# E7: the two "identical" mixed-dtype bundles side by side (pick vs hand-written) in one problem for an HLO diff
PROBLEMS["md_pick_vs_hand"] = dict(make=PROBLEMS["layer_grads_mixed_dtype"]["make"], baseline="two_dtype_groups", cands={"two_dtype_groups": _md_two_dtype_groups, "p3_pick": PROBLEMS["layer_grads_mixed_dtype"]["cands"]["p3_pick"]})

# ================================================================ Phase-3 picks with the unequal-concat term (v5uneq, 2026-10-28)
def p3uneq_dp_grad_bucketing_64(grads, rank, ws, xm, torch):
    """Size-sorted equal-count bucketing strategy"""
    
    # Step 1: Create a list of (index, size, tensor) tuples and sort by size
    indexed_grads = [(i, t.numel(), t) for i, t in enumerate(grads)]
    indexed_grads.sort(key=lambda x: x[1])
    
    # Step 2: Partition into fixed-count buckets (8 buckets of 8 tensors each)
    num_buckets = 8
    bucket_size = len(grads) // num_buckets  # 64 // 8 = 8
    
    # Step 3: Process each bucket with all-reduce
    reduced_tensors = [None] * len(grads)
    
    for bucket_idx in range(num_buckets):
        start_idx = bucket_idx * bucket_size
        end_idx = start_idx + bucket_size
        bucket = indexed_grads[start_idx:end_idx]
        
        # Concatenate tensors in this bucket
        bucket_tensors = [item[2] for item in bucket]
        flat = torch.cat([t.reshape(-1) for t in bucket_tensors])
        
        # Perform all-reduce on concatenated bucket
        reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat)
        
        # Split back and place in original positions
        offset = 0
        for orig_idx, size, orig_tensor in bucket:
            reduced_tensors[orig_idx] = reduced_flat[offset:offset + size].view(orig_tensor.shape)
            offset += size
    
    return reduced_tensors
PROBLEMS["hs_a_64x64k_512k"]["cands"]["p3_pick2"] = p3uneq_dp_grad_bucketing_64
def p3uneq_probe_het_48_small(grads, rank, ws, xm, torch):
    """Optimized size-binned all-reduce with minimal local ops"""
    
    # Define size bins (in number of elements)
    SMALL_THRESHOLD = 256
    LARGE_THRESHOLD = 25600
    
    # Categorize tensors into bins with their original indices
    small_bin = []
    medium_bin = []
    large_bin = []
    
    for idx, t in enumerate(grads):
        numel = t.numel()
        if numel < SMALL_THRESHOLD:
            small_bin.append((idx, t))
        elif numel < LARGE_THRESHOLD:
            medium_bin.append((idx, t))
        else:
            large_bin.append((idx, t))
    
    # Result list to preserve order
    results = [None] * len(grads)
    
    # Process each bin with minimal local ops
    def process_bin(bin_list):
        if not bin_list:
            return
        
        if len(bin_list) == 1:
            # Single tensor: direct all-reduce, no concatenation needed
            idx, t = bin_list[0]
            flat = t.reshape(-1)
            reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
            results[idx] = reduced.reshape(t.shape)
            return
        
        # Extract indices and tensors
        indices = [item[0] for item in bin_list]
        tensors = [item[1] for item in bin_list]
        
        # Pre-flatten tensors to avoid repeated reshape calls
        flats = [t.reshape(-1) for t in tensors]
        
        # Concatenate all tensors in this bin
        flat = torch.cat(flats)
        
        # All-reduce the concatenated tensor
        reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
        
        # Split back using narrow (metadata-only) instead of slicing
        offset = 0
        for idx, t in zip(indices, tensors):
            n = t.numel()
            # Use narrow for zero-copy slice, then reshape
            results[idx] = reduced.narrow(0, offset, n).reshape(t.shape)
            offset += n
    
    # Process each bin
    process_bin(small_bin)
    process_bin(medium_bin)
    process_bin(large_bin)
    
    return results
PROBLEMS["probe_het_48_small"]["cands"]["p3_pick2"] = p3uneq_probe_het_48_small
def p3uneq_probe_het_48_mid(grads, rank, ws, xm, torch):
    """Single flat buffer all-reduce with minimal copying and view-based outputs"""
    # Calculate total size
    total_size = sum(t.numel() for t in grads)
    
    # Pre-allocate flat buffer
    flat = torch.empty(total_size, dtype=grads[0].dtype, device=grads[0].device)
    
    # Copy tensors into flat buffer
    offset = 0
    shapes = []
    for t in grads:
        n = t.numel()
        flat[offset:offset + n] = t.reshape(-1)
        shapes.append((t.shape, n))
        offset += n
    
    # Perform single all-reduce in-place
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Return views (metadata-only) into the reduced buffer
    outs = []
    offset = 0
    for shape, n in shapes:
        outs.append(reduced[offset:offset + n].view(shape))
        offset += n
    
    return outs
PROBLEMS["probe_het_48_mid"]["cands"]["p3_pick2"] = p3uneq_probe_het_48_mid
def p3uneq_hetero_grad_ar_many(grads, rank, ws, xm, torch):
    """Two-phase reduction by size class with minimal local ops."""
    
    # Pre-compute metadata without creating intermediate tensors
    n = len(grads)
    grad_metadata = []
    
    for i in range(n):
        t = grads[i]
        num_cols = t.shape[1] if t.dim() > 1 else 1
        num_elems = t.numel()
        grad_metadata.append((i, num_cols, num_elems, t.shape))
    
    # Define size classes using tertiles on column counts
    col_counts = sorted([meta[1] for meta in grad_metadata])
    
    if n >= 3:
        small_thresh = col_counts[n // 3]
        large_thresh = col_counts[2 * n // 3]
    else:
        small_thresh = col_counts[0] if n > 0 else 0
        large_thresh = col_counts[-1] if n > 0 else float('inf')
    
    # Partition indices into size classes (metadata only)
    small_indices = []
    medium_indices = []
    large_indices = []
    
    for idx, num_cols, num_elems, shape in grad_metadata:
        if num_cols <= small_thresh:
            small_indices.append(idx)
        elif num_cols <= large_thresh:
            medium_indices.append(idx)
        else:
            large_indices.append(idx)
    
    # Process each size class
    def process_class(indices):
        if not indices:
            return []
        
        # Build single flat tensor by concatenating views
        flat_parts = [grads[i].view(-1) for i in indices]
        flat = torch.cat(flat_parts)
        
        # Single all_reduce
        reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
        
        # Split using narrow (metadata-only view operation)
        results = []
        offset = 0
        for i in indices:
            size = grad_metadata[i][2]
            # Use narrow for zero-copy slicing, then reshape to original
            results.append((i, reduced.narrow(0, offset, size).reshape(grad_metadata[i][3])))
            offset += size
        
        return results
    
    # Process all classes
    small_results = process_class(small_indices)
    medium_results = process_class(medium_indices)
    large_results = process_class(large_indices)
    
    # Recombine in original order - use a pre-allocated list
    output = [None] * n
    for idx, tensor in small_results + medium_results + large_results:
        output[idx] = tensor
    
    return output
PROBLEMS["hetero_grad_ar_many"]["cands"]["p3_pick2"] = p3uneq_hetero_grad_ar_many
def p3uneq_hs_c_32mixed(grads, rank, ws, xm, torch):
    """Single flat concatenation all-reduce: concatenate all 32 tensors, one all_reduce, then slice back."""
    if not grads:
        return []
    
    # Concatenate all gradients into one flat buffer
    flat = torch.cat([t.reshape(-1) for t in grads])
    
    # Single all_reduce on the concatenated buffer
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Slice and reshape back to original tensor shapes
    outs = []
    offset = 0
    for t in grads:
        n = t.numel()
        outs.append(reduced[offset:offset + n].view(t.shape))
        offset += n
    
    return outs
PROBLEMS["hs_c_32x256k_2m"]["cands"]["p3_pick2"] = p3uneq_hs_c_32mixed

# ================================================================ Phase-3 picks, simulator v8 (tiny-mix gain, count knee, unequal-concat + assembled-buffer term), 2026-10-28
def p3v8_dp_grad_bucketing_64(grads, rank, ws, xm, torch):
    """Optimized tensor-count bucketed all-reduce with minimal local ops"""
    
    # Configuration: fewer buckets to reduce overhead
    num_buckets = 4
    tensors_per_bucket = len(grads) // num_buckets
    
    outs = [None] * len(grads)
    
    for bucket_idx in range(num_buckets):
        # Determine the range of tensors for this bucket
        start_idx = bucket_idx * tensors_per_bucket
        
        # Last bucket gets any remaining tensors
        if bucket_idx == num_buckets - 1:
            end_idx = len(grads)
        else:
            end_idx = start_idx + tensors_per_bucket
        
        # Store shapes and flatten tensors (pre-flatten to avoid repeated reshapes)
        bucket_tensors = []
        shapes = []
        for i in range(start_idx, end_idx):
            t = grads[i]
            shapes.append(t.shape)
            # Only flatten if not already 1D to minimize ops
            if t.dim() > 1:
                bucket_tensors.append(t.view(-1))
            else:
                bucket_tensors.append(t)
        
        # Single concatenation operation instead of manual copy loop
        flat = torch.cat(bucket_tensors)
        
        # Perform all-reduce on the concatenated tensor
        reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
        
        # Split using narrow (metadata-only view) instead of slicing
        offset = 0
        for i, shape in enumerate(shapes):
            n = grads[start_idx + i].numel()
            # Use narrow for zero-copy slicing, then view to restore shape
            outs[start_idx + i] = reduced.narrow(0, offset, n).view(shape)
            offset += n
    
    return outs
PROBLEMS["hs_a_64x64k_512k"]["cands"]["p3_pick3"] = p3v8_dp_grad_bucketing_64
def p3v8_probe_het_48_small(grads, rank, ws, xm, torch):
    """
    Optimized reduce-scatter + all-gather with minimal local ops
    Reduces local operations by avoiding unnecessary copies and concatenations
    """
    if ws == 1:
        return grads
    
    # Concatenate all gradients into a single flat buffer (unavoidable for collective)
    flat = torch.cat([t.flatten() for t in grads])
    total_size = flat.numel()
    
    # Pad to make size divisible by ws for even chunking
    padded_size = ((total_size + ws - 1) // ws) * ws
    if padded_size > total_size:
        # Pad in-place by resizing instead of creating new tensor
        pad_amount = padded_size - total_size
        flat = torch.nn.functional.pad(flat, (0, pad_amount))
    
    # Phase 1: Reduce-scatter using collective primitive
    # This reduces and scatters in one operation
    scatter_output = xm.reduce_scatter(
        xm.REDUCE_SUM,
        flat,
        scale=1.0,
        scatter_dim=0,
        shard_count=ws,
        groups=None
    )
    
    # Phase 2: All-gather to collect all reduced chunks on all ranks
    gathered = xm.all_gather(scatter_output, dim=0)
    
    # Split back into original gradient shapes using views where possible
    outs = []
    offset = 0
    for t in grads:
        n = t.numel()
        # Use narrow (a view operation) instead of slicing when possible
        if offset + n <= total_size:
            chunk = gathered.narrow(0, offset, n).reshape(t.shape)
        else:
            # Only slice if we need to handle padding edge case
            chunk = gathered[offset:offset + n].reshape(t.shape)
        outs.append(chunk)
        offset += n
    
    return outs
PROBLEMS["probe_het_48_small"]["cands"]["p3_pick3"] = p3v8_probe_het_48_small
def p3v8_probe_het_48_mid(grads, rank, ws, xm, torch):
    """Optimized size-binned all-reduce with minimal local ops."""
    
    # Pre-flatten all gradients once (unavoidable, but do it upfront)
    flat_grads = [g.reshape(-1) for g in grads]
    sizes = [fg.numel() for fg in flat_grads]
    
    # Sort indices by size (metadata-only operation)
    sorted_indices = sorted(range(len(grads)), key=lambda i: sizes[i])
    
    # Determine number of bins
    num_bins = min(8, max(4, len(grads) // 6))
    tensors_per_bin = len(grads) // num_bins
    remainder = len(grads) % num_bins
    
    # Allocate result list
    results = [None] * len(grads)
    
    # Process bins
    idx = 0
    for bin_idx in range(num_bins):
        bin_size = tensors_per_bin + (1 if bin_idx < remainder else 0)
        if bin_size == 0:
            continue
        
        # Get indices for this bin (metadata-only)
        bin_indices = sorted_indices[idx:idx + bin_size]
        idx += bin_size
        
        # Concatenate only the tensors in this bin
        bin_tensors = [flat_grads[i] for i in bin_indices]
        
        # Single concatenation per bin
        concatenated = torch.cat(bin_tensors)
        
        # Single all-reduce per bin
        reduced = xm.all_reduce(xm.REDUCE_SUM, concatenated)
        
        # Split using narrow (metadata-only view operation)
        offset = 0
        for i in bin_indices:
            size = sizes[i]
            # Use narrow for zero-copy slicing, then reshape to original shape
            results[i] = reduced.narrow(0, offset, size).view(grads[i].shape)
            offset += size
    
    return results
PROBLEMS["probe_het_48_mid"]["cands"]["p3_pick3"] = p3v8_probe_het_48_mid
def p3v8_hetero_grad_ar_many(grads, rank, ws, xm, torch):
    """Two-phase bucketed by shape-class: groups tensors by column dimension into homogeneous classes"""
    
    if not grads:
        return []
    
    # Phase 1: Group tensors by column dimension (shape[1]) into shape classes
    shape_classes = {}  # column_dim -> list of (index, tensor)
    for idx, t in enumerate(grads):
        col_dim = t.shape[1] if len(t.shape) > 1 else t.shape[0]
        if col_dim not in shape_classes:
            shape_classes[col_dim] = []
        shape_classes[col_dim].append((idx, t))
    
    # Prepare output list (preserve order)
    outs = [None] * len(grads)
    
    # Phase 2: Process each shape class separately
    for col_dim, tensor_list in shape_classes.items():
        if len(tensor_list) == 1:
            # Singleton: handle individually with direct all_reduce
            idx, t = tensor_list[0]
            outs[idx] = xm.all_reduce(xm.REDUCE_SUM, t)
        else:
            # Multiple tensors with same column dimension: stack into 3-D
            # All tensors in this class have shape [rows, col_dim]
            indices = [idx for idx, _ in tensor_list]
            tensors = [t for _, t in tensor_list]
            
            # Stack tensors along a new dimension (dimension 0)
            # Result shape: [num_tensors, rows, col_dim]
            stacked = torch.stack(tensors, dim=0)
            
            # Single all_reduce on the stacked 3-D tensor
            reduced_stack = xm.all_reduce(xm.REDUCE_SUM, stacked)
            
            # Unstack back to individual tensors
            for i, idx in enumerate(indices):
                outs[idx] = reduced_stack[i]
    
    return outs
PROBLEMS["hetero_grad_ar_many"]["cands"]["p3_pick3"] = p3v8_hetero_grad_ar_many
def p3v8_layer_grads_hetero_ar(grads, rank, ws, xm, torch):
    """Single flat concatenation all-reduce: concatenate all 6 tensors into one 1-D buffer, 
    perform a single all_reduce, then slice and reshape back (1 dispatch).
    """
    # Concatenate all gradient tensors into a single flat 1-D buffer
    flat = torch.cat([t.reshape(-1) for t in grads])
    
    # Perform single all-reduce operation
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Slice and reshape back to original shapes
    outs = []
    offset = 0
    for t in grads:
        n = t.numel()
        outs.append(reduced[offset:offset + n].view(t.shape))
        offset += n
    
    return outs
PROBLEMS["layer_grads_hetero_ar"]["cands"]["p3_pick3"] = p3v8_layer_grads_hetero_ar
def p3v8_model_grads_4layers(grads, rank, ws, xm, torch):
    """Reduce-scatter + all-gather with minimal local ops"""
    
    outs = [None] * len(grads)
    
    for layer_idx in range(4):
        start_idx = layer_idx * 6
        layer_grads = grads[start_idx:start_idx + 6]
        
        # Concatenate once (unavoidable for reduce-scatter input)
        flat = torch.cat([t.reshape(-1) for t in layer_grads])
        total_size = flat.numel()
        
        # Calculate chunk size with ceiling division
        chunk_size = (total_size + ws - 1) // ws
        padded_size = chunk_size * ws
        
        # Only pad if necessary, reuse flat when possible
        if padded_size > total_size:
            # Minimize padding overhead
            flat_padded = torch.nn.functional.pad(flat, (0, padded_size - total_size))
        else:
            flat_padded = flat
        
        # Reduce-scatter
        reduced_chunk = xm.reduce_scatter(
            xm.REDUCE_SUM,
            flat_padded,
            scale=1.0,
            scatter_dim=0,
            shard_count=ws,
            groups=None
        )
        
        # All-gather
        gathered = xm.all_gather(reduced_chunk, dim=0)
        
        # Use narrow instead of slicing to remove padding (metadata-only op)
        if padded_size > total_size:
            result = gathered.narrow(0, 0, total_size)
        else:
            result = gathered
        
        # Unpack using narrow/view (avoid indexing)
        offset = 0
        for i, t in enumerate(layer_grads):
            n = t.numel()
            # Use narrow (metadata-only) instead of slice
            chunk = result.narrow(0, offset, n)
            # Reshape is metadata-only when contiguous
            outs[start_idx + i] = chunk.reshape(t.shape)
            offset += n
    
    return outs
PROBLEMS["model_grads_4layers"]["cands"]["p3_pick3"] = p3v8_model_grads_4layers
def p3v8_layer_grads_mixed_dtype(grads, rank, ws, xm, torch):
    """
    Optimized parallel bf16-bucket and fp32-bucket strategy:
    Minimize local ops by using views and avoiding explicit unpacking loops.
    """
    # Separate gradients by dtype
    bf16_grads = grads[:4]
    fp32_grads = grads[4:]
    
    # Store shapes for later reconstruction
    bf16_shapes = [t.shape for t in bf16_grads]
    fp32_shapes = [t.shape for t in fp32_grads]
    
    # Compute offsets once
    bf16_sizes = [t.numel() for t in bf16_grads]
    fp32_sizes = [t.numel() for t in fp32_grads]
    
    # Concatenate using views (reshape(-1) is metadata-only on contiguous tensors)
    bf16_flat = torch.cat([t.reshape(-1) for t in bf16_grads])
    fp32_flat = torch.cat([t.reshape(-1) for t in fp32_grads])
    
    # Launch both all_reduce operations
    bf16_reduced = xm.all_reduce(xm.REDUCE_SUM, bf16_flat)
    fp32_reduced = xm.all_reduce(xm.REDUCE_SUM, fp32_flat)
    
    # Unpack using narrow (metadata-only view) instead of slicing
    bf16_results = []
    offset = 0
    for size, shape in zip(bf16_sizes, bf16_shapes):
        bf16_results.append(bf16_reduced.narrow(0, offset, size).view(shape))
        offset += size
    
    # Unpack fp32 using narrow
    fp32_results = []
    offset = 0
    for size, shape in zip(fp32_sizes, fp32_shapes):
        fp32_results.append(fp32_reduced.narrow(0, offset, size).view(shape))
        offset += size
    
    # Return in original order
    return bf16_results + fp32_results
PROBLEMS["layer_grads_mixed_dtype"]["cands"]["p3_pick3"] = p3v8_layer_grads_mixed_dtype

def flatten(out):
    if torch.is_tensor(out): return [out]
    res = []
    for o in out: res += flatten(o)
    return res
