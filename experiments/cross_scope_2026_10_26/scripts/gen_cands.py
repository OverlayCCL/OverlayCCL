#!/usr/bin/env python3
"""Emit the (g) candidate pool: new strategy classes (algebraic recomposition RS/AG/AR swaps, hierarchical
intra-/inter-node, chunking, pre/post-processing placement) for the eight problems, plus the production
baseline and the paper's deployed strategy as anchors. Writes cands_g/*.py and manifest_g_{local,node}.json."""
import json, os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cands_g")
os.makedirs(OUT, exist_ok=True)
NODE_DIR = "/home/ubuntu/g2/cands_g"
C = {}

# ---------------- uniform_a2a: input (ws*chunk) flat, row d = chunk for dest d; output rows = src order
C[("uniform_a2a", "rs_masked")] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    x = input_tensor.view(world_size, chunk_size)
    buf = torch.zeros(world_size, world_size, chunk_size, dtype=input_tensor.dtype, device=input_tensor.device)
    buf[:, rank, :] = x
    out = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
    return out.reshape(-1)
'''
C[("uniform_a2a", "hier_ag")] = '''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    node, local = rank // rpn, rank % rpn
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    g1 = xm.all_gather(input_tensor.unsqueeze(0), dim=0, groups=intra)
    if num_nodes == 1:
        return g1[:, rank * chunk_size:(rank + 1) * chunk_size].reshape(-1)
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    cols = torch.stack([g1[:, (n * rpn + local) * chunk_size:(n * rpn + local + 1) * chunk_size] for n in range(num_nodes)], dim=0)
    g2 = xm.all_gather(cols.unsqueeze(0), dim=0, groups=inter)
    return g2[:, node, :, :].reshape(-1)
'''
for K in (2, 4):
    C[("uniform_a2a", f"chunked_ag_{K}")] = f'''\
def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = {K}
    part = chunk_size // K
    x = input_tensor.view(world_size, chunk_size)
    outs = []
    for k in range(K):
        g = xm.all_gather(x[:, k * part:(k + 1) * part].contiguous().unsqueeze(0), dim=0)
        outs.append(g[:, rank, :])
    return torch.cat(outs, dim=1).reshape(-1)
'''

# ---------------- ring_kv: list of 1-D per-slot chunks; output per slot = rank-ordered concat
C[("ring_kv", "hier_ag")] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    outs = []
    for c in kv_chunks:
        g1 = xm.all_gather(c.unsqueeze(0), dim=0, groups=intra)
        if num_nodes == 1:
            outs.append(g1.reshape(-1)); continue
        g2 = xm.all_gather(g1.unsqueeze(0), dim=0, groups=inter)
        outs.append(g2.reshape(-1))
    return outs
'''
C[("ring_kv", "ar_masked")] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for c in kv_chunks:
        buf = torch.zeros(world_size, c.numel(), dtype=c.dtype, device=c.device)
        buf[rank] = c
        outs.append(xm.all_reduce(xm.REDUCE_SUM, buf).reshape(-1))
    return outs
'''
C[("ring_kv", "chunked_ag_4")] = '''\
def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 4
    outs = []
    for c in kv_chunks:
        part = c.numel() // K
        gs = [xm.all_gather(c[k * part:(k + 1) * part].unsqueeze(0), dim=0) for k in range(K)]
        outs.append(torch.cat(gs, dim=1).reshape(-1))
    return outs
'''

# ---------------- dxe: logits_local (T, V_local), targets (T,)
C[("dxe", "hier_ar_stacked")] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    lo, hi = rank * V_local, (rank + 1) * V_local
    local_max = logits_local.max(dim=-1).values
    gmax = xm.all_reduce(xm.REDUCE_MAX, local_max, groups=intra)
    if num_nodes > 1:
        gmax = xm.all_reduce(xm.REDUCE_MAX, gmax, groups=inter)
    sumexp = (logits_local - gmax.unsqueeze(-1)).exp().sum(dim=-1)
    in_shard = (targets >= lo) & (targets < hi)
    t = logits_local.gather(1, (targets - lo).clamp(0, V_local - 1).unsqueeze(1)).squeeze(1)
    t = torch.where(in_shard, t, torch.zeros_like(t))
    pack = torch.stack([sumexp, t], dim=1)
    g = xm.all_reduce(xm.REDUCE_SUM, pack, groups=intra)
    if num_nodes > 1:
        g = xm.all_reduce(xm.REDUCE_SUM, g, groups=inter)
    return (g[:, 0].log() + gmax - g[:, 1]).mean()
'''
C[("dxe", "ag_chunked_4")] = '''\
def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 4
    part = V_local // K
    r = targets // V_local
    m = None; s = None; tl = None
    for k in range(K):
        g = xm.all_gather(logits_local[:, k * part:(k + 1) * part].contiguous(), dim=1)
        cm = g.max(dim=-1).values
        off = targets - r * V_local - k * part
        inside = (off >= 0) & (off < part)
        col = r * part + off.clamp(0, part - 1)
        tk = g.gather(1, col.unsqueeze(1)).squeeze(1)
        if m is None:
            m = cm
            s = (g - m.unsqueeze(-1)).exp().sum(dim=-1)
            tl = torch.where(inside, tk, torch.zeros_like(tk))
        else:
            nm = torch.where(m > cm, m, cm)
            s = s * (m - nm).exp() + (g - nm.unsqueeze(-1)).exp().sum(dim=-1)
            m = nm
            tl = torch.where(inside, tk, tl)
    return (s.log() + m - tl).mean()
'''

# ---------------- pp_send_recv: activations list M of (B,S,DM); src ranks return own, dst ranks receive pair's
C[("pp_send_recv", "rs_masked_per_mb")] = '''\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    outs = []
    for m in range(M):
        a = activations[m]
        buf = torch.zeros(world_size, *a.shape, dtype=a.dtype, device=a.device)
        if stage == src_stage:
            buf[rank] = a
            buf[rank + half if src_stage == 0 else rank - half] = a
        r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
        outs.append(r[0])
    return outs
'''
C[("pp_send_recv", "rs_masked_all_mb")] = '''\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    a0 = activations[0]
    buf = torch.zeros(world_size, M, *a0.shape, dtype=a0.dtype, device=a0.device)
    if stage == src_stage:
        st = torch.stack(activations, dim=0)
        buf[rank] = st
        buf[rank + half if src_stage == 0 else rank - half] = st
    r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
    return [r[0, m] for m in range(M)]
'''
C[("pp_send_recv", "ag_gather_per_mb")] = '''\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    src = rank if stage == src_stage else (rank - half if src_stage == 0 else rank + half)
    outs = []
    for m in range(M):
        g = xm.all_gather(activations[m].unsqueeze(0), dim=0)
        outs.append(g[src])
    return outs
'''
C[("pp_send_recv", "ag_gather_all_mb")] = '''\
def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    src = rank if stage == src_stage else (rank - half if src_stage == 0 else rank + half)
    st = torch.stack(activations, dim=0)
    g = xm.all_gather(st.unsqueeze(0), dim=0)
    return [g[src, m] for m in range(M)]
'''

# ---------------- tp_mlp: partials[m][L] (B,S,DM) -> AR each
_TP_PACK = '''\
    shapes = [partials[m][L].shape for m in range(M) for L in range(N_LAYERS)]
    flat = torch.cat([partials[m][L].flatten() for m in range(M) for L in range(N_LAYERS)], dim=0)
'''
_TP_UNPACK = '''\
    outs = []; off = 0
    for m in range(M):
        row = []
        for L in range(N_LAYERS):
            n = 1
            for d in shapes[m * N_LAYERS + L]: n *= d
            row.append(r[off:off + n].reshape(shapes[m * N_LAYERS + L])); off += n
        outs.append(row)
    return outs
'''
C[("tp_mlp", "rs_ag_bundled")] = '''\
def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
''' + _TP_PACK + '''\
    n = flat.numel(); pad = (-n) % world_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)], dim=0)
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=world_size)
    r = xm.all_gather(rs, dim=0)
''' + _TP_UNPACK
C[("tp_mlp", "hier_ar_bundled")] = '''\
def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
''' + _TP_PACK + '''\
    r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=intra)
    if num_nodes > 1:
        r = xm.all_reduce(xm.REDUCE_SUM, r, groups=inter)
''' + _TP_UNPACK
C[("tp_mlp", "chunked_ar_bundled_4")] = '''\
def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 4
''' + _TP_PACK + '''\
    parts = torch.chunk(flat, K, dim=0)
    r = torch.cat([xm.all_reduce(xm.REDUCE_SUM, p) for p in parts], dim=0)
''' + _TP_UNPACK

# ---------------- fsdp_prefetch: shards[m][L] (DM, shard) -> gathered (DM, ws*shard) per (m, L)
C[("fsdp_prefetch", "ag_dim0_permute_per_mb")] = '''\
def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        L, DM, S = st.shape
        g = xm.all_gather(st.unsqueeze(0), dim=0)
        g = g.permute(1, 2, 0, 3).reshape(L, DM, world_size * S)
        outs.append([g[L_] for L_ in range(N_LAYERS)])
    return outs
'''
C[("fsdp_prefetch", "hier_ag_per_mb")] = '''\
def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        g = xm.all_gather(st, dim=-1, groups=intra)
        if num_nodes > 1:
            g = xm.all_gather(g, dim=-1, groups=inter)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
'''
C[("fsdp_prefetch", "chunked_ag_2_per_mb")] = '''\
def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        h = st.shape[1] // 2
        g0 = xm.all_gather(st[:, :h, :].contiguous(), dim=-1)
        g1 = xm.all_gather(st[:, h:, :].contiguous(), dim=-1)
        g = torch.cat([g0, g1], dim=1)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
'''

# ---------------- llama_block_ar: two (B,S,DM) partials -> two full sums
C[("llama_block_ar", "rs_ag_stack")] = '''\
def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([attn_partial, mlp_partial], dim=0)
    flat = st.reshape(-1)
    n = flat.numel(); pad = (-n) % world_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)], dim=0)
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=world_size)
    r = xm.all_gather(rs, dim=0)[:n].reshape(st.shape)
    return [r[0], r[1]]
'''
C[("llama_block_ar", "hier_ar_stack")] = '''\
def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
    st = torch.stack([attn_partial, mlp_partial], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, st, groups=intra)
    if num_nodes > 1:
        r = xm.all_reduce(xm.REDUCE_SUM, r, groups=inter)
    return [r[0], r[1]]
'''

# ---------------- anchors: the paper's deployed strategies (from the 224-rank pool) and the production baselines
V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cluster_224rank", "scripts", "cands_v2")
ANCH = {"uniform_a2a": "uniform_a2a__g_extract-view_col_structure-ag_slice.py",
        "ring_kv": "ring_kv__g_bundle-per_slot.py",
        "dxe": "dxe__g_structure-max_plus_stacked_sum.py",
        "pp_send_recv": "pp_send_recv__g_bundle-all_mb_pack-stack.py",
        "tp_mlp": "tp_mlp__g_bundle-all_pack-flatten.py",
        "fsdp_prefetch": "fsdp_prefetch__g_bundle-per_mb.py",
        "llama_block_ar": "llama_block_ar__g_bundle-one_ar_pack-stack.py"}
for p, f in ANCH.items():
    C[(p, "paper")] = open(os.path.join(V2, f)).read()
# extra anchors that the new classes are compared against
C[("pp_send_recv", "ar_masked_per_mb")] = open(os.path.join(V2, "pp_send_recv__g_bundle-per_mb_pack-stack.py")).read()
C[("tp_mlp", "ar_per_mb")] = open(os.path.join(V2, "tp_mlp__g_bundle-per_mb_pack-flatten.py")).read()
C[("fsdp_prefetch", "ag_all")] = open(os.path.join(V2, "fsdp_prefetch__g_bundle-all.py")).read()

man_local, man_node = [], []
for (p, name), code in C.items():
    fn = f"{p}__{name}.py"
    open(os.path.join(OUT, fn), "w").write(code)
    man_local.append({"problem": p, "cand": name, "kind": "code", "code": os.path.join(OUT, fn)})
    man_node.append({"problem": p, "cand": name, "kind": "code", "code": f"{NODE_DIR}/{fn}"})
for p in ("uniform_a2a", "ring_kv", "dxe", "pp_send_recv", "tp_mlp", "fsdp_prefetch", "llama_block_ar"):
    man_local.append({"problem": p, "cand": "harness_baseline", "kind": "baseline"})
    man_node.append({"problem": p, "cand": "harness_baseline", "kind": "baseline"})
json.dump(man_local, open(os.path.join(os.path.dirname(OUT), "manifest_g_local.json"), "w"), indent=1)
json.dump(man_node, open(os.path.join(os.path.dirname(OUT), "manifest_g_node.json"), "w"), indent=1)
print(len(C), "candidate files;", len(man_local), "manifest entries ->", OUT)
