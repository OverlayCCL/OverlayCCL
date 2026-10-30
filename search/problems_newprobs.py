"""New collective problems outside the paper's eight, registered in the same format as problems.py / problems_modext.py
so the search loop (experiments/run_search.py --problem <name>) can run on them. Sandbox shapes are small; the
hardware harnesses live in experiments/cross_scope_2026_10_26/scripts/g_newprobs.py (same strategies, same names).

Problems: zero1_param_ag, zero2_grad_rs, ulysses_seq_a2a, hetero_grad_ar, dp_grad_ar_groups.
"""
import torch
from .problems import CollectiveProblem, register_problem

_TAIL = "rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1"

# ------------------------------------------------------------------ ZeRO-1 parameter all-gather
def _zero1_reference(inputs, world_size):
    L = len(inputs[0]); outs = [torch.cat([inputs[r][i] for r in range(world_size)]) for i in range(L)]
    return [list(outs) for _ in range(world_size)]
def _zero1_generate(world_size, pattern="uniform", shard_size=64, L=4, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(shard_size) for _ in range(L)] for _ in range(world_size)]
    return {"per_rank_args": [{"shards": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _zero1_reference(per, world_size)}
def _zero1_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["shards"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_ZERO1_BUILTINS = {"per_tensor_ag": f'''\
def evolved_zero1_param_ag(shards, {_TAIL}):
    """developer baseline: one all_gather per parameter shard"""
    return [xm.all_gather(s, dim=0) for s in shards]
''', "flat_ag": f'''\
def evolved_zero1_param_ag(shards, {_TAIL}):
    flat = torch.cat(shards); g = xm.all_gather(flat.unsqueeze(0), dim=0); n = shards[0].numel()
    return [g[:, i * n:(i + 1) * n].reshape(-1) for i in range(len(shards))]
'''}
register_problem(CollectiveProblem(
    name="zero1_param_ag", display_name="ZeRO-1 parameter all-gather after the optimizer step",
    evolved_fn_name="evolved_zero1_param_ag",
    signature=f"def evolved_zero1_param_ag(shards, {_TAIL}):",
    signature_doc="""    Args:
        shards: list[Tensor] of L 1-D shards (this rank's slice of each flat parameter tensor; equal sizes).
    Returns: list[Tensor] of L 1-D tensors, each the rank-ordered concatenation of that shard across all ranks.""",
    reference_fn=_zero1_reference, generate_test_case=_zero1_generate, call_candidate=_zero1_call,
    builtin_templates=_ZERO1_BUILTINS,
    optimization_hints="""### Hardware constraints:
- xm.all_gather with the default pin_layout=True is lowered to a zero-padded all_reduce (2x the bytes of a native
  all-gather); pin_layout=False selects the native all-gather, which compiles only for dim-0 gathers."""))

# ------------------------------------------------------------------ ZeRO-2 gradient reduce-scatter
def _zero2_reference(inputs, world_size):
    L = len(inputs[0]); outs = []
    for r in range(world_size):
        row = []
        for i in range(L):
            s = sum(inputs[q][i] for q in range(world_size)); n = s.numel() // world_size; row.append(s[r * n:(r + 1) * n])
        outs.append(row)
    return outs
def _zero2_generate(world_size, pattern="uniform", shard_size=64, L=4, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(shard_size * world_size) * 0.01 for _ in range(L)] for _ in range(world_size)]
    return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _zero2_reference(per, world_size)}
def _zero2_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["grads"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_ZERO2_BUILTINS = {"per_tensor_rs": f'''\
def evolved_zero2_grad_rs(grads, {_TAIL}):
    """developer baseline: one reduce_scatter per gradient tensor"""
    return [xm.reduce_scatter(xm.REDUCE_SUM, t.reshape(-1), scale=1.0, scatter_dim=0, shard_count=world_size) for t in grads]
''', "flat_rs": f'''\
def evolved_zero2_grad_rs(grads, {_TAIL}):
    per = [t.numel() // world_size for t in grads]
    view = torch.cat([t.reshape(world_size, -1) for t in grads], dim=1).reshape(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=world_size)
    outs = []; off = 0
    for n in per: outs.append(r[off:off + n]); off += n
    return outs
'''}
register_problem(CollectiveProblem(
    name="zero2_grad_rs", display_name="ZeRO-2 gradient reduce-scatter",
    evolved_fn_name="evolved_zero2_grad_rs",
    signature=f"def evolved_zero2_grad_rs(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list[Tensor] of L 1-D gradient tensors (numel divisible by world_size), identical shapes on every rank.
    Returns: list[Tensor] of L 1-D tensors: this rank's shard (rank-th slice of numel/world_size) of the SUM over ranks.""",
    reference_fn=_zero2_reference, generate_test_case=_zero2_generate, call_candidate=_zero2_call,
    builtin_templates=_ZERO2_BUILTINS))

# ------------------------------------------------------------------ Ulysses sequence-parallel attention all-to-all
_U_S, _U_H, _U_D = 4, 8, 4   # sandbox: S_local, H, D (H must be divisible by world_size)
def _ulysses_reference(inputs, world_size):
    hl = _U_H // world_size; outs = []
    for d in range(world_size):
        outs.append(torch.cat([inputs[s][:, d * hl:(d + 1) * hl, :] for s in range(world_size)], dim=0))   # (S, H/ws, D)
    return outs
def _ulysses_generate(world_size, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed); H = max(_U_H, world_size)
    xs = [torch.randn(_U_S, H, _U_D) for _ in range(world_size)]
    hl = H // world_size
    expected = [torch.cat([xs[s][:, d * hl:(d + 1) * hl, :] for s in range(world_size)], dim=0) for d in range(world_size)]
    return {"per_rank_args": [{"x": xs[r]} for r in range(world_size)], "shared_args": {}, "expected": expected}
def _ulysses_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["x"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_ULYSSES_BUILTINS = {"ag_transpose_rs": f'''\
def evolved_ulysses_seq_a2a(x, {_TAIL}):
    """developer baseline (same shape as the AllToAllV baseline): all-gather, transpose to destination-major, reduce-scatter"""
    S, H, D = x.shape; hl = H // world_size
    c = x.view(S, world_size, hl, D).permute(1, 0, 2, 3).contiguous(); n = c.numel() // world_size
    g = xm.all_gather(c.reshape(-1).unsqueeze(0), dim=0).view(world_size, world_size, n).permute(1, 0, 2).contiguous().view(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, g, scale=1.0 / world_size, scatter_dim=0, shard_count=world_size)
    return r.view(world_size * S, hl, D)
''', "ag_slice": f'''\
def evolved_ulysses_seq_a2a(x, {_TAIL}):
    S, H, D = x.shape; hl = H // world_size
    c = x.view(S, world_size, hl, D).permute(1, 0, 2, 3).contiguous(); n = c.numel() // world_size
    g = xm.all_gather(c.reshape(-1).unsqueeze(0), dim=0).view(world_size, world_size, n)
    return g[:, rank, :].contiguous().view(world_size * S, hl, D)
'''}
register_problem(CollectiveProblem(
    name="ulysses_seq_a2a", display_name="Ulysses sequence-parallel attention all-to-all (sequence shard -> head shard)",
    evolved_fn_name="evolved_ulysses_seq_a2a",
    signature=f"def evolved_ulysses_seq_a2a(x, {_TAIL}):",
    signature_doc="""    Args:
        x: (S_local, H, D) tensor: this rank's sequence block with all heads. H is divisible by world_size.
    Returns: (S_local*world_size, H/world_size, D): every rank's sequence block (rank order) for this rank's head slice.""",
    reference_fn=_ulysses_reference, generate_test_case=_ulysses_generate, call_candidate=_ulysses_call,
    builtin_templates=_ULYSSES_BUILTINS,
    optimization_hints="""### Hardware constraints:
- xm.all_to_all does not compile at multi-node scale; collective_permute rings abort at 64+ ranks.
- xm.all_gather with the default pin_layout=True is lowered to a zero-padded all_reduce; pin_layout=False is native."""))

# ------------------------------------------------------------------ heterogeneous gradient all-reduce (pad-and-stack vs exact)
_HET_SIZES = (16, 32, 16, 64, 16, 128, 32, 16)
def _hetero_reference(inputs, world_size):
    outs = [sum(inputs[r][i] for r in range(world_size)) for i in range(len(inputs[0]))]
    return [list(outs) for _ in range(world_size)]
def _hetero_generate(world_size, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(n) * 0.01 for n in _HET_SIZES] for _ in range(world_size)]
    return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _hetero_reference(per, world_size)}
def _hetero_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["grads"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_HETERO_BUILTINS = {"per_tensor_ar": f'''\
def evolved_hetero_grad_ar(grads, {_TAIL}):
    """developer baseline: one all_reduce per gradient tensor"""
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
''', "flat_cat_ar": f'''\
def evolved_hetero_grad_ar(grads, {_TAIL}):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
'''}
register_problem(CollectiveProblem(
    name="hetero_grad_ar", display_name="Replicated gradient all-reduce over heterogeneous tensor sizes",
    evolved_fn_name="evolved_hetero_grad_ar",
    signature=f"def evolved_hetero_grad_ar(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list[Tensor] of gradient tensors of DIFFERENT sizes (identical shapes on every rank).
    Returns: list[Tensor]: the element-wise SUM over ranks of each tensor, same shapes, same order.""",
    reference_fn=_hetero_reference, generate_test_case=_hetero_generate, call_candidate=_hetero_call,
    builtin_templates=_HETERO_BUILTINS))

# ------------------------------------------------------------------ DP gradient all-reduce inside replica groups (TP x DP)
def _dpg_groups(ws):
    tp = max(1, ws // 4); return [[r for r in range(ws) if r % tp == s] for s in range(tp)], tp
def _dpg_reference(inputs, world_size):
    groups, tp = _dpg_groups(world_size); outs = [None] * world_size
    for g in groups:
        s = [sum(inputs[r][i] for r in g) for i in range(len(inputs[0]))]
        for r in g: outs[r] = list(s)
    return outs
def _dpg_generate(world_size, pattern="uniform", shard_size=64, L=4, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(shard_size) * 0.01 for _ in range(L)] for _ in range(world_size)]
    return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _dpg_reference(per, world_size)}
def _dpg_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["grads"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_DPG_BUILTINS = {"per_tensor_grouped_ar": f'''\
def evolved_dp_grad_ar_groups(grads, {_TAIL}):
    """developer baseline: one all_reduce per tensor over this rank's data-parallel replica group"""
    tp = max(1, world_size // 4)
    groups = [[r for r in range(world_size) if r % tp == s] for s in range(tp)]
    return [xm.all_reduce(xm.REDUCE_SUM, t, groups=groups) for t in grads]
'''}
register_problem(CollectiveProblem(
    name="dp_grad_ar_groups", display_name="Data-parallel gradient all-reduce inside replica groups (TP x DP mesh)",
    evolved_fn_name="evolved_dp_grad_ar_groups",
    signature=f"def evolved_dp_grad_ar_groups(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list[Tensor] of gradient tensors (identical shapes on every rank).
    The data-parallel group of rank r is {q : q % tp == r % tp} with tp = world_size // 4 (4 ranks per group; on a
    1-node cluster the group is intra-node, on a multi-node cluster it spans nodes).
    Returns: list[Tensor]: the SUM over the rank's data-parallel group of each tensor.""",
    reference_fn=_dpg_reference, generate_test_case=_dpg_generate, call_candidate=_dpg_call,
    builtin_templates=_DPG_BUILTINS,
    optimization_hints="""### Hardware constraints:
- Replica groups of 2 ranks fail at execution (NRT_INVALID); groups of 4 or more work.
- A world-wide collective with masking is a valid substitute for grouped collectives but moves tp x the bytes."""))


# ------------------------------------------------------------------ heterogeneous-bundle problems that invert between microbench and training step
# (2026-10-26 2-node study): the microbench lets XLA combine back-to-back all-reduces; the training graph does not.
_DM, _HID = 2048, 5376
_LAYER_SHAPES_SB = [(3 * 64, 64), (64, 64), (2 * 168, 64), (64, 168), (64,), (64,)]   # sandbox-sized layer (same aspect as DM=2048/HID=5376)
def _hetero_ar_reference(inputs, world_size):
    outs = [sum(inputs[r][i] for r in range(world_size)) for i in range(len(inputs[0]))]
    return [list(outs) for _ in range(world_size)]
def _mk_hetero_generate(shapes, dtypes=None):
    def gen(world_size, pattern="uniform", shard_size=64, seed=0):
        torch.manual_seed(seed)
        per = [[(torch.randn(*sh) * 0.01).to(dtypes[i] if dtypes else torch.float32) for i, sh in enumerate(shapes)] for _ in range(world_size)]
        return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _hetero_ar_reference(per, world_size)}
    return gen
def _grads_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["grads"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
def _ar_builtins(fn_name):
    return {"per_tensor_ar": f'''\
def {fn_name}(grads, {_TAIL}):
    """developer baseline: one all_reduce per gradient tensor (DDP hook per parameter)"""
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
''', "flat_cat_ar": f'''\
def {fn_name}(grads, {_TAIL}):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat)
    outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
''', "bucketed_ar": f'''\\
def {fn_name}(grads, {_TAIL}):
    """in-order buckets of 8 tensors each (size-independent; a few dispatches, exact bytes)"""
    outs = []
    for b in range(0, len(grads), 8):
        chunk = grads[b:b + 8]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
''', "bucketed_ar_4": f'''\\
def {fn_name}(grads, {_TAIL}):
    """in-order buckets of 4 tensors each"""
    outs = []
    for b in range(0, len(grads), 4):
        chunk = grads[b:b + 4]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
''', "bucketed_ar_16": f'''\\
def {fn_name}(grads, {_TAIL}):
    """in-order buckets of 16 tensors each"""
    outs = []
    for b in range(0, len(grads), 16):
        chunk = grads[b:b + 16]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
'''}
_HINT_BUNDLE = """### Sizes: the tensors in this sandbox are the deployment's tensors scaled down (1/64 to 1/1024 of the element
counts). Write size-independent code: bucket by tensor COUNT or by RELATIVE size (sorted order, fractions of the total),
never by hardcoded element or byte thresholds, and keep the deployment order of the tensors (do not reorder pieces
inside a bundle unless the strategy needs it).
### Measured on Trainium (2-node study):
- A lone-call microbench lets XLA combine back-to-back all_reduce calls into one; inside a training step they stay
  separate dispatches, so per-tensor all-reduce looks cheap on the bench and is 1.4-4.9x slower in the step than bundles.
- Concatenating UNEQUAL 1-D pieces into one collective can be pathologically slow (7-11x) inside the training graph while
  being fast on the bench; equal-sized stacks and per-layer buckets avoid it.
- Measured bundle orderings: a few IN-ORDER buckets of adjacent tensors (e.g. 4-8 buckets) ran at the optimum; the same
  tensors size-SORTED into buckets ran 2-3x slower; preallocate-and-copy buffers and reduce-scatter/all-gather
  decompositions of small pieces lost to simple in-order buckets."""
register_problem(CollectiveProblem(
    name="layer_grads_hetero_ar", display_name="DP gradient all-reduce of one transformer layer (6 heterogeneous tensors)",
    evolved_fn_name="evolved_layer_grads_hetero_ar", signature=f"def evolved_layer_grads_hetero_ar(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 6 gradient tensors of one transformer layer (qkv, out, up+gate, down, 2 norms), heterogeneous shapes,
            identical on every rank. Returns: list of the element-wise SUM over ranks, same shapes and order.""",
    reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate(_LAYER_SHAPES_SB), call_candidate=_grads_call,
    builtin_templates=_ar_builtins("evolved_layer_grads_hetero_ar"), optimization_hints=_HINT_BUNDLE))
register_problem(CollectiveProblem(
    name="model_grads_4layers", display_name="DP gradient all-reduce of a 4-layer transformer (24 heterogeneous tensors)",
    evolved_fn_name="evolved_model_grads_4layers", signature=f"def evolved_model_grads_4layers(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 24 gradient tensors (4 layers x 6 parameters), heterogeneous shapes, identical on every rank.
    Returns: list of the element-wise SUM over ranks, same shapes and order.""",
    reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate(_LAYER_SHAPES_SB * 4), call_candidate=_grads_call,
    builtin_templates=dict(_ar_builtins("evolved_model_grads_4layers"), per_layer_bucket=f'''\
def evolved_model_grads_4layers(grads, {_TAIL}):
    """one all_reduce per layer (6 tensors each), exact bytes"""
    outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
'''), optimization_hints=_HINT_BUNDLE))
register_problem(CollectiveProblem(
    name="layer_grads_mixed_dtype", display_name="DP gradient all-reduce of one layer with fp32 norm parameters (mixed dtypes)",
    evolved_fn_name="evolved_layer_grads_mixed_dtype", signature=f"def evolved_layer_grads_mixed_dtype(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 6 gradient tensors; the first 4 are bf16 (weights), the last 2 are fp32 (norms). Identical shapes on
            every rank. Returns: list of the element-wise SUM over ranks, same shapes, same dtypes, same order.""",
    reference_fn=_hetero_ar_reference,
    generate_test_case=_mk_hetero_generate(_LAYER_SHAPES_SB, dtypes=[torch.bfloat16] * 4 + [torch.float32] * 2), call_candidate=_grads_call,
    builtin_templates={"per_tensor_ar": f'''\
def evolved_layer_grads_mixed_dtype(grads, {_TAIL}):
    """developer baseline: one all_reduce per tensor in its native dtype"""
    return [xm.all_reduce(xm.REDUCE_SUM, t) for t in grads]
''', "cast_flat_fp32": f'''\
def evolved_layer_grads_mixed_dtype(grads, {_TAIL}):
    flat = torch.cat([t.reshape(-1).float() for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view(t.shape).to(t.dtype)); off += n
    return outs
''', "cast_flat_bf16": f'''\\
def evolved_layer_grads_mixed_dtype(grads, {_TAIL}):
    flat = torch.cat([t.reshape(-1).to(torch.bfloat16) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view(t.shape).to(t.dtype)); off += n
    return outs
''', "two_dtype_groups": f'''\\
def evolved_layer_grads_mixed_dtype(grads, {_TAIL}):
    outs = [None] * len(grads)
    for dt in (torch.bfloat16, torch.float32):
        idx = [i for i, t in enumerate(grads) if t.dtype == dt]
        if not idx: continue
        flat = torch.cat([grads[i].reshape(-1) for i in idx]); r = xm.all_reduce(xm.REDUCE_SUM, flat); off = 0
        for i in idx: n = grads[i].numel(); outs[i] = r[off:off + n].view(grads[i].shape); off += n
    return outs
'''}, optimization_hints=_HINT_BUNDLE))
# gradient-bucketing size sweep instance: 64 mixed-size gradients (sandbox 1/1024 of the hardware sizes)
import random as _rnd
_rng_hs = _rnd.Random(7); _HS_A = [_rng_hs.choice([64, 128, 256, 512]) for _ in range(64)]   # mixed sizes (one RNG, not one per draw)
register_problem(CollectiveProblem(
    name="dp_grad_bucketing_64", display_name="DP gradient all-reduce of 64 mixed-size tensors (bucketing sweep instance)",
    evolved_fn_name="evolved_dp_grad_bucketing_64", signature=f"def evolved_dp_grad_bucketing_64(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 64 1-D gradient tensors of mixed sizes (identical on every rank).
    Returns: list of the element-wise SUM over ranks, same sizes and order.""",
    reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate([(n,) for n in _HS_A]), call_candidate=_grads_call,
    builtin_templates=_ar_builtins("evolved_dp_grad_bucketing_64"), optimization_hints=_HINT_BUNDLE))
# ------------------------------------------------------------------ 2026-10-27: remaining inverting problems + the same workload under a different setup
_rng_hs_c = _rnd.Random(7); _HS_C = [_rng_hs_c.choice([256, 512, 1024, 2048]) for _ in range(32)]   # hardware sizes 256K..2M elements / 1024
register_problem(CollectiveProblem(
    name="hs_c_32mixed", display_name="DP gradient all-reduce of 32 mixed-size tensors (256K-2M elements on hardware)",
    evolved_fn_name="evolved_hs_c_32mixed", signature=f"def evolved_hs_c_32mixed(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 32 1-D gradient tensors of mixed sizes (identical on every rank).
    Returns: list of the element-wise SUM over ranks, same sizes and order.""",
    reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate([(n,) for n in _HS_C]), call_candidate=_grads_call,
    builtin_templates=_ar_builtins("evolved_hs_c_32mixed"), optimization_hints=_HINT_BUNDLE))
# 48 heterogeneous gradients whose flat concat hits the unequal-operand pathology on hardware (sandbox: 1/1024 of (1024, 1024*k))
_S8_SB = [(32, 32 * k) for k in (1, 3, 1, 5, 1, 7, 3, 1, 1, 2, 1, 3, 1, 1, 5, 1) * 3]
register_problem(CollectiveProblem(
    name="hetero_grad_ar_many", display_name="DP gradient all-reduce of 48 heterogeneous 2-D gradients (flat concat pathology instance)",
    evolved_fn_name="evolved_hetero_grad_ar_many", signature=f"def evolved_hetero_grad_ar_many(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 48 2-D gradient tensors (rows equal, columns an odd mix), identical on every rank.
    Returns: list of the element-wise SUM over ranks, same shapes and order.""",
    reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate(_S8_SB), call_candidate=_grads_call,
    builtin_templates=_ar_builtins("evolved_hetero_grad_ar_many"), optimization_hints=_HINT_BUNDLE))
# the 4-layer model gradients all-reduced inside DP groups of 4 ranks that span nodes at 64 ranks (2 per node)
def _mg_groups(ws):
    tp = max(1, ws // 4); return [[r for r in range(ws) if r % tp == s] for s in range(tp)], tp
def _mg_grouped_reference(inputs, world_size):
    groups, _ = _mg_groups(world_size); outs = [None] * world_size
    for g in groups:
        s = [sum(inputs[r][i] for r in g) for i in range(len(inputs[0]))]
        for r in g: outs[r] = list(s)
    return outs
def _mk_grouped_generate(shapes):
    def gen(world_size, pattern="uniform", shard_size=64, seed=0):
        torch.manual_seed(seed)
        per = [[(torch.randn(*sh) * 0.01) for sh in shapes] for _ in range(world_size)]
        return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _mg_grouped_reference(per, world_size)}
    return gen
_GRP = "groups = [[r for r in range(world_size) if r % max(1, world_size // 4) == s] for s in range(max(1, world_size // 4))]"
register_problem(CollectiveProblem(
    name="model_grads_ep_groups", display_name="DP gradient all-reduce of a 4-layer transformer inside 4-rank DP groups (TP=world/4)",
    evolved_fn_name="evolved_model_grads_ep_groups", signature=f"def evolved_model_grads_ep_groups(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 24 gradient tensors (4 layers x 6 parameters), heterogeneous shapes. The reduction runs inside DP
            groups of 4 ranks: ranks with equal (rank % (world_size // 4)) form a group (pass groups=... to xm.all_reduce).
    Returns: list of the element-wise SUM over the caller's group, same shapes and order.""",
    reference_fn=_mg_grouped_reference, generate_test_case=_mk_grouped_generate(_LAYER_SHAPES_SB * 4), call_candidate=_grads_call,
    builtin_templates={"per_tensor_grouped": f'''\
def evolved_model_grads_ep_groups(grads, {_TAIL}):
    """developer baseline: one grouped all_reduce per gradient tensor"""
    {_GRP}
    return [xm.all_reduce(xm.REDUCE_SUM, t, groups=groups) for t in grads]
''', "per_layer_grouped": f'''\
def evolved_model_grads_ep_groups(grads, {_TAIL}):
    {_GRP}
    outs = []
    for b in range(0, len(grads), 6):
        chunk = grads[b:b + 6]; flat = torch.cat([t.reshape(-1) for t in chunk]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=groups); off = 0
        for t in chunk: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
''', "flat_grouped": f'''\
def evolved_model_grads_ep_groups(grads, {_TAIL}):
    {_GRP}
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=groups); outs = []; off = 0
    for t in grads: n = t.numel(); outs.append(r[off:off + n].view(t.shape)); off += n
    return outs
'''}, optimization_hints=_HINT_BUNDLE))
# ------------------------------------------------------------------ 2026-10-27: the three probe-derived 1-D inversions (sizes = hardware sizes / 1024, seed 11 draw order)
def _probe_sizes(n, choices, seed=11):
    rng = _rnd.Random(seed); return [rng.choice(choices) for _ in range(n)]
for _n, _lab, _ch in ((24, "mid", [1024, 2048, 4096, 8192]), (48, "mid", [1024, 2048, 4096, 8192]), (48, "small", [64, 128, 256, 512])):
    _pname = f"probe_het_{_n}_{_lab}"
    register_problem(CollectiveProblem(
        name=_pname, display_name=f"DP gradient all-reduce of {_n} mixed-size 1-D tensors ({_lab} sizes)",
        evolved_fn_name=f"evolved_{_pname}", signature=f"def evolved_{_pname}(grads, {_TAIL}):",
        signature_doc=f"""    Args:
        grads: list of {_n} 1-D gradient tensors of mixed sizes (identical on every rank).
    Returns: list of the element-wise SUM over ranks, same sizes and order.""",
        reference_fn=_hetero_ar_reference, generate_test_case=_mk_hetero_generate([(k,) for k in _probe_sizes(_n, _ch)]), call_candidate=_grads_call,
        builtin_templates=_ar_builtins(f"evolved_{_pname}"), optimization_hints=_HINT_BUNDLE))
# ------------------------------------------------------------------ 2026-10-28: ZeRO-2 heterogeneous reduce-scatter (hardware: cat along dim 1 then RS is 11x slower in training, 1.14x faster at the 64-rank bench)
_Z2H_SIZES = [12288, 4096, 21504, 10752, 64, 64]   # hardware sizes / 1024 (vectors rounded up to 64 so every piece divides the scoring world size) (3DMxDM, DMxDM, 2HIDxDM, DMxHID, two DM-vectors), all divisible by the sandbox world size
def _z2h_reference(inputs, world_size):
    outs = []
    for r in range(world_size):
        row = []
        for i in range(len(inputs[0])):
            s = sum(inputs[q][i] for q in range(world_size)); n = s.numel() // world_size; row.append(s[r * n:(r + 1) * n])
        outs.append(row)
    return outs
def _z2h_generate(world_size, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(n) * 0.01 for n in _Z2H_SIZES] for _ in range(world_size)]
    return {"per_rank_args": [{"grads": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _z2h_reference(per, world_size)}
register_problem(CollectiveProblem(
    name="zero2_hetero_rs", display_name="ZeRO-2 reduce-scatter of 6 heterogeneous 1-D gradients (each rank keeps its shard)",
    evolved_fn_name="evolved_zero2_hetero_rs", signature=f"def evolved_zero2_hetero_rs(grads, {_TAIL}):",
    signature_doc="""    Args:
        grads: list of 6 1-D gradient tensors of heterogeneous sizes (each divisible by world_size), identical on every rank.
    Returns: list of 6 tensors; the i-th is the caller's rank-th contiguous shard (size numel/world_size) of the SUM over ranks.""",
    reference_fn=_z2h_reference, generate_test_case=_z2h_generate, call_candidate=_grads_call,
    builtin_templates={"per_tensor_rs": f'''\
def evolved_zero2_hetero_rs(grads, {_TAIL}):
    """developer baseline: one reduce_scatter per gradient tensor"""
    return [xm.reduce_scatter(xm.REDUCE_SUM, t, scale=1.0, scatter_dim=0, shard_count=world_size) for t in grads]
''', "cat_dim1_rs": f'''\
def evolved_zero2_hetero_rs(grads, {_TAIL}):
    """one reduce_scatter over the dim-1 concatenation of the (world_size, -1) views"""
    per = [t.numel() // world_size for t in grads]
    view = torch.cat([t.reshape(world_size, -1) for t in grads], dim=1).reshape(-1)
    r = xm.reduce_scatter(xm.REDUCE_SUM, view, scale=1.0, scatter_dim=0, shard_count=world_size); outs = []; off = 0
    for n in per: outs.append(r[off:off + n]); off += n
    return outs
''', "flat_ar_then_slice": f'''\
def evolved_zero2_hetero_rs(grads, {_TAIL}):
    flat = torch.cat([t.reshape(-1) for t in grads]); r = xm.all_reduce(xm.REDUCE_SUM, flat); outs = []; off = 0
    for t in grads: n = t.numel() // world_size; outs.append(r[off + rank * n: off + (rank + 1) * n]); off += t.numel()
    return outs
'''}, optimization_hints=_HINT_BUNDLE))

# ------------------------------------------------------------------ FSDP parameter prefetch: all-gather LAYOUT (2026-10-29 hunt)
# Three sharded weights per layer (two column shards gathered along dim 1, one row shard gathered along dim 0) for two layers.
# On Neuron, xm.all_gather with the default pin_layout=True is lowered as a zero-padded all-reduce of the output buffer; the
# native lowering (pin_layout=False) only exists for a gather along the leading dimension.
_FSDPL_SHAPES = [(2048, 24), (2048, 24), (24, 2048)] * 2
def _fsdpl_reference(inputs, world_size):
    outs = []
    for i, sh in enumerate(_FSDPL_SHAPES):
        dim = 1 if sh[0] == 2048 else 0
        outs.append(torch.cat([inputs[r][i] for r in range(world_size)], dim=dim))
    return [list(outs) for _ in range(world_size)]
def _fsdpl_generate(world_size, pattern="uniform", shard_size=64, seed=0):
    torch.manual_seed(seed)
    per = [[torch.randn(*sh) * 0.01 for sh in _FSDPL_SHAPES] for _ in range(world_size)]
    return {"per_rank_args": [{"shards": per[r]} for r in range(world_size)], "shared_args": {}, "expected": _fsdpl_reference(per, world_size)}
def _fsdpl_call(fn, ra, sa, rank, ws, nd, cpd, xm, torch_, num_nodes=1):
    return fn(ra["shards"], rank, ws, nd, cpd, xm, torch_, num_nodes=num_nodes)
_FSDPL_BUILTINS = {"ag_pinned": f'''\
def evolved_fsdp_ag_layout(shards, {_TAIL}):
    """developer baseline: all_gather along the shard dimension with the default (pinned) layout"""
    return [xm.all_gather(s, dim=1 if s.shape[0] == 2048 else 0) for s in shards]
''', "ag_dim0_t": f'''\
def evolved_fsdp_ag_layout(shards, {_TAIL}):
    """gather every shard along the leading dimension (column shards transposed first), transpose back"""
    outs = []
    for s in shards:
        if s.shape[0] == 2048: outs.append(xm.all_gather(s.transpose(0, 1).contiguous(), dim=0).transpose(0, 1))
        else: outs.append(xm.all_gather(s, dim=0))
    return outs
''', "ag_dim0_t_native": f'''\
def evolved_fsdp_ag_layout(shards, {_TAIL}):
    """leading-dimension gathers with the native all-gather lowering (pin_layout=False), column shards transposed around it"""
    outs = []
    for s in shards:
        if s.shape[0] == 2048: outs.append(xm.all_gather(s.transpose(0, 1).contiguous(), dim=0, pin_layout=False).transpose(0, 1))
        else: outs.append(xm.all_gather(s, dim=0, pin_layout=False))
    return outs
'''}
register_problem(CollectiveProblem(
    name="fsdp_ag_layout", display_name="FSDP parameter prefetch: six all-gathers per two layers (layout of the gather)",
    evolved_fn_name="evolved_fsdp_ag_layout",
    signature=f"def evolved_fsdp_ag_layout(shards, {_TAIL}):",
    signature_doc="""    Args:
        shards: list of 6 per-rank weight shards: (2048, 24), (2048, 24), (24, 2048) for each of two layers.
    Returns: list of 6 gathered full weights in the same order: column shards gathered along dim 1 -> (2048, 24*world_size),
             row shards gathered along dim 0 -> (24*world_size, 2048); the rank order of the gather is the rank order.""",
    reference_fn=_fsdpl_reference, generate_test_case=_fsdpl_generate, call_candidate=_fsdpl_call,
    builtin_templates=_FSDPL_BUILTINS,
    optimization_hints="""### Target: AWS Trainium (trn1.32xlarge), 7 nodes x 32 ranks, EFA between nodes.
- xm.all_gather(t, dim, pin_layout=True) (the default) is lowered as a zero-padded all-reduce over the OUTPUT buffer; it is the
  only lowering available for a gather along a non-leading dimension.
- xm.all_gather(t, dim=0, pin_layout=False) uses the native all-gather lowering (about half the bytes on the wire) but only
  compiles for a gather along the leading dimension (dim=0); a non-leading dim with pin_layout=False fails to compile.
- Transposes/relayouts are local copies (HBM bandwidth)."""))
