"""amp7: amp6 with generic PROBLEM SITES (the new problems' gradient sets as replicated parameter groups; developer baseline = each
problem's microbench-winning strategy synced per microbatch, OverlayCCL = per-tensor once per step). Original amp6 docstring:
End-to-end Llama shell (amp3's four model-extension sites: PP transfer, TP MLP all-reduce, FSDP prefetch, vocab dxe) PLUS
two data-parallel gradient-sync sites that carry, exactly, the per-problem gradient sets of the two new problems whose
winners add an ingredient beyond bundling:

  site A (layer_grads_mixed_dtype): per layer a replicated Llama sub-block at DM=2048 / HID=5376 whose parameters are the
         per-problem set  qkv (3DM x DM) bf16, o (DM x DM) bf16, gate_up (2HID x DM) bf16, down (DM x HID) bf16,
         two RMSNorm gains (DM) fp32.  Winner (simulator-decides, v11): one flat bf16 bundle per layer (cast_flat_bf16).
  site B (probe_het_48_small):       48 replicated 1-D vectors of 64K/128K/256K/512K elements (the per-problem size list).
         Winner (simulator-decides after the correctness gate removes a non-running first pick): in-order 8-piece buckets.

The embedding and the LM-head shard are replicated too and are synced per tensor in every arm. Arms:
  per_mb  : baseline. Per-microbatch graphs at the four amp3 sites (as in amp3/amp5) and textbook-DDP gradient sync:
            every microbatch's backward all-reduces each replicated gradient per tensor in its native dtype
            (BASE_SYNC=per_mb, the default); BASE_SYNC=once reproduces amp5's convention (one per-tensor sync per step).
  bundled : OverlayCCL. The four sites as in amp3 (one graph for all microbatches) and ONE gradient sync per step with
            GRAD_SYNC = per_tensor | sites | nosim | flat | bf16_flat | buckets_<k>.
              sites : site A cast_flat_bf16 per layer, site B 8-piece in-order buckets (the two per-problem winners)
              nosim : the no-simulator Phase-3 winners (site A: 3 in-order pairs with dtype preserved; site B: one flat concat)
Knobs (KEY=VALUE argv tokens): N_LAYERS_PER_STAGE (2), N_MB (4), T_A (64 tokens fed to the site-A block), SYNC_GRAPH
(fused | isolated: mark_step before the sync | isolated2: mark_step before and after the sync), SYNC_REPEAT (1),
HID_PER_RANK (24), VOCAB_PER_RANK (32), BASE_SYNC (per_mb | once).
"""
import os, sys, time, json, statistics, random
import time as _t
import argparse as _ap
_p = _ap.ArgumentParser(); _p.add_argument('--backend', default='per_mb'); _p.add_argument('--steps', type=int, default=1000)
_p.add_argument('--warmup', type=int, default=5); _p.add_argument('--seed', type=int, default=0)
_a, _rest = _p.parse_known_args(sys.argv[1:])
for _tok in _rest:
    if '=' in _tok: _k, _v = _tok.split('=', 1); os.environ[_k] = _v
sys.argv = [sys.argv[0], _a.backend, str(_a.steps), str(_a.seed)]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

sys.path.insert(0, '/home/ubuntu/OverlayCCL')
from training import _percall_probe as probe

DM = 2048
HIDR = 5376                      # replicated sub-block hidden size = the per-problem layer set's HID_
T_A = int(os.environ.get('T_A', '64'))
HID = int(os.environ.get('HID', '0')) or None
N_LAYERS_PER_STAGE = int(os.environ.get('N_LAYERS_PER_STAGE', '2'))
N_MB = int(os.environ.get('N_MB', '4'))
GRAD_SYNC = os.environ.get('GRAD_SYNC', 'per_tensor')
BASE_SYNC = os.environ.get('BASE_SYNC', 'per_mb')
BASE_STRAT = os.environ.get('BASE_STRAT', 'microbench')   # strategy of the developer baseline at the problem sites
SYNC_GRAPH = os.environ.get('SYNC_GRAPH', 'isolated')
SYNC_REPEAT = int(os.environ.get('SYNC_REPEAT', '1'))
B = 1
S = int(os.environ.get('S', '1024'))
VOCAB = None
LR = 1e-4
WARMUP = 5
STEADY_FROM = 200


def _mixed_sizes(n, choices, seed=7):
    rng = random.Random(seed); return [rng.choice(choices) for _ in range(n)]
SITE_B_SIZES = _mixed_sizes(48, [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024], 11)   # = probe_het_48_small


def main():
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed)
    dev = xm.xla_device()
    ws = int(os.environ.get('WORLD_SIZE', xr.world_size()))
    rank = xr.global_ordinal()
    global HID, VOCAB
    HID = HID or int(os.environ.get('HID_PER_RANK', '24')) * ws; VOCAB = int(os.environ.get('VOCAB_PER_RANK', '32')) * ws
    half = ws // 2
    stage = 0 if rank < half else 1
    pair_id = rank if stage == 0 else rank - half
    assert HID % ws == 0 and VOCAB % ws == 0
    shard_hid = HID // ws
    shard_vocab = VOCAB // ws

    embed = nn.Embedding(VOCAB, DM, dtype=torch.bfloat16).to(dev)
    w_gate = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01) for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_up = nn.ParameterList([nn.Parameter(torch.randn(DM, shard_hid, dtype=torch.bfloat16) * 0.01) for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    w_down = nn.ParameterList([nn.Parameter(torch.randn(shard_hid, DM, dtype=torch.bfloat16) * 0.01) for _ in range(N_LAYERS_PER_STAGE)]).to(dev)
    lm_head_shard = nn.Parameter(torch.randn(DM, shard_vocab, dtype=torch.bfloat16, device=dev) * 0.01)

    # ---- problem sites: each named problem's gradient set becomes a replicated parameter group of the model (amp7).
    # PROBLEM_SITES=comma list of g_newprobs problem names (default: the seven on which the original simulator picks per-tensor,
    # minus zero2). The pieces are used the way g_train_v10 uses them (2-D as nn.Linear-style weights, DM-sized vectors as
    # RMSNorm gains, other vectors through the mean hidden state) in one pass over the hidden state after the layer loop.
    import importlib.util as _ilu, math as _math
    _spec = _ilu.spec_from_file_location("g_newprobs", "/home/ubuntu/OverlayCCL/training/g_newprobs.py"); G_NP = _ilu.module_from_spec(_spec); _spec.loader.exec_module(G_NP)
    PROBLEM_SITES = [s for s in os.environ.get('PROBLEM_SITES', 'layer_grads_hetero_ar,layer_grads_mixed_dtype,hs_a_64x64k_512k,hs_c_32x256k_2m,probe_het_48_small').split(',') if s]   # the five per-tensor sets that compile quickly in the shell (model/hetero-many/mid probes stay per-problem only)
    MICROBENCH_WINNER = {"layer_grads_hetero_ar": "flat_cat_ar", "model_grads_4layers": "flat_cat_ar", "probe_het_24_mid": "bucket6", "probe_het_48_mid": "bucket6",
                         "hetero_grad_ar_many": "p3_pick3", "hs_c_32x256k_2m": "bucket_4mb", "layer_grads_mixed_dtype": "cast_flat_bf16", "hs_a_64x64k_512k": "bucket_4mb", "probe_het_48_small": "bucket6"}
    for k_, v_ in (tok.split('=') for tok in os.environ.get('MICROBENCH_WINNER', '').split(',') if '=' in tok): MICROBENCH_WINNER[k_] = v_
    SITES = len(PROBLEM_SITES) > 0
    SITEA_LAYOUT = 'n/a'
    sites = []   # list of dicts: name, params (nn.Parameters), template, rest
    torch.manual_seed(1234)
    for pn in PROBLEM_SITES:
        P_ = G_NP.PROBLEMS[pn]; args_ = P_["make"](ws, rank, dev); tmpl = args_[0]; tens = G_NP.flatten(tmpl)
        prms = [nn.Parameter((torch.randn(*t.shape, dtype=t.dtype) * (1.0 if t.dim() == 1 and t.numel() == DM else 0.01)).to(dev)) for t in tens]
        sites.append(dict(name=pn, params=prms, template=tmpl, rest=tuple(args_[1:]), cands=P_["cands"]))
    torch.manual_seed(seed)
    def _rebuild(tmpl, it):
        if torch.is_tensor(tmpl): return next(it)
        return [_rebuild(x, it) for x in tmpl]
    def rmsnorm(x, g):
        xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xn * g.float()).to(torch.bfloat16)
    def _fit_width(x, b):
        if b == DM: return x
        if b < DM: return x[..., :b]
        return x.repeat(*([1] * (x.dim() - 1)), _math.ceil(b / DM))[..., :b]
    def problem_sites_apply(x):
        """every piece of every problem site touches the hidden state once: 2-D pieces as nn.Linear weights (gradient layout from
        the matmul backward), DM-vectors as RMSNorm gains, other vectors through an elementwise term scaled by the mean hidden state"""
        xs_ = x.float().mean()
        for st in sites:
            extra = 0.0
            for w in st["params"]:
                if w.dim() == 2:
                    o, i = w.shape
                    y = torch.matmul(_fit_width(x, i).to(w.dtype), w.t()).to(torch.bfloat16)
                    x = x + 0.01 * (y[..., :DM] if o >= DM else y.repeat(*([1] * (y.dim() - 1)), _math.ceil(DM / o))[..., :DM])
                elif w.numel() == DM:
                    x = rmsnorm(x, w)
                else:
                    extra = extra + ((w.float() * xs_) ** 2).mean()
            if torch.is_tensor(extra): x = x + (1e-3 * extra).to(torch.bfloat16)
        return x
    def site_a_block(x, L): return x
    def site_b_apply(x): return problem_sites_apply(x)
    site_params = [p_ for st in sites for p_ in st["params"]]
    rep_params = [embed.weight, lm_head_shard] + site_params
    N_EMB = 2; N_A = 0; N_B = len(site_params)
    sharded_params = list(w_gate) + list(w_up) + list(w_down)
    all_params = rep_params + sharded_params

    _agent_avail = True
    try:
        from runtime.trainium_tp_mlp_7node import evolved_tp_mlp, init_tp_mlp
        from runtime.trainium_pp_send_recv_7node import evolved_pp_send_recv, init_pp_send_recv
        from runtime.trainium_dxe_7node import dxe_loss, init_dxe
        init_tp_mlp(); init_pp_send_recv(); init_dxe()
    except Exception as _e:
        _agent_avail = False
        if rank == 0: print(f"[init] agent runtimes unavailable: {_e}; bundled mode will fall back to inline equivalents", flush=True)

    inputs = [torch.randint(0, VOCAB, (B, S), dtype=torch.int64).to(dev) for _ in range(N_MB)]
    targets = [torch.randint(0, VOCAB, (B, S), dtype=torch.int64).to(dev) for _ in range(N_MB)]

    FSDP_AG = os.environ.get('FSDP_AG', 'pinned')
    def ag_col(w):   # gather a column shard (DM, shard) -> (DM, HID)
        if FSDP_AG == 'dim0_t_native': return xm.all_gather(w.t().contiguous(), dim=0, pin_layout=False).t()
        if FSDP_AG == 'dim0_t': return xm.all_gather(w.t().contiguous(), dim=0).t()
        return xm.all_gather(w, dim=1)
    def ag_row(w):   # gather a row shard (shard, DM) -> (HID, DM)
        if FSDP_AG == 'dim0_t_native': return xm.all_gather(w, dim=0, pin_layout=False)
        return xm.all_gather(w, dim=0)
    def tp_fsdp_block(x, L):
        x = site_a_block(x, L)
        if probe.in_window():
            _t0 = _t.time()
            w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
            _ = w_gate_full.float().sum().item()
            probe.record("fsdp_prefetch", (_t.time() - _t0) * 1000.0)
        else:
            w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
        h = F.silu(torch.matmul(x, w_gate_full)) * torch.matmul(x, w_up_full)
        partial = torch.matmul(h, w_down_full)
        if probe.in_window():
            _t0 = _t.time()
            ared = xm.all_reduce(xm.REDUCE_SUM, partial)
            _ = ared.float().sum().item()
            probe.record("tp_mlp", (_t.time() - _t0) * 1000.0)
        else:
            ared = xm.all_reduce(xm.REDUCE_SUM, partial)
        return x + ared / ws

    def transfer(act, src_stage):
        buf = torch.zeros(half, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone(); buf[pair_id] = act
        return xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id]
    def transfer_batched(acts, src_stage):
        M = acts.shape[0]
        buf = torch.zeros(half, M, B, S, DM, device=dev, dtype=torch.bfloat16)
        if stage == src_stage:
            buf = buf.clone(); buf[pair_id] = acts
        return xm.all_reduce(xm.REDUCE_SUM, buf)[pair_id]

    def vocab_parallel_loss(h, tgt):
        logits = torch.matmul(h, lm_head_shard).float()
        sum_exp_local = torch.exp(logits).sum(dim=-1)
        sum_exp = xm.all_reduce(xm.REDUCE_SUM, sum_exp_local)
        rank_lo = rank * shard_vocab; rank_hi = rank_lo + shard_vocab
        local_tgt_mask = (tgt >= rank_lo) & (tgt < rank_hi)
        local_tgt_idx = (tgt - rank_lo).clamp(0, shard_vocab - 1)
        target_logit_local = torch.where(local_tgt_mask, logits.gather(-1, local_tgt_idx.unsqueeze(-1)).squeeze(-1), torch.zeros_like(sum_exp))
        target_logit = xm.all_reduce(xm.REDUCE_SUM, target_logit_local)
        return (torch.log(sum_exp + 1e-20) - target_logit).mean()
    def vocab_parallel_loss_bundled(h_all, targets_stacked):
        logits = torch.matmul(h_all, lm_head_shard).float()
        logits_flat = logits.reshape(-1, shard_vocab); tgt_flat = targets_stacked.reshape(-1)
        if _agent_avail:
            return dxe_loss(logits_flat, tgt_flat, shard_vocab)
        local_max = logits_flat.max(dim=-1).values
        global_max = xm.all_reduce(xm.REDUCE_MAX, local_max)
        shifted = logits_flat - global_max.unsqueeze(-1)
        local_sum_exp = shifted.exp().sum(dim=-1)
        lo, hi = rank * shard_vocab, (rank + 1) * shard_vocab
        target_local = tgt_flat - lo; in_shard = (tgt_flat >= lo) & (tgt_flat < hi)
        target_local_safe = target_local * in_shard.long()
        local_target = logits_flat.gather(1, target_local_safe.unsqueeze(1)).squeeze(1) * in_shard.to(logits_flat.dtype)
        packed = torch.stack([local_sum_exp, local_target], dim=0)
        g_packed = xm.all_reduce(xm.REDUCE_SUM, packed)
        return (g_packed[0].log() + global_max - g_packed[1]).mean()

    # ---------------- gradient-sync strategies (operate on the list of replicated gradients in rep_params order) ----------------
    def _slice_back(r, ts):
        outs = []; off = 0
        for t in ts: n = t.numel(); outs.append(r[off:off + n].view_as(t).to(t.dtype)); off += n
        return outs
    def sync_per_tensor(grads): return [xm.all_reduce(xm.REDUCE_SUM, g) for g in grads]
    def bundle(ts, cast=None):
        """one all-reduce over the concatenation of ts (cast: wire dtype; None = upcast to fp32 if dtypes differ)"""
        if cast is not None: pieces = [t.reshape(-1).to(cast) for t in ts]
        elif len({t.dtype for t in ts}) == 1: pieces = [t.reshape(-1) for t in ts]
        else: pieces = [t.reshape(-1).float() for t in ts]
        return _slice_back(xm.all_reduce(xm.REDUCE_SUM, torch.cat(pieces)), ts)
    def by_groups(grads, groups, cast=None):
        out = [None] * len(grads)
        for idx in groups:
            res = bundle([grads[i] for i in idx], cast)
            for i, r in zip(idx, res): out[i] = r
        return out
    def sync_sites(grads, which):
        """which: 'per_tensor' | 'microbench' (each problem's microbench winner) | '<candidate name>' applied to every site"""
        out = sync_per_tensor(grads[:N_EMB]); off = N_EMB
        for st in sites:
            n = len(st["params"]); gs = grads[off:off + n]; off += n
            cname = 'per_tensor_ar' if which == 'per_tensor' else (MICROBENCH_WINNER[st["name"]] if which == 'microbench' else ('nosim_1' if which == 'nosim' else which))
            fn_ = st["cands"][cname]
            outs = G_NP.flatten(fn_(*((_rebuild(st["template"], iter(gs)),) + st["rest"]), rank, ws, xm, torch))
            out += [o.to(g.dtype).view_as(g) for o, g in zip(outs, gs)]
        return out
    def grad_sync(grads): return sync_sites(grads, GRAD_SYNC)
    def grads_of(params):
        gs = []
        for p in params:
            gs.append(p.grad if p.grad is not None else torch.zeros_like(p))
        return gs
    # GRAD_BUFFER=flat: the replicated gradients live in one contiguous per-dtype buffer (Megatron-LM / DeepSpeed layout);
    # .grad is a view of the buffer, autograd accumulates into it, and the sync strategies operate on those views.
    GRAD_BUFFER = os.environ.get('GRAD_BUFFER', 'separate')
    _gb = {}; _goff = []
    for p_ in rep_params:
        _goff.append((p_.dtype, _gb.get(p_.dtype, 0))); _gb[p_.dtype] = _gb.get(p_.dtype, 0) + p_.numel()
    _gb = {dt: torch.zeros(n, dtype=dt, device=dev) for dt, n in _gb.items()} if GRAD_BUFFER == 'flat' else {}
    def reset_grads():
        for p_ in all_params: p_.grad = None
        if GRAD_BUFFER == 'flat':
            for b in _gb.values(): b.zero_()
            for p_, (dt, off) in zip(rep_params, _goff): p_.grad = _gb[dt][off:off + p_.numel()].view_as(p_)

    def step_per_mb(s):
        """baseline: per-microbatch graphs at the four sites + textbook-DDP per-microbatch per-tensor gradient sync"""
        reset_grads()
        acc = [None] * len(rep_params)
        loss_total = 0.0
        for m in range(N_MB):
            if stage == 0:
                h = embed(inputs[m]).to(torch.bfloat16)
                for L in range(N_LAYERS_PER_STAGE):
                    if probe.in_window():
                        _t0 = _t.time(); h = tp_fsdp_block(h, L); _ = h.float().sum().item(); probe.record("tp_fsdp", (_t.time() - _t0) * 1000.0)
                    else:
                        h = tp_fsdp_block(h, L)
                h = site_b_apply(h)
            else:
                h = torch.zeros(B, S, DM, device=dev, dtype=torch.bfloat16)
            if probe.in_window():
                _t0 = _t.time(); h_t = transfer(h, src_stage=0); _ = h_t.float().sum().item(); probe.record("pp_send_recv", (_t.time() - _t0) * 1000.0)
            else:
                h_t = transfer(h, src_stage=0)
            if stage == 1:
                h2 = h_t
                for L in range(N_LAYERS_PER_STAGE):
                    if probe.in_window():
                        _t0 = _t.time(); h2 = tp_fsdp_block(h2, L); _ = h2.float().sum().item(); probe.record("tp_fsdp", (_t.time() - _t0) * 1000.0)
                    else:
                        h2 = tp_fsdp_block(h2, L)
                h2 = site_b_apply(h2)
                if probe.in_window():
                    _t0 = _t.time(); loss = vocab_parallel_loss(h2, targets[m]); _ = loss.float().sum().item(); probe.record("vocab_dxe", (_t.time() - _t0) * 1000.0)
                else:
                    loss = vocab_parallel_loss(h2, targets[m])
                loss_total = loss_total + loss.detach()
                (loss / N_MB).backward()
            else:
                (h.sum() * 0).backward()   # stage 0: zero-weighted backward through its layers so every replicated parameter has a gradient
            if BASE_SYNC == 'per_mb':
                if SYNC_GRAPH in ('isolated', 'isolated2'): xm.mark_step()
                with torch.no_grad():
                    gs = grads_of(rep_params)
                    for _r in range(SYNC_REPEAT): synced = sync_sites(gs, BASE_STRAT)
                    for i, g in enumerate(synced): acc[i] = g if acc[i] is None else acc[i] + g
                    reset_grads()
                if SYNC_GRAPH == 'isolated2': xm.mark_step()
            xm.mark_step()
        if BASE_SYNC == 'once':
            if SYNC_GRAPH in ('isolated', 'isolated2'): xm.mark_step()
            with torch.no_grad():
                gs = grads_of(rep_params)
                for _r in range(SYNC_REPEAT): acc = sync_sites(gs, BASE_STRAT)
            if SYNC_GRAPH == 'isolated2': xm.mark_step()
        return (loss_total if stage == 1 else None), acc

    def step_bundled(s):
        """OverlayCCL: one graph for all microbatches at the four sites; one gradient sync per step with GRAD_SYNC"""
        reset_grads()
        if stage == 0:
            h_all = torch.stack([embed(inputs[m]).to(torch.bfloat16) for m in range(N_MB)], dim=0)
            for L in range(N_LAYERS_PER_STAGE):
                h_all = site_a_block(h_all, L)
                if probe.in_window():
                    _t0 = _t.time()
                    w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
                    _ = w_gate_full.float().sum().item(); probe.record("fsdp_prefetch_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
                h = F.silu(torch.matmul(h_all, w_gate_full)) * torch.matmul(h_all, w_up_full)
                partial = torch.matmul(h, w_down_full)
                if probe.in_window():
                    _t0 = _t.time(); ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws; _ = ared.float().sum().item(); probe.record("tp_mlp_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                h_all = h_all + ared
            h_all = site_b_apply(h_all)
        else:
            h_all = torch.zeros(N_MB, B, S, DM, device=dev, dtype=torch.bfloat16)
        if probe.in_window():
            _t0 = _t.time(); h_all = transfer_batched(h_all, src_stage=0); _ = h_all.float().sum().item(); probe.record("pp_send_recv", (_t.time() - _t0) * 1000.0)
        else:
            h_all = transfer_batched(h_all, src_stage=0)
        loss_total = None
        if stage == 1:
            for L in range(N_LAYERS_PER_STAGE):
                h_all = site_a_block(h_all, L)
                if probe.in_window():
                    _t0 = _t.time()
                    w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
                    _ = w_gate_full.float().sum().item(); probe.record("fsdp_prefetch_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    w_gate_full = ag_col(w_gate[L]); w_up_full = ag_col(w_up[L]); w_down_full = ag_row(w_down[L])
                h = F.silu(torch.matmul(h_all, w_gate_full)) * torch.matmul(h_all, w_up_full)
                partial = torch.matmul(h, w_down_full)
                if probe.in_window():
                    _t0 = _t.time(); ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws; _ = ared.float().sum().item(); probe.record("tp_mlp_bundled", (_t.time() - _t0) * 1000.0)
                else:
                    ared = xm.all_reduce(xm.REDUCE_SUM, partial) / ws
                h_all = h_all + ared
            h_all = site_b_apply(h_all)
            targets_stacked = torch.stack(targets, dim=0)
            if probe.in_window():
                _t0 = _t.time(); loss_total = vocab_parallel_loss_bundled(h_all, targets_stacked); _ = loss_total.float().sum().item(); probe.record("vocab_dxe_bundled", (_t.time() - _t0) * 1000.0)
            else:
                loss_total = vocab_parallel_loss_bundled(h_all, targets_stacked)
            loss_total.backward()
        else:
            (h_all.sum() * 0).backward()
        if SYNC_GRAPH in ('isolated', 'isolated2'): xm.mark_step()
        with torch.no_grad():
            gs = grads_of(rep_params)
            for _r in range(SYNC_REPEAT): acc = grad_sync(gs)
        if SYNC_GRAPH == 'isolated2': xm.mark_step()
        return loss_total, acc

    backend = sys.argv[1] if len(sys.argv) > 1 else 'per_mb'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    fn = step_per_mb if backend == 'per_mb' else step_bundled
    tag = f"amp7_{backend}_{GRAD_SYNC if backend != 'per_mb' else 'base_' + BASE_SYNC + '_' + BASE_STRAT}_{SYNC_GRAPH}_L{N_LAYERS_PER_STAGE}_M{N_MB}_S{S}_sites{len(sites)}" + (f"_rep{SYNC_REPEAT}" if SYNC_REPEAT != 1 else "") + ("_flatbuf" if GRAD_BUFFER == 'flat' else "") + (f"_ag{FSDP_AG}" if FSDP_AG != 'pinned' else "") + ("_nosites" if not SITES else "") + f"_ws{ws}_seed{seed}"

    bench_rank = half
    if rank == bench_rank:
        print(f'[init] ws={ws} half={half} N_MB={N_MB} N_LAYERS={N_LAYERS_PER_STAGE}/stage T_A={T_A}', flush=True)
        print(f'[init] DM={DM} HID={HID} VOCAB={VOCAB} B={B} S={S} backend={backend} steps={steps} grad_sync={GRAD_SYNC} base_strat={BASE_STRAT} sites={PROBLEM_SITES} base_sync={BASE_SYNC} sync_graph={SYNC_GRAPH} sync_repeat={SYNC_REPEAT} rep_params={len(rep_params)} (emb {N_EMB}, site pieces {N_B}) rep_elems={sum(p.numel() for p in rep_params)}', flush=True)

    def apply_update(acc):
        with torch.no_grad():
            for p, g in zip(rep_params, acc): p.data = p.data - LR * (g / ws).to(p.dtype)
            for p in sharded_params:
                if p.grad is not None:
                    g = xm.all_reduce(xm.REDUCE_SUM, p.grad) / ws
                    p.data = p.data - LR * g.to(p.dtype)

    for _ in range(WARMUP):
        loss, acc = fn(0); apply_update(acc); xm.mark_step()
        if loss is not None: _ = loss.item()
    if rank == bench_rank: print('[init] warmup done', flush=True)

    times, losses = [], []
    for s in range(steps):
        probe.set_step(s)
        xm.mark_step()
        t0 = time.time()
        _ts = _t.time()
        loss, acc = fn(s)
        apply_update(acc)
        xm.mark_step()
        if loss is not None: losses.append(loss.item())
        else: _ = rep_params[0].data.float().sum().item()   # stage-0 ranks: force execution so the timing covers the step
        if probe.in_window(): probe.record("step_incl_sync", (_t.time() - _ts) * 1000.0)
        times.append((time.time() - t0) * 1000)
        if rank == bench_rank and (s + 1) % 100 == 0:
            print(f'  step {s+1}: median_ms={statistics.median(times[-50:]):.1f} loss={statistics.mean(losses[-50:]) if losses else float("nan"):.3f}', flush=True)

    if rank == bench_rank:
        steady = times[STEADY_FROM:] if len(times) > STEADY_FROM else times
        steady_loss = losses[STEADY_FROM:] if len(losses) > STEADY_FROM else losses
        print(f'[bench] {backend} steady_mean={statistics.mean(steady):.2f}ms steady_median={statistics.median(steady):.2f}ms final_loss={steady_loss[-1] if steady_loss else float("nan"):.3f}', flush=True)
        os.makedirs('/tmp/tp_search', exist_ok=True)
        with open(f'/tmp/tp_search/llama_e2e_{tag}.json', 'w') as f:
            json.dump({'backend': backend, 'steady_mean_ms': statistics.mean(steady), 'steady_median_ms': statistics.median(steady), 'all_ms': times,
                       'losses': losses[-100:] if losses else [], 'final_loss': steady_loss[-1] if steady_loss else None,
                       'DM': DM, 'HID': HID, 'HIDR': HIDR, 'T_A': T_A, 'N_LAYERS_PER_STAGE': N_LAYERS_PER_STAGE, 'N_MB': N_MB, 'B': B, 'S': S, 'VOCAB': VOCAB, 'steps': steps,
                       'grad_sync': GRAD_SYNC, 'base_sync': BASE_SYNC, 'sync_graph': SYNC_GRAPH, 'sync_repeat': SYNC_REPEAT, 'n_rep_params': len(rep_params),
                       'site_b_sizes': SITE_B_SIZES, 'ws': ws, 'tag': tag, 'grad_buffer': GRAD_BUFFER, 'sitea_layout': SITEA_LAYOUT, 'fsdp_ag': FSDP_AG, 'sites': PROBLEM_SITES, 'base_strat': BASE_STRAT, 'microbench_winner': MICROBENCH_WINNER}, f)
        probe.dump(f'/tmp/tp_search/llama_e2e_{tag}_percall.json', extra={'backend': backend, 'steps': steps, 'seed': seed})


if __name__ == '__main__':
    main()
