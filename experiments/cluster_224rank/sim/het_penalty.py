"""Heterogeneous-set collective penalty for simulator v5 (training scope only), shared by the scoring drivers.

excess_us_per_call_per_direction = (bytes / HET_BW_BPUS) * n_calls ** HET_COUNT_EXP * spread_factor
spread_factor = max(HET_SPREAD_FLOOR, log(max_bytes / min_bytes) / log(HET_SPREAD_BASE)) ** HET_SPREAD_EXP   (1.0 when HET_SPREAD_EXP = 0)
                 * (1 + HET_2D_GAIN * frac_2d)   where frac_2d = share of the set's collectives issued on operands of rank >= 2 (0 when HET_2D_GAIN = 0)
Applies only to operands >= HET_MIN_BYTES of the collective types in HET_OPS, and only to sets with >= 2 distinct operand
sizes in one graph (the simulator decides the latter). All knobs are environment variables so the drivers stay unchanged.
Defaults = the constants frozen on 2026-10-27 from the 2-node calibration (probe_het sweep + layer/model gradient
diagnostics, see experiments/cross_scope_2026_10_26/new_problems/REPORT.md): HET_BW_BPUS=1086120 (1.09 GB/ms),
HET_COUNT_EXP=1.2, HET_TINYMIX_GAIN=8 (replaces the 2-D gain: the sets that pay the large penalty mix tiny
norm vectors with large matrices; hetero_grad_ar_many is 2-D without tiny members and is fast), HET_COUNT_KNEE=32 with
HET_KNEE_US=12.5 per piece above the knee (catk sweep: sets of >= 48 pieces pay extra), HET_2D_GAIN=0, HET_SPREAD_EXP=0,
HET_MIN_BYTES=262144, HET_OPS=all_reduce.
Set HET_BW_BPUS=0 to disable the term (reproduces the v4 scores exactly)."""
import math, os

def make_het_penalty():
    bw = float(os.environ.get("HET_BW_BPUS", "1086120") or 0)
    if bw <= 0: return None
    a = float(os.environ.get("HET_A_US", "0")); e = float(os.environ.get("HET_COUNT_EXP", "1.2"))
    min_bytes = float(os.environ.get("HET_MIN_BYTES", "262144")); ops = set(os.environ.get("HET_OPS", "all_reduce").split(","))
    g2 = float(os.environ.get("HET_2D_GAIN", "0")); se = float(os.environ.get("HET_SPREAD_EXP", "0")); sb = float(os.environ.get("HET_SPREAD_BASE", "8")); sf = float(os.environ.get("HET_SPREAD_FLOOR", "0.3"))
    gt = float(os.environ.get("HET_TINYMIX_GAIN", "8"))          # set mixes tiny (< HET_MIN_BYTES) and large operands (layer norms + matrices)
    knee = float(os.environ.get("HET_COUNT_KNEE", "32")); knee_us = float(os.environ.get("HET_KNEE_US", "12.5"))   # many-piece sets (>= 48 pieces) pay extra per call
    def het(op, n, nbytes, set_min=None, set_max=None, frac_2d=0.0):
        if op not in ops or nbytes < min_bytes: return 0.0
        f = 1.0
        if se != 0 and set_min and set_max and set_max > set_min:
            f = max(sf, math.log(set_max / set_min) / math.log(sb)) ** se
        if g2 != 0: f *= (1.0 + g2 * float(frac_2d or 0.0))
        tiny = float(os.environ.get("HET_TINY_BYTES", "65536")); spread_min = float(os.environ.get("HET_TINYMIX_SPREAD", "128"))
        if gt != 0 and set_min is not None and set_max is not None and set_min < tiny and set_max >= spread_min * max(1.0, set_min): f *= (1.0 + gt)
        knee_term = knee_us * max(0.0, float(n) - knee) if (set_max is None or set_max >= 4 * min_bytes) else 0.0
        return (a + nbytes / bw) * (float(n) ** e) * f + knee_term
    return het

def make_unequal_piece_penalty():
    """Per-call cost (us, per direction) of a collective fed by a torch.cat of k UNEQUAL pieces (catk sweep 2026-10-27:
    flat concats of <= 32 unequal pieces are free, >= 48 pieces cost ~bytes / 2 GB/s extra or fail to compile; concats of
    tiny pieces only (< 256 KB mean) show no penalty). Default mode "bytes": nbytes / UNEQ_BW_BPUS (2000 B/us) when
    npieces > UNEQ_KNEE (40) and mean piece >= UNEQ_MIN_PIECE_BYTES; set UNEQ_BW_BPUS=0 to disable."""
    per = float(os.environ.get("UNEQ_PIECE_US", "0") or 0); bw = float(os.environ.get("UNEQ_BW_BPUS", "2000") or 0)
    mode = os.environ.get("UNEQ_MODE", "bytes")          # "bytes": nbytes / UNEQ_BW_BPUS when npieces > knee; "pieces": per * (npieces - knee)
    if (mode == "pieces" and per <= 0) or (mode == "bytes" and bw <= 0): return None
    knee = float(os.environ.get("UNEQ_KNEE", "40")); ops = set(os.environ.get("UNEQ_OPS", "all_reduce,reduce_scatter,all_gather").split(","))
    min_piece = float(os.environ.get("UNEQ_MIN_PIECE_BYTES", "262144"))
    def pen(op, npieces, nbytes):
        if op not in ops or npieces <= knee: return 0.0
        if nbytes / max(1.0, float(npieces)) < min_piece: return 0.0   # concats of tiny pieces only (hs_b) show no penalty
        return (nbytes / bw) if mode == "bytes" else per * (float(npieces) - knee)
    return pen
