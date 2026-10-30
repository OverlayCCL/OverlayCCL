# correctness_test_v5.py — training-scope terms for heterogeneous collective sets

Additive to v4 (the paper's calibrated simulator). Both terms act only when `coll_run_reset == "fusion"` (training
scope). With the terms off (`HET_BW_BPUS=0`, `UNEQ_FACTOR=1`) v5 reproduces the v4 scores exactly (71/71 pool
candidates, 84/84 new-problem candidates).

1. **Heterogeneous-set penalty** `het_set_penalty(op, n_calls, scaled_bytes, set_min, set_max, frac_2d) -> us`
   (`het_penalty.py`), charged (x2 for fwd+bwd) to every collective of a type whose calls in one graph have >= 2 distinct
   operand sizes:

       excess_us = (bytes / HET_BW_BPUS) * n_calls ** HET_COUNT_EXP * (1 + HET_2D_GAIN * frac_2d)

   `frac_2d` = share of the set's collectives issued on operands of rank >= 2 (the sandbox records the operand rank of
   every collective event). Operands below `HET_MIN_BYTES` (256 KB) are exempt; `HET_OPS` = all_reduce.
   **Frozen constants (2026-10-27):** `HET_BW_BPUS=1086120` (1.09 GB/ms), `HET_COUNT_EXP=1.2`, `HET_2D_GAIN=8`
   (`HET_SPREAD_EXP=0`; a size-spread factor is available but not used). These are the module defaults.

   *Motivation and calibration.* A lone-call microbench lets XLA combine back-to-back same-type collectives; inside a
   training graph a set with heterogeneous operand sizes is not combined. Measured per-call excess of per-tensor
   all-reduce over the best bundle at 64 ranks (2 nodes): 1-D mixed-size sets 0.02-0.9 ms (`probe_het_*`, hs_a, hs_c,
   layer/model gradients flattened to 1-D), 2-D layer-structured sets 2.0 ms (6 tensors) to 7.0 ms (24 tensors); the
   same model gradients cost 420 ms per step as 2-D operands and 102 ms as 1-D operands, so the operand rank, not size
   heterogeneity alone, carries most of the cost. Fit: `fit_het2.py` (log-space grid over BW, exponent and 2-D gain on
   19 measured sets, pathology sets excluded). Ranking check (`rank_eval.py`, training scope, 22 problems): the
   per-tensor-vs-best-bundle direction matches hardware on every set where hardware shows a >2% difference and no
   unequal-concat pathology (model/layer/mixed-dtype/hs_a/hs_c/probe 24-mid, 48-mid, model 1-D); the remaining
   mismatches are hardware ties (emb_ln, hs_b, 48-small) or pathology sets (see 2).
   *Regression.* Every candidate of the paper's eight problems issues equal-size sets, so the term never fires on the
   pool: 0 of 71 pool candidates change under the frozen constants (`scripts/score_v5.py`, arms `sim_v4_v5_{off,het_only,plain,spread,2d}.json` in
   `../results/sensitivity_arms`), fidelity table identical to the campaign's (rho A2AV .90, dxe .56, FSDP .81, LB .90,
   PP .89, RKV .75, TP .90, UA2A .90).

2. **Unequal-concat factor** `unequal_concat_factor`: multiplies a collective fed by a `torch.cat` of unequal operands
   (marker event `cat_unequal`). Measured 7-11x on some workloads (hetero_grad_ar_many, zero2_hetero_rs, hs_a flat concat
   0.86x, layer/model "no-norm" flat concats 0.6-0.8x) and absent on others (layer/model flat concats 1.8-4.9x faster);
   the trigger is not understood, so this stays OFF (1.0) and the pathology is left to a training-scope hardware gate.

3. **Sandbox dtypes (2026-10-27).** `MockTorch` exposes `bfloat16` / `float16` / `float64`, `TrackedTensor` gained
   `.bfloat16()` / `.half()` / `.double()`, and any dtype-changing cast (`.to`, `.float`, ...) records a `cast` volume op
   (bytes of the output). Mixed-dtype gradient bundles are now expressible; with this the strategy-enumerate loop picks
   the two-dtype-group bundle on `layer_grads_mixed_dtype` (sim 9.4 vs 11.1 ms per-tensor; hardware 31.7 vs 56.0 ms).
   Pool regression unchanged (0 of 71, `sim_v4_v5_2d_dtype.json`).

4. **Revision v8 (2026-10-28), after the day-4 sweeps.** Four changes, all with 0 of 71 pool candidates changed
   (`../results/sensitivity_arms/sim_v4_v8.json`):
   - the large per-tensor gain is keyed on **tiny-mix sets** (a member below `HET_TINY_BYTES` = 64 KB together with a member
     at least `HET_TINYMIX_SPREAD` = 128x larger; `HET_TINYMIX_GAIN` = 8) instead of operand rank: the sets that pay the
     large penalty are norm vectors interleaved with weight matrices, while a 48-piece 2-D set without tiny members
     (hetero_grad_ar_many) runs per-tensor at full speed; `HET_2D_GAIN` is now 0;
   - a **piece-count knee**: `HET_KNEE_US` = 12.5 us per call per piece above `HET_COUNT_KNEE` = 32 pieces, for sets whose
     largest member is >= 1 MB (the probe sweep: 24-piece 1-D sets pay nothing, 48- and 64-piece sets pay 0.2-0.9 ms per
     call);
   - the **unequal-concat term** (`make_unequal_piece_penalty`, bytes / `UNEQ_BW_BPUS` = 2000 B/us when more than
     `UNEQ_KNEE` = 40 unequal pieces of mean size >= 256 KB feed one all-reduce; calibrated on the catk sweep: flat
     concats of <= 32 pieces are free, >= 48 pieces cost ~bytes / 2 GB/s or fail to compile) now also fires on buffers
     **assembled by slice assignment** into a preallocated tensor (`TrackedTensor.__setitem__` keeps the piece list; the
     all-reduce mocks emit the `cat_unequal` marker), which is how a refined search candidate evaded the concat marker;
   - sandbox dtypes (bf16/fp16, `cast` volume op) and a ZeRO-2 heterogeneous reduce-scatter registration
     (`zero2_hetero_rs`; its dim-1 concatenation candidate is expressible but its 11x hardware slowdown has no term).
   Ranking against hardware (`rank_eval.py`, 27 sets): per-tensor-vs-best-bundle direction right on 20; the 7 misses
   are hardware ties scored as small differences (hs_b, catk_32_small, emb_ln), the equal-size control (hs_d), and the
   three unequal-concat pathology sets (hetero_grad_ar_many, model/layer no-norm, path_many_half) whose bundles are slow
   for a backend reason invisible at the HLO level (results/hlo). Those remain a training-scope hardware-gate matter.

5. **Revision v11 (2026-10-28), two sandbox holes found by the search itself.** (a) The unequal-concat term keyed on
   all-reduce consumers only; the strategy enumerator evaded it with "concatenate, reduce-scatter, all-gather", so
   `UNEQ_OPS` now defaults to `all_reduce,reduce_scatter,all_gather`. (b) `TrackedTensor.__setitem__` did not mark the
   assigned value as consumed, so a collective result written into a preallocated buffer was treated as dead code and its
   all-reduce dropped from the score (a 12-all-reduce "hierarchical bucketing" refinement scored 11.9 ms against 580 ms
   for its parent); the assignment now unwraps the value through `_unwrap`, which marks it live. Pool regression after
   both fixes: 0 of 71 candidates changed (`../results/sensitivity_arms/sim_v4_v11.json`). Search-side changes made at
   the same time (registrations in `search/problems_newprobs.py`): the `bucketed_ar` builtin buckets by count (8 in-order
   tensors) instead of a byte threshold that never fills at sandbox scale; `layer_grads_mixed_dtype` seeds a bf16 flat
   bundle and a two-dtype bundle; the optimization hints state that sandbox sizes are scaled down and that bucket code must
   be size-independent and keep the deployment order.
