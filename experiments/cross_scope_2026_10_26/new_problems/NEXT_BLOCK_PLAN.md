# Plan for the next hardware block (7 nodes preferred; 2 nodes acceptable for items 1-4)

Deadline rule: Capacity Block instances are reclaimed 30 minutes before the block's end time; schedule the last pull at
end - 45 min. Runner changes already in place: `g_train.py` records `collective_effective` (output differs from input)
and `g_newprobs_run2.sh` marks a run INVALID when the node logs show `ncclInitComm` / NRT init failures.

1. **Re-validate grouped collectives** at 64 and 224 ranks with the effectiveness check: dp_grad_ar_groups,
   grp_small_dp_ar, expert_grads_ep_groups, model_grads_ep_groups (their ~5-8 ms training results are suspect: cross-node
   4-rank replica groups failed NCCL init at 64 ranks). Try group layouts with >= 4 members per node (TP=8 on 2 nodes,
   TP=16 on 7 nodes) and the `xm.all_reduce(..., groups=..., pin_layout=False)` variant.
2. **Finish day 3e**: model_grads_intra_node_groups (placement control), model_grads_fp32, and bench64 of the diagnostics
   (`layer_grads_1d`, `model_grads_1d`, `layer_grads_no_norm`, `model_grads_no_norm`); repeat the probe_het sweep once.
3. **7-node check of the inverting set** (model_grads_4layers, layer_grads_hetero_ar, layer_grads_mixed_dtype, hs_a, hs_c,
   probe 24-mid/48-mid/48-small, hetero_grad_ar_many, zero2_hetero_rs): bench at 32 ranks on one node, training at 224
   ranks; does the per-tensor penalty scale with world size (the v5 term does not depend on world size)?
4. **Operand-rank mechanism**: 2-D sets with power-of-two rows and columns (hetero_grad_ar_many is 2-D yet fast per-tensor)
   vs the layer shapes (6144 / 10752 rows, 5376 columns); 2-D sets with and without tiny 1-D members; a 3-D variant.
   Goal: a predictive rule for `frac_2d` (or replace it by a shape-based condition) and for the unequal-concat pathology.
5. **Different-setup inversions still untested**: reduce-scatter / all-gather heterogeneous sets at 224 ranks (ZeRO-2/-3
   style), `collective_permute` pipeline transfers at 224 ranks (3-3.8x faster at both benches, failed at 64-rank
   training), Ulysses all-to-all with `pin_layout=False`, and the OLMoE end-to-end step with the masked reduce-scatter.
6. **Search convergence on hardware**: re-run Phase 3 (v5 frozen) on every registered problem and verify each pick in
   training; add a training-scope hardware gate to the search harness so that unequal-concat picks are rejected.
