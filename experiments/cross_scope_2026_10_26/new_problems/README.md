# New problems outside the paper's eight (2-node cluster, 2026-10-26)

`g_newprobs.py` registers eight collective problems with a production baseline and alternative strategies each
(DP gradient all-reduce in replica groups, ZeRO-2 gradient reduce-scatter, ZeRO-1 parameter all-gather, Ulysses
sequence-parallel all-to-all, expert-parallel all-to-all in groups, large bucketed DP all-reduce, heterogeneous
gradient all-reduce x2). `g_bench2.py` = 1-node bench + on-device correctness (32 ranks); `g_train.py` = 2-node
training-scope micro-runner (matmul -> collective -> loss -> backward re-issuing the collective, 300 steps); runners
`g_newprobs_run.sh` / `g_newprobs_run2.sh` (NP_OUT); `newprobs_table.py <dirs...>` prints bench-vs-training ratios
averaged over rounds (`results/newprobs_table.txt`). `../../search/problems_newprobs.py` registers five of the
problems in the search loop's format (validate with `sandbox_newprobs.py`).

## Day 2 (2026-10-26/27): rigid cross-scope inversions

Added problems (g_newprobs.py): `hs_a/b/c/d/e` (mixed-size gradient all-reduce sweeps), `grp_small_dp_ar`,
`pp_small_transfer`, `layer_grads_hetero_ar`, `emb_ln_hetero_ar`, `zero1_hetero_ag`, `fsdp_hetero_ag`, `zero2_hetero_rs`,
`layer_grads_mixed_dtype`, `model_grads_4layers`, `pp_hetero_transfer`. Three scopes per candidate: 1-node bench (32
ranks, `results/day2*/newprob_bench_*.json`), 2-node forward bench at 64 ranks (`results/bench64/*_ws64.json`,
`g_bench64.sh`), 2-node training step at 64 ranks (`results/*/train/`). `results/day2_table.txt` = all rounds merged.
Controls (`g_controls.sh`): the heterogeneous-concat pathology reproduces in the training runner at 32 ranks on one node
and is absent from the forward bench at 64 ranks, so it is a graph-context effect, not a node-count effect. Mechanism
diagnostic `per_tensor_marked` (graph cut after each all-reduce) shows the lone-call bench combines back-to-back
all-reduces (per-tensor 2.5-3.7x faster unmarked than marked), the training graph does not.

## Simulator check on the inverting problems (sim/)

`sim_newprobs.py` scores every candidate of the eight inverting problems with the paper's calibrated simulator at bench
scope (32 ranks, one node) and training scope (64 ranks, two nodes); `sim_newprobs.json` = default constants,
`sim_newprobs_op.json` = dispatch-run reset after every op, `sim_newprobs_op_whatif15000.json` = what-if with a 15 ms
un-amortized cost per collective at training scope. Result: with the shipped constants every candidate scores within
0.2-4% of the others at both scopes (time is byte-dominated; dispatch overhead is 1.7 ms and amortized), so the simulator
does not separate the training winners from the bench winners; the 15 ms what-if recovers the measured ordering on the six
bundling problems but not on the two unequal-concat pathologies, which no cost term predicts.

## Day 3 (2026-10-27): calibration, diagnostics, search convergence — see REPORT.md

`REPORT.md` consolidates the study: the eight (plus three probe-derived) strict inversions, the mechanism (bench combiner,
operand-rank effect, unequal-concat pathology, API-level failures of collective_permute and cross-node replica groups),
simulator v5 with its frozen constants and zero pool regression (`../../cluster_224rank/sim/README_v5.md`,
`fit_het2.py`, `rank_eval.py`), and the Phase-3 convergence table with hardware verification (`phase3/`,
`results/phase3_picks_hw_table.txt`). New result dirs: `results/day3` (API/algebraic variants), `results/day3b`
(probe_het sweep), `results/day3diag` (1-D / no-norm diagnostics).

**Correction (2026-10-27):** all training-scope (64-rank) results of candidates that pass replica `groups`
(dp_grad_ar_groups, ep_a2a_grouped, grp_small_dp_ar, expert_grads_ep_groups, model_grads_ep_groups) are invalid: the node
logs show NCCL communicator init failures for the cross-node 4-rank groups and the step times are the dispatch floor. The
statements about a "group placement blind spot" in the day-1/day-2 notes above are withdrawn; see REPORT.md section 1
and NEXT_BLOCK_PLAN.md item 1. `g_train.py` now records `collective_effective` and the runner marks such runs INVALID.
