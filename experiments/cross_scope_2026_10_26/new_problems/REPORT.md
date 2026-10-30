# Cross-scope inversions outside the paper's eight problems: findings, mechanism, simulator fix, search convergence

Consolidated report of the 2-node study (Capacity Blocks 2026-10-26 and 2026-10-26/27, 2 x trn1.32xlarge, 64 ranks).
Definition used throughout (the strict one): strategy A is significantly faster than B at a lone-call microbench and
significantly slower than B inside a training step. Three scopes per candidate: `bench32` = 1-node forward bench at
32 ranks (`g_bench2.py`), `bench64` = 2-node forward bench at 64 ranks (`g_bench64.sh`), `train64` = 2-node training
step at 64 ranks (`g_train.py`: matmul -> collective -> loss -> backward, the backward re-issues the collective through an
autograd Function; 300 steps; median). Ratios are baseline / candidate (>1 = candidate faster). n = rounds.

## 1. Eight new problems with rigid pairwise inversions (results/day2_table_final.txt)

| problem | bench winner | training winner | bench32 / bench64 ratio of the training winner | train64 ratio | n |
|---|---|---|---|---|---|
| model_grads_4layers (24 heterogeneous DP grads, 4 layers) | per_tensor_ar | per_layer_bucket / flat_cat / bucket_32mb | 0.85 / 1.01, 0.89 / 1.07, 0.70 / 0.78 | 4.71, 4.92, 2.67 | 3 |
| model_grads_bucket_sweep (same grads, 1-256 MB buckets) | per_tensor_ar | every bucket size | 0.64-0.94 | 2.66-4.28 | 2 |
| layer_grads_hetero_ar (one layer's 6 grads) | per_tensor_ar | bucket_32mb / equal_group_stacks / flat_cat | 0.86 / 0.90, 0.86 / 0.87, 0.91 / 1.08 | 1.44, 1.44, 1.76 | 3 |
| layer_grads_mixed_dtype (4 bf16 + 2 fp32 grads; dtype-honoring runner, day 4) | per_tensor_ar | cast_flat_bf16 / cast_flat_fp32 / two_dtype_groups | 0.92 / -, 0.70 / -, 0.50 / - | 1.67, 1.39, 1.19 | 1 (day-2 rows were a harness confound, see section 7) |
| model_grads_fp32 (24 fp32 grads; dtype-honoring runner) | per_tensor_ar | flat_cat / per_layer_bucket | 0.76 / - | 3.37, 3.36 | 1 |
| probe_het_24_mid / 48_mid / 48_small (24 or 48 mixed-size 1-D grads) | per_tensor_ar | bucket6 | 0.80, 0.75, 0.77 at bench32 | 1.12-1.13, 1.17, 1.55-1.75 | 2 |
| hs_a_64x64k_512k (64 mixed 1-D grads) | per_tensor_ar | bucket_4mb | 0.74 / 0.89 | 2.04 | 3 |
| hs_c_32x256k_2m (32 mixed 1-D grads) | per_tensor_ar | bucket_4mb | 0.77 / 0.87 | 1.19 | 3 |
| hetero_grad_ar_many (48 odd-width 2-D grads) | flat_cat_ar (1.16x at bench64) | per_tensor_ar | 1.01 / 1.16 | 0.14 (7x slower) | 2 |
| zero2_hetero_rs (heterogeneous reduce-scatter) | cat_dim1_rs (1.14x at bench64) | per_tensor_rs | 0.78 / 1.14 | 0.09 (11x slower) | 3 |

Controls: `hs_d_64x256k_equal` (equal sizes) shows no inversion (flat 0.87 / 1.01 -> 1.09); `probe_eq8_*` (8 equal
all-reduces vs one stacked, 64 KB-16 MB) tie at both scopes (train 14.7/14.4, 15.9/15.0, 26.0/25.0, 75.0/66.3 ms).
Problems that do NOT invert: hs_b, hs_e (tiny operands), emb_ln_hetero_ar, zero1_hetero_ag, fsdp_hetero_ag (bundles
fail to compile), pp_small / pp_hetero / pp_4stage (same direction at both scopes), grad_norm_clip_stat, and the day-1
problems zero2_grad_rs, zero1_param_ag, ulysses_seq_a2a, dp_grad_ar_buckets (amplifications or same direction).
**INVALID (found 2026-10-27):** every training-scope run of a candidate that uses replica `groups` at 64 ranks
(dp_grad_ar_groups, ep_a2a_grouped, grp_small_dp_ar, expert_grads_ep_groups, model_grads_ep_groups; all grouped
candidates) failed NCCL communicator init in the node logs (`ENC:ncclInitComm failed`, 26-32 occurrences per log) and
reported dispatch-floor step times (~5-8 ms) without executing the collective; the runner had no output check. Their
'training' numbers and the derived 'group-placement blind spot' statements (world-wide substitutes 0.35x in training)
are withdrawn; only the un-grouped (masked world-wide) candidates of those problems ran. The 32-rank benches of the
same candidates are valid (intra-node groups initialize).

## 2. Mechanism

* **Bench combiner.** `per_tensor_marked` (an `xm.mark_step()` after each all-reduce, i.e. the DDP-hook behavior) is
  2.5-3.7x slower than the unmarked per-tensor loop at the 32-rank bench (layer_grads 24.5 vs 9.5 ms, model_grads 92.7 vs
  25.1 ms): a lone-call bench lets XLA combine back-to-back same-type collectives into one, so a per-tensor loop looks like
  a single collective. Inside the training graph the per-tensor set is not combined when the operand sizes are
  heterogeneous (equal-size sets still are, see the controls), so per-tensor pays one uncombined collective per tensor
  (~15 ms each in model_grads vs ~1.7 ms on the bench) and bundles win 1.2-4.9x.
* **Unequal-concat pathology** (hetero_grad_ar_many, zero2_hetero_rs): a concat of unequal operands feeding one
  collective is 7-11x slower inside the training graph and fast on the forward bench at the same world size
  (`g_controls.sh` reproduces it in the training runner at 32 ranks on one node, so it is a graph-context effect, not a
  node-count effect). Padding to 16 M / pow2, splitting in two and size-bucketing do not remove it; the flat cats of
  layer_grads / model_grads (also unequal operands) are fast. The trigger is not understood; only a hardware gate catches it.
* **API-level cross-scope failure** (`pp_api_variants`): `collective_permute` point-to-point transfers are 3.0-3.8x
  faster than the masked all-reduce at both benches and fail to load the NEFF at 64-rank training (NRT rendezvous error,
  twice). `all_to_all` fails to compile even at 32 ranks.
* **Algebraic / API knobs on the inverting workload** (`layer_grads_api_variants`): `scale=` on the flat all-reduce,
  reduce-scatter + all-gather with native layout, and `pin_layout=False` all behave like the flat bundle (0.88-0.91 at
  bench32, 1.69-1.76 in training) = the same inversion; `pin_layout=False` on the per-tensor loop changes nothing; the
  two-stage grouped all-reduce (intra-node then inter-node) loses the bench (0.68 / 0.57) and ties training (1.05).
  Ulysses variants (masked reduce-scatter 1.23 / 1.60 -> 1.62) amplify rather than invert.

## 3. Simulator v5 to v8 (../../cluster_224rank/sim/correctness_test_v5.py, het_penalty.py, README_v5.md)

The paper's calibrated simulator scores all candidates of the inverting problems within 0.2-4% of each other and slightly
favors per-tensor (sim/sim_newprobs.json). The revision adds training-scope-only terms (they fire only when
`coll_run_reset == "fusion"`); with every term off the v4 scores are reproduced exactly (71/71 pool, 84/84 new).

**Terms (frozen 2026-10-28, module defaults in `het_penalty.py`).**

1. *Heterogeneous-set penalty* on every all-reduce of a set with >= 2 distinct operand sizes in one graph:
   `(bytes / 1,086,120 B/us) * n^1.2 * (1 + 8 * tinymix) + 12.5 us * max(0, n - 32)`, operands below 256 KB exempt, the
   count knee only for sets whose largest member is >= 1 MB. `tinymix` = the set has a member below 64 KB and a member
   at least 128x larger (norm vectors interleaved with weight matrices).
2. *Unequal-concat term* on an all-reduce fed by a concatenation of more than 40 unequal pieces of mean size >= 256 KB:
   `bytes / 2000 B/us`; buffers assembled by slice assignment into a preallocated tensor count as concatenations (the
   sandbox keeps the piece list per tensor and the all-reduce mocks emit the marker).
3. Sandbox dtypes (bf16/fp16 and a `cast` volume op) and a rank-preserving shrink in the scoring drivers.

**Calibration data (results/day3b, day3diag, day4a, day4b, day4e; tables `day3b_sweep_table.txt`,
`day4a_catk_table.txt`, `day4b_path_table.txt`, `day4_experiments_table.txt`).**

| experiment | finding |
|---|---|
| probe_het sweep (6/24/48 x small/mid, 1-D) | per-tensor penalty absent at <= 24 pieces, 0.2-0.9 ms per call at 48 pieces; three strict inversions (24-mid, 48-mid, 48-small) |
| layer / model gradients flattened to 1-D at identical sizes | per-tensor step 420 -> 102 ms (model), 55.8 -> 35.2 (layer): most of the large penalty is tied to the set's structure, not its sizes |
| layer / model without the two norm vectors | per-tensor 420 -> 153 ms (model), 55.8 -> 36.8 (layer): the tiny-plus-large mix drives the large gain |
| hetero_grad_ar_many (48 2-D pieces, no tiny member) | per-tensor fast (68-75 ms for 233 MB): rank-2 operands alone are not the trigger (the earlier "2-D gain" was replaced by the tiny-mix gain) |
| catk piece-count sweep (8-128 unequal 1-D pieces) | flat concat free up to 32 pieces; loses to 8-piece buckets from 64 pieces (76 vs 42 ms at 31 MB; 337 vs 276 at 61 MB); at 48 mid-size pieces the flat concat does not compile within 15 min |
| tiny-piece cure (path_*): one appended 4 KB tensor | makes the 4-matrix concat fast (61 -> 36 ms) and the 48-piece concat fast (486 -> 64 ms); appending 1024 elements cures, 65536 does not; equal pieces with the same total are fast; HLO of slow and cured variants is structurally identical (results/hlo) |

**Regression.** 0 of 71 pool candidates change under the frozen terms (`../../cluster_224rank/results/sensitivity_arms/
sim_v4_v8.json`; two intermediate variants that charged slice assignment as a volume op moved 12 and 2 pool candidates
and were withdrawn), so the fidelity table is the campaign's (rho A2AV .90, dxe .56, FSDP .81, LB .90, PP .89, RKV .75,
TP .90, UA2A .90).

**Ranking against hardware** (`rank_eval.py`, sim/sim_newprobs_v5_arm_v8.json, 27 sets): the per-tensor-vs-best-bundle
direction matches hardware on 20; the seven misses are hardware ties scored as small differences (hs_b, catk_32_small,
emb_ln), the equal-size control (hs_d), and the three unequal-concat pathology sets (hetero_grad_ar_many, model/layer
no-norm, path_many_half), whose bundles are slow for a backend reason invisible at the HLO level. The hardware winner is
ranked highest on model_grads, layer_grads, hs_a, probe 24-mid, 48-mid, 48-small and (with real dtypes) mixed-dtype;
on hs_c flat and buckets tie; on hetero_grad_ar_many and zero2_hetero_rs a pathological bundle stays ranked first.

## 4. Does OverlayCCL converge to the training winners? (phase3/, results/gate_v11/convergence_table.txt)

> **Selection rule restored (2026-10-29).** The paper's loop ships the candidate with the lowest *simulated* time among the
> candidates that pass the correctness/TV gates (Phase 5, `survivors.sort(key=sim_time_us)`; "correctness gates only" in the
> system model). The training-scope *performance* gate described below was a method change introduced on 2026-10-28; it is
> now a validation column only. The loop's output under the paper's rule with the fitted v11 simulator is in
> `results/loop_simdecides/convergence_simdecides.md`: 8 of 10 problems converge (6 at the optimum, 24-mid and hs_c within
> 3% and 12% of it), hetero_grad_ar_many ships a 2.9x regression and 48-mid a bimodal candidate. The two problems carried
> forward into the Llama e2e are the two whose winners add an ingredient beyond bundling and that converge under the
> paper's rule: layer_grads_mixed_dtype (bf16 wire cast, 1.67x at 64 ranks / 1.57x at 224) and probe_het_48_small
> (in-order 8-piece buckets after the correctness gate removes a non-running first pick, 1.56x / 1.64x).

**Final loop (2026-10-28, simulator v11, 2-node pair of day 5).** Phase 2 builtins + Phase 3 strategy-enumerate
(Sonnet 4.5) scored at training scope, then a **training-scope hardware gate** (`gate_training.py`): the top-3 scored
candidates (LLM strategies and builtins) plus the per-tensor baseline are each run in the 64-rank training harness and the
fastest measured one is the loop's output. Hand-written candidates are measured on the same pair for reference only.
Ratio = per-tensor / candidate.

| problem | gate output | hardware | best hand-written | verdict |
|---|---|---|---|---|
| layer_grads_hetero_ar | in-order 2-bucket split | 36.6 vs 61.1 ms, 1.67x | flat 1.67x | at optimum |
| model_grads_4layers | flat concat | 96.8 vs 431.4, 4.46x | flat 4.46x | at optimum |
| layer_grads_mixed_dtype (real dtypes) | bf16 flat bundle | 36.5 vs 61.0, 1.67x | bf16 flat 1.66x | at optimum |
| hs_a (dp_grad_bucketing_64) | alternating in-order buckets | 73.0 vs 98.9, 1.35x | bucket_4mb 1.38x | at optimum (2%) |
| probe_het_24_mid | flat concat | 267.4 vs 304.0, 1.14x | bucket6 1.14x | at optimum |
| probe_het_48_small | in-order 8-buckets | 31.8 vs 50.0, 1.57x | bucket6 1.57x | at optimum |
| probe_het_48_mid | in-order 4-buckets (second round) | 492.4 vs 596.4, 1.21x | bucket6 1.18x | above the hand-written best |
| hs_c_32mixed | in-order 16-buckets (second round) | 90.4 vs 104.6, 1.16x | bucket_4mb 1.18x | at optimum (2%) |
| hetero_grad_ar_many | shape-class bundles (second round, finalists merged across Phase-3 runs) | 65.2 vs 74.3, 1.14x | shape-class 1.14x; per-tensor was the prior best | at optimum |
| zero2_hetero_rs | per-tensor reduce-scatter (baseline) | 24.1 ms | baseline is the optimum | at optimum (gate rejects the dim-1 concat finalist, 236 ms) |

All ten converge: nine at or above the hand-written optimum, hs_c within 2%. A repeat of every gate decision the same
night (results/gate_rep2) reproduces each winner's step time within 1% and the same choice on all ten (two ties between
equivalent candidates resolve either way). One rejected finalist, the adaptive dual-bucket variant on 48-mid, is bimodal: 522.6 ms in
the first round and 1173.7 ms in the repeat with no code change, i.e. the backend lowers the same graph differently from
one compile to the next; the chosen 4-bucket candidate is stable (492.4 / 492.8). The first gate round (results/gate_v11)
already reached the optimum on seven; the second round (results/gate_final; Phase 3 with twice the refinement budget,
4/8/16-count in-order bucket seeds, finalists merged across runs) closed hetero_grad_ar_many, hs_c and 48-mid. Without the
gate the simulator-ranked top pick would have lost on hetero_grad_ar_many (bundle 3.2x slower), zero2 (dim-1 concat 10x
slower) and, in earlier runs, on hs_a and 48-mid (sorted buckets 2-3x slower): all candidates the simulator cannot
separate from a good bundle by bytes and dispatch count.

**History of the search-side fixes that got here** (results/day4_picks_table.txt): the first picks lost on hs_a (flat
0.89x, size-sorted buckets 0.57x), 48-small (sandbox-scale thresholds, preallocated flat), 48-mid (flat, sorted bins
0.75x) and hetero_grad_ar_many (flat 7x slower). Fixes: size-independent in-order bucket builtins (4/8/16 tensors),
bf16-flat and two-dtype seeds for the mixed-dtype problem, prompt hints that sandbox sizes are scaled and that bundles must
keep the deployment order, two sandbox holes closed (unequal concat feeding a reduce-scatter; slice-assigned collective
results treated as dead code), and the gate.

## 5. Same workload, different setup (days 3e and 4)

| variant | result |
|---|---|
| model_grads_fp32 (fp32, dtype-honoring runner) | inversion holds: bench 0.76 -> training 3.37x (flat) |
| layer_grads_mixed_dtype (real dtypes) | inversion holds: bf16 flat 0.92 -> 1.67x; fp32 flat 0.70 -> 1.39x; two-dtype 0.50 -> 1.19x |
| model_grads_ep_groups (4-rank DP groups spanning nodes) | grouped collectives fail NCCL communicator init at 64 ranks inside the training graph (runner now marks these INVALID) |
| model_grads_intra_node_groups | lost with the day-3 block; not rerun |

## 6. Harness corrections

* **Dtype confound (found 2026-10-28 from an HLO diff).** The training runner derived all pieces from one bf16 matmul
  buffer, so every tensor was bf16 in training; the day-2 "two_dtype_groups" candidate degenerated into one flat concat
  of all six tensors and the day-2 mixed-dtype and fp32 numbers were the plain layer/model inversions. `g_train.py` v5
  casts each piece to its declared dtype; the rows above are from that runner.
* **Grouped collectives.** All 64-rank training results of candidates with replica `groups` were invalid (NCCL init
  failure, dispatch-floor step times); withdrawn, runner records `collective_effective` and rejects such runs.
* **Pair-to-pair variance.** Latency-bound problems (hs_a, 48-small) run ~1.7x slower on the day-4 instance pair than
  on the day-2 pair with identical bench numbers; compare ratios within a day.

## 7. Open items

* The training runner (`g_train.py`) has no output check; grouped collectives that fail NCCL init at 64 ranks
  produce floor-time 'results'. Add a reference check (as `g_bench2.py` has) before the next block.
* Capacity Blocks reclaim instances 30 minutes before the end time: plan the last pull accordingly.

* The unequal-concat pathology has no predictive cost term (trigger below the HLO level: a 4 KB appended piece cures
  it, equal pieces of the same total are fast); hetero_grad_ar_many and zero2_hetero_rs need a training-scope hardware
  gate. The search framework's training-validation hook (`_run_training_validation`) is the place to wire it.
* `collective_permute` is a correctness/availability cross-scope failure rather than a speed inversion.
* 7-node confirmation of the eight problems and of the fitted constants (needs a 7-node block).

## 8. End-to-end footprint of the new problems in real 10B models (2026-10-29, 7-node block)

Two questions were asked of the new problems: would their winners show up in a real ~10B model, and would a 10B Llama
built on them reach the 3x of the amplified Llama shell? Both answers are no, for a measured reason.

**OLMoE-10B.** The OLMoE harness times each step section with a host sync (`STEP_TIMER`). In the 7-node campaign runs
(224 ranks, paper configuration) the per-tensor replicated-gradient sync is 38 ms of a 4257 ms baseline step (0.9%) and
37 ms of a 3035 ms OverlayCCL step (1.2%); both arms run it per-tensor (the paper toggles AllToAllV and dxe only). The
replicated set is ~204M elements in ~50 pieces (25 MB qkv, 8 MB o, a 132 MB embedding, routers, 2 KB norm gains). A 1.7x
win on that site is 15 ms of the step.

**10B Llama, TP=32 x DP=7.** The real per-rank DP gradient set of a 9.75B Llama (48 layers, DM 4096, FF 11008: per
layer qkv 384x4096, o 4096x128, gate_up 688x4096, down 4096x344, two RMSNorm gains; plus embedding/head shards and the
final norm; 291 pieces, 312M elements bf16) registered as `llama10b_dp_grads` (g_newprobs_10b.py) and run in the
224-rank training harness (`results/sy7_10b`, 60 steps):

| strategy | measured step (ms) | v11 simulator (us, 64-rank scale) | paper's original simulator |
|---|---|---|---|
| per-tensor (291 all-reduces) | 331.9 | 11,349,140 | 774,429 |
| 32 MB buckets | 327.5 | 952,358 | 788,968 |
| per-layer buckets (6 pieces) | 344.0 | 928,698 | 785,630 |
| 16-piece in-order buckets | 322.1 | 824,849 | 785,630 |
| 128 MB buckets | 349.7 | 1,329,765 | 785,630 |

The set is bandwidth-bound and strategy-insensitive on hardware (all five within +-3%). The paper's original simulator
(heterogeneous-set and unequal-concat terms off: `HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0`) predicts exactly that, a tie
within 2%. The v11 heterogeneous-set term predicts a 12x per-tensor penalty that does not exist: it was fitted at 64 ranks on
sets of 6 to 64 pieces and a few hundred MB, and its bytes-proportional form does not extrapolate to a 291-piece, 624 MB set
at 224 ranks. No problem in the 71-candidate regression pool resembles this set, so the zero-regression check could not catch
it. Disabling the tiny-mix gain alone leaves a 3.7x predicted penalty; disabling the count knee changes little
(`sim/sim_newprobs_v5_llama10b*.json`). The term has to be re-fit as a bounded per-set excess against 224-rank data before
it is used for anything at this scale; a piece-count sweep of the same set truncated to 8 and 24 layers
(`llama10b_dp_grads_L8/_L24`, `results/sy7_10b_sweep`) was run for that purpose.

**Consequences.** (1) Gradient sync of a ~10B-class replicated set is a few percent of a real step at these shapes, so the new
problems cannot move a real 10B model's end-to-end number; the 3x of the Llama shell comes from the four amplified sites and
would in any 10B shell. (2) The new problems' two genuinely new ingredients, the bf16 wire cast (layer_grads_mixed_dtype) and
pathology-aware grouping (count-bounded in-order buckets, the two-way split on hs_a, shape-class bundles on
hetero_grad_ar_many), pay only in the per-problem regime: 48 or more pieces of 64 KB to a few MB, a few hundred MB in total.
An e2e shell that shows them must carry a replicated set built in that regime, and should be labeled as such; the amp5
shell's 12 pieces of 58 MB was the wrong regime (per-tensor won there in both graph forms). (3) Under the paper's
simulator-decides rule, the cast converges (the simulator ranks the bf16 flat bundle first because the cast halves bytes;
1.67x on hardware); grouping converges where dodging the pathology also minimizes dispatch count (hs_a first pick 1.35x;
48-small after the correctness gate removes a non-running first pick, 1.57x), is partial on 48-mid and hs_c (bimodal dual
bucket 1.14x/0.51x; RS-pipelined 1.03x), and fails on hetero_grad_ar_many (shape-class and 16-piece buckets 0.5% apart in
the simulator, 65 vs 237 ms on hardware). The AR -> reduce-scatter + all-gather decomposition proposed by the LLM on 24-mid
and hs_c ranks first in the simulator but never wins on hardware (1.11x, 1.03x against 1.14x, 1.16x).

**Piece-count sweep of the same set (224 ranks, 60 steps, `results/sy7_10b_sweep`).** The 10B layer set truncated to 8 and 24
layers (49 and 145 pieces, ~100 and ~300 MB bf16):

| layers (pieces) | per-tensor | 16-piece buckets | flat concat | flat speedup |
|---|---|---|---|---|
| 8 (49) | 62.9 ms | 57.2 | 49.2 | 1.28x |
| 24 (145) | 182.9 | 141.7 | 130.6 | 1.40x |
| 48 (291) | 331.9 | 322.1 | not run (32 MB buckets 327.5) | ~1.03x |

The per-tensor penalty is real on this set up to 24 layers and disappears at 48 layers (624 MB), so the heterogeneous-set term is
right in sign but wrong in shape at the top end: a bounded (saturating) excess rather than one proportional to bytes. Note that a
flat concatenation of 49 and 145 unequal pieces compiled and was the fastest candidate here, so the >40-piece unequal-concat
pathology is not universal either; it has so far appeared on 2-D operand sets (hetero_grad_ar_many) and the catk sweeps.

**No-simulator ablation on the chosen problems (2026-10-29, 224 ranks, `results/sy7_nosim*`, `run_phase3_nosim.py`).** The
paper's no-sim arm (strategy enumeration without refinement, pick = first correct candidate in enumeration order) ships:
mixed-dtype grads -> "hierarchical bucketing with dtype preservation" (3 in-order pairs): 41.0 ms vs per-tensor 58.8 (1.43x; the
simulator's pick, the bf16 flat cast, is 37.4 ms, 1.57x); 48-small -> "single flat concatenation": 52.5 vs 50.4 ms (0.96x, a
loss; simulator's pick 30.8 ms, 1.64x); hs_a -> "two-phase hierarchical bucketing": 103.6 vs 101.9 ms (0.98x; simulator's
pick 70.8 ms, 1.44x). Without the cost signal the loop lands on bundles that the pathology makes slow on two of the three.

## 9. Placement of the gradient sync decides the ranking (2026-10-29, 224 ranks)

The amp6 Llama shell (`experiments/model_extension/train_llama_e2e_amp6.py`: amp3's four sites plus two gradient-sync sites
carrying exactly the mixed-dtype layer set and the 48-small set) gave, with one sync per step in a separate graph after the
backward: per-tensor 23.4 ms, the two per-problem winners 50.75 ms, textbook-DDP baseline (per-microbatch per-tensor) 96.0 ms.
The winners lose by 2.2x at the sync site, as every bundle did in amp5. A runner that reproduces that placement on the
per-problem sets themselves (`g_train_v8.py`: every piece its own parameter, gradients synced in their own graph after the
backward, `results/sy7_v8`) shows the same thing and is 4-5x faster per step than the harness the inversions were measured
with:

| set (224 ranks) | g_train (collective in the forward on views of one flat buffer, re-issued in the backward): per-tensor / winner | v8 (DDP placement): per-tensor / winner |
|---|---|---|
| hs_a (64 x 64K-512K) | 101.9 / 70.9 ms (1.44x) | 22.1 / 24.1 ms (0.92x) |
| 48-small | 50.4 / 30.8 (1.64x) | 17.1 / 18.4 (0.93x) |
| mixed-dtype layer set | 59.0 / 37.8 (1.56x) | 13.8 / 14.9 (0.92x) |

So the heterogeneous-gradient inversions are a property of the harness placement, not of gradient synchronization in a
training step with separate gradient buffers. The remaining question, under test (`g_train_v9.py`, `GRAD_BUFFER=flat` in
amp6), is whether the slow per-tensor regime reappears when the gradients live in one contiguous per-dtype buffer
(Megatron-LM / DeepSpeed layout, where each gradient is a view of the buffer), which would make the inversion a
gradient-buffer-layout effect rather than an artifact. Until that is settled the new problems' training-scope rows must be
read as "forward-collective placement on a flat buffer".

**Contiguous-buffer test (v9, `results/sy7_v9`, 224 ranks).** With every gradient a view of one preallocated per-dtype buffer
(Megatron-LM / DeepSpeed layout) the mixed-dtype layer set runs at 80.1 ms per-tensor and 81.4 ms with the bf16 flat cast:
both strategies slow down ~6x relative to separate buffers (13.8 / 14.9 ms) and stay tied, so the layout does not recreate the
inversion either. The two 1-D sets (hs_a, 48-small) do not compile in this layout (neuronx-cc internal error NCC_ITEN404,
"LegalizeSundaAccess", in the gradient-accumulation graph). Conclusion so far: the heterogeneous-gradient inversions belong to
the original harness's forward-collective placement; neither DDP-style separate buffers nor a contiguous buffer reproduces them.

**Shell-level contiguous buffer (amp6 `GRAD_BUFFER=flat`):** the training graph with every replicated gradient a view of one
buffer does not compile at 224 ranks (neuronx-cc NCC_EBVF030, >5M instructions) for either strategy, so the layout cannot be
compared in the shell either.

**Where the +27 ms is not (diag2, 224 ranks).** Re-issuing the sync twice per step (SYNC_REPEAT=2) adds 3.8 ms for the two
winners (50.75 -> 54.6) and nothing measurable for per-tensor (23.4 -> 22.4), so the collectives themselves account for at most
~4 ms of the 27 ms gap; the rest is paid elsewhere in the sync+update graph. Storing the site-A weights (in, out) and applying
them without a transpose makes the winners' arm slower still (96.6 ms; per-tensor unchanged at 23.1), so a simple
"transposed gradient layout" story is not it either. On the bare mixed-dtype set, consuming the matrices through x @ W.t()
(v8t) leaves per-tensor vs cast at 13.5 vs 14.6 ms. Attribution runs (site A only / site B only / cast without bundling,
split sync and update graphs, HLO dumps) follow.

**Attribution (diag3, 224 ranks, isolated sync graph).** Site B's 8-piece buckets are harmless: with site A per-tensor the step is
22.0 ms (per-tensor everywhere: 23.4). Site A's flat bf16 bundle is the whole cost: with site B per-tensor the step is 42.5 ms.
The cast ingredient without the bundle (per-tensor all-reduce in bf16, cast back) costs ~1 ms: 24.5 ms with site B's buckets.
Splitting the sync and the update into separate graphs makes the bundle worse (61.7 vs 24.5 for per-tensor), so the cost sits in
materializing the flat 100 MB bundle and its six slices as graph outputs/inputs, i.e. in layout conversion of the matrix
gradients, not in the collective (re-issuing the collectives adds 3.8 ms). On the bare set the same bundle costs ~1 ms because
the gradients of an elementwise or a 64-row matmul loss carry the default layout; in the shell the compiler assigns the
site-A weight gradients the layout of the big matmuls, and flattening them is a relayout copy.

## 10. Hunt for inversions with other strategy types (2026-10-29, 7-node block)

Candidates at sites the Llama shell already has, with strategy types other than bundling, grouping or casting
(`g_newprobs_hunt.py`, `results/sy7_hunt*`): all-gather LAYOUT at the FSDP prefetch (pinned default vs native lowering vs
leading-dim gather with transposes), all-reduce DECOMPOSITION at the TP MLP site (one all-reduce vs reduce-scatter +
all-gather, each in both layouts), sequence-parallel RECOMPOSITION (reduce-scatter, norm on the token shard, all-gather) and a
4-chunk pipelined all-reduce, a two-level HIERARCHICAL all-reduce (intra-node reduce-scatter, inter-node all-reduce, intra-node
all-gather), a reduce-scatter PIPELINE TRANSFER replacing the masked all-reduce, all-gather prefetch ORDERING with the matmuls
inside the problem, and a one-all-gather cross-entropy statistics merge.

**First confirmed candidate: all-gather layout (fsdp_ag_layout), 224-rank training step (g_train, forward placement):**
pinned default 37.4 ms, leading-dim gather + transposes (pinned) 40.4 ms, leading-dim gather with the native lowering 24.8 ms
(1.50x). The native lowering only compiles for a gather along the leading dimension (a dim-1 native gather fails with
NCC_ITEN404 "Vague replica_groups"), which is why the strategy needs the transpose pair. 32-rank bench: the two transposed
variants fail to compile in the bench harness, so the bench scope cannot even rank them. 7-node microbench pending.

**Simulator v12 (additive, `correctness_test_v5.py`):** `xm.all_gather` accepts `pin_layout`; the native lowering is charged
`AG_NATIVE_BYTES_FACTOR` = 0.5 of the pinned bytes (the pinned lowering is a zero-padded all-reduce of the output buffer, i.e.
~2x the wire bytes; the all-gather curve was calibrated on pinned gathers). `all_reduce` accepts and ignores `pin_layout`;
`TrackedTensor.t()` added. Pool regression: 0 of 71 candidates change (`/tmp/simrun/sim_v4_v12.json` vs v11). Scores:
pinned 52358 us, dim0+transpose 52856, native 30552 (1.71x; hardware 1.50x), order correct.

**Loop (search registration `fsdp_ag_layout` in `search/problems_newprobs.py`, strategy-enumerate, Sonnet 4.5, 108 s):**
the LLM proposes the transpose-to-leading-dim native gather itself (strategy 2) and a stacked variant; refinement produces
"refined-from-Transpose-normalize to dim-0 native gather" at 30054.6 us, the paper-rule pick. Its hardware time and the
7-node microbench are queued (`results/sy7_hunt_picks`); the e2e knob `FSDP_AG=dim0_t_native` in amp6 is queued as well.

**Hunt round 2 outcome (224-rank training step, g_train forward placement; `results/sy7_hunt/train`).**
| problem | baseline | alternatives | verdict |
|---|---|---|---|
| tp_seq_parallel (AR+norm vs RS, norm on shard, AG; 4-chunk AR) | 56.3 ms | 57.5 / 58.4 / 57.2 | tie |
| fsdp_prefetch_order (sequential vs next-layer prefetch vs all-first) | 58.0 | 57.3 / 57.1 | tie (ordering invisible to the compiler-scheduled graph) |
| dxe_ag_stats (two ARs vs one AG of per-rank stats) | 6.54 | 8.94 / 8.83 | all-reduce wins; no inversion |
| hier_ar (flat vs intra-node RS + inter-node AR + intra-node AG, AR+AR, 2 chunks) | timeout | timeouts | replica-group collectives at 224 ranks hang; flat run hung in the same batch; not measurable here |
| pp_transfer_rs (masked AR vs RS over a (ws, ...) buffer) | NRT_INVALID | 70.5 / 70.4 | baseline invalid in this harness (the shell's masked AR works at e2e scope); inconclusive |
| fsdp_ag_layout (pinned vs native leading-dim gather) | 37.4 | 40.4 / **24.8** | **inversion, 1.50x** (see above) |
So of seven strategy types tried, one (collective lowering/layout at the FSDP site) gives a confirmed inversion; recomposition,
chunking and ordering are ties at this scale; hierarchy and the RS transfer could not be measured.

**7-node microbench (224-rank lone call, `results/sy7_hunt/logs/bench224_*`).** fsdp_ag_layout: pinned 16.1 ms; every transposed
or native variant FAILS TO COMPILE in the lone-call graph (NCC_ITEN404 "Vague replica_groups"), although the same code compiles
and runs inside the training graph (40.4 / 24.8 ms). So for this problem the microbench cannot rank the winner at all: it is a
compilability inversion between scopes on top of the 1.50x speed inversion, and only the simulator (v12) or the training run can
pick it. Others at 224 ranks lone-call: hier_ar flat 8.2 / 2-chunk 9.3 / RS+AR+AG 10.8 / AR+AR 46.4 (replica-group collectives
run in the lone-call graph but hang inside the training graph); dxe two-AR 3.5 vs one-AG 5.4 / 4.6 (AR wins in both scopes);
tp_seq_parallel AR+norm 5.1 vs RS variants 6.1-6.3 (AR wins at bench, tie in training); fsdp_prefetch_order 17.3-17.5 (tie);
pp_transfer_rs masked AR 0.51 ms vs native RS 27.0 (the masked-AR lone call is at the dispatch floor and its training run is
NRT_INVALID; inconclusive).

**fsdp_ag_layout in the shell (amp6, 224 ranks, isolated sync graph, the two sync sites present and synced per tensor;
`results/amp6`).** Bundled arm (6 gathers per step, hoisted): pinned 22.17 ms, native leading-dim 22.82, pinned leading-dim +
transposes 24.02: no gain. Per-microbatch baseline (24 gathers per step): pinned 96.0 ms, native 74.2 ms (-22 ms, ~0.9 ms per
gather). So the native lowering is faster inside the shell when the gathers sit in the per-microbatch graphs, and invisible when
the six hoisted gathers share one graph with the step's compute. Under the paper's convention (the developer baseline keeps the
default pinned gather) the OverlayCCL arm gains nothing from it. **tp_ar_decomp (corrected token count):** bench 3.2 / 3.1 / 3.6 /
3.0 ms, training 55.7 / 55.6 / 58.1 / 56.1 (AR, native AR, RS+AG, native RS+AG): tie.

## 11. Reframing (2026-10-29 evening): the forward-collective harness is the microbench, the e2e-like placement is the training

With the shell measurements above, the scopes of the heterogeneous-gradient problems are re-labeled:

* **microbench** = `g_train.py` (v4/v5): the collective issued inside the forward on views of one flat buffer and re-issued in
  the backward. It is a microbench of the collective itself (its ratios are what a developer sees when benchmarking strategies in
  isolation), and it favors the bundles: cast+flat 1.67x, 8-piece buckets 1.57x, two-bucket split 1.35x at 64 ranks.
* **training** = `g_train_v10.py` (the "mirror" harness): the set's pieces are parameters of a small model on 1024 tokens, 2-D
  pieces applied nn.Linear-style (x @ W.t()), DM-sized vectors as RMSNorm gains, other vectors through the mean hidden state;
  N_MB=4 microbatches accumulate into separate gradient buffers; one synchronization per step in its own graph after the
  backward, then the SGD update. This is the placement of the Llama e2e shell, where per-tensor beat the bundles (23.4 vs
  50.75 ms) and the DDP-placement runner v8 showed per-tensor ahead by 7-8%.
* **developer baseline** for these problems = the microbench winner (what a developer who benchmarked strategies in isolation
  would deploy: the bf16 flat bundle, the 8-piece buckets); **OverlayCCL's pick** should be per-tensor, which the paper's
  ORIGINAL simulator constants already rank first on mixed-dtype grads (123910 vs 124131 us), 48-small (32292 vs 33973),
  hs_c, 24-mid and 48-mid. The fitted v11 heterogeneous-set term (calibrated on the microbench) inverted those rankings and
  made the loop ship the microbench winners into the shell; it is withdrawn. The additive sandbox API extensions (bf16/fp16
  dtypes, `pin_layout`, `.t()`) stay; the v12 layout term stays (it changes no pool score).
* This is the same inversion the paper reports for its eight problems, mirrored: there the developer default is per-call and
  bundling wins in training; here the microbench recommends bundling and per-tensor wins in training.

Runs in flight: Phase 3 with the original constants on all ten problems (`phase3/phase3_orig_*.log`, tag `orig`), and the v10
harness at 224 ranks on the nine all-reduce problems (`results/sy7_v10`): per-tensor vs the microbench winners.

**Loop with the original constants (tag `orig`, `phase3/phase3_orig_*.log`), Phase-5 pick = lowest sim among all survivors:**
layer grads -> per-tensor (16707 us; flat 16958); model grads -> per-tensor (63214; flat 64218); 24-mid, 48-mid, hetero-many,
zero2 -> per-tensor; hs_c -> per-tensor (122487; the LLM's RS+AG 123790); **hs_a -> LLM "single flat all-reduce" refinement
(72951 vs per-tensor 74225, 1.7%)**; **48-small -> LLM "in-order buckets of 12" refinement (59956 vs 60032, 0.1%)**;
**mixed-dtype -> cast_flat_bf16 (9281 vs per-tensor 10458, 11%, because the cast halves the bytes)**. So the original
constants pick per-tensor or an equivalent on 7 of 10 and a bundle on three; whether those three are right is what the v10
mirror harness measures (`results/sy7_v10`, candidates `orig_1`). If the shell's 2x per-tensor advantage on the mixed set
reproduces there, the missing cost is the relayout of flattening matmul-produced 2-D gradients, a local-copy term.

**Clean amp3-equivalent shell (amp6 `SITES=0`: embedding + head the only replicated tensors), isolated sync graph, 300 steps
(`results_amp6_7node`, tags `c7_/c5_/c3_`):**

| nodes | baseline (per-mb, pinned AG) | bundled, pinned AG | bundled, native leading-dim AG | speedup pinned / native |
|---|---|---|---|---|
| 7 (seed 0) | 20.41 ms | 5.93 | 4.95 | 3.44x / 4.12x |
| 7 (seed 1) | 21.26 | 5.50 | 4.10 | 3.87x / 5.19x |
| 5 | 18.36 | 5.00 | 5.04 | 3.67x / 3.64x |
| 3 | 14.76 | 5.34 | 4.52 | 2.76x / 3.27x |
| 7, baseline with the native AG | 19.36 | | | (baseline gains 5%) |

Without the two gradient-sync sites the native all-gather DOES transfer to the OverlayCCL arm: -17% / -25% at 7 nodes and -15%
at 3 nodes (tie at 5 nodes), i.e. the FSDP-layout inversion is an e2e gain once the step is not dominated by the 250 MB sync.
Loss identical across arms (3.466 per-microbatch-normalized).

**Per-site probe inside the clean shell (PERCALL_PROBE=1, host sync around each probed call; `results_amp6_7node/json/*p7_*percall*`).**
FSDP gathers measured in isolation inside the step: pinned 0.776 ms, native 0.771 ms per call, identical; yet the step medians of
the same runs are 5.75 vs 4.65 ms. The native lowering pays off only when the gathers are scheduled inside the step graph (the
probe's graph cut turns each gather into a lone call and the difference vanishes), which is one more instance of the same
cross-scope effect: lone-call measurement, even inside the training script, misses it.

**Loop convergence on fsdp_ag_layout under the paper's rule (`results/sy7_hunt_picks`, 224 ranks, repeat run).** The
simulator's first pick, the LLM's "dimension-aware native gather" (native lowering along dim 1 without transposes, 30055 us),
does not compile at training shape (NCC_ITEN404 "Vague replica_groups"), so the correctness gate removes it; the next survivor,
the builtin leading-dim native gather with transposes (30552 us), measures 24.87 ms against the pinned developer baseline's
36.62 ms (1.47x; first round 1.50x). The loop therefore ships the measured winner by its own rule, with no performance gate.

**Mirror-harness validation at 1 node (32 ranks, 10 steps; `v10_test`).** Gradient harness v10: hs_a per-tensor 68.98 ms vs
the two-bucket winner 69.39 ms (tie); forward-gather harness v11ag on fsdp_ag_layout: pinned 21.78 ms vs native leading-dim
20.71 ms (1.05x). Harness notes: one graph per microbatch; 2-D pieces are applied nn.Linear-style (x @ W.t()); vectors enter
through an elementwise term (both an einsum and a matmul over the concatenation of a 64-vector set compiled to a
million-instruction graph on this compiler); DM-sized vectors act as RMSNorm gains.

## 12. Per-problem TRAINING in the mirror harness, 224 ranks, round 1 (`results/sy7_v10`, 100 steps, N_MB=4)

| problem | per-tensor | microbench winner (developer baseline) | other candidate | per-tensor vs microbench winner |
|---|---|---|---|---|
| hs_a (64 x 64K-512K) | 79.8 ms | 4 MB buckets 124.2 | two-bucket split 78.8 | **1.56x** |
| 48-small | 60.7 | 6 buckets 63.0 | 8-piece buckets 61.8 | 1.04x |
| layer grads | 33.3 | flat 35.2 | flat (loop) 36.0 | 1.06x |
| mixed-dtype grads | 34.2 | bf16 flat cast 35.8 | dtype pairs 36.3 | 1.05x |
| model grads (24) | 117.6 | flat 118.2 | two-bucket 118.3 | 1.01x |
| 24-mid | 50.5 | 6 buckets 57.2 | flat 66.7 | 1.13x (1.32x vs flat) |
| hs_c | 46.5 | 4 MB buckets 49.0 | 16-piece buckets 49.2 | 1.05x |
| 48-mid | 93.3 | 6 buckets 114.2 | 4-piece buckets 113.4 | 1.22x |
| hetero-many (48 2-D) | 129.9 | shape-class bundles 122.3 | two-phase shape-class 121.8 | 0.94x (bundling wins) |

In the e2e-like placement per-tensor is the fastest strategy on 8 of 9 sets, by 1 to 6% on the medium sets and by 1.13-1.56x on
the probe sets whose microbench favored buckets; hetero_grad_ar_many is the exception, where grouping by shape class stays 6%
ahead. The paper's original simulator constants pick per-tensor on all of these except hs_a, 48-small and mixed-dtype (where
it prefers a bundle by 1.7%, 0.1% and 11%) and hetero_grad_ar_many (per-tensor, 0.25% ahead of shape-class); rounds 2-5 and
the loop picks follow.

**Original-constant loop picks measured in the mirror harness (224 ranks, `results/sy7_v10`, candidates `orig_1`).** hs_a: the
LLM's single-flat refinement 87.4 ms vs per-tensor 79.8 (the pick, 1.7% ahead in the simulator, is 10% behind on hardware);
48-small: the 12-piece in-order buckets 61.3 vs per-tensor 60.7 (tie, as the 0.1% simulator margin suggests); mixed-dtype:
the bf16 flat cast 35.8 vs per-tensor 34.2 (the 11% simulator margin is a 5% loss on hardware); hs_c: the RS+AG decomposition
does not run at 224 ranks (operand not divisible by the rank count), so the correctness gate removes it and the loop ships
per-tensor. Net: under the paper's rule with the original constants the loop ships per-tensor or an equivalent on 7 of 10, a
10% loss on hs_a, a 5% loss on mixed-dtype and a tie on 48-small; no pick is a large regression.

**fsdp_ag_layout per-problem TRAINING (v11ag: gather inside the forward feeding a matmul, true reduce-scatter backward, N_MB=4;
`results/sy7_v11ag`), 224 ranks, round 1:** pinned 102.9 ms, leading-dim + transposes pinned 117.6, leading-dim native 91.6
(1.12x over the pinned developer default; 1.05x at 1 node; 1.50x in the forward-collective microbench; 17-25% on the e2e step).

**Search-cost rows with the original constants (tab:ablation-cost / cost-calculus), ten new problems, laptop LLM wall:**
strategy-enumerate 854 s (14 min, ~10 LLM calls per problem), cc-react 4026 s (67 min), multi-island 3704 s (62 min). Picks:
strategy-enumerate as listed above; multi-island converges to flat concatenation evolutions on 7 of 10 (bucket variants on hs_a
and 48-small, cast_flat_fp32 on mixed-dtype, flat-then-slice on zero2), cc-react returns a single final candidate per problem.
No-simulator picks (`phase3/phase3_nosim_*.log`): single flat all-reduce on 7 of 9 all-reduce problems, hierarchical bucketing
on model grads and mixed-dtype, two-phase buckets on hs_a; they are measured in the mirror harness (`sy7_v10_nosim`) and as the
e2e no-sim arm.

## 13. amp7 e2e: four amp3 sites + the FSDP gather layout + five gradient-sync sites (layer, mixed-dtype, hs_a, hs_c, 48-small)

Developer baseline: per-microbatch graphs at the four sites, pinned gathers, and at every gradient site the microbench-winning
strategy (flat / bf16 flat cast / 4 MB buckets / 6 buckets) synced per microbatch. OverlayCCL: bundled graphs, native leading-dim
gathers, accumulate-then-sync with per-tensor all-reduce. Control: OverlayCCL's schedule with the microbench strategies.
300 steps, isolated sync graph, loss identical across arms (3.466 per-microbatch-normalized). `results_amp7_7node`.

| nodes | baseline | OverlayCCL | control (schedule only, microbench strategies) | speedup | strategy share at equal schedule |
|---|---|---|---|---|---|
| 7 (224r) | 600.1 ms | 79.1 ms | 164.1 ms | 7.59x | 2.07x |
| 5 (160r) | 600.6 | 70.3 | — | 8.54x | — |
| 3 (96r) | 599.0 | 69.6 | — | 8.61x | — |
| 7, S=2048 M=4 | 644.4 | (bundled arm lost to the chain swap; re-run if time permits) | | | |

Reading: the shell is gradient-sync dominated (the four amp3 sites are ~5 ms of the 79 ms OverlayCCL step), so the ratio
decomposes as 3.66x from accumulate-then-sync (bundling in time) times 2.07x from choosing per-tensor over the microbench
bundles; the latter is the new problems' contribution at e2e scale. Compile walls (for the break-even count) come from the
cold-cache pair, queued.

**amp7 e2e, seed 1 (7 nodes):** baseline 605.6 ms, OverlayCCL 70.4 ms (8.60x; seed 0: 7.59x). Knob grid, 7 nodes: S=2048 M=4
baseline 644.4 (bundled arm lost to a chain swap); S=1024 M=8 baseline (see chain log) / OverlayCCL 106.9. Cold-cache pair
(fresh compile cache on every node, 300 steps): baseline wall 857 s incl. compile with steady 595.4 ms; OverlayCCL wall 491 s
with steady 72.3 ms, so OverlayCCL is ahead from the first step (no break-even point: its graphs also compile faster).

**Knob grid (7 nodes, tab:llama-sweep rows):** S=2048 M=4: baseline 644.4 ms (OverlayCCL arm lost to a chain swap); S=1024 M=8:
baseline 1181.5 ms, OverlayCCL 106.9 ms (11.05x; the baseline pays eight per-microbatch syncs). Rounds 2 and 3 of the per-problem
training at 224 ranks completed without failures (`sy7_v10_r2/r3`, `sy7_v11ag_r2/r3`; 60 steps each); table below from the
regenerated paper_tables_new.md (n=3 so far, rounds 4-5 queued after the 1-node block).

## 14. Simulator fidelity and constant-swap rows for the new problems (`results/fidelity_new.md`, `fidelity_new.py`)

Simulator = the paper's original constants; hardware = mirror-harness 224-rank training medians over all rounds; candidates =
per-tensor, the microbench winner, the loop's pick, the no-sim pick (3-4 per problem). Rho 0.74-1.00 on eight of nine (hs_a 0.40,
where the loop's flat refinement is ranked first and measures 11% behind per-tensor); top-1 correct on 7 of 9 (hetero-many: 0.4%
regret; hs_a: 11%). Constant swaps: the pick is unchanged on 9/9 problems for the op-cost, graph-reset, knee and fwd/bwd arms
and for network x2; network x1/2, x1/4 and x4 flip 1-3 picks (hetero-many, hs_a, 48-small, hs_c).
# Simulator fidelity (original constants) vs mirror-harness 224-rank training, new problems

| Problem | n | rho | top-1 | regret | ratio-MAPE |
|---|---|---|---|---|---|
| hetero_grad_ar_many | 4 | 0.74 | no (p3_pick3) | 0.4% | 6% |
| hs_a_64x64k_512k | 4 | 0.40 | no (orig_1) | 11.0% | 20% |
| hs_c_32x256k_2m | 4 | 0.77 | yes | 0.0% | 3% |
| layer_grads_hetero_ar | 4 | 0.77 | yes | 0.0% | 6% |
| layer_grads_mixed_dtype | 3 | 0.87 | yes | 0.0% | 5% |
| model_grads_4layers | 4 | 0.77 | yes | 0.0% | 11% |
| probe_het_24_mid | 4 | 0.77 | yes | 0.0% | 23% |
| probe_het_48_mid | 4 | 0.77 | yes | 0.0% | 49% |
| probe_het_48_small | 4 | 1.00 | yes | 0.0% | 6% |

# Constant-swap arms (* = the swap changes the simulator's pick)

| Arm | hetero_grad_ar_many | hs_a_64x64k_512k | hs_c_32x256k_2m | layer_grads_hetero_ar | layer_grads_mixed_dtype | model_grads_4layers | probe_het_24_mid | probe_het_48_mid | probe_het_48_small | agree. |
|---|---|---|---|---|---|---|---|---|---|---|
| calibrated (original constants) | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| network x1/2 | 0.26* | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 0.95 | 8/9 |
| network x1/4 | 0.26* | 0.40* | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 0.74* | 6/9 |
| network x2 | 0.74 | 0.40 | 0.32 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| network x4 | 0.74 | 0.40 | -0.74* | 0.77 | 0.87 | 0.95 | 0.77 | 0.32 | 0.80* | 7/9 |
| op costs = isolated | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| run reset per graph | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 0.95 | 9/9 |
| amort. knee 64 KB | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| amort. knee 1 MB | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| fwd+bwd factor 0 | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
| fwd+bwd factor 2 | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | 0.77 | 0.77 | 0.77 | 1.00 | 9/9 |
fsdp_ag_layout row: the harness candidates share a helper the sandbox adapter does not carry, so its fidelity row uses the search
registration's builtin scores from the Phase-3 run (v12 layout term): pinned 52358, leading-dim pinned 52856, leading-dim native
30552 us against 102.9 / 117.6 / 90.3 ms in training: rho 1.00, top-1 correct, ratio-MAPE 24% (the simulator over-predicts the
native gain, 1.71x vs 1.14x measured).
