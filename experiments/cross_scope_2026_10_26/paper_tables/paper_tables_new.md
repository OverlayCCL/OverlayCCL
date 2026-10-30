# New-problem rows for every paper table (2-node data of 2026-10-27/28; 7-node columns pending)

## tab:problems — rows to add ("Gradient-sync side")

| Problem | Role in the training step |
|---|---|
| Layer grads AR | DP all-reduce of one layer's heterogeneous gradients (q/k/v/o, up+gate, down, norms) |
| Model grads AR | DP all-reduce of all layers' gradients; bucket-count choice |
| Mixed-dtype grads AR | bf16 weight gradients with fp32 norm gradients |
| Hetero-many grads AR | 48 odd-width 2-D gradients; grouping choice |
| ZeRO-2 hetero RS | reduce-scatter of heterogeneous gradient shards |
| Bucket-count probes (hs_a, hs_c, 24/48-piece) | mixed-size 1-D gradient sets; bucket-count calibration instances |

## tab:setup — column to add

| | Grad-sync harness |
|---|---|
| Scale, steps | 2n·64r, 300 (7n·224r pending) |
| Model shell | matmul → collective → loss → backward (collective re-issued in backward) |
| Shapes | Llama-7B-class layer (DM 2048, HID 5376), 1-D mixed sets 64K-8M |
| Collective(s) | one per script (gradient all-reduce / reduce-scatter) |

## tab:perproblem — rows to add (ms; 1n bench = 32 ranks, 2n bench = 64 ranks, 2n train = 64-rank training step; 7n columns pending)

| Problem | 1n bench base | 1n bench Overlay | 2n bench base | 2n bench Overlay | 2n train base | 2n train Overlay | train speedup | Overlay strategy |
|---|---|---|---|---|---|---|---|---|
| Layer grads (6 hetero) | **9.46** | 10.43 | 13.75 | **12.88** | 61.1 | **36.6** | 1.67x | Single flat concatenation all- |
| Model grads (24 hetero) | **25.01** | 28.25 | 37.78 | **35.89** | 431.4 | **96.9** | 4.45x | flat_cat_ar |
| Mixed-dtype grads | **9.68** | 10.40 | 13.95 | **13.21** | 61.0 | **36.5** | 1.67x | cast_flat_bf16 |
| Bucket count A (64 x 64K-512K) | **8.94** | 11.30 | **13.35** | 14.19 | 98.9 | **72.7** | 1.36x | refined-from-Alternating bucke |
| Bucket count C (32 x 256K-2M) | **8.95** | 11.00 | **12.60** | 13.40 | 104.6 | **90.6** | 1.15x | bucketed_ar_16 |
| Probe 24 x 1-8M | **15.42** | 16.70 | 23.16 | **20.26** | 304.0 | **267.3** | 1.14x | flat_cat_ar |
| Probe 48 x 1-8M | 26.56 | — | **39.21** | 53.17 | 596.4 | **492.8** | 1.21x | bucketed_ar_4 |
| Probe 48 x 64-512K | **8.28** | 10.48 | **11.51** | 13.17 | 50.0 | **32.1** | 1.56x | bucketed_ar |
| Hetero-many (48 2-D) | **20.51** | 29.42 | 29.45 | **25.89** | 74.3 | **65.2** | 1.14x | Two-phase bucketed by shape-cl |
| ZeRO-2 hetero RS | 7.01 | **6.88** | **9.41** | 9.42 | 24.1 | **24.1** | 1.00x | per_tensor_rs |

Inversion figure data (fig:disagree): bars = base/Overlay at the three scopes above.

## tab:multiseed — rows to add (2-node pair, n = gate rounds; 7-node n>=5 pending)

| Primitive | n | baseline (ms/step) | Overlay (ms/step) | Speedup |
|---|---|---|---|---|
| Layer grads (6 hetero) | 2 | 61.10 ± 0.00 | 36.76 ± 0.19 | 1.66x |
| Model grads (24 hetero) | 2 | 431.41 ± 0.00 | 96.86 ± 0.12 | 4.45x |
| Mixed-dtype grads | 2 | 60.99 ± 0.00 | 36.53 ± 0.02 | 1.67x |
| Bucket count A (64 x 64K-512K) | 2 | 98.92 ± 0.00 | 72.84 ± 0.26 | 1.36x |
| Bucket count C (32 x 256K-2M) | 2 | 104.64 ± 0.00 | 90.48 ± 0.17 | 1.16x |
| Probe 24 x 1-8M | 2 | 304.03 ± 0.00 | 267.33 ± 0.03 | 1.14x |
| Probe 48 x 1-8M | 2 | 596.37 ± 0.00 | 492.59 ± 0.33 | 1.21x |
| Probe 48 x 64-512K | 2 | 50.01 ± 0.00 | 31.98 ± 0.23 | 1.56x |
| Hetero-many (48 2-D) | 2 | 74.35 ± 0.00 | 65.18 ± 0.00 | 1.14x |
| ZeRO-2 hetero RS | 2 | 24.09 ± 0.00 | 24.10 ± 0.04 | 1.00x |

## tab:cost-calculus — new-problem search cost (strategy-enumerate, simulator v11, + training-scope gate)

| Problem | Phase-3 wall (s) | LLM calls | gate finalists + baseline (HW runs) | gate cluster time (min) |
|---|---|---|---|---|
| Layer grads (6 hetero) | 90 | 10 | 4 | 9 |
| Model grads (24 hetero) | 92 | 10 | 4 | 23 |
| Mixed-dtype grads | 74 | 10 | 4 | — |
| Bucket count A (64 x 64K-512K) | 101 | 10 | 4 | 20 |
| Bucket count C (32 x 256K-2M) | 159 | 14 | 5 | 7 |
| Probe 24 x 1-8M | 91 | 10 | 4 | 38 |
| Probe 48 x 1-8M | 130 | 14 | 5 | 58 |
| Probe 48 x 64-512K | 72 | 10 | 4 | 15 |
| Hetero-many (48 2-D) | 132 | 14 | 5 | 11 |
| ZeRO-2 hetero RS | 76 | 10 | 4 | 12 |
| **total (10 problems)** | 1017 s = 17 min | 112 | | 192 min of 2-node cluster |

LLM cost: at the paper's observed Sonnet 4.5 rate (~$19 for ~80 calls over 8 problems, i.e. ~$0.24/call) the new problems cost about $27 in LLM calls (estimate; token counts not logged by this driver).

## tab:simfidelity — rows to add (simulator v11 at training scope vs 64-rank training step; day-5 pair measurements)

| Problem | n | rho | top-1 | regret | ratio-MAPE |
|---|---|---|---|---|---|
| Layer grads (6 hetero) | 6 | 0.85 | yes | 0.0% | 31% |
| Model grads (24 hetero) | 6 | 0.65 | no | 0.2% | 63% |
| Mixed-dtype grads | 0 | — | — | — | — |
| Bucket count A (64 x 64K-512K) | 6 | 0.75 | no | 1.4% | 40% |
| Bucket count C (32 x 256K-2M) | 6 | -0.09 | no | 14.1% | 6% |
| Probe 24 x 1-8M | 6 | 0.12 | no | 3.1% | 6% |
| Probe 48 x 1-8M | 6 | -0.37 | no | 138.2% | 32% |
| Probe 48 x 64-512K | 5 | 0.87 | no | 0.7% | 7% |
| Hetero-many (48 2-D) | 6 | 0.58 | yes | 0.0% | 166% |
| ZeRO-2 hetero RS | 0 | — | — | — | — |
| pooled rho on speedup ratios | 39 | 0.40 | | | direction agreement 32/39 |

## tab:constswap — new-problem columns (per-problem Spearman of each arm against the 64-rank training step; hand-written candidates measured on the day-5 pair)

| Arm | Layer grads | Model grads | Mixed-dtype grads | Bucket count A | Bucket count C | Probe 24 x 1-8M | Probe 48 x 1-8M | Probe 48 x 64-512K | Hetero-many | ZeRO-2 hetero RS |
|---|---|---|---|---|---|---|---|---|---|---|
| calibrated (reference) | 0.85 (n=6) | 0.65 (n=6) | — | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | — |
| network x1/2 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| network x1/4 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| network x2 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| network x4 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.36 (n=5) | 0.58 (n=6) | nan (n=3) |
| op costs = isolated | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| run reset per graph | -0.65 (n=6) | -0.65 (n=6) | -0.87 (n=5) | 0.27 (n=6) | -0.85 (n=6) | -0.44 (n=6) | -0.39 (n=6) | -0.29 (n=5) | 0.79 (n=6) | nan (n=3) |
| amort. knee 64 KB | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| amort. knee 1 MB | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| fwd+bwd factor 0 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |
| fwd+bwd factor 2 | 0.85 (n=6) | 0.65 (n=6) | 0.87 (n=5) | 0.75 (n=6) | -0.09 (n=6) | 0.12 (n=6) | -0.37 (n=6) | 0.87 (n=5) | 0.58 (n=6) | nan (n=3) |

## tab:ablation-cost — new-problem rows (wall = LLM search time on the laptop, 10 problems; cluster gate time separate)

| Phase-3 style | Wall (min) | problems with a result | note |
|---|---|---|---|
| strategy-enumerate (default) | 17 | 10/10 | picks gated on hardware: results/gate_final |
| cc-react | 66 | 10/10 | picks gated on hardware: results/gate_style_ccreact (pending) |
| multi-island | 62 | 10/10 | picks gated on hardware: results/gate_style_multiisland (pending) |

LLM cost per style: the driver does not log tokens; at the paper's observed rate the three styles cost roughly in proportion to their wall time.

## tab:ablation-nonllm — new-problem row

| Problem | Best non-LLM find | Sim vs base | HW (ms) | OverlayCCL (ms) |
|---|---|---|---|---|
| all 10 new problems | none (every edit sequence scores >= baseline) | 1.00x | --- | see tab:perproblem rows |

(random walk, GA, beam, rule-based; 400 simulator evaluations per problem from the per-tensor baseline; the generic edit space never produces concatenation, bucketing, dtype casts or shape-class grouping, as in the paper.)

## tab:e2e / tab:llama-sweep — 2-node preview from train_llama_e2e_amp5.py (64 ranks, 2 layers/stage, 4 microbatches, 300 steps; 7-node numbers pending)

| backend | gradient sync | mixed dtype | sync x | step (ms) | loss (per-mb normalized) |
|---|---|---|---|---|---|
| bundled | bf16_flat | True | 4 | 20.91 | 4.719 |
| bundled | buckets_8 | False | 4 | 21.83 | 4.719 |
| bundled | buckets_8 | False | 1 | 24.90 | 4.719 |
| bundled | buckets_8 | False | 1 | 14.75 | 3.466 |
| bundled | flat | False | 4 | 21.27 | 4.719 |
| bundled | flat | False | 1 | 15.68 | 3.466 |
| bundled | flat | True | 1 | 20.57 | 4.719 |
| bundled | per_layer | False | 1 | 19.61 | 4.719 |
| bundled | per_layer | False | 4 | 73.23 | 4.719 |
| bundled | per_layer | False | 1 | 13.40 | 3.466 |
| bundled | per_tensor | False | 4 | 17.25 | 4.719 |
| bundled | per_tensor | False | 4 | 15.05 | 4.719 |
| bundled | per_tensor | False | 1 | 13.69 | 3.466 |
| bundled | per_tensor | True | 1 | 17.98 | 4.719 |
| bundled | per_tensor | True | 1 | 15.12 | 4.719 |
| per_mb | per_tensor | False | 4 | 39.91 | 4.719 |
| per_mb | per_tensor | False | 4 | 39.64 | 4.719 |
| per_mb | per_tensor | False | 1 | 39.03 | 3.466 |
| per_mb | per_tensor | True | 1 | 38.68 | 4.719 |

Reading: the four amp3 sites give 2.2x (38.6 -> 17.5 ms); on this replicated set (8 equal-size attention matrices + 4 norm gains + embedding + head shard) the per-tensor sync is already fast and every bundle adds 2-4 ms of concatenation, i.e. the heterogeneous-set inversion does not appear when most members are equal-sized. A grouped-query (GQA) variant with a heterogeneous replicated set is being measured (results_amp5_2node/gqa_*).

## tab:e2e (Llama) — 7-node rows, train_llama_e2e_amp5.py (224 ranks, GQA=4, 2 layers/stage, 4 microbatches, 300 steps, loss identical 3.466 per-mb-normalized)

| sync graph | arm | step (ms) | speedup |
|---|---|---|---|
| fused | baseline: per-microbatch, per-tensor sync | 39.49 | 1.00x |
| fused | OverlayCCL: 4 sites, per-tensor sync (gate choice) | 14.98 | 2.64x |
| fused | 4 sites + 8-count buckets | 17.89 | 2.21x |
| fused | 4 sites + per-layer bundle | 70.17 | 0.56x |
| fused | 4 sites + flat / bf16 flat | no result in 20 min | — |
| isolated | baseline | 38.55 | 1.00x |
| isolated | OverlayCCL: 4 sites, per-tensor sync (gate choice) | 12.76 | 3.02x |
| isolated | 4 sites + per-layer bundle | 13.98 | 2.76x |
| isolated | 4 sites + 8-count buckets | 15.16 | 2.54x |
| isolated | 4 sites + flat | 16.29 | 2.37x |
| isolated | 4 sites + bf16 flat (mixed dtype) | 16.27 | 2.37x |

Reading: at 224 ranks the gradient-sync site keeps per-tensor as the best strategy in BOTH graph forms; every bundle of the replicated set costs 1.2-4.7x the per-tensor step in the fused graph and 1.1-1.3x in the isolated graph. The loop's gate therefore selects per-tensor and the Llama headline is the four-site speedup (2.64x fused, 3.01x isolated).


## tab:perproblem — 7-node columns (224-rank bench, 224-rank training step; OverlayCCL = gate winner; n = training rounds)

| Problem | 7n bench base | 7n bench Overlay | 7n train base | 7n train Overlay | train speedup | n |
|---|---|---|---|---|---|---|
| Layer grads (6 hetero) | 12.88 | 14.48 | 59.0 ± 0.1 | 37.5 ± 0.3 | 1.57x | 2 |
| Model grads (24 hetero) | 40.69 | 43.53 | 155.5 ± 0.2 | 111.8 ± 0.0 | 1.39x | 2 |
| Mixed-dtype grads | 13.01 | 14.49 | 59.0 ± 0.3 | 37.8 ± 0.1 | 1.56x | 5 |
| Bucket count A (64 x 64K-512K) | 14.00 | 12.57 | 101.9 ± 0.1 | 70.9 ± 0.1 | 1.44x | 5 |
| Bucket count C (32 x 256K-2M) | 12.63 | 12.10 | 104.2 ± 0.0 | 97.7 ± 0.1 | 1.07x | 2 |
| Probe 24 x 1-8M | 23.70 | 26.28 | 307.6 ± 0.1 | 286.6 ± 0.1 | 1.07x | 2 |
| Probe 48 x 1-8M | 42.94 | 55.47 | 594.8 | — | — | 0 |
| Probe 48 x 64-512K | 12.13 | 11.37 | 50.4 ± 0.3 | 30.8 ± 0.2 | 1.64x | 5 |
| Hetero-many (48 2-D) | 31.53 | 26.87 | 76.9 ± 0.4 | 66.6 ± 0.0 | 1.15x | 2 |
| ZeRO-2 hetero RS | — | — | — | — | — | 0 |

zero2_hetero_rs: no 224-rank training result (shard sizes not divisible by 224); probe 48-mid buckets: see sy7_train_48mid if the longer-timeout rerun finished.

## tab:3node — Llama rows (amp5, fused sync graph, HID = 24 x ranks, VOCAB = 32 x ranks)

| Harness | base (ms) | OverlayCCL (ms) | Speedup |
|---|---|---|---|
| Llama amp5 7n (224r) | 39.49 | 14.98 | 2.64x |
| Llama amp5 5n (160r) | 38.29 | 14.58 | 2.63x |
| Llama amp5 3n (96r) | 36.34 | 14.36 | 2.53x |

Second seed at 7n (fused): baseline 39.89, OverlayCCL 15.11 (2.64x); isolated second seed at 7n: baseline 38.04, OverlayCCL 13.07 (2.91x; seed 0: 38.55 / 12.76 = 3.02x).


## e2e footprint — gradient-sync share of real 10B steps (224 ranks)

OLMoE-10B (paper configuration, 7-node phase timers): per-tensor replicated-gradient sync = 38 ms of a 4257 ms baseline step (0.9%), 37 ms of a 3035 ms OverlayCCL step (1.2%); both arms per-tensor.

Real 10B Llama TP=32 DP gradient set (291 pieces, 312M elements bf16) in the 224-rank training harness, 60 steps (results/sy7_10b):

| strategy | step (ms) |
|---|---|
| bucket_128mb | 349.7 |
| bucket_32mb | 327.5 |
| bucketed_ar_16 | 322.1 |
| per_layer_bucket | 344.0 |
| per_tensor_ar | 331.9 |

All strategies within +-3%: bandwidth-bound, no inversion; the paper's original simulator predicts a tie within 2%, the v11 heterogeneous-set term wrongly predicts a 12x per-tensor penalty (see REPORT.md section 8).


## chosen problems (paper rule = simulator-decides with the fitted v11 terms; training gate = validation only)

| Problem | ingredient | paper-rule pick | 64r speedup | 224r speedup (n) | no-sim pick | no-sim 224r |
|---|---|---|---|---|---|---|
| Mixed-dtype grads | bf16 wire cast | baseline:cast_flat_bf16 | 1.67x | 1.56x (5) | 3 in-order pairs, dtype preserved | 1.44x |
| Probe 48 x 64-512K | pathology-aware grouping (8-piece in-order) | baseline:bucketed_ar | 1.56x | 1.64x (5) | single flat concat | 0.96x |
| Bucket count A (64 x 64K-512K) | pathology-aware grouping (2 alternating buckets) | strat:refined-from-Alternating bucke | 1.36x | 1.44x (5) | two-phase hierarchical buckets | 0.98x |

## tab:e2e (Llama) — amp6: four amp3 sites + the two per-problem gradient-sync sites (224 ranks unless noted)

| arm | sync graph | nodes | step (ms) | loss | tag |
|---|---|---|---|---|---|
| bundled / per_tensor | isolated2 | 7 | 24.48 | 3.466 | amp6_bundled_per_tensor_isolated2_L2_M4_seed0 |
| bundled / per_tensor | isolated | 5 | 5.04 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_agdim0_t_native_nosites_seed0 |
| bundled / per_tensor | isolated | 7 | 4.10 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_agdim0_t_native_nosites_seed1 |
| bundled / per_tensor | isolated | 7 | 22.82 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_agdim0_t_native_seed0 |
| bundled / per_tensor | isolated | 7 | 24.02 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_agdim0_t_seed0 |
| bundled / per_tensor | isolated | 7 | 23.06 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_direct_seed0 |
| bundled / per_tensor | isolated | 5 | 5.00 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_nosites_seed0 |
| bundled / per_tensor | isolated | 7 | 5.50 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_nosites_seed1 |
| bundled / per_tensor | isolated | 7 | 22.37 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_rep2_seed0 |
| bundled / per_tensor | isolated | 7 | 22.17 | 3.466 | amp6_bundled_per_tensor_isolated_L2_M4_seed0 |
| bundled / sites_a_cast_pt | isolated | 7 | 24.51 | 3.466 | amp6_bundled_sites_a_cast_pt_isolated_L2_M4_seed0 |
| bundled / sites_a_only | isolated | 7 | 42.50 | 3.466 | amp6_bundled_sites_a_only_isolated_L2_M4_seed0 |
| bundled / sites_b_only | isolated | 7 | 22.03 | 3.466 | amp6_bundled_sites_b_only_isolated_L2_M4_seed0 |
| bundled / sites | isolated2 | 7 | 61.74 | 3.466 | amp6_bundled_sites_isolated2_L2_M4_seed0 |
| bundled / sites | isolated | 7 | 96.55 | 3.466 | amp6_bundled_sites_isolated_L2_M4_direct_seed0 |
| bundled / sites | isolated | 7 | 54.59 | 3.466 | amp6_bundled_sites_isolated_L2_M4_rep2_seed0 |
| bundled / sites | isolated | 7 | 43.07 | 3.466 | amp6_bundled_sites_isolated_L2_M4_seed0 |
| per_mb / base sync per_mb | isolated | 7 | 19.36 | 13.863 | amp6_per_mb_base_per_mb_isolated_L2_M4_agdim0_t_native_nosites_seed0 |
| per_mb / base sync per_mb | isolated | 7 | 74.21 | 13.863 | amp6_per_mb_base_per_mb_isolated_L2_M4_agdim0_t_native_seed0 |
| per_mb / base sync per_mb | isolated | 5 | 18.36 | 13.863 | amp6_per_mb_base_per_mb_isolated_L2_M4_nosites_seed0 |
| per_mb / base sync per_mb | isolated | 7 | 21.26 | 13.863 | amp6_per_mb_base_per_mb_isolated_L2_M4_nosites_seed1 |
| per_mb / base sync per_mb | isolated | 7 | 95.99 | 13.863 | amp6_per_mb_base_per_mb_isolated_L2_M4_seed0 |

## tab:perproblem — hunt row: fsdp_ag_layout (FSDP prefetch gathers; strategy type = collective lowering/layout)

| candidate | 1n bench (32r) | 7n bench (224r lone call) | 7n training step | sim v12 (us) |
|---|---|---|---|---|
| pinned default (developer baseline) | ok | 16.1 | 37.4 | 52358 |
| leading-dim gather + transposes, pinned | does not compile | does not compile | 40.4 | 52856 |
| leading-dim gather + transposes, native lowering | does not compile | does not compile | 24.8 | 30552 |

Speedup (training step): native 1.50x over pinned; the lone-call scopes cannot rank the winner (compile failure outside the training graph).


## tab:e2e / tab:3node — clean amp3-equivalent shell (amp6 SITES=0), isolated sync graph, 300 steps

| nodes | baseline (per-mb) | OverlayCCL (bundled, pinned AG) | OverlayCCL + native AG | speedup pinned / native |
|---|---|---|---|---|
| 7 | 20.41 | 5.93 | 4.95 | 3.44x / 4.12x |
| 7 (seed 1) | 21.26 | 5.50 | 4.10 | 3.87x / 5.19x |
| 5 | 18.36 | 5.00 | 5.04 | 3.67x / 3.64x |
| 3 | 14.76 | 5.34 | 4.52 | 2.76x / 3.27x |

7-node baseline with the native gather: 19.36 ms (the developer baseline keeps the pinned default by convention).


## tab:perproblem (re-labeled scopes) — developer baseline = microbench winner, OverlayCCL = per-tensor (native gather for FSDP AG)

| Problem | 1n bench base/tool | 7n bench base/tool | microbench (fwd-collective harness, 224r) base/tool | 1n TRAINING base/tool | 7n TRAINING base/tool | 7n train speedup | n |
|---|---|---|---|---|---|---|---|
| Layer grads (6 hetero) | see tab:perproblem rows above | —/— | 37.9/59.0 | pending | 35.9±0.6/34.5±1.1 | 1.04x | 3 |
| Model grads (24 hetero) | see tab:perproblem rows above | —/— | 111.8/155.5 | pending | 118.5±0.2/118.0±0.3 | 1.00x | 3 |
| Mixed-dtype grads | see tab:perproblem rows above | —/— | 37.8/59.0 | pending | 35.9±0.3/34.6±0.7 | 1.04x | 3 |
| Bucket count A (64 x 64K-512K) | see tab:perproblem rows above | —/— | 70.1/101.9 | pending | 125.4±1.2/81.4±1.4 | 1.54x | 3 |
| Bucket count C (32 x 256K-2M) | see tab:perproblem rows above | —/— | 87.3/104.2 | pending | 48.8±0.4/46.8±0.7 | 1.04x | 3 |
| Probe 24 x 1-8M | see tab:perproblem rows above | —/— | 265.1/307.6 | pending | 57.7±0.5/51.2±0.7 | 1.13x | 3 |
| Probe 48 x 1-8M | see tab:perproblem rows above | —/— | — | pending | 115.1±0.9/93.8±0.4 | 1.23x | 3 |
| Probe 48 x 64-512K | see tab:perproblem rows above | —/— | 31.6/50.4 | pending | 62.2±0.8/61.2±0.5 | 1.02x | 3 |
| Hetero-many (48 2-D) | see tab:perproblem rows above | —/— | 66.2/76.9 | pending | 122.4±0.3/130.4±0.6 | 0.94x | 3 |
| fsdp_ag_layout | see tab:perproblem rows above | —/— | — | pending | 102.9±0.8/90.3±1.2 | 1.14x | 3 |

(1n/7n bench = lone call; 'microbench' = the former per-problem harness (collective in the forward on flat-buffer views); TRAINING = mirror harness: the set as parameters of a small model, N_MB=4 accumulate, sync after the backward.)


## pending

* tab:e2e (OLMoE / Llama end-to-end) — needs the 7-node runs; the 2-node amp5 matrix (results/amp5) gives a preview of the gradient-sync site.
* tab:llama-sweep, tab:3node, tab:ablation-nosim, appendix nosim per-problem — 7-node.
* appendix per-problem style cells — cc-react and multi-island picks are being gated on hardware (results/gate_style_*).
* tab:probe-tools / tab:cost-calculus text — add the training-scope gate as a tool/row (text change, later).

