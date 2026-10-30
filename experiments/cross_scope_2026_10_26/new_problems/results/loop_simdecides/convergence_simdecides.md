# Loop output under the paper's selection rule (simulator v11 + correctness/TV gates; training-scope gate = validation only)

Rule: Phase 5 ships the candidate with the lowest simulated time among the candidates that pass the correctness gates (compiles, runs at training shape, output effective). The 'training gate (validation)' column is the fastest measured candidate of 2026-10-28, kept for reference; it is NOT the selector. 64r = 2-node training step (day-5 pair), 224r = 7-node.

| problem | paper-rule pick (lowest sim among gate survivors) | 64r: pick / per-tensor / best measured (ms) | 64r speedup | 224r pick vs per-tensor | training gate (validation) |
|---|---|---|---|---|---|
| hetero_grad_ar_many | gate_1: baseline:bucketed_ar_16 (sim 577203 us) | 215.9 / 74.3 / 65.2 (gate_2) | 0.34x | — | gate_2 (65.2) |
| hs_a_64x64k_512k | gate_1: strat:refined-from-Alternating bucke (sim 73227 us) | 72.7 / 98.9 / 71.7 (bucket_4mb) | 1.36x | 1.44x (n=2) | gate_1 (72.7) |
| hs_c_32x256k_2m | gate_1: strat:refined-from-Reduce-Scatter th (sim 123478 us) | 101.0 / 104.6 / 88.5 (bucket_4mb) | 1.04x | — | gate_3 (90.6) |
| layer_grads_hetero_ar | gate_1: baseline:flat_cat_ar (sim 16958 us) | 36.7 / 61.1 / 36.6 (flat_cat_ar) | 1.67x | — | gate_3 (36.6) |
| layer_grads_mixed_dtype | gate_1: baseline:cast_flat_bf16 (sim 9281 us) | 36.5 / 61.0 / 36.5 (gate_1) | 1.67x | 1.57x (n=2) | gate_1 (36.5) |
| model_grads_4layers | gate_1: strat:Two-bucket layer-pair bundling (sim 64218 us) | 97.2 / 431.4 / 96.8 (flat_cat_ar) | 4.44x | — | gate_3 (96.9) |
| probe_het_24_mid | gate_1: strat:refined-from-Reduce-scatter th (sim 466668 us) | 275.0 / 304.0 / 266.6 (bucket6) | 1.11x | — | gate_2 (267.3) |
| probe_het_48_mid | gate_1: strat:refined-from-Two-phase size-st (sim 936503 us) | 1173.7 / 596.4 / 492.8 (gate_4) | 0.51x | — | gate_4 (492.8) |
| probe_het_48_small | gate_2: baseline:bucketed_ar (sim 63144 us) | 32.1 / 50.0 / 31.9 (bucket6) | 1.56x | 1.64x (n=2) | gate_2 (32.1) |
| zero2_hetero_rs | gate_1: baseline:per_tensor_rs (sim 124100 us) | 24.1 / 24.1 / 24.1 (gate_1) | 1.00x | — | gate_1 (24.1) |
