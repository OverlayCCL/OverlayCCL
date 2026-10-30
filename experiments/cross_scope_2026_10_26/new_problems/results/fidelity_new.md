# Simulator fidelity (original constants) vs mirror-harness 224-rank training, new problems

| Problem | n | rho | top-1 | regret | ratio-MAPE |
|---|---|---|---|---|---|
| hetero_grad_ar_many | 4 | 0.74 | no (p3_pick3) | 0.4% | 6% |
| hs_a_64x64k_512k | 4 | 0.40 | no (orig_1) | 11.0% | 20% |
| hs_c_32x256k_2m | 4 | 0.77 | yes | 0.0% | 2% |
| layer_grads_hetero_ar | 4 | 0.77 | yes | 0.0% | 5% |
| layer_grads_mixed_dtype | 3 | 0.87 | yes | 0.0% | 5% |
| model_grads_4layers | 4 | -0.26 | no (per_tensor_ar) | 0.3% | 11% |
| probe_het_24_mid | 4 | 0.77 | yes | 0.0% | 23% |
| probe_het_48_mid | 4 | 0.77 | yes | 0.0% | 49% |
| probe_het_48_small | 4 | 1.00 | yes | 0.0% | 6% |

# Constant-swap arms (* = the swap changes the simulator's pick)

| Arm | hetero_grad_ar_many | hs_a_64x64k_512k | hs_c_32x256k_2m | layer_grads_hetero_ar | layer_grads_mixed_dtype | model_grads_4layers | probe_het_24_mid | probe_het_48_mid | probe_het_48_small | agree. |
|---|---|---|---|---|---|---|---|---|---|---|
| calibrated (original constants) | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| network x1/2 | 0.26* | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 0.95 | 8/9 |
| network x1/4 | 0.26* | 0.40* | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 0.74* | 6/9 |
| network x2 | 0.74 | 0.40 | 0.32 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| network x4 | 0.74 | 0.40 | -0.74* | 0.77 | 0.87 | 0.32 | 0.77 | 0.32 | 0.80* | 7/9 |
| local terms = snapshot | — | nan* | nan | nan | nan | nan | nan | nan | nan* | 6/8 |
| op costs = isolated | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| run reset per graph | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 0.95 | 9/9 |
| amort. knee 64 KB | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| amort. knee 1 MB | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| fwd+bwd factor 0 | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |
| fwd+bwd factor 2 | 0.74 | 0.40 | 0.77 | 0.77 | 0.87 | -0.26 | 0.77 | 0.77 | 1.00 | 9/9 |

