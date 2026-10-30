| problem | candidate | bench 32r ms | bench ratio | train 64r ms (n) | train ratio | sim bench | sim train64 | sim train224 | note |
|---|---|---|---|---|---|---|---|---|---|
| alltoallv | ag_slice_pl0 | 5.8 | 1.00x | 13.3 (1) | 1.08x | — | — | — | bench-tie, training win |
| alltoallv | paper | 5.8 | 1.00x | 13.4±0.1 (2) | 1.07x | — | — | — | bench-tie, training win |
| alltoallv | rs_masked | 5.9 | 0.99x | 12.8±0.1 (2) | 1.13x | — | — | — | bench-tie, training win |
| alltoallv | harness_baseline | 5.8 | 1.00x | 14.4±0.1 (2) | 1.00x | 1.00x | 1.00x | 1.00x | |
| dxe | ag_chunked_4 | 7.7 | 0.82x | 20.1 (1) | 0.90x | 0.52x | 0.56x | 0.66x |  |
| dxe | hier_ar_stacked | 6.0 | 1.05x | 17.7 (1) | 1.02x | 0.70x | 0.68x | 1.00x | bench win, training tie |
| dxe | paper | 5.9 | 1.07x | 15.0 (1) | 1.21x | 0.70x | 0.68x | 1.00x |  |
| dxe | harness_baseline | 6.3 | 1.00x | 18.1 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| fsdp_prefetch | ag_all | 8.3 | 1.16x | 16.2 (1) | 1.16x | 1.78x | 1.65x | 1.51x |  |
| fsdp_prefetch | ag_dim0_permute_per_mb | 11.7 | 0.82x | 21.8 (1) | 0.86x | 0.92x | 0.93x | 0.94x |  |
| fsdp_prefetch | chunked_ag_2_per_mb | 11.6 | 0.83x | 20.7 (1) | 0.91x | 0.60x | 0.62x | 0.66x |  |
| fsdp_prefetch | hier_ag_per_mb | 10.2 | 0.94x | 49.0 (1) | 0.38x | 1.00x | 0.12x | 0.14x |  |
| fsdp_prefetch | paper | 10.3 | 0.94x | 19.2 (1) | 0.98x | 1.00x | 1.00x | 1.00x |  |
| fsdp_prefetch | harness_baseline | 9.6 | 1.00x | 18.9 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| llama_block_ar | hier_ar_stack | 5.7 | 0.99x | 49.5 (1) | 0.47x | 1.33x | 0.73x | 0.73x |  |
| llama_block_ar | paper | 5.7 | 0.99x | 21.2 (1) | 1.10x | 1.33x | 1.33x | 1.33x | bench-tie, training win |
| llama_block_ar | rs_ag_stack | 5.9 | 0.95x | 24.5 (1) | 0.96x | 1.15x | 1.16x | 1.14x |  |
| llama_block_ar | harness_baseline | 5.6 | 1.00x | 23.4 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| pp_send_recv | ag_gather_all_mb | 11.8 | 0.95x | fail (0) | — | 3.27x | 3.40x | 3.51x |  |
| pp_send_recv | ag_gather_all_mb_pl0 | 8.9 | 1.27x | 34.3 (1) | 1.30x | — | — | — |  |
| pp_send_recv | ag_gather_per_mb | 13.3 | 0.85x | 59.5 (1) | 0.75x | 2.26x | 2.72x | 3.26x |  |
| pp_send_recv | ag_gather_per_mb_pl0 | 8.8 | 1.28x | 36.0 (1) | 1.24x | — | — | — |  |
| pp_send_recv | ar_masked_per_mb | 11.2 | 1.00x | 45.3 (1) | 0.99x | 1.00x | 1.00x | 1.00x |  |
| pp_send_recv | paper | 9.0 | 1.24x | 36.2 (1) | 1.23x | 1.11x | 1.06x | 1.02x |  |
| pp_send_recv | rs_masked_all_mb | 11.6 | 0.97x | 45.5 (1) | 0.98x | 2.92x | 2.97x | 3.01x |  |
| pp_send_recv | rs_masked_per_mb | 10.3 | 1.09x | 42.2 (1) | 1.06x | 2.17x | 2.50x | 2.85x |  |
| pp_send_recv | harness_baseline | 11.2 | 1.00x | 44.7 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| ring_kv | ar_masked | 5.8 | 1.35x | 7.2 (1) | 1.18x | 1.40x | 1.60x | 2.32x |  |
| ring_kv | chunked_ag_4 | 6.1 | 1.28x | 8.0 (1) | 1.07x | 1.21x | 1.19x | 1.19x |  |
| ring_kv | hier_ag | 5.9 | 1.33x | 10.2 (1) | 0.83x | 1.30x | 1.12x | 0.95x | INVERSION |
| ring_kv | paper | 5.5 | 1.42x | 7.0 (1) | 1.21x | 1.30x | 1.44x | 1.24x |  |
| ring_kv | per_slot_pl0 | 5.7 | 1.37x | 7.0 (1) | 1.21x | — | — | — |  |
| ring_kv | harness_baseline | 7.8 | 1.00x | 8.5 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| tp_mlp | ar_per_mb | 6.5 | 0.96x | 18.0 (1) | 0.99x | 1.00x | 1.00x | 1.00x |  |
| tp_mlp | chunked_ar_bundled_4 | 6.9 | 0.91x | 18.4 (1) | 0.97x | 0.96x | 0.96x | 0.96x |  |
| tp_mlp | hier_ar_bundled | 6.3 | 0.98x | 29.2 (1) | 0.61x | 1.64x | 0.88x | 0.87x |  |
| tp_mlp | paper | 6.3 | 0.99x | 17.3 (1) | 1.03x | 1.64x | 1.63x | 1.62x |  |
| tp_mlp | rs_ag_bundled | 6.6 | 0.95x | 18.2 (1) | 0.98x | 1.88x | 1.88x | 1.83x |  |
| tp_mlp | harness_baseline | 6.2 | 1.00x | 17.9 (1) | 1.00x | 1.00x | 1.00x | 1.00x | |
| uniform_a2a | ag_slice_pl0 | 6.4 | 1.23x | 346.8 (1) | 1.32x | — | — | — |  |
| uniform_a2a | ag_t_rs_pl0 | 6.7 | 1.18x | 397.5 (1) | 1.15x | — | — | — |  |
| uniform_a2a | chunked_ag_2 | 7.9 | 0.99x | 421.9±0.8 (3) | 1.09x | 2.19x | 2.29x | 2.89x | bench-tie, training win |
| uniform_a2a | chunked_ag_4 | 7.9 | 0.99x | 483.9 (1) | 0.95x | 2.07x | 2.23x | 2.87x |  |
| uniform_a2a | hier_ag | 7.3 | 1.07x | 348.9 (1) | 1.32x | 2.27x | 1.17x | 1.45x |  |
| uniform_a2a | paper | 7.7 | 1.02x | 406.8±0.6 (3) | 1.13x | 2.27x | 2.33x | 2.90x | bench-tie, training win |
| uniform_a2a | rs_masked | 6.4 | 1.23x | 322.3±0.7 (3) | 1.42x | 1.95x | 1.92x | 1.63x |  |
| uniform_a2a | harness_baseline | 7.9 | 1.00x | 459.0±0.5 (3) | 1.00x | 1.00x | 1.00x | 1.00x | |
