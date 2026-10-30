# amp7 end-to-end JSON (224 / 160 / 96 ranks)

One file per arm, named by the run tag. The 224-rank seed-0 pair (`..._ws224_seed0.json`, baseline and OverlayCCL) was run twice:
a warm-cache run (chain log `a7_base_microbench` / `a7_bund_pertensor_native`: 600.1 / 79.1 ms steady median) and a cold-cache
repeat for the wall-clock measurement (`cold_base_microbench` / `cold_bund_pertensor_native`: 595.4 / 72.3 ms). The tag does not
encode the cache state, so the files hold the cold repeat; the warm values survive only in `../chain_amp7.out`.
