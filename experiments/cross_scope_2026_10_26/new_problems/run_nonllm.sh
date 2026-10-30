#!/bin/bash
cd /tmp/g2; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1
for p in layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype; do OUTDIR=/tmp/g2/hw/nonllm/$p BUDGET=400 PROBLEMS=$p nice -n 10 python generic_search_newprobs.py > hw/nonllm_$p.log 2>&1; done
for p in dp_grad_bucketing_64 hs_c_32mixed hetero_grad_ar_many probe_het_24_mid probe_het_48_mid probe_het_48_small zero2_hetero_rs; do OUTDIR=/tmp/g2/hw/nonllm/$p BUDGET=400 SCALE=1024 PROBLEMS=$p nice -n 10 python generic_search_newprobs.py > hw/nonllm_$p.log 2>&1; done
echo NONLLM_DONE
