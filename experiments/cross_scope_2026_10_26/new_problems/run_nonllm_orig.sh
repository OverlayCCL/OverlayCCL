#!/bin/bash
# Non-LLM searches on the nine gradient-sync / prefetch problems with the paper's ORIGINAL simulator constants, two seeds,
# 400 evaluations per search, starting from (a) the paper's developer baseline (the microbench-winning bundle) and (b) per-tensor.
cd /tmp/g2; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1 HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0
BUNDLE='layer_grads_hetero_ar:flat_cat_ar,model_grads_4layers:flat_cat_ar,layer_grads_mixed_dtype:cast_flat_bf16,dp_grad_bucketing_64:bucketed_ar,hs_c_32mixed:bucketed_ar,probe_het_24_mid:bucketed_ar,probe_het_48_mid:bucketed_ar,probe_het_48_small:bucketed_ar,fsdp_ag_layout:ag_pinned'
PT='fsdp_ag_layout:ag_dim0_t_native'
for start in bundle pertensor; do
  [ $start = bundle ] && ST=$BUNDLE || ST=$PT
  for seed in 0 1; do
    for p in layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype fsdp_ag_layout; do
      STARTS=$ST SEED=$seed OUTDIR=/tmp/g2/hw/nonllm_orig/$start/$p BUDGET=400 PROBLEMS=$p python generic_search_newprobs.py > hw/nonllm_orig_${start}_${p}_s$seed.log 2>&1; grep -h "start=" hw/nonllm_orig_${start}_${p}_s$seed.log | cut -c1-260; done
    for p in dp_grad_bucketing_64 hs_c_32mixed probe_het_24_mid probe_het_48_mid probe_het_48_small; do
      STARTS=$ST SEED=$seed SCALE=1024 OUTDIR=/tmp/g2/hw/nonllm_orig/$start/$p BUDGET=400 PROBLEMS=$p python generic_search_newprobs.py > hw/nonllm_orig_${start}_${p}_s$seed.log 2>&1; grep -h "start=" hw/nonllm_orig_${start}_${p}_s$seed.log | cut -c1-260; done
  done
done
echo NONLLM_ORIG_DONE
