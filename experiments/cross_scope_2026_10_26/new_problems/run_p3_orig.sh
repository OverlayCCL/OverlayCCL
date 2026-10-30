#!/bin/bash
# OverlayCCL Phase 3 on the new problems with the PAPER'S ORIGINAL simulator constants (heterogeneous-set and unequal-concat terms off)
cd ~/work/paper/overlayccl_public; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1 HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0
for p in layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype; do TAGP3=orig python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_orig_$p.log 2>&1; done
for p in dp_grad_bucketing_64 hs_c_32mixed hetero_grad_ar_many probe_het_24_mid probe_het_48_mid probe_het_48_small zero2_hetero_rs; do SCALE=1024 TAGP3=orig python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_orig_$p.log 2>&1; done
echo P3_ORIG_DONE
