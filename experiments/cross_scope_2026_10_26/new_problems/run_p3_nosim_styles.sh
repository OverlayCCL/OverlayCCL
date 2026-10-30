#!/bin/bash
# local LLM runs for the ablation tables: no-simulator arm for the 7 remaining problems, then cc-react / multi-island styles (original constants) on all 10
cd ~/work/paper/overlayccl_public; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1 HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0
for p in layer_grads_hetero_ar model_grads_4layers; do TAGP3=nosim python /tmp/g2/run_phase3_nosim.py $p 4 > /tmp/g2/hw/phase3_nosim_$p.log 2>&1; done
for p in probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32mixed; do SCALE=1024 TAGP3=nosim python /tmp/g2/run_phase3_nosim.py $p 4 > /tmp/g2/hw/phase3_nosim_$p.log 2>&1; done
echo P3_NOSIM7_DONE
for st in cc-react multi-island; do tag=orig_$(echo $st | tr -d '-')
  for p in layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype; do env P3_STYLE=$st TAGP3=$tag python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_${tag}_$p.log 2>&1; done
  for p in dp_grad_bucketing_64 hs_c_32mixed hetero_grad_ar_many probe_het_24_mid probe_het_48_mid probe_het_48_small zero2_hetero_rs; do env SCALE=1024 P3_STYLE=$st TAGP3=$tag python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_${tag}_$p.log 2>&1; done
done
echo P3_STYLES_ORIG_DONE
