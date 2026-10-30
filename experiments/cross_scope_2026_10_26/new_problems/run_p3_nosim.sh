#!/bin/bash
cd ~/work/paper/overlayccl_public; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1
env TAGP3=nosim python /tmp/g2/run_phase3_nosim.py layer_grads_mixed_dtype 4 > /tmp/g2/hw/phase3_nosim_layer_grads_mixed_dtype.log 2>&1
for p in dp_grad_bucketing_64 probe_het_48_small; do env SCALE=1024 TAGP3=nosim python /tmp/g2/run_phase3_nosim.py $p 4 > /tmp/g2/hw/phase3_nosim_$p.log 2>&1; done
echo P3_NOSIM_DONE
