#!/bin/bash
# the other two Phase-3 styles on the ten new problems (LLM only; picks gated on hardware later)
cd ~/work/paper/overlayccl_public; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1
for st in cc-react multi-island; do
  tag=style_$(echo $st | tr -d '-')
  for p in layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype; do /usr/bin/time -f "%e s wall" env P3_STYLE=$st TAGP3=$tag python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_${tag}_$p.log 2>&1; done
  for p in dp_grad_bucketing_64 hs_c_32mixed hetero_grad_ar_many probe_het_24_mid probe_het_48_mid probe_het_48_small zero2_hetero_rs; do /usr/bin/time -f "%e s wall" env SCALE=1024 P3_STYLE=$st TAGP3=$tag python /tmp/g2/run_phase3_v5.py $p 4 > /tmp/g2/hw/phase3_${tag}_$p.log 2>&1; done
done
echo P3_STYLES_DONE
