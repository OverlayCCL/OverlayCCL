#!/bin/bash
# HLO dumps (training runner at 32 ranks on the master only, rank 0 dump) for pairs that differ on hardware
# although they look equivalent: mixed-dtype pick vs hand-written bundle; layer no-norm flat with/without a tiny piece.
set -uo pipefail
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; OUT=$G/hlo; mkdir -p $OUT
source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO
cp $G/g_newprobs.py $G/g_train.py $REPO/training/
run() { P=$1; C=$2; D=$OUT/${P}__${C}; mkdir -p $D
  echo "--- [$(date -u +%T)] hlo $P $C" | tee -a $OUT/hlo.log
  XLA_FLAGS="--xla_dump_to=$D --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.* " XLA_SAVE_TENSORS_FILE= RESULTS_DIR=$OUT/train32 timeout 900 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29705 $REPO/training/g_train.py --problem $P --backend $C --steps 30 --warmup 3 > $D/run.log 2>&1
  grep -h "\[train\]\|\[check\]" $D/run.log | tee -a $OUT/hlo.log; ls $D | wc -l | tee -a $OUT/hlo.log; }
run md_pick_vs_hand two_dtype_groups
run md_pick_vs_hand p3_pick
run path_no_norm_clone flat_cat_ar
run path_no_norm_plus_tiny flat_cat_ar
run path_many_clone flat_cat_ar
run path_many_plus_tiny flat_cat_ar
echo HLO_DONE >> $OUT/hlo.log
