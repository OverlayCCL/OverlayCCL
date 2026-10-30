#!/bin/bash
# constant-swap arms (paper Table constswap) rescored on the new problems' candidates at training scope
cd /tmp/g2; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1 SCOPES=train
export SETS="model_grads_4layers,layer_grads_hetero_ar,layer_grads_mixed_dtype,hs_a_64x64k_512k,hs_c_32x256k_2m,hetero_grad_ar_many,model_grads_bucket_sweep,zero2_hetero_rs:rs,probe_het_24_mid,probe_het_48_mid,probe_het_48_small"
while pgrep -f "TAGV5=v11all" > /dev/null || pgrep -f "sim_newprobs_v5.py" > /dev/null; do sleep 30; done
run() { tag=$1; shift; env "$@" TAGV5=arm_$tag python sim_newprobs_v5.py > hw/sim_v5_arm_$tag.log 2>&1; echo "done $tag"; }
run net_x0p5 NET_SCALE=0.5 & run net_x0p25 NET_SCALE=0.25 & run net_x2 NET_SCALE=2 & run net_x4 NET_SCALE=4 & wait
run opc_isolated OPC=isolated & run reset_graph RESET=graph & run knee_64k KNEE=65536 & run knee_1m KNEE=1048576 & wait
run fwdbwd_0 FWDBWD=0 & run fwdbwd_2 FWDBWD=2 & wait
echo ARMS_NEWPROBS_DONE
