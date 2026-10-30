#!/bin/bash
# Day-2 controls for the heterogeneous-concat pathology:
#  (1) training runner at 32 ranks on ONE node (same code as the 64-rank runs)  -> is it scale or the fwd/bwd graph?
#  (2) plain bench at 64 ranks on two nodes                                     -> does the pathology show in forward-only?
set -uo pipefail
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; AP=$REPO/experiments/cluster_224rank/scripts; OUT=$G/controls; mkdir -p $OUT
source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO
cp $G/g_newprobs.py $G/g_train.py $G/g_bench2.py $REPO/training/
for C in per_tensor_ar flat_cat_ar; do
  echo "--- [$(date -u +%T)] control-1 train@32 hetero_grad_ar_many $C" | tee -a $OUT/controls.log
  RESULTS_DIR=$OUT/train32 timeout 900 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29704 $REPO/training/g_train.py --problem hetero_grad_ar_many --backend $C --steps 300 --warmup 5 > $OUT/train32_$C.log 2>&1
  grep -h "\[train\]" $OUT/train32_$C.log | tee -a $OUT/controls.log
done
echo "--- [$(date -u +%T)] control-2 bench@64 hetero_grad_ar_many" | tee -a $OUT/controls.log
export RESULTS_DIR=$OUT/bench64; mkdir -p $RESULTS_DIR
for ip in $WORKERS; do ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "mkdir -p $RESULTS_DIR" </dev/null; done
timeout 1500 bash $AP/run_ap7.sh training/g_bench2.py agent 1 bench64_hetero --problem hetero_grad_ar_many >> $OUT/controls.log 2>&1
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip:"$RESULTS_DIR/*.json" $RESULTS_DIR/ 2>/dev/null; done
grep -a -h "\[bench2\]" $RESULTS_DIR/logs/bench64_hetero_node*.log | cut -c1-170 | tee -a $OUT/controls.log
echo "=== controls done $(date -u +%T) ===" | tee -a $OUT/controls.log
