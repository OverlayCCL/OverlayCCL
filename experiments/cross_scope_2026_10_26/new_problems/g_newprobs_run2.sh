#!/bin/bash
# new problems: 1-node bench+correctness per problem (master), then 2-node training-scope runs per candidate
set -uo pipefail
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; AP=$REPO/experiments/cluster_224rank/scripts; OUT=${NP_OUT:-$G/newprobs}; mkdir -p $OUT
source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO OUT=$OUT
cp $G/g_newprobs.py $G/g_train.py $G/g_bench2.py $REPO/training/
PROBS=${PROBS:-dp_grad_ar_groups zero2_grad_rs zero1_param_ag ulysses_seq_a2a}
for P in $PROBS; do
  [ -f $OUT/newprob_bench_$P.json ] && continue
  echo "--- [$(date -u +%T)] bench $P" | tee -a $OUT/run.log
  for i in $(seq 1 30); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
  $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29703 $REPO/training/g_bench2.py --problem $P > $OUT/bench_$P.log 2>&1
  grep -h "\[bench2\]" $OUT/bench_$P.log | cut -c1-200 | tee -a $OUT/run.log
done
export RESULTS_DIR=$OUT/train; mkdir -p $RESULTS_DIR
for P in $PROBS; do
  for C in $(python3 -c "import sys; sys.path.insert(0,'$G'); from g_newprobs import PROBLEMS; print(' '.join(PROBLEMS['$P']['cands']))"); do
    [ -f $RESULTS_DIR/newprob_train_${P}__${C}.json ] && continue
    echo "--- [$(date -u +%T)] train $P $C" | tee -a $OUT/run.log
    timeout ${RUN_TIMEOUT:-1500} bash $AP/run_ap7.sh training/g_train.py $C ${STEPS:-300} newprob_${P}__${C} --problem $P ${TRAIN_EXTRA:-} >> $OUT/run_train.log 2>&1
    for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip:$RESULTS_DIR/newprob_train_${P}__${C}.json $RESULTS_DIR/ 2>/dev/null; done
    if grep -q "ncclInitComm\|failed to init NCCL comm\|NRT_INVALID\|NRT_FAILURE" $RESULTS_DIR/logs/newprob_${P}__${C}_node*.log 2>/dev/null; then
      echo "   INVALID (collective init failure in node logs)" | tee -a $OUT/run.log; mv $RESULTS_DIR/newprob_train_${P}__${C}.json $RESULTS_DIR/INVALID_newprob_train_${P}__${C}.json 2>/dev/null
    else
      python3 -c "import json; d=json.load(open('$RESULTS_DIR/newprob_train_${P}__${C}.json')); print('   median_ms', round(d['measured_ms'],2), 'effective', d.get('collective_effective'))" 2>/dev/null | tee -a $OUT/run.log || echo "   NO RESULT" | tee -a $OUT/run.log
    fi
  done
done
echo "=== newprobs done $(date -u +%T) ===" | tee -a $OUT/run.log
