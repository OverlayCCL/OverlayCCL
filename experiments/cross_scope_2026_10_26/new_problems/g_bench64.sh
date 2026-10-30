#!/bin/bash
# forward-only bench at 64 ranks (both nodes) for a list of new problems: same world size as the training runs
set -uo pipefail
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; AP=$REPO/experiments/cluster_224rank/scripts; OUT=${B64_OUT:-$G/bench64}; mkdir -p $OUT
source /home/ubuntu/venv.env
cp $G/g_newprobs.py $G/g_bench2.py $REPO/training/
export RESULTS_DIR=$OUT
for ip in $WORKERS; do ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "mkdir -p $OUT" </dev/null; done
for P in ${PROBS:?}; do
  [ -f $OUT/newprob_bench_${P}_ws64.json ] && continue
  echo "--- [$(date -u +%T)] bench64 $P" | tee -a $OUT/bench64.log
  timeout ${RUN_TIMEOUT:-1500} bash $AP/run_ap7.sh training/g_bench2.py agent 1 bench64_$P --problem $P >> $OUT/bench64_run.log 2>&1
  for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip:"$OUT/newprob_bench_${P}_ws64.json" $OUT/ 2>/dev/null; done
  grep -a -h "\[bench2\]" $OUT/logs/bench64_${P}_node*.log 2>/dev/null | cut -c1-170 | tee -a $OUT/bench64.log
done
echo "=== bench64 done $(date -u +%T) ===" | tee -a $OUT/bench64.log
