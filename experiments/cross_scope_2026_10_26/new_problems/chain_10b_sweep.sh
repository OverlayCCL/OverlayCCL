#!/bin/bash
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q SY7_10B_DONE $G/chain_10b.out 2>/dev/null; do sleep 15; done
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
cp $G/g_newprobs_10b.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
mkdir -p $G/sy7_10b_sweep; for p in llama10b_dp_grads_L8 llama10b_dp_grads_L24; do echo '{"skipped": true}' > $G/sy7_10b_sweep/newprob_bench_$p.json; done
echo "=== [$(date -u +%T)] sweep start" > $G/chain_10b_sweep.out
PROBS='llama10b_dp_grads_L8 llama10b_dp_grads_L24' STEPS=60 RUN_TIMEOUT=900 NP_OUT=$G/sy7_10b_sweep bash $G/g_newprobs_run2.sh >> $G/chain_10b_sweep.out 2>&1
echo SY7_10B_SWEEP_DONE >> $G/chain_10b_sweep.out
