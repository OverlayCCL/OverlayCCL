#!/bin/bash
# 10B DP-gradient-set probe at 224 ranks: wait for the second-seed batch to finish, stop the 48_mid long rerun, run the probe.
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q SY7B_S2_DONE $G/chain_sy7b.out 2>/dev/null; do sleep 20; done
echo "=== [$(date -u +%T)] S2 done; stopping chain_sy7b" > $G/chain_10b.out
bash $G/kill_sy7b.sh >> $G/chain_10b.out 2>&1
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
cp $G/g_newprobs_10b.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
mkdir -p $G/sy7_10b; echo '{"skipped": true}' > $G/sy7_10b/newprob_bench_llama10b_dp_grads.json
echo "=== [$(date -u +%T)] probe start" >> $G/chain_10b.out
PROBS='llama10b_dp_grads' STEPS=60 RUN_TIMEOUT=1200 NP_OUT=$G/sy7_10b bash $G/g_newprobs_run2.sh >> $G/chain_10b.out 2>&1
echo SY7_10B_DONE >> $G/chain_10b.out
