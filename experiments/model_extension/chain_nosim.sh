#!/bin/bash
# 224-rank training-step measurement of the NO-SIM Phase-3 winners (paper ablation arm) for the cast + two grouping problems
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q SY7_MS_DONE $G/chain_ms.out 2>/dev/null; do sleep 15; done
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
echo "=== [$(date -u +%T)] nosim start" > $G/chain_nosim.out
cp $G/g_newprobs_nosim.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
mkdir -p $G/sy7_nosim; for p in hs_a_64x64k_512k probe_het_48_small; do echo '{"skipped": true}' > $G/sy7_nosim/newprob_bench_$p.json; done
PROBS='hs_a_64x64k_512k probe_het_48_small' RUN_TIMEOUT=900 NP_OUT=$G/sy7_nosim bash $G/g_newprobs_run2.sh >> $G/chain_nosim.out 2>&1
cp $G/g_train_v5.py $G/g_train.py; mkdir -p $G/sy7_nosim_dtype; echo '{"skipped": true}' > $G/sy7_nosim_dtype/newprob_bench_layer_grads_mixed_dtype.json
PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_nosim_dtype bash $G/g_newprobs_run2.sh >> $G/chain_nosim.out 2>&1
echo SY7_NOSIM_DONE >> $G/chain_nosim.out
