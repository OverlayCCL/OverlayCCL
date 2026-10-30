#!/bin/bash
# fused-graph check on the 2-node pair: the collective inside the step graph (no graph cut), baseline vs best bundle vs gate winner
G=/home/ubuntu/g2
while ! grep -q AMP5C_DONE $G/amp5/chainc.out 2>/dev/null; do sleep 30; done
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v7.py $G/g_train.py
PROBS='layer_grads_hetero_ar hs_a_64x64k_512k model_grads_4layers probe_het_48_small hetero_grad_ar_many' RUN_TIMEOUT=1500 TRAIN_EXTRA=--fused NP_OUT=$G/day5fused bash $G/g_newprobs_run2.sh > $G/day5fused.out 2>&1
echo FUSED_DONE >> $G/day5fused.out
