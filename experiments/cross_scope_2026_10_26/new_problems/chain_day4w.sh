#!/bin/bash
# after chain_day4z's day4f: hardware verification of the simulator-v8 picks (p3_pick3); runner v4 except the dtype problem
G=/home/ubuntu/g2
while ! grep -q DAY4F_DONE $G/day4f.out 2>/dev/null; do sleep 30; done
pkill -f 'chain_day4[z]'; sleep 1; while pgrep -f g_newprobs_run2.sh > /dev/null; do sleep 20; done
cp $G/g_newprobs_v23.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
mkdir -p $G/day4w/train; cp -n $G/day4g/train/newprob_train_*.json $G/day4w/train/ 2>/dev/null; rm -f $G/day4w/train/*p3_pick3*
PROBS='hs_a_64x64k_512k probe_het_48_small hetero_grad_ar_many probe_het_48_mid layer_grads_hetero_ar model_grads_4layers' RUN_TIMEOUT=1500 NP_OUT=$G/day4w bash $G/g_newprobs_run2.sh > $G/day4w.out 2>&1; echo DAY4W_DONE >> $G/day4w.out
cp $G/g_train_v5.py $G/g_train.py; mkdir -p $G/day4w5/train; cp -n $G/day4f/train/newprob_train_layer_grads_mixed_dtype__*.json $G/day4w5/train/ 2>/dev/null; rm -f $G/day4w5/train/*p3_pick3*
PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/day4w5 bash $G/g_newprobs_run2.sh > $G/day4w5.out 2>&1; echo DAY4W5_DONE >> $G/day4w5.out
cp $G/g_train_v4.py $G/g_train.py
PATH_P='path_no_norm_plus_tiny path_no_norm_1d path_no_norm_clone path_many_1d path_many_plus_tiny path_many_clone path_many_half eq95_512k eq111_1m eq4_unequal_total path_no_norm_plus_1024 path_no_norm_plus_65536 path_many_tiny_first path_many_sorted'
PROBS="$PATH_P" RUN_TIMEOUT=900 NP_OUT=$G/day4e_rep2 bash $G/g_newprobs_run2.sh > $G/day4e_rep2.out 2>&1; echo DAY4E2_DONE >> $G/day4e_rep2.out
