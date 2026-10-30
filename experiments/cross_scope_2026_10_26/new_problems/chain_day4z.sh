#!/bin/bash
# picks with the ORIGINAL runner (v4, comparable with all earlier rounds); dtype-honoring runner (v5) only for day4f
G=/home/ubuntu/g2
cp $G/g_newprobs_v22.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
rm -rf $G/day4g; mkdir -p $G/day4g/train
PROBS='hs_a_64x64k_512k probe_het_48_small hetero_grad_ar_many hs_c_32x256k_2m probe_het_48_mid' RUN_TIMEOUT=1500 NP_OUT=$G/day4g bash $G/g_newprobs_run2.sh > $G/day4g.out 2>&1; echo DAY4G_DONE >> $G/day4g.out
cp $G/g_train_v5.py $G/g_train.py
PROBS='layer_grads_mixed_dtype model_grads_fp32' RUN_TIMEOUT=900 NP_OUT=$G/day4f bash $G/g_newprobs_run2.sh > $G/day4f.out 2>&1; echo DAY4F_DONE >> $G/day4f.out
cp $G/g_train_v4.py $G/g_train.py
PATH_P='path_no_norm_plus_tiny path_no_norm_1d path_no_norm_clone path_many_1d path_many_plus_tiny path_many_clone path_many_half eq95_512k eq111_1m eq4_unequal_total path_no_norm_plus_1024 path_no_norm_plus_65536 path_many_tiny_first path_many_sorted'
PROBS="$PATH_P" RUN_TIMEOUT=900 NP_OUT=$G/day4e_rep2 bash $G/g_newprobs_run2.sh > $G/day4e_rep2.out 2>&1; echo DAY4E2_DONE >> $G/day4e_rep2.out
echo CHAIN_DAY4Z_DONE >> $G/day4e_rep2.out
