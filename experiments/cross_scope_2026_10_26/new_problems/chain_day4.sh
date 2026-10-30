#!/bin/bash
# day-4 campaign on the 2026-10-27/28 block (instances reclaimed at 11:00 UTC): piece-count sweep, pathology diags,
# grouped re-validation with the effectiveness check, lost day-3e items, bench64 of diagnostics, repeats.
G=/home/ubuntu/g2
cp $G/g_newprobs_v20.py $G/g_newprobs.py
CATK=$(python3 -c "print(' '.join([f'catk_{k}_small' for k in (8,16,32,64,128)] + [f'catk_{k}_mid' for k in (8,16,32,48)]))")
PATH_P='path_no_norm_plus_tiny path_no_norm_1d path_no_norm_clone path_many_1d path_many_plus_tiny path_many_clone path_many_half'
PROBS="$CATK" RUN_TIMEOUT=900 NP_OUT=$G/day4a bash $G/g_newprobs_run2.sh > $G/day4a.out 2>&1; echo DAY4A_DONE >> $G/day4a.out
PROBS="$PATH_P" RUN_TIMEOUT=900 NP_OUT=$G/day4b bash $G/g_newprobs_run2.sh > $G/day4b.out 2>&1; echo DAY4B_DONE >> $G/day4b.out
PROBS='layer_grads_mixed_dtype probe_het_24_mid probe_het_48_mid probe_het_48_small' RUN_TIMEOUT=1500 NP_OUT=$G/day4p bash $G/g_newprobs_run2.sh > $G/day4p.out 2>&1; echo DAY4P_DONE >> $G/day4p.out
PROBS='dp_grad_ar_groups grp_small_dp_ar model_grads_ep_groups' RUN_TIMEOUT=900 NP_OUT=$G/day4c bash $G/g_newprobs_run2.sh > $G/day4c.out 2>&1; echo DAY4C_DONE >> $G/day4c.out
PROBS='model_grads_intra_node_groups model_grads_fp32' RUN_TIMEOUT=900 NP_OUT=$G/day4d bash $G/g_newprobs_run2.sh > $G/day4d.out 2>&1; echo DAY4D_DONE >> $G/day4d.out
PROBS="layer_grads_1d model_grads_1d layer_grads_no_norm model_grads_no_norm $PATH_P" B64_OUT=$G/bench64_4 bash $G/g_bench64.sh > $G/bench64_4.out 2>&1; echo BENCH64_4_DONE >> $G/bench64_4.out
PROBS="$CATK" RUN_TIMEOUT=900 NP_OUT=$G/day4a_rep2 bash $G/g_newprobs_run2.sh > $G/day4a_rep2.out 2>&1; echo DAY4A2_DONE >> $G/day4a_rep2.out
PROBS="$PATH_P" RUN_TIMEOUT=900 NP_OUT=$G/day4b_rep2 bash $G/g_newprobs_run2.sh > $G/day4b_rep2.out 2>&1; echo DAY4B2_DONE >> $G/day4b_rep2.out
echo CHAIN_DAY4_DONE >> $G/day4b_rep2.out
