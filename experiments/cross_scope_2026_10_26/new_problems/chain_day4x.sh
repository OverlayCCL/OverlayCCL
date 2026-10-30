#!/bin/bash
# takes over from chain_day4 after the pick verification (day4p): E1-E7 experiments, HLO dumps, then the rest of day 4
G=/home/ubuntu/g2
while pgrep -f g_newprobs_run2.sh > /dev/null; do sleep 30; done
echo DAY4P_DONE >> $G/day4p.out
cp $G/g_newprobs_v21.py $G/g_newprobs.py
E='eq95_512k eq111_1m eq4_unequal_total path_no_norm_plus_1 path_no_norm_plus_64 path_no_norm_plus_1024 path_no_norm_plus_65536 path_many_tiny_first path_many_sorted md_pick_vs_hand'
PROBS="$E" RUN_TIMEOUT=900 NP_OUT=$G/day4e bash $G/g_newprobs_run2.sh > $G/day4e.out 2>&1; echo DAY4E_DONE >> $G/day4e.out
bash $G/hlo_dump.sh > $G/hlo_dump.out 2>&1
PROBS='dp_grad_ar_groups grp_small_dp_ar model_grads_ep_groups' RUN_TIMEOUT=900 NP_OUT=$G/day4c bash $G/g_newprobs_run2.sh > $G/day4c.out 2>&1; echo DAY4C_DONE >> $G/day4c.out
PROBS='model_grads_intra_node_groups model_grads_fp32' RUN_TIMEOUT=900 NP_OUT=$G/day4d bash $G/g_newprobs_run2.sh > $G/day4d.out 2>&1; echo DAY4D_DONE >> $G/day4d.out
CATK=$(python3 -c "print(' '.join([f'catk_{k}_small' for k in (8,16,32,64,128)] + [f'catk_{k}_mid' for k in (8,16,32,48)]))")
PATH_P='path_no_norm_plus_tiny path_no_norm_1d path_no_norm_clone path_many_1d path_many_plus_tiny path_many_clone path_many_half'
PROBS="layer_grads_1d model_grads_1d layer_grads_no_norm model_grads_no_norm $PATH_P $E" B64_OUT=$G/bench64_4 bash $G/g_bench64.sh > $G/bench64_4.out 2>&1; echo BENCH64_4_DONE >> $G/bench64_4.out
PROBS="$E $PATH_P" RUN_TIMEOUT=900 NP_OUT=$G/day4e_rep2 bash $G/g_newprobs_run2.sh > $G/day4e_rep2.out 2>&1; echo DAY4E2_DONE >> $G/day4e_rep2.out
PROBS="$CATK" RUN_TIMEOUT=900 NP_OUT=$G/day4a_rep2 bash $G/g_newprobs_run2.sh > $G/day4a_rep2.out 2>&1; echo DAY4A2_DONE >> $G/day4a_rep2.out
echo CHAIN_DAY4X_DONE >> $G/day4a_rep2.out
