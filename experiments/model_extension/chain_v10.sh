#!/bin/bash
# v10 "mirror" per-problem training (e2e-like placement) for the nine all-reduce problems: per_tensor + hand-written microbench winner + gate winner
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
until grep -q PROBE_DONE $G/chain_probe.out 2>/dev/null && grep -q HUNT5_DONE $G/chain_hunt5.out 2>/dev/null && ! grep -q KeyError $G/chain_hunt5.out; do sleep 20; done
cleanall; OUT=$G/chain_v10.out; echo "=== [$(date -u +%T)] v10 mirror harness start" > $OUT
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py
mkdir -p $G/sy7_v10; for p in hs_a_64x64k_512k probe_het_48_small layer_grads_hetero_ar layer_grads_mixed_dtype model_grads_4layers probe_het_24_mid hs_c_32x256k_2m hetero_grad_ar_many probe_het_48_mid; do echo '{"skipped": true}' > $G/sy7_v10/newprob_bench_$p.json; done
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_hetero_ar layer_grads_mixed_dtype model_grads_4layers probe_het_24_mid hs_c_32x256k_2m hetero_grad_ar_many probe_het_48_mid' STEPS=100 RUN_TIMEOUT=2400 NP_OUT=$G/sy7_v10 bash $G/g_newprobs_run2.sh >> $OUT 2>&1
cleanall; echo V10_DONE >> $OUT
