#!/bin/bash
# after the lost knob arm: seeds 2-4 of the Llama-7B e2e pair at 224 ranks (n=5 for the multi-seed table)
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
OUT=$G/chain_amp7.out
until grep -q AMP7_K7S2048_DONE $OUT 2>/dev/null; do sleep 60; done
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
export RESULTS_DIR=$G/amp7 PERCALL_PROBE=0
R() { local tg=$1; shift; local t0=$(date +%s); echo "=== [$(date -u +%T)] $tg" >> $OUT; timeout ${RT:-1800} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp7.py "$@" > $G/amp7/$tg.out 2>&1; echo "wall_s=$(( $(date +%s) - t0 ))" >> $OUT; grep -h "\[bench\]" $G/amp7/logs/*${tg}*node*.log 2>/dev/null | sort -u | head -1 | cut -c1-200 >> $OUT; cleanall; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp7_*.json" $G/amp7/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp7_*.json $G/amp7/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated"
for sd in 2 3 4; do
  # stop starting new arms after 09:45 UTC (final pull 10:30)
  [ "$(date -u +%H%M)" -gt 0945 ] && break
  R a7_base_microbench_s$sd per_mb 300 a7_base_microbench_s$sd BASE_STRAT=microbench $C --seed $sd
  R a7_bund_pertensor_native_s$sd bundled 300 a7_bund_pertensor_native_s$sd GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C --seed $sd
  collect; echo "AMP7_SEED${sd}_DONE" >> $OUT
done
echo AMP7_SEEDS234_DONE >> $OUT
