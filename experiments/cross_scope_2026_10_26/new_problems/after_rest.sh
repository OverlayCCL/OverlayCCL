#!/bin/bash
# after chain_rest finishes: re-run the lost knob arm (S2048 M4, OverlayCCL pick) if the cluster is still alive
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
OUT=$G/chain_amp7.out
until grep -q AMP7_ALL_DONE $OUT 2>/dev/null; do sleep 60; done
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
export RESULTS_DIR=$G/amp7 PERCALL_PROBE=0
R() { local tg=$1; shift; local t0=$(date +%s); echo "=== [$(date -u +%T)] $tg" >> $OUT; timeout ${RT:-2400} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp7.py "$@" > $G/amp7/$tg.out 2>&1; echo "wall_s=$(( $(date +%s) - t0 ))" >> $OUT; grep -h "\[bench\]" $G/amp7/logs/*${tg}*node*.log 2>/dev/null | sort -u | head -1 | cut -c1-200 >> $OUT; cleanall; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp7_*.json" $G/amp7/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp7_*.json $G/amp7/json/ 2>/dev/null; }
cleanall
cp $G/g_newprobs_sy7.py /home/ubuntu/OverlayCCL/training/g_newprobs.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/training/g_newprobs.py ubuntu@$ip:/home/ubuntu/OverlayCCL/training/; done
R k7_bund_S2048_M4 bundled 300 k7_bund_S2048_M4 GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native SYNC_GRAPH=isolated S=2048 N_MB=4 N_LAYERS_PER_STAGE=2
collect; echo AMP7_K7S2048_DONE >> $OUT
