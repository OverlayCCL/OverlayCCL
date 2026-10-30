#!/bin/bash
# per-site timing inside the shell (PERCALL_PROBE=1: the FSDP gathers are timed with a host sync in the probe window) for pinned vs native
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
until grep -q CLEAN_ALL_DONE $G/chain_clean.out 2>/dev/null; do sleep 20; done
cleanall; OUT=$G/chain_probe.out; echo "=== [$(date -u +%T)] probe start" > $OUT
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=1
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-1200} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; cleanall; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated SITES=0"
R p7_bund_pinned bundled 300 p7_bund_pinned GRAD_SYNC=per_tensor FSDP_AG=pinned $C
R p7_bund_native bundled 300 p7_bund_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*percall*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*percall*.json $G/amp6/json/ 2>/dev/null
echo PROBE_DONE >> $OUT
