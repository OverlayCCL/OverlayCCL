#!/bin/bash
# clean amp3-equivalent shell (no site A/B): baseline vs bundled, pinned vs native FSDP gather; 7 nodes, then 5 and 3 nodes
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
true
cleanall; OUT=$G/chain_clean.out; echo "=== [$(date -u +%T)] clean shell start" > $OUT
cp $G/train_llama_e2e_amp6_clean.py /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-1200} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; cleanall; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*.json $G/amp6/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated SITES=0"
R c7_base_pinned per_mb 300 c7_base_pinned BASE_SYNC=per_mb FSDP_AG=pinned $C
R c7_bund_pinned bundled 300 c7_bund_pinned GRAD_SYNC=per_tensor FSDP_AG=pinned $C
R c7_bund_native bundled 300 c7_bund_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
R c7_base_native per_mb 300 c7_base_native BASE_SYNC=per_mb FSDP_AG=dim0_t_native $C
collect; echo CLEAN7_DONE >> $OUT
set -- $WORKERS; W5="$1 $2 $3 $4"; W3="$1 $2"
WORKERS_OVERRIDE="$W5" R c5_base_pinned per_mb 300 c5_base_pinned BASE_SYNC=per_mb FSDP_AG=pinned $C
WORKERS_OVERRIDE="$W5" R c5_bund_pinned bundled 300 c5_bund_pinned GRAD_SYNC=per_tensor FSDP_AG=pinned $C
WORKERS_OVERRIDE="$W5" R c5_bund_native bundled 300 c5_bund_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
WORKERS_OVERRIDE="$W3" R c3_base_pinned per_mb 300 c3_base_pinned BASE_SYNC=per_mb FSDP_AG=pinned $C
WORKERS_OVERRIDE="$W3" R c3_bund_pinned bundled 300 c3_bund_pinned GRAD_SYNC=per_tensor FSDP_AG=pinned $C
WORKERS_OVERRIDE="$W3" R c3_bund_native bundled 300 c3_bund_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
collect; echo CLEAN_SUBSETS_DONE >> $OUT
R c7_base_pinned_s1 per_mb 300 c7_base_pinned_s1 BASE_SYNC=per_mb FSDP_AG=pinned $C --seed 1
R c7_bund_pinned_s1 bundled 300 c7_bund_pinned_s1 GRAD_SYNC=per_tensor FSDP_AG=pinned $C --seed 1
R c7_bund_native_s1 bundled 300 c7_bund_native_s1 GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C --seed 1
collect; echo CLEAN_ALL_DONE >> $OUT
