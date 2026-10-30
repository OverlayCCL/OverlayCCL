#!/bin/bash
# 2-node: isolated-sync-graph e2e arms (after the fused harness check), to compare with the fused matrix on the same pair
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; export RESULTS_DIR=$G/amp5 PERCALL_PROBE=0
while ! grep -q FUSED_DONE $G/day5fused.out 2>/dev/null; do sleep 30; done
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag"; timeout 1800 bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp5.py "$@" > $G/amp5/$tag.out 2>&1; grep -h "\[bench\]" $G/amp5/logs/*${tag}*node*.log 2>/dev/null | head -1; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 GQA=4 SYNC_GRAPH=isolated"
R iso_base_per_tensor per_mb 300 iso_base_per_tensor GRAD_SYNC=per_tensor $C
R iso_bund_per_tensor bundled 300 iso_bund_per_tensor GRAD_SYNC=per_tensor $C
R iso_bund_per_layer bundled 300 iso_bund_per_layer GRAD_SYNC=per_layer $C
R iso_bund_buckets_8 bundled 300 iso_bund_buckets_8 GRAD_SYNC=buckets_8 $C
R iso_bund_flat bundled 300 iso_bund_flat GRAD_SYNC=flat $C
cp /tmp/tp_search/llama_e2e_amp5_*.json $G/amp5/json/ 2>/dev/null; source /home/ubuntu/venv.env; for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp5_*.json" $G/amp5/json/ 2>/dev/null
echo ISO2N_DONE
