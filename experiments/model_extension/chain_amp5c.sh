#!/bin/bash
# GQA variant (heterogeneous replicated set): baseline vs bundled with each sync strategy, plain and 4x-amplified
cd /home/ubuntu/OverlayCCL; export RESULTS_DIR=/home/ubuntu/g2/amp5 PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag"; timeout 1800 bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp5.py "$@" > /home/ubuntu/g2/amp5/$tag.out 2>&1; grep -h "\[bench\]" /home/ubuntu/g2/amp5/logs/*${tag}*node*.log 2>/dev/null | head -1; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 GQA=4"
R gqa_base_per_tensor per_mb 300 gqa_base_per_tensor GRAD_SYNC=per_tensor $C
R gqa_bund_per_tensor bundled 300 gqa_bund_per_tensor GRAD_SYNC=per_tensor $C
R gqa_bund_per_layer bundled 300 gqa_bund_per_layer GRAD_SYNC=per_layer $C
R gqa_bund_flat bundled 300 gqa_bund_flat GRAD_SYNC=flat $C
R gqa_bund_buckets_8 bundled 300 gqa_bund_buckets_8 GRAD_SYNC=buckets_8 $C
R gqa_amp4_base_per_tensor per_mb 300 gqa_amp4_base_per_tensor GRAD_SYNC=per_tensor SYNC_REPEAT=4 $C
R gqa_amp4_bund_per_tensor bundled 300 gqa_amp4_bund_per_tensor GRAD_SYNC=per_tensor SYNC_REPEAT=4 $C
R gqa_amp4_bund_flat bundled 300 gqa_amp4_bund_flat GRAD_SYNC=flat SYNC_REPEAT=4 $C
R gqa_amp4_bund_per_layer bundled 300 gqa_amp4_bund_per_layer GRAD_SYNC=per_layer SYNC_REPEAT=4 $C
R gqa_mixed_bund_per_tensor bundled 300 gqa_mixed_bund_per_tensor GRAD_SYNC=per_tensor MIXED=1 $C
R gqa_mixed_bund_bf16_flat bundled 300 gqa_mixed_bund_bf16_flat GRAD_SYNC=bf16_flat MIXED=1 $C
mkdir -p /home/ubuntu/g2/amp5/json; cp /tmp/tp_search/llama_e2e_amp5_*.json /home/ubuntu/g2/amp5/json/ 2>/dev/null
source /home/ubuntu/venv.env; for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp5_*.json" /home/ubuntu/g2/amp5/json/ 2>/dev/null; done
echo AMP5C_DONE
