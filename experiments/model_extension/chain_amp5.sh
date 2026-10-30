#!/bin/bash
# amp5 preliminary matrix on 2 nodes: baseline vs bundled with each gradient-sync strategy; plain and mixed-dtype; then amplified site
cd /home/ubuntu/OverlayCCL; export RESULTS_DIR=/home/ubuntu/g2/amp5 PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag"; timeout 1800 bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp5.py "$@" > /home/ubuntu/g2/amp5/$tag.out 2>&1; grep -h "\[bench\]" /home/ubuntu/g2/amp5/logs/*${tag}*node*.log 2>/dev/null | head -1; }
R base_per_tensor per_mb 300 base_per_tensor GRAD_SYNC=per_tensor
R bund_per_tensor bundled 300 bund_per_tensor GRAD_SYNC=per_tensor
R bund_per_layer bundled 300 bund_per_layer GRAD_SYNC=per_layer
R bund_flat bundled 300 bund_flat GRAD_SYNC=flat
R bund_buckets_8 bundled 300 bund_buckets_8 GRAD_SYNC=buckets_8
R bund_shape_class bundled 300 bund_shape_class GRAD_SYNC=shape_class
R mixed_base_per_tensor per_mb 300 mixed_base_per_tensor GRAD_SYNC=per_tensor MIXED=1
R mixed_bund_per_tensor bundled 300 mixed_bund_per_tensor GRAD_SYNC=per_tensor MIXED=1
R mixed_bund_bf16_flat bundled 300 mixed_bund_bf16_flat GRAD_SYNC=bf16_flat MIXED=1
R mixed_bund_flat_fp32 bundled 300 mixed_bund_flat_fp32 GRAD_SYNC=flat MIXED=1
R mixed_bund_shape_class bundled 300 mixed_bund_shape_class GRAD_SYNC=shape_class MIXED=1
R amp4_base_per_tensor per_mb 300 amp4_base_per_tensor GRAD_SYNC=per_tensor SYNC_REPEAT=4
R amp4_bund_per_tensor bundled 300 amp4_bund_per_tensor GRAD_SYNC=per_tensor SYNC_REPEAT=4
R amp4_bund_flat bundled 300 amp4_bund_flat GRAD_SYNC=flat SYNC_REPEAT=4
R amp4_bund_buckets_8 bundled 300 amp4_bund_buckets_8 GRAD_SYNC=buckets_8 SYNC_REPEAT=4
R amp4_mixed_bund_bf16_flat bundled 300 amp4_mixed_bund_bf16_flat GRAD_SYNC=bf16_flat MIXED=1 SYNC_REPEAT=4
mkdir -p /home/ubuntu/g2/amp5/json; cp /tmp/tp_search/llama_e2e_amp5_*.json /home/ubuntu/g2/amp5/json/ 2>/dev/null
source /home/ubuntu/venv.env; for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp5_*.json" /home/ubuntu/g2/amp5/json/ 2>/dev/null; done
echo AMP5_DONE
