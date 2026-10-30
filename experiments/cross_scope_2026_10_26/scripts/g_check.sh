#!/bin/bash
# correctness of every code candidate vs the production baseline, one torchrun per problem (1 node)
set -uo pipefail
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; OUT=$G/bench_1node; mkdir -p $OUT
source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO:$G REPO=$REPO OUT=$OUT
for P in ${PROBLEMS:-uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar}; do
  echo "--- [$(date -u +%T)] check $P" | tee -a $OUT/check.log
  for i in $(seq 1 30); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
  $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29702 $G/g_check.py --problem $P --manifest ${MANIFEST:-$G/manifest_g_node.json} > $OUT/check_${CHECK_TAG:-}$P.log 2>&1
  grep -h "\[check\]" $OUT/check_${CHECK_TAG:-}$P.log | cut -c1-160 | tee -a $OUT/check.log
done
echo "=== check done $(date -u +%T) ===" | tee -a $OUT/check.log
