#!/bin/bash
# 1-node (32-rank) timing test of the harnesses: v10 on the layer set and the 64-vector set; v11ag on the gather problem
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=/home/ubuntu/OverlayCCL
mkdir -p $G/v10_test; echo "start $(date -u +%T)" > $G/v10_test/status.txt
T() { tag=$1; mod=$2; harness=$3; prob=$4; cand=$5; cp $G/$mod training/g_newprobs.py; cp $G/$mod $G/g_newprobs.py; cp $G/$harness training/g_train.py; cp $G/$harness $G/g_train.py
  RESULTS_DIR=$G/v10_test timeout 1500 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29711 training/g_train.py --problem $prob --backend $cand --steps 10 --warmup 2 > $G/v10_test/$tag.log 2>&1
  echo "$tag rc=$? $(date -u +%T): $(grep -h '\[train\]\|\[check\]' $G/v10_test/$tag.log | tr '\n' ' ' | cut -c1-160) $(grep -h 'Error' $G/v10_test/$tag.log | grep -v 'known hosts\|Deprecat\|elastic\|ChildFailed' | sort -u | head -1 | cut -c1-120)" >> $G/v10_test/status.txt; bash $G/node_clean.sh > /dev/null; }
T hsa_pt g_newprobs_sy7.py g_train_v10.py hs_a_64x64k_512k per_tensor_ar
T hsa_g1 g_newprobs_sy7.py g_train_v10.py hs_a_64x64k_512k gate_1
T ag_pinned g_newprobs_hunt.py g_train_v11ag.py fsdp_ag_layout ag_pinned
T ag_native g_newprobs_hunt.py g_train_v11ag.py fsdp_ag_layout ag_dim0_t_native
echo V10TEST_DONE >> $G/v10_test/status.txt
