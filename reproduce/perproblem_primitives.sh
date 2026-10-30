#!/bin/bash
# Per-problem measurements of the eight MoE / Llama primitives (Table 3 upper block, Table 6 upper blocks, Figure 2).
# Run ON the cluster master after reproduce/setup_layout.sh master.
#
#   bash reproduce/perproblem_primitives.sh bench1n            # 1-node bench column: 20-iteration isolated call at 32 ranks (experiments/h7_bench)
#   bash reproduce/perproblem_primitives.sh train7n [seed]     # 7-node bench + 7-node training columns: training/train_<primitive>_7node.py, both backends
#
# Each 7-node script writes <RESULTS_DIR>/<primitive>_<backend>*.json with steady_median_ms (per-step) and, with PERCALL_PROBE=1
# (the launcher default), the per-call probe used for the Llama-side rows of Table 6 (median per call x calls per step).
# Table 6 = this script at five seeds (PYTHONHASHSEED / --seed), mean +- sd per cell. The deployed strategies are the
# runtime/trainium_<primitive>_7node.py modules the search emitted; `baseline` selects the production strategy.
set -uo pipefail
MODE=${1:?bench1n|train7n}; SEED=${2:-0}
REPO=/home/ubuntu/OverlayCCL; cd $REPO; source /home/ubuntu/venv.env
export RESULTS_DIR=${RESULTS_DIR:-$REPO/training/results/7node}
case $MODE in
  bench1n)
    export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 PYTHONPATH=$REPO
    for b in a2av ua2a rkv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
      echo "=== bench_$b"; $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29703 experiments/h7_bench/bench_$b.py 2>&1 | grep -i 'median\|ms' | tail -4
    done ;;
  train7n)
    for p in alltoallv uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
      for bk in baseline agent; do
        PYTHONHASHSEED=$SEED bash experiments/cluster_224rank/scripts/run_ap7.sh training/train_${p}_7node.py $bk ${STEPS:-300} ${p}_${bk}_s$SEED --backend $bk --steps ${STEPS:-300}
      done
    done ;;
  *) echo "usage: $0 bench1n|train7n [seed]"; exit 1 ;;
esac
