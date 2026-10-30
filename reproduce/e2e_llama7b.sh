#!/bin/bash
# Llama-7B end-to-end shell (experiments/model_extension/train_llama_e2e_amp7.py) on the cluster. Run ON the master after
# reproduce/setup_layout.sh master; needs /home/ubuntu/venv.env and $SSH_KEY (see reproduce/env.example).
#
#   bash reproduce/e2e_llama7b.sh baseline   [S] [N_MB] [seed]   # developer baseline: per-microbatch sync, microbench-winning bundles, pinned gather
#   bash reproduce/e2e_llama7b.sh overlayccl [S] [N_MB] [seed]   # OverlayCCL: bundled across microbatches, per-tensor sync once per step, native dim-0 gather
#   bash reproduce/e2e_llama7b.sh control    [S] [N_MB] [seed]   # OverlayCCL's schedule with the microbench strategies (strategy share, Table 4 text)
#   bash reproduce/e2e_llama7b.sh nosim      [S] [N_MB] [seed]   # the no-simulator picks at every gradient site (Table 9)
#
# Defaults S=1024 N_MB=4 seed=0 (Table 4 at 7 nodes). Table 5 knobs: S=2048 N_MB=4 and S=1024 N_MB=8. Table 5 (lower block) cluster
# sizes: WORKERS_OVERRIDE="<4 workers>" (5 nodes, 160 ranks) or "<2 workers>" (3 nodes, 96 ranks). Table 6 Llama-7B row:
# seeds 0..4. Wall clock from a cold cache (text of Sec. 3): COLD=1. 300 steps, steady-state = steps 100-300.
# Results: one JSON per arm under $RESULTS_DIR/json (default /home/ubuntu/g2/amp7/json), fields steady_median_ms /
# steady_mean_ms / final_loss / ws; the [bench] line is also appended to $RESULTS_DIR/chain.out.
set -uo pipefail
ARM=${1:?baseline|overlayccl|control|nosim}; S=${2:-1024}; M=${3:-4}; SEED=${4:-0}
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; cd $REPO; source /home/ubuntu/venv.env
export RESULTS_DIR=${RESULTS_DIR:-$G/amp7} PERCALL_PROBE=0
C="N_LAYERS_PER_STAGE=2 N_MB=$M S=$S SYNC_GRAPH=isolated"
case $ARM in
  baseline)   BK=per_mb;  ENVS="BASE_STRAT=microbench $C" ;;
  overlayccl) BK=bundled; ENVS="GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C" ;;
  control)    BK=bundled; ENVS="GRAD_SYNC=microbench FSDP_AG=dim0_t_native $C" ;;
  nosim)      BK=bundled; ENVS="GRAD_SYNC=nosim $C"
              # the no-sim picks live in g_newprobs_nosim7.py; the shell imports training/g_newprobs.py on every node
              cp $G/g_newprobs_nosim7.py $REPO/training/g_newprobs.py ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[ "$ARM" != nosim ] && cp $G/g_newprobs_sy7.py $REPO/training/g_newprobs.py
TAG=e2e_${ARM}_S${S}_M${M}_seed${SEED}
echo "=== [$(date -u +%T)] $TAG" >> $RESULTS_DIR/chain.out
bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp7.py $BK 300 $TAG $ENVS --seed $SEED > $RESULTS_DIR/$TAG.out 2>&1
grep -h "\[bench\]" $RESULTS_DIR/logs/*${TAG}*node*.log 2>/dev/null | sort -u | head -1 | tee -a $RESULTS_DIR/chain.out
mkdir -p $RESULTS_DIR/json; cp /tmp/tp_search/llama_e2e_amp7_*.json $RESULTS_DIR/json/ 2>/dev/null
for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp7_*.json" $RESULTS_DIR/json/ 2>/dev/null; done
# every run leaves compile locks / partial NEFF dirs if interrupted; clean all nodes so the next run cannot hang on them
bash $REPO/experiments/model_extension/node_clean.sh >/dev/null 2>&1
for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash $REPO/experiments/model_extension/node_clean.sh" >/dev/null 2>&1; done
