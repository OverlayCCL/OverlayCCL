#!/bin/bash
# Per-problem measurements of the nine gradient-sync and prefetch problems at the three scopes of Table 3 (lower block),
# Table 6 and Table 7 (lower blocks). Run ON the cluster master after reproduce/setup_layout.sh master.
#
#   bash reproduce/perproblem_gradsync.sh microbench            # 224-rank microbench: collective issued in the forward on flat-buffer views (g_train_v5.py)
#   bash reproduce/perproblem_gradsync.sh train7n  [round]      # 224-rank training, mirror harness (g_train_v10.py; FSDP AG problem: g_train_v11ag.py); one round = one seed
#   bash reproduce/perproblem_gradsync.sh train1n               # same harnesses on one node (32 ranks)
#   bash reproduce/perproblem_gradsync.sh nosim                 # 224-rank training of the no-simulator picks (g_newprobs_nosim7.py; Table S7)
#
# Candidates per problem: per_tensor_ar (OverlayCCL's pick) and the microbench winner (developer baseline): flat_cat_ar for the
# layer and model sets, cast_flat_bf16 for the mixed-dtype set, bucket_4mb for hs_a / hs_c, bucket6 for the 24/48-piece probes;
# ag_pinned vs ag_dim0_t_native for the FSDP all-gather lowering. Results: $G/sy7_<mode>[_r<round>]/train/newprob_train_<problem>__<cand>.json
# (measured_ms = median step over the last 3/4 of the steps, collective_effective = the collective really ran). Rounds 1-5 of the
# paper are results/sy7_v10{,_r2..r5} and results/sy7_v11ag{,_r2..r5}; results/sy7_v10_1n, results/sy7_v11ag_1n are the 1-node column;
# results/sy7_train* are the microbench. Table rows: python experiments/cross_scope_2026_10_26/paper_tables/paper_fill.py <results dir> <amp7 json dir>.
set -uo pipefail
MODE=${1:?microbench|train7n|train1n|nosim}; ROUND=${2:-1}
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; cd $REPO; source /home/ubuntu/venv.env
OUT=$G/chain.out
PL='layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small'
PRUNE='
_MS2 = {"layer_grads_hetero_ar": ("per_tensor_ar","flat_cat_ar"), "model_grads_4layers": ("per_tensor_ar","flat_cat_ar"), "probe_het_24_mid": ("per_tensor_ar","bucket6"), "probe_het_48_mid": ("per_tensor_ar","bucket6"), "hs_c_32x256k_2m": ("per_tensor_ar","bucket_4mb"), "layer_grads_mixed_dtype": ("per_tensor_ar","cast_flat_bf16"), "hs_a_64x64k_512k": ("per_tensor_ar","bucket_4mb"), "probe_het_48_small": ("per_tensor_ar","bucket6")}
for _p, _k in _MS2.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'
cleanall() { bash $REPO/experiments/model_extension/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash $REPO/experiments/model_extension/node_clean.sh" >/dev/null 2>&1; done; }
prep_grads() { cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/$1 $G/g_train.py; python3 -c "s=open('$G/g_newprobs.py').read(); s+='''$PRUNE'''; open('$G/g_newprobs.py','w').write(s)"; }
prep_ag() { cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_train_v11ag.py $G/g_train.py
  python3 -c "s=open('$G/g_newprobs.py').read(); s+='\nPROBLEMS[\"fsdp_ag_layout\"][\"cands\"] = {c: f for c, f in PROBLEMS[\"fsdp_ag_layout\"][\"cands\"].items() if c in (\"ag_pinned\",\"ag_dim0_t_native\")}\n'; open('$G/g_newprobs.py','w').write(s)"; }
skip_bench() { mkdir -p $1; for p in $2; do echo '{"skipped": true}' > $1/newprob_bench_$p.json; done; }   # the 1-node bench+correctness pass is not needed here
case $MODE in
  microbench)
    prep_grads g_train_v5.py; D=$G/sy7_microbench; skip_bench $D "$PL"
    PROBS="$PL" STEPS=300 RUN_TIMEOUT=2400 NP_OUT=$D bash $G/g_newprobs_run2.sh >> $OUT 2>&1
    prep_ag; cp $G/g_train_v5.py $G/g_train.py; D=$G/sy7_microbench_ag; skip_bench $D fsdp_ag_layout
    PROBS=fsdp_ag_layout STEPS=300 RUN_TIMEOUT=2400 NP_OUT=$D bash $G/g_newprobs_run2.sh >> $OUT 2>&1; cleanall ;;
  train7n)
    prep_grads g_train_v10.py; D=$G/sy7_v10_r$ROUND; skip_bench $D "$PL"
    PROBS="$PL" STEPS=60 RUN_TIMEOUT=1200 NP_OUT=$D bash $G/g_newprobs_run2.sh >> $OUT 2>&1
    prep_ag; D=$G/sy7_v11ag_r$ROUND; skip_bench $D fsdp_ag_layout
    PROBS=fsdp_ag_layout STEPS=60 RUN_TIMEOUT=1200 NP_OUT=$D bash $G/g_newprobs_run2.sh >> $OUT 2>&1; cleanall ;;
  nosim)
    cp $G/g_newprobs_nosim7.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py; D=$G/sy7_v10_nosim; skip_bench $D "$PL"
    PROBS="$PL" STEPS=60 RUN_TIMEOUT=1200 NP_OUT=$D bash $G/g_newprobs_run2.sh >> $OUT 2>&1; cleanall ;;
  train1n)
    export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO
    run1n() { local D=$1 P=$2 Cn=$3; mkdir -p $D; cp $G/g_newprobs.py $G/g_train.py training/
      RESULTS_DIR=$D timeout 1200 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29703 training/g_train.py --problem $P --backend $Cn --steps 100 --warmup 3 > $D/${P}__${Cn}.log 2>&1
      echo "--- 1n $P $Cn: $(grep -h '\[train\]' $D/${P}__${Cn}.log | head -1 | cut -c1-120)" | tee -a $OUT; bash $REPO/experiments/model_extension/node_clean.sh >/dev/null 2>&1; }
    prep_grads g_train_v10.py
    for P in $PL; do for Cn in $(python3 -c "import sys; sys.path.insert(0,'$G'); from g_newprobs import PROBLEMS; print(' '.join(PROBLEMS['$P']['cands']))"); do run1n $G/sy7_v10_1n $P $Cn; done; done
    prep_ag; for Cn in ag_pinned ag_dim0_t_native; do run1n $G/sy7_v11ag_1n fsdp_ag_layout $Cn; done ;;
  *) echo "usage: $0 microbench|train7n [round]|train1n|nosim"; exit 1 ;;
esac
echo "=== $MODE done $(date -u +%T)" | tee -a $OUT
