#!/bin/bash
# amp7 program, ordered by value per cluster-minute (block reclaimed ~11:00 UTC 10-30):
#  smoke -> 7n e2e (5 arms) -> 5n/3n e2e -> single-site e2e -> knob sweep -> cold-cache break-even pair -> e2e seed 1
#  -> per-problem rounds 2-3 at 7n (v10: per-tensor + microbench winner; v11ag 3 cands) -> 1-node per-problem training (last)
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
cleanall; OUT=$G/chain_amp7.out; echo "=== [$(date -u +%T)] amp7 start" > $OUT
cp $G/g_newprobs_sy7.py /home/ubuntu/OverlayCCL/training/g_newprobs.py; cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/train_llama_e2e_amp7.py /home/ubuntu/OverlayCCL/experiments/model_extension/
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/training/g_newprobs.py ubuntu@$ip:/home/ubuntu/OverlayCCL/training/; scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp7.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
export RESULTS_DIR=$G/amp7 PERCALL_PROBE=0; mkdir -p $G/amp7/json
R() { tag=$1; shift; t0=$(date +%s); echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-2400} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp7.py "$@" > $G/amp7/$tag.out 2>&1; echo "wall_s=$(( $(date +%s) - t0 ))" >> $OUT; grep -h "\[bench\]" $G/amp7/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 | cut -c1-200 >> $OUT; grep -h "Error\|NCC_" $G/amp7/logs/*${tag}*node0.log 2>/dev/null | grep -v "known hosts\|Deprecat\|socket\|elastic\|ChildFailed" | sort -u | head -2 | cut -c1-200 >> $OUT; cleanall; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp7_*.json" $G/amp7/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp7_*.json $G/amp7/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated"
RT=2400 R s7_base_microbench per_mb 60 s7_base_microbench BASE_STRAT=microbench $C
collect; echo AMP7_SMOKE_DONE >> $OUT
# ---- 7-node e2e (table e2e): baseline = microbench winners per microbatch; OverlayCCL = per-tensor once (+ native gather)
R a7_base_microbench per_mb 300 a7_base_microbench BASE_STRAT=microbench $C
R a7_bund_pertensor_native bundled 300 a7_bund_pertensor_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
R a7_bund_microbench bundled 300 a7_bund_microbench GRAD_SYNC=microbench FSDP_AG=dim0_t_native $C
collect; echo AMP7_E2E_DONE >> $OUT
# ---- cluster-size generalization (table 9)
set -- $WORKERS; W5="$1 $2 $3 $4"; W3="$1 $2"
WORKERS_OVERRIDE="$W5" R a5_base_microbench per_mb 300 a5_base_microbench BASE_STRAT=microbench $C
WORKERS_OVERRIDE="$W5" R a5_bund_pertensor_native bundled 300 a5_bund_pertensor_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
WORKERS_OVERRIDE="$W3" R a3_base_microbench per_mb 300 a3_base_microbench BASE_STRAT=microbench $C
WORKERS_OVERRIDE="$W3" R a3_bund_pertensor_native bundled 300 a3_bund_pertensor_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C
collect; echo AMP7_SUBSETS_DONE >> $OUT
# ---- configuration-knob sweep (table 7): microbatches and layers per stage
for K in "S=2048 N_MB=4 N_LAYERS_PER_STAGE=2" "S=1024 N_MB=8 N_LAYERS_PER_STAGE=2" "S=2048 N_MB=8 N_LAYERS_PER_STAGE=2"; do
  tag=$(echo $K | tr ' =' '__')
  R k7_base_$tag per_mb 300 k7_base_$tag BASE_STRAT=microbench SYNC_GRAPH=isolated $K
  R k7_bund_$tag bundled 300 k7_bund_$tag GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native SYNC_GRAPH=isolated $K
done
collect; echo AMP7_KNOBS_DONE >> $OUT
# ---- cold-cache pair for the break-even step count (fresh compile cache on every node)
for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "rm -rf /tmp/neuron_cache_cold" >/dev/null 2>&1; done; rm -rf /tmp/neuron_cache_cold
NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache_cold R cold_base_microbench per_mb 300 cold_base_microbench BASE_STRAT=microbench $C NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache_cold
NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache_cold R cold_bund_pertensor_native bundled 300 cold_bund_pertensor_native GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache_cold
collect; echo AMP7_COLD_DONE >> $OUT
# ---- e2e second seed
R a7_base_microbench_s1 per_mb 300 a7_base_microbench_s1 BASE_STRAT=microbench $C --seed 1
R a7_bund_pertensor_native_s1 bundled 300 a7_bund_pertensor_native_s1 GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C --seed 1
collect; echo AMP7_SEED_DONE >> $OUT
# ---- per-problem rounds 2-3 at 7 nodes (multi-seed, table 8): per-tensor + microbench winner
for r in 2 3 ${EXTRA_ROUNDS:-}; do
  cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_MS2 = {"layer_grads_hetero_ar": ("per_tensor_ar","flat_cat_ar"), "model_grads_4layers": ("per_tensor_ar","flat_cat_ar"), "probe_het_24_mid": ("per_tensor_ar","bucket6"), "probe_het_48_mid": ("per_tensor_ar","bucket6"), "hetero_grad_ar_many": ("per_tensor_ar","p3_pick3"), "hs_c_32x256k_2m": ("per_tensor_ar","bucket_4mb"), "layer_grads_mixed_dtype": ("per_tensor_ar","cast_flat_bf16"), "hs_a_64x64k_512k": ("per_tensor_ar","bucket_4mb"), "probe_het_48_small": ("per_tensor_ar","bucket6")}
for _p, _k in _MS2.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  mkdir -p $G/sy7_v10_r$r; for p in layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small; do echo '{"skipped": true}' > $G/sy7_v10_r$r/newprob_bench_$p.json; done
  PROBS='layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small' STEPS=100 RUN_TIMEOUT=1500 NP_OUT=$G/sy7_v10_r$r bash $G/g_newprobs_run2.sh >> $OUT 2>&1
  cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_train_v11ag.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["fsdp_ag_layout"]["cands"] = {c: f for c, f in PROBLEMS["fsdp_ag_layout"]["cands"].items() if c in ("ag_pinned","ag_dim0_t_native")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  mkdir -p $G/sy7_v11ag_r$r; echo '{"skipped": true}' > $G/sy7_v11ag_r$r/newprob_bench_fsdp_ag_layout.json
  PROBS='fsdp_ag_layout' STEPS=100 RUN_TIMEOUT=1500 NP_OUT=$G/sy7_v11ag_r$r bash $G/g_newprobs_run2.sh >> $OUT 2>&1
  cleanall; echo "ROUND_${r}_DONE" >> $OUT
done
echo AMP7_ROUNDS_DONE >> $OUT
# ---- no-simulator arm (table ablation-nosim-perproblem / ablation-nosim): picks materialized locally into g_newprobs_nosim7.py
if [ -f $G/g_newprobs_nosim7.py ]; then
  cp $G/g_newprobs_nosim7.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py
  mkdir -p $G/sy7_v10_nosim; for p in layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small; do echo '{"skipped": true}' > $G/sy7_v10_nosim/newprob_bench_$p.json; done
  PROBS='layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small' STEPS=100 RUN_TIMEOUT=1500 NP_OUT=$G/sy7_v10_nosim bash $G/g_newprobs_run2.sh >> $OUT 2>&1
  cleanall; cp $G/g_newprobs_nosim7.py /home/ubuntu/OverlayCCL/training/g_newprobs.py; for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/training/g_newprobs.py ubuntu@$ip:/home/ubuntu/OverlayCCL/training/; done
  R a7_bund_nosim bundled 300 a7_bund_nosim GRAD_SYNC=nosim $C
  collect
fi
echo AMP7_NOSIM_DONE >> $OUT
# ---- 1-node per-problem training (32 ranks on the master), last
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=/home/ubuntu/OverlayCCL
mkdir -p $G/sy7_v10_1n $G/sy7_v11ag_1n; echo "=== [$(date -u +%T)] 1-node per-problem training" >> $OUT
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_newprobs_sy7.py training/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py; cp $G/g_train_v10.py training/g_train.py
for P in layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small; do
  for Cn in $(python3 -c "import sys; sys.path.insert(0,'$G'); from g_newprobs import PROBLEMS; print(' '.join(PROBLEMS['$P']['cands']))"); do
    RESULTS_DIR=$G/sy7_v10_1n timeout 1500 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29703 training/g_train.py --problem $P --backend $Cn --steps 100 --warmup 3 > $G/sy7_v10_1n/${P}__${Cn}.log 2>&1
    echo "--- 1n $P $Cn: $(grep -h '\[train\]' $G/sy7_v10_1n/${P}__${Cn}.log | head -1 | cut -c1-120)" >> $OUT; bash $G/node_clean.sh >/dev/null 2>&1
  done
done
cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_newprobs_hunt_picks.py training/g_newprobs.py; cp $G/g_train_v11ag.py $G/g_train.py; cp $G/g_train_v11ag.py training/g_train.py
for Cn in ag_pinned ag_dim0_t ag_dim0_t_native; do
  RESULTS_DIR=$G/sy7_v11ag_1n timeout 1500 $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29703 training/g_train.py --problem fsdp_ag_layout --backend $Cn --steps 100 --warmup 3 > $G/sy7_v11ag_1n/fsdp_ag_layout__${Cn}.log 2>&1
  echo "--- 1n fsdp_ag_layout $Cn: $(grep -h '\[train\]' $G/sy7_v11ag_1n/fsdp_ag_layout__${Cn}.log | head -1 | cut -c1-120)" >> $OUT; bash $G/node_clean.sh >/dev/null 2>&1
done
echo V10_1N_DONE >> $OUT
# ---- per-problem rounds 4-5 (to n=5), if time permits
EXTRA_ROUNDS="4 5"
for r in 4 5; do
  cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_MS2 = {"layer_grads_hetero_ar": ("per_tensor_ar","flat_cat_ar"), "model_grads_4layers": ("per_tensor_ar","flat_cat_ar"), "probe_het_24_mid": ("per_tensor_ar","bucket6"), "probe_het_48_mid": ("per_tensor_ar","bucket6"), "hetero_grad_ar_many": ("per_tensor_ar","p3_pick3"), "hs_c_32x256k_2m": ("per_tensor_ar","bucket_4mb"), "layer_grads_mixed_dtype": ("per_tensor_ar","cast_flat_bf16"), "hs_a_64x64k_512k": ("per_tensor_ar","bucket_4mb"), "probe_het_48_small": ("per_tensor_ar","bucket6")}
for _p, _k in _MS2.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  mkdir -p $G/sy7_v10_r$r; for p in layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small; do echo '{"skipped": true}' > $G/sy7_v10_r$r/newprob_bench_$p.json; done
  PROBS='layer_grads_hetero_ar model_grads_4layers probe_het_24_mid probe_het_48_mid hetero_grad_ar_many hs_c_32x256k_2m layer_grads_mixed_dtype hs_a_64x64k_512k probe_het_48_small' STEPS=100 RUN_TIMEOUT=1500 NP_OUT=$G/sy7_v10_r$r bash $G/g_newprobs_run2.sh >> $OUT 2>&1
  cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_train_v11ag.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["fsdp_ag_layout"]["cands"] = {c: f for c, f in PROBLEMS["fsdp_ag_layout"]["cands"].items() if c in ("ag_pinned","ag_dim0_t_native")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  mkdir -p $G/sy7_v11ag_r$r; echo '{"skipped": true}' > $G/sy7_v11ag_r$r/newprob_bench_fsdp_ag_layout.json
  PROBS='fsdp_ag_layout' STEPS=100 RUN_TIMEOUT=1500 NP_OUT=$G/sy7_v11ag_r$r bash $G/g_newprobs_run2.sh >> $OUT 2>&1
  cleanall; echo "ROUND_${r}_DONE" >> $OUT
done
echo AMP7_ROUNDS45_DONE >> $OUT
# ---- e2e seeds 2-4 (multi-seed e2e row), if time permits
for sd in 2 3 4; do
  R a7_base_microbench_s$sd per_mb 300 a7_base_microbench_s$sd BASE_STRAT=microbench $C --seed $sd
  R a7_bund_pertensor_native_s$sd bundled 300 a7_bund_pertensor_native_s$sd GRAD_SYNC=per_tensor FSDP_AG=dim0_t_native $C --seed $sd
done
collect; echo AMP7_SEEDS_DONE >> $OUT
echo AMP7_ALL_DONE >> $OUT
