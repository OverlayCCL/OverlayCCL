#!/bin/bash
# after the bundled per-tensor arm: stop chain_amp6, run the v8 (DDP-placement) per-problem diagnostic, then the remaining amp6 arms
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q "iso_base_once" $G/chain_amp6.out 2>/dev/null; do sleep 15; done
sleep 20; bash $G/kill_amp6.sh >> $G/chain_amp6.out 2>&1
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
OUT=$G/chain_amp6.out; echo "=== [$(date -u +%T)] chain_amp6b: v8 DDP-placement diagnostic" >> $OUT
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v8.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_V8 = {"hs_a_64x64k_512k": ("per_tensor_ar","gate_1","flat_cat_ar"), "probe_het_48_small": ("per_tensor_ar","gate_2","flat_cat_ar"), "layer_grads_mixed_dtype": ("per_tensor_ar","gate_1","two_dtype_groups")}
for _p, _k in _V8.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
mkdir -p $G/sy7_v8; for p in hs_a_64x64k_512k probe_het_48_small layer_grads_mixed_dtype; do echo '{"skipped": true}' > $G/sy7_v8/newprob_bench_$p.json; done
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_v8 bash $G/g_newprobs_run2.sh >> $OUT 2>&1
echo V8_DIAG_DONE >> $OUT
# ---- remaining amp6 arms
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-2400} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*.json $G/amp6/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4"
R iso_base_once per_mb 300 iso_base_once BASE_SYNC=once SYNC_GRAPH=isolated $C
R iso_bund_per_tensor_rep2 bundled 300 iso_bund_per_tensor_rep2 GRAD_SYNC=per_tensor SYNC_GRAPH=isolated SYNC_REPEAT=2 $C
R iso_bund_sites_rep2 bundled 300 iso_bund_sites_rep2 GRAD_SYNC=sites SYNC_GRAPH=isolated SYNC_REPEAT=2 $C
R iso_bund_nosim bundled 300 iso_bund_nosim GRAD_SYNC=nosim SYNC_GRAPH=isolated $C
R iso_bund_flat bundled 300 iso_bund_flat GRAD_SYNC=flat SYNC_GRAPH=isolated $C
collect; echo AMP6_ISO_DONE >> $OUT
R fused_base_permb per_mb 300 fused_base_permb BASE_SYNC=per_mb SYNC_GRAPH=fused $C
R fused_bund_sites bundled 300 fused_bund_sites GRAD_SYNC=sites SYNC_GRAPH=fused $C
R fused_bund_per_tensor bundled 300 fused_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=fused $C
R iso2_base_permb per_mb 300 iso2_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated2 $C
R iso2_bund_sites bundled 300 iso2_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated2 $C
R iso2_bund_per_tensor bundled 300 iso2_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated2 $C
collect; echo AMP6_FORMS_DONE >> $OUT
set -- $WORKERS; W5="$1 $2 $3 $4"; W3="$1 $2"
WORKERS_OVERRIDE="$W5" R n5_base_permb per_mb 300 n5_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W5" R n5_bund_sites bundled 300 n5_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_base_permb per_mb 300 n3_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_bund_sites bundled 300 n3_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
collect; echo AMP6_SUBSETS_DONE >> $OUT
R iso_base_permb_s1 per_mb 300 iso_base_permb_s1 BASE_SYNC=per_mb SYNC_GRAPH=isolated $C --seed 1
R iso_bund_sites_s1 bundled 300 iso_bund_sites_s1 GRAD_SYNC=sites SYNC_GRAPH=isolated $C --seed 1
collect; echo AMP6_ALL_DONE >> $OUT
