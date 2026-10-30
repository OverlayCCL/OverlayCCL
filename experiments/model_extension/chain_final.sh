#!/bin/bash
# lean, self-cleaning chain: v9 diagnostic -> amp6 flat-buffer arms -> remaining amp6 arms -> hunt. Every run has a 20-min cap and
# every node is cleaned (leftover processes, stale compile locks, incomplete cache modules) after each run.
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
cp $G/train_llama_e2e_amp6_flatbuf.py /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
OUT9=$G/chain_v9.out; OUT6=$G/chain_amp6.out; echo "=== [$(date -u +%T)] chain_final: v9 contiguous-buffer diagnostic" > $OUT9
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v9.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_V9 = {"hs_a_64x64k_512k": ("per_tensor_ar","gate_1"), "probe_het_48_small": ("per_tensor_ar","gate_2"), "layer_grads_mixed_dtype": ("per_tensor_ar","gate_1")}
for _p, _k in _V9.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
mkdir -p $G/sy7_v9; for p in hs_a_64x64k_512k probe_het_48_small layer_grads_mixed_dtype; do echo '{"skipped": true}' > $G/sy7_v9/newprob_bench_$p.json; done
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_v9 bash $G/g_newprobs_run2.sh >> $OUT9 2>&1
cleanall; echo V9_DIAG_DONE >> $OUT9
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0; mkdir -p $G/amp6/json
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT6; timeout ${RT:-1200} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT6; cleanall; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*.json $G/amp6/json/ 2>/dev/null; }
echo "=== [$(date -u +%T)] chain_final: amp6 flat-buffer arms" >> $OUT6
C="N_LAYERS_PER_STAGE=2 N_MB=4"; CF="$C GRAD_BUFFER=flat"
R fb_iso_bund_per_tensor bundled 300 fb_iso_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $CF
R fb_iso_bund_sites bundled 300 fb_iso_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $CF
R fb_iso_base_permb per_mb 300 fb_iso_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $CF
R fb_iso_base_once per_mb 300 fb_iso_base_once BASE_SYNC=once SYNC_GRAPH=isolated $CF
R fb_iso_bund_nosim bundled 300 fb_iso_bund_nosim GRAD_SYNC=nosim SYNC_GRAPH=isolated $CF
collect; echo AMP6_FLATBUF_DONE >> $OUT6
echo "=== [$(date -u +%T)] chain_final: remaining amp6 arms" >> $OUT6
R iso_base_once per_mb 300 iso_base_once BASE_SYNC=once SYNC_GRAPH=isolated $C
R iso_bund_per_tensor_rep2 bundled 300 iso_bund_per_tensor_rep2 GRAD_SYNC=per_tensor SYNC_GRAPH=isolated SYNC_REPEAT=2 $C
R iso_bund_sites_rep2 bundled 300 iso_bund_sites_rep2 GRAD_SYNC=sites SYNC_GRAPH=isolated SYNC_REPEAT=2 $C
R iso_bund_nosim bundled 300 iso_bund_nosim GRAD_SYNC=nosim SYNC_GRAPH=isolated $C
R iso_bund_flat bundled 300 iso_bund_flat GRAD_SYNC=flat SYNC_GRAPH=isolated $C
collect; echo AMP6_ISO_DONE >> $OUT6
R fused_base_permb per_mb 300 fused_base_permb BASE_SYNC=per_mb SYNC_GRAPH=fused $C
R fused_bund_sites bundled 300 fused_bund_sites GRAD_SYNC=sites SYNC_GRAPH=fused $C
R fused_bund_per_tensor bundled 300 fused_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=fused $C
R iso2_base_permb per_mb 300 iso2_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated2 $C
R iso2_bund_sites bundled 300 iso2_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated2 $C
R iso2_bund_per_tensor bundled 300 iso2_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated2 $C
collect; echo AMP6_FORMS_DONE >> $OUT6
set -- $WORKERS; W5="$1 $2 $3 $4"; W3="$1 $2"
WORKERS_OVERRIDE="$W5" R n5_base_permb per_mb 300 n5_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W5" R n5_bund_per_tensor bundled 300 n5_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W5" R n5_bund_sites bundled 300 n5_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_base_permb per_mb 300 n3_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_bund_per_tensor bundled 300 n3_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_bund_sites bundled 300 n3_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
collect; echo AMP6_SUBSETS_DONE >> $OUT6
R iso_base_permb_s1 per_mb 300 iso_base_permb_s1 BASE_SYNC=per_mb SYNC_GRAPH=isolated $C --seed 1
R iso_bund_per_tensor_s1 bundled 300 iso_bund_per_tensor_s1 GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C --seed 1
R iso_bund_sites_s1 bundled 300 iso_bund_sites_s1 GRAD_SYNC=sites SYNC_GRAPH=isolated $C --seed 1
collect; echo AMP6_ALL_DONE >> $OUT6
echo "=== [$(date -u +%T)] chain_final: hunt" >> $OUT9
cp $G/g_newprobs_hunt.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
PROBS='fsdp_ag_layout tp_ar_decomp' RUN_TIMEOUT=900 NP_OUT=$G/sy7_hunt bash $G/g_newprobs_run2.sh >> $OUT9 2>&1
cleanall; echo SY7_HUNT_DONE >> $OUT9
