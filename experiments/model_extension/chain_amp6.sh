#!/bin/bash
# amp6 Llama e2e: four amp3 sites + the two per-problem gradient-sync sites (site A mixed-dtype layer set, site B 48 small vectors)
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q SY7_NOSIM_DONE $G/chain_nosim.out 2>/dev/null; do sleep 15; done
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0; mkdir -p $RESULTS_DIR $G/amp6/json
OUT=$G/chain_amp6.out; echo "=== [$(date -u +%T)] amp6 start" > $OUT
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-2400} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]\|\[init\] DM" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -3 >> $OUT; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*.json $G/amp6/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4"
# 0) smoke (60 steps)
RT=1800 R smoke_base per_mb 60 smoke_base SYNC_GRAPH=isolated $C
RT=1800 R smoke_sites bundled 60 smoke_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
collect; echo AMP6_SMOKE_DONE >> $OUT
# 1) isolated sync graph, 7 nodes
R iso_base_permb per_mb 300 iso_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
R iso_bund_sites bundled 300 iso_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
R iso_bund_per_tensor bundled 300 iso_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C
R iso_base_once per_mb 300 iso_base_once BASE_SYNC=once SYNC_GRAPH=isolated $C
R iso_bund_nosim bundled 300 iso_bund_nosim GRAD_SYNC=nosim SYNC_GRAPH=isolated $C
R iso_bund_flat bundled 300 iso_bund_flat GRAD_SYNC=flat SYNC_GRAPH=isolated $C
collect; echo AMP6_ISO_DONE >> $OUT
# 2) fused and isolated2 graph forms
R fused_base_permb per_mb 300 fused_base_permb BASE_SYNC=per_mb SYNC_GRAPH=fused $C
R fused_bund_sites bundled 300 fused_bund_sites GRAD_SYNC=sites SYNC_GRAPH=fused $C
R fused_bund_per_tensor bundled 300 fused_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=fused $C
R iso2_base_permb per_mb 300 iso2_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated2 $C
R iso2_bund_sites bundled 300 iso2_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated2 $C
R iso2_bund_per_tensor bundled 300 iso2_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated2 $C
collect; echo AMP6_FORMS_DONE >> $OUT
# 3) 5-node / 3-node subsets (isolated)
set -- $WORKERS; W5="$1 $2 $3 $4"; W3="$1 $2"
WORKERS_OVERRIDE="$W5" R n5_base_permb per_mb 300 n5_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W5" R n5_bund_sites bundled 300 n5_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_base_permb per_mb 300 n3_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
WORKERS_OVERRIDE="$W3" R n3_bund_sites bundled 300 n3_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
collect; echo AMP6_SUBSETS_DONE >> $OUT
# 4) second seed of the headline pair
R iso_base_permb_s1 per_mb 300 iso_base_permb_s1 BASE_SYNC=per_mb SYNC_GRAPH=isolated $C --seed 1
R iso_bund_sites_s1 bundled 300 iso_bund_sites_s1 GRAD_SYNC=sites SYNC_GRAPH=isolated $C --seed 1
collect; echo AMP6_ALL_DONE >> $OUT
