#!/bin/bash
# where do the winners' +27 ms go? (1) HLO dumps of the sync+update graph for sites vs per-tensor, (2) split sync/update graphs,
# (3) site A vs site B attribution and the cast ingredient without bundling
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
cp $G/train_llama_e2e_amp6_diag3.py /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
OUT=$G/chain_diag3.out; echo "=== [$(date -u +%T)] diag3 start" > $OUT
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0; mkdir -p $G/amp6/json
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-1200} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; cleanall; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated"
rm -rf /tmp/hlo_sites /tmp/hlo_pt
R h_sites bundled 60 h_sites GRAD_SYNC=sites XLA_FLAGS=--xla_dump_to=/tmp/hlo_sites $C
R h_pt bundled 60 h_pt GRAD_SYNC=per_tensor XLA_FLAGS=--xla_dump_to=/tmp/hlo_pt $C
R d_sites_b_only bundled 300 d_sites_b_only GRAD_SYNC=sites_b_only $C
R d_sites_a_only bundled 300 d_sites_a_only GRAD_SYNC=sites_a_only $C
R d_sites_a_cast_pt bundled 300 d_sites_a_cast_pt GRAD_SYNC=sites_a_cast_pt $C
R d_iso2_sites bundled 300 d_iso2_sites GRAD_SYNC=sites N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated2
R d_iso2_pt bundled 300 d_iso2_pt GRAD_SYNC=per_tensor N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated2
for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done
echo DIAG3_DONE >> $OUT
