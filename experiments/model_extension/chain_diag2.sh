#!/bin/bash
# why do the winners cost +27 ms in the shell when they cost +1-2 ms on the bare sets? layout + marginal-cost diagnostics
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
cp $G/train_llama_e2e_amp6_layout.py /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
OUT=$G/chain_diag2.out; echo "=== [$(date -u +%T)] diag2 start" > $OUT
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0; mkdir -p $G/amp6/json
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-1200} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; cleanall; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 SYNC_GRAPH=isolated"
R d_sites_direct bundled 300 d_sites_direct GRAD_SYNC=sites SITEA_LAYOUT=direct $C
R d_per_tensor_direct bundled 300 d_per_tensor_direct GRAD_SYNC=per_tensor SITEA_LAYOUT=direct $C
R d_sites_rep2 bundled 300 d_sites_rep2 GRAD_SYNC=sites SYNC_REPEAT=2 $C
R d_per_tensor_rep2 bundled 300 d_per_tensor_rep2 GRAD_SYNC=per_tensor SYNC_REPEAT=2 $C
for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done
echo DIAG2_E2E_DONE >> $OUT
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v8t.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["layer_grads_mixed_dtype"]["cands"] = {c: f for c, f in PROBLEMS["layer_grads_mixed_dtype"]["cands"].items() if c in ("per_tensor_ar","gate_1")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
mkdir -p $G/sy7_v8t; echo '{"skipped": true}' > $G/sy7_v8t/newprob_bench_layer_grads_mixed_dtype.json
PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_v8t bash $G/g_newprobs_run2.sh >> $OUT 2>&1
cleanall; echo DIAG2_DONE >> $OUT
