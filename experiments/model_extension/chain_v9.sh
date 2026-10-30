#!/bin/bash
# after the amp6 matrix: (1) v9 contiguous-buffer diagnostic on the three per-problem sets, (2) amp6 arms with GRAD_BUFFER=flat, (3) the hunt problems
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q AMP6_ALL_DONE $G/chain_amp6.out 2>/dev/null; do sleep 20; done
pkill -f chain_hunt.sh; sleep 1; pkill -f run_ap7.sh; pkill -f 'training/g_train.py'; pkill -f torchrun; sleep 3
for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "pkill -f 'training/g_train.py'; pkill -f torchrun" >/dev/null 2>&1; done
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
OUT=$G/chain_v9.out; echo "=== [$(date -u +%T)] v9 contiguous-buffer diagnostic" > $OUT
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
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_v9 bash $G/g_newprobs_run2.sh >> $OUT 2>&1
echo V9_DIAG_DONE >> $OUT
# ship the patched amp6 to the workers, then the flat-buffer arms
cp /home/ubuntu/g2/train_llama_e2e_amp6_flatbuf.py /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py; for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/experiments/model_extension/train_llama_e2e_amp6.py ubuntu@$ip:/home/ubuntu/OverlayCCL/experiments/model_extension/; done
export RESULTS_DIR=$G/amp6 PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag" >> $OUT; timeout ${RT:-2400} bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp6.py "$@" > $G/amp6/$tag.out 2>&1; grep -h "\[bench\]" $G/amp6/logs/*${tag}*node*.log 2>/dev/null | sort -u | head -1 >> $OUT; }
collect() { for w in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "ubuntu@$w:/tmp/tp_search/llama_e2e_amp6_*.json" $G/amp6/json/ 2>/dev/null; done; cp /tmp/tp_search/llama_e2e_amp6_*.json $G/amp6/json/ 2>/dev/null; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 GRAD_BUFFER=flat"
R fb_iso_base_permb per_mb 300 fb_iso_base_permb BASE_SYNC=per_mb SYNC_GRAPH=isolated $C
R fb_iso_bund_per_tensor bundled 300 fb_iso_bund_per_tensor GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C
R fb_iso_bund_sites bundled 300 fb_iso_bund_sites GRAD_SYNC=sites SYNC_GRAPH=isolated $C
R fb_iso_base_once per_mb 300 fb_iso_base_once BASE_SYNC=once SYNC_GRAPH=isolated $C
R fb_iso_bund_nosim bundled 300 fb_iso_bund_nosim GRAD_SYNC=nosim SYNC_GRAPH=isolated $C
collect; echo AMP6_FLATBUF_DONE >> $OUT
# the hunt problems
cp $G/g_newprobs_hunt.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
PROBS='fsdp_ag_layout tp_ar_decomp' RUN_TIMEOUT=900 NP_OUT=$G/sy7_hunt bash $G/g_newprobs_run2.sh >> $OUT 2>&1
echo SY7_HUNT_DONE >> $OUT
