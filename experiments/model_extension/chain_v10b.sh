#!/bin/bash
# v10 mirror harness: the original-simulator loop picks (orig_1) + flat_cat_ar / two_dtype_groups controls, after chain_v10
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
until grep -q V10_DONE $G/chain_v10.out 2>/dev/null; do sleep 30; done
cleanall; OUT=$G/chain_v10b.out; echo "=== [$(date -u +%T)] v10b (orig picks) start" > $OUT
cp $G/g_newprobs_v10picks.py $G/g_newprobs.py; cp $G/g_train_v10.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_V10B = {"hs_a_64x64k_512k": ("orig_1","flat_cat_ar"), "probe_het_48_small": ("orig_1","flat_cat_ar"), "hs_c_32x256k_2m": ("orig_1","flat_cat_ar"), "layer_grads_mixed_dtype": ("two_dtype_groups",)}
for _p, _k in _V10B.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
PROBS='hs_a_64x64k_512k probe_het_48_small hs_c_32x256k_2m layer_grads_mixed_dtype' STEPS=100 RUN_TIMEOUT=2400 NP_OUT=$G/sy7_v10 bash $G/g_newprobs_run2.sh >> $OUT 2>&1
cleanall; echo V10B_DONE >> $OUT
