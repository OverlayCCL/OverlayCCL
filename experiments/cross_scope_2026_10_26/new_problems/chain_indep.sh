#!/bin/bash
# independent-pieces check (g_train v6, G_TRAIN_INDEPENDENT=1): are in-order buckets only "free" because the harness
# hands bundles adjacent views of one buffer?  hs_a: per_tensor / bucket_4mb (in-order) / p3_pick2 (size-sorted 8x8) /
# p3_pick3 (in-order 4 buckets) / flat; hetero_many: per_tensor / p3_pick3; 48_mid: per_tensor / bucket6 / p3_pick3 (sorted bins)
G=/home/ubuntu/g2
cp $G/g_newprobs_v23.py $G/g_newprobs.py; cp $G/g_train_v6.py $G/g_train.py; export G_TRAIN_INDEPENDENT=1
python3 - <<'PY'
import re
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
for _p, _keep in (("hs_a_64x64k_512k", ("per_tensor_ar","bucket_4mb","p3_pick2","p3_pick3","flat_cat_ar")), ("hetero_grad_ar_many", ("per_tensor_ar","p3_pick3","flat_cat_ar")), ("probe_het_48_mid", ("per_tensor_ar","bucket6","p3_pick3"))):
    PROBLEMS[_p]["cands"] = {k: v for k, v in PROBLEMS[_p]["cands"].items() if k in _keep}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
PROBS='hs_a_64x64k_512k hetero_grad_ar_many probe_het_48_mid' RUN_TIMEOUT=1500 NP_OUT=$G/day4indep bash $G/g_newprobs_run2.sh > $G/day4indep.out 2>&1; echo DAY4INDEP_DONE >> $G/day4indep.out
cp $G/g_train_v4.py $G/g_train.py
