#!/bin/bash
# extra repeats (rounds 3-5) of baseline vs gate winner at 224 ranks for the cast problem and the two best grouping problems
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
until grep -q SY7_10B_SWEEP_DONE $G/chain_10b_sweep.out 2>/dev/null; do sleep 15; done
for i in $(seq 1 40); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
echo "=== [$(date -u +%T)] ms start" > $G/chain_ms.out
for r in 3 4 5; do
  cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_MS = {"hs_a_64x64k_512k": ("per_tensor_ar","gate_1"), "probe_het_48_small": ("per_tensor_ar","gate_2")}
for _p, _k in _MS.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  PROBS='hs_a_64x64k_512k probe_het_48_small' RUN_TIMEOUT=900 NP_OUT=$G/sy7_train_s$r bash $G/g_newprobs_run2.sh >> $G/chain_ms.out 2>&1
  cp $G/g_train_v5.py $G/g_train.py
  python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["layer_grads_mixed_dtype"]["cands"] = {c: f for c, f in PROBLEMS["layer_grads_mixed_dtype"]["cands"].items() if c in ("per_tensor_ar","gate_1")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
  PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=900 NP_OUT=$G/sy7_train_s${r}_dtype bash $G/g_newprobs_run2.sh >> $G/chain_ms.out 2>&1
  echo "ROUND_${r}_DONE" >> $G/chain_ms.out
done
echo SY7_MS_DONE >> $G/chain_ms.out
