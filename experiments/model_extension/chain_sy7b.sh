#!/bin/bash
# remaining 7-node time: isolated e2e seed 1; second seed of baseline + gate winner per problem at 224 ranks; 48-mid buckets with a longer timeout
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
export RESULTS_DIR=$G/sy7_e2e PERCALL_PROBE=0
R() { tag=$1; shift; echo "=== [$(date -u +%T)] $tag"; timeout 2400 bash experiments/cluster_224rank/scripts/run_ap7.sh experiments/model_extension/train_llama_e2e_amp5.py "$@" > $G/sy7_e2e/$tag.out 2>&1; grep -h "\[bench\]" $G/sy7_e2e/logs/*${tag}*node*.log 2>/dev/null | head -1; }
C="N_LAYERS_PER_STAGE=2 N_MB=4 GQA=4"
R e2e_iso_base_s1 per_mb 300 e2e_iso_base_s1 GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C --seed 1
R e2e_iso_bund_per_tensor_s1 bundled 300 e2e_iso_bund_per_tensor_s1 GRAD_SYNC=per_tensor SYNC_GRAPH=isolated $C --seed 1
echo SY7B_E2E_DONE
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='''
_SEED2 = {"hs_a_64x64k_512k": ("per_tensor_ar","gate_1"), "probe_het_48_small": ("per_tensor_ar","gate_2"), "layer_grads_hetero_ar": ("per_tensor_ar","gate_3"), "probe_het_24_mid": ("per_tensor_ar","gate_2"), "hs_c_32x256k_2m": ("per_tensor_ar","gate_3"), "hetero_grad_ar_many": ("per_tensor_ar","gate_2"), "model_grads_4layers": ("per_tensor_ar","gate_3")}
for _p, _k in _SEED2.items():
    if _p in PROBLEMS: PROBLEMS[_p]["cands"] = {c: f for c, f in PROBLEMS[_p]["cands"].items() if c in _k}
'''
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
PROBS='hs_a_64x64k_512k probe_het_48_small layer_grads_hetero_ar probe_het_24_mid hs_c_32x256k_2m hetero_grad_ar_many model_grads_4layers' RUN_TIMEOUT=1500 NP_OUT=$G/sy7_train_s2 bash $G/g_newprobs_run2.sh > $G/sy7_train_s2.out 2>&1
cp $G/g_train_v5.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["layer_grads_mixed_dtype"]["cands"] = {c: f for c, f in PROBLEMS["layer_grads_mixed_dtype"]["cands"].items() if c in ("per_tensor_ar","gate_1")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
PROBS='layer_grads_mixed_dtype' RUN_TIMEOUT=1500 NP_OUT=$G/sy7_train_s2_dtype bash $G/g_newprobs_run2.sh > $G/sy7_train_s2_dtype.out 2>&1
echo SY7B_S2_DONE
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["probe_het_48_mid"]["cands"] = {c: f for c, f in PROBLEMS["probe_het_48_mid"]["cands"].items() if c in ("bucket6","gate_4")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
PROBS='probe_het_48_mid' RUN_TIMEOUT=2700 NP_OUT=$G/sy7_train_48mid bash $G/g_newprobs_run2.sh > $G/sy7_train_48mid.out 2>&1
echo SY7B_ALL_DONE
