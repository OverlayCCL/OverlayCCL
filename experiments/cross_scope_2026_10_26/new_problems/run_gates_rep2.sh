#!/bin/bash
# overnight repeat of every gate decision (only the gate candidates rerun; baseline/hand-written are pre-seeded)
cd /tmp/g2; source ~/work/ncc/venv/bin/activate
/tmp/g2/m.sh "cd /home/ubuntu/g2 && for d in gate_*; do [ -d \$d/train ] && mkdir -p rep1_\$d && cp -r \$d/train rep1_\$d/ ; done; ls -d rep1_gate_* | wc -l"
G() { echo "=== [$(date -u +%T)] gate-rep2 $*"; python3 /tmp/g2/gate_training.py "$@" --run 2>&1 | grep "\[gate\]"; }
GATE_SEED_DIRS="gate_hetero_grad_ar_many" GATE_KEEP=per_tensor_ar,p3_pick3 G hetero_grad_ar_many hetero_grad_ar_many v12 4 --extra=v8
GATE_SEED_DIRS="gate_hs_c_32x256k_2m" GATE_KEEP=per_tensor_ar,bucket_4mb G hs_c_32mixed hs_c_32x256k_2m v12 4
GATE_SEED_DIRS="gate_probe_het_48_mid" GATE_KEEP=per_tensor_ar,bucket6 G probe_het_48_mid probe_het_48_mid v12 4
GATE_SEED_DIRS="gate_hs_a_64x64k_512k" GATE_KEEP=per_tensor_ar,bucket_4mb,flat_cat_ar G dp_grad_bucketing_64 hs_a_64x64k_512k v11 3
GATE_SEED_DIRS="gate_probe_het_48_small" GATE_KEEP=per_tensor_ar,bucket6,flat_cat_ar G probe_het_48_small probe_het_48_small v11 3
GATE_SEED_DIRS="gate_layer_grads_hetero_ar" GATE_KEEP=per_tensor_ar,flat_cat_ar,bucket_32mb G layer_grads_hetero_ar layer_grads_hetero_ar v11 3
GATE_SEED_DIRS="gate_probe_het_24_mid" GATE_KEEP=per_tensor_ar,bucket6,flat_cat_ar G probe_het_24_mid probe_het_24_mid v11 3
GATE_SEED_DIRS="gate_model_grads_4layers" GATE_KEEP=per_tensor_ar,flat_cat_ar,per_layer_bucket G model_grads_4layers model_grads_4layers v11 3
GATE_SEED_DIRS="gate_zero2_hetero_rs" GATE_KEEP=per_tensor_rs,cat_dim1_rs,bucket_dim1_rs G zero2_hetero_rs zero2_hetero_rs v11 3
GATE_SEED_DIRS="gate_layer_grads_mixed_dtype_dtype" GATE_KEEP=per_tensor_ar,cast_flat_bf16,two_dtype_groups G layer_grads_mixed_dtype layer_grads_mixed_dtype v11 3 --dtype
echo GATES_REP2_DONE
