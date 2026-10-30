#!/bin/bash
cd /tmp/g2; source ~/work/ncc/venv/bin/activate
G() { echo "=== [$(date -u +%T)] gate $*"; python3 /tmp/g2/gate_training.py "$@" --run 2>&1 | grep "\[gate\]"; }
GATE_SEED_DIRS="gate_hetero_grad_ar_many" GATE_KEEP=per_tensor_ar,p3_pick3 G hetero_grad_ar_many hetero_grad_ar_many v12 4 --extra=v8
GATE_SEED_DIRS="gate_hs_c_32x256k_2m" GATE_KEEP=per_tensor_ar,bucket_4mb G hs_c_32mixed hs_c_32x256k_2m v12 4
GATE_SEED_DIRS="gate_probe_het_48_mid" GATE_KEEP=per_tensor_ar,bucket6 G probe_het_48_mid probe_het_48_mid v12 4
echo GATES2_DONE
