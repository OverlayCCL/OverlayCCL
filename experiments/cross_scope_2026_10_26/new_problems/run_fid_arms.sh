#!/bin/bash
# simulator scores (ORIGINAL constants) for the new problems' measured candidates under the constant-swap arms; HW = mirror harness 224r
cd /tmp/g2; source ~/work/ncc/venv/bin/activate; export OMP_NUM_THREADS=1 HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0 SCOPES=train G_MODULE=/tmp/g2/g_newprobs_fid.py
SETS9='layer_grads_hetero_ar,model_grads_4layers,probe_het_24_mid,probe_het_48_mid,hetero_grad_ar_many,hs_c_32x256k_2m,layer_grads_mixed_dtype,hs_a_64x64k_512k,probe_het_48_small'
run() { tag=$1; shift; env "$@" SETS=$SETS9 TAGV5=fid_$tag python sim_newprobs_v5.py > /tmp/g2/hw/fid_$tag.log 2>&1; echo "done $tag"; }
run calib
run net_half NET_SCALE=0.5
run net_quarter NET_SCALE=0.25
run net_x2 NET_SCALE=2
run net_x4 NET_SCALE=4
run local_snapshot LOCAL=snapshot
run opc_isolated OPC=isolated
run reset_graph RESET=graph
run knee_64k KNEE=65536
run knee_1m KNEE=1048576
run fwdbwd_0 FWDBWD=0
run fwdbwd_2 FWDBWD=2
G_MODULE=/tmp/g2/g_newprobs_hunt_picks.py SETS=fsdp_ag_layout TAGV5=fid_ag_calib python sim_newprobs_v5.py > /tmp/g2/hw/fid_ag_calib.log 2>&1; echo "done ag"
echo FID_ARMS_DONE
