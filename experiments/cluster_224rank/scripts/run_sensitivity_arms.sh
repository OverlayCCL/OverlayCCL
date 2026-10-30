#!/bin/bash
# simulator sensitivity arms (paper task c): rescore the measured candidate pool under constant swaps
set -u; cd ~/work/paper/overlayccl_public/experiments/cluster_224rank; source ~/work/ncc/venv/bin/activate
R=$PWD; export REPO=~/work/paper/overlayccl_public CT_V4=$R/sim/correctness_test_v4.py TABLE=$R/results/hwprobe/hw_probe_measured.json TABLES=0 SCOPE=harness RESET=fusion MULTS='{"ring_kv":2,"alltoallv":32,"uniform_a2a":384}' OUTDIR=/tmp/simrun MANIFEST=/tmp/simrun/manifest_local.json
run () { local tag=$1; shift; (cd $REPO && env "$@" TAG=$tag python $R/scripts/score_v4.py > /tmp/simrun/score_$tag.log 2>&1); echo "arm $tag done: $(grep -c "V4=" /tmp/simrun/score_$tag.log) scored, $(grep -c "V4=None" /tmp/simrun/score_$tag.log) failed"; }
run base_v16b X=1
run net_x0p25 NET_SCALE=0.25
run net_x0p5 NET_SCALE=0.5
run net_x2 NET_SCALE=2
run net_x4 NET_SCALE=4
run net_x8 NET_SCALE=8
run local_snapshot LOCAL=snapshot
run reset_graph RESET=graph
run knee_64k KNEE=65536
run knee_1m KNEE=1048576
run opc_isolated OPC=isolated
run fwdbwd_0 FWDBWD=0
run fwdbwd_2 FWDBWD=2
echo ARMS_DONE
