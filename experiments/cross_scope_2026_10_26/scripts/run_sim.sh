#!/bin/bash
cd ~/work/paper/overlayccl_public/experiments/cluster_224rank; source ~/work/ncc/venv/bin/activate
R=$PWD; export OMP_NUM_THREADS=1 REPO=$HOME/work/paper/overlayccl_public CT_V4=$R/sim/correctness_test_v4.py TABLE=$R/results/hwprobe/hw_probe_measured.json MANIFEST=/tmp/g2/manifest_g_local.json OUTDIR=/tmp/g2/sim
cd $REPO
SCOPE=bench WORLD=32  MULTS='{"ring_kv":2,"uniform_a2a":1664}' TAG=bench_32  python /tmp/g2/score_scopes.py > /tmp/g2/sim/log_bench_32.txt 2>&1 &
SCOPE=train WORLD=64  MULTS='{"ring_kv":2,"uniform_a2a":1024}' TAG=train_64  python /tmp/g2/score_scopes.py > /tmp/g2/sim/log_train_64.txt 2>&1 &
SCOPE=train WORLD=224 MULTS='{"ring_kv":2,"uniform_a2a":384}'  TAG=train_224 python /tmp/g2/score_scopes.py > /tmp/g2/sim/log_train_224.txt 2>&1 &
wait; echo SIM_DONE
