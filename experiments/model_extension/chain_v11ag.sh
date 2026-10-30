#!/bin/bash
# 7-node per-problem TRAINING of the all-gather layout problem (v11ag: forward gather feeding a matmul, true reduce-scatter backward)
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
until grep -q V10B_DONE $G/chain_v10b.out 2>/dev/null; do sleep 30; done
cleanall; OUT=$G/chain_v11ag.out; echo "=== [$(date -u +%T)] v11ag start" > $OUT
cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_train_v11ag.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["fsdp_ag_layout"]["cands"] = {c: f for c, f in PROBLEMS["fsdp_ag_layout"]["cands"].items() if c in ("ag_pinned","ag_dim0_t","ag_dim0_t_native")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
mkdir -p $G/sy7_v11ag; echo '{"skipped": true}' > $G/sy7_v11ag/newprob_bench_fsdp_ag_layout.json
PROBS='fsdp_ag_layout' STEPS=100 RUN_TIMEOUT=2400 NP_OUT=$G/sy7_v11ag bash $G/g_newprobs_run2.sh >> $OUT 2>&1
cleanall; echo V11AG_DONE >> $OUT
