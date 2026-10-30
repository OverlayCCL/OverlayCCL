#!/bin/bash
# measure the loop's picks for fsdp_ag_layout at 224 ranks (training + 7-node microbench), after hunt4
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
cleanall() { bash $G/node_clean.sh >/dev/null 2>&1; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done; }
until grep -q PROBE_DONE $G/chain_probe.out 2>/dev/null; do sleep 20; done
cleanall; OUT=$G/chain_hunt5.out; echo "=== [$(date -u +%T)] hunt5 start" > $OUT
cp $G/g_newprobs_hunt_picks.py $G/g_newprobs.py; cp $G/g_train_v4.py $G/g_train.py
python3 - <<'PY'
s=open('/home/ubuntu/g2/g_newprobs.py').read()
s+='\nPROBLEMS["fsdp_ag_layout"]["cands"] = {c: f for c, f in PROBLEMS["fsdp_ag_layout"]["cands"].items() if c in ("ag_pinned","ag_dim0_t_native","loop_1")}\n'
open('/home/ubuntu/g2/g_newprobs.py','w').write(s)
PY
mkdir -p $G/sy7_hunt_picks; echo '{"skipped": true}' > $G/sy7_hunt_picks/newprob_bench_fsdp_ag_layout.json
PROBS='fsdp_ag_layout' RUN_TIMEOUT=900 NP_OUT=$G/sy7_hunt_picks bash $G/g_newprobs_run2.sh >> $OUT 2>&1
export RESULTS_DIR=$G/sy7_hunt_picks OUT=$G/sy7_hunt_picks; cp $G/g_newprobs.py /home/ubuntu/OverlayCCL/training/g_newprobs.py
for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR /home/ubuntu/OverlayCCL/training/g_newprobs.py ubuntu@$ip:/home/ubuntu/OverlayCCL/training/; done
timeout 900 bash experiments/cluster_224rank/scripts/run_ap7.sh training/g_bench2.py bench 30 bench224_picks_fsdp_ag_layout --problem fsdp_ag_layout > $G/sy7_hunt_picks/bench224.out 2>&1
grep -h "\[bench2\]" $G/sy7_hunt_picks/logs/*bench224_picks*node0.log 2>/dev/null | cut -c1-200 >> $G/chain_hunt5.out; cleanall
echo HUNT5_DONE >> $G/chain_hunt5.out
