#!/bin/bash
# fill the one empty Table-3 cell: 224-rank microbench (forward-collective harness v5) of probe_het_48_mid with bucket6
G=/home/ubuntu/g2; cd /home/ubuntu/OverlayCCL; source /home/ubuntu/venv.env
OUT=$G/chain_amp7.out; echo "=== [$(date -u +%T)] 48mid bucket6 microbench" >> $OUT
cp $G/g_newprobs_sy7.py $G/g_newprobs.py; cp $G/g_train_v5.py $G/g_train.py
python3 -c "s=open('/home/ubuntu/g2/g_newprobs.py').read(); s+='\nPROBLEMS[\"probe_het_48_mid\"][\"cands\"] = {c: f for c, f in PROBLEMS[\"probe_het_48_mid\"][\"cands\"].items() if c in (\"bucket6\",)}\n'; open('/home/ubuntu/g2/g_newprobs.py','w').write(s)"
mkdir -p $G/sy7_train_48mid; echo '{"skipped": true}' > $G/sy7_train_48mid/newprob_bench_probe_het_48_mid.json
PROBS='probe_het_48_mid' STEPS=60 RUN_TIMEOUT=2400 NP_OUT=$G/sy7_train_48mid bash $G/g_newprobs_run2.sh >> $OUT 2>&1
bash $G/node_clean.sh >/dev/null 2>&1
for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh" >/dev/null 2>&1; done
echo MB48_DONE >> $OUT
