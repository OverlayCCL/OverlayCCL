#!/bin/bash
cd /home/ubuntu/g2; source /home/ubuntu/venv.env
pkill -f 'chain_v1[0]'; pkill -f 'chain_v11a[g]'; pkill -f 'chain_amp[7]'; sleep 1; pkill -f 'run_ap[7].sh'; pkill -f 'g_newprobs_run[2]'; sleep 2
bash /home/ubuntu/g2/node_clean.sh; for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "bash /home/ubuntu/node_clean.sh"; done
rm -rf sy7_v10/train sy7_v10/run.log sy7_v10/run_train.log chain_v10.out chain_v10b.out chain_v11ag.out chain_amp7.out
chmod +x *.sh
nohup bash /home/ubuntu/g2/chain_v10.sh > chain_v10.log 2>&1 &
nohup bash /home/ubuntu/g2/chain_v10b.sh > chain_v10b.log 2>&1 &
nohup bash /home/ubuntu/g2/chain_v11ag.sh > chain_v11ag.log 2>&1 &
nohup bash /home/ubuntu/g2/chain_amp7.sh > chain_amp7.log 2>&1 &
disown -a; sleep 3; pgrep -af "chain_" | grep -v pgrep | awk '{print $NF}' | tr '\n' ' '; echo; date -u +%T
