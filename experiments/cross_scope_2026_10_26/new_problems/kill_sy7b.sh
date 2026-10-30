#!/bin/bash
# stop chain_sy7b and whatever run_ap7 launched, on master and the six workers
pkill -f chain_sy7b.sh; sleep 1
pkill -f run_ap7.sh; pkill -f g_newprobs_run2; pkill -f 'training/g_train.py'; pkill -f torchrun; sleep 2
source /home/ubuntu/venv.env
for ip in $WORKERS; do ssh -n -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "pkill -f 'training/g_train.py'; pkill -f torchrun; pkill -f 'training/g_train.py'" >/dev/null 2>&1; done
sleep 3; pgrep -af 'g_train|torchrun|chain_sy7b' | grep -v pgrep | wc -l
