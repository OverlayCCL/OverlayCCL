#!/bin/bash
# bring up the 7-node Sydney cluster: ssh wait, key + bootstrap_ap7 (clone public repo, venv.env, worker sync), day-5 payload
set -uo pipefail
source /tmp/sy7/cluster.env
KEY=$SSH_KEY
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 ubuntu@$MASTER_PUB"
SCP="scp -q -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
for i in $(seq 1 60); do $SSH true 2>/dev/null && break; echo "$(date -u +%T) waiting for ssh ($i)"; sleep 20; done
$SSH true || { echo "master ssh failed"; exit 1; }
$SCP $KEY ubuntu@$MASTER_PUB:$SSH_KEY
$SCP ~/work/paper/overlayccl_public/experiments/cluster_224rank/scripts/bootstrap_ap7.sh ubuntu@$MASTER_PUB:/home/ubuntu/bootstrap_ap7.sh
$SSH "mkdir -p /home/ubuntu/g2 /tmp/tp_search"
for w in $WORKER_PRIVS; do for i in $(seq 1 40); do $SSH "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 ubuntu@$w true" 2>/dev/null && break; sleep 15; done; done
$SSH "chmod 600 $SSH_KEY && bash /home/ubuntu/bootstrap_ap7.sh '$WORKER_PRIVS'" 2>&1 | tail -8
$SCP /tmp/g2/g_newprobs_gates_all.py /tmp/g2/g_newprobs_sy7.py /tmp/g2/g_bench2.py /tmp/g2/g_train_v4.py /tmp/g2/g_train_v5.py /tmp/g2/g_train_v6.py /tmp/g2/g_train_v7.py /tmp/g2/g_newprobs_run.sh /tmp/g2/g_newprobs_run2.sh /tmp/g2/g_bench64.sh /tmp/g2/newprobs_table.py ubuntu@$MASTER_PUB:/home/ubuntu/g2/
$SCP ~/work/paper/overlayccl_public/experiments/model_extension/train_llama_e2e_amp5.py ubuntu@$MASTER_PUB:/home/ubuntu/OverlayCCL/experiments/model_extension/
$SSH "cd /home/ubuntu/OverlayCCL && python3 experiments/cluster_224rank/scripts/patch_pp.py >/dev/null 2>&1; chmod +x /home/ubuntu/g2/*.sh; cp /home/ubuntu/g2/g_newprobs_gates_all.py /home/ubuntu/g2/g_newprobs.py; cp /home/ubuntu/g2/g_train_v4.py /home/ubuntu/g2/g_train.py; for w in $WORKER_PRIVS; do ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@\$w 'mkdir -p /tmp/tp_search' </dev/null; done; cat /home/ubuntu/venv.env; nproc; neuron-ls 2>/dev/null | head -3"
echo "=== sy7 bootstrap done: $MASTER_PUB ==="
