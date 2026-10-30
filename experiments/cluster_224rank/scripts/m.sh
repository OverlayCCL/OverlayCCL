#!/bin/bash
# run a command on the master: bash /tmp/ap7/m.sh "<remote command>"
source /tmp/ap7/cluster.env
exec ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 ubuntu@$MASTER_PUB "$@"
