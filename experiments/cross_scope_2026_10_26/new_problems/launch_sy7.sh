#!/bin/bash
# Launch 7x trn1.32xlarge into the Sydney Capacity Block (1 master + 6 workers, 8 EFA NICs each, 500G gp3, no placement group).
set -euo pipefail
export AWS_PAGER=""
REGION=ap-southeast-2; CR=<CAPACITY_RESERVATION_ID>; AMI=<AMI_ID>; ITYPE=trn1.32xlarge; KEY=Kaiyao
SUBNET=subnet-0802b633757043bfe; SG=sg-041ed0ec1d7764349; NIC=/tmp/sy7/nic.json; OUTF=/tmp/sy7/cluster.env
python3 - <<PY > $NIC
import json
nics=[{"NetworkCardIndex":0,"DeviceIndex":0,"InterfaceType":"efa","SubnetId":"$SUBNET","Groups":["$SG"],"AssociatePublicIpAddress":True}]
nics+=[{"NetworkCardIndex":i,"DeviceIndex":1,"InterfaceType":"efa","SubnetId":"$SUBNET","Groups":["$SG"]} for i in range(1,8)]
print(json.dumps(nics))
PY
launch_one () { aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $ITYPE --key-name $KEY --count 1 \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CR}" --instance-market-options 'MarketType=capacity-block' \
  --network-interfaces "file://$NIC" --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$1}]" --query 'Instances[0].InstanceId' --output text; }
MASTER_ID=$(launch_one sy7-master); echo "master $MASTER_ID"
WORKER_IDS=""; for i in 1 2 3 4 5 6; do id=$(launch_one sy7-worker-$i); WORKER_IDS="$WORKER_IDS $id"; echo "worker $i $id"; done
for id in $MASTER_ID $WORKER_IDS; do aws ec2 wait instance-running --region $REGION --instance-ids "$id"; done
get_priv () { aws ec2 describe-instances --region $REGION --instance-ids "$1" --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].PrivateIpAddress' --output text; }
MASTER_PUB=$(aws ec2 describe-instances --region $REGION --instance-ids "$MASTER_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
MASTER_PRIV=$(get_priv "$MASTER_ID"); WP=""; for id in $WORKER_IDS; do WP="$WP $(get_priv $id)"; done
{ echo "REGION=$REGION"; echo "CR=$CR"; echo "MASTER_ID=$MASTER_ID"; echo "WORKER_IDS=\"$(echo $WORKER_IDS)\""; echo "MASTER_PUB=$MASTER_PUB"; echo "MASTER_PRIV=$MASTER_PRIV"; echo "WORKER_PRIVS=\"$(echo $WP)\""; echo "LAUNCHED_UTC=$(date -u +%FT%TZ)"; } | tee $OUTF
