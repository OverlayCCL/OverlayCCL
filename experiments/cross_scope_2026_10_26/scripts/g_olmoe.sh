#!/bin/bash
# 2-node OLMoE end-to-end (training/train_olmoe10b_3node.py, shapes derived from ws=64): one run per AllToAllV arm.
# arms: baseline (AG+T+RS inline) | paper (public runtime helper: AG + column slice) | rs_masked (one masked
# reduce-scatter) | ag_slice_pl0 (AG + column slice with the native all-gather, pin_layout=False).
# The harness imports runtime.trainium_alltoallv_3node for --backend agent, so each arm's runtime file is copied there.
set -uo pipefail
REPO=/home/ubuntu/OverlayCCL; AP=$REPO/experiments/cluster_224rank/scripts; G=/home/ubuntu/g2
OUT=${OLMOE_OUT:-$G/olmoe_2node}; mkdir -p $OUT
STEPS=${STEPS:-300}
source /home/ubuntu/venv.env
HARNESS=training/train_olmoe10b_3node.py
if [ -n "${SEQLEN:-}" ]; then
  HARNESS=training/train_olmoe10b_2node_s${SEQLEN}.py
  sed -E "s/^SEQLEN *= *256/SEQLEN    = ${SEQLEN}/" $REPO/training/train_olmoe10b_3node.py > $REPO/$HARNESS
  grep -n "^SEQLEN" $REPO/$HARNESS
fi
declare -A FILE=( [paper]=$G/cands_g/alltoallv__paper_runtime.py [rs_masked]=$G/cands_g/alltoallv__rs_masked_runtime.py [ag_slice_pl0]=$G/cands_g/alltoallv__ag_slice_pl0_runtime.py )
for ARM in ${ARMS:-baseline paper rs_masked ag_slice_pl0}; do
  [ -f $OUT/olmoe_$ARM.json ] && { echo "skip $ARM"; continue; }
  echo "--- [$(date -u +%T)] olmoe arm=$ARM" | tee -a $OUT/olmoe.log
  if [ "$ARM" = baseline ]; then BACKEND=baseline; else BACKEND=agent; cp ${FILE[$ARM]} $REPO/runtime/trainium_alltoallv_3node.py; fi
  export RESULTS_DIR=$OUT/raw_$ARM; mkdir -p $RESULTS_DIR
  for ip in $WORKERS; do ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip "mkdir -p $RESULTS_DIR" </dev/null; done
  timeout ${RUN_TIMEOUT:-3600} bash $AP/run_ap7.sh $HARNESS $BACKEND $STEPS olmoe_$ARM --grad-sync baseline --ce baseline >> $OUT/olmoe.log 2>&1
  RC=$?
  for ip in $WORKERS; do scp -q -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$ip:"$RESULTS_DIR/olmoe10b_*.json" $RESULTS_DIR/ 2>/dev/null; done
  F=$(ls $RESULTS_DIR/olmoe10b_*.json 2>/dev/null | grep -v _phase | head -1)
  if [ -n "$F" ]; then
    python3 - "$F" "$OUT/olmoe_$ARM.json" "$ARM" "$RC" <<'PY'
import json, sys, statistics
d = json.load(open(sys.argv[1])); ts = d.get("step_times_ms") or []
d["arm"] = sys.argv[3]; d["rc"] = int(sys.argv[4])
d["measured_ms"] = statistics.median(ts[len(ts)//2:int(len(ts)*0.93)]) if len(ts) > 60 else d.get("steady_avg_step_ms")
d["steady_window_ms"] = ts[len(ts)//2:int(len(ts)*0.93)]
for k in ("step_times_ms", "losses"): d.pop(k, None)
json.dump(d, open(sys.argv[2], "w"), indent=1)
print(f"   arm={sys.argv[3]} measured_ms={d['measured_ms']} steady_avg={d.get('steady_avg_step_ms')} final_loss={d.get('final_loss')} rc={d['rc']}")
PY
  else
    echo "   NO RESULT (rc=$RC)" | tee -a $OUT/olmoe.log
  fi
  rm -f $REPO/runtime/trainium_alltoallv_3node.py
done
echo "=== olmoe done $(date -u +%T) ===" | tee -a $OUT/olmoe.log
