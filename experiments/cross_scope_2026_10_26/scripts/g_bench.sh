#!/bin/bash
# 1-node bench of every manifest candidate (run ON master; uses only the master's 32 cores).
# Usage: bash g_bench.sh [problem-filter-regex]   -> /home/ubuntu/g2/bench_1node/<problem>__<cand>.json + bench.log
set -uo pipefail
FILTER=${1:-.}
G=/home/ubuntu/g2; REPO=/home/ubuntu/OverlayCCL; OUT=$G/bench_1node; mkdir -p $OUT
source /home/ubuntu/venv.env
export PATH=$VENV/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_NUM_CORES=32 NEURON_COMPILE_CACHE_URL=/tmp/neuron_cache PYTHONPATH=$REPO REPO=$REPO OUT=$OUT
python3 - "$FILTER" <<'PY' > /tmp/g_bench_plan.txt
import json, sys, re, os
flt = re.compile(sys.argv[1])
for c in json.load(open(os.environ.get('MANIFEST','/home/ubuntu/g2/manifest_g_node.json'))):
    if flt.search(c['problem']): print(c['problem'], c['cand'], c['kind'], c.get('code') or c.get('path') or '-')
PY
echo "=== bench: $(wc -l < /tmp/g_bench_plan.txt) runs ===" | tee -a $OUT/bench.log
while read -r P C K X; do
  TAG="${P}__${C}"
  [ -f $OUT/$TAG.json ] && python3 -c "import json,sys; d=json.load(open('$OUT/$TAG.json')); sys.exit(0 if d.get('med_ms') is not None else 1)" && { echo "skip $TAG"; continue; }
  echo "--- [$(date -u +%T)] $TAG" | tee -a $OUT/bench.log
  pkill -9 -f 'torchru[n]' 2>/dev/null; pkill -9 -f 'g_benc[h].py' 2>/dev/null; sleep 2
  for i in $(seq 1 30); do [ -z "$(sudo fuser /dev/neuron* 2>/dev/null)" ] && break; sleep 3; done
  if [ "$K" = baseline ]; then SRC="--baseline"; else SRC="--code $X"; fi
  timeout ${RUN_TIMEOUT:-1500} $VENV/bin/torchrun --nproc_per_node=32 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29701 \
      $G/g_bench.py --problem $P --cand $C $SRC > $OUT/$TAG.log 2>&1
  RC=$?
  grep -h "\[bench\]" $OUT/$TAG.log | head -1 | tee -a $OUT/bench.log
  [ -f $OUT/$TAG.json ] || echo "{\"problem\":\"$P\",\"cand\":\"$C\",\"status\":\"no result rc=$RC\"}" > $OUT/$TAG.json
done < /tmp/g_bench_plan.txt
echo "=== bench done $(date -u +%T) ===" | tee -a $OUT/bench.log
