#!/bin/bash
# OLMoE-10B end-to-end (training/train_olmoe10b.py) at 224 ranks. Run ON the master after reproduce/setup_layout.sh master.
#
#   bash reproduce/e2e_olmoe10b.sh baseline   [steps]   # production strategy at every site
#   bash reproduce/e2e_olmoe10b.sh overlayccl [steps]   # deployed OverlayCCL strategies (AllToAllV pack-and-gather, dxe 2-AR)
#
# Table 4 uses 2500 real-OpenWebText steps (--realtok); Table 5 knob rows are the sibling scripts
# training/olmoe_sweep_s128.py, training/olmoe_sweep_d1024_s512.py, training/olmoe_sweep_l4.py (same flags, 1000 steps);
# Table 5 (lower block) 3-node row is training/train_olmoe10b_3node.py with WORKERS_OVERRIDE="<2 workers>". Appendix B's collective-share
# runs (EXDIM / TOPK / stub) are in experiments/cluster_224rank/scripts/queue*_ap7.sh.
set -uo pipefail
ARM=${1:?baseline|overlayccl}; STEPS=${2:-2500}
REPO=/home/ubuntu/OverlayCCL; cd $REPO; source /home/ubuntu/venv.env
export RESULTS_DIR=${RESULTS_DIR:-$REPO/training/results/7node}
case $ARM in
  baseline)   A="--backend baseline --ce baseline" ;;
  overlayccl) A="--backend agent --ce agent" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
SCRIPT=${OLMOE_SCRIPT:-training/train_olmoe10b.py}
bash experiments/cluster_224rank/scripts/run_ap7.sh $SCRIPT ${ARM} $STEPS olmoe10b_${ARM}_${STEPS} $A --steps $STEPS ${REALTOK:---realtok}
grep -h "steady\|final" $RESULTS_DIR/logs/olmoe10b_${ARM}_${STEPS}_node0.log | tail -3
