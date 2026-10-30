#!/bin/bash
# Run the OverlayCCL search loop (Phases 1-5) on all seventeen collective problems of the paper.
#
#   bash reproduce/search_all.sh original     # the eight MoE/Llama primitives (Table 1, upper rows) - runs ON the cluster master
#   bash reproduce/search_all.sh gradsync     # the nine gradient-sync / prefetch problems            - runs on the analysis host (CPU + LLM)
#   bash reproduce/search_all.sh gradsync-nosim   # the same nine with the simulator hidden (Table 9 / Table S7 no-sim arm)
#
# The eight primitives are registered in search/problems.py; experiments/run_search.py runs calibration (Phase 1), seed
# scoring (Phase 2), the LLM proposer (Phase 3), hardware gates (Phase 4) and codegen (Phase 5) and writes
# runtime/trainium_<problem>[_7node].py. The nine gradient-sync and prefetch problems are registered in
# search/problems_newprobs.py; their Phase 2-3 run is run_phase3_v5.py (same loop, simulator at training scope), the
# emitted candidates are materialized into the hardware harness module with materialize_picks.py, and Phases 4-5 are the
# correctness/TV gates plus the lowest-simulator-score rule (simdecides_table.py prints the shipped pick per problem).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
MODE=${1:-original}
case $MODE in
  original)
    : "${MASTER_ADDR:?set MASTER_ADDR}"; : "${WORKER_ADDRS:?set WORKER_ADDRS (comma-separated)}"
    for prob in alltoallv uniform_a2a ring_kv dxe pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
      python experiments/run_search.py --problem $prob --hw-eval --num-nodes 7 \
        --master-addr "$MASTER_ADDR" --worker-addrs "$WORKER_ADDRS" \
        --phase3-style ${PHASE3_STYLE:-strategy-enumerate} --llm-model ${LLM_MODEL:-sonnet} ${EXTRA_SEARCH_ARGS:-}
    done ;;
  gradsync|gradsync-nosim)
    # original simulator constants (the heterogeneous-set and unequal-concat terms fitted during the study are OFF)
    export OMP_NUM_THREADS=1 HET_BW_BPUS=0 HET_KNEE_US=0 UNEQ_BW_BPUS=0
    NP=experiments/cross_scope_2026_10_26/new_problems; mkdir -p $NP/results/phase3
    RUNNER=$NP/run_phase3_v5.py; TAG=orig; [ "$MODE" = gradsync-nosim ] && { RUNNER=$NP/run_phase3_nosim.py; TAG=nosim; }
    # search-problem name -> hardware-problem name (sizes differ by the 1/64 sandbox scale; see materialize_picks.py)
    for p in fsdp_ag_layout layer_grads_hetero_ar model_grads_4layers layer_grads_mixed_dtype; do
      TAGP3=$TAG python $RUNNER $p ${MAX_ROUNDS:-4} > $NP/results/phase3/phase3_${TAG}_$p.log 2>&1; echo "done $p"; done
    for p in dp_grad_bucketing_64 hs_c_32mixed probe_het_24_mid probe_het_48_mid probe_het_48_small; do
      SCALE=1024 TAGP3=$TAG python $RUNNER $p ${MAX_ROUNDS:-4} > $NP/results/phase3/phase3_${TAG}_$p.log 2>&1; echo "done $p"; done
    echo "Phase-3 logs: $NP/results/phase3/phase3_${TAG}_*.log"
    echo "next: python $NP/materialize_picks.py <search problem> <hw problem> $TAG $NP/g_newprobs_sy7.py <out module> name_substr:cand ..."
    echo "      then reproduce/perproblem_gradsync.sh on the cluster for the correctness/TV gates, and python $NP/simdecides_table.py" ;;
  *) echo "usage: $0 original|gradsync|gradsync-nosim"; exit 1 ;;
esac
