#!/bin/bash
# Regenerate every numeric table of the paper from the measurements in this repository (CPU only; no cluster needed).
# Run on the analysis host after `bash reproduce/setup_layout.sh` (the simulator-scoring scripts use the symlinked layout).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
RB=experiments/cluster_224rank; NP=experiments/cross_scope_2026_10_26/new_problems; PT=experiments/cross_scope_2026_10_26/paper_tables
hdr() { printf '\n\n################ %s\n' "$*"; }
hdr "Tables 3 (lower block), 6 (lower block + Llama-7B shell row): gradient-sync and prefetch problems, mirror harness, n=5"
python3 $PT/paper_fill.py $NP/results $NP/results/amp7_json
hdr "Tables 4, 5, 9 Llama-7B rows"
python3 reproduce/e2e_tables.py $NP/results/amp7_json
hdr "Table 4 / 5 OLMoE rows: training/parse_olmoe_results.py over $RB/results/olmoe_results"
python3 training/parse_olmoe_results.py $RB/results/olmoe_results 2>/dev/null || echo "(see $RB/docs/results_summary.md)"
hdr "Table 7 (upper block) and Table 10 (eight-problem columns): 74-candidate pool vs 224-rank step time"
( cd $RB && EXTRA_DIRS="results/sweep_a2av,results/sweep_a2av_r2,results/sweep_a2av_r3,results/sweep_ua2a_fix,results/sweep_ua2a_fix_r2,results/sweep_found" \
  python3 scripts/fidelity_table.py results/sweep_v2 results/sweep_v2/sim_v4_v16b.json --arms 'results/sensitivity_arms/sim_v4_*.json' ) 2>/dev/null || echo "(see $RB/results/sim_ablation_table.md and $RB/docs/sim_correlation_final.md)"
hdr "Table 7 (lower block) and Table 10 (grad. column): gradient-sync problems, original constants vs mirror-harness medians"
python3 $NP/fidelity_new.py 2>/dev/null || cat $NP/results/fidelity_new.md
hdr "Table 8 / Table S9 / Tables S4-S8: search cost per style, non-LLM searches, per-problem ablations"
echo "Phase-3 wall and token cost per style: last lines of the run_search.py logs ($RB/results/queue.log; $NP/results/phase3/*.log)."
echo "Non-LLM searches: $RB/results/gs_b400 (scripts/generic_search.py) and $NP/results/nonllm (run_nonllm.sh)."
echo "No-sim per-problem picks and their 224-rank training cost: $NP/results/sy7_v10_nosim vs $NP/results/sy7_v10* (python3 $NP/simdecides_table.py)."
hdr "Figure 2 (cross-scope inversion bars) and the workflow figure"
python3 figures/gen_paper_figures.py 2>/dev/null && echo "wrote figures/disagreement.pdf figures/workflow.pdf" || echo "(needs matplotlib)"
