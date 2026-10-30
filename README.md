# OverlayCCL

Anonymous artifact for the MLSys 2027 submission **_OverlayCCL: Composing Collectives Above a Vendor Library via a
Closed-Stack Search Loop_**.

OverlayCCL is a search loop that finds faster collective-communication strategies on top of a closed-source vendor
library. On AWS Trainium the vendor library (Neuron) exposes a fixed API; the loop never modifies it and never sees inside
it. It only chooses which primitives to call (`xm.all_gather`, `xm.reduce_scatter`, `xm.all_reduce`, ...), in what order,
how they are lowered, and how tensors are reshaped, padded or sliced around each call. An LLM agent proposes strategies at
this layer, a cost-model simulator whose constants the agent calibrates on the target device scores them, hardware
correctness gates filter them, and the lowest-simulator-score survivor is emitted as a drop-in runtime module.

**Headline results** (paper Table 4; 7 nodes x `trn1.32xlarge`, 224 ranks, steady-state ms/step over the developer's
production baseline):

| Shell | Baseline | OverlayCCL | Speedup |
|---|---|---|---|
| Llama-7B (ten collective sites, 300 steps, bit-identical loss) | 600.1 | 79.1 | **7.59x** (five seeds: 8.60x) |
| OLMoE-10B (2500 real-OpenWebText steps, matched descent) | 4404.3 | 3139.8 | **1.40x** |

Across seventeen collective problems the shipped strategies fall into five families: bundling across microbatches,
lifting graph boundaries, algebraic recomposition, collective-lowering selection with relayout, and de-bundling.

---

## Repository layout

```
search/          Phases 2-5: problem registry (problems.py; problems_newprobs.py for the gradient-sync and prefetch
                 problems), sandbox correctness test, strategy-enumerate / cc-react / multi-island Phase-3 loops
simulator/       the cost model (cost_model.py) and the topology object its constants live in
runtime/         deployed strategies: trainium_<problem>.py (1-node reference), trainium_<problem>_7node.py (224 ranks)
codegen/         Phase-5 codegen and validation glue
prompts/         LLM proposer prompt scaffolds
training/        OLMoE-10B shell, per-primitive 7-node training scripts, knob-sweep scripts
experiments/
  run_search.py                  the loop entry point (Phases 1-5) for the eight MoE / Llama primitives
  h7_bench/                      1-node lone-call benches (Table 3, 1n bench column)
  model_extension/               the Llama-7B shell (train_llama_e2e_amp7.py) and its launch chains; results_amp*_7node/
  cluster_224rank/              224-rank measurement campaign: N-node launcher (scripts/run_ap7.sh), Phase-1 probe table
                                 (results/hwprobe), 74-candidate pool and simulator arms (Tables 7 and 10), non-LLM searches
                                 (Table S9), OLMoE and Llama-7B result JSON, faithful simulator fork (sim/correctness_test_v5.py)
  cross_scope_2026_10_26/        gradient-sync and prefetch problems: harnesses, Phase-3 runs, 2-node and 7-node results
    new_problems/                g_newprobs_sy7.py (problem + candidate registry), g_train_v5.py (microbench harness),
                                 g_train_v10.py / g_train_v11ag.py (mirror training harnesses), run_phase3_v5.py (Phase 2-3),
                                 materialize_picks.py, fidelity_new.py, results/ (all runs), REPORT.md (study log)
    paper_tables/                paper_fill.py (Table 3 / 6 rows), paper_rows_cached.tex (the rows as pasted in the paper)
figures/         gen_paper_figures.py (Figures 1-2), plot_loss_curve.py (Figure 3), the static PDFs used in the paper
reproduce/       one script per experiment family + tables.sh; start here
references.bib
```

The seventeen problems (Table 1): PP cross-stage, TP MLP, FSDP weight prefetch, Layer-block AR, FSDP all-gather lowering,
Layer grads AR, Mixed-dtype grads AR, three bucket-count sets (hs_a 64x64K-512K, hs_c 32x256K-2M, 48x64-512K), Model grads AR,
24- and 48-piece probes (Llama side); AllToAllV, Uniform AllToAll, Ring KV, Distributed CE (MoE side). Names used by the
scripts: `pp_send_recv tp_mlp fsdp_prefetch llama_block_ar` / `fsdp_ag_layout layer_grads_hetero_ar layer_grads_mixed_dtype
hs_a_64x64k_512k hs_c_32x256k_2m probe_het_48_small model_grads_4layers probe_het_24_mid probe_het_48_mid` /
`alltoallv uniform_a2a ring_kv dxe`.

---

## Setting up a cluster

The 224-rank numbers need 7 x AWS `trn1.32xlarge` in one VPC with EFA (8 adapters per node); the 3- and 5-node rows use
the same scripts with fewer workers, the 1-node columns run on a single instance.

```bash
# 1. Launch the instances. The security group must allow all traffic within the group INCLUDING a self-referencing
#    egress rule on all protocols (without it EFA reports peer=self and no cross-node collective runs).
# 2. On every node: Neuron SDK (aws-neuronx-tools, -collectives, -runtime-lib, -dkms) and a venv with
#    torch-neuronx, torch-xla, neuronx-distributed (the DLAMI ships one at /opt/aws_neuronx_venv_pytorch_2_8).
#    The paper used neuronx-cc 2.23 and torch-neuronx 2.8.
# 3. On the master: clone this repo to /home/ubuntu/OverlayCCL, then
cd /home/ubuntu/OverlayCCL && bash reproduce/setup_layout.sh master
cp reproduce/env.example /home/ubuntu/venv.env    # edit VENV and WORKERS (worker private IPs)
export SSH_KEY=~/.ssh/<cluster-key>.pem            # every launcher ssh/scp/rsyncs to the workers with this key
export ANTHROPIC_API_KEY=...                       # LLM proposer; search/_anthropic_route.py also supports AWS Bedrock
```

The N-node launcher `experiments/cluster_224rank/scripts/run_ap7.sh` rsyncs the repo to every worker, kills stale Neuron
processes, waits for the devices and rendezvous ports to free up, and starts one `torchrun` per node. Two operational rules
matter: never interrupt a run while it compiles without running `experiments/model_extension/node_clean.sh` on every node
afterwards (stale compile locks hang the next run), and always warm the compile cache (or use `COLD=1` deliberately) before
reading a step time.

On the analysis host (CPU only; simulator scoring, Phase 3 for the gradient problems, table generation):

```bash
bash reproduce/setup_layout.sh            # symlinks the fixed paths the scripts expect to this clone
python3 -m venv ~/work/ncc/venv && source ~/work/ncc/venv/bin/activate && pip install torch numpy anthropic matplotlib
```

---

## Running the search

```bash
bash reproduce/search_all.sh original        # eight primitives, 7 nodes, LLM = Claude Sonnet 4.5 (~56 min, ~$19 of tokens)
bash reproduce/search_all.sh gradsync        # nine gradient-sync / prefetch problems (analysis host; simulator at training scope)
bash reproduce/search_all.sh gradsync-nosim  # the no-simulator arm of the same nine
```

For a single primitive: `python experiments/run_search.py --problem alltoallv --hw-eval --num-nodes 7 --master-addr
$MASTER_ADDR --worker-addrs $WORKER_ADDRS --phase3-style strategy-enumerate --llm-model sonnet`. `--no-llm` substitutes a
heuristic mutator (1-node smoke test, ~5 min), `--phase3-style {cc-react,multi-island}` selects the alternative Phase-3
loops of Table 8, `--no-simulator` hides the cost model from Phase 3 (Table 9). The loop writes
`runtime/trainium_<problem>[_7node].py`. For the gradient problems the Phase-3 output is materialized into the hardware
harness with `materialize_picks.py`, gated on the cluster with `reproduce/perproblem_gradsync.sh`, and
`simdecides_table.py` prints the shipped pick (lowest simulator score among the correctness/TV-gate survivors) per problem.

---

## Running the models end to end

```bash
# Llama-7B shell (Table 4; 7 nodes, S=1024, M=4, seed 0). Arms: baseline | overlayccl | control | nosim
bash reproduce/e2e_llama7b.sh baseline
bash reproduce/e2e_llama7b.sh overlayccl
bash reproduce/e2e_llama7b.sh overlayccl 2048 4 0           # Table 5 knob rows: S=2048,M=4 and S=1024,M=8
WORKERS_OVERRIDE="w1 w2 w3 w4" bash reproduce/e2e_llama7b.sh overlayccl   # Table 5 lower block: 5 nodes; two workers = 3 nodes
for s in 0 1 2 3 4; do bash reproduce/e2e_llama7b.sh overlayccl 1024 4 $s; done   # Table 6 Llama-7B row
COLD=1 bash reproduce/e2e_llama7b.sh overlayccl              # wall clock from a cold compile cache (no break-even point)

# OLMoE-10B (Table 4; 2500 real-OpenWebText steps)
bash reproduce/e2e_olmoe10b.sh baseline
bash reproduce/e2e_olmoe10b.sh overlayccl
```

The Llama-7B shell carries the four original sites (PP send/recv, TP MLP all-reduce, FSDP weight prefetch, vocabulary-sharded
loss), the FSDP all-gather lowering problem at the prefetch site and five data-parallel gradient-synchronization sites whose
parameter groups are the per-problem sets. The baseline synchronizes every microbatch with the microbench-winning bundles and
the pinned gather; OverlayCCL bundles the four original sites across microbatches, gathers with the native dim-0 lowering and
synchronizes once per step per tensor. Both arms report the same per-microbatch-normalized loss (3.466).

---

## Reproducing each table and figure

Numbers as of the submitted draft (Tables 1-10 and Figures 1-2 are in the main paper, S-numbered items in the supplementary PDF); `bash reproduce/tables.sh` regenerates every numeric table from the measurements in the
repository without a cluster. The cluster-side commands below produce the raw JSON those tables read.

| Paper artifact | Data in the repo | Reproduction |
|---|---|---|
| **Table 1** problems | descriptive | `search/problems.py`, `search/problems_newprobs.py`, `new_problems/g_newprobs_sy7.py` |
| **Table 2** evaluation harnesses | descriptive | `training/train_olmoe10b.py`, `experiments/model_extension/train_llama_e2e_amp7.py`, `training/train_<primitive>_7node.py` |
| **Table 3** per-problem latency at three scopes | upper block: `cluster_224rank/results/{sweep_v2,olmoe_results,llama7b_results}`, `training/results`; lower block: `new_problems/results/sy7_train*` (microbench), `sy7_v10_1n`, `sy7_v11ag_1n` (1-node), `sy7_v10*`, `sy7_v11ag*` (224 ranks) | upper: `reproduce/perproblem_primitives.sh bench1n` then `train7n`; lower: `reproduce/perproblem_gradsync.sh microbench`, `train1n`, `train7n`; rows: `paper_tables/paper_fill.py` |
| **Table 4** end-to-end speedups | `new_problems/results/amp7_json` (Llama), `cluster_224rank/results/olmoe_results` (OLMoE) | `reproduce/e2e_llama7b.sh {baseline,overlayccl}`, `reproduce/e2e_olmoe10b.sh {baseline,overlayccl}`; rows: `reproduce/e2e_tables.py` |
| **Table 5** configuration-knob sweep (upper) and cluster-size generalization (lower) | same JSON dirs (ws224 / ws160 / ws96); OLMoE knob and 3-node runs in `olmoe_results`, `llama7b_results` | Llama: `e2e_llama7b.sh <arm> 2048 4`, `<arm> 1024 8`, `WORKERS_OVERRIDE=... e2e_llama7b.sh`; OLMoE: `training/olmoe_sweep_{s128,d1024_s512,l4}.py`, `training/train_olmoe10b_3node.py` |
| **Table 6** multi-seed retest | primitives: `training/results`, `cluster_224rank/results`; gradient problems: `sy7_v10{,_r2..r5}`, `sy7_v11ag{,_r2..r5}`; Llama shell: `amp7_json` seeds 0-4 | `perproblem_primitives.sh train7n <seed>` x5; `perproblem_gradsync.sh train7n <round>` x5; `e2e_llama7b.sh overlayccl 1024 4 <seed>` x5; cells: `paper_fill.py`, `e2e_tables.py` |
| **Table 7** simulator fidelity | upper: `cluster_224rank/results/sweep_v2` (+ `sweep_a2av*`, `sweep_ua2a_fix*`, `sweep_found`) and `sweep_v2/sim_v4_v16b.json`; lower: `new_problems/results/sim_fid_arms/sim_newprobs_v5_fid_calib.json` vs `sy7_v10*` | `cluster_224rank/scripts/score_v4.py` + `scripts/fidelity_table.py`; `new_problems/run_fid_arms.sh` + `fidelity_new.py` |
| **Table 8** search cost per Phase-3 style | `cluster_224rank/results/queue.log`, `new_problems/results/phase3/*.log` | `run_search.py --phase3-style {strategy-enumerate,cc-react,multi-island}` prints wall and token cost at the end of each run |
| **Table 9** no-simulator ablation | `amp7_json` (`bundled_nosim`), `olmoe_results` | `search_all.sh gradsync-nosim`, `materialize_nosim.py`, then `e2e_llama7b.sh nosim`; OLMoE: `run_search.py --no-simulator` then `e2e_olmoe10b.sh` |
| **Table S9** non-LLM searches (appendix) | `cluster_224rank/results/gs_b400`, `generic_search_default.json`, `grammar_search.json`, `rewriter_search.json`; `new_problems/results/nonllm` | `cluster_224rank/scripts/generic_search.py` (`BUDGET`, `SEEDS`), `grammar_search.py`, `rewriter_search.py`; `new_problems/run_nonllm.sh` |
| **Table 10** constant-swap arms | `cluster_224rank/results/sensitivity_arms`, `sweep_v2/sim_v4_*.json`; `new_problems/results/sim_fid_arms/` | `cluster_224rank/scripts/run_sensitivity_arms.sh` + `fidelity_table.py --arms`; `new_problems/run_fid_arms.sh` + `fidelity_new.py` |
| **Table S1** / **Figure S1** OLMoE loss match | `cluster_224rank/results/olmoe_results` (per-step loss) | `figures/plot_loss_curve.py` |
| **Table S2** probe tools, **Table S3** cost calculus | descriptive | tool implementations in `search/profiling.py` and `cluster_224rank/scripts/train_hwprobe_7node.py`; measured probe table `cluster_224rank/results/hwprobe/hw_probe_measured.json` |
| **Tables S4-S8** appendix ablations (e2e, Llama sweep, per-problem, no-sim per-problem, payload) | `cluster_224rank/results`, `new_problems/results/{sy7_v10_nosim,loop_simdecides}` | same scripts as Tables 4-5, 9-10; `new_problems/simdecides_table.py` |
| **Figure 1** workflow | drawn | `figures/gen_paper_figures.py` (`fig_workflow`) |
| **Figure 2** cross-scope inversion bars | Table 3 upper-block numbers | `figures/gen_paper_figures.py` (`fig_disagreement`) |
| **Figures S2-S3** Llama-block descent and M-sweep | `experiments/model_extension/results_amp*_7node`, `cluster_224rank/results/llama7b_results` | static PDFs in `figures/` from the block-harness runs (`experiments/model_extension/train_llama_e2e_amp3.py`) |
| **Figures S4-S13** code listings, prompt scaffolds, phase snippets | `runtime/trainium_alltoallv*.py`, `prompts/`, `search/` | listings of the files named in each caption |

Operational notes that affect the numbers: Llama-7B step times are steady medians over steps 100-300 of a 300-step run,
OLMoE step times steady means; the 224-rank seed-0 Llama pair was measured twice (warm cache 600.1 / 79.1 ms, cold-cache
repeat 595.4 / 72.3 ms) and the JSON directory keeps the second run (same tag), the first survives in
`new_problems/results/chain_amp7.out`; the per-call probe (`PERCALL_PROBE=1`) must be off for shell step times.

---

## Requirements

- 1-7 x AWS `trn1.32xlarge`; Neuron SDK 2.2x (neuronx-cc 2.23), PyTorch/XLA 2.8, NeuronX-Distributed 0.18+.
- Anthropic API or AWS Bedrock access for the LLM proposer (Claude Sonnet 4.5 in the paper); `--no-llm` for heuristic-only runs.
- Analysis host: Python 3.10+, torch (CPU), numpy, matplotlib.
