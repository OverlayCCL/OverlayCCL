# OverlayCCL

Anonymous artifact for the PPoPP '27 submission
**_OverlayCCL: Composing Collectives Above a Vendor Library via a
Closed-Stack Search Loop_**.

The paper (double-blind PDF, 10 pages of main text + references) is
`paper.pdf`. The appendix (loss-match evidence, methodology
details, per-problem ablation tables, deployed strategies,
extended AG+RS case study, cost-model implementation details,
profiling tool details, search-style prompt scaffolds, per-phase
code snippets) is available in two forms:

- **`appendix.pdf`** — typeset supplementary material as a
  standalone PDF (submitted alongside the paper).
- **`SUPPLEMENTARY.md`** — the same appendix content as Markdown
  for easy browsing in the repo.

---

## What OverlayCCL is

A closed-stack search loop that finds faster collective-communication
strategies on top of a closed-source vendor library. On AWS Trainium
the vendor library (Neuron) exposes a fixed API — the loop never
modifies it and never sees inside it. It only calls its primitives
(`xm.all_gather`, `xm.reduce_scatter`, `xm.all_reduce`,
`xm.collective_permute`, `xm.all_to_all`) and reshapes, pads, or
slices tensors around each call. An LLM agent proposes strategies at
this layer; a workload-calibrated cost-model simulator scores them;
the top candidates are hardware-validated before deployment as
drop-in runtime modules.

**Headline results** (paper Table 6):

| Model | Baseline | OverlayCCL | Speedup |
|---|---|---|---|
| OLMoE-10B (224 ranks, 2500 real-OpenWebText steps) | 4404.3 ms/step | 3139.8 ms/step | **1.40×** |
| Llama-7B (224 ranks, 300 random-token steps) | 71.3 ms/step | 22.0 ms/step | **3.24×** |

Both speedups are steady-state per-step, with algorithm-equivalent
descent (real OpenWebText for OLMoE; bit-identical
per-microbatch-normalized loss for Llama-7B).

**Steady-state vs wall-clock.** On short (~300-step) runs the
Llama-7B wall time regresses (2.5→5.9 min) because OverlayCCL's
bundled strategy pays a ~3.7-minute one-time compile overhead;
break-even is ≈4400 steps, well below any real pretraining. Paper §3
walks through this.

---

## Repository layout

```
paper.pdf                — the submitted paper (double-blind, PPoPP '27)
appendix.pdf             — typeset supplementary material
SUPPLEMENTARY.md         — Markdown mirror of appendix.pdf
references.bib           — bibliography

runtime/                 — deployed strategies (the paper's artifact)
                           trainium_<problem>.py       (1-node reference)
                           trainium_<problem>_7node.py (deployed at 224 ranks)

simulator/               — Phase-1-calibrated cost model
search/                  — Phase-3 strategy-enumerate / cc-react /
                           multi-island search loops
codegen/                 — Phase-5 codegen and validation glue
prompts/                 — LLM proposer prompt scaffolds

training/                — end-to-end training harnesses that consume
                           the deployed runtimes (OLMoE and Llama)
experiments/             — 1-node microbenchmarks and per-problem
                           training scripts

figures/                 — SUPPLEMENTARY.md and paper figures
```

The eight collective problems evaluated:

- **MoE-side:** AllToAllV, Uniform AllToAll, Ring KV, Distributed CE.
- **Llama-side:** PP cross-stage, TP MLP AR, FSDP weight prefetch,
  Layer-block AR.

---

## Setting up a cluster

The headline numbers require a 7-node cluster of AWS `trn1.32xlarge`
instances (224 NeuronCores total). Smaller clusters (3 or 5 nodes)
work for the cluster-size ablation.

```bash
# 1. Launch 7×trn1.32xlarge in one VPC with EFA enabled.
#    Security group must allow all traffic within the SG *including
#    self-referential egress on all protocols* (missing this trips
#    EFA peer=self and prevents any cross-node collective).

# 2. On every node, install the Neuron stack.
sudo apt-get install -y aws-neuronx-tools aws-neuronx-collectives \
                       aws-neuronx-runtime-lib aws-neuronx-dkms

# 3. Create the Python venv (DLAMI ami-01807ad0e6484b5a8 ships one
#    pre-built at /opt/aws_neuronx_venv_pytorch_2_8/).
python -m venv /opt/aws_neuronx_venv_pytorch_2_8
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com/ \
    torch-neuronx torch-xla \
    neuronx-distributed neuronx-distributed-training \
    transformers datasets

# 4. Anthropic API key (for the LLM proposer).
export ANTHROPIC_API_KEY=...

# 5. On the master, clone this repo and sync it to workers via rsync/SSH.
git clone <this-repo-URL> OverlayCCL && cd OverlayCCL
```

For multi-node runs, key-based SSH and rsync must be configured
between master and workers. Set `MASTER_ADDR`, `WORKER_ADDRS` (comma-
separated), and `MASTER_PORT` as environment variables — every
reproduction script below reads them from the environment rather
than hard-coding IPs.

---

## Running the search

The agent entry point is `experiments/run_search.py`. It runs Phase 1
(on-device calibration) → Phase 2 (seed pool) → Phase 3 (LLM
proposer) → Phase 4 (hardware validation) → Phase 5 (codegen), and
writes the deployed strategy to
`runtime/trainium_<problem>[_7node].py`.

**Smoke test (1-node, no LLM):**

```bash
python experiments/run_search.py \
  --problem alltoallv --pattern moe \
  --no-llm --phase3-style strategy-enumerate
```

Uses a heuristic mutation oracle instead of the LLM. Writes
`runtime/trainium_alltoallv.py`. About 5 minutes on a single
`trn1.32xlarge`.

**Full 7-node search (LLM, all 8 problems, paper headline setup):**

```bash
for prob in alltoallv uniform_a2a ring_kv dxe \
            pp_send_recv tp_mlp fsdp_prefetch llama_block_ar; do
  python experiments/run_search.py \
    --problem $prob --hw-eval --num-nodes 7 \
    --master-addr $MASTER_ADDR --worker-addrs $WORKER_ADDRS \
    --phase3-style strategy-enumerate --llm-model sonnet
done
```

Roughly 56 wall-minutes across the 8 problems, ~$19 in Sonnet 4.5
tokens.

**Ablations** — replace `--phase3-style strategy-enumerate` with
`cc-react` or `multi-island` for the alternative Phase-3 loops (paper
§4.1); add `--no-simulator` to hide the cost model from Phase 3
(paper §4.2). The no-simulator run collapses the OLMoE headline to
1.00×.

---

## Training with the deployed strategies

The strategies emitted under `runtime/` are drop-in replacements that
the training harnesses select via a `--backend {baseline,agent}` (or
`per_mb` / `bundled` for the Llama harnesses) toggle.

**OLMoE-10B end-to-end (paper Table 6 top row, 1.40× steady):**

```bash
torchrun --nnodes=7 --node_rank=0 --nproc_per_node=32 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  training/train_olmoe10b.py \
  --backend agent --ce agent \
  --steps 2500 --realtok
```

Compare to `--backend baseline --ce baseline` for the 1.40× ratio.
Workers run with matching `--node_rank {1..6}`.

**Llama-7B end-to-end (paper Table 6 bottom row, 3.24× steady):**

```bash
torchrun --nnodes=7 --node_rank=0 --nproc_per_node=32 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  experiments/model_extension/train_llama_e2e_7b.py \
  bundled 300
```

Replace `bundled` with `per_mb` for the developer baseline. Break-
even (compile overhead vs. steady-state saving) is ~4400 steps; the
300-step harness undersells wall-time savings for that reason.

---

## Reproducing each paper table and figure

Every reproduction script reads `MASTER_ADDR`, `WORKER_ADDRS`, and
`MASTER_PORT` from the environment; the pipeline itself is
platform-agnostic once those are set.

| Paper artifact | Reproduction |
|---|---|
| **Table 1** — the eight collective problems | Descriptive; no reproduction needed. |
| **Table 2** — Phase-1 probe tools | Descriptive; the tool implementations live in `simulator/` and `search/profiling.py`. |
| **Table 3** — cost calculus (loop vs. e2e trials) | Numbers are derived; the underlying strategy-enumerate wall/cost lines print at the end of any full 7-node `experiments/run_search.py --phase3-style strategy-enumerate` run over all 8 problems. |
| **Table 4** — evaluation harnesses (setup) | Descriptive; the three harnesses are the training scripts in `training/` (per-problem microbench and OLMoE-10B) and `experiments/model_extension/train_llama_e2e_7b.py` (Llama-7B). |
| **Table 5** — per-primitive latency at three scopes | For 1-node bench columns: run each `experiments/h7_bench/bench_<primitive>.py` (`bench_a2av.py`, `bench_dxe.py`, `bench_pp_send_recv.py`, `bench_tp_mlp.py`, `bench_fsdp_prefetch.py`, `bench_llama_block_ar.py`, `bench_grad_ar.py`, `bench_ring_kv_v6.py`). For 7-node bench and 7-node training columns: run each `training/train_<primitive>_7node.py` with the two backends and read `steady_median_ms` from the emitted JSON. |
| **Table 6** — end-to-end speedups (headline) | See "Training with the deployed strategies" above: `training/train_olmoe10b.py --realtok --steps 2500` (OLMoE row) and `experiments/model_extension/train_llama_e2e_7b.py {per_mb,bundled} 300` (Llama-7B row). |
| **Table 7** — configuration-knob sweep (OLMoE SEQLEN/DM/LAYERS + Llama-block amp1–amp4) | OLMoE knob rows: `training/olmoe_sweep_s128.py` (SEQLEN=128), `training/olmoe_sweep_d1024_s512.py` (SEQLEN=512, DM=1024), `training/olmoe_sweep_l4.py` (LAYERS=4), and the SEQLEN=256 default from `training/train_olmoe10b.py`, each with both `--backend baseline` and `--backend agent`. Llama-block amp rows: `experiments/model_extension/train_llama_e2e_amp{1,2,3,4}.py {per_mb,bundled} 300`. |
| **Table 8** — multi-seed retest | Same as Table 5 (per-primitive) and Table 7 (Llama-block amp3 e2e) but with 5+ seeds; set `PYTHONHASHSEED` per run. Aggregation is a plain per-cell mean±stdev of the emitted `steady_median_ms`. |
| **Table 9** — cluster-size generalization (7n → 5n → 3n) | For OLMoE at 3-nodes: `training/train_olmoe10b_3node.py`. For Llama-7B at 3-nodes: `experiments/model_extension/train_llama_e2e_7b_3node.py`. 5-nodes uses the same scripts with `--num-nodes 5` and the corresponding topology constants noted in the paper. |
| **Table 10** — per-style search cost | Wall-time and Anthropic-cost lines print at the end of each `experiments/run_search.py --phase3-style {strategy-enumerate,cc-react,multi-island}` run. |
| **Table 11** — no-simulator ablation | Same as the full 7-node search with `--no-simulator` added: `experiments/run_search.py --problem <p> --phase3-style strategy-enumerate --no-simulator` over all 8 problems, then run the deployed strategies through the same end-to-end harnesses as Table 6. |
| **Figure 1** — workflow diagram | Static image at `figures/workflow.png`; no reproduction script (a hand-authored diagram, not generated from measurements). |
| **Figure 2** — cross-scope inversion bars | `python figures/gen_paper_figures.py` regenerates `figures/disagreement.pdf`; the bars are sourced from the Table 5 numbers in-tree. |

For per-run measurement details and multi-seed numbers, see
`appendix.pdf` sections B (methodology details) and C
(per-problem ablation tables); the same content is browsable as
`SUPPLEMENTARY.md`.

---

## Requirements

- Cluster: 1–7×AWS `trn1.32xlarge`. Headline numbers require 7 nodes;
  the 3-node ablation and 1-node microbenchmarks work standalone.
- Neuron SDK 2.x, PyTorch/XLA 2.8+, NeuronX-Distributed 0.18+,
  NeuronX-Distributed-Training 1.7+.
- Anthropic API access for the LLM proposer (Claude Sonnet 4.5 in
  the paper). Pass `--no-llm` for heuristic-only runs.

---

## License

See `LICENSE`.
