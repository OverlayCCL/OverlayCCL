# Cross-scope probe of further strategy classes (2-node cluster, 2026-10-26)

Companion to the paper section "Beyond the three mechanisms: a cross-scope probe of other strategy classes".
Hand-written candidates in five classes the search did not reach, each measured at two scopes on a 2-node
trn1.32xlarge cluster (64 ranks; same AMI / Neuron SDK as the 7-node runs): the 1-node microbench at the
per-problem harness shapes (32 ranks) and the per-problem training harness `training/train_<p>_7node.py` at 64 ranks
(300 steps, warm compile cache, median of the steady window).

## Layout

| path | content |
|---|---|
| `scripts/gen_cands.py` | writes `cands/*.py` (algebraic recomposition, hierarchical, chunking, pre/post-processing) + manifests |
| `cands/` | candidate sources; `*_pl0` = same strategy with `pin_layout=False` (native all-gather); `alltoallv__*_runtime.py` = whole runtime files for the AllToAllV helper |
| `scripts/score_scopes.py`, `scripts/run_sim.sh`, `sim/` | the paper's calibrated simulator scored at bench (32 ranks) and training (64 / 224 ranks) scope |
| `scripts/g_bench.py`, `scripts/g_bench.sh` | 1-node bench of one candidate at harness shapes (initializes the xla process group; asserts the world size) |
| `scripts/g_check.py`, `scripts/g_check.sh` | on-device correctness of every candidate against the production baseline |
| `scripts/g_sweep.sh` | 64-rank training-scope sweep via `deploy_candidate.py` + `run_ap7.sh` (from `experiments/cluster_224rank`) |
| `scripts/g_groups_test.py` | probe: grouped (intra/inter-node) collectives at 64 ranks, `pin_layout` True/False |
| `scripts/collect_sweep.py`, `scripts/analyze_scopes.py`, `scripts/g_report.py` | tables; `results/report_table.md` is `g_report.py results sim` |
| `results/bench_1node/` | 32-rank bench JSON per candidate (`med_ms`); `check_*.json` = correctness |
| `results/sweep_2node*/` | 64-rank step time per candidate (`measured_ms`); `_rep2/_rep3`, `_a2av_rep*`, `_hier`, `_pl0` = repeat / follow-up rounds |
| `results/olmoe_2node_s256/`, `results/olmoe_2node_s128/` | OLMoE end-to-end at 64 ranks (`training/train_olmoe10b_3node.py`, shapes derived from the world size), one run per AllToAllV arm |
| `results/groups_probe_node0.log` | grouped-collective probe output |

## Reproduce (on the master of a 2-node cluster bootstrapped with `experiments/cluster_224rank/scripts/bootstrap_ap7.sh`)

```bash
G=/home/ubuntu/g2; mkdir -p $G && cp -r cands $G/cands_g && cp cands/manifest_*.json scripts/g_*.{py,sh} $G/
bash $G/g_check.sh                      # correctness, one torchrun per problem
bash $G/g_bench.sh                      # 1-node bench, every manifest entry
bash $G/g_sweep.sh                      # 64-rank training-scope sweep (needs the PP harness patch: scripts/patch_pp.py)
SEQLEN=128 bash $G/g_olmoe.sh           # OLMoE end-to-end, four AllToAllV arms
python3 scripts/g_report.py results sim # table
```

Notes: `xr.world_size()` is 1 until `dist.init_process_group("xla")` runs (the bench asserts this); hierarchical
candidates derive the node count from the world size because the runtime wrappers pass `num_nodes=7`; the harness's
rank 0 may land on a worker under the c10d rendezvous, so result files are harvested from every node.
