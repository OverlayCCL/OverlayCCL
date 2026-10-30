#!/usr/bin/env python3
"""Append the NO-SIM Phase-3 winner of each (search problem -> hardware problem) as candidate `nosim_1` to a copy of the
sy7 harness module, keeping only per_tensor + the sim-decides winner + nosim_1 so the 224-rank run measures just those."""
import json, os, re, ast, sys
PAIRS = {"layer_grads_mixed_dtype": ("layer_grads_mixed_dtype", "Hierarchical bucketing with dt", ("per_tensor_ar", "gate_1")),
         "dp_grad_bucketing_64": ("hs_a_64x64k_512k", "Two-phase hierarchical bucketi", ("per_tensor_ar", "gate_1")),
         "probe_het_48_small": ("probe_het_48_small", "Single flat concatenation all-", ("per_tensor_ar", "gate_2"))}
base = open("/tmp/g2/g_newprobs_sy7.py").read(); block = "\n# ================================================================ NO-SIM Phase-3 winners (paper ablation arm, 2026-10-29)\n"
for sp, (hw, winner, keep) in PAIRS.items():
    recs = json.load(open(os.path.expanduser(f"~/work/g2_backup/phase3_runs/{sp}_nosim.json"))); code = None
    for r in recs:
        raw = r.get("raw");
        if not raw: continue
        lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if winner.lower()[:20] not in str(lst[0]).lower(): continue
        try: dct = ast.literal_eval(lst[1])
        except Exception: continue
        code = dct.get("_params", {}).get("evolved_code") or dct.get("_params", {}).get("builtin_code"); name = str(lst[0]); break
    assert code, (sp, winner)
    fn = f"nosim_1_{re.sub('[^a-z0-9]+', '_', name.lower())[:40]}"
    c = re.sub(rf"def evolved_{sp}\(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1\):", f"def {fn}(grads, rank, ws, xm, torch):", code)
    assert c != code, name; c = c.replace("world_size", "ws")
    block += c.rstrip() + f'\nPROBLEMS["{hw}"]["cands"]["nosim_1"] = {fn}   # NO-SIM winner: {name}\n'
    block += f'PROBLEMS["{hw}"]["cands"] = {{k: v for k, v in PROBLEMS["{hw}"]["cands"].items() if k in {keep!r} or k == "nosim_1"}}\n'
    print(f"{hw}: nosim_1 = {name}")
mod = base.replace("\ndef flatten(out):", block + "\ndef flatten(out):"); ast.parse(mod); open("/tmp/g2/g_newprobs_nosim.py", "w").write(mod); print("wrote /tmp/g2/g_newprobs_nosim.py")
