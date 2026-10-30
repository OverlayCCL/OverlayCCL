#!/usr/bin/env python3
"""Append chosen Phase-3 candidates of a search problem as harness candidates. usage: materialize_picks.py <search problem> <hw problem> <tag> <base module> <out module> name_substr:cand_name [...]"""
import json, os, re, ast, sys
sp, hw, tag, base_mod, out_mod = sys.argv[1:6]; picks = [a.split(":") for a in sys.argv[6:]]
recs = json.load(open(os.path.expanduser(f"~/work/g2_backup/phase3_runs/{sp}_{tag}.json")))
base = open(base_mod).read(); block = f"\n# ================================================================ Phase-3 picks for {hw} ({sp}, tag {tag})\n"
for sub, cname in picks:
    code = None
    for r in recs:
        raw = r.get("raw")
        if not raw: continue
        lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if sub.lower() not in str(lst[0]).lower(): continue
        dct = ast.literal_eval(lst[1]) if isinstance(lst[1], str) else lst[1]
        code = dct.get("_params", {}).get("evolved_code") or dct.get("_params", {}).get("builtin_code"); name = str(lst[0]); break
    assert code, (sp, sub)
    fn = f"{cname}_{re.sub('[^a-z0-9]+', '_', name.lower())[:40]}"
    c = re.sub(rf"def evolved_{sp}\([a-z_]+, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1\):", f"def {fn}(xs, rank, ws, xm, torch):", code)
    assert c != code, name
    first_arg = re.search(rf"def evolved_{sp}\(([a-z_]+),", code).group(1)
    c = c.replace("world_size", "ws"); c = f"def {fn}(xs, rank, ws, xm, torch):\n    {first_arg} = xs\n" + c.split("\n", 1)[1]
    block += c.rstrip() + f'\nPROBLEMS["{hw}"]["cands"]["{cname}"] = {fn}   # Phase-3 {tag}: {name}\n'; print(f"{hw}: {cname} = {name}")
mod = base.replace("\ndef flatten(out):", block + "\ndef flatten(out):"); ast.parse(mod); open(out_mod, "w").write(mod); print("wrote", out_mod)
