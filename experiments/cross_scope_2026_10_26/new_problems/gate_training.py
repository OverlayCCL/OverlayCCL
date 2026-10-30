#!/usr/bin/env python3
"""Training-scope hardware gate for the search's finalists.

usage: gate_training.py <search problem> <hardware problem> <phase3 tag> [topk=3] [--run] [--independent]
  1. loads ~/work/g2_backup/phase3_runs/<search problem>_<tag>.json, orders every scored candidate (LLM strategies and
     builtins) by simulated time and takes the top-k with code;
  2. appends them as candidates gate_1..gate_k of PROBLEMS[<hardware problem>] in a copy of the harness module
     (/tmp/g2/g_newprobs_gate_<hw>.py), adapting the search signature to the harness signature;
  3. with --run: ships the module to the master, pre-seeds the output dir with the baseline/hand-written results already
     measured on this cluster, runs bench32 + 64-rank training for the gate candidates (g_newprobs_run2.sh), pulls the
     JSONs and writes /tmp/g2/hw/gate/<hw>.json with the measured step time of every candidate and the gate's choice
     (= fastest measured, with the per-tensor baseline always in the set).
Env: GATE_SEED_DIRS (master dirs to pre-seed from, default 'day4w day4g day4p day4f day4w5'), GATE_TIMEOUT (1500)."""
import json, os, sys, ast, subprocess, re, glob, statistics
sp, hw, tag = sys.argv[1], sys.argv[2], sys.argv[3]
topk = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 3
seen_codes = set()
RUN = "--run" in sys.argv; INDEP = "--independent" in sys.argv; DTYPE = "--dtype" in sys.argv   # --dtype: dtype-honoring runner (v5)
MASTER = "<MASTER_PUBLIC_IP>"; KEY = os.path.expanduser("$SSH_KEY")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", f"ubuntu@{MASTER}"]
SCP = ["scp", "-q", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
recs = json.load(open(os.path.expanduser(f"~/work/g2_backup/phase3_runs/{sp}_{tag}.json")))
for _a in [x.split("=", 1)[1] for x in sys.argv if x.startswith("--extra=")]:   # --extra=<tag>: merge finalists of another Phase-3 run
    recs += json.load(open(os.path.expanduser(f"~/work/g2_backup/phase3_runs/{sp}_{_a}.json")))
cands = []
for r in recs:
    raw = r.get("raw")
    if not raw: continue
    lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
    try: dct = ast.literal_eval(lst[1])
    except Exception: continue
    code = dct.get("_params", {}).get("evolved_code") or dct.get("_params", {}).get("builtin_code")
    t = dct.get("sim_time_us")
    if code and t and code not in seen_codes: seen_codes.add(code); cands.append((float(t), str(lst[0]), code))
cands.sort(key=lambda x: x[0]); picked = cands[:topk]
base = open("/tmp/g2/g_newprobs_v23.py").read()
block = f"\n# ================================================================ training-scope gate candidates for {hw} ({sp}, phase3 {tag})\n"
names = []
for i, (t, nm, code) in enumerate(picked, 1):
    fn = f"gate_{i}_{re.sub('[^a-z0-9]+', '_', nm.lower())[:40]}"
    c = re.sub(rf"def evolved_{sp}\(grads, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1\):", f"def {fn}(grads, rank, ws, xm, torch):", code)
    assert c != code, nm
    c = c.replace("world_size", "ws")
    block += c.rstrip() + "\n" + f'PROBLEMS["{hw}"]["cands"]["gate_{i}"] = {fn}   # sim {t:.0f} us, {nm}\n'; names.append((f"gate_{i}", nm, t))
keep = os.environ.get("GATE_KEEP")   # comma-separated hand-written candidates to keep next to the gate set (default: all)
if keep:
    block += f'PROBLEMS["{hw}"]["cands"] = {{k: v for k, v in PROBLEMS["{hw}"]["cands"].items() if k in {tuple(keep.split(","))!r} or k.startswith("gate_")}}\n'
mod = base.replace("\ndef flatten(out):", block + "\ndef flatten(out):")
ast.parse(mod); out_mod = f"/tmp/g2/g_newprobs_gate_{hw}.py"; open(out_mod, "w").write(mod)
print(f"[gate] {hw}: {len(cands)} scored candidates, gate set:"); [print(f"   {g}: sim {t:.0f} us  {nm}") for g, nm, t in names]
if not RUN: sys.exit(0)
OUT = f"/home/ubuntu/g2/gate_{hw}" + ("_indep" if INDEP else "") + ("_dtype" if DTYPE else ""); seeds = os.environ.get("GATE_SEED_DIRS", "day4w day4g day4p day4f day4w5").split()
subprocess.run(SCP + [out_mod, f"ubuntu@{MASTER}:/home/ubuntu/g2/g_newprobs_gate.py"], check=True)
pre = f"mkdir -p {OUT}/train; " + "; ".join(f"cp -n /home/ubuntu/g2/{d}/train/newprob_train_{hw}__*.json {OUT}/train/ 2>/dev/null" for d in seeds) + f"; rm -f {OUT}/train/*gate_*; cp /home/ubuntu/g2/g_newprobs_gate.py /home/ubuntu/g2/g_newprobs.py"
runner = f"cp /home/ubuntu/g2/g_train_{'v6' if INDEP else ('v5' if DTYPE else 'v4')}.py /home/ubuntu/g2/g_train.py; " + ("export G_TRAIN_INDEPENDENT=1; " if INDEP else "") + f"PROBS='{hw}' RUN_TIMEOUT={os.environ.get('GATE_TIMEOUT', '1500')} NP_OUT={OUT} bash /home/ubuntu/g2/g_newprobs_run2.sh > {OUT}.out 2>&1; echo GATE_DONE >> {OUT}.out"
subprocess.run(SSH + [pre + "; " + runner], check=False)
loc = f"/tmp/g2/hw/gate/{hw}{'_indep' if INDEP else ''}{'_dtype' if DTYPE else ''}"; os.makedirs(f"{loc}/train", exist_ok=True)
subprocess.run(SCP + [f"ubuntu@{MASTER}:{OUT}/train/*.json", f"{loc}/train/"], check=False); subprocess.run(SCP + [f"ubuntu@{MASTER}:{OUT}/run.log", f"{loc}/"], check=False)
res = {}
for f in glob.glob(f"{loc}/train/newprob_train_{hw}__*.json"):
    j = json.load(open(f)); res[j["cand"]] = (j["measured_ms"], j.get("collective_effective"))
base_c = next((c for c in res if c.startswith("per_tensor")), None)
valid = {c: v for c, v in res.items() if v[1] is not False}
choice = min(valid, key=lambda c: valid[c][0]) if valid else None
gate_set = [base_c] + [g for g, _, _ in names]
gate_choice = min((c for c in gate_set if c in valid), key=lambda c: valid[c][0], default=None)
summary = {"hw_problem": hw, "search_problem": sp, "phase3_tag": tag, "gate_candidates": names, "measured_ms": {c: v[0] for c, v in res.items()},
           "effective": {c: v[1] for c, v in res.items()}, "gate_choice": gate_choice, "best_overall": choice, "independent_pieces": INDEP}
json.dump(summary, open(f"{loc}.json", "w"), indent=1)
print(f"[gate] {hw}: measured", {c: round(v[0], 1) for c, v in sorted(res.items(), key=lambda kv: kv[1][0])})
print(f"[gate] choice among baseline+finalists: {gate_choice} ({valid.get(gate_choice, ('?',))[0]} ms); best overall candidate: {choice}")
