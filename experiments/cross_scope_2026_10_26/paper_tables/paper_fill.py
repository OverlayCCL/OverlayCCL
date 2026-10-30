#!/usr/bin/env python3
"""Emit LaTeX rows for the paper's new-problem blocks from the pulled sy7 results.
usage: python3 paper_fill.py <results root with sy7_v10*, sy7_v11ag*, sy7_v10_1n, sy7_v11ag_1n, sy7_train*, sy7_hunt> <amp7 json dir>"""
import sys, glob, json, statistics as st, collections, os
G, AJ = sys.argv[1], sys.argv[2]
MB_WIN = {"layer_grads_hetero_ar": "flat_cat_ar", "model_grads_4layers": "flat_cat_ar", "probe_het_24_mid": "bucket6", "probe_het_48_mid": "bucket6",
          "hs_c_32x256k_2m": "bucket_4mb", "layer_grads_mixed_dtype": "cast_flat_bf16", "hs_a_64x64k_512k": "bucket_4mb",
          "probe_het_48_small": "bucket6", "fsdp_ag_layout": "ag_pinned"}
TOOL = {k: "per_tensor_ar" for k in MB_WIN}; TOOL["fsdp_ag_layout"] = "ag_dim0_t_native"
SHORT = {"fsdp_ag_layout": "FSDP AG lowering", "layer_grads_hetero_ar": "Layer grads (6)", "layer_grads_mixed_dtype": "Mixed-dtype grads", "model_grads_4layers": "Model grads (24)",
         "hs_a_64x64k_512k": "hs\\_a ($64\\times$64K--512K)", "hs_c_32x256k_2m": "hs\\_c ($32\\times$256K--2M)", "probe_het_48_small": "48$\\times$64--512K",
         "probe_het_24_mid": "24$\\times$1--8M", "probe_het_48_mid": "48$\\times$1--8M"}
ORDER = ["fsdp_ag_layout", "layer_grads_hetero_ar", "layer_grads_mixed_dtype", "model_grads_4layers", "hs_a_64x64k_512k", "hs_c_32x256k_2m", "probe_het_48_small", "probe_het_24_mid", "probe_het_48_mid"]
def load(pats):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for pat in pats:
        for f in glob.glob(f"{G}/{pat}"):
            j = json.load(open(f)); out[j["problem"]][j["cand"]].append(j["measured_ms"])
    return out
mir = load(["sy7_v10/train/*.json", "sy7_v10_r[2-9]/train/*.json", "sy7_v11ag/train/*.json", "sy7_v11ag_r[2-9]/train/*.json"])
one = load(["sy7_v10_1n/*.json", "sy7_v11ag_1n/*.json"])
mb = load(["sy7_train/train/*.json", "sy7_train_s[2-9]/train/*.json", "sy7_train_dtype/train/*.json", "sy7_train_s[2-9]_dtype/train/*.json", "sy7_train_48mid/train/*.json", "sy7_hunt/train/*.json"])
def ms(v): return f"{st.mean(v):.1f}" if v else "---"
def pm(v): return f"{st.mean(v):.1f} $\\pm$ {st.stdev(v) if len(v) > 1 else 0:.1f}" if v else "---"
def bold(a, b, fa, fb):
    if not a or not b: return fa, fb
    return (f"\\textbf{{{fa}}}", fb) if st.mean(a) < st.mean(b) else (fa, f"\\textbf{{{fb}}}")
print("% ---- tab:perproblem new block (microbench 224r / 1n training / 7n training, ms/step)")
print("\\midrule\n\\multicolumn{7}{l}{\\textit{Gradient-sync and prefetch problems} --- \\textbf{7n microbench} / \\textbf{1n train} / \\textbf{7n train} (ms/step)} \\\\")
for p in ORDER:
    b, o = MB_WIN[p], TOOL[p]
    cells = []
    for src in (mb, one, mir):
        a, c = src.get(p, {}).get(b), src.get(p, {}).get(o)
        fa, fb = bold(a, c, ms(a), ms(c)); cells += [fa, fb]
    print(f"{SHORT[p]:28s} & " + " & ".join(cells) + " \\\\")
print("\n% ---- tab:multiseed new block")
print("\\midrule\n\\multicolumn{5}{l}{\\textit{Gradient-sync and prefetch problems (224-rank training, ms/step)}} \\\\")
for p in ORDER:
    a, c = mir.get(p, {}).get(MB_WIN[p]), mir.get(p, {}).get(TOOL[p])
    if not a or not c: print(f"% {p}: missing"); continue
    n = min(len(a), len(c)); sp = st.mean(a) / st.mean(c)
    se = ((st.stdev(a)**2 / len(a) if len(a) > 1 else 0) + (st.stdev(c)**2 / len(c) if len(c) > 1 else 0)) ** 0.5
    clean = abs(st.mean(a) - st.mean(c)) > 2 * se
    sps = f"$\\mathbf{{{sp:.2f}\\times}}$" if (sp > 1 and clean) else f"${sp:.2f}\\times$"
    print(f"{SHORT[p]:28s} & {n} & {pm(a)} & {pm(c)} & {sps} \\\\")
print("\n% ---- Llama-7B shell seeds (224 ranks)")
base, tool = [], []
for f in sorted(glob.glob(f"{AJ}/llama_e2e_amp7_*M4_S1024_sites5*ws224_seed*.json")):
    j = json.load(open(f)); t = j["tag"]
    if "per_mb_base_per_mb_microbench" in t: base.append((t, j["steady_median_ms"]))
    elif "bundled_per_tensor" in t and "agdim0_t_native" in t: tool.append((t, j["steady_median_ms"]))
print("% base:", base); print("% tool:", tool)
if base and tool:
    a = [v for _, v in base]; c = [v for _, v in tool]
    print(f"Llama-7B shell (224r) & {min(len(a), len(c))} & {pm(a)} & {pm(c)} & $\\mathbf{{{st.mean(a)/st.mean(c):.2f}\\times}}$ \\\\")
