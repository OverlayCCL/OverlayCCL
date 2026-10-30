#!/usr/bin/env python3
"""Convergence table from the training-scope gate outputs (/tmp/g2/hw/gate/*.json)."""
import json, glob, os
rows = []
for f in sorted(x for x in glob.glob("/tmp/g2/hw/gate/*.json") if not x.endswith(".v11.json")):
    d = json.load(open(f)); m = d["measured_ms"]; base = next((c for c in m if c.startswith("per_tensor")), None)
    gate = d["gate_choice"]; hand = {c: v for c, v in m.items() if not c.startswith("gate_") and c != base}
    best_hand = min(hand, key=hand.get) if hand else None
    r_gate = m[base] / m[gate] if gate and base else float("nan"); r_hand = m[base] / hand[best_hand] if best_hand else float("nan")
    verdict = "at optimum" if r_gate >= 0.98 * max(r_hand, 1.0) else ("win" if r_gate > 1.03 else ("baseline kept" if gate == base else "loss"))
    rows.append((d["hw_problem"] + (" (dtype)" if d.get("independent_pieces") is False and "_dtype" in f else ""), m[base], gate, m[gate], best_hand, hand.get(best_hand), r_gate, r_hand, verdict))
print(f"{'problem':30s} {'per-tensor':>10s} {'gate choice':>22s} {'ms':>7s} {'gate x':>7s} {'best hand-written':>20s} {'ms':>7s} {'hand x':>7s}  verdict")
for p, b, g, gm, h, hm, rg, rh, v in rows:
    print(f"{p:30s} {b:10.1f} {g:>22s} {gm:7.1f} {rg:7.2f} {str(h):>20s} {hm if hm else float('nan'):7.1f} {rh:7.2f}  {v}")
