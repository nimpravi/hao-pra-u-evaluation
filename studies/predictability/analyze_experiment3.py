#!/usr/bin/env python3
"""
analyze_experiment3.py -- implements EXPERIMENT3_PREREGISTRATION.md Section 5
decision rule VERBATIM. Do not edit after observing results.

Per predictability level p:
  1. per-seed composite (min-max normalized across the 6 schemes within
     each seed, anchor weights w_E=0.0188) -- identical to
     hao_pra_u_sim.composite_scores / analyze_shift.composites.
  2. rank schemes by mean composite over the 12 seeds; winner, runner-up.
  3. decisive win iff Welch two-sided p < 0.05 (winner vs runner-up).
  4. "HAO-PRA-U wins level p" iff HAO-PRA-U is winner AND decisive.

Primary endpoint: the set L_win of levels where HAO-PRA-U wins.
  H1 supported  iff L_win is a nonempty CONTIGUOUS up-set (all p >= p*);
                report smallest such p* as the threshold.
  H0 supported  iff L_win is empty.
  Isolated/non-monotone wins -> reported as non-threshold-consistent.

Tripwire (Section 3): p=0.0 must NOT be a HAO-PRA-U win. If it is, the
result is flagged as a probable harness/regression error, not a finding.

Usage: python3 analyze_experiment3.py [exp3_results/results_master.csv]
"""
import csv
import json
import sys
from collections import defaultdict
import numpy as np
from scipy import stats

BENEFIT = ["throughput", "utilization", "acceptance", "reliability",
           "isolation", "energy_eff"]
COST = ["sla_violation", "latency", "fragmentation", "overhead"]
KEYMAP = dict(sla_violation="sla", latency="latency", throughput="throughput",
              acceptance="acceptance", utilization="utilization",
              reliability="reliability", isolation="isolation",
              energy_eff="energy", fragmentation="fragmentation",
              overhead="overhead")
ANCHOR_W = dict(sla=0.2256, latency=0.1880, throughput=0.1692,
                acceptance=0.1316, utilization=0.1128, reliability=0.0658,
                isolation=0.0282, energy=0.0188, fragmentation=0.0390,
                overhead=0.0210)
PROPOSED = "HAO-PRA-U"


def composites(rows):
    """rows: KPI rows of one level -> {method: [composite per seed]}."""
    seeds = sorted({r["seed"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    out = {m: [] for m in methods}
    for sd in seeds:
        sub = {r["method"]: r for r in rows if r["seed"] == sd}
        comp = {m: 0.0 for m in methods}
        for metric in BENEFIT + COST:
            vals = np.array([float(sub[m][metric]) for m in methods])
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo + 1e-12)
            if metric in COST:
                norm = 1.0 - norm
            for m, v in zip(methods, norm):
                comp[m] += ANCHOR_W[KEYMAP[metric]] * float(v)
        for m in methods:
            out[m].append(comp[m])
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "exp3_results/results_master.csv"
    rows = list(csv.DictReader(open(path)))
    levels = sorted({float(r["predictability"]) for r in rows})

    report = {}
    win_levels = []
    for p in levels:
        lr = [r for r in rows if float(r["predictability"]) == p]
        comp = composites(lr)
        means = {m: float(np.mean(v)) for m, v in comp.items()}
        ranked = sorted(means, key=means.get, reverse=True)
        winner, runner = ranked[0], ranked[1]
        t, pv = stats.ttest_ind(comp[winner], comp[runner], equal_var=False)
        decisive = bool(pv < 0.05)
        hao_win = bool(winner == PROPOSED and decisive)
        if hao_win:
            win_levels.append(p)
        report[p] = dict(
            mean_composite={m: round(v, 4) for m, v in means.items()},
            winner=winner, runner_up=runner, p_vs_runner=round(float(pv), 5),
            decisive=decisive, hao_pra_u_wins=hao_win)

    # ---- primary endpoint: contiguous up-set test (Section 5) ----
    win_set = sorted(win_levels)
    upset = bool(win_set) and win_set == [p for p in levels
                                          if p >= win_set[0]]
    p_star = win_set[0] if upset else None

    if not win_set:
        verdict = ("H0 supported: HAO-PRA-U does not achieve a decisive "
                   "regime win at any predictability level. The negative "
                   "result is robust to the generality objection.")
    elif upset:
        verdict = (f"H1 supported: contiguous up-set; threshold p* = "
                   f"{p_star}. HAO-PRA-U wins for all p >= {p_star}. The "
                   "original negative result is conditional on "
                   "unpredictability.")
    else:
        verdict = ("Non-monotone / isolated wins: NOT threshold-consistent. "
                   f"Win levels {win_set} do not form an up-set. Reported as "
                   "non-threshold-consistent per Section 5; leans H0.")

    # ---- tripwire (Section 3) ----
    tripwire = ""
    if 0.0 in report and report[0.0]["hao_pra_u_wins"]:
        tripwire = ("TRIPWIRE FIRED: HAO-PRA-U 'wins' at p=0.0, which "
                    "contradicts Experiments 1-2 at the memoryless limit. "
                    "Treat as a probable harness/regression error and "
                    "investigate BEFORE making any claim.")

    out = dict(summary=dict(win_levels=win_set, is_contiguous_upset=upset,
                            threshold_p_star=p_star, verdict=verdict,
                            tripwire=tripwire),
               levels=report)
    with open("exp3_results/decision_report_exp3.json", "w") as f:
        json.dump(out, f, indent=1)

    print("Predictability-sweep winners:")
    for p in levels:
        r = report[p]
        flag = "  <-- HAO-PRA-U WIN" if r["hao_pra_u_wins"] else ""
        print(f"  p={p:<4} {r['winner']:<26} (p_vs_runner="
              f"{r['p_vs_runner']}, decisive={r['decisive']}){flag}")
    print()
    if tripwire:
        print(tripwire + "\n")
    print(verdict)
    print("\nFull details: exp3_results/decision_report_exp3.json")


if __name__ == "__main__":
    main()
