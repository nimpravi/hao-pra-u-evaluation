#!/usr/bin/env python3
"""
Regime-shift study analysis — implements SHIFT_STUDY_PREREGISTRATION.md
decision rules VERBATIM. Do not edit after observing results.

Per scenario instance:
  * per-seed composite over the POST-shift window only (Phase B min-max
    normalization across the six schemes within each seed, anchor weights
    w_E = 0.0188).
  * "wins an instance" = best post-shift mean composite AND
    Welch p < 0.05 vs runner-up.
GATE 2 PASS = HAO-PRA-U wins >= 2 of the 4 instances.
Secondary (reported regardless): pre->post degradation per scheme.

Usage: python3 analyze_shift.py [shift_results/all_windows.csv]
"""
import csv, json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

BENEFIT = ["throughput", "utilization", "acceptance", "reliability",
           "isolation", "energy_eff"]
COST = ["sla_violation", "latency", "fragmentation", "overhead"]
KEYMAP = dict(sla_violation="sla", latency="latency",
              throughput="throughput", acceptance="acceptance",
              utilization="utilization", reliability="reliability",
              isolation="isolation", energy_eff="energy",
              fragmentation="fragmentation", overhead="overhead")
ANCHOR_W = dict(sla=0.2256, latency=0.1880, throughput=0.1692,
                acceptance=0.1316, utilization=0.1128, reliability=0.0658,
                isolation=0.0282, energy=0.0188, fragmentation=0.0390,
                overhead=0.0210)
PROPOSED = "HAO-PRA-U"
INSTANCES = ["S1a_load15", "S1b_load30", "S2_mixshift", "S3_bursts"]


def composites(rows):
    """rows: post-window rows of one scenario -> {method: [comp per seed]}"""
    seeds = sorted({r["seed"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    out = {m: [] for m in methods}
    for sd in seeds:
        sub = {r["method"]: dict(r) for r in rows if r["seed"] == sd}
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
        "shift_results/all_windows.csv"
    rows = list(csv.DictReader(open(path)))
    report, wins = {}, []
    for sc in INSTANCES:
        post = [r for r in rows if r["scenario"] == sc
                and r["window"] == "post"]
        pre = [r for r in rows if r["scenario"] == sc
               and r["window"] == "pre"]
        if not post:
            print(f"missing scenario {sc}; aborting")
            return
        comp = composites(post)
        means = {m: float(np.mean(v)) for m, v in comp.items()}
        ranked = sorted(means, key=means.get, reverse=True)
        winner, runner = ranked[0], ranked[1]
        t, p = stats.ttest_ind(comp[winner], comp[runner],
                               equal_var=False)
        decisive = bool(p < 0.05)
        if winner == PROPOSED and decisive:
            wins.append(sc)
        deg = {}
        for m in means:
            v_pre = np.mean([float(r["sla_violation"]) for r in pre
                             if r["method"] == m])
            v_post = np.mean([float(r["sla_violation"]) for r in post
                              if r["method"] == m])
            l_pre = np.mean([float(r["latency"]) for r in pre
                             if r["method"] == m])
            l_post = np.mean([float(r["latency"]) for r in post
                              if r["method"] == m])
            deg[m] = dict(viol_pre=round(v_pre, 4),
                          viol_post=round(v_post, 4),
                          delta_viol=round(v_post - v_pre, 4),
                          delta_latency=round(l_post - l_pre, 3))
        report[sc] = dict(post_composite_means={m: round(v, 4)
                                                for m, v in means.items()},
                          winner=winner, runner_up=runner,
                          p_vs_runner=round(float(p), 5),
                          decisive=decisive, degradation=deg)

    gate2 = len(wins) >= 2
    out = dict(summary=dict(hao_instance_wins=wins, n_wins=len(wins),
                            GATE2_PASS=bool(gate2)), instances=report)
    with open("shift_results/decision_report_shift.json", "w") as f:
        json.dump(out, f, indent=1)

    print("Post-shift winners:")
    for sc in INSTANCES:
        r = report[sc]
        print(f"  {sc:<14} {r['winner']:<26} "
              f"(p={r['p_vs_runner']}, decisive={r['decisive']})")
    print(f"\nHAO-PRA-U instance wins: {len(wins)} of 4  ->  GATE 2: "
          f"{'PASS - architecture paper proceeds with adaptation evidence' if gate2 else 'FAIL - negative-result paper or claims-free architecture paper'}")
    print("Full details: shift_results/decision_report_shift.json")


if __name__ == "__main__":
    main()
