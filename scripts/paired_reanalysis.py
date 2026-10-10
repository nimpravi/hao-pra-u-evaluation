#!/usr/bin/env python3
"""
Step 2: Paired-statistics recomputation of the regime sweep.

All schemes are evaluated on the same seeds and traffic realizations, so the
samples are paired; the registered analysis used Welch's t-test, which assumes
independent samples. This script recomputes the comparison with paired tests.

This script:
  1. Reproduces the registered Welch analysis (sanity check vs the paper).
  2. Recomputes the same decisions with paired tests:
       - paired t-test (parametric)
       - Wilcoxon signed-rank (non-parametric)
  3. Reports whether the headline "18 of 20 decisive" survives.

Composite construction is copied verbatim from studies/regime_sweep/
analyze_sweep.py so the only thing that changes is the significance test.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

PROPOSED = "HAO-PRA-U"

OBJ_W_BASE = dict(sla=0.2256, latency=0.1880, throughput=0.1692,
                  acceptance=0.1316, utilization=0.1128, reliability=0.0658,
                  isolation=0.0282, energy=0.0188, fragmentation=0.0390,
                  overhead=0.0210)

BENEFIT = ["throughput", "utilization", "acceptance", "reliability",
           "isolation", "energy_eff"]
COST = ["sla_violation", "latency", "fragmentation", "overhead"]

KEYMAP = dict(sla_violation="sla", latency="latency", throughput="throughput",
              utilization="utilization", reliability="reliability",
              isolation="isolation", energy_eff="energy",
              fragmentation="fragmentation", overhead="overhead",
              acceptance="acceptance")


def cell_weights(w_E):
    others = {k: v for k, v in OBJ_W_BASE.items() if k != "energy"}
    s = (1.0 - w_E) / sum(others.values())
    w = {k: v * s for k, v in others.items()}
    w["energy"] = w_E
    return w


def per_seed_composites(rows, w_E):
    """Verbatim from analyze_sweep.py. Returns {method: [composite per seed]}
    with seed order preserved, so columns are aligned for paired tests."""
    W = cell_weights(w_E)
    seeds = sorted({r["seed"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    out = {m: [] for m in methods}
    for sd in seeds:
        sub = {r["method"]: dict(r) for r in rows if r["seed"] == sd}
        for metric in BENEFIT + COST:
            vals = np.array([float(sub[m][metric]) for m in methods])
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo + 1e-12)
            if metric in COST:
                norm = 1.0 - norm
            for m, v in zip(methods, norm):
                sub[m].setdefault("_comp", 0.0)
                sub[m]["_comp"] += W[KEYMAP[metric]] * float(v)
        for m in methods:
            out[m].append(sub[m].pop("_comp"))
    return out, seeds


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "all_cells.csv"
    rows = list(csv.DictReader(open(path)))
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(float(r["w_E"]), float(r["headroom"]))].append(r)
    cells = sorted(by_cell)

    n_welch = n_paired_t = n_wilcoxon = 0
    hao_wins = 0
    table = []

    for cell in cells:
        wE, h = cell
        comp, seeds = per_seed_composites(by_cell[cell], wE)
        means = {m: float(np.mean(v)) for m, v in comp.items()}
        ranked = sorted(means, key=means.get, reverse=True)
        winner, runner = ranked[0], ranked[1]

        a = np.array(comp[winner])
        b = np.array(comp[runner])

        # registered test (independent)
        _, p_welch = stats.ttest_ind(a, b, equal_var=False)
        # paired parametric
        _, p_pair = stats.ttest_rel(a, b)
        # paired non-parametric
        try:
            _, p_wil = stats.wilcoxon(a, b)
        except ValueError:
            p_wil = float("nan")

        n_welch += p_welch < 0.05
        n_paired_t += p_pair < 0.05
        n_wilcoxon += p_wil < 0.05
        hao_wins += (winner == PROPOSED)

        table.append((wE, h, winner, means[winner], means[runner],
                      p_welch, p_pair, p_wil))

    hdr = (f"{'w_E':>7} {'h':>5} {'winner':<22} {'win':>6} {'run':>6} "
           f"{'p_Welch':>9} {'p_paired':>9} {'p_wilcox':>9}  flags")
    print(hdr)
    print("-" * len(hdr))
    for wE, h, win, mw, mr, pw, pp, pl in table:
        flag = ""
        if (pw < 0.05) != (pp < 0.05):
            flag += " DIFFERS(t)"
        if (pw < 0.05) != (pl < 0.05):
            flag += " DIFFERS(W)"
        print(f"{wE:>7} {h:>5} {win:<22} {mw:>6.3f} {mr:>6.3f} "
              f"{pw:>9.2e} {pp:>9.2e} {pl:>9.2e} {flag}")

    n = len(cells)
    print()
    print(f"Regimes total                     : {n}")
    print(f"Decisive under Welch (registered) : {n_welch}/{n}")
    print(f"Decisive under paired t           : {n_paired_t}/{n}")
    print(f"Decisive under Wilcoxon           : {n_wilcoxon}/{n}")
    print(f"Regimes won by {PROPOSED:<18}: {hao_wins}/{n}")


if __name__ == "__main__":
    main()
