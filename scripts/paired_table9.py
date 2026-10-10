#!/usr/bin/env python3
"""
Step 2b: Paired recomputation of the Table IX per-metric comparisons
(HAO-PRA-U vs baseline SLA-violation ratio), plus paired effect sizes.

Registered analysis used Welch + Cohen's d (independent) and Cliff's delta.
Here we add the paired t-test, Wilcoxon signed-rank, and Cohen's d_z
(the paired standardized effect size), which is the correct companion
to a paired test.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

PROPOSED = "HAO-PRA-U"


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cohens_d_indep(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def cohens_dz_paired(a, b):
    """Paired standardized effect size: mean difference / sd of differences."""
    d = np.asarray(a) - np.asarray(b)
    return d.mean() / d.std(ddof=1)


def main():
    path = sys.argv[1]
    rows = list(csv.DictReader(open(path)))
    by_cell = defaultdict(dict)
    for r in rows:
        cell = (float(r["w_E"]), float(r["headroom"]))
        by_cell[cell].setdefault(r["method"], {})[int(r["seed"])] = r

    targets = [
        ((0.0188, 2.0), "Static Allocation",      "Anchor  (h=2.0) vs Static"),
        ((0.0188, 2.0), "Centralized Predictive", "Anchor  (h=2.0) vs Centr.Pred."),
        ((0.0188, 2.0), "Flat-Agent Allocation",  "Anchor  (h=2.0) vs Flat-Agent"),
        ((0.0188, 1.4), "Static Allocation",      "Tight   (h=1.4) vs Static"),
        ((0.0188, 1.4), "Centralized Predictive", "Tight   (h=1.4) vs Centr.Pred."),
    ]

    metric = "sla_violation"
    print(f"Metric: {metric}   (values are HAO-PRA-U - baseline; "
          f"negative favors HAO-PRA-U)\n")
    hdr = (f"{'comparison':<34} {'p_Welch':>9} {'p_paired':>9} "
           f"{'p_wilcox':>9} {'d_indep':>8} {'d_z':>8} {'cliff':>7}")
    print(hdr)
    print("-" * len(hdr))

    for cell, base, label in targets:
        seeds = sorted(set(by_cell[cell][PROPOSED]) & set(by_cell[cell][base]))
        a = np.array([float(by_cell[cell][PROPOSED][s][metric]) for s in seeds])
        b = np.array([float(by_cell[cell][base][s][metric]) for s in seeds])

        _, p_w = stats.ttest_ind(a, b, equal_var=False)
        _, p_p = stats.ttest_rel(a, b)
        try:
            _, p_l = stats.wilcoxon(a, b)
        except ValueError:
            p_l = float("nan")

        print(f"{label:<34} {p_w:>9.2e} {p_p:>9.2e} {p_l:>9.2e} "
              f"{cohens_d_indep(a, b):>8.2f} {cohens_dz_paired(a, b):>8.2f} "
              f"{cliffs_delta(a, b):>7.2f}")

    print(f"\n(n = {len(seeds)} paired seeds per comparison)")
    print("Note: Wilcoxon two-sided floor for n=10 is p = 1.95e-03.")


if __name__ == "__main__":
    main()
