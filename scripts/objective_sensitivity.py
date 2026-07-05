#!/usr/bin/env python3
"""
Objective-weight sensitivity analysis.

The pre-registered sweep varies only the energy/resource-cost weight w_E.
This script perturbs the two highest-leverage
weights -- SLA-compliance (alpha_1) and latency (alpha_2) -- by +/-50% of
their registered values, renormalizes the remaining weights to keep the
composite a convex combination, and recomputes the per-regime winner across
the full 5x4 sweep for each perturbed weighting.

It reports, for each perturbation: (a) how many of the 20 regimes Static
still wins, and (b) whether HAO-PRA-U wins any regime. This is a robustness
check, NOT a re-registration: it is reported as post-hoc sensitivity.

USAGE
-----
  python3 objective_sensitivity.py results/regime_sweep/all_cells.csv
"""
import sys
import csv
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
OBJ_W_BASE = dict(sla=0.2256, latency=0.1880, throughput=0.1692,
                  acceptance=0.1316, utilization=0.1128, reliability=0.0658,
                  isolation=0.0282, energy=0.0188, fragmentation=0.0390,
                  overhead=0.0210)
PROPOSED = "HAO-PRA-U"


def perturbed_base(scale_sla=1.0, scale_lat=1.0):
    """Return an objective-weight dict with sla/latency scaled, renormalized
    over the non-energy weights so they sum as before (energy handled per
    cell)."""
    w = dict(OBJ_W_BASE)
    w["sla"] = OBJ_W_BASE["sla"] * scale_sla
    w["latency"] = OBJ_W_BASE["latency"] * scale_lat
    non_energy = {k: v for k, v in w.items() if k != "energy"}
    target = sum(v for k, v in OBJ_W_BASE.items() if k != "energy")
    s = target / sum(non_energy.values())
    for k in non_energy:
        w[k] = non_energy[k] * s
    return w


def cell_weights(base, w_E):
    others = {k: v for k, v in base.items() if k != "energy"}
    s = (1.0 - w_E) / sum(others.values())
    w = {k: v * s for k, v in others.items()}
    w["energy"] = w_E
    return w


def per_seed_composites(rows, W):
    seeds = sorted({r["seed"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    out = {m: [] for m in methods}
    for sd in seeds:
        sub = {r["method"]: dict(r) for r in rows if r["seed"] == sd}
        acc = {m: 0.0 for m in methods}
        for metric in BENEFIT + COST:
            vals = np.array([float(sub[m][metric]) for m in methods])
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo + 1e-12)
            if metric in COST:
                norm = 1.0 - norm
            for m, v in zip(methods, norm):
                acc[m] += W[KEYMAP[metric]] * float(v)
        for m in methods:
            out[m].append(acc[m])
    return out


def evaluate(path, base):
    cells = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            cells[(r["w_E"], r["headroom"])].append(r)
    static_wins = 0
    hao_wins = 0
    decisive = 0
    for (wE, h), rows in cells.items():
        W = cell_weights(base, float(wE))
        comp = per_seed_composites(rows, W)
        means = {m: float(np.mean(v)) for m, v in comp.items()}
        order = sorted(means, key=lambda m: -means[m])
        winner, runner = order[0], order[1]
        _, p = stats.ttest_ind(comp[winner], comp[runner], equal_var=False)
        if p < 0.05:
            decisive += 1
        if winner == "Static Allocation":
            static_wins += 1
        if winner == PROPOSED:
            hao_wins += 1
    return static_wins, hao_wins, decisive, len(cells)


def main(path):
    grid = [("registered", 1.0, 1.0),
            ("SLA -50%", 0.5, 1.0), ("SLA +50%", 1.5, 1.0),
            ("latency -50%", 1.0, 0.5), ("latency +50%", 1.0, 1.5),
            ("SLA+50%,lat-50%", 1.5, 0.5), ("SLA-50%,lat+50%", 0.5, 1.5)]
    print(f"{'perturbation':<20}{'Static wins':>12}{'HAO wins':>10}"
          f"{'decisive':>10}{'/regimes':>10}")
    print("-" * 62)
    for name, ss, sl in grid:
        sw, hw, dec, n = evaluate(path, perturbed_base(ss, sl))
        print(f"{name:<20}{sw:>12}{hw:>10}{dec:>10}{n:>10}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results/regime_sweep/all_cells.csv")
