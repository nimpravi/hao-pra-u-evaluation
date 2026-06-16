#!/usr/bin/env python3
"""
Phase B analysis — implements SWEEP_PREREGISTRATION.md decision rules
VERBATIM. Do not edit after observing sweep results.

Per cell:
  * per-seed composite: min-max normalize each metric across the six
    schemes WITHIN that seed, weight by the cell's Table II weights
    (energy weight = w_E, others rescaled), costs inverted.
  * "wins a cell"  = best mean composite AND Welch p < 0.05 vs runner-up.
  * Pareto (H2)    = HAO-PRA-U better than the best-composite baseline on
    BOTH sla_violation and utilization (resource proxy; capacity is equal
    within a cell), each Welch p < 0.05.
Gate 1: HAO-PRA-U winning >= 3 contiguous cells (edge-adjacent in the
5x4 grid) => proceed with the reframed paper.

Usage: python3 analyze_sweep.py [sweep_results/all_cells.csv]
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
OBJ_W_BASE = dict(sla=0.2256, latency=0.1880, throughput=0.1692,
                  acceptance=0.1316, utilization=0.1128,
                  reliability=0.0658, isolation=0.0282, energy=0.0188,
                  fragmentation=0.0390, overhead=0.0210)
PROPOSED = "HAO-PRA-U"


def cell_weights(w_E):
    others = {k: v for k, v in OBJ_W_BASE.items() if k != "energy"}
    s = (1.0 - w_E) / sum(others.values())
    w = {k: v * s for k, v in others.items()}
    w["energy"] = w_E
    return w


def per_seed_composites(rows, w_E):
    """rows: all rows of one cell. Returns {method: [composite per seed]}"""
    W = cell_weights(w_E)
    seeds = sorted({r["seed"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    out = {m: [] for m in methods}
    for sd in seeds:
        sub = {r["method"]: r for r in rows if r["seed"] == sd}
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
    return out


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "sweep_results/all_cells.csv"
    rows = list(csv.DictReader(open(path)))
    cells = sorted({(float(r["w_E"]), float(r["headroom"])) for r in rows})
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(float(r["w_E"]), float(r["headroom"]))].append(r)

    report, hao_wins = {}, []
    for cell in cells:
        wE, h = cell
        comp = per_seed_composites(by_cell[cell], wE)
        means = {m: float(np.mean(v)) for m, v in comp.items()}
        ranked = sorted(means, key=means.get, reverse=True)
        winner, runner = ranked[0], ranked[1]
        t, p = welch(comp[winner], comp[runner])
        wins = p < 0.05
        # Pareto check (H2) for HAO vs best-composite baseline
        best_base = next(m for m in ranked if m != PROPOSED)
        hao_rows = [r for r in by_cell[cell] if r["method"] == PROPOSED]
        base_rows = [r for r in by_cell[cell] if r["method"] == best_base]
        pareto = {}
        for metric in ("sla_violation", "utilization"):
            a = [float(r[metric]) for r in hao_rows]
            b = [float(r[metric]) for r in base_rows]
            tt, pp = welch(a, b)
            pareto[metric] = dict(hao=float(np.mean(a)),
                                  base=float(np.mean(b)), p=pp,
                                  better=bool(np.mean(a) < np.mean(b)
                                              and pp < 0.05))
        report[f"wE={wE},h={h}"] = dict(
            composite_means={m: round(v, 4) for m, v in means.items()},
            winner=winner, runner_up=runner, p_vs_runner=round(p, 5),
            decisive=wins, pareto_vs=best_base, pareto=pareto)
        if winner == PROPOSED and wins:
            hao_wins.append(cell)

    # contiguity on the 5x4 grid (edge adjacency)
    def adjacent(c1, c2):
        i1, j1 = (sorted({c[0] for c in cells}).index(c1[0]),
                  sorted({c[1] for c in cells}).index(c1[1]))
        i2, j2 = (sorted({c[0] for c in cells}).index(c2[0]),
                  sorted({c[1] for c in cells}).index(c2[1]))
        return abs(i1 - i2) + abs(j1 - j2) == 1
    contiguous = 0
    if hao_wins:
        comp_sets = []
        for c in hao_wins:
            placed = False
            for s in comp_sets:
                if any(adjacent(c, o) for o in s):
                    s.add(c); placed = True; break
            if not placed:
                comp_sets.append({c})
        merged = True
        while merged:
            merged = False
            for a in comp_sets:
                for b in comp_sets:
                    if a is not b and any(adjacent(x, y) for x in a
                                          for y in b):
                        a |= b; comp_sets.remove(b); merged = True; break
                if merged:
                    break
        contiguous = max(len(s) for s in comp_sets)

    gate1 = contiguous >= 3
    summary = dict(hao_decisive_wins=[list(c) for c in hao_wins],
                   largest_contiguous_block=contiguous,
                   GATE1_PASS=bool(gate1))
    with open("sweep_results/decision_report.json", "w") as f:
        json.dump(dict(summary=summary, cells=report), f, indent=1)

    print("Winner map (rows = w_E, cols = headroom):")
    wEs = sorted({c[0] for c in cells}); hs = sorted({c[1] for c in cells})
    for wE in wEs:
        line = []
        for h in hs:
            r = report[f"wE={wE},h={h}"]
            tag = r["winner"][:12] + ("*" if r["decisive"] else "")
            line.append(f"{tag:<14}")
        print(f"w_E={wE:<7}" + " ".join(line))
    print(f"\nHAO-PRA-U decisive wins: {len(hao_wins)} cells; largest "
          f"contiguous block: {contiguous}\nGATE 1: "
          f"{'PASS - proceed with reframed paper' if gate1 else 'FAIL - return to method development or write up the negative result'}")
    print("Full details: sweep_results/decision_report.json")


if __name__ == "__main__":
    main()
