#!/usr/bin/env python3
"""
eps_sensitivity.py  --  Step 3 evidence for Reviewer 4's epsilon question.

R4 asked: how was the division-guard epsilon chosen, what is its order of
magnitude, and how sensitive are the results to it?

This sweeps EPS over several orders of magnitude at the anchor regime and
reports whether the per-scheme metrics and the winner change.

WHERE TO PUT THIS FILE
----------------------
Place it in the SAME directory as the simulator:

    hao-pra-u-evaluation/studies/regime_sweep/eps_sensitivity.py

HOW TO RUN
----------
    cd hao-pra-u-evaluation/studies/regime_sweep
    python3 eps_sensitivity.py                # 3 seeds  (~few minutes)
    python3 eps_sensitivity.py --seeds 5      # more seeds, slower

Send back the whole stdout, plus the file eps_sensitivity_results.csv that
it writes next to itself.

NOTE ON RUNTIME
---------------
Each (eps, seed) pair retrains the GRU and runs all six schemes, so cost is
roughly (number of eps values) x (seeds) x one full anchor run. Start with
the default 3 seeds; only increase if it finishes quickly.
"""
import argparse
import csv
import importlib
import os
import sys

import numpy as np

EPS_GRID = [1e-9, 1e-8, 1e-6, 1e-4, 1e-3]   # 1e-6 is the registered value
REGISTERED_EPS = 1e-6

METRICS_REPORT = ["sla_violation", "latency", "throughput", "utilization",
                  "energy_eff", "fragmentation", "acceptance"]


def run_one(eps, seeds):
    """Reload the simulator with a patched EPS and run the anchor regime."""
    if "hao_pra_u_sim" in sys.modules:
        del sys.modules["hao_pra_u_sim"]
    sim = importlib.import_module("hao_pra_u_sim")

    # patch the guard constant, then set the anchor regime
    sim.EPS = eps
    sim.set_regime(0.0188, 2.00)

    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        U = sim.generate_traffic(rng)
        UHAT, _ = sim.train_gru_forecasts(U, seed)
        tuned = sim.tune_baselines(U, UHAT, seed)
        for scheme in sim.SCHEMES:
            res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
            rows.append(dict(eps=eps, seed=seed,
                             method=sim.PAPER_NAMES[scheme],
                             **{k: float(res[k]) for k in METRICS_REPORT}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    if not os.path.exists(os.path.join(here, "hao_pra_u_sim.py")):
        sys.exit("ERROR: hao_pra_u_sim.py not found next to this script.\n"
                 "Put eps_sensitivity.py in studies/regime_sweep/ and rerun.")

    all_rows = []
    for eps in EPS_GRID:
        print(f"running eps = {eps:.0e} ...", flush=True)
        all_rows.extend(run_one(eps, args.seeds))

    out = os.path.join(here, "eps_sensitivity_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # ---- summary: per-scheme mean SLA violation vs eps -------------------
    methods = sorted({r["method"] for r in all_rows})
    print("\nMean SLA violation ratio by scheme and eps "
          f"(anchor regime, {args.seeds} seeds)\n")
    hdr = f"{'method':<26}" + "".join(f"{e:>12.0e}" for e in EPS_GRID)
    print(hdr)
    print("-" * len(hdr))
    max_dev = 0.0
    for m in methods:
        line = f"{m:<26}"
        ref = None
        for e in EPS_GRID:
            vals = [r["sla_violation"] for r in all_rows
                    if r["method"] == m and r["eps"] == e]
            mu = float(np.mean(vals))
            if e == REGISTERED_EPS:
                ref = mu
            line += f"{mu:>12.5f}"
        for e in EPS_GRID:
            vals = [r["sla_violation"] for r in all_rows
                    if r["method"] == m and r["eps"] == e]
            max_dev = max(max_dev, abs(float(np.mean(vals)) - ref))
        print(line)

    print(f"\nLargest absolute deviation from the registered "
          f"eps={REGISTERED_EPS:.0e}: {max_dev:.6f}")

    # ---- did the best-SLA scheme ever change? ----------------------------
    print("\nBest-SLA scheme at each eps:")
    stable = True
    ref_best = None
    for e in EPS_GRID:
        means = {m: float(np.mean([r["sla_violation"] for r in all_rows
                                   if r["method"] == m and r["eps"] == e]))
                 for m in methods}
        best = min(means, key=means.get)
        if e == REGISTERED_EPS:
            ref_best = best
        print(f"  eps={e:.0e}: {best}")
    for e in EPS_GRID:
        means = {m: float(np.mean([r["sla_violation"] for r in all_rows
                                   if r["method"] == m and r["eps"] == e]))
                 for m in methods}
        if min(means, key=means.get) != ref_best:
            stable = False
    print(f"\nOrdering stable across the eps sweep: {stable}")
    print(f"Per-run detail written to: {out}")


if __name__ == "__main__":
    main()
