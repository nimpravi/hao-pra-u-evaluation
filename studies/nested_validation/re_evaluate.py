#!/usr/bin/env python3
"""
re_evaluate.py -- recover the full metric set WITHOUT re-tuning.

WHY THIS EXISTS
    nested_driver.py logged only sim.METRICS (7 metrics). The registered
    composite needs 10: reliability, isolation, and overhead were dropped.
    That was a driver bug, not a problem with your run.

WHY IT IS CHEAP
    The expensive part of the study was model selection (10,800 validation
    evaluations). Those results are already saved in
    tuned_params_nested.json. run_scheme() is deterministic given (seed,
    tuned), so replaying the test-window evaluation with the SAME saved
    parameters reproduces exactly the same runs and simply logs the three
    missing columns. Cost is 200 x 6 = 1,200 run_scheme calls, roughly a
    ninth of the original run, with no re-tuning at all.

    Nothing is re-selected. No decision is revisited. This is a logging
    fix, and it does not touch the pre-registration.

PLACEMENT
    studies/nested_validation/re_evaluate.py

USAGE
    python3 re_evaluate.py --workers 8

OUTPUT
    ../../results/nested_validation/all_cells_nested_full.csv
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hao_pra_u_sim as sim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "nested_validation")

# The registered composite needs all ten.
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]


def rebuild(entry):
    """Re-run the six schemes for one (regime, seed) with saved params."""
    w_e, h, seed = entry["w_E"], entry["headroom"], entry["seed"]
    tuned = dict(entry["tuned"])
    if "flat_headroom" in tuned:
        tuned["flat_headroom"] = np.array(tuned["flat_headroom"], float)

    sim.set_regime(w_e, h)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)

    rows = []
    for scheme in sim.SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        rows.append(dict(w_E=w_e, headroom=h,
                         method=sim.PAPER_NAMES[scheme], seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(
        RES, "tuned_params_nested.json"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.params):
        sys.exit(f"tuned params not found: {args.params}\n"
                 "Pass --params /path/to/tuned_params_nested.json")

    entries = json.load(open(args.params))
    print(f"Replaying {len(entries)} (regime, seed) cells x 6 schemes")
    print("using the ALREADY-SELECTED parameters. No re-tuning occurs.\n")

    out = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(rebuild, e): e for e in entries}
            for i, f in enumerate(as_completed(futs), 1):
                out.extend(f.result())
                e = futs[f]
                print(f"  [{i}/{len(entries)}] "
                      f"({e['w_E']}, {e['headroom']}, seed={e['seed']})",
                      flush=True)
    else:
        for i, e in enumerate(entries, 1):
            out.extend(rebuild(e))
            print(f"  [{i}/{len(entries)}] "
                  f"({e['w_E']}, {e['headroom']}, seed={e['seed']})",
                  flush=True)

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "all_cells_nested_full.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"\nWrote {path}")
    print(f"Columns: {', '.join(out[0].keys())}")

    # Consistency check against the 7-metric run, if present.
    old = os.path.join(RES, "all_cells_nested.csv")
    if os.path.exists(old):
        prev = {(r["w_E"], r["headroom"], r["method"], r["seed"]):
                float(r["sla_violation"])
                for r in csv.DictReader(open(old))}
        bad = 0
        for r in out:
            k = (str(r["w_E"]), str(r["headroom"]), r["method"],
                 str(r["seed"]))
            if k in prev and abs(prev[k] - r["sla_violation"]) > 1e-9:
                bad += 1
        if bad:
            print(f"\nWARNING: {bad} cells differ from the original run. "
                  "Replay should be exact; investigate before using.")
        else:
            print("\nConsistency check: SLA violations match the original "
                  "run exactly. Replay is faithful.")


if __name__ == "__main__":
    main()
