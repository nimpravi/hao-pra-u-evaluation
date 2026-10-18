#!/usr/bin/env python3
"""
confirm_static_edge_driver.py -- confirmatory replication of Static + edge
correction on fresh seeds (reviewer M1.1)

Registered in PREREGISTRATION_STATIC_EDGE_CONFIRM.md. Do not run this until
that document is hashed, committed and deposited with an independent
timestamping service. Running first makes this the eighth exploratory study
rather than the first confirmatory one, and the paper already has enough
exploratory studies.

WHY THIS IS A WRAPPER AND NOT A COPY

A replication should execute the same code as the thing it replicates, not a
retyped version of it. So this driver imports `static_edge_driver` and runs
its `run_cell` unchanged, overriding only:

  * the seed set, to 11..30, which no prior study in this paper has used
  * the output filenames

Every scheme definition, the ported edge block, the equivalence check and the
tuning path are therefore bit-identical to the exploratory run by
construction, and that property is checked rather than asserted: the import
is verified against the SHA-256 recorded below before anything runs.

Usage, from studies/nested_validation/ :

    python confirm_static_edge_driver.py --verify-only
    python confirm_static_edge_driver.py --workers 4

Outputs to results/nested_validation/ :
    all_cells_confirm_edge.csv  /  tuned_params_confirm_edge.json
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "static_edge_driver.py")

# Registered seed set. Seeds 0..9 were used in Studies 4..13; seed 10 is a
# buffer against off-by-one reuse. Fixed in the registration, Sec. 3.
CONFIRM_SEEDS = list(range(11, 31))

# Paste the digest of the static_edge_driver.py you are replicating here,
# from preregistrations/hashes.txt, before freezing the registration. Leave
# as None only if you accept that the "same code" guarantee is unchecked.
EXPECTED_DRIVER_SHA256 = "d006fe5c5fdb5b5578ccc641785e939a01fccafcaf9435c2c0394f42869fda0f"


def load_driver():
    if not os.path.exists(DRIVER):
        sys.exit(f"ABORT: {DRIVER} not found. The confirmatory run must "
                 "execute the exploratory driver, not a copy of it.")
    digest = hashlib.sha256(open(DRIVER, "rb").read()).hexdigest()
    print(f"static_edge_driver.py SHA-256: {digest}")
    if EXPECTED_DRIVER_SHA256 is None:
        print("  WARNING: EXPECTED_DRIVER_SHA256 is not set, so the "
              "replicated-code guarantee is unverified.")
    elif digest != EXPECTED_DRIVER_SHA256:
        sys.exit(f"ABORT: driver digest does not match the registered value\n"
                 f"  expected {EXPECTED_DRIVER_SHA256}\n"
                 f"  found    {digest}\n"
                 "The confirmatory run would not be replicating the "
                 "registered scheme.")
    else:
        print("  OK: matches the digest recorded in the registration.")
    spec = importlib.util.spec_from_file_location("static_edge_driver", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["static_edge_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify-only", action="store_true",
                    help="run the equivalence check and stop")
    a = ap.parse_args()

    sed = load_driver()
    out_dir = sed.OUT_DIR

    print("\nEQUIVALENCE CHECK (the exploratory driver's own)")
    compared, mismatches = sed.equivalence_check()
    print(f"  intervals compared: {compared}, mismatching: {mismatches}")
    if mismatches or compared == 0:
        print("  FAIL: nothing downstream would be valid.")
        return 2
    print("  OK.\n")
    if a.verify_only:
        return 0

    jobs = [(w, h, s)
            for w in sed.W_E_GRID for h in sed.H_GRID
            for s in CONFIRM_SEEDS]
    print(f"CONFIRMATORY: {len(jobs)} cell-seeds "
          f"(seeds {CONFIRM_SEEDS[0]}..{CONFIRM_SEEDS[-1]}), "
          f"workers={a.workers}")
    print("Registered primary statistic: absolute cost Jbar, fixed "
          "normalizers.")
    print("Registered test: paired Wilcoxon, Holm across 20 regimes, "
          "alpha=0.05.\n")

    rows, params = [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (r, p) in enumerate(ex.map(sed.run_cell, jobs), 1):
                rows.extend(r); params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, p = sed.run_cell(j)
            rows.extend(r); params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    print("\nTRIPWIRE")
    f1 = [p["tuned"]["edge_fires_per_interval_nofc"] for p in params]
    print(f"  edge step fires per interval, median {np.median(f1):.3f}")
    ok = max(f1) > 0
    print("  OK: the edge step is active." if ok else
          "  FAIL: the edge step never fired; the comparison is vacuous.")

    seeds_seen = sorted({r["seed"] for r in rows})
    print(f"  seeds in output: {seeds_seen[0]}..{seeds_seen[-1]} "
          f"({len(seeds_seen)} distinct)")
    if set(seeds_seen) & set(range(0, 11)):
        print("  FAIL: a previously used seed appears in the confirmatory "
              "output. This is not an independent replication.")
        ok = False
    else:
        print("  OK: no previously used seed appears.")

    print("\n  NOTE: the registered verdict is NOT computed here. Score this "
          "with\n        the analysis script and read the verdict off the "
          "rule in\n        PREREGISTRATION_STATIC_EDGE_CONFIRM.md Sec. 4.")

    os.makedirs(out_dir, exist_ok=True)
    cp = os.path.join(out_dir, "all_cells_confirm_edge.csv")
    jp = os.path.join(out_dir, "tuned_params_confirm_edge.json")
    cols = ["w_E", "headroom", "method", "seed"] + sed.FULL_METRICS
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(jp, "w") as f:
        json.dump(params, f, indent=1)
    print(f"\nwrote {cp} ({len(rows)} rows)")
    print(f"wrote {jp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
