#!/usr/bin/env python3
"""
nested_driver.py -- Study 4: equal-budget nested validation.

Runs the registered 5x4 regime sweep with:
  * a three-way train / validation / test split, and
  * an equal 9-point search budget for every scheme, HAO-PRA-U included.

See preregistrations/PREREGISTRATION_NESTED_VALIDATION.md. This driver
implements that document verbatim; if the two disagree, the pre-registration
governs and this script is the bug.

PLACEMENT
    hao-pra-u-evaluation/studies/nested_validation/nested_driver.py
alongside the patched hao_pra_u_sim.py.

USAGE
    python3 nested_driver.py --validate        # tripwire run, anchor cell
    python3 nested_driver.py --workers 8       # full study

OUTPUT (written to ../../results/nested_validation/)
    all_cells_nested.csv        per (regime, seed, scheme) test-window KPIs
    tuned_params_nested.json    selected parameters per (regime, seed)
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

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
ANCHOR = (0.0188, 2.00)
N_SEEDS = 10
EXPECTED_BUDGET = 9

# The registered composite needs ten metrics; sim.METRICS holds only seven.
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "results", "nested_validation")


def check_patches():
    """Fail fast if the simulator has not been patched as specified."""
    problems = []
    for name in ("VALID_START", "VALID_END", "HAO_GAMMA_GRID",
                 "HAO_RESERVE_SCALE_GRID"):
        if not hasattr(sim, name):
            problems.append(f"missing {name} (Patch 1 or 2 not applied)")
    if hasattr(sim, "N_TRAIN") and sim.N_TRAIN != 550:
        problems.append(f"N_TRAIN is {sim.N_TRAIN}, expected 550 (Patch 1)")
    if not (hasattr(sim, "tune_all_schemes") or
            hasattr(sim, "tune_baselines")):
        problems.append("no tuning function found")
    if problems:
        sys.exit("PATCH CHECK FAILED:\n  " + "\n  ".join(problems))


def tuner():
    return getattr(sim, "tune_all_schemes", None) or sim.tune_baselines


def run_cell(w_e, h, seed):
    """Tune on validation, evaluate on test, for one (regime, seed)."""
    sim.set_regime(w_e, h)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)      # fits on 0..N_TRAIN-1
    tuned = tuner()(U, UHAT, seed)                  # selects on validation

    # Tripwire T3: every scheme must have received exactly 9 evaluations.
    budget = tuned.get("_budget", {})
    bad = {k: v for k, v in budget.items() if v != EXPECTED_BUDGET}
    if budget and bad:
        raise RuntimeError(f"T3 violated at ({w_e},{h},seed={seed}): {bad}")

    rows = []
    for scheme in sim.SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        rows.append(dict(w_E=w_e, headroom=h,
                         method=sim.PAPER_NAMES[scheme], seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def tripwire_checks(params):
    """T2: selected values must lie inside the registered grids."""
    grid9 = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    issues = []
    for p in params:
        t = p["tuned"]
        for key, grid in (("static_margin", grid9),
                          ("reactive_margin", grid9),
                          ("cp_margin", grid9)):
            if key in t and round(float(t[key]), 4) not in grid:
                issues.append(f"{key}={t[key]} out of grid "
                              f"at w_E={p['w_E']} h={p['headroom']} "
                              f"seed={p['seed']}")
        if "hao_gamma" in t and float(t["hao_gamma"]) not in \
                sim.HAO_GAMMA_GRID:
            issues.append(f"hao_gamma={t['hao_gamma']} out of grid")
        if "hao_reserve_scale" in t and float(t["hao_reserve_scale"]) not in \
                sim.HAO_RESERVE_SCALE_GRID:
            issues.append(f"hao_reserve_scale={t['hao_reserve_scale']} out of grid")
    return issues


def summarize_selection(params):
    """Report where HAO-PRA-U's selected parameters landed."""
    gammas = [p["tuned"].get("hao_gamma") for p in params
              if "hao_gamma" in p["tuned"]]
    scales = [p["tuned"].get("hao_reserve_scale") for p in params
              if "hao_reserve_scale" in p["tuned"]]
    if not gammas:
        return
    print("\nHAO-PRA-U selected parameters across all (regime, seed):")
    for name, vals, grid in (("hao_gamma", gammas, sim.HAO_GAMMA_GRID),
                             ("hao_reserve_scale", scales,
                              sim.HAO_RESERVE_SCALE_GRID)):
        counts = {g: vals.count(g) for g in grid}
        total = len(vals)
        line = "  ".join(f"{g}: {c}/{total}" for g, c in counts.items())
        print(f"  {name:<16} {line}")
        edges = {grid[0], grid[-1]}
        at_edge = sum(counts[g] for g in edges)
        if total and at_edge / total > 0.8:
            print(f"    NOTE: {at_edge}/{total} selections at a grid "
                  f"boundary. Report as a boundary optimum; do NOT widen "
                  f"the grid post hoc.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="tripwire run: anchor cell, 2 seeds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    check_patches()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.validate:
        cells = [ANCHOR]
        seeds = range(2)
        print("TRIPWIRE RUN: anchor cell, 2 seeds. Checks T1/T2/T3.")
        print("Compare HAO-PRA-U SLA violation against the published "
              "anchor value (0.0121 +/- 0.0059) for T1.\n")
    else:
        cells = [(w, h) for w in W_E_GRID for h in H_GRID]
        seeds = range(args.seeds)
        print(f"FULL STUDY: {len(cells)} regimes x {args.seeds} seeds "
              f"x 6 schemes x {EXPECTED_BUDGET} validation points")
        print(f"= {len(cells) * args.seeds * 6 * EXPECTED_BUDGET} "
              f"validation evaluations. This will take a while.\n")

    jobs = [(w, h, s) for (w, h) in cells for s in seeds]
    all_rows, all_params = [], []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, w, h, s): (w, h, s)
                    for (w, h, s) in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                rows, params = fut.result()
                all_rows.extend(rows)
                all_params.append(params)
                print(f"  [{i}/{len(jobs)}] done {futs[fut]}", flush=True)
    else:
        for i, (w, h, s) in enumerate(jobs, 1):
            rows, params = run_cell(w, h, s)
            all_rows.extend(rows)
            all_params.append(params)
            print(f"  [{i}/{len(jobs)}] done ({w}, {h}, seed={s})",
                  flush=True)

    issues = tripwire_checks(all_params)
    if issues:
        print("\nT2 TRIPWIRE VIOLATIONS (run is void):")
        for m in issues:
            print("  " + m)
        sys.exit(1)
    print("\nTripwires T2/T3: passed.")

    suffix = "_validate" if args.validate else ""
    csv_path = os.path.join(OUT_DIR, f"all_cells_nested{suffix}.csv")
    json_path = os.path.join(OUT_DIR, f"tuned_params_nested{suffix}.json")

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(json_path, "w") as fh:
        json.dump(all_params, fh, indent=2)

    summarize_selection(all_params)

    if args.validate:
        hao = [r["sla_violation"] for r in all_rows
               if r["method"] == "HAO-PRA-U"]
        print(f"\nTuned HAO-PRA-U mean SLA violation = {np.mean(hao):.4f}")
        print("  (published anchor, DEFAULT config: 0.0121 +/- 0.0059)")
        print("  These two are NOT expected to match: this run uses the")
        print("  TUNED configuration, and a lower value here is the intended")
        print("  effect of the equal search budget, not a patching error.")
        print("  Tripwire T1 is an equivalence test at the DEFAULT config;")
        print("  run t1_check.py for it.")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
