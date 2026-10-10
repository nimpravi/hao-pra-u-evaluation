#!/usr/bin/env python3
"""
ablation_driver.py -- Study 6: per-component ablation of the hierarchy.

Reviewer 5 (comment 2) observed that the paper's ablation ladder
(Centralized Predictive -> Flat-Agent -> HAO-PRA-U) changes several things
at once, so it cannot say WHICH hierarchical component is responsible for
the outcome. This study isolates each one.

Design: leave-one-out from the full hierarchy. Four components are toggled
independently, each removed while the other three stay active:

  A  abl_reserve    learned reserve selection -> fixed reserve (0.20)
  B  abl_risk       SLA-risk weighting removed from the fill order
  C  abl_negotiate  inter-domain capacity negotiation skipped
  D  abl_edge       edge-level correction skipped

Leave-one-out (rather than an additive ladder) is chosen because it
measures each component's marginal contribution IN THE PRESENCE of the
others, which is the question R5 asks. The GRU forecaster and all tuned
parameters are held fixed at the Study 4 selections, so only the
architecture varies.

USAGE
    python3 ablation_driver.py --validate           # 2 regimes, 3 seeds
    python3 ablation_driver.py --workers 8          # full 20 regimes
    python3 ablation_driver.py --params <tuned_params_nested.json>
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

CONFIGS = {
    "FULL": {},
    "no_reserve": {"abl_reserve": False},
    "no_risk": {"abl_risk": False},
    "no_negotiate": {"abl_negotiate": False},
    "no_edge": {"abl_edge": False},
}

FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "ablation_study")


def load_tuned(params_path):
    """Map (w_E, h, seed) -> tuned dict from the Study 4 selections."""
    if not params_path or not os.path.exists(params_path):
        return {}
    out = {}
    for e in json.load(open(params_path)):
        t = dict(e["tuned"])
        if "flat_headroom" in t:
            t["flat_headroom"] = np.array(t["flat_headroom"], float)
        out[(float(e["w_E"]), float(e["headroom"]), int(e["seed"]))] = t
    return out


def run_cell(w_e, h, seed, tuned_base):
    sim.set_regime(w_e, h)
    U = sim.generate_traffic(np.random.default_rng(seed))
    UHAT, _ = sim.train_gru_forecasts(U, seed)

    rows = []
    neg_fires = None
    for name, toggles in CONFIGS.items():
        t = dict(tuned_base)
        t.update(toggles)
        res = sim.run_scheme("HAO-PRA-U", U, UHAT, seed, t)
        rows.append(dict(w_E=w_e, headroom=h, seed=seed, config=name,
                         **{k: float(res[k]) for k in FULL_METRICS}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--params", default=os.path.join(
        HERE, "..", "..", "results", "nested_validation",
        "tuned_params_nested.json"))
    args = ap.parse_args()

    # --- guard: refuse to run against a simulator lacking the toggles ---
    if not getattr(sim, "ABLATION_TOGGLES_SUPPORTED", False):
        sys.exit(
            "PATCH CHECK FAILED: the imported hao_pra_u_sim.py does not\n"
            "implement the Study 6 ablation toggles, so every configuration\n"
            "would run an identical FULL hierarchy and the results would be\n"
            "silently meaningless.\n\n"
            f"  imported from: {os.path.abspath(sim.__file__)}\n\n"
            "Copy the Step 6 hao_pra_u_sim.py into the directory containing\n"
            "this driver and rerun.")

    tuned_map = load_tuned(args.params)
    if tuned_map:
        print(f"Using Study 4 tuned parameters from {args.params}")
    else:
        print("WARNING: no tuned params found; using registered defaults.")

    default_tuned = {"static_margin": 1.40, "reactive_margin": 1.40,
                     "cent_period": 3, "cent_margin": 1.15,
                     "cp_margin": 1.40,
                     "flat_headroom": np.full(len(sim.DOMAINS), 1.40),
                     "hao_gamma": 0.85, "hao_reserve_scale": 1.5}

    if args.validate:
        cells, seeds = [(0.0188, 1.4), (0.0188, 2.3)], range(3)
        print("VALIDATION: 2 regimes x 3 seeds x 5 configs")
    else:
        cells = [(w, h) for w in W_E_GRID for h in H_GRID]
        seeds = range(args.seeds)
        print(f"FULL ABLATION: {len(cells)} regimes x {args.seeds} seeds "
              f"x {len(CONFIGS)} configs")

    jobs = []
    for (w, h) in cells:
        for s in seeds:
            jobs.append((w, h, s,
                         tuned_map.get((w, h, s), default_tuned)))

    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, *j): j[:3] for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                rows += f.result()
                print(f"  [{i}/{len(jobs)}] {futs[f]}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows += run_cell(*j)
            print(f"  [{i}/{len(jobs)}] {j[:3]}", flush=True)

    os.makedirs(RES, exist_ok=True)
    tag = "_validate" if args.validate else ""
    path = os.path.join(RES, f"ablation_results{tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}")

    # ---- summary: marginal contribution of each component ----
    print("\nMarginal effect of REMOVING each component "
          "(mean SLA violation; positive = removal hurts):\n")
    hdr = f"{'config':<14}{'sla':>10}{'delta_vs_FULL':>15}{'identical':>11}"
    print(hdr)
    print("-" * len(hdr))
    full = np.mean([r["sla_violation"] for r in rows
                    if r["config"] == "FULL"])
    for name in CONFIGS:
        v = np.mean([r["sla_violation"] for r in rows
                     if r["config"] == name])
        same = all(
            abs(a["sla_violation"] - b["sla_violation"]) < 1e-12
            for a, b in zip(
                sorted([r for r in rows if r["config"] == "FULL"],
                       key=lambda r: (r["w_E"], r["headroom"], r["seed"])),
                sorted([r for r in rows if r["config"] == name],
                       key=lambda r: (r["w_E"], r["headroom"], r["seed"]))))
        flag = "YES" if (same and name != "FULL") else ""
        d = "" if name == "FULL" else f"{v - full:+.5f}"
        print(f"{name:<14}{v:>10.5f}{d:>15}{flag:>11}")
    print("\n'identical = YES' means the component never changed any "
          "outcome:\nit is inert in this environment.")

    # --- sanity: not EVERY component can be inert ---
    n_inert = 0
    ref = sorted([r for r in rows if r["config"] == "FULL"],
                 key=lambda r: (r["w_E"], r["headroom"], r["seed"]))
    for name in CONFIGS:
        if name == "FULL":
            continue
        cur = sorted([r for r in rows if r["config"] == name],
                     key=lambda r: (r["w_E"], r["headroom"], r["seed"]))
        if all(abs(a["sla_violation"] - b["sla_violation"]) < 1e-12
               for a, b in zip(ref, cur)):
            n_inert += 1
    if n_inert == len(CONFIGS) - 1:
        print("\nERROR: ALL components report inert. That is not a finding,\n"
              "it means the toggles had no effect. Verify you are importing\n"
              "the Step 6 simulator. Results are VOID.")
        sys.exit(2)


if __name__ == "__main__":
    main()
