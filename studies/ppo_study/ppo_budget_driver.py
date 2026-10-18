#!/usr/bin/env python3
"""
ppo_budget_driver.py -- does HAO-PPO's result change with more training?

Reviewer R5: Table XII reports about 6,600 training steps for the PPO
controller, and Sec. VI-K lists "the training budget was too small" as a
candidate explanation for its failure. Stating that a third explanation
exists is not evidence against it. This driver tests it.

Does NOT modify the frozen simulator. The per-run training budget is read
from `tuned["_ppo_episodes"]`, which the simulator already supports, so the
pinned digest is unchanged.

Two modes:

  --mode budget   Full 20-regime sweep at ONE enlarged budget (default 96
                  episodes, 8x the registered 12). Directly comparable to
                  Study 5: same schemes, same splits, same statistics.
                  Answers "does the verdict survive 8x training?"

  --mode curve    Learning curve: a subset of regimes, every budget in
                  --budgets, 10 seeds each. Answers "is it still improving,
                  or has it plateaued?" Cheaper than the full sweep and the
                  more informative of the two if you only run one.

Usage, from studies/ppo_study/ :

    python ppo_budget_driver.py --mode budget --validate
    python ppo_budget_driver.py --mode budget --episodes 96 --workers 4
    python ppo_budget_driver.py --mode curve  --validate
    python ppo_budget_driver.py --mode curve  --workers 4

Outputs to results/ppo_study/ :
    all_cells_ppo_budget<N>.csv   /  tuned_params_ppo_budget<N>.json
    ppo_learning_curve.csv        /  tuned_params_ppo_curve.json
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "hao_pra_u_sim", os.path.join(HERE, "hao_pra_u_sim.py"))
sim = importlib.util.module_from_spec(_spec)
sys.modules["hao_pra_u_sim"] = sim
_spec.loader.exec_module(sim)

_pspec = importlib.util.spec_from_file_location(
    "ppo_reserve", os.path.join(HERE, "ppo_reserve.py"))
ppo_reserve = importlib.util.module_from_spec(_pspec)
sys.modules["ppo_reserve"] = ppo_reserve
_pspec.loader.exec_module(ppo_reserve)

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
N_SEEDS = 10
ANCHOR = (0.0188, 2.0)
# Regimes for the learning curve: the anchor, the tightest, the most
# abundant, and the highest resource-cost cell. Chosen before seeing
# results, to span the registered space rather than to flatter the curve.
CURVE_CELLS = [(0.0188, 2.0), (0.0188, 1.4), (0.0188, 2.3), (0.30, 2.0)]
# 0 is an untrained control: it must differ from the registered 12,
# or the learned policy is not reaching evaluation at all.
DEFAULT_BUDGETS = [0, 4, 12, 24, 48, 96]

SCHEMES7 = ["Static", "Reactive", "Centralized", "CentralizedPredictive",
            "FlatAgent", "HAO-PRA-U", "HAO-PPO"]
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "ppo_study")


def preflight():
    problems = []
    if getattr(sim, "N_TRAIN", None) != 550:
        problems.append(f"N_TRAIN is {getattr(sim,'N_TRAIN',None)}, expected 550")
    if not hasattr(sim, "tune_all_schemes"):
        problems.append("tune_all_schemes missing")
    if not hasattr(ppo_reserve, "PPO_EPISODES"):
        problems.append("ppo_reserve.PPO_EPISODES missing")
    if "HAO-PPO" not in getattr(sim, "SCHEMES", []):
        problems.append("HAO-PPO not in sim.SCHEMES")
    reg = getattr(ppo_reserve, "PPO_EPISODES", None)
    if reg != 12:
        problems.append(f"PPO_EPISODES is {reg}, expected the registered 12")
    if problems:
        sys.exit("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))
    print(f"Preflight OK. Registered PPO_EPISODES = {reg}")


def run_cell(args):
    w_e, h, seed, budgets = args
    sim.set_regime(w_e, h)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)
    tuned = sim.tune_all_schemes(U, UHAT, seed)

    rows = []
    # Non-PPO schemes do not depend on the PPO budget; run them once and
    # replicate per budget so each budget has a complete comparison set.
    base = {}
    for scheme in SCHEMES7:
        if scheme == "HAO-PPO":
            continue
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        base[sim.PAPER_NAMES[scheme]] = {k: float(res[k]) for k in FULL_METRICS}

    for ep in budgets:
        t = dict(tuned)
        t["_ppo_episodes"] = int(ep)
        res = sim.run_scheme("HAO-PPO", U, UHAT, seed, t)
        for name, vals in base.items():
            rows.append(dict(w_E=w_e, headroom=h, ppo_episodes=int(ep),
                             method=name, seed=seed, **vals))
        rows.append(dict(w_E=w_e, headroom=h, ppo_episodes=int(ep),
                         method="HAO-PPO", seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["budget", "curve"])
    ap.add_argument("--episodes", type=int, default=96,
                    help="budget mode: training episodes (registered is 12)")
    ap.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                    help="curve mode: budgets to sweep")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    preflight()

    if a.mode == "budget":
        budgets = [a.episodes]
        cells = [ANCHOR] if a.validate else \
            [(w, h) for w in W_E_GRID for h in H_GRID]
        print(f"budget mode: {a.episodes} episodes "
              f"({a.episodes/12:.0f}x the registered 12)")
    else:
        budgets = list(a.budgets)
        cells = [ANCHOR] if a.validate else CURVE_CELLS
        print(f"curve mode: budgets {budgets} over {len(cells)} regime(s)")

    seeds = range(2) if a.validate else range(a.seeds)
    jobs = [(w, h, s, budgets) for (w, h) in cells for s in seeds]
    print(f"{'VALIDATE' if a.validate else 'FULL'}: {len(jobs)} cell-seeds "
          f"x {len(budgets)} budget(s), workers={a.workers}")

    all_rows, all_params = [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (r, p) in enumerate(ex.map(run_cell, jobs), 1):
                all_rows.extend(r); all_params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, p = run_cell(j)
            all_rows.extend(r); all_params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    # Tripwire. NOTE the direction: outcomes being IDENTICAL across trained
    # budgets is the expected finding (the policy saturates), not a failure.
    # What must hold is that the UNTRAINED control differs from the
    # registered budget; if it does not, the learned policy never reaches
    # evaluation and every PPO number in this study is meaningless.
    per = {}
    for r in all_rows:
        if r["method"] == "HAO-PPO":
            per.setdefault(r["ppo_episodes"], []).append(r["sla_violation"])
    means = {k: float(np.mean(v)) for k, v in sorted(per.items())}
    print("\nTRIPWIRE  mean SLA violation by training budget:")
    for k, v in means.items():
        tag = "  (untrained control)" if k == 0 else ""
        print(f"  {k:>4} episodes: {v:.6f}{tag}")

    ok = True
    if 0 in means:
        trained = [v for k, v in means.items() if k > 0]
        if trained and all(abs(v - means[0]) < 1e-12 for v in trained):
            print("  FAIL: trained budgets match the untrained control, so the")
            print("        PPO policy is not reaching evaluation. Every PPO")
            print("        result in this study would be invalid. Stop here.")
            ok = False
        else:
            print("  OK: training changes the outcome, so the policy is live.")
    else:
        print("  NOTE: no untrained control in --budgets; add 0 to include one.")

    trained_vals = {k: v for k, v in means.items() if k > 0}
    if len(trained_vals) > 1:
        if len(set(round(v, 12) for v in trained_vals.values())) == 1:
            print("  FINDING: identical across every trained budget. The policy")
            print("           saturates; more training changes nothing.")
        else:
            print("  FINDING: outcomes still vary across trained budgets.")
    if not ok:
        return 1

    if a.validate:
        print("\nValidation complete.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    if a.mode == "budget":
        stem = f"ppo_budget{a.episodes}"
        csv_path = os.path.join(OUT_DIR, f"all_cells_{stem}.csv")
    else:
        stem = "ppo_curve"
        csv_path = os.path.join(OUT_DIR, "ppo_learning_curve.csv")
    json_path = os.path.join(OUT_DIR, f"tuned_params_{stem}.json")

    cols = ["w_E", "headroom", "ppo_episodes", "method", "seed"] + FULL_METRICS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)
    with open(json_path, "w") as f:
        json.dump(all_params, f, indent=1)
    print(f"\nwrote {csv_path} ({len(all_rows)} rows)")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
