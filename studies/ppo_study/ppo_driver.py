#!/usr/bin/env python3
"""
ppo_driver.py -- Study 5: deep-RL reserve control.

Runs the registered 5x4 sweep with SEVEN schemes (the six from Study 4 plus
HAO-PPO), under the identical nested-validation protocol and equal search
budgets. Also records the policy diagnostics needed to test H5.3.

See preregistrations/PREREGISTRATION_PPO_STUDY.md. If this script and that
document disagree, the document governs and this script is the bug.

PLACEMENT
    studies/ppo_study/ppo_driver.py
alongside hao_pra_u_sim.py and ppo_reserve.py.

USAGE
    python3 ppo_driver.py --validate            # anchor cell, 2 seeds
    python3 ppo_driver.py --workers 8           # primary study
    python3 ppo_driver.py --robustness --workers 8   # 36-point corners
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
import ppo_reserve  # noqa: E402

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
ANCHOR = (0.0188, 2.00)
ROBUST_CELLS = [(0.0188, 1.4), (0.0188, 2.3), (0.30, 1.4), (0.30, 2.3)]

PPO_GRID = [(lr, cl) for lr in (1e-4, 3e-4, 1e-3)
            for cl in (0.1, 0.2, 0.3)]                       # 9, primary
PPO_GRID_ROBUST = [(lr, cl, en, hd)
                   for lr in (1e-4, 3e-4, 1e-3)
                   for cl in (0.1, 0.2, 0.3)
                   for en in (0.0, 0.01)
                   for hd in (64, 128)]                      # 36, robustness

FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "ppo_study")

# Probe states for the H5.3 policy-diagnostic sweep.
PROBE = [(a, m) for a in np.linspace(0.3, 1.2, 6)
         for m in np.linspace(0.3, 1.4, 6)]


def tune_ppo(U, UHAT, seed, tuned_base, robust=False):
    """Select PPO hyperparameters on the validation window."""
    grid = PPO_GRID_ROBUST if robust else PPO_GRID
    best, n = (-1e18, None), 0
    for cfg in grid:
        t = dict(tuned_base)
        if robust:
            lr, cl, en, hd = cfg
            t.update(ppo_lr=lr, ppo_clip=cl, ppo_entropy=en, ppo_hidden=hd)
        else:
            lr, cl = cfg
            t.update(ppo_lr=lr, ppo_clip=cl)
        t["_ppo_episodes"] = ppo_reserve.PPO_EPISODES_TUNE

        st = sim.SchemeState()
        sim.pretrain_ppo(st, U, UHAT, t, seed,
                         episodes=ppo_reserve.PPO_EPISODES_TUNE)
        rng = np.random.default_rng(seed + 104729)
        hist, rs = [], []
        for tt in range(sim.VALID_START, sim.VALID_END):
            u = U[tt]
            hist.append(u)
            if len(hist) > sim.ROLL_WIN:
                hist.pop(0)
            ubar = np.mean(hist, axis=0)
            if tt > 0 and not np.isnan(UHAT[tt - 1]).any():
                sigma = np.abs(U[tt - 1] - UHAT[tt - 1]) / (ubar + sim.EPS)
            else:
                sigma = np.zeros(len(sim.SLICES))
            X = sim.allocate("HAO-PPO", st, tt, u, U, UHAT, ubar, sigma,
                             False, rng, t)
            rs.append(sim.reward_from_metrics(
                sim.evaluate_interval("HAO-PPO", X, u, None)))
            st.prev_util = float(np.mean(rs[-1:]) * 0 + st.prev_util)
        sc = float(np.mean(rs))
        n += 1
        if sc > best[0]:
            best = (sc, cfg)
    return best[1], n


def run_cell(w_e, h, seed, robust=False):
    sim.set_regime(w_e, h)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)

    tuned = sim.tune_all_schemes(U, UHAT, seed)      # 6 schemes, 9 each
    cfg, nppo = tune_ppo(U, UHAT, seed, tuned, robust=robust)
    if robust:
        lr, cl, en, hd = cfg
        tuned.update(ppo_lr=lr, ppo_clip=cl, ppo_entropy=en, ppo_hidden=hd)
    else:
        lr, cl = cfg
        tuned.update(ppo_lr=lr, ppo_clip=cl)

    budget = dict(tuned.get("_budget", {}))
    budget["HAO-PPO"] = nppo
    expected = len(PPO_GRID_ROBUST) if robust else len(PPO_GRID)
    if nppo != expected:
        raise RuntimeError(f"T5.4 violated: HAO-PPO got {nppo} evals, "
                           f"expected {expected}")

    rows = []
    for scheme in sim.SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        rows.append(dict(w_E=w_e, headroom=h,
                         method=sim.PAPER_NAMES[scheme], seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))

    # ---- H5.3 diagnostic: is the learned policy state-dependent? ----
    st = sim.SchemeState()
    sim.pretrain_ppo(st, U, UHAT, tuned, seed)
    acts = [st.ppo.act(ppo_reserve.build_state(a, m, 0.09, 0.70, 0.50),
                       training=False) for a, m in PROBE]
    diag = dict(w_E=w_e, headroom=h, seed=seed,
                distinct_actions=len(set(acts)),
                modal_action=int(max(set(acts), key=acts.count)),
                modal_share=acts.count(max(set(acts), key=acts.count))
                / len(acts),
                action_counts=json.dumps(
                    {int(a): acts.count(a) for a in sorted(set(acts))}))

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged,
                      budget=budget), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    if "HAO-PPO" not in sim.SCHEMES:
        sys.exit("PATCH CHECK FAILED: HAO-PPO not registered in sim.SCHEMES")

    if args.validate:
        cells, seeds = [ANCHOR], range(2)
        print("TRIPWIRE RUN: anchor cell, 2 seeds.")
    elif args.robustness:
        cells, seeds = ROBUST_CELLS, range(args.seeds)
        print(f"ROBUSTNESS: {len(cells)} corner regimes x {args.seeds} "
              f"seeds, HAO-PPO gets {len(PPO_GRID_ROBUST)} evaluations")
        print("(deliberate asymmetry favouring the learned method)")
    else:
        cells = [(w, h) for w in W_E_GRID for h in H_GRID]
        seeds = range(args.seeds)
        print(f"PRIMARY STUDY: {len(cells)} regimes x {args.seeds} seeds "
              f"x 7 schemes, equal 9-point budgets")

    jobs = [(w, h, s) for (w, h) in cells for s in seeds]
    rows, params, diags = [], [], []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, w, h, s, args.robustness): (w, h, s)
                    for (w, h, s) in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                r, p, d = f.result()
                rows += r
                params.append(p)
                diags.append(d)
                print(f"  [{i}/{len(jobs)}] {futs[f]}", flush=True)
    else:
        for i, (w, h, s) in enumerate(jobs, 1):
            r, p, d = run_cell(w, h, s, args.robustness)
            rows += r
            params.append(p)
            diags.append(d)
            print(f"  [{i}/{len(jobs)}] ({w}, {h}, seed={s})", flush=True)

    os.makedirs(RES, exist_ok=True)
    tag = ("_validate" if args.validate else
           "_robust" if args.robustness else "")
    for name, data in (("all_cells_ppo", rows),
                       ("policy_diagnostics", diags)):
        p = os.path.join(RES, f"{name}{tag}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"Wrote {p}")
    p = os.path.join(RES, f"tuned_params_ppo{tag}.json")
    json.dump(params, open(p, "w"), indent=2)
    print(f"Wrote {p}")

    # ---- H5.3 summary ----
    dist = [d["distinct_actions"] for d in diags]
    share = [d["modal_share"] for d in diags]
    print(f"\nH5.3 diagnostic (policy state-dependence):")
    print(f"  distinct greedy actions per cell: mean {np.mean(dist):.2f}, "
          f"max {max(dist)}, cells with >1: {sum(d > 1 for d in dist)}"
          f"/{len(dist)}")
    print(f"  modal action share: mean {np.mean(share):.1%}")
    # Modal share is the discriminating statistic: a policy that uses a
    # second action on a handful of probe states is not meaningfully
    # state-dependent. Registered threshold: H5.3 is supported only if the
    # modal action accounts for less than 80% of greedy selections.
    if np.mean(share) >= 0.80:
        print(f"  -> policy is effectively CONSTANT "
              f"(modal share {np.mean(share):.1%} >= 80%): a richer policy")
        print("     class finds nothing a tuned margin does not already")
        print("     capture. H5.3 NOT supported.")
    else:
        print(f"  -> policy is STATE-DEPENDENT "
              f"(modal share {np.mean(share):.1%} < 80%): H5.3 supported.")


if __name__ == "__main__":
    main()
