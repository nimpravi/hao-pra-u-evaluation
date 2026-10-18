#!/usr/bin/env python3
"""
static_oracle_driver.py -- does state-dependent provisioning pay once you
stop charging it for coordination it does not use? (reviewer M1.2 / S5.1)

THE QUESTION, and why this run is the one that matters.

Sec. VII-E reports that the clairvoyant reserve oracle loses to tuned Static
on Jbar in all 20 regimes. The per-term decomposition then shows that +0.0178
of that +0.0189 gap, 94%, is the control-overhead term the oracle inherits by
running inside the hierarchy. Net of it the gap is +0.0011. Removing the
alpha5*U term as well leaves the oracle AHEAD by -0.0027.

So the paper's negative conclusion currently rests on one modelled cost. The
clean test is a clairvoyant margin on the STATIC control architecture: choose
per interval, against realized demand, from the baselines' own registered
margin grid, and pay Static's one control message and 0.2 ms delay.

  If Static-Oracle still loses to tuned Static, the environment attribution
  survives and the paper can state it as a result rather than a reading.

  If Static-Oracle wins, the paper's claim narrows to "state-dependent
  provisioning does not pay for THIS hierarchy's coordination", which is a
  smaller and different paper.

Both outcomes are reportable. Only the second changes the headline.

WHY THE ORACLE IS CLEAN HERE. sim.allocate's Static branch is
    X = cap_proportional(ALPHA * (margin * BASE_DEMAND)[:, None])
which depends on the margin and nothing else: not on t, not on realized
demand, not on any carried state. So the per-interval problems are exactly
separable, the greedy per-interval choice is a true upper bound over the
margin grid, and no snapshot/restore of simulator state is needed. This is a
stronger separability argument than the one Sec. VII-E makes for the reserve
oracle, because here there is no state at all.

Does NOT modify the frozen simulator. Every allocation and every metric comes
from sim.allocate and sim.evaluate_interval under the "Static" scheme label,
so Static-Oracle pays exactly Static's control cost, delay and orchestration
energy. That label choice is the whole point of the experiment.

Usage, from studies/nested_validation/ :

    python static_oracle_driver.py --validate
    python static_oracle_driver.py --workers 4

Outputs to results/nested_validation/ :
    all_cells_static_oracle.csv  /  tuned_params_static_oracle.json
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "hao_pra_u_sim", os.path.join(HERE, "hao_pra_u_sim.py"))
sim = importlib.util.module_from_spec(_spec)
sys.modules["hao_pra_u_sim"] = sim
_spec.loader.exec_module(sim)

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
N_SEEDS = 10
ANCHOR = (0.0188, 2.0)

# The baselines' own registered 9-point margin grid, copied from
# sim.tune_all_schemes. The oracle is given exactly the actions a tuned
# Static scheme may choose between, so the only difference between the two
# is WHEN the choice is made: once per regime, or once per interval.
MARGIN_GRID = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]

CAPPED_SCALE_GRID = [0.5, 0.75, 1.0]      # symmetric 1.40 ceiling, as Sec. VII-B
STUDY4_SCHEMES = ["Static", "Reactive", "Centralized",
                  "CentralizedPredictive", "FlatAgent", "HAO-PRA-U"]
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "nested_validation")


def grid_check():
    """The oracle's action set must be the baselines' action set.

    If sim's registered margin grid ever diverges from MARGIN_GRID above, the
    comparison stops being like-for-like and silently favours whichever side
    got the richer set. Catch that here rather than in the results.
    """
    src = ""
    try:
        import inspect
        src = inspect.getsource(sim.tune_all_schemes)
    except Exception:
        return None                       # cannot check; reported, not fatal
    want = "grid9 = [" + ", ".join(f"{g:.2f}" for g in MARGIN_GRID) + "]"
    return want in src


def static_alloc(margin):
    """sim.allocate's Static branch, parameterized by margin.

    Routed through sim.allocate rather than reimplemented, so there is no
    port to drift. state/t/demand arguments are accepted and ignored by that
    branch, which is exactly the property this experiment relies on.
    """
    st = sim.SchemeState()
    rng = np.random.default_rng(0)
    return np.array(sim.allocate("Static", st, 0, None, None, None, None,
                                 None, False, rng, {"static_margin": margin}),
                    copy=True)


def run_static_oracle(U, seed):
    """Per-interval clairvoyant margin, charged Static's control cost."""
    # Static's allocation depends only on the margin, so the nine candidate
    # allocations can be built once and reused at every interval.
    cand = [static_alloc(g) for g in MARGIN_GRID]
    rows, chosen, spreads = [], [], []
    for t in range(len(U)):
        u = U[t]
        best_r, best_i = -np.inf, 0
        rs = []
        for i, X in enumerate(cand):
            m = sim.evaluate_interval("Static", X, u, None)
            r = sim.reward_from_metrics(m)
            rs.append(r)
            if r > best_r:
                best_r, best_i = r, i
        m = sim.evaluate_interval("Static", cand[best_i], u, None)
        if t >= sim.N_TRAIN:
            rows.append(m)
            chosen.append(MARGIN_GRID[best_i])
            spreads.append(float(max(rs) - min(rs)))
    return rows, chosen, spreads


def run_static_fixed(U, margin):
    """Tuned Static under the identical evaluation loop.

    Used only by the validation tripwire. The headline comparison uses
    sim.run_scheme('Static', ...) so that tuned Static is scored by exactly
    the code path every other study uses.
    """
    X = static_alloc(margin)
    return [sim.evaluate_interval("Static", X, U[t], None)
            for t in range(sim.N_TRAIN, len(U))]


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in FULL_METRICS}


def run_cell(args):
    w_e, h, seed = args
    sim.set_regime(w_e, h)
    sim.HAO_RESERVE_SCALE_GRID = list(CAPPED_SCALE_GRID)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)
    tuned = sim.tune_all_schemes(U, UHAT, seed)

    rows = []
    for scheme in STUDY4_SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        rows.append(dict(w_E=w_e, headroom=h, method=sim.PAPER_NAMES[scheme],
                         seed=seed, **{k: float(res[k]) for k in FULL_METRICS}))

    orows, chosen, spreads = run_static_oracle(U, seed)
    rows.append(dict(w_E=w_e, headroom=h, method="Static-Oracle", seed=seed,
                     **agg(orows)))

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    cnt = Counter(chosen)
    logged["oracle_distinct_margins"] = len(cnt)
    logged["oracle_modal_margin"] = float(cnt.most_common(1)[0][0])
    logged["oracle_modal_share"] = cnt.most_common(1)[0][1] / max(len(chosen), 1)
    logged["oracle_mean_margin"] = float(np.mean(chosen))
    logged["oracle_median_reward_spread"] = float(np.median(spreads))
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    print("ACTION-SET CHECK: oracle grid vs the simulator's registered grid")
    gc = grid_check()
    if gc is None:
        print("  could not read sim.tune_all_schemes; check by hand")
    elif gc:
        print(f"  OK: both use {MARGIN_GRID}\n")
    else:
        print("  FAIL: the simulator's registered margin grid is not the grid")
        print(f"        this driver gives the oracle ({MARGIN_GRID}).")
        print("        The comparison would not be like-for-like.")
        return 2

    cells = [ANCHOR] if a.validate else \
        [(w, h) for w in W_E_GRID for h in H_GRID]
    seeds = range(2) if a.validate else range(a.seeds)
    jobs = [(w, h, s) for (w, h) in cells for s in seeds]
    print(f"{'VALIDATE' if a.validate else 'FULL'}: {len(jobs)} cell-seeds, "
          f"workers={a.workers}")

    rows, params = [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (r, p) in enumerate(ex.map(run_cell, jobs), 1):
                rows.extend(r); params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, p = run_cell(j)
            rows.extend(r); params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    print("\nTRIPWIRES")
    ok = True

    # 1. The candidate margins must be genuinely distinguishable, or a
    #    per-interval choice is meaningless and "the oracle ties" says
    #    nothing about the environment.
    spread = [p["tuned"]["oracle_median_reward_spread"] for p in params]
    print(f"  median reward spread across the 9 margins: {np.median(spread):.5f}")
    if np.median(spread) <= 0.0:
        print("  FAIL: the nine margins are indistinguishable under the")
        print("        reward, so there is nothing to choose between.")
        ok = False
    else:
        print("  OK: the margins are distinguishable, so the choice is live.")

    # 2. The oracle must not be beaten by tuned Static on SLA violation in
    #    every single cell. It optimizes reward, not SLA, so this is a smoke
    #    test for a wiring error rather than a result.
    byk = {}
    for r in rows:
        byk.setdefault(r["method"], []).append(r["sla_violation"])
    if "Static-Oracle" in byk and "Static Allocation" in byk:
        o = np.array(byk["Static-Oracle"]); s = np.array(byk["Static Allocation"])
        worse = int((o > s).sum())
        print(f"  oracle worse than tuned Static on SLA in {worse}/{len(o)} "
              f"cell-seeds")
        if worse == len(o):
            print("  FAIL: the clairvoyant scheme is worse everywhere, which")
            print("        points at a wiring error rather than a finding.")
            ok = False

    # Direction is the result and is never a failure condition.
    dm = [p["tuned"]["oracle_distinct_margins"] for p in params]
    ms = [p["tuned"]["oracle_modal_share"] for p in params]
    print(f"  distinct margins used: median {np.median(dm):.1f} of "
          f"{len(MARGIN_GRID)};  modal share median {np.median(ms):.3f}")
    print("  mean SLA violation:")
    for m in ("Static Allocation", "Static-Oracle", "HAO-PRA-U"):
        if m in byk:
            print(f"    {m:<22}{np.mean(byk[m]):.5f}")
    print("\n  NOTE: do not read the direction off these two anchor seeds.")
    print("        Run the sweep and score it with analyze_static_edge.py.")

    if a.validate:
        print("\nValidation complete.")
        return 0 if ok else 1

    os.makedirs(OUT_DIR, exist_ok=True)
    cp = os.path.join(OUT_DIR, "all_cells_static_oracle.csv")
    jp = os.path.join(OUT_DIR, "tuned_params_static_oracle.json")
    cols = ["w_E", "headroom", "method", "seed"] + FULL_METRICS
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(jp, "w") as f:
        json.dump(params, f, indent=1)
    print(f"\nwrote {cp} ({len(rows)} rows)")
    print(f"wrote {jp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
