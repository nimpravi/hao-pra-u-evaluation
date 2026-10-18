#!/usr/bin/env python3
"""
static_edge_driver.py -- does edge correction pay outside the hierarchy?
(reviewer R5)

THE QUESTION. Study 6 found edge correction to be the one component that
earns its place: removing it costs 0.139 of composite and hurts in 19 of 20
regimes, the most consistent effect anywhere in the study. It has only ever
been measured INSIDE the hierarchy. If it also works bolted onto tuned
static over-provisioning, the paper gains its first positive, deployable
recommendation: keep tuned margins and one cheap per-slice reallocation
step, and discard the learned reserve, the risk weighting, the negotiation
and the hierarchy around them. If it only works inside the hierarchy, that
is a narrower but still publishable answer.

TWO VARIANTS, because the edge step needs a risk signal and the choice of
signal decides what is being tested:

  Static+Edge     risk computed from the rolling mean. No forecast
                  anywhere. This is the deployable scheme: a margin and a
                  reallocation rule, nothing learned, nothing predicted.

  Static+Edge-F   risk computed from the same GRU forecast HAO-PRA-U uses.
                  Not deployable without a forecaster, but it isolates the
                  hierarchy exactly: identical inputs to HAO's edge step,
                  with everything else stripped away.

Reporting both is the honest move. The first answers "is this worth
building", the second answers "is the hierarchy doing anything the edge
step needs".

Does NOT modify the frozen simulator. The edge-correction block is ported
verbatim into this driver and applied to the simulator's own Static
allocation. Because a hand-port can silently diverge from the original,
--validate runs an EQUIVALENCE CHECK first: it reconstructs HAO-PRA-U's
allocation using the ported block and requires it to match the simulator's
own output bitwise. If that check fails, nothing else runs.

Usage, from studies/nested_validation/ :

    python static_edge_driver.py --validate
    python static_edge_driver.py --workers 4

Outputs to results/nested_validation/ :
    all_cells_static_edge.csv  /  tuned_params_static_edge.json
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

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
N_SEEDS = 10
ANCHOR = (0.0188, 2.0)
CAPPED_SCALE_GRID = [0.5, 0.75, 1.0]      # symmetric 1.40 ceiling
STUDY4_SCHEMES = ["Static", "Reactive", "Centralized",
                  "CentralizedPredictive", "FlatAgent", "HAO-PRA-U"]
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "nested_validation")


def edge_correct(X, u, R):
    """Verbatim port of the edge-correction block in sim.allocate.

    Kept character-for-character faithful to the original rather than
    tidied, so the equivalence check below is meaningful. X is modified in
    place and returned.
    """
    nS = len(sim.SLICES)
    edge = 1
    lat_crit = [sim.SLICES.index(s) for s in ("URLLC", "XR", "IIoT")]
    donors = [j for j in range(nS) if j not in lat_crit]
    fired = 0
    for j in lat_crit:
        need = sim.ALPHA[j, edge] * u[j]
        if X[j, edge] < need and R[j] > 0.55:
            fired += 1
            want = need - X[j, edge]
            for k_ in sorted(donors, key=lambda x: R[x]):
                surplus = X[k_, edge] - 0.8 * sim.ALPHA[k_, edge] * u[k_]
                give = min(max(surplus, 0.0), want)
                X[k_, edge] -= give
                X[j, edge] += give
                want -= give
                if want <= 0:
                    break
    return X, fired


def equivalence_check(seed=0):
    """Rebuild HAO-PRA-U's allocation with the ported block and require it
    to match the simulator bitwise. Guards against a divergent port."""
    sim.set_regime(*ANCHOR)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)
    tuned = sim.tune_all_schemes(U, UHAT, seed)

    mismatches, compared = 0, 0
    for use_sim_edge in (True,):
        st_a, st_b = sim.SchemeState(), sim.SchemeState()
        t_on = dict(tuned); t_on["abl_edge"] = True
        t_off = dict(tuned); t_off["abl_edge"] = False
        sim.pretrain_hao(st_a, U, UHAT, t_on, seed)
        sim.pretrain_hao(st_b, U, UHAT, t_off, seed)
        hist_a, hist_b = [], []
        for t in range(sim.N_TRAIN, min(sim.N_TRAIN + 120, len(U))):
            u = U[t]
            for h in (hist_a, hist_b):
                h.append(u)
                if len(h) > sim.ROLL_WIN:
                    h.pop(0)
            ubar = np.mean(hist_a, axis=0)
            sigma = (np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
                     if t > 0 and not np.isnan(UHAT[t - 1]).any()
                     else np.zeros(len(sim.SLICES)))
            rng_a = np.random.default_rng(1234 + t)
            rng_b = np.random.default_rng(1234 + t)
            Xa = sim.allocate("HAO-PRA-U", st_a, t, u, U, UHAT, ubar, sigma,
                              False, rng_a, t_on)
            Xb = sim.allocate("HAO-PRA-U", st_b, t, u, U, UHAT, ubar, sigma,
                              False, rng_b, t_off)
            uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
            R = sim.risk_scores(u, uhat, ubar, sigma)
            Xb, _ = edge_correct(np.array(Xb, copy=True), u, R)
            compared += 1
            if not np.allclose(Xa, Xb, rtol=0, atol=0):
                mismatches += 1
    return compared, mismatches


def run_variant(U, UHAT, seed, tuned, use_forecast):
    """Static allocation plus the ported edge step."""
    rng = np.random.default_rng(seed + 7919)
    st = sim.SchemeState()
    hist, rows, fires = [], [], 0
    for t in range(len(U)):
        u = U[t]
        hist.append(u)
        if len(hist) > sim.ROLL_WIN:
            hist.pop(0)
        ubar = np.mean(hist, axis=0)
        sigma = (np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
                 if t > 0 and not np.isnan(UHAT[t - 1]).any()
                 else np.zeros(len(sim.SLICES)))
        X = np.array(sim.allocate("Static", st, t, u, U, UHAT, ubar, sigma,
                                  t < sim.N_TRAIN, rng, tuned), copy=True)
        if use_forecast:
            ref = UHAT[t] if not np.isnan(UHAT[t]).any() else u
        else:
            ref = ubar          # forecast-free: rolling mean stands in
        R = sim.risk_scores(u, ref, ubar, sigma)
        X, f = edge_correct(X, u, R)
        X = np.maximum(X, 0.0)
        m = sim.evaluate_interval("Static", X, u, None)
        if t >= sim.N_TRAIN:
            rows.append(m)
            fires += f
    return rows, fires


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

    r1, f1 = run_variant(U, UHAT, seed, tuned, use_forecast=False)
    rows.append(dict(w_E=w_e, headroom=h, method="Static+Edge", seed=seed,
                     **agg(r1)))
    r2, f2 = run_variant(U, UHAT, seed, tuned, use_forecast=True)
    rows.append(dict(w_E=w_e, headroom=h, method="Static+Edge-F", seed=seed,
                     **agg(r2)))

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    n = len(r1)
    logged["edge_fires_per_interval_nofc"] = f1 / max(n, 1)
    logged["edge_fires_per_interval_fc"] = f2 / max(n, 1)
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    print("EQUIVALENCE CHECK: ported edge block vs the simulator's own")
    compared, mismatches = equivalence_check()
    print(f"  intervals compared: {compared}, mismatching: {mismatches}")
    if mismatches:
        print("  FAIL: the ported edge-correction block does not reproduce")
        print("        the simulator. Everything downstream would be wrong.")
        return 2
    print("  OK: the port is bitwise identical to the simulator's block.\n")

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

    print("\nTRIPWIRE")
    f1 = [p["tuned"]["edge_fires_per_interval_nofc"] for p in params]
    f2 = [p["tuned"]["edge_fires_per_interval_fc"] for p in params]
    print(f"  edge step fires per interval: forecast-free median "
          f"{np.median(f1):.3f}, forecast-informed median {np.median(f2):.3f}")
    ok = True
    if max(f1) == 0 and max(f2) == 0:
        print("  FAIL: the edge step never fired. Static's allocation never")
        print("        leaves a latency-critical slice short, so there is")
        print("        nothing for the correction to do and the comparison")
        print("        is vacuous.")
        ok = False
    else:
        print("  OK: the edge step is active, so the comparison is live.")

    # Direction is a result, never a failure condition.
    byk = {}
    for r in rows:
        byk.setdefault(r["method"], []).append(r["sla_violation"])
    print("  mean SLA violation:")
    for m in ("Static Allocation", "Static+Edge", "Static+Edge-F",
              "HAO-PRA-U"):
        if m in byk:
            print(f"    {m:<22}{np.mean(byk[m]):.5f}")

    if a.validate:
        print("\nValidation complete.")
        return 0 if ok else 1

    os.makedirs(OUT_DIR, exist_ok=True)
    cp = os.path.join(OUT_DIR, "all_cells_static_edge.csv")
    jp = os.path.join(OUT_DIR, "tuned_params_static_edge.json")
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
