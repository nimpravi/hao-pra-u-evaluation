#!/usr/bin/env python3
"""
edge_audit_driver.py -- the two accounting questions about Static+Edge
(reviewer M1.4 and M2)

TWO QUESTIONS, both about cost accounting rather than mechanism.

M2, donor cost. Sec. VII-G bounds the harm edge correction can do to the
slices it takes capacity from, but does not measure it. The bound is loose
because the aggregate metrics cannot see per-slice effects: the swap
conserves each domain's total, so utilization, fragmentation and isolation
cannot move by construction, and SLA violation is reported as a mean over
slices, where a donor regression can hide behind a larger beneficiary gain.

  It turns out no simulator change is needed. evaluate_interval already
  returns `viol_per_slice`; every driver in this study has simply been
  discarding it, because METRICS does not list it. This driver keeps it.

M1.4, control delay. Static+Edge is evaluated under the "Static" label and so
inherits Static's 0.2 ms synchronous control delay, where the hierarchy is
charged 0.8 ms for a step of the same kind. Edge correction acts partly
through the latency term, which carries alpha2 = 0.188, so the margin may not
survive honest delay accounting.

  Also no simulator change. CONTROL_DELAY is a module-level dict; this driver
  rebinds one entry at run time inside the worker process. The frozen file on
  disk is untouched and its digest is unchanged. The override is applied for
  one arm only and asserted back afterwards.

ARMS
  Static           0.2 ms, the tuned baseline
  Static+Edge      0.2 ms, exactly as Sec. VII-G reports it
  Static+Edge@0.8  0.8 ms, the hierarchy's delay charged to the edge step
  HAO-PRA-U        for context, unchanged

Usage, from studies/nested_validation/ :

    python edge_audit_driver.py --validate
    python edge_audit_driver.py --workers 4

Outputs to results/nested_validation/ :
    all_cells_edge_audit.csv            aggregate metrics, four arms
    per_slice_edge_audit.csv            per-slice violation rates
    tuned_params_edge_audit.json
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
CAPPED_SCALE_GRID = [0.5, 0.75, 1.0]
STUDY4_SCHEMES = ["Static", "HAO-PRA-U"]
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "nested_validation")

HIER_DELAY = 0.8          # what the hierarchy is charged, from CONTROL_DELAY
LAT_CRIT = ("URLLC", "XR", "IIoT")


def edge_correct(X, u, R):
    """Verbatim port of the edge-correction block in sim.allocate, identical
    to the copy in static_edge_driver.py. Modified in place and returned."""
    nS = len(sim.SLICES)
    edge = 1
    lat_crit = [sim.SLICES.index(s) for s in LAT_CRIT]
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
    """Rebuild HAO-PRA-U's allocation with the ported block and require it to
    match the simulator bitwise. Identical in construction to the check in
    static_edge_driver.py: both states are pretrained under their own tuned
    dict, each candidate interval gets identical fresh randomness, and only
    the 120 test-window intervals are compared.
    """
    sim.set_regime(*ANCHOR)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)
    tuned = sim.tune_all_schemes(U, UHAT, seed)

    st_a, st_b = sim.SchemeState(), sim.SchemeState()
    t_on = dict(tuned); t_on["abl_edge"] = True
    t_off = dict(tuned); t_off["abl_edge"] = False
    sim.pretrain_hao(st_a, U, UHAT, t_on, seed)
    sim.pretrain_hao(st_b, U, UHAT, t_off, seed)

    mismatches = compared = 0
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


def run_edge_arm(U, UHAT, seed, tuned, delay):
    """Static + edge correction, forecast-free risk, at a chosen control delay.

    The delay is charged by rebinding sim.CONTROL_DELAY["Static"] for the
    duration of this call. The frozen file is not touched; the original value
    is restored and asserted before returning.
    """
    original = sim.CONTROL_DELAY["Static"]
    sim.CONTROL_DELAY["Static"] = float(delay)
    try:
        st = sim.SchemeState()
        rng = np.random.default_rng(seed + 7919)
        hist, rows, pslice, fires = [], [], [], 0
        for t in range(len(U)):
            u = U[t]
            hist.append(u)
            if len(hist) > sim.ROLL_WIN:
                hist.pop(0)
            ubar = np.mean(hist, axis=0)
            sigma = (np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
                     if t > 0 and not np.isnan(UHAT[t - 1]).any()
                     else np.zeros(len(sim.SLICES)))
            X = np.array(sim.allocate("Static", st, t, u, U, UHAT, ubar,
                                      sigma, t < sim.N_TRAIN, rng, tuned),
                         copy=True)
            R = sim.risk_scores(u, ubar, ubar, sigma)
            X, f = edge_correct(X, u, R)
            X = np.maximum(X, 0.0)
            m = sim.evaluate_interval("Static", X, u, None)
            if t >= sim.N_TRAIN:
                rows.append(m)
                pslice.append(np.asarray(m["viol_per_slice"], dtype=float))
                fires += f
    finally:
        sim.CONTROL_DELAY["Static"] = original
    assert sim.CONTROL_DELAY["Static"] == original, \
        "control delay was not restored; downstream arms would be corrupted"
    return rows, np.mean(pslice, axis=0), fires


def run_plain_arm(U, UHAT, seed, tuned, scheme):
    """A stock scheme through the simulator's own loop, keeping per-slice
    violations that run_scheme discards."""
    st = sim.SchemeState()
    if scheme == "HAO-PRA-U":
        sim.pretrain_hao(st, U, UHAT, tuned, seed)
    rng = np.random.default_rng(seed + 7919)
    hist, rows, pslice = [], [], []
    n = len(U)
    for t in range(n):
        u = U[t]
        hist.append(u)
        if len(hist) > sim.ROLL_WIN:
            hist.pop(0)
        ubar = np.mean(hist, axis=0)
        sigma = (np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
                 if t > 0 and not np.isnan(UHAT[t - 1]).any()
                 else np.zeros(len(sim.SLICES)))
        training = t < sim.N_TRAIN
        X = sim.allocate(scheme, st, t, u, U, UHAT, ubar, sigma, training,
                         rng, tuned)
        m = sim.evaluate_interval(scheme, X, u, None)
        if scheme == "HAO-PRA-U":
            r = sim.reward_from_metrics(m)
            sim.q_update(st, r, U[t + 1] if t + 1 < n else u, sigma)
            if training:
                st.eps = max(sim.EPS_END, st.eps * sim.EPS_DECAY)
        if t >= sim.N_TRAIN:
            rows.append(m)
            pslice.append(np.asarray(m["viol_per_slice"], dtype=float))
    return rows, np.mean(pslice, axis=0)


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

    rows, pslice = [], []

    def record(method, r, ps):
        rows.append(dict(w_E=w_e, headroom=h, method=method, seed=seed,
                         **agg(r)))
        pslice.append(dict(w_E=w_e, headroom=h, method=method, seed=seed,
                           **{s: float(v) for s, v in zip(sim.SLICES, ps)}))

    for scheme in STUDY4_SCHEMES:
        r, ps = run_plain_arm(U, UHAT, seed, tuned, scheme)
        record(sim.PAPER_NAMES[scheme], r, ps)

    r2, ps2, f2 = run_edge_arm(U, UHAT, seed, tuned, delay=0.2)
    record("Static+Edge", r2, ps2)
    r8, ps8, f8 = run_edge_arm(U, UHAT, seed, tuned, delay=HIER_DELAY)
    record("Static+Edge@0.8ms", r8, ps8)

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    logged["edge_fires_per_interval"] = f2 / max(len(r2), 1)
    logged["delay_02_latency"] = float(np.mean([m["latency"] for m in r2]))
    logged["delay_08_latency"] = float(np.mean([m["latency"] for m in r8]))
    return rows, pslice, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    print("EQUIVALENCE CHECK: ported edge block vs the simulator's own")
    compared, mismatches = equivalence_check()
    print(f"  intervals compared: {compared}, mismatching: {mismatches}")
    if mismatches or compared == 0:
        print("  FAIL: the ported block does not reproduce the simulator.")
        return 2
    print("  OK: the port is bitwise identical.\n")

    print("DELAY OVERRIDE CHECK")
    base = sim.CONTROL_DELAY["Static"]
    print(f"  CONTROL_DELAY['Static'] = {base} before; "
          f"hierarchy is charged {sim.CONTROL_DELAY['HAO-PRA-U']}")
    if base != 0.2:
        print("  FAIL: the frozen table does not say 0.2 ms for Static.")
        return 2
    print("  OK: override will raise it to 0.8 for one arm only.\n")

    cells = [ANCHOR] if a.validate else \
        [(w, h) for w in W_E_GRID for h in H_GRID]
    seeds = range(2) if a.validate else range(a.seeds)
    jobs = [(w, h, s) for (w, h) in cells for s in seeds]
    print(f"{'VALIDATE' if a.validate else 'FULL'}: {len(jobs)} cell-seeds, "
          f"workers={a.workers}")

    rows, pslice, params = [], [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (r, ps, p) in enumerate(ex.map(run_cell, jobs), 1):
                rows.extend(r); pslice.extend(ps); params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, ps, p = run_cell(j)
            rows.extend(r); pslice.extend(ps); params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    print("\nTRIPWIRES")
    ok = True

    # 1. The delay override must actually move latency, by about 0.6 ms.
    d02 = np.mean([p["tuned"]["delay_02_latency"] for p in params])
    d08 = np.mean([p["tuned"]["delay_08_latency"] for p in params])
    print(f"  mean latency at 0.2 ms: {d02:.5f}; at 0.8 ms: {d08:.5f}; "
          f"difference {d08 - d02:+.5f}")
    if abs((d08 - d02) - 0.6) > 0.05:
        print("  FAIL: the override did not shift latency by ~0.6 ms, so it")
        print("        did not take effect the way this driver assumes.")
        ok = False
    else:
        print("  OK: the override is live.")

    # 2. Per-slice violations must average to the reported aggregate.
    byk = {}
    for r in rows:
        byk.setdefault(r["method"], []).append(r["sla_violation"])
    psk = {}
    for r in pslice:
        psk.setdefault(r["method"], []).append(
            np.mean([r[s] for s in sim.SLICES]))
    worst = 0.0
    for m in byk:
        worst = max(worst, abs(np.mean(byk[m]) - np.mean(psk[m])))
    print(f"  max |mean(per-slice) - aggregate SLA violation|: {worst:.2e}")
    if worst > 1e-9:
        print("  FAIL: per-slice violations do not reconstruct the aggregate,")
        print("        so one of the two is not what it claims to be.")
        ok = False
    else:
        print("  OK: per-slice decomposition reconstructs the aggregate.")

    print("\n  per-slice violation rate (donors are eMBB and mMTC):")
    hdr = "".join(f"{s:>9}" for s in sim.SLICES)
    print(f"    {'scheme':<20}{hdr}")
    for m in ("Static Allocation", "Static+Edge", "Static+Edge@0.8ms",
              "HAO-PRA-U"):
        sub = [r for r in pslice if r["method"] == m]
        if sub:
            vals = "".join(f"{np.mean([r[s] for r in sub]):>9.5f}"
                           for s in sim.SLICES)
            print(f"    {m:<20}{vals}")
    print("\n  NOTE: two anchor seeds. Do not read a direction off this;")
    print("        score the sweep with the analysis script.")

    if a.validate:
        print("\nValidation complete.")
        return 0 if ok else 1

    os.makedirs(OUT_DIR, exist_ok=True)
    cp = os.path.join(OUT_DIR, "all_cells_edge_audit.csv")
    pp = os.path.join(OUT_DIR, "per_slice_edge_audit.csv")
    jp = os.path.join(OUT_DIR, "tuned_params_edge_audit.json")
    cols = ["w_E", "headroom", "method", "seed"] + FULL_METRICS
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    pcols = ["w_E", "headroom", "method", "seed"] + list(sim.SLICES)
    with open(pp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pcols); w.writeheader(); w.writerows(pslice)
    with open(jp, "w") as f:
        json.dump(params, f, indent=1)
    print(f"\nwrote {cp} ({len(rows)} rows)")
    print(f"wrote {pp} ({len(pslice)} rows)")
    print(f"wrote {jp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
