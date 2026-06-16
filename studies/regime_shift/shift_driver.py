#!/usr/bin/env python3
"""
Regime-shift study driver (SHIFT_STUDY_PREREGISTRATION.md).

The simulator (hao_pra_u_sim.py, DESIGN_FREEZE_v2) is imported UNCHANGED;
this driver only composes its frozen primitives (generate_traffic,
train_gru_forecasts, tune_baselines, allocate, evaluate_interval,
reward_from_metrics, q_update, pretrain_hao).

Trace construction guarantees neutrality: intervals 0-999 are produced by
the frozen generate_traffic with the same seed (bit-identical to Phase B);
the 300-interval extension uses a separate deterministic stream shared
across scenarios, with the scenario transform applied from t = 1000.

Usage:
    python3 shift_driver.py --workers 3              # full study (4 x 10)
    python3 shift_driver.py --validate               # S0 no-shift, seed 0
Outputs:
    shift_results/all_windows.csv     (scenario x method x seed x window)
    shift_results/recovery_curves.csv (per-interval violations, t>=950)
Then run:  python3 analyze_shift.py
"""
import argparse, csv, json, os
import numpy as np
import importlib.util

SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hao_pra_u_sim.py")
N_TOTAL, T_SHIFT, T_PRE = 1300, 1000, 700   # windows: pre 700-999, post 1000-1299
EXT_STREAM_OFFSET = 424243                   # extension rng = seed + this

# Registered scenario set (S0 is the neutrality control, not a result cell)
SCENARIOS = {
    "S0_noshift":  dict(factors={}, burst_p=0.05, burst_mag=(0.30, 0.60)),
    "S1a_load15":  dict(factors={"__ALL__": 1.15}, burst_p=0.05,
                        burst_mag=(0.30, 0.60)),
    "S1b_load30":  dict(factors={"__ALL__": 1.30}, burst_p=0.05,
                        burst_mag=(0.30, 0.60)),
    "S2_mixshift": dict(factors={"URLLC": 1.40, "XR": 1.40, "eMBB": 0.80},
                        burst_p=0.05, burst_mag=(0.30, 0.60)),
    "S3_bursts":   dict(factors={}, burst_p=0.10, burst_mag=(0.40, 0.80)),
}
RESULT_SCENARIOS = ["S1a_load15", "S1b_load30", "S2_mixshift", "S3_bursts"]


def load_sim():
    spec = importlib.util.spec_from_file_location("sim", SIM_PATH)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim


def recover_phases(sim, seed):
    """Replay the frozen generator's stream to recover per-slice phases
    without modifying the frozen code."""
    rng = np.random.default_rng(seed)
    n = sim.N_INTERVALS
    phases = []
    for s in sim.SLICES:
        base = sim.BASE_DEMAND[s]
        phases.append(rng.uniform(0, 2 * np.pi))
        rng.normal(0, sim.STOCH_STD_FRAC * base, n)
        rng.random(n)
        rng.uniform(*sim.BURST_MAG_RANGE, n)
    return phases


def build_trace(sim, seed, scen):
    """1300-interval trace: frozen 0-999 + scenario-transformed extension."""
    U0 = sim.generate_traffic(np.random.default_rng(seed))     # bit-exact
    phases = recover_phases(sim, seed)
    rext = np.random.default_rng(seed + EXT_STREAM_OFFSET)
    t_ext = np.arange(T_SHIFT, N_TOTAL)
    EXT = np.zeros((N_TOTAL - T_SHIFT, len(sim.SLICES)))
    for j, s in enumerate(sim.SLICES):
        # shared extension randoms (same draws for every scenario)
        z = rext.normal(0, 1, len(t_ext))
        bu = rext.random(len(t_ext))
        mu = rext.uniform(0, 1, len(t_ext))
        f = scen["factors"].get(s, scen["factors"].get("__ALL__", 1.0))
        base = sim.BASE_DEMAND[s] * f
        periodic = base * (1 + sim.PERIODIC_AMPLITUDE *
                           np.sin(2 * np.pi * t_ext / sim.PERIODIC_PERIOD
                                  + phases[j]))
        noise = z * sim.STOCH_STD_FRAC * base
        lo, hi = scen["burst_mag"]
        bursts = (bu < scen["burst_p"]) * (lo + (hi - lo) * mu) * base
        EXT[:, j] = np.maximum(periodic + noise + bursts, 1.0)
    return np.vstack([U0, EXT])


def run_one(sim, scheme, U, UHAT, seed, tuned):
    """Replicates the frozen run_scheme loop over the 1300-interval trace
    with window tagging. Identical state evolution to v2 for t < 1000."""
    st = sim.SchemeState()
    if scheme == "HAO-PRA-U":
        sim.pretrain_hao(st, U, UHAT, tuned, seed)
    rng = np.random.default_rng(seed + 7919)
    hist, pre, post, series = [], [], [], []
    n = len(U)
    for t in range(n):
        u = U[t]
        hist.append(u)
        if len(hist) > sim.ROLL_WIN:
            hist.pop(0)
        ubar = np.mean(hist, axis=0)
        if t > 0 and not np.isnan(UHAT[t - 1]).any():
            sigma = np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
        else:
            sigma = np.zeros(len(sim.SLICES))
        training = t < sim.N_TRAIN
        X = sim.allocate(scheme, st, t, u, U, UHAT, ubar, sigma, training,
                         rng, tuned)
        m = sim.evaluate_interval(scheme, X, u, None)
        r = sim.reward_from_metrics(m)
        if scheme == "HAO-PRA-U":
            sim.q_update(st, r, U[t + 1] if t + 1 < n else u, sigma)
            if training:
                st.eps = max(sim.EPS_END, st.eps * sim.EPS_DECAY)
        if T_PRE <= t < T_SHIFT:
            pre.append(m)
        elif t >= T_SHIFT:
            post.append(m)
        if t >= 950:
            series.append(m["sla_violation"])
    keys = sim.METRICS + ["reliability", "isolation", "overhead"]
    agg = {}
    for win, rows in (("pre", pre), ("post", post)):
        agg[win] = {k: float(np.mean([r_[k] for r_ in rows])) for k in keys}
    return agg, series


def task(args):
    scenario, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sim = load_sim()
    sim.set_regime()                       # anchor regime, frozen weights
    scen = SCENARIOS[scenario]
    U = build_trace(sim, seed, scen)
    UHAT, _ = sim.train_gru_forecasts(U, seed)   # trains on 0-699 only
    tuned = sim.tune_baselines(U, UHAT, seed)    # tunes on 0-699 only
    rows, curves = [], []
    for scheme in sim.SCHEMES:
        agg, series = run_one(sim, scheme, U, UHAT, seed, tuned)
        for win in ("pre", "post"):
            rows.append(dict(scenario=scenario, method=sim.PAPER_NAMES[
                scheme], seed=seed, window=win, **agg[win]))
        curves.append(dict(scenario=scenario,
                           method=sim.PAPER_NAMES[scheme], seed=seed,
                           series=json.dumps([round(v, 4)
                                              for v in series])))
    return rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()))
    ap.add_argument("--validate", action="store_true",
                    help="neutrality control: S0 no-shift, seed 0 only")
    args = ap.parse_args()

    if args.validate:
        tasks = [("S0_noshift", 0)]
    else:
        tasks = [(sc, sd) for sc in RESULT_SCENARIOS for sd in range(10)]

    os.makedirs("shift_results", exist_ok=True)
    all_rows, all_curves = [], []
    if args.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (rows, curves) in enumerate(ex.map(task, tasks)):
                all_rows += rows
                all_curves += curves
                print(f"done {i+1}/{len(tasks)}", flush=True)
    else:
        for i, t in enumerate(tasks):
            rows, curves = task(t)
            all_rows += rows
            all_curves += curves
            print(f"done {i+1}/{len(tasks)}: {t}", flush=True)

    with open("shift_results/all_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open("shift_results/recovery_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "method", "seed",
                                          "series"])
        w.writeheader()
        w.writerows(all_curves)
    print("wrote shift_results/all_windows.csv")


if __name__ == "__main__":
    main()
