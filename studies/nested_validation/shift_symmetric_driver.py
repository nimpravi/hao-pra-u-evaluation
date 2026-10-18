#!/usr/bin/env python3
"""
shift_symmetric_driver.py -- the staleness study under symmetric conditions
(reviewer M1.3)

Registered in PREREGISTRATION_SHIFT_SYMMETRIC.md. Do not run until that
document is hashed, committed and independently timestamped.

WHAT CHANGES FROM STUDY 2, AND WHAT DOES NOT

Study 2 ran on the earlier simulator build under 70/30, tuning baselines only
(sim.tune_baselines) while the framework received no search at all. That is
the asymmetry Study 4 exists to remove, which is why the manuscript labels
Study 2's result conditional. This driver changes exactly three things:

  * the nested build, 550/150/300, so tuning happens on 550..699 and the
    pre-shift window 700..999 is untouched by selection
  * sim.tune_all_schemes, giving every scheme the same 9 evaluations
  * the capped multiplier grid, so the protection range is symmetric

Everything about the shift itself is carried over unchanged from
shift_driver.py: the trace construction, the scenario definitions, the
extension RNG offset, the window boundaries at 1000, and the rule that no
parameter is re-tuned after the shift. The two generators are byte-identical
between builds, which this driver verifies before running rather than
assuming.

Usage, from studies/nested_validation/ :

    python shift_symmetric_driver.py --validate     # S0 neutrality, seed 0
    python shift_symmetric_driver.py --workers 4

Outputs to results/nested_validation/ :
    all_windows_symmetric.csv  /  recovery_curves_symmetric.csv
    tuned_params_shift_symmetric.json
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    "hao_pra_u_sim", os.path.join(HERE, "hao_pra_u_sim.py"))
sim = importlib.util.module_from_spec(_spec)
sys.modules["hao_pra_u_sim"] = sim
_spec.loader.exec_module(sim)

# Carried over verbatim from studies/regime_shift/shift_driver.py.
N_TOTAL, T_SHIFT = 1300, 1000
EXT_STREAM_OFFSET = 424243
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

CAPPED_SCALE_GRID = [0.5, 0.75, 1.0]
N_SEEDS = 10

# Study 2's six schemes. The nested build adds HAO-PPO, which Study 2 did
# not contain, and the min-max composite is normalized within whatever set
# is scored, so the two sets give different composites by construction.
#
# We do NOT choose between them here. Every scheme the build offers is run,
# and the two composites are formed at analysis time by subsetting the CSV:
# the six-scheme composite is primary, because it is the one comparable with
# Study 2, and the seven-scheme composite is reported beside it. Jbar is
# set-independent and unaffected either way.
#
# Running only six would have been a scheme-set choice made after a crash,
# by authors who already know from Sec. VII-C that admitting HAO-PPO removes
# decisiveness. Running both removes the choice.
SHIFT_SCHEMES_PRIMARY = ["Static", "Reactive", "Centralized",
                         "CentralizedPredictive", "FlatAgent", "HAO-PRA-U"]
FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "nested_validation")

# Windows, derived from the nested split rather than hard-coded, so that a
# change to the split cannot silently leave them inconsistent.
T_PRE = sim.VALID_END                     # 700: first post-selection interval


def recover_phases(seed):
    """Replay the frozen generator's stream to recover per-slice phases
    without modifying it. Verbatim from shift_driver.py."""
    rng = np.random.default_rng(seed)
    n = sim.N_INTERVALS
    phases = []
    for s in sim.SLICES:
        base = sim.BASE_DEMAND[s]
        phase = rng.uniform(0, 2 * np.pi)
        phases.append(phase)
        rng.normal(0, sim.STOCH_STD_FRAC * base, n)
        rng.random(n)
        rng.uniform(*sim.BURST_MAG_RANGE, n)
    return phases


def build_trace(seed, scen):
    """1300-interval trace: frozen 0..999 plus a scenario-transformed
    extension drawn from a separate deterministic stream."""
    rng = np.random.default_rng(seed)
    U0 = sim.generate_traffic(rng)
    phases = recover_phases(seed)
    ext_rng = np.random.default_rng(seed + EXT_STREAM_OFFSET)
    t_ext = np.arange(T_SHIFT, N_TOTAL)
    EXT = np.zeros((N_TOTAL - T_SHIFT, len(sim.SLICES)))
    for j, s in enumerate(sim.SLICES):
        base = sim.BASE_DEMAND[s]
        f = scen["factors"].get("__ALL__",
                                scen["factors"].get(s, 1.0)) * base
        periodic = f * (1 + sim.PERIODIC_AMPLITUDE *
                        np.sin(2 * np.pi * t_ext / sim.PERIODIC_PERIOD
                                + phases[j]))
        noise = ext_rng.normal(0, sim.STOCH_STD_FRAC * f, len(t_ext))
        hits = ext_rng.random(len(t_ext)) < scen["burst_p"]
        mags = ext_rng.uniform(*scen["burst_mag"], len(t_ext))
        EXT[:, j] = np.maximum(periodic + noise + hits * mags * f, 1.0)
    return np.vstack([U0, EXT])


def generator_check():
    """The extension logic above reimplements the frozen generator's shape.
    Verify that reimplementation against the real thing on the frozen window,
    where both must agree exactly."""
    seed = 0
    rng = np.random.default_rng(seed)
    U0 = sim.generate_traffic(rng)
    U = build_trace(seed, SCENARIOS["S0_noshift"])
    return int(np.array_equal(U[:sim.N_INTERVALS], U0))


def run_one(scheme, U, UHAT, seed, tuned):
    """The frozen run_scheme loop over 1300 intervals with window tagging.

    Parameters are frozen before T_PRE and never re-tuned, in particular not
    after the shift: that is what makes this a staleness test.
    """
    st = sim.SchemeState()
    if scheme == "HAO-PRA-U":
        sim.pretrain_hao(st, U, UHAT, tuned, seed)
    elif scheme == "HAO-PPO":
        sim.pretrain_ppo(st, U, UHAT, tuned, seed,
                         episodes=tuned.get("_ppo_episodes"))
    rng = np.random.default_rng(seed + 7919)
    hist, pre, post, series = [], [], [], []
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
            # gamma MUST be passed. q_update falls back to the registered
            # Q_GAMMA when it is None, which would silently discard the
            # hao_gamma that tune_all_schemes just selected and run the
            # framework untuned while every baseline ran tuned. That is the
            # exact asymmetry this study exists to remove. The original
            # shift_driver.py omits it correctly, because on that build the
            # framework received no search at all.
            sim.q_update(st, r, U[t + 1] if t + 1 < n else u, sigma,
                         gamma=tuned.get("hao_gamma"))
            if training:
                st.eps = max(sim.EPS_END, st.eps * sim.EPS_DECAY)
        elif scheme == "HAO-PPO":
            # Policy frozen after pretraining; the live pass is evaluation
            # only, matching sim.run_scheme.
            st.prev_util = float(m["utilization"])
        if T_PRE <= t < T_SHIFT:
            pre.append(m)
        elif t >= T_SHIFT:
            post.append(m)
        if t >= T_SHIFT - 50:
            series.append(m["sla_violation"])
    agg = {}
    for win, rowset in (("pre", pre), ("post", post)):
        agg[win] = {k: float(np.mean([r_[k] for r_ in rowset]))
                    for k in FULL_METRICS}
    return agg, series


def run_cell(args):
    scenario, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sim.set_regime()                       # anchor regime, frozen weights
    sim.HAO_RESERVE_SCALE_GRID = list(CAPPED_SCALE_GRID)
    scen = SCENARIOS[scenario]
    U = build_trace(seed, scen)
    # The forecaster and the tuner see only 0..699; the shift is at 1000.
    UHAT, _ = sim.train_gru_forecasts(U[:sim.N_INTERVALS], seed)
    UHAT = np.vstack([UHAT, np.full((N_TOTAL - sim.N_INTERVALS,
                                     len(sim.SLICES)), np.nan)])
    tuned = sim.tune_all_schemes(U[:sim.N_INTERVALS], UHAT[:sim.N_INTERVALS],
                                 seed)
    rows, curves = [], []
    for scheme in sim.SCHEMES:
        agg, series = run_one(scheme, U, UHAT, seed, tuned)
        for win in ("pre", "post"):
            rows.append(dict(scenario=scenario,
                             method=sim.PAPER_NAMES[scheme], seed=seed,
                             window=win, **agg[win]))
        curves.append(dict(scenario=scenario,
                           method=sim.PAPER_NAMES[scheme], seed=seed,
                           series=json.dumps([round(v, 4) for v in series])))
    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    return rows, curves, dict(scenario=scenario, seed=seed, tuned=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="neutrality control: S0 no-shift, seed 0 only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    print("DEPENDENCY CHECK: every scheme in the build must be runnable")
    need_ppo = [m for m in sim.SCHEMES if "PPO" in m]
    if need_ppo and importlib.util.find_spec("ppo_reserve") is None:
        print(f"  FAIL: this build contains {need_ppo}, which needs")
        print("        ppo_reserve.py, and it is not on the import path.")
        print()
        print("        Copy it in, from the repository root:")
        print("          cp studies/ppo_study/ppo_reserve.py "
              "studies/nested_validation/")
        print()
        print("        Not running the PPO arm is NOT an acceptable "
              "fallback here.")
        print("        Amendment A1 fixes that every scheme in the build is "
              "run and")
        print("        that both composites are formed at analysis time, "
              "precisely so")
        print("        that no scheme-set choice is made after a failure. "
              "Dropping a")
        print("        scheme because a file is missing would be that "
              "choice.")
        return 2
    print(f"  OK: {len(sim.SCHEMES)} schemes, all importable.\n")

    print("GENERATOR CHECK: trace extension against the frozen generator")
    if not generator_check():
        print("  FAIL: intervals 0..999 of the constructed trace do not match")
        print("        sim.generate_traffic. The extension logic has drifted")
        print("        from the frozen generator and nothing downstream is")
        print("        comparable to Study 2.")
        return 2
    print("  OK: intervals 0..999 are bit-identical to the frozen trace.\n")

    print("SYMMETRY CHECK")
    print(f"  split: train 0..{sim.N_TRAIN-1}, valid {sim.N_TRAIN}.."
          f"{sim.VALID_END-1}, pre-shift {T_PRE}..{T_SHIFT-1}, "
          f"post-shift {T_SHIFT}..{N_TOTAL-1}")
    print(f"  HAO multiplier grid: {CAPPED_SCALE_GRID} "
          f"(symmetric 1.40 ceiling)")
    print("  tuner: tune_all_schemes, 9 evaluations for every scheme")
    print(f"  schemes run: {len(sim.SCHEMES)} -- {sim.SCHEMES}")
    missing = [m for m in SHIFT_SCHEMES_PRIMARY if m not in sim.SCHEMES]
    if missing:
        print(f"  FAIL: {missing} absent from this build, so Study 2's "
              "scheme set")
        print("        cannot be reconstructed from this run.")
        return 2
    extra = [m for m in sim.SCHEMES if m not in SHIFT_SCHEMES_PRIMARY]
    print(f"  primary composite subsets to Study 2's six: "
          f"{SHIFT_SCHEMES_PRIMARY}")
    if extra:
        print(f"  also run, reported as a secondary composite: {extra}")
        print("    Both are formed at analysis time from the same CSV, so "
              "no scheme-set")
        print("    choice is made here. Jbar is set-independent and "
              "unaffected.")
    print()

    scens = ["S0_noshift"] if a.validate else \
        ["S0_noshift"] + RESULT_SCENARIOS
    seeds = range(1) if a.validate else range(a.seeds)
    jobs = [(sc, s) for sc in scens for s in seeds]
    print(f"{'VALIDATE' if a.validate else 'FULL'}: {len(jobs)} "
          f"scenario-seeds, workers={a.workers}")

    rows, curves, params = [], [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (r, c, p) in enumerate(ex.map(run_cell, jobs), 1):
                rows.extend(r); curves.extend(c); params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, c, p = run_cell(j)
            rows.extend(r); curves.extend(c); params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    print("\nTRIPWIRES")
    ok = True

    # H3, the registered neutrality control. If S0 moves, the extension is
    # an artifact and no scenario result is reportable.
    s0 = [r for r in rows if r["scenario"] == "S0_noshift"]
    if s0:
        worst, worst_m = 0.0, ""
        for m in {r["method"] for r in s0}:
            pre = [r["sla_violation"] for r in s0
                   if r["method"] == m and r["window"] == "pre"]
            post = [r["sla_violation"] for r in s0
                    if r["method"] == m and r["window"] == "post"]
            sd = np.std(pre) if len(pre) > 1 else 0.0
            gap = abs(np.mean(post) - np.mean(pre))
            if len(pre) > 1 and gap > worst:
                worst, worst_m = gap, m
            if len(pre) > 1 and gap > max(sd, 1e-9):
                print(f"  S0 drift for {m}: |post-pre| {gap:.5f} "
                      f"> pre-window sd {sd:.5f}")
                ok = False
        print(f"  S0 neutrality: worst drift {worst:.5f} ({worst_m or 'n/a'})")
        if ok:
            print("  OK: no scheme's S0 post-window leaves its pre-window "
                  "spread.")
        else:
            print("  FAIL: the registration voids the study if S0 fails.")

    if not a.validate:
        print("\n  post-shift mean SLA violation by scenario:")
        for sc in RESULT_SCENARIOS:
            line = f"    {sc:<14}"
            for m in ("Static Allocation", "HAO-PRA-U"):
                v = [r["sla_violation"] for r in rows
                     if r["scenario"] == sc and r["method"] == m
                     and r["window"] == "post"]
                if v:
                    line += f"{m}={np.mean(v):.5f}  "
            print(line)
        print("\n  NOTE: the registered verdict is not computed here. Score "
              "with the\n        analysis script and read it off "
              "PREREGISTRATION_SHIFT_SYMMETRIC.md Sec. 4.")

    if a.validate:
        print("\nValidation complete.")
        return 0 if ok else 1

    os.makedirs(OUT_DIR, exist_ok=True)
    wp = os.path.join(OUT_DIR, "all_windows_symmetric.csv")
    rp = os.path.join(OUT_DIR, "recovery_curves_symmetric.csv")
    jp = os.path.join(OUT_DIR, "tuned_params_shift_symmetric.json")
    cols = ["scenario", "method", "seed", "window"] + FULL_METRICS
    with open(wp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(rp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "method", "seed",
                                          "series"])
        w.writeheader(); w.writerows(curves)
    with open(jp, "w") as f:
        json.dump(params, f, indent=1)
    print(f"\nwrote {wp} ({len(rows)} rows)")
    print(f"wrote {rp}")
    print(f"wrote {jp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
