#!/usr/bin/env python3
"""
robustness_driver.py -- two post-review robustness arms for Study 4.

Neither arm modifies the frozen simulator. Arm A narrows a registered grid
by assignment; arm B passes flags the simulator already reads from `tuned`.
The pinned SHA-256 of hao_pra_u_sim.py is unchanged by running this.

ARM A  --arm capped
    Study 4 re-run with the reserve multiplier grid narrowed from
    [0.5, 1.0, 1.5] to [0.5, 0.75, 1.0]. The framework's reserve factor is
    rho = RESERVE_ACTIONS[a] * scale, and RESERVE_ACTIONS tops out at 0.40,
    so the original grid let it reach rho = 0.60, an effective provisioning
    ceiling of 1.60 against the baselines' 1.40. Under the narrowed grid the
    ceiling is exactly 1.40 and the protection range is symmetric.
    Everything else, including the 9-evaluation budget (3 gammas x 3
    scales), is identical to Study 4.
    Answers: do the framework's two composite wins survive range symmetry?

ARM B  --arm constreserve
    Adds a seventh scheme, HAO-ConstReserve: the full hierarchy with the
    LEARNED reserve replaced by a single constant rho, selected per
    (regime, seed) on the validation window from a 9-point grid under the
    same 9-evaluation budget. No learning, but tuned.
    The Study 6 ablation compared a *tuned* learned reserve against an
    *untuned* fixed one (rho = 0.20), so it conflated the value of learning
    with the value of tuning. This arm separates them: if a tuned constant
    matches the learned reserve, "the reserve is a provisioning level" is
    shown directly rather than inferred.

Usage, from studies/nested_validation/ :

    python robustness_driver.py --arm capped       --validate
    python robustness_driver.py --arm capped       --workers 4
    python robustness_driver.py --arm constreserve --validate
    python robustness_driver.py --arm constreserve --workers 4

--validate runs the anchor regime at two seeds and prints a tripwire
report. Run it first; it takes a few minutes and catches a broken setup
before the full sweep.

Outputs, written next to the Study 4 results:
    all_cells_capped.csv        / tuned_params_capped.json
    all_cells_constreserve.csv  / tuned_params_constreserve.json
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
SIM_PATH = os.path.join(HERE, "hao_pra_u_sim.py")

_spec = importlib.util.spec_from_file_location("hao_pra_u_sim", SIM_PATH)
sim = importlib.util.module_from_spec(_spec)
sys.modules["hao_pra_u_sim"] = sim
_spec.loader.exec_module(sim)

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
N_SEEDS = 10
EXPECTED_BUDGET = 9
ANCHOR = (0.0188, 2.0)

# Arm A: narrowed multiplier grid. max(RESERVE_ACTIONS) * 1.0 = 0.40, so the
# effective provisioning ceiling is 1.40, matching the baselines' grid.
CAPPED_SCALE_GRID = [0.5, 0.75, 1.0]

# Arm B: 9-point constant-reserve grid, same cardinality as the baselines'
# 9-point margin grid and spanning the same provisioning range.
CONST_RHO_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Study 4's scheme set, which the composite's min-max normalization is
# taken over. sim.SCHEMES also contains HAO-PPO (Study 5); including it
# here would change the normalization and make these arms incomparable to
# Table VII, so it is excluded deliberately rather than by omission.
STUDY4_SCHEMES = ["Static", "Reactive", "Centralized",
                  "CentralizedPredictive", "FlatAgent", "HAO-PRA-U"]

FULL_METRICS = list(sim.METRICS) + ["reliability", "isolation", "overhead"]
OUT_DIR = os.path.join(HERE, "..", "..", "results", "nested_validation")


def preflight():
    """Fail loudly if the simulator is not the Study 4 one."""
    problems = []
    for name in ("VALID_START", "VALID_END", "HAO_GAMMA_GRID",
                 "HAO_RESERVE_SCALE_GRID", "RESERVE_ACTIONS",
                 "PRETRAIN_EPOCHS_TUNE", "ROLL_WIN", "EPS", "SLICES"):
        if not hasattr(sim, name):
            problems.append(f"missing {name}")
    if getattr(sim, "N_TRAIN", None) != 550:
        problems.append(f"N_TRAIN is {getattr(sim,'N_TRAIN',None)}, expected 550")
    if not getattr(sim, "ABLATION_TOGGLES_SUPPORTED", False):
        problems.append("ABLATION_TOGGLES_SUPPORTED is not True; arm B needs it")
    if getattr(sim, "RESERVE_ACTIONS", [None])[-1] != 0.40:
        problems.append("RESERVE_ACTIONS does not end at 0.40")
    if not hasattr(sim, "tune_all_schemes"):
        problems.append("tune_all_schemes missing (not the Study 4 simulator)")
    if problems:
        sys.exit("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))


def valid_score(scheme, tuned, U, UHAT, seed):
    """Mean reward over the validation window, replicating the scoring that
    tune_all_schemes uses internally so arm B's selection is comparable."""
    rng = np.random.default_rng(seed + 104729)
    st = sim.SchemeState()
    if scheme == "HAO-PRA-U":
        sim.pretrain_hao(st, U, UHAT, tuned, seed,
                         epochs=sim.PRETRAIN_EPOCHS_TUNE)
    hist, rs = [], []
    for t in range(sim.VALID_START, sim.VALID_END):
        u = U[t]
        hist.append(u)
        if len(hist) > sim.ROLL_WIN:
            hist.pop(0)
        ubar = np.mean(hist, axis=0)
        if t > 0 and not np.isnan(UHAT[t - 1]).any():
            sigma = np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + sim.EPS)
        else:
            sigma = np.zeros(len(sim.SLICES))
        X = sim.allocate(scheme, st, t, u, U, UHAT, ubar, sigma, False,
                         rng, tuned)
        rs.append(sim.reward_from_metrics(
            sim.evaluate_interval(scheme, X, u, None)))
    return float(np.mean(rs))


def tune_const_reserve(U, UHAT, seed, base_tuned):
    """Arm B: pick one constant rho on validation. Exactly 9 evaluations."""
    best, best_r, n = None, -np.inf, 0
    for rho in CONST_RHO_GRID:
        t = dict(base_tuned)
        t["abl_reserve"] = False       # disable the learned selection
        t["abl_fixed_rho"] = float(rho)
        r = valid_score("HAO-PRA-U", t, U, UHAT, seed)
        n += 1
        if r > best_r:
            best_r, best = r, float(rho)
    assert n == EXPECTED_BUDGET, f"const-reserve budget {n} != {EXPECTED_BUDGET}"
    return best, best_r


def run_cell(args):
    w_e, h, seed, arm = args
    sim.set_regime(w_e, h)
    if arm in ("capped", "constreserve_capped"):
        sim.HAO_RESERVE_SCALE_GRID = list(CAPPED_SCALE_GRID)

    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT, _ = sim.train_gru_forecasts(U, seed)
    tuned = sim.tune_all_schemes(U, UHAT, seed)

    budget = {k: v for k, v in tuned.get("_budget", {}).items()
              if k in STUDY4_SCHEMES or k in
              [sim.PAPER_NAMES.get(x, x) for x in STUDY4_SCHEMES]}
    bad = {k: v for k, v in budget.items() if v != EXPECTED_BUDGET}
    if budget and bad:
        raise RuntimeError(f"budget violated at ({w_e},{h},seed={seed}): {bad}")

    rows = []
    for scheme in STUDY4_SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        rows.append(dict(w_E=w_e, headroom=h,
                         method=sim.PAPER_NAMES[scheme], seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))

    extra = {}
    if arm in ("constreserve", "constreserve_capped"):
        rho_star, val_r = tune_const_reserve(U, UHAT, seed, tuned)
        ct = dict(tuned)
        ct["abl_reserve"] = False
        ct["abl_fixed_rho"] = rho_star
        res = sim.run_scheme("HAO-PRA-U", U, UHAT, seed, ct)
        rows.append(dict(w_E=w_e, headroom=h, method="HAO-ConstReserve",
                         seed=seed,
                         **{k: float(res[k]) for k in FULL_METRICS}))
        extra = {"const_rho": rho_star, "const_valid_reward": val_r}

    logged = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in tuned.items() if not k.startswith("_")}
    logged.update(extra)
    return rows, dict(w_E=w_e, headroom=h, seed=seed, tuned=logged)


def check_capped(params):
    """Arm A: no selected multiplier may exceed 1.0, and no realized rho may
    exceed 0.40. A violation means the narrowed grid did not take effect."""
    bad = [p for p in params
           if float(p["tuned"].get("hao_reserve_scale", 1.0)) > 1.0 + 1e-9]
    if bad:
        print(f"  FAIL: {len(bad)} selections above scale 1.0; the narrowed "
              f"grid did not take effect.")
        return False
    sel = [float(p["tuned"].get("hao_reserve_scale", 1.0)) for p in params]
    top = sum(1 for s in sel if abs(s - 1.0) < 1e-9)
    print(f"  OK: all {len(sel)} selections <= 1.0 "
          f"(effective ceiling 1.40, symmetric with the baselines).")
    print(f"  {top}/{len(sel)} selections at the new top of grid "
          f"(was 114/200 at the old top).")
    return True


def check_const(params):
    rhos = [p["tuned"].get("const_rho") for p in params
            if "const_rho" in p["tuned"]]
    if not rhos:
        print("  FAIL: no constant reserve was selected in any cell.")
        return False
    bad = [r for r in rhos if round(float(r), 4) not in CONST_RHO_GRID]
    if bad:
        print(f"  FAIL: {len(bad)} selections outside the registered grid.")
        return False
    at_top = sum(1 for r in rhos if abs(float(r) - 0.40) < 1e-9)
    print(f"  OK: {len(rhos)} constants selected, all inside the 9-point grid.")
    print(f"  {at_top}/{len(rhos)} at the top of the grid (0.40).")
    print(f"  distribution: " + ", ".join(
        f"{g}:{sum(1 for r in rhos if abs(float(r)-g)<1e-9)}"
        for g in CONST_RHO_GRID))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["capped", "constreserve",
                             "constreserve_capped"])
    ap.add_argument("--validate", action="store_true",
                    help="anchor regime, 2 seeds, tripwire only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    preflight()
    print(f"Simulator preflight OK. Arm: {a.arm}")
    if a.arm in ("capped", "constreserve_capped"):
        print(f"  multiplier grid {sim.HAO_RESERVE_SCALE_GRID} -> "
              f"{CAPPED_SCALE_GRID}")
        print(f"  max rho {max(sim.RESERVE_ACTIONS)*1.5:.2f} -> "
              f"{max(sim.RESERVE_ACTIONS)*1.0:.2f}  (ceiling 1.60 -> 1.40)")
    if a.arm in ("constreserve", "constreserve_capped"):
        print(f"  constant-reserve grid {CONST_RHO_GRID} (9 evaluations)")

    if a.validate:
        cells = [ANCHOR]
        seeds = range(2)
        tag = "VALIDATE"
    else:
        cells = [(w, h) for w in W_E_GRID for h in H_GRID]
        seeds = range(a.seeds)
        tag = "FULL"
    jobs = [(w, h, s, a.arm) for (w, h) in cells for s in seeds]
    print(f"{tag}: {len(jobs)} cells x seeds, workers={a.workers}")

    all_rows, all_params = [], []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (rows, p) in enumerate(ex.map(run_cell, jobs), 1):
                all_rows.extend(rows)
                all_params.append(p)
                print(f"  {i}/{len(jobs)} done", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows, p = run_cell(j)
            all_rows.extend(rows)
            all_params.append(p)
            print(f"  {i}/{len(jobs)} done", flush=True)

    print("\nTRIPWIRE")
    if a.arm == "capped":
        ok = check_capped(all_params)
    elif a.arm == "constreserve":
        ok = check_const(all_params)
    else:
        ok = check_capped(all_params) and check_const(all_params)

    if a.validate:
        print("\nValidation complete. If the tripwire passed, run the full "
              "sweep with --workers N.")
        return 0 if ok else 1

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"all_cells_{a.arm}.csv")
    json_path = os.path.join(OUT_DIR, f"tuned_params_{a.arm}.json")
    cols = ["w_E", "headroom", "method", "seed"] + FULL_METRICS
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(all_rows)
    with open(json_path, "w") as f:
        json.dump(all_params, f, indent=1)
    print(f"\nwrote {csv_path} ({len(all_rows)} rows)")
    print(f"wrote {json_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
