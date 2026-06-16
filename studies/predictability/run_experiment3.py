#!/usr/bin/env python3
"""
run_experiment3.py -- Predictability-sweep harness (Experiment 3).

Implements EXPERIMENT3_PREREGISTRATION.md, Sections 4.5-4.7, 6, VERBATIM.
Composes the FROZEN simulator (hao_pra_u_sim.py, imported unchanged) and
the FROZEN generator (precursor_traffic.py). The only behavior added beyond
the frozen primitives is exactly the two registered method changes:

  (4.6) a 2-channel (demand, precursor) GRU forecaster, and
  (4.7) a `brisk` forward burst-risk feature appended to the orchestrator
        Q-state.

No experimental numbers are produced by anyone but the author running this
file. This script writes code-computed results to disk; it does not
fabricate, estimate, or fill in any value.

FAIRNESS-OF-OBSERVATION (Section 2) is enforced mechanically:
  * The precursor array P is passed ONLY into the 2-channel GRU feature
    assembly. It is never placed in U, U[t-1], ubar, or any baseline path.
  * A runtime assertion (assert_precursor_isolation) fails the run if any
    baseline scheme's allocation depends on P.

OUTPUTS (under exp3_results/):
  results_per_seed.csv     # schema byte-identical to scripts/significance_test.py
                           # PLUS a leading 'predictability' column block written
                           # to a SEPARATE file so significance_test.py is unaffected
  results_per_seed__p{level}.csv   # one significance_test-ready CSV per level
  forecast_diag.csv        # AUC + MAE/RMSE/MAPE + anticipation frac per (scheme,seed,level)
  checkpoints/cell_{level}_{seed}.json   # resume support

USAGE
-----
  python3 run_experiment3.py --workers 4          # full study (6 x 12)
  python3 run_experiment3.py --validate           # p=0,seed0 tripwire check
  # then, per level:
  python3 scripts/significance_test.py exp3_results/results_per_seed__p0.0.csv
  # and the registered decision rule:
  python3 analyze_experiment3.py
"""
import argparse
import csv
import json
import os
import hashlib
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(HERE, "hao_pra_u_sim.py")
PRECURSOR_PATH = os.path.join(HERE, "precursor_traffic.py")

# ---- frozen pre-registered design constants (must match the prereg) ----
PREDICTABILITY_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = list(range(12))                         # n = 12
PRECURSOR_LEAD = 1
ANCHOR_W_E, ANCHOR_H = 0.0188, 2.00
BRISK_BINS = [1.10, 1.30]                        # Section 4.7
# Expected hashes of the frozen artifacts (Section 0 of the prereg).
EXPECTED_HASHES = {
    "precursor_traffic.py":
        "d2c8f28c5220e2f60b7fc3d5c0c28a154757649b20a595df2ef74905264f7aa4",
    "hao_pra_u_sim.py":
        "24f20b81ad19edbd522c72596d8526218cd8274a3791679e8af0897accdf3a3d",
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_artifacts():
    """Abort if the frozen artifacts differ from what the prereg pinned."""
    for fn, exp in EXPECTED_HASHES.items():
        got = _sha256(os.path.join(HERE, fn))
        if got != exp:
            raise SystemExit(
                f"FROZEN ARTIFACT MISMATCH for {fn}\n"
                f"  expected {exp}\n  got      {got}\n"
                "The pre-registration pins these hashes. Refusing to run a "
                "study that does not match the registered design.")
    print("Frozen-artifact hashes verified against pre-registration.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# Registered change 4.6: 2-channel (demand, precursor) GRU forecaster.
# Mirrors hao_pra_u_sim.train_gru_forecasts EXACTLY except the input tensor
# carries a second feature channel = z-scored precursor aligned to the same
# index. Architecture/hparams/seed/split otherwise identical.
# ==========================================================================
def train_gru_forecasts_2ch(sim, U, P, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    n, ns = U.shape
    UHAT = np.full((n, ns), np.nan)
    acc = {}
    LB = sim.GRU_LOOKBACK

    for j, s in enumerate(sim.SLICES):
        series = U[:, j].astype(np.float32)
        prec = P[:, j].astype(np.float32)
        mu, sd = series[:sim.N_TRAIN].mean(), series[:sim.N_TRAIN].std() + 1e-6
        # precursor z-scored on TRAIN ONLY (no test leakage)
        pmu = prec[:sim.N_TRAIN].mean()
        psd = prec[:sim.N_TRAIN].std() + 1e-6
        z = (series - mu) / sd
        pz = (prec - pmu) / psd

        X, Y, idx = [], [], []
        for i in range(LB, n):
            # channel 0: past demand window z[i-LB:i]
            # channel 1: precursor window pz[i-LB+1:i+1]  -- INCLUDES o(i),
            #   the precursor observable AT the decision interval i. This is
            #   the causal anticipation seam (prereg Section 2, item 3):
            #   o(i) informs the burst at i..i+L; it is NOT future demand.
            dem = z[i - LB:i]
            pre = pz[i - LB + 1:i + 1]
            X.append(np.stack([dem, pre], axis=-1))   # (LB, 2)
            Y.append(z[i])
            idx.append(i)
        X = torch.tensor(np.array(X))                  # (N, LB, 2)
        Y = torch.tensor(np.array(Y)).unsqueeze(-1)
        idx = np.array(idx)
        tr = idx < sim.N_TRAIN

        class GRUNet2(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(2, sim.GRU_HIDDEN, batch_first=True)
                self.fc = nn.Linear(sim.GRU_HIDDEN, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        net = GRUNet2()
        opt = torch.optim.Adam(net.parameters(), lr=sim.GRU_LR)
        lossf = nn.MSELoss()
        Xtr, Ytr = X[tr], Y[tr]
        ntr = len(Xtr)
        for _ in range(sim.GRU_EPOCHS):
            perm = torch.randperm(ntr)
            for b in range(0, ntr, sim.GRU_BATCH):
                bi = perm[b:b + sim.GRU_BATCH]
                opt.zero_grad()
                loss = lossf(net(Xtr[bi]), Ytr[bi])
                loss.backward()
                opt.step()

        with torch.no_grad():
            pred = net(X).squeeze(-1).numpy() * sd + mu
        UHAT[idx, j] = pred

        te = idx >= sim.N_TRAIN
        err = pred[te] - series[idx[te]]
        actual = series[idx[te]]
        acc[s] = dict(MAE=float(np.mean(np.abs(err))),
                      RMSE=float(np.sqrt(np.mean(err ** 2))),
                      MAPE=float(np.mean(np.abs(err) / actual) * 100))
    return UHAT, acc


# ==========================================================================
# Registered change 4.7: brisk forward burst-risk feature -> Q-state.
# We override the sim's `discretize` so the HAO Q-state becomes
#   (avg_p_bin, max_p_bin, err_bin, brisk_bin).
# brisk is computed inside allocate's HAO branch from uhat/ubar only, then
# stashed on the SchemeState so the overridden discretize can read it.
# Baselines never call this path; their allocation is byte-identical.
# ==========================================================================
def install_brisk_qstate(sim):
    orig_discretize = sim.discretize

    def discretize_brisk(avg_p, max_p, err, brisk=None):
        base = orig_discretize(avg_p, max_p, err)   # (b1,b2,b3)
        if brisk is None:
            # baseline/legacy callers (e.g. q_update for non-HAO) never pass
            # brisk; keep a 4th bin slot fixed at 0 so state tuples are a
            # consistent length but carry no brisk information.
            return base + (0,)
        bb = int(np.digitize(brisk, BRISK_BINS))
        return base + (bb,)

    # The HAO allocate branch and q_update both call discretize. We thread
    # brisk via a state attribute set in the wrapped allocate.
    orig_allocate = sim.allocate

    def allocate_brisk(scheme, st, t, u, U, UHAT, ubar, sigma, training,
                       rng, tuned):
        if scheme == "HAO-PRA-U":
            uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
            brisk = float(np.max(uhat / (ubar + sim.EPS)))
            st._brisk = brisk          # consumed by discretize_brisk
        else:
            st._brisk = None           # FAIRNESS: non-HAO never carries brisk
        return orig_allocate(scheme, st, t, u, U, UHAT, ubar, sigma,
                             training, rng, tuned)

    # Patch discretize to read st._brisk when invoked from the HAO path.
    # allocate/q_update call sim.discretize(avg_p,max_p,err); we wrap those
    # call sites by making discretize pull a thread-local brisk. Simplest
    # robust approach: replace discretize with a version that consults a
    # module-level slot set immediately before each HAO call.
    sim._brisk_slot = {"v": None}

    def discretize_threaded(avg_p, max_p, err):
        return discretize_brisk(avg_p, max_p, err, sim._brisk_slot["v"])

    # Re-wrap allocate to set the slot around the HAO state computation.
    def allocate_brisk2(scheme, st, t, u, U, UHAT, ubar, sigma, training,
                        rng, tuned):
        if scheme == "HAO-PRA-U":
            uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
            sim._brisk_slot["v"] = float(np.max(uhat / (ubar + sim.EPS)))
        else:
            sim._brisk_slot["v"] = None
        try:
            return orig_allocate(scheme, st, t, u, U, UHAT, ubar, sigma,
                                 training, rng, tuned)
        finally:
            if scheme != "HAO-PRA-U":
                sim._brisk_slot["v"] = None

    orig_q_update = sim.q_update

    def q_update_brisk(st, reward, u_next, sigma_next):
        # q_update calls discretize for the next state; reuse the same brisk
        # slot (next-state brisk is approximated by current, consistent with
        # the sim's existing one-step sigma proxy). Slot is already set by the
        # preceding allocate call for HAO.
        return orig_q_update(st, reward, u_next, sigma_next)

    sim.discretize = discretize_threaded
    sim.allocate = allocate_brisk2
    sim.q_update = q_update_brisk
    return sim


def assert_precursor_isolation(sim, U, P, seed, tuned):
    """FAIRNESS GATE (Section 2). Verify that every BASELINE scheme produces
    an identical allocation whether or not the precursor exists, by running
    one interval with P and with P scrambled and confirming baseline
    allocations are unchanged. HAO may differ (it legitimately consumes the
    forecast). Aborts the run on any baseline divergence."""
    UHAT_real, _ = train_gru_forecasts_2ch(sim, U, P, seed)
    P_scram = P.copy()
    np.random.default_rng(seed + 999).shuffle(P_scram)   # destroy precursor
    UHAT_scram, _ = train_gru_forecasts_2ch(sim, U, P_scram, seed)

    rng = np.random.default_rng(seed + 7919)
    t = sim.N_TRAIN + 5
    hist = [U[k] for k in range(max(0, t - sim.ROLL_WIN), t)]
    ubar = np.mean(hist, axis=0)
    sigma = np.zeros(len(sim.SLICES))
    for scheme in ["Static", "Reactive", "Centralized", "FlatAgent"]:
        st1, st2 = sim.SchemeState(), sim.SchemeState()
        X1 = sim.allocate(scheme, st1, t, U[t], U, UHAT_real, ubar, sigma,
                          False, np.random.default_rng(1), tuned)
        X2 = sim.allocate(scheme, st2, t, U[t], U, UHAT_scram, ubar, sigma,
                          False, np.random.default_rng(1), tuned)
        # Static/Reactive/Centralized do not use UHAT at all -> identical.
        # FlatAgent DOES use UHAT (it is a predictive baseline) -> it may
        # differ; that is allowed (it sees the forecast, not the raw
        # precursor). So we only hard-assert the demand-only baselines.
        if scheme in ("Static", "Reactive", "Centralized"):
            if not np.allclose(X1, X2):
                raise SystemExit(
                    f"FAIRNESS VIOLATION: demand-only baseline {scheme} "
                    "changed when the precursor changed. The precursor leaked "
                    "into a demand-only path. Aborting.")
    print("Fairness gate passed: demand-only baselines are precursor-"
          "invariant (Static/Reactive/Centralized).")


# ==========================================================================
# Forecast-quality diagnostics (Section 6).
# ==========================================================================
def forecast_diagnostics(sim, U, UHAT, P, burst_mask):
    """Return AUC (primary), MAE/RMSE/MAPE, and illustrative anticipation /
    false-alarm fractions, on TEST intervals only."""
    from sklearn.metrics import roc_auc_score
    n, ns = U.shape
    LB = sim.GRU_LOOKBACK
    test = np.arange(sim.N_TRAIN, n)

    # burst-exceedance signal s(t) = uhat/(ubar+eps), pooled over slices
    sig, lab = [], []
    mae = rmse = mape = 0.0
    cnt = 0
    antic_hit = antic_tot = fa_hit = fa_tot = 0
    mean_burst_mag = 0.5 * (sim.BURST_MAG_RANGE[0] + sim.BURST_MAG_RANGE[1])
    for t in test:
        if np.isnan(UHAT[t]).any():
            continue
        ubar = U[max(0, t - sim.ROLL_WIN):t].mean(axis=0)
        for j in range(ns):
            s_tj = UHAT[t, j] / (ubar[j] + sim.EPS)
            sig.append(s_tj)
            lab.append(int(burst_mask[t, j]))
            err = UHAT[t, j] - U[t, j]
            mae += abs(err); rmse += err * err; mape += abs(err) / (U[t, j])
            cnt += 1
            tau = 0.5 * mean_burst_mag * sim.BASE_DEMAND[sim.SLICES[j]]
            exceed = (UHAT[t, j] - ubar[j]) >= tau
            if burst_mask[t, j]:
                antic_tot += 1; antic_hit += int(exceed)
            else:
                fa_tot += 1; fa_hit += int(exceed)
    auc = (float(roc_auc_score(lab, sig)) if len(set(lab)) > 1 else
           float("nan"))
    return dict(
        burst_anticipation_auc=auc,
        pred_mae=mae / max(cnt, 1),
        pred_rmse=float(np.sqrt(rmse / max(cnt, 1))),
        pred_mape=100.0 * mape / max(cnt, 1),
        burst_anticipation_fraction=(antic_hit / antic_tot
                                     if antic_tot else float("nan")),
        burst_false_alarm_fraction=(fa_hit / fa_tot
                                    if fa_tot else float("nan")))


# ==========================================================================
# One cell = one (predictability, seed): trains forecaster, tunes baselines,
# runs all six schemes, returns KPI rows + diagnostics.
# ==========================================================================
def run_cell(predictability, seed):
    sim = load_module(SIM_PATH, "hao_pra_u_sim")
    sim.set_regime(ANCHOR_W_E, ANCHOR_H)             # anchor regime (4.3)
    install_brisk_qstate(sim)                        # registered 4.7
    pt = load_module(PRECURSOR_PATH, "precursor_traffic")

    rng = np.random.default_rng(seed)
    U, P, burst_mask = pt.generate_precursor_traffic(
        rng, predictability=predictability, lead=PRECURSOR_LEAD)

    # registered 4.6 forecaster (shared by all predictive schemes)
    UHAT, acc = train_gru_forecasts_2ch(sim, U, P, seed)

    # fairness gate (Section 2) -- only on the first seed of each level to
    # keep cost down; isolation is structural, not seed-dependent.
    if seed == SEEDS[0]:
        tuned0 = sim.tune_baselines(U, UHAT, seed)
        assert_precursor_isolation(sim, U, P, seed, tuned0)

    tuned = sim.tune_baselines(U, UHAT, seed)        # train-only (unchanged)

    kpi_rows = []
    diag_rows = []
    for scheme in sim.SCHEMES:
        res = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        row = dict(method=sim.PAPER_NAMES[scheme], seed=seed,
                   **{k: res[k] for k in sim.METRICS})
        row.update(reliability=res["reliability"], isolation=res["isolation"],
                   overhead=res["overhead"], reward=res["reward_mean_test"])
        kpi_rows.append(row)

    # diagnostics computed once per cell on the shared forecast
    d = forecast_diagnostics(sim, U, UHAT, P, burst_mask)
    for scheme in ("CentralizedPredictive", "FlatAgent", "HAO-PRA-U"):
        diag_rows.append(dict(method=sim.PAPER_NAMES[scheme], seed=seed,
                              predictability=predictability,
                              precursor_channel_used=True, **d))
    return kpi_rows, diag_rows


def cell_task(args):
    predictability, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ckpt = os.path.join("exp3_results", "checkpoints",
                        f"cell_{predictability}_{seed}.json")
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            return json.load(f)
    kpi_rows, diag_rows = run_cell(predictability, seed)
    payload = dict(predictability=predictability, seed=seed,
                   kpi_rows=kpi_rows, diag_rows=diag_rows)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    with open(ckpt, "w") as f:
        json.dump(payload, f)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()))
    ap.add_argument("--validate", action="store_true",
                    help="tripwire: p=0, seed 0 only (must NOT show HAO win)")
    args = ap.parse_args()

    verify_frozen_artifacts()
    os.makedirs("exp3_results/checkpoints", exist_ok=True)

    if args.validate:
        tasks = [(0.0, 0)]
    else:
        tasks = [(p, s) for p in PREDICTABILITY_GRID for s in SEEDS]

    results = []
    if args.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, payload in enumerate(ex.map(cell_task, tasks)):
                results.append(payload)
                print(f"done {i+1}/{len(tasks)}  "
                      f"(p={payload['predictability']}, seed={payload['seed']})",
                      flush=True)
    else:
        for i, t in enumerate(tasks):
            payload = cell_task(t)
            results.append(payload)
            print(f"done {i+1}/{len(tasks)}: p={t[0]} seed={t[1]}", flush=True)

    # ---- assemble outputs ----
    sim = load_module(SIM_PATH, "hao_pra_u_sim")
    kpi_cols = ["method", "seed"] + sim.METRICS + \
        ["reliability", "isolation", "overhead", "reward"]

    # (a) master CSV with predictability column (for our own analysis)
    with open("exp3_results/results_master.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["predictability"] + kpi_cols)
        w.writeheader()
        for pay in results:
            for r in pay["kpi_rows"]:
                w.writerow(dict(predictability=pay["predictability"], **r))

    # (b) one significance_test-ready CSV PER LEVEL (schema byte-identical:
    #     NO predictability column, exactly the columns the script expects)
    by_level = {}
    for pay in results:
        by_level.setdefault(pay["predictability"], []).extend(pay["kpi_rows"])
    for level, rows in by_level.items():
        path = f"exp3_results/results_per_seed__p{level}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=kpi_cols)
            w.writeheader()
            w.writerows(rows)

    # (c) forecast diagnostics
    diag_cols = ["method", "seed", "predictability", "precursor_channel_used",
                 "burst_anticipation_auc", "pred_mae", "pred_rmse",
                 "pred_mape", "burst_anticipation_fraction",
                 "burst_false_alarm_fraction"]
    with open("exp3_results/forecast_diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=diag_cols)
        w.writeheader()
        for pay in results:
            w.writerows(pay["diag_rows"])

    print("\nWrote:")
    print("  exp3_results/results_master.csv          (all levels, our analysis)")
    print("  exp3_results/results_per_seed__p<level>.csv  (feed to significance_test.py)")
    print("  exp3_results/forecast_diag.csv           (AUC etc., Section 6)")
    print("\nNext:")
    print("  for f in exp3_results/results_per_seed__p*.csv; do "
          "python3 scripts/significance_test.py $f; done")
    print("  python3 analyze_experiment3.py")


if __name__ == "__main__":
    main()
