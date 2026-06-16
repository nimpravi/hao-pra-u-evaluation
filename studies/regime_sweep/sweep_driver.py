#!/usr/bin/env python3
"""
Phase B confirmatory sweep driver (SWEEP_PREREGISTRATION.md).

Runs the pre-registered 5x4 grid: energy/resource-cost weight
w_E in {0.0188, 0.05, 0.10, 0.20, 0.30} x capacity headroom
h in {1.4, 1.7, 2.0, 2.3}, 10 seeds per cell, baselines re-tuned per
(cell, seed) on training intervals under that cell's reward.

GRU forecasts depend only on the seed (traffic is regime-independent),
so they are trained once per seed and cached in uhat_cache/.

Usage:
    python3 sweep_driver.py --workers 8          # full sweep
    python3 sweep_driver.py --validate           # anchor cell, seeds 0-1
Outputs:
    sweep_results/all_cells.csv   (one row per cell x method x seed)
    sweep_results/tuned_params.json
Then run:  python3 analyze_sweep.py
"""
import argparse, csv, itertools, json, os
import numpy as np
import importlib.util

W_E_GRID = [0.0188, 0.05, 0.10, 0.20, 0.30]
H_GRID = [1.4, 1.7, 2.0, 2.3]
N_SEEDS = 10
SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hao_pra_u_sim.py")


def load_sim():
    spec = importlib.util.spec_from_file_location("sim", SIM_PATH)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim


def build_uhat_cache(seeds):
    os.makedirs("uhat_cache", exist_ok=True)
    sim = load_sim()
    acc_all = {}
    for seed in seeds:
        path = f"uhat_cache/seed{seed}.npz"
        if os.path.exists(path):
            continue
        rng = np.random.default_rng(seed)
        U = sim.generate_traffic(rng)
        UHAT, acc = sim.train_gru_forecasts(U, seed)
        np.savez(path, UHAT=UHAT)
        acc_all[seed] = acc
        print(f"cached GRU forecasts seed {seed}", flush=True)
    if acc_all:
        with open("uhat_cache/prediction_accuracy.json", "a") as f:
            json.dump(acc_all, f)


def run_cell_seed(args):
    """One (w_E, h, seed) task: re-tune baselines under this regime,
    run all six schemes. Returns list of result rows."""
    wE, h, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sim = load_sim()
    sim.set_regime(wE, h)
    rng = np.random.default_rng(seed)
    U = sim.generate_traffic(rng)
    UHAT = np.load(f"uhat_cache/seed{seed}.npz")["UHAT"]
    tuned = sim.tune_baselines(U, UHAT, seed)
    rows = []
    for scheme in sim.SCHEMES:
        r = sim.run_scheme(scheme, U, UHAT, seed, tuned)
        row = dict(w_E=wE, headroom=h, method=sim.PAPER_NAMES[scheme],
                   seed=seed)
        for k in (sim.METRICS + ["reliability", "isolation", "overhead"]):
            row[k] = r[k]
        rows.append(row)
    tuned_ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in tuned.items()}
    return rows, dict(w_E=wE, headroom=h, seed=seed, tuned=tuned_ser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()))
    ap.add_argument("--validate", action="store_true",
                    help="pipeline check: anchor cell only, seeds 0-1")
    args = ap.parse_args()

    if args.validate:
        cells, seeds = [(0.0188, 2.0)], [0, 1]
    else:
        cells, seeds = list(itertools.product(W_E_GRID, H_GRID)), \
            list(range(N_SEEDS))

    build_uhat_cache(seeds)
    tasks = [(wE, h, s) for (wE, h) in cells for s in seeds]
    print(f"{len(tasks)} tasks on {args.workers} workers", flush=True)

    os.makedirs("sweep_results", exist_ok=True)
    all_rows, tuned_log = [], []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (rows, tl) in enumerate(ex.map(run_cell_seed, tasks)):
                all_rows += rows
                tuned_log.append(tl)
                print(f"done {i+1}/{len(tasks)}: w_E={tl['w_E']} "
                      f"h={tl['headroom']} seed={tl['seed']}", flush=True)
    else:
        for i, t in enumerate(tasks):
            rows, tl = run_cell_seed(t)
            all_rows += rows
            tuned_log.append(tl)
            print(f"done {i+1}/{len(tasks)}: w_E={t[0]} h={t[1]} "
                  f"seed={t[2]}", flush=True)

    fields = list(all_rows[0].keys())
    out = "sweep_results/all_cells.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    with open("sweep_results/tuned_params.json", "w") as f:
        json.dump(tuned_log, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
