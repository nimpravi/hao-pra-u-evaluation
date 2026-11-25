#!/usr/bin/env python3
"""Study 7 analysis: Static + edge correction.

Self-contained. Reproduces every number in STATIC_EDGE_RESULTS.md from the
two run artifacts, with no imports outside numpy/scipy.

    python analyze_static_edge.py \
        results/nested_validation/all_cells_static_edge.csv \
        results/nested_validation/tuned_params_static_edge.json \
        [results/nested_validation/all_cells_oracle.csv]

The third argument is optional. Supplied, it adds the cross-run comparisons
against HAO-Oracle and HAO-ConstReserve, and first checks that the two runs
are on the same scale.

Place in scripts/ and record its SHA-256 in preregistrations/hashes.txt.
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# ----------------------------------------------------------------- objective
# Registered weights. w_E is swept; the rest are rescaled to fill 1 - w_E.
OBJ = dict(sla=0.2256, latency=0.1880, throughput=0.1692, acceptance=0.1316,
           utilization=0.1128, reliability=0.0658, isolation=0.0282,
           energy=0.0188, fragmentation=0.0390, overhead=0.0210)
BEN = ["throughput", "utilization", "acceptance", "reliability", "isolation",
       "energy_eff"]
COST = ["sla_violation", "latency", "fragmentation", "overhead"]
KEY = dict(sla_violation="sla", latency="latency", throughput="throughput",
           acceptance="acceptance", utilization="utilization",
           reliability="reliability", isolation="isolation",
           energy_eff="energy", fragmentation="fragmentation",
           overhead="overhead")
# Registered fixed normalizers for the absolute statistic.
LNORM = 40.0
TNORM = 1.2 * (200 + 90 + 135 + 170 + 115)
# Control-message counts the simulator charges per scheme label.
ORCH_MSGS_MAX = 24


def wts(wE):
    o = {k: v for k, v in OBJ.items() if k != "energy"}
    s = (1.0 - wE) / sum(o.values())
    w = {k: v * s for k, v in o.items()}
    w["energy"] = wE
    return w


def comps(rows, wE):
    """Min-max composite, normalized WITHIN the scheme set. Higher is better."""
    W = wts(wE)
    seeds = sorted({r["seed"] for r in rows})
    ms = sorted({r["method"] for r in rows})
    out = {m: [] for m in ms}
    for sd in seeds:
        sub = {r["method"]: r for r in rows if r["seed"] == sd}
        acc = {m: 0.0 for m in ms}
        for met in BEN + COST:
            vals = np.array([float(sub[m][met]) for m in ms])
            lo, hi = vals.min(), vals.max()
            n = (vals - lo) / (hi - lo + 1e-12)
            if met in COST:
                n = 1.0 - n
            for m, v in zip(ms, n):
                acc[m] += W[KEY[met]] * float(v)
        for m in ms:
            out[m].append(acc[m])
    return out


def jbar_terms(row, W):
    """Per-term contributions to Jbar. Signed so the sum is Jbar."""
    d = {k: float(row[k]) for k in row
         if k not in ("w_E", "headroom", "method", "seed")}
    return dict(
        sla=W["sla"] * d["sla_violation"],
        latency=W["latency"] * (d["latency"] / LNORM),
        energy=W["energy"] * (1.0 - min(d["energy_eff"], 1.0)),
        frag=W["fragmentation"] * d["fragmentation"],
        overhead=W["overhead"] * d["overhead"],
        throughput=-W["throughput"] * (d["throughput"] / TNORM),
        util=-W["utilization"] * d["utilization"],
        accept=-W["acceptance"] * d["acceptance"],
        rel=-W["reliability"] * d["reliability"],
        iso=-W["isolation"] * d["isolation"],
    )


def jbar(rows, wE):
    """Absolute cost, fixed normalizers, no within-set coupling. Lower better."""
    W = wts(wE)
    seeds = sorted({r["seed"] for r in rows})
    ms = sorted({r["method"] for r in rows})
    out = {m: [] for m in ms}
    for sd in seeds:
        sub = {r["method"]: r for r in rows if r["seed"] == sd}
        for m in ms:
            out[m].append(sum(jbar_terms(sub[m], W).values()))
    return out


def dz(a, b):
    d = np.asarray(a) - np.asarray(b)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 1e-15 else 0.0


def holm_bh(ps):
    ps = np.asarray(ps)
    n = len(ps)
    order = np.argsort(ps)
    holm = np.empty(n)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (n - i) * ps[idx])
        holm[idx] = min(1.0, run)
    bh = np.empty(n)
    run = 1.0
    for i, idx in enumerate(order[::-1]):
        run = min(run, ps[idx] * n / (n - i))
        bh[idx] = min(1.0, run)
    return holm, bh


def load(path):
    rows = list(csv.DictReader(open(path)))
    c = defaultdict(list)
    for r in rows:
        c[(float(r["w_E"]), float(r["headroom"]))].append(r)
    return c


def paired(cells, A, B, fn, better, cells_B=None):
    """Per-cell paired Wilcoxon of A against B, with Holm and BH over 20 cells."""
    cb = cells_B if cells_B is not None else cells
    recs = []
    for cell in sorted(cells):
        wE = cell[0]
        a = np.array(fn(cells[cell], wE)[A])
        b = np.array(fn(cb[cell], wE)[B])
        d = a - b
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:
            p = 1.0
        recs.append(dict(cell=cell, p=float(p), dz=dz(a, b),
                         delta=float(d.mean()), muB=float(b.mean()),
                         win=(d.mean() > 0) if better == "hi" else (d.mean() < 0)))
    holm, bh = holm_bh([r["p"] for r in recs])
    for r, hm, bb in zip(recs, holm, bh):
        r["holm"], r["bh"] = float(hm), float(bb)
    return recs


def report(label, recs):
    n = len(recs)
    w = sum(r["win"] for r in recs)
    print(f"  {label:<46} wins {w:>2}/{n}"
          f"  raw {sum(1 for r in recs if r['p'] < 0.05 and r['win']):>2}"
          f"  Holm {sum(1 for r in recs if r['holm'] < 0.05 and r['win']):>2}"
          f"  BH {sum(1 for r in recs if r['bh'] < 0.05 and r['win']):>2}"
          f"  delta {np.mean([r['delta'] for r in recs]):+.5f}"
          f"  rel {np.mean([abs(r['delta']) / abs(r['muB']) for r in recs]) * 100:>5.1f}%"
          f"  dz_med {np.median([r['dz'] for r in recs]):+7.2f}")


def main():
    csv_path, json_path = sys.argv[1], sys.argv[2]
    oracle_path = sys.argv[3] if len(sys.argv) > 3 else None
    SE = load(csv_path)
    tp = json.load(open(json_path))

    print(f"cells {len(SE)}   seeds/cell {len(SE[sorted(SE)[0]]) // 8}   "
          f"schemes {sorted({r['method'] for v in SE.values() for r in v})}")

    # --- 1. per-cell winner ------------------------------------------------
    for stat, fn, better in (("composite", comps, "hi"), ("Jbar", jbar, "lo")):
        wins = defaultdict(int)
        for cell in sorted(SE):
            c = fn(SE[cell], cell[0])
            mu = {m: float(np.mean(v)) for m, v in c.items()}
            wins[max(mu, key=mu.get) if better == "hi"
                 else min(mu, key=mu.get)] += 1
        print(f"\n[1] per-cell winner, {stat}: {dict(wins)}")

    # --- 2. head-to-head ---------------------------------------------------
    PAIRS = [("Static+Edge", "Static Allocation"),
             ("Static+Edge", "HAO-PRA-U"),
             ("Static+Edge-F", "Static Allocation"),
             ("Static+Edge-F", "HAO-PRA-U"),
             ("Static+Edge", "Static+Edge-F")]
    for stat, fn, better in (("composite", comps, "hi"), ("Jbar", jbar, "lo")):
        print(f"\n[2] head-to-head, {stat}")
        for A, B in PAIRS:
            report(f"{A} vs {B}", paired(SE, A, B, fn, better))

    # --- 3. mechanism ------------------------------------------------------
    flat = defaultdict(dict)
    for v in SE.values():
        for r in v:
            flat[(r["w_E"], r["headroom"], r["seed"])][r["method"]] = r
    ks = sorted(flat)
    print("\n[3] mechanism: per-metric means")
    print(f"  {'metric':<16}{'Static':>11}{'S+Edge':>11}{'S+Edge-F':>11}"
          f"{'HAO-PRA-U':>11}{'d% vs S':>10}")
    for met in ["sla_violation", "latency", "throughput", "utilization",
                "energy_eff", "fragmentation", "acceptance", "reliability",
                "isolation", "overhead"]:
        g = lambda m: np.array([float(flat[k][m][met]) for k in ks])
        s, e = g("Static Allocation"), g("Static+Edge")
        print(f"  {met:<16}{s.mean():>11.5f}{e.mean():>11.5f}"
              f"{g('Static+Edge-F').mean():>11.5f}{g('HAO-PRA-U').mean():>11.5f}"
              f"{(e.mean() - s.mean()) / abs(s.mean()) * 100:>+9.1f}%")
    s = np.array([float(flat[k]["Static Allocation"]["sla_violation"]) for k in ks])
    e = np.array([float(flat[k]["Static+Edge"]["sla_violation"]) for k in ks])
    print(f"  SLA: S+Edge better in {int((e < s).sum())}/{len(s)} cell-seeds, "
          f"tied {int((e == s).sum())}, worse {int((e > s).sum())}")

    # --- 4. Jbar gap decomposition ----------------------------------------
    for B in ["HAO-PRA-U", "Static Allocation"]:
        agg = defaultdict(list)
        for k in ks:
            W = wts(float(k[0]))
            a = jbar_terms(flat[k]["Static+Edge"], W)
            b = jbar_terms(flat[k][B], W)
            for t in a:
                agg[t].append(a[t] - b[t])
        tot = sum(np.mean(v) for v in agg.values())
        print(f"\n[4] Jbar(Static+Edge) - Jbar({B}) = {tot:+.5f}"
              f"   (negative favours Static+Edge)")
        for t, v in sorted(agg.items(), key=lambda x: np.mean(x[1])):
            m = np.mean(v)
            print(f"    {t:<11}{m:>+10.5f}{m / abs(tot) * 100:>+8.1f}% of gap")

    # --- 5. control-cost sensitivity --------------------------------------
    # The harness charges control cost by SCHEME LABEL. Static+Edge is
    # evaluated under the "Static" label, so it is charged 1 message/interval
    # while performing a per-interval cross-domain reallocation. Report the
    # break-even message count rather than assuming one.
    fires = defaultdict(list)
    for ent in tp:
        fires[(ent["w_E"], ent["headroom"])].append(
            ent["tuned"]["edge_fires_per_interval_nofc"])
    print("\n[5] control-cost break-even for Static+Edge (overhead term only)")
    print(f"  {'w_E':>7}{'h':>5}{'fires/int':>11}{'m* vs Static':>14}"
          f"{'msgs/firing':>13}{'m* vs HAO':>11}")
    bs, bh_, mpf = [], [], []
    for cell in sorted(SE):
        wE = cell[0]
        J = jbar(SE[cell], wE)
        W = wts(wE)["overhead"]
        dS = np.mean(np.array(J["Static+Edge"]) - np.array(J["Static Allocation"]))
        dH = np.mean(np.array(J["Static+Edge"]) - np.array(J["HAO-PRA-U"]))
        mS = 1.0 + (-dS) * ORCH_MSGS_MAX / W
        mH = 1.0 + (-dH) * ORCH_MSGS_MAX / W
        fr = float(np.mean(fires[cell]))
        k = (mS - 1.0) / fr
        bs.append(mS); bh_.append(mH); mpf.append(k)
        print(f"  {wE:>7}{cell[1]:>5}{fr:>11.3f}{mS:>14.1f}{k:>13.1f}{mH:>11.1f}")
    f = lambda v: f"min {min(v):.1f} median {np.median(v):.1f} max {max(v):.1f}"
    print(f"  break-even vs Static : {f(bs)} messages/interval, {f(mpf)} per firing")
    print(f"  break-even vs HAO    : {f(bh_)} messages/interval "
          f"(HAO is charged {ORCH_MSGS_MAX})")
    print(f"  mean fires/interval  : "
          f"{np.mean([np.mean(v) for v in fires.values()]):.3f}")

    # Worst case: charge Static+Edge HAO's entire control budget.
    recs = []
    for cell in sorted(SE):
        wE = cell[0]
        J = jbar(SE[cell], wE)
        pen = wts(wE)["overhead"] * (ORCH_MSGS_MAX - 1) / ORCH_MSGS_MAX
        a = np.array(J["Static+Edge"]) + pen
        for B in ["Static Allocation", "HAO-PRA-U"]:
            b = np.array(J[B])
            try:
                _, p = stats.wilcoxon(a, b)
            except ValueError:
                p = 1.0
            recs.append((B, cell, float(p), a.mean() < b.mean()))
    print(f"  charged all {ORCH_MSGS_MAX} messages:", end="")
    for B in ["Static Allocation", "HAO-PRA-U"]:
        sub = [r for r in recs if r[0] == B]
        hm, _ = holm_bh([r[2] for r in sub])
        print(f"  beats {B} {sum(1 for r in sub if r[3])}/{len(sub)}"
              f" (Holm {sum(1 for r, h in zip(sub, hm) if r[3] and h < 0.05)})",
              end="")
    print()

    # --- 6. cross-run comparisons -----------------------------------------
    if not oracle_path:
        print("\n[6] skipped: no oracle CSV given")
        return
    OR = load(oracle_path)
    drift = max(np.abs(np.array(jbar(SE[c], c[0])["Static Allocation"])
                       - np.array(jbar(OR[c], c[0])["Static Allocation"])).max()
                for c in sorted(SE))
    print(f"\n[6] cross-run scale check: max |dJbar(Static)| = {drift:.2e}")
    if drift > 1e-12:
        print("    NOT COMPARABLE. Static Allocation differs between runs.")
        print("    Do not report the cross-run rows.")
        return
    print("    Static Allocation is identical across runs; Jbar is comparable.")
    for B in ["HAO-Oracle", "HAO-ConstReserve"]:
        if B not in {r["method"] for v in OR.values() for r in v}:
            print(f"  {B}: absent from oracle CSV, skipped")
            continue
        report(f"Static+Edge vs {B} [Jbar, cross-run]",
               paired(SE, "Static+Edge", B, jbar, "lo", cells_B=OR))


if __name__ == "__main__":
    main()
