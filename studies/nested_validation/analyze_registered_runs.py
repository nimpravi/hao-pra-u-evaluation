#!/usr/bin/env python3
"""Scores the two registered runs against the decision rules fixed in their
registrations. Self-contained: numpy and scipy only.

    python analyze_registered_runs.py \
        results/nested_validation/all_cells_confirm_edge.csv \
        results/nested_validation/all_windows_symmetric.csv

Reproduces every number in RESULTS.md. The verdict logic is a transcription
of PREREGISTRATION_STATIC_EDGE_CONFIRM.md Sec. 4 and
PREREGISTRATION_SHIFT_SYMMETRIC.md Sec. 4. Nothing here is tunable.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# ------------------------------------------------------------------ objective
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
LNORM = 40.0
TNORM = 1.2 * (200 + 90 + 135 + 170 + 115)
WE_ANCHOR = 0.0188


def wts(wE):
    o = {k: v for k, v in OBJ.items() if k != "energy"}
    s = (1.0 - wE) / sum(o.values())
    w = {k: v * s for k, v in o.items()}
    w["energy"] = wE
    return w


def comps(rows, wE):
    """Min-max composite, normalized WITHIN the scheme set. Higher better."""
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


def jbar(rows, wE):
    """Absolute cost, fixed normalizers, set-independent. Lower better."""
    W = wts(wE)
    seeds = sorted({r["seed"] for r in rows})
    ms = sorted({r["method"] for r in rows})
    out = {m: [] for m in ms}
    for sd in seeds:
        sub = {r["method"]: r for r in rows if r["seed"] == sd}
        for m in ms:
            d = {k: float(sub[m][k]) for k in KEY}
            out[m].append(
                W["sla"] * d["sla_violation"]
                + W["latency"] * (d["latency"] / LNORM)
                + W["energy"] * (1.0 - min(d["energy_eff"], 1.0))
                + W["fragmentation"] * d["fragmentation"]
                + W["overhead"] * d["overhead"]
                - W["throughput"] * (d["throughput"] / TNORM)
                - W["utilization"] * d["utilization"]
                - W["acceptance"] * d["acceptance"]
                - W["reliability"] * d["reliability"]
                - W["isolation"] * d["isolation"])
    return out


def dz(a, b):
    d = np.asarray(a) - np.asarray(b)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 1e-15 else 0.0


def holm(ps):
    ps = np.asarray(ps)
    n = len(ps)
    order = np.argsort(ps)
    out = np.empty(n)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (n - i) * ps[idx])
        out[idx] = min(1.0, run)
    return out


def wil(x, y):
    try:
        return stats.wilcoxon(x, y, zero_method="wilcox",
                              alternative="two-sided", method="exact").pvalue
    except Exception:
        return stats.wilcoxon(x, y, alternative="two-sided").pvalue


# ================================================== STUDY 14: confirmatory
def study14(path):
    rows = list(csv.DictReader(open(path)))
    C = defaultdict(list)
    for r in rows:
        C[(float(r["w_E"]), float(r["headroom"]))].append(r)
    cells = sorted(C)
    seeds = sorted({int(r["seed"]) for r in rows})

    print("=" * 74)
    print("STUDY 14  Confirmatory replication of Static + edge correction")
    print("=" * 74)
    print(f"  regimes {len(cells)}   seeds {seeds[0]}-{seeds[-1]} "
          f"(n={len(seeds)})   rows {len(rows)}")
    bad = sorted(set(seeds) & set(range(11)))
    print(f"  registered seed set 11-30: "
          f"{'OK, no seed 0-10 present' if not bad else 'CONTAMINATED ' + str(bad)}")
    if bad:
        print("  >>> registration Sec. 3 violated; do not report this run.")
        return
    for k in cells:
        assert len(C[k]) == 8 * len(seeds), (k, len(C[k]))
    print("  grid complete: 20 regimes x 8 schemes x 20 seeds")

    def fam(a, b, stat):
        out = []
        for (wE, h) in cells:
            rs = C[(wE, h)]
            if stat == "jbar":
                d = jbar(rs, wE)
            elif stat == "comp":
                d = comps(rs, wE)
            else:
                sd_ = sorted({r["seed"] for r in rs})
                sub = {(r["method"], r["seed"]): r for r in rs}
                d = {m: [float(sub[(m, s)][stat]) for s in sd_]
                     for m in {r["method"] for r in rs}}
            x, y = np.array(d[a]), np.array(d[b])
            out.append(dict(wE=wE, h=h, diff=x.mean() - y.mean(),
                            p=wil(x, y), dz=dz(x, y)))
        for r, hh in zip(out, holm([r["p"] for r in out])):
            r["holm"] = hh
        return out

    def show(tag, a, b, rs, higher=False):
        wins = sum((r["diff"] > 0) if higher else (r["diff"] < 0) for r in rs)
        surv = sum(r["holm"] < 0.05 for r in rs)
        print(f"\n  {tag}")
        print(f"    {a} better in {wins}/{len(rs)};  {surv}/{len(rs)} "
              f"survive Holm(0.05)")
        print(f"    mean margin {np.mean([r['diff'] for r in rs]):+.6f}   "
              f"median d_z {np.median([r['dz'] for r in rs]):+.2f}   "
              f"max Holm p {max(r['holm'] for r in rs):.2e}")
        return wins, surv

    w1, s1 = show("H1 (primary, Jbar)", "Static+Edge", "Static Allocation",
                  fam("Static+Edge", "Static Allocation", "jbar"))
    w2, s2 = show("H2 (primary, Jbar)", "Static+Edge", "HAO-PRA-U",
                  fam("Static+Edge", "HAO-PRA-U", "jbar"))
    w4, _ = show("H4 (secondary, SLA violation)", "Static+Edge",
                 "Static Allocation",
                 fam("Static+Edge", "Static Allocation", "sla_violation"))
    ad = [abs(np.mean(jbar(C[k], k[0])["Static+Edge"])
              - np.mean(jbar(C[k], k[0])["Static+Edge-F"])) for k in cells]
    print(f"\n  H3 (secondary): mean |dJbar| between the two edge variants "
          f"= {np.mean(ad):.2e}  (threshold 1e-3)")

    def v(w, s, n=20):
        if w >= 11 and s >= 1:
            return "SUPPORTED"
        return "REFUTED" if (n - w) >= 11 else "INCONCLUSIVE"
    print("\n  VERDICTS, registration Sec. 4")
    print(f"    H1 {v(w1, s1)}    H2 {v(w2, s2)}"
          f"    H3 {'SUPPORTED' if np.mean(ad) < 1e-3 else 'REFUTED'}"
          f"    H4 {'supported' if w4 >= 11 else 'not supported'}")


# ============================================== STUDY 15: symmetric shift
SIX = ["Static Allocation", "Reactive Allocation", "Centralized Orchestration",
       "Centralized Predictive", "Flat-Agent Allocation", "HAO-PRA-U"]
SCEN = ["S1a_load15", "S1b_load30", "S2_mixshift", "S3_bursts"]
UHAT_CONSUMERS = ["Centralized Predictive", "Flat-Agent Allocation",
                  "HAO-PRA-U", "HAO-PPO"]


def study15(path):
    R = list(csv.DictReader(open(path)))
    METS = BEN + COST
    seeds = sorted({int(r["seed"]) for r in R})
    schemes = sorted({r["method"] for r in R})

    print("\n" + "=" * 74)
    print("STUDY 15  Regime shift under symmetric conditions")
    print("=" * 74)
    print(f"  seeds {seeds}   schemes {len(schemes)}   rows {len(R)}")

    # integrity: intervals 0-999 are shared, so the pre-window must be
    # identical across all five scenarios for every scheme and seed.
    ref = defaultdict(dict)
    for r in R:
        if r["window"] == "pre":
            ref[(r["method"], r["seed"])][r["scenario"]] = r
    mism = defaultdict(int)
    for v in ref.values():
        base = v["S0_noshift"]
        for sc, rr in v.items():
            if sc == "S0_noshift":
                continue
            for m in METS:
                if abs(float(rr[m]) - float(base[m])) > 1e-12:
                    mism[v[sc]["method"]] += 1
    print(f"  pre-window reproducibility across scenarios: "
          f"{sum(mism.values())} mismatches"
          + (f"  {dict(mism)}" if mism else "  (as required)"))

    print("\n  H3 / S0 neutrality control  [gate: study void if this fails]")
    print("  rule: |mean_post - mean_pre| <= across-seed SD of the pre-window")
    fails = []
    for m in schemes:
        bad = []
        for met in METS:
            pre = np.array([float(r[met]) for r in R
                            if r["scenario"] == "S0_noshift"
                            and r["window"] == "pre" and r["method"] == m])
            post = np.array([float(r[met]) for r in R
                             if r["scenario"] == "S0_noshift"
                             and r["window"] == "post" and r["method"] == m])
            if abs(post.mean() - pre.mean()) > pre.std(ddof=1) + 1e-15:
                bad.append(met)
        print(f"    {m:28s} {'PASS' if not bad else 'FAIL  ' + ','.join(bad)}")
        if bad:
            fails.append(m)

    print("\n  signed S0 drift (stationary trace, so this should be noise)")
    print(f"    {'scheme':28s} {'d_sla':>9s} {'d_thru':>9s}  uses forecast")
    for m in schemes:
        def g(w, met):
            return np.array([float(r[met]) for r in R
                             if r["scenario"] == "S0_noshift"
                             and r["window"] == w and r["method"] == m])
        print(f"    {m:28s} "
              f"{g('post','sla_violation').mean()-g('pre','sla_violation').mean():+9.5f} "
              f"{g('post','throughput').mean()-g('pre','throughput').mean():+9.4f}"
              f"  {'YES' if m in UHAT_CONSUMERS else 'no'}")

    six_fail = [f for f in fails if f in SIX]
    if six_fail:
        print(f"\n  S0 FAILED for {six_fail}")
        print("  Registration Sec. 4: 'If S0 fails, the study is void and no")
        print("  result from it is reported.'  H1 and H2 are NOT computed.")
        print("  See AMENDMENT_A2_SHIFT_SYMMETRIC.md for the cause.")
        return
    print("\n  S0 passed; scoring H1 and H2.")
    for stat, label in (("comp", "H1 post-shift composite"),
                        ("dsla", "H2 pre-to-post change in SLA violation")):
        rs = []
        for sc in SCEN:
            if stat == "comp":
                d = comps([dict(r) for r in R if r["scenario"] == sc
                           and r["window"] == "post" and r["method"] in SIX],
                          WE_ANCHOR)
                x, y = np.array(d["Static Allocation"]), np.array(d["HAO-PRA-U"])
                hi = True
            else:
                def ch(m):
                    pre = {int(r["seed"]): float(r["sla_violation"]) for r in R
                           if r["scenario"] == sc and r["window"] == "pre"
                           and r["method"] == m}
                    po = {int(r["seed"]): float(r["sla_violation"]) for r in R
                          if r["scenario"] == sc and r["window"] == "post"
                          and r["method"] == m}
                    return np.array([po[s] - pre[s] for s in seeds])
                x, y = ch("Static Allocation"), ch("HAO-PRA-U")
                hi = False
            rs.append(dict(sc=sc, diff=x.mean() - y.mean(), p=wil(x, y)))
        for r, hh in zip(rs, holm([r["p"] for r in rs])):
            r["holm"] = hh
        wins = sum((r["diff"] > 0) if hi else (r["diff"] < 0) for r in rs)
        print(f"    {label}: Static better in {wins}/4; "
              f"{sum(r['holm'] < 0.05 for r in rs)}/4 survive Holm -> "
              + ("SUPPORTED" if wins >= 3 else
                 "REFUTED" if (4 - wins) >= 3 else "INCONCLUSIVE"))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        study14(sys.argv[1])
    if len(sys.argv) > 2:
        study15(sys.argv[2])
