#!/usr/bin/env python3
"""Edge audit: control-delay sensitivity (M1.4) and per-slice violations (M2).

    python analyze_edge_audit.py \
        results/nested_validation/all_cells_edge_audit.csv \
        results/nested_validation/per_slice_edge_audit.csv

Not a registered study and not a hypothesis test. Reproduces every number in
Sec. VII-G of the manuscript. Imports the objective from
analyze_registered_runs.py so the two cannot drift apart.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path
                              .abspath(__file__)))
from analyze_registered_runs import jbar, dz, holm, wil   # noqa: E402

SLICES = ["eMBB", "URLLC", "mMTC", "XR", "IIoT"]
A_02, A_08 = "Static+Edge", "Static+Edge@0.8ms"
STATIC, HAO = "Static Allocation", "HAO-PRA-U"


def load_cells(path):
    C = defaultdict(list)
    for r in csv.DictReader(open(path)):
        C[(float(r["w_E"]), float(r["headroom"]))].append(r)
    return C


def family(C, a, b, stat="jbar"):
    out = []
    for (wE, h) in sorted(C):
        rows = C[(wE, h)]
        if stat == "jbar":
            d = jbar(rows, wE)
        else:
            sd = sorted({r["seed"] for r in rows})
            sub = {(r["method"], r["seed"]): r for r in rows}
            d = {m: [float(sub[(m, s)][stat]) for s in sd]
                 for m in {r["method"] for r in rows}}
        x, y = np.array(d[a]), np.array(d[b])
        out.append(dict(wE=wE, h=h, diff=x.mean() - y.mean(),
                        p=wil(x, y), dz=dz(x, y)))
    for r, hh in zip(out, holm([r["p"] for r in out])):
        r["holm"] = hh
    return out


def main(cells_path, slice_path):
    C = load_cells(cells_path)
    cells = sorted(C)
    seeds = sorted({int(r["seed"]) for rs in C.values() for r in rs})
    for k in cells:
        assert len(C[k]) == 4 * len(seeds), (k, len(C[k]))
    print(f"{len(cells)} regimes x 4 arms x {len(seeds)} seeds complete")

    print("\n" + "=" * 70)
    print("M1.4  control-delay sensitivity, every-interval convention")
    print("=" * 70)
    for a, b, tag in ((A_02, STATIC, "Static+Edge @0.2ms vs tuned Static"),
                      (A_08, STATIC, "Static+Edge @0.8ms vs tuned Static"),
                      (A_08, HAO, "Static+Edge @0.8ms vs HAO-PRA-U")):
        rs = family(C, a, b)
        wins = sum(r["diff"] < 0 for r in rs)
        surv = sum(r["holm"] < 0.05 for r in rs)
        print(f"  {tag:42s} {a.split('@')[0]} lower in {wins:2d}/{len(rs)}"
              f"; {surv:2d} survive Holm; mean {np.mean([r['diff'] for r in rs]):+.6f}")

    print("\n  per-regime margin against tuned Static")
    r02 = family(C, A_02, STATIC)
    r08 = family(C, A_08, STATIC)
    print(f"    {'w_E':>7s} {'h':>4s} {'@0.2ms':>10s} {'@0.8ms':>10s}"
          f" {'holm':>8s} {'verdict':>10s}")
    for a, b in zip(r02, r08):
        v = "edge" if b["diff"] < 0 else "Static"
        print(f"    {a['wE']:7.4f} {a['h']:4.1f} {a['diff']:+10.5f}"
              f" {b['diff']:+10.5f} {b['holm']:8.5f} {v:>10s}")

    print("\n  where the extra 0.6 ms goes (0.8ms arm minus 0.2ms arm)")
    for met in ("latency", "sla_violation"):
        rs = family(C, A_08, A_02, met)
        d = [r["diff"] for r in rs]
        print(f"    {met:15s} mean {np.mean(d):+.6f}"
              f"  range [{min(d):+.5f}, {max(d):+.5f}]")
    rs = family(C, A_08, A_02)
    print(f"    {'Jbar':15s} mean {np.mean([r['diff'] for r in rs]):+.6f}")

    print("\n" + "=" * 70)
    print("M2  per-slice SLA violation, Static+Edge minus tuned Static")
    print("=" * 70)
    D = defaultdict(dict)
    for r in csv.DictReader(open(slice_path)):
        D[(float(r["w_E"]), float(r["headroom"]),
           int(r["seed"]))][r["method"]] = r
    print(f"  {'slice':8s} {'mean delta':>12s} {'regimes worse':>14s}")
    total_worse = 0
    for s in SLICES:
        deltas, worse = [], 0
        for (wE, h) in cells:
            a = np.mean([float(D[(wE, h, sd)][A_02][s]) for sd in seeds])
            b = np.mean([float(D[(wE, h, sd)][STATIC][s]) for sd in seeds])
            deltas.append(a - b)
            if a - b > 1e-12:
                worse += 1
        total_worse += worse
        print(f"  {s:8s} {np.mean(deltas):+12.5f} {worse:11d}/{len(cells)}")
    print(f"\n  regime-slice cells where Static+Edge is worse: "
          f"{total_worse} of {len(cells) * len(SLICES)}")

    worse08 = 0
    for s in SLICES:
        for (wE, h) in cells:
            a = np.mean([float(D[(wE, h, sd)][A_08][s]) for sd in seeds])
            b = np.mean([float(D[(wE, h, sd)][STATIC][s]) for sd in seeds])
            if a - b > 1e-12:
                worse08 += 1
    print(f"  same at the 0.8 ms charge: {worse08} of "
          f"{len(cells) * len(SLICES)}  (donor cost is the reallocation, "
          f"not the delay)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
