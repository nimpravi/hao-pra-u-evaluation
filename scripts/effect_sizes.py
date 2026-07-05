#!/usr/bin/env python3
"""
Effect-size computation for HAO-PRA-U vs. baselines.

Complements significance_test.py (which reports Welch's t and p) by adding
practical-significance measures.

  * Cohen's d           -- standardized mean difference (parametric).
  * Cliff's delta       -- non-parametric, rank-based, robust to non-normality;
                           pairs with the Mann-Whitney U validation already in
                           the paper.

Sign convention: effect sizes are computed as (PROPOSED - baseline). For
sla_violation, latency, fragmentation, energy/overhead a NEGATIVE value means
HAO-PRA-U is better (lower); for throughput, utilization, acceptance,
reliability, isolation a POSITIVE value means HAO-PRA-U is better (higher).

Cliff's delta magnitude thresholds (Romano et al., 2006):
  |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
Cohen's d magnitude thresholds (Cohen, 1988):
  |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, else large.

USAGE
-----
  python3 effect_sizes.py <results_per_seed.csv> [--metric sla_violation]
                          [--proposed "HAO-PRA-U"]

The CSV must have columns: method, seed, and the metric columns. One row per
(method, seed). Works on the anchor / predictability per-seed files directly;
for the sweep, pre-filter by (w_E, headroom) before calling, or pass the cell
CSV and use --group w_E,headroom (see filter_cell()).
"""
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:
    sys.exit("Install scipy:  pip install scipy")

METRICS = ["sla_violation", "latency", "throughput", "utilization",
           "energy_eff", "fragmentation", "acceptance", "reliability",
           "isolation", "overhead"]


def cliffs_delta(a, b):
    """Cliff's delta for (a - b). Returns delta in [-1, 1]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    n = len(a) * len(b)
    return (gt - lt) / n if n else float("nan")


def cliff_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def cohens_d(a, b):
    """Cohen's d for (a - b) with pooled SD."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def cohen_magnitude(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def load(path, group=None):
    """method -> metric -> [values]; optionally only rows matching group dict."""
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            if group and any(
                abs(float(row[k]) - v) > 1e-9 for k, v in group.items()
            ):
                continue
            m = row["method"].strip()
            for k in METRICS:
                if row.get(k, "").strip():
                    data[m][k].append(float(row[k]))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--proposed", default="HAO-PRA-U")
    ap.add_argument("--metric", default=None,
                    help="restrict to one metric (default: all)")
    ap.add_argument("--group", default=None,
                    help="comma list col=val to filter, e.g. w_E=0.0188,headroom=1.4")
    args = ap.parse_args()

    group = None
    if args.group:
        group = {}
        for tok in args.group.split(","):
            k, v = tok.split("=")
            group[k.strip()] = float(v)

    data = load(args.csv, group)
    if args.proposed not in data:
        sys.exit(f"No rows with method == '{args.proposed}'.")

    metrics = [args.metric] if args.metric else METRICS
    baselines = sorted(m for m in data if m != args.proposed)

    hdr = (f"{'baseline':<26}{'metric':<14}{'t':>8}{'p':>9}"
           f"{'cohen_d':>9} {'(mag)':<11}{'cliff':>8} {'(mag)':<11}")
    print(hdr)
    print("-" * len(hdr))
    for b in baselines:
        for k in metrics:
            a = data[args.proposed].get(k, [])
            c = data[b].get(k, [])
            if len(a) >= 2 and len(c) >= 2:
                t, p = stats.ttest_ind(a, c, equal_var=False)
                cd = cohens_d(a, c)
                cl = cliffs_delta(a, c)
                print(f"{b:<26}{k:<14}{t:>8.3f}{p:>9.4f}"
                      f"{cd:>9.3f} {('('+cohen_magnitude(cd)+')'):<11}"
                      f"{cl:>8.3f} {('('+cliff_magnitude(cl)+')'):<11}")


if __name__ == "__main__":
    main()
