#!/usr/bin/env python3
"""
Significance testing for HAO-PRA-U vs. baselines (Welch's two-sided t-test).

USAGE
-----
1) Export your per-seed, run-level results to a CSV named results_per_seed.csv
   with columns:
       method, seed, sla_violation, latency, throughput, utilization,
       energy_eff, fragmentation, acceptance
   One row per (method, seed). 'method' values must include 'HAO-PRA-U' and
   each baseline name used in the paper.

2) Run:  python3 significance_test.py results_per_seed.csv

3) Paste the printed p-values into Section VII where marked <TBD>.
   ONLY claim significance where p < 0.05. If key comparisons are not
   significant at n=5 seeds, increase to 10+ seeds and re-run.

Welch's t-test is used because run-level variances may differ across methods.
With n=5 per group the test is low-powered; treat non-significance as
"insufficient evidence," not equivalence.
"""
import sys
import csv
from collections import defaultdict

try:
    from scipy import stats
except ImportError:
    sys.exit("Install scipy:  pip install scipy")

METRICS = ["sla_violation", "latency", "throughput", "utilization",
           "energy_eff", "fragmentation", "acceptance"]
PROPOSED = "HAO-PRA-U"

def main(path):
    data = defaultdict(lambda: defaultdict(list))  # method -> metric -> [values]
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"].strip()
            for k in METRICS:
                if row.get(k, "").strip():
                    data[m][k].append(float(row[k]))

    if PROPOSED not in data:
        sys.exit(f"No rows with method == '{PROPOSED}' found.")

    baselines = [m for m in data if m != PROPOSED]
    print(f"{'baseline':<28}{'metric':<16}{'t':>8}{'p':>10}  sig(p<0.05)")
    print("-" * 70)
    for b in sorted(baselines):
        for k in METRICS:
            a, c = data[PROPOSED][k], data[b][k]
            if len(a) >= 2 and len(c) >= 2:
                t, p = stats.ttest_ind(a, c, equal_var=False)
                print(f"{b:<28}{k:<16}{t:>8.3f}{p:>10.4f}  {'YES' if p < 0.05 else 'no'}")
    print("\nReport exact p-values in the paper; do not claim significance "
          "for any 'no' rows.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
