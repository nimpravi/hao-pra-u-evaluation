#!/usr/bin/env python3
"""Print the verified HAO-PRA-U anchor row for Table tab:anchor from your
archived Phase B sweep output. Usage:
    python3 make_anchor_row.py path/to/sweep_results/all_cells.csv
"""
import csv, sys
import numpy as np
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["method"] == "HAO-PRA-U"
        and abs(float(r["w_E"]) - 0.0188) < 1e-9
        and abs(float(r["headroom"]) - 2.0) < 1e-9]
assert len(rows) == 10, f"expected 10 seeds, got {len(rows)}"
def ms(k, f=4):
    v = [float(r[k]) for r in rows]
    return f"{np.mean(v):.{f}f}\\pm{np.std(v, ddof=1):.{f}f}"
print("HAO-PRA-U                & "
      f"${ms('sla_violation')}$ & ${ms('latency',2)}$ & ${ms('throughput',1)}$ & "
      f"${ms('utilization',3)}$ & ${ms('energy_eff',3)}$ & ${ms('acceptance',3)}$ \\\\")
