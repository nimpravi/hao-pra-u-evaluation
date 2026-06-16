# Committed results

This directory holds the result files that back the numbers in the paper.

## Present in this commit
- `predictability/decision_report_exp3.json` — Experiment 3 decision report
  (win levels, per-level composites, Welch p-values, verdict).
- `predictability/forecast_diag.csv` — per (scheme, seed, level) forecast
  diagnostics: burst-anticipation AUC, MAE/RMSE/MAPE, anticipation/false-
  alarm fractions.

## To add from your run (not fabricated here)
Copy these from your machine's run outputs so the record is complete:

- `regime_sweep/all_cells.csv` — one row per (cell, method, seed) across the
  5x4x10 sweep.
- `regime_sweep/decision_report.json` — Experiment 1 decision report
  (winner map, decisive wins, contiguous-block gate).
- `regime_sweep/tuned_params.json` — per-cell tuned baseline parameters.
  (`regime_sweep/sweep.log` is already committed as run provenance.)
- `predictability/results_master.csv` — all levels, all schemes, per seed.
- `predictability/results_per_seed__p0.0.csv` … `p1.0.csv` — one
  significance-test-ready CSV per predictability level.
- `regime_shift/all_windows.csv` — scenario × method × seed × window KPIs.
- `regime_shift/recovery_curves.csv` — per-interval violation series.
- `regime_shift/decision_report_shift.json` — Experiment 2 decision report.
- (regime sweep) the Experiment-1 sweep outputs and decision report.

These are produced by the commands in the top-level README and
`docs/reproducibility.md`. They were generated on the author's machine; this
repository does not synthesize result data.
