# Reproducibility notes

## Environment
- Python 3.10+ (tested on 3.12)
- Install: `pip install -r requirements.txt`
- Results are seeded; exact bit-reproduction depends on library versions
  (see `requirements.txt`). The decision-level conclusions are robust to
  minor version drift; per-seed floats may differ at the last digits.

## Integrity checks
- `run_experiment3.py` computes SHA-256 of `precursor_traffic.py` and
  `hao_pra_u_sim.py` and compares to the values pinned in the
  pre-registration (`preregistrations/hashes.txt`). Any modification to the
  frozen design aborts the run. This is intentional: the committed code is
  byte-identical to the registered, frozen design.
- Verify ALL frozen artifacts at once, from the repo root:
  - Linux/macOS: `sha256sum -c preregistrations/hashes.txt`
  - This covers every study's sim copy (Experiments 1, 2, 3) plus the
    Experiment-3 precursor generator.
- Or verify one file by hand:
  - Linux/macOS: `sha256sum studies/predictability/precursor_traffic.py`
  - Windows: `certutil -hashfile studies\predictability\precursor_traffic.py SHA256`

## Command → output file map (Experiment 3)
| Command | Writes |
|---|---|
| `python run_experiment3.py --workers N` | `exp3_results/results_master.csv`, `results_per_seed__p<level>.csv`, `forecast_diag.csv`, `checkpoints/` |
| `python ../../scripts/significance_test.py results_per_seed__p1.0.csv` | stdout: per-metric Welch tests |
| `python analyze_experiment3.py` | `exp3_results/decision_report_exp3.json` + stdout summary |

The committed files under `results/predictability/` are the published run's
outputs. To regenerate from scratch, delete `exp3_results/` (including
`checkpoints/`) before re-running so no prior cell is reused.

## Command → output file map (Experiment 2)
| Command | Writes |
|---|---|
| `python shift_driver.py --workers N` | `shift_results/all_windows.csv`, `recovery_curves.csv` |
| `python analyze_shift.py` | `shift_results/decision_report_shift.json` + stdout |

## Notes on the predictability generator
- `predictability=0` recovers the memoryless Bernoulli burst law; this is the
  registered tripwire and must reproduce the regime-sweep ordering.
- The marginal burst rate is held at 0.05 for every predictability level, so
  the knob varies anticipability only, not burst frequency.
- The precursor is forecaster-only; demand-only baselines are verified
  precursor-invariant at runtime (the fairness gate).

## Command → output file map (Experiment 1, regime sweep)
| Command | Writes |
|---|---|
| `python sweep_driver.py --workers N` | `uhat_cache/seed*.npz`, `sweep_results/all_cells.csv`, `tuned_params.json` |
| `python analyze_sweep.py` | `sweep_results/decision_report.json` + stdout winner map |

GRU forecasts are regime-independent and cached once per seed in
`uhat_cache/`. The committed `results/regime_sweep/sweep.log` records the
completed 200-cell run as provenance.
