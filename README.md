# A Pre-Registered Protocol for Evaluating Learned Cross-Domain Slice Orchestration in 6G

<!-- Zenodo: create a release on GitHub, link the repo in Zenodo, then paste
     the badge Markdown Zenodo gives you, replacing the two placeholders below.
     Use the CONCEPT DOI (always resolves to the latest version), not a
     version-specific DOI. -->
<!--[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX) -->

This repository contains the simulator, pre-registrations, run/analysis
harnesses, and committed result files for a pre-registered
evaluation of a hierarchical agentic slice-orchestration framework
(HAO-PRA-U) against per-regime-tuned baselines.

**Headline finding.** Across a 20-regime sweep, a regime-shift study, and a
predictability sweep, the learned hierarchical framework does **not** beat
well-tuned static over-provisioning. Tuned over-provisioning wins 18 of 20
registered regimes decisively; under regime shift, simple demand-following
adapts best; and when bursts are made anticipable, the framework still wins
no predictability level while reactive control absorbs the benefit. The
contribution is a rigorous, reproducible **negative result** and a set of
strong, reusable baselines.

---

## Why pre-registration

Each study's hypotheses, seeds, parameter grid, statistical decision rule,
and metrics were frozen and timestamped **before** results were observed.
The run harness verifies the SHA-256 of the frozen simulator and generator
against the values pinned in the pre-registration and **refuses to run** on
a mismatch. This makes "registered before results" mechanically checkable
rather than a claim of good faith. See `preregistrations/` and
`preregistrations/hashes.txt`.

---

## Repository layout

```
hao-pra-u-evaluation/
├── preregistrations/        # frozen study designs + pinned artifact hashes
│   ├── SWEEP_PREREGISTRATION.md          # Experiment 1 (dated 2026-06-11)
│   ├── SHIFT_STUDY_PREREGISTRATION.md    # Experiment 2 (dated 2026-06-12)
│   ├── EXPERIMENT3_PREREGISTRATION.md    # Experiment 3 (dated 2026-06-13)
│   └── hashes.txt
├── scripts/                 # shared analysis (Welch significance test, etc.)
├── studies/
│   ├── regime_sweep/        # Experiment 1: 5x4 sweep driver + analysis (+ frozen sim copy)
│   ├── regime_shift/        # Experiment 2: driver + analysis (+ frozen sim copy)
│   └── predictability/      # Experiment 3: generator, run, analysis (+ frozen sim copy)
├── results/                 # committed CSV/JSON that back every number in the paper
│   ├── regime_sweep/
│   ├── regime_shift/
│   └── predictability/
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

**On the duplicated simulator.** `hao_pra_u_sim.py` is intentionally
co-located (vendored) with each study's harness rather than placed in a
shared package directory. This keeps the frozen file **byte-identical**
wherever it is imported, so the SHA-256 pinned in the pre-registration holds
at every call site. All copies share the same hash
(`24f20b81…`); see `preregistrations/hashes.txt`.

---

## Reproducing the results

Requires Python 3.10+ and the packages in `requirements.txt`
(`pip install -r requirements.txt`).

### Experiment 1 — regime sweep (5x4 grid)
```bash
cd studies/regime_sweep
python sweep_driver.py --validate             # anchor cell, seeds 0-1
python sweep_driver.py --workers 8            # full 5x4x10 = 200-cell sweep
python analyze_sweep.py
```

### Experiment 3 — predictability sweep
```bash
cd studies/predictability
python run_experiment3.py --validate          # tripwire: p=0 must show no HAO win
python run_experiment3.py --workers 4         # full 6 x 12 study (heavy)
for f in exp3_results/results_per_seed__p*.csv; do
    python ../../scripts/significance_test.py "$f"
done
python analyze_experiment3.py                 # applies the registered decision rule
```

### Experiment 2 — regime shift
```bash
cd studies/regime_shift
python shift_driver.py --validate             # S0 neutrality control
python shift_driver.py --workers 3            # full study
python analyze_shift.py
```

The harness verifies frozen-artifact hashes on startup and will abort if the
simulator or generator has been modified from the registered design.

---

## Claim-to-artifact map

Every headline number in the paper traces to a committed file produced by a
named command. (Numbers below are from the committed result files in this
repository.)

| Claim in paper | Value | Artifact | Produced by |
|---|---|---|---|
| Tuned static wins the regime sweep | 18/20 decisive | `results/regime_sweep/decision_report.json` | `sweep_driver.py` → `analyze_sweep.py` |
| Under shift, demand-following adapts best; learned hierarchy among worst | GATE 2 outcome | `results/regime_shift/decision_report_shift.json` | `shift_driver.py` → `analyze_shift.py` |
| Predictability sweep: HAO-PRA-U wins **no** level | win_levels = [] | `results/predictability/decision_report_exp3.json` | `analyze_experiment3.py` |
| Forecaster anticipation rises with predictability | AUC 0.48 → 0.69 (p=0 → 1.0) | `results/predictability/forecast_diag.csv` | `run_experiment3.py` |
| Static's lead over Reactive erodes to non-decisive at p=1.0 | p_Welch = 0.105 | `results/predictability/decision_report_exp3.json` | `analyze_experiment3.py` |
| HAO reduces SLA violations but pays a fixed fragmentation cost | SLA t→−3.4; frag t≈13–14 | per-level `results_per_seed__p*.csv` | `significance_test.py` |
| Effect sizes accompany every confirmatory test | e.g. tight-capacity Static vs HAO d=2.61, δ=1.00 | `results/regime_sweep/all_cells.csv` | `effect_sizes.py` |
| Component ablation localizes failure to the architecture, not the predictor | CP (predictor only) best SLA 0.0090 | `results/regime_sweep/decision_report.json` (anchor cell) | `analyze_sweep.py` |
| Negative result robust to objective-weight perturbation | HAO wins 0/20 under all ±50% α₁,α₂ weightings | `results/regime_sweep/all_cells.csv` | `objective_sensitivity.py` |
| p=0 control reproduces memoryless ordering (no regression) | tripwire clean | `decision_report_exp3.json` | `analyze_experiment3.py` |

> The full per-seed predictability CSVs (`results_per_seed__p0.0.csv` …
> `p1.0.csv`) and `results_master.csv` are regenerated by the full
> `run_experiment3.py` run; commit them alongside the report files for a
> complete record.

---

## Paper

The manuscript is **not** included in this repository (publisher copyright).
This repo holds the code, data, and pre-registrations that back the paper.

> Hierarchical Agentic Orchestration for Cross-Domain Zero-Touch Slicing:
> A Pre-Registered Evaluation Against Tuned Baselines for 6G.
> IEEE Transactions on Network and Service Management (under review / DOI to appear).

See `CITATION.cff` for citation metadata.

---

## Citation

If you use the simulator or the baselines, please cite **both** the paper
and the archived software.

**Paper** — see `CITATION.cff` for full metadata.

**Software (this repository)** — archived on Zenodo:

> Hegde, P. (2026). *Hierarchical Agentic Orchestration for
> Cross-Domain Zero-Touch Slicing: evaluation code, pre-registrations, and
> data* [Software]. Zenodo. https://doi.org/XX.XXXX/zenodo.XXXXXXX

<!-- Replace XX.XXXX/zenodo.XXXXXXX above and in the badge with the DOI
     Zenodo assigns after you archive the repo. The concept DOI cites all
     versions; a version DOI cites one specific release. -->

## License

MIT — see `LICENSE`.
