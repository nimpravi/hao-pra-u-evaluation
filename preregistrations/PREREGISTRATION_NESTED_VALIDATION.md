# PRE-REGISTRATION: Equal-Budget Nested Validation (Study 4)

**Status:** FROZEN BEFORE RESULTS. This document must be committed, and its
SHA-256 recorded in `preregistrations/hashes.txt`, before any run of
`nested_driver.py` is executed. No section may be edited after the first
confirmatory run; corrections are made by appending a dated amendment
section at the end, as in the previous design freezes.

**Supersedes:** the tuning protocol of `SWEEP_PREREGISTRATION.md`, which is
retained unchanged as the record of the original study.

---

## 1. Motivation

The original protocol claimed symmetric baseline strengthening. Audit of
the released simulator shows the claim does not hold for model selection.
Per (regime, seed), `tune_baselines()` performs a grid search for every
baseline, while HAO-PRA-U reads no tuned parameter at all:

| Scheme | Tuned parameters | Validation evaluations |
|---|---|---|
| Static | margin | 7 |
| Reactive | margin | 7 |
| Centralized | period x margin | 9 |
| Centralized Predictive | margin | 7 |
| Flat-Agent | per-domain headroom | 6 |
| HAO-PRA-U | none | **0** |

HAO-PRA-U's discount factor, learning-rate schedule, replay depth, and
SLA-risk weights are fixed module constants. The comparison is therefore
symmetric in *protection range* (all schemes may over-provision within
[1.00, 1.40]) but asymmetric in *search budget*: baselines are fitted to
each regime, the proposed framework is not.

Two readings follow, and the original study cannot distinguish them:

- **R-A:** tuned over-provisioning genuinely dominates this environment.
- **R-B:** tuned over-provisioning dominates partly because it alone
  received per-regime model selection.

This study is designed to separate them.

A second defect is addressed simultaneously. Baseline tuning currently
maximizes reward on intervals 0-699, the same intervals used to fit the GRU
forecaster. Hyperparameter selection and model fitting therefore share data.
This study introduces a held-out validation window.

---

## 2. Hypotheses (registered before observation)

- **H4.1 (primary).** Under equal per-scheme search budgets and nested
  validation, per-regime-tuned static over-provisioning still wins a
  majority of the 20 registered regimes decisively.
- **H4.2.** Under the same conditions, HAO-PRA-U wins strictly more than
  zero regimes.
- **H4.3.** The number of regimes won by HAO-PRA-U under equal budgets is
  greater than under the original asymmetric protocol (0 of 20).

**Interpretation fixed in advance.** If H4.1 holds and H4.2 fails, reading
R-A is supported and the original conclusion stands with its fairness
objection answered. If H4.2 or H4.3 holds, reading R-B is supported to that
extent, and the paper's central claim must be narrowed accordingly. We
commit now to reporting and honoring either outcome. **A result in which
HAO-PRA-U improves materially under equal budgets is a publishable and
reportable finding, not a failure of the study, and will not be reframed.**

---

## 3. Nested validation protocol

The 1000-interval trace is partitioned into three contiguous windows:

| Window | Intervals | Purpose |
|---|---|---|
| Train | 0-549 (55%) | fit the GRU forecaster only |
| Validation | 550-699 (15%) | select all free parameters, every scheme |
| Test | 700-999 (30%) | frozen confirmatory evaluation |

Rules, fixed in advance:

1. The GRU is fitted on the train window only. Its architecture and
   hyperparameters are unchanged from the original study and are **not**
   tuned here, for either the predictive baselines or HAO-PRA-U, so that
   the forecaster remains a shared, constant component.
2. All free parameters of all six schemes are selected on the validation
   window by maximizing mean reward, using the identical scoring function
   `reward_from_metrics()` for every scheme.
3. Selected parameters are frozen before the test window is touched.
4. No parameter, grid, or rule in this document may be revised after any
   test-window result is observed.
5. The test window (700-999) is identical to the original study's test
   window, so test-set results remain comparable across studies.

---

## 4. Equal search budgets

Every scheme receives **exactly 9 validation evaluations** per
(regime, seed). Nine is chosen because it is the maximum budget any scheme
received under the original protocol (Centralized, 3 x 3), so no baseline
is weakened relative to the published study.

### Baseline grids (9 points each)

| Scheme | Grid | Count |
|---|---|---|
| Static | margin in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40} | 9 |
| Reactive | margin in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40} | 9 |
| Centralized | period in {3, 5, 8} x margin in {1.05, 1.10, 1.15} | 9 |
| Centralized Predictive | margin in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40} | 9 |
| Flat-Agent | uniform headroom in {1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40} | 9 |

Static, Reactive, Centralized Predictive, and Flat-Agent grids are widened
from 7, 7, 7, and 6 points to 9 by inserting {1.25, 1.35} (and 1.40 for
Flat-Agent). Every previously searched value remains in the grid, so the
baselines can only improve relative to the published study, never degrade.

### HAO-PRA-U grid (9 points)

HAO-PRA-U's two most influential free parameters, neither previously
searched:

| Parameter | Values | Meaning |
|---|---|---|
| `Q_GAMMA` | {0.85, 0.90, 0.95} | discount factor of the reserve learner |
| `RESERVE_SCALE` | {0.5, 1.0, 1.5} | multiplier on the Q-selected reserve factor rho |

Full factorial, 3 x 3 = 9. The registered original configuration
(`Q_GAMMA` = 0.90, `RESERVE_SCALE` = 1.0) is a grid point, so HAO-PRA-U
cannot perform worse than its published configuration except through
validation noise.

`RESERVE_SCALE` multiplies the reserve factor selected by the Q-learner,
`rho_eff = clip(RESERVE_SCALE * rho, 0, 0.60)`, recovering the published
behaviour exactly at `RESERVE_SCALE` = 1.0. It is the direct analogue of
the margin the baselines tune: it controls provisioning magnitude, which
this environment rewards most.

**Knob selection, recorded before the confirmatory run.** An earlier
candidate second parameter, a multiplier on the SLA-risk term in the
domain priority (19), was rejected during harness validation. Because the
risk term acts only through the priority sort, and the sort binds only when
capacity is exhausted, that multiplier is numerically inert for h >= 1.7 and
moves SLA violation only in the third decimal at h = 1.4. It would have
given HAO-PRA-U a nominal budget of 9 but an effective budget of 3. This
rejection was made on mechanism, before any confirmatory result was
observed, and is recorded here for audit.

**Parameters deliberately NOT tuned**, for either side: GRU architecture and
training, reserve action set, state discretization thresholds, replay depth
(30 epochs), objective weights alpha, and the SLA-risk weights w1..w6. These
remain fixed at their registered values so that this study isolates the
effect of equalizing search budget, not of enlarging the proposed method's
design space.

---

## 5. Experimental design

Unchanged from the original sweep except as stated above:

- Regimes: 5 x 4 factorial, `w_E` in {0.0188, 0.05, 0.10, 0.20, 0.30},
  `h` in {1.4, 1.7, 2.0, 2.3}. 20 regimes.
- Seeds: 10 per regime, seeds 0-9, identical traffic generation.
- Tuning is repeated independently for every (regime, seed) pair, for every
  scheme, on that seed's validation window.
- Total validation evaluations: 20 regimes x 10 seeds x 6 schemes x 9
  points = 10,800.

---

## 6. Statistical decision rules

Fixed in advance, and revised from the original study in one respect
following the Study-2 re-analysis:

1. A scheme **wins a regime** if it has the highest mean composite score
   over the 10 seeds AND is significantly better than the runner-up.
2. Significance uses the **Wilcoxon signed-rank test**, paired by seed,
   at alpha = 0.05. Paired testing is registered as primary because the
   composite is min-max normalized across schemes within each seed and is
   therefore paired by construction.
3. The paired t-test and Welch's t-test are reported alongside as
   secondary, for comparability with the published study.
4. Family-wise error across the 20 regimes is controlled with
   Holm-Bonferroni; false-discovery rate with Benjamini-Hochberg. Both are
   reported.
5. Effect sizes accompany every confirmatory comparison: Cohen's d_z
   (paired) as primary, with Cliff's delta and independent-sample Cohen's d
   reported for comparability.
6. Composite construction is unchanged (per-seed min-max across the six
   schemes, weights of Table I).

**Note on the Wilcoxon floor.** With n = 10 seeds the two-sided Wilcoxon
p-value cannot fall below 1.95e-3. Regimes attaining that floor are reported
as p <= 1.95e-3 and are not distinguished from one another.

---

## 7. Tripwires

Registered checks that indicate harness error rather than a finding:

- **T1.** At the anchor regime (`w_E` = 0.0188, `h` = 2.0), selecting
  `Q_GAMMA` = 0.90 and `RISK_SCALE` = 1.0 must reproduce the published
  HAO-PRA-U test-window behaviour to within seed noise. A gross mismatch
  indicates a patching error.
- **T2.** Baseline margins selected on the validation window must lie in
  the registered grid. Any out-of-grid value indicates an implementation
  fault.
- **T3.** If any scheme receives other than exactly 9 validation
  evaluations, the run is void.

Tripwire failures void the run; they are not reported as results.

---

## 8. Artifacts

On completion the following are committed:

- `studies/nested_validation/nested_driver.py`
- `studies/nested_validation/hao_pra_u_sim.py` (frozen copy, hash pinned)
- `results/nested_validation/all_cells_nested.csv` (per regime, seed, scheme)
- `results/nested_validation/tuned_params_nested.json` (selected parameters
  per regime, seed, scheme)
- `results/nested_validation/decision_report_nested.json`

The SHA-256 of this document and of the frozen simulator copy are recorded
in `preregistrations/hashes.txt` before the first confirmatory run.

---

## 9. Relationship to reported results

Whatever the outcome, the paper will report both protocols: the original
asymmetric-budget study and this equal-budget study, with the difference
between them stated explicitly. The original study is not withdrawn or
retro-fitted; it is presented as what it was, with its fairness limitation
identified by peer review and quantified here.

---

*Registered by: Praveen Hegde*
*Date frozen: __________ (fill before first run)*
*SHA-256 of this file recorded in preregistrations/hashes.txt*
