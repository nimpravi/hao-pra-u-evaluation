# PRE-REGISTRATION: Regime-shift study under symmetric conditions, corrected (Study 16)

**Status:** FROZEN BEFORE RESULTS. Content-hash this document, record the hash
in `preregistrations/hashes.txt`, commit it, and deposit it with an
independent timestamping service **before** `shift_symmetric_v2_driver.py` is
executed. No section is edited after the first run; corrections are appended
as dated amendments.

**Date:** 10/18/2026

**Independent timestamp:** _(DOI or receipt, filled in before running)_

**Supersedes:** nothing. Study 15 is void under its own registered rule and
stays in the record as void. This is a separate study with a separate
registration, because a registration cannot be amended into a study that
passes.

---

## 1. Why this study exists

Study 15 was registered to test staleness under symmetric conditions. Its
S0 neutrality control failed, so under its registered rule the study is void
and no result from it is reported. `AMENDMENT_A2_SHIFT_SYMMETRIC.md` records
the outcome and the cause: the driver built the forecast array over the
frozen window only and padded the shift window with `NaN`, and every
forecast-consuming branch of `sim.allocate` falls back to the realized
demand when the forecast is not finite. Four of seven schemes therefore
allocated against true demand for all 300 post-shift intervals.

This study is the corrected run. The question it asks is unchanged from
Study 15, and from Study 2 before it.

We restate what makes it hard to call. Sec. V-B records that the framework
takes the largest available reserve wherever margin is cheap, which is the
condition at the anchor regime. Under equal budgets the framework may
degrade differently than it did in Study 2. We do not know the direction and
we are not predicting one.

We are also not blind. Under the asymmetric conditions of Study 2, tuned
static over-provisioning won the post-shift comparison and the framework's
degradation was the larger. Study 2's protocol is what this study exists to
correct, so that result carries no weight as a prior. Study 15 produced no
scorable result, so it supplies no prior either; the only thing we know from
it is the direction of its defect, which favoured the framework.

## 2. Hypotheses

Two-sided throughout. Carried over from Study 15 unchanged.

- **H1 (primary).** Under equal search budgets and a symmetric protection
  range, tuned static over-provisioning attains the better post-shift
  composite than HAO-PRA-U in a majority of the four registered shift
  scenarios.
- **H2 (primary).** The framework's post-shift degradation, measured as the
  change in SLA violation from the pre-shift to the post-shift window, is
  larger than tuned Static's in a majority of the four scenarios.
- **H3 (secondary, gate).** The neutrality control S0 reproduces the
  pre-shift window to within Monte Carlo error, confirming that the trace
  extension and the evaluation pipeline introduce no artifact.

H3's scope is widened deliberately. Study 15's H3 was justified as a test
that "the trace extension introduces no artifact," but the statistic it used
ranges over the whole pipeline, and that is what caught the defect. We now
say so in the hypothesis rather than only in the rule.

## 3. Design, fixed here

| item | value |
|---|---|
| Simulator | `hao_pra_u_sim.py`, SHA-256 beginning `30bedec8`, unmodified |
| Driver | `shift_symmetric_v2_driver.py`, hash recorded alongside this document |
| Trace | 1300 intervals: 0--999 from the frozen generator, 1000--1299 the registered scenario extension |
| Forecaster | one GRU per slice, fitted on intervals 0--549 only, predicting 550--1299 in one pass, never refitted |
| Windows | tune 550--699, pre-shift 700--999, post-shift 1000--1299 |
| Scenarios | `S1a_load15`, `S1b_load30`, `S2_mixshift`, `S3_bursts`, plus `S0_noshift` as the neutrality control |
| Schemes | all seven in the build are run. The **six Study-2 schemes** are the primary set: Static, Reactive, Centralized, Centralized Predictive, Flat-Agent, HAO-PRA-U |
| Seeds | 0 to 9, matching Study 2 and Study 15 so the three are paired |
| Search budget | 9 validation evaluations per scheme, every scheme, on 550--699 |
| Protection range | symmetric; HAO multiplier grid capped at $\{0.5, 0.75, 1.0\}$ |
| Freeze | all parameters frozen at the end of the tuning window and never re-tuned, in particular not after the shift |

The forecaster row is the only design change from Study 15, and it is a
correction rather than a choice. `train_gru_forecasts` fits on intervals
below `N_TRAIN` (550) and predicts every interval it is given, so passing
the full 1300 trains on exactly the same 550 intervals as before and extends
the prediction through the shift. The forecaster is stale after interval 550
by construction, which is what the study tests. Study 15 gave it no
prediction past 999 at all, which is not staleness but clairvoyance via the
fallback.

Forecasts over intervals 0 to 999 are unchanged to within 1.5e-05 absolute,
a float32 batched-inference difference and not a change of model, so the
tuning window and pre-shift window remain comparable to Study 15's and to
Study 2's. The driver reports any tuned parameter that differs from Study
15's rather than assuming none do.

**Scheme set.** Amendment A1 to Study 15 established that every scheme in
the build is run and both composites are formed at analysis time, so that no
scheme-set choice is made after a failure. That stands here. The six-scheme
composite is primary, because it is the one comparable with Study 2; the
seven-scheme composite is reported beside it. $\bar{J}$ is set-independent
and unaffected either way.

We record in advance that HAO-PPO's pre-shift window was not reproducible
across scenarios in Study 15: 114 metric values differed where the shared
intervals 0--999 require them to be identical, and all 114 were HAO-PPO's.
HAO-PPO is not in the primary set, so this does not affect the primary
composite. If the same irreproducibility appears in this run, the
seven-scheme composite is reported with that fact stated and is not used to
decide anything.

## 4. Decision rule, fixed here

**Gate, evaluated before H1 and H2.**

- For every scheme in the build and every one of the ten scored metrics, S0
  passes if
  $$|\overline{x}_\text{post} - \overline{x}_\text{pre}| \le \max\!\left(s_\text{pre},\ 0.02\,|\overline{x}_\text{pre}|\right)$$
  where $s_\text{pre}$ is the across-seed standard deviation of the
  pre-shift window.
- **Directional check.** No forecast-consuming scheme (Centralized
  Predictive, Flat-Agent, HAO-PRA-U, HAO-PPO) may *improve* on SLA violation
  from the pre- to the post-shift window by more than the same tolerance.
  Improvement on a stationary trace is the signature the Study 15 defect
  left, and a symmetric bound alone does not name a direction.
- **H3 is supported** if both checks pass for every scheme. **If S0 fails,
  the study is void and no result from it is reported.**

The relative floor is new and is registered here, before the run. Study 15's
rule compared against the standard deviation alone, and three of its six
failing metrics had zero or near-zero across-seed variance, where any
nonzero drift fails regardless of magnitude. We verified before freezing
this document that the floor does not admit the Study 15 defect: with the
floor in place, that defect still fails the gate on SLA violation and on
fragmentation. We state the verification so that the floor cannot later be
read as a weakening chosen to let a known failure through.

**Primary tests, evaluated only if the gate passes.**

- **Test:** paired Wilcoxon signed-rank across the 10 seeds within each
  scenario, two-sided. The exact two-sided floor at $n=10$ is
  $2/2^{10} \approx 0.002$.
- **Correction:** Holm--Bonferroni across the four result scenarios. S0 is a
  control and is not in the family.
- **Threshold:** $\alpha = 0.05$ after correction.
- **Effect size:** Cohen's $d_z$, reported with every comparison.
- **H1 is supported** if tuned Static has the better mean post-shift
  composite in at least 3 of 4 scenarios. **Refuted** if HAO-PRA-U does.
  Two-all is **inconclusive**.
- **H2** uses the same rule on the pre-to-post change in SLA violation.

## 5. What is NOT permitted after results are observed

- Re-tuning any parameter after the shift, which would convert a staleness
  test into an adaptation test.
- Refitting the forecaster after interval 549, for any reason.
- Changing the scenario set, the windows, the seeds, or the budget.
- Dropping a scenario because its outcome is inconvenient, including S0.
- Changing the gate tolerance, in either direction, or dropping a metric
  from it.
- Switching the primary statistic between composite and $\bar{J}$. Both are
  reported; the composite decides, matching Study 2.
- Making the primary set seven schemes, or any set other than the six named
  in Sec. 3.
- Presenting a refutation of H1 or H2 as a limitation rather than a finding.
  If the framework degrades more gracefully than tuned Static under
  symmetric conditions, that is a result and it is reported as one in the
  abstract.
- Reporting this study without reporting that Study 15 preceded it and was
  void. A corrected study that conceals its predecessor is a study with an
  unreported attempt.

## 6. Outputs

`results/nested_validation/all_windows_symmetric_v2.csv`,
`recovery_curves_symmetric_v2.csv` and
`tuned_params_shift_symmetric_v2.json`, committed with the analysis script
and the console output. The Study 15 artifacts are not overwritten and stay
in the repository as the record of the void run.
