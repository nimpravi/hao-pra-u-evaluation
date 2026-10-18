# PRE-REGISTRATION: Regime-shift study under symmetric conditions (Study 15)

**Status:** FROZEN BEFORE RESULTS. Content-hash this document, record the hash
in `preregistrations/hashes.txt`, commit it, and deposit it with an
independent timestamping service **before** `shift_symmetric_driver.py` is
executed. No section is edited after the first confirmatory run; corrections
are appended as dated amendments.

**Date:** 10/17/2026

**Independent timestamp:** https://doi.org/10.5281/zenodo.23444965

---

## 1. Why this study exists

Staleness testing is one of the three components of the protocol in
Sec. III. Its only demonstration in this paper is Study 2, which was run
under the asymmetric conditions Study 4 exists to remove: the baselines were
tuned per regime and the framework was not, the split was 70/30 rather than
nested, and the simulator was the earlier build.

The manuscript currently labels Study 2's result conditional on the protocol
that produced it, and declines to predict what re-running would show. That
is honest but unsatisfying for the component the paper argues hardest for.
This study removes the conditional.

We state in advance what makes this study hard to call: Sec. V-B records
that the framework takes the largest available reserve wherever margin is
cheap, which is the condition at the anchor regime where the shift study
runs. Under equal budgets the framework may therefore degrade differently
than it did. We do not know the direction and we are not predicting one.

## 2. Hypotheses

Two-sided throughout.

- **H1 (primary).** Under equal search budgets and a symmetric protection
  range, tuned static over-provisioning attains the better post-shift
  composite than HAO-PRA-U in a majority of the four registered shift
  scenarios.
- **H2 (primary).** The framework's post-shift degradation, measured as the
  change in SLA violation from the pre-shift to the post-shift window,
  is larger than tuned Static's in a majority of the four scenarios.
- **H3 (secondary).** The neutrality control S0 reproduces the pre-shift
  window to within Monte Carlo error, confirming that the trace extension
  introduces no artifact.

H1 is the registered decision rule of the original Study 2, carried over
unchanged so the two studies are comparable. H2 is the quantity a reader
intuitively means by "degrades gracefully," and the original study's
registration did not make it primary; we make it primary here and say so.

## 3. Design, fixed here

| item | value |
|---|---|
| Simulator | `hao_pra_u_sim.py`, SHA-256 beginning `30bedec8`, unmodified |
| Trace | 1300 intervals: 0--999 from the frozen generator, 1000--1299 the registered scenario extension |
| Windows | tune 550--699, pre-shift 700--999, post-shift 1000--1299 |
| Scenarios | `S1a_load15`, `S1b_load30`, `S2_mixshift`, `S3_bursts`, plus `S0_noshift` as the neutrality control |
| Seeds | 0 to 9, matching the original Study 2 so the two are paired |
| Search budget | 9 validation evaluations per scheme, every scheme, on 550--699 |
| Protection range | symmetric; HAO multiplier grid capped at $\{0.5, 0.75, 1.0\}$ |
| Freeze | all parameters frozen at the end of the tuning window and never re-tuned, in particular not after the shift |

The seeds deliberately match the original study. This is a re-run under
changed conditions, not an independent replication, and pairing the seeds
isolates the conditions.

## 4. Decision rule, fixed here

- **Test:** paired Wilcoxon signed-rank across the 10 seeds within each
  scenario, two-sided.
- **Correction:** Holm--Bonferroni across the four result scenarios. S0 is a
  control and is not in the family.
- **Threshold:** $\alpha = 0.05$ after correction.
- **H1 is supported** if tuned Static has the better mean post-shift
  composite in at least 3 of 4 scenarios. **Refuted** if HAO-PRA-U does.
  Two-all is **inconclusive**.
- **H2** uses the same rule on the pre-to-post change in SLA violation.
- **H3 is supported** if, in S0, every scheme's post-window metrics fall
  within the pre-window's across-seed standard deviation. If S0 fails, the
  study is void and no result from it is reported.

## 5. What is NOT permitted after results are observed

- Re-tuning any parameter after the shift, which would convert a staleness
  test into an adaptation test.
- Changing the scenario set, the windows, the seeds, or the budget.
- Dropping a scenario because its outcome is inconvenient, including S0.
- Switching the primary statistic between composite and $\bar{J}$. Both are
  reported; the composite decides, matching the original Study 2.
- Presenting a refutation of H1 or H2 as a limitation rather than a finding.
  If the framework degrades more gracefully than tuned Static under symmetric
  conditions, that is a result and it is reported as one in the abstract.

## 6. Disclosure of known results

Under the asymmetric conditions of the original Study 2, tuned static
over-provisioning won the post-shift comparison and the framework's
degradation was the larger. We are therefore not blind to the direction, and
we record here that the original study's protocol is exactly what this study
exists to correct, so its result carries no weight as a prior.

## 7. Outputs

`results/nested_validation/all_windows_symmetric.csv`,
`recovery_curves_symmetric.csv` and `tuned_params_shift_symmetric.json`,
committed with the analysis script and the console output.
