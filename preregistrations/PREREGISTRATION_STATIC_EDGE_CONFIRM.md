# PRE-REGISTRATION: Confirmatory replication of Static + edge correction (Study 14)

**Status:** FROZEN BEFORE RESULTS. This document must be content-hashed, its
hash recorded in `preregistrations/hashes.txt`, committed, and deposited with
an independent timestamping service **before** `confirm_static_edge_driver.py`
is executed on any seed in the confirmatory set. No section may be edited
after the first confirmatory run; corrections are appended as dated
amendments.

**Date:** 10/17/2026

**Independent timestamp:** https://doi.org/10.5281/zenodo.23444965
running)_

**Protocol:** Sec. III-A of the manuscript as amended, which requires the
registration document itself to be hashed and deposited independently of the
authors, with no results in the same deposit.

---

## 1. Why this study exists

Sec. VII-G reports that tuned static margins plus edge correction beat every
other scheme evaluated, in all 20 regimes, under both statistics. That study
is **exploratory**: the scheme was specified after the Study-6 ablation
identified edge correction as the one component carrying value. It is also
the paper's only positive recommendation, which makes reporting it as
exploratory unsatisfying and makes an unregistered replication worthless.

This study replicates it on seeds never used in any prior study in this
paper, under a decision rule fixed here, in advance.

## 2. Hypotheses

Registered before observation. All are two-sided; we state the direction we
expect but the test does not assume it.

- **H1 (primary).** On the absolute cost statistic $\bar{J}$ with the
  registered fixed normalizers, `Static+Edge` has a lower mean than
  `Static Allocation` in a majority of the 20 registered regimes.
- **H2 (primary).** On $\bar{J}$, `Static+Edge` has a lower mean than
  `HAO-PRA-U` in a majority of the 20 registered regimes.
- **H3 (secondary).** `Static+Edge` and `Static+Edge-F` are not
  distinguishable: the mean absolute difference in $\bar{J}$ is below
  $0.001$, which is the smallest difference any comparison in this paper
  treats as meaningful.
- **H4 (secondary).** Mean SLA violation for `Static+Edge` is lower than for
  `Static Allocation`.

## 3. Design, fixed here

| item | value |
|---|---|
| Simulator | `hao_pra_u_sim.py`, SHA-256 beginning `30bedec8`, unmodified |
| Regimes | the 20 registered cells, $w_E \in \{0.0188, 0.05, 0.10, 0.20, 0.30\}$ by $h \in \{1.4, 1.7, 2.0, 2.3\}$ |
| **Seeds** | **11 to 30 inclusive, 20 seeds, none used in any prior study** |
| Split | nested 550 / 150 / 300, as Study 4 |
| Search budget | 9 validation evaluations per scheme per (regime, seed) |
| Protection range | symmetric; HAO multiplier grid capped at $\{0.5, 0.75, 1.0\}$ |
| Schemes | the six Study-4 schemes, plus `Static+Edge` and `Static+Edge-F` |
| Equivalence check | the ported edge block must reproduce HAO-PRA-U's allocation bitwise over 120 test intervals, or the run aborts |

Seeds 0 to 9 were used in Studies 4 to 13 and are excluded. Seed 10 is
excluded as a buffer against off-by-one reuse.

## 4. Decision rule, fixed here

- **Test:** paired Wilcoxon signed-rank on per-seed $\bar{J}$ within each
  regime, two-sided.
- **Correction:** Holm--Bonferroni across the 20 regimes, per hypothesis
  family. H1 and H2 are separate families.
- **Threshold:** $\alpha = 0.05$ after correction.
- **Effect size:** Cohen's $d_z$, reported with every comparison.
- **H1 is supported** if `Static+Edge` has the lower mean $\bar{J}$ in at
  least 11 of 20 regimes **and** at least one regime survives Holm. It is
  **refuted** if `Static Allocation` has the lower mean in at least 11 of 20.
  Any other outcome is recorded as **inconclusive**, and the exploratory
  result stands as exploratory.
- **H2** uses the same rule with `HAO-PRA-U` in place of
  `Static Allocation`.
- **H3 is supported** if the mean $|\Delta\bar{J}|$ between the two variants
  is below $0.001$. We do not test H3 for significance, because a
  significance test of a null difference at ten or twenty seeds answers a
  question we are not asking.

## 5. What is NOT permitted after results are observed

- Changing the seed set, the regime grid, the budget, the protection range or
  the scheme set.
- Changing the primary statistic from $\bar{J}$ to the composite, or the test
  from paired Wilcoxon, or the correction from Holm.
- Adding a scheme to the comparison set. The min--max composite is reported
  for continuity only and is explicitly **not** the decision statistic here,
  precisely because it moves when the set moves.
- Reporting the per-regime break-even analysis of Sec. VII-G as if it were
  registered. It is not, and it remains exploratory.
- Describing any outcome other than those in Sec. 4 as support for H1.

## 6. Disclosure of known results

We are not blind. The exploratory version of this study, on seeds 0 to 9,
found `Static+Edge` ahead of `Static Allocation` in 20 of 20 regimes with a
mean margin of $-0.00165$, and ahead of `HAO-PRA-U` in 20 of 20 with a mean
margin of $-0.02833$. We expect replication. Registering a study whose result
we expect is still worth doing: the alternative is a positive recommendation
that rests on a single exploratory pass, and if the replication fails that is
the single most important thing this paper could report.

We also record, so that it cannot be presented later as a discovery, that the
confound analysis in Sec. VII-G stands unchanged: `Static+Edge` inherits
Static's control-message count and control delay, and the margin against
`Static Allocation` reverses if one firing of the rule costs more than roughly
8 to 17 control messages.

## 7. Outputs

`results/nested_validation/all_cells_confirm_edge.csv` and
`tuned_params_confirm_edge.json`, both committed with the analysis script and
the console output.
