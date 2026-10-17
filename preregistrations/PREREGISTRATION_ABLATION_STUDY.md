# Study 6: Per-component ablation of the hierarchy

## PRE-REGISTRATION

**Status:** Freeze and hash before the confirmatory run.
**Builds on:** Studies 4 and 5. Tuned parameters, nested split, decision
rules and statistics are inherited unchanged.

### 1. Motivation

The published ablation ladder
(Centralized Predictive -> Flat-Agent -> HAO-PRA-U) varies several things
simultaneously: the global Q-learning reserve, risk-weighted allocation,
inter-domain negotiation and edge correction all arrive together at the
final rung. The paper's sharpest claim, that hierarchical organization
rather than prediction is what fails to pay, rests on that ladder and
cannot presently attribute the deficit to any specific component.

### 2. Design

Leave-one-out from the full hierarchy. Each component is removed while the
other three remain active, so the measured quantity is each component's
marginal contribution *in the presence of the others*, which is the
question asked. An additive ladder answers a different question and is not
used.

| Toggle | Component removed | Counterfactual |
|---|---|---|
| `abl_reserve` | learned reserve selection | fixed reserve rho = 0.20 (registered, not fitted) |
| `abl_risk` | SLA-risk weighting in the fill order | demand-proportional priority with intent bias retained |
| `abl_negotiate` | inter-domain capacity negotiation | step skipped |
| `abl_edge` | edge-level correction | step skipped |

The GRU forecaster and all tuned parameters are held at the Study 4
selections for the matching (regime, seed), so only architecture varies.
Removing the *learned* reserve while retaining *a* reserve separates
"learning which reserve to use" from "having a reserve at all"; conflating
the two was the flaw in the published ladder.

### 3. Hypotheses

- **H6.1.** At least one component has no measurable effect on any metric
  in any regime, i.e. it is inert.
- **H6.2.** The learned reserve selection accounts for the largest share of
  the hierarchy's SLA-violation performance among the four components.
- **H6.3.** No single component's removal improves the composite score,
  i.e. every component is at worst neutral.

### 4. Design and decision rules

5 x 4 regimes, 10 seeds, five configurations (FULL plus four leave-one-out),
scored on the registered ten-metric composite. Paired Wilcoxon signed-rank
against FULL at alpha = 0.05, Holm-Bonferroni and Benjamini-Hochberg both
reported, Cohen's d_z as the primary effect size.

A component is declared **inert** if removing it leaves every metric
bitwise identical to FULL across all regimes and seeds.

### 5. Artifacts

`studies/ablation_study/ablation_driver.py`,
`results/ablation_study/ablation_results.csv`.

---

## VALIDATION RESULTS (pre-confirmatory)

Two regimes (w_E = 0.0188 at h = 1.4 and h = 2.3), 3 seeds, using the
Study 4 tuned parameters.

| Configuration | Mean SLA violation | Change vs FULL | Inert? |
|---|---|---|---|
| FULL | 0.02356 | | |
| no_reserve | 0.03867 | **+0.01511** | |
| no_risk | 0.02430 | +0.00074 | |
| no_negotiate | 0.02356 | **+0.00000** | **YES** |
| no_edge | 0.02667 | +0.00311 | |

### The headline finding: the Coordination Layer is dead code

Inter-domain negotiation fires only when the most loaded domain exceeds
90% utilization **and** the least loaded falls below 75%. Instrumenting
17,500 intervals at h = 1.4:

- max domain utilization > 0.90 in **69.1%** of intervals
- min domain utilization < 0.75 in **8.3%** of intervals
- **both simultaneously: 0.0%**

The two conditions are individually common but never co-occur, because the
allocator fills domains proportionally enough that a heavily loaded system
has no idle domain to donate to. The negotiation step therefore **never
executes** in this environment, while the framework is still charged its
orchestration overhead (24 messages per interval, the highest of any
scheme, and 0.06 orchestration energy).

This is the sharpest available answer to R5 #2. One of the four
hierarchical components contributes literally nothing and costs overhead.

### Ordering of the remaining three

Learned reserve selection dominates (+0.0151 when removed), edge correction
is second (+0.0031), risk weighting is a distant third (+0.0007). Combined
with the Study 5 finding that even a PPO policy converges to a
near-constant reserve, the picture is consistent: **almost all of the
hierarchy's value is in choosing a provisioning level, which is exactly
what a tuned static margin also does.**

Risk weighting is close to inert too: it affects only the priority sort,
which binds only under contention, so it is numerically inactive at
h >= 1.7 (confirmed separately during Study 4 harness validation).

---

## To run the confirmatory study

**Important:** the driver must import the **Step 6** simulator, the one
carrying `ABLATION_TOGGLES_SUPPORTED = True`. Running it from a directory
holding an earlier copy (for example `studies/ppo_study/`) makes every
configuration execute an identical FULL hierarchy, producing a null result
that looks like a finding. The driver now refuses to start in that case,
and additionally voids any run in which every component reports inert.

```bash
mkdir -p studies/ablation_study && cd studies/ablation_study
# copy the STEP 6 hao_pra_u_sim.py and ablation_driver.py here
python3 ablation_driver.py --validate \
    --params ../../results/nested_validation/tuned_params_nested.json
python3 ablation_driver.py --workers 8 \
    --params ../../results/nested_validation/tuned_params_nested.json
```

Roughly 100 minutes at 8 workers for the full 20 regimes x 10 seeds x 5
configurations. Send back `ablation_results.csv`.

## Caveats

1. The validation above is 2 regimes and 3 seeds. Treat the ordering as
   indicative until the full run completes; the inertness finding for
   negotiation is already solid because it rests on a trigger-condition
   count over 17,500 intervals, not on a performance difference.
2. The fixed-reserve counterfactual (rho = 0.20) is one choice among
   several. A different constant would change the magnitude attributed to
   the reserve component, though not its rank.
3. Leave-one-out does not capture interactions. If two components are
   jointly necessary but individually removable, this design will
   understate both. Given negotiation never fires and risk weighting is
   near-inert, that risk is low here, but it should be stated.
