> **What this document is.** An internal claims audit performed during the
> honest rewrite of this manuscript, after the pre-registered sweep closed.
> Each prior-draft claim was binned as supported (A), testable-but-untested
> (B), or falsified/unsupported (C), and Bin-C claims were removed from the
> paper. It is retained here for transparency: it documents that unsupported
> superiority claims were systematically identified and struck rather than
> quietly dropped. It is NOT a pre-registration and contains no results.
>
> **Status: historical record, partly superseded.** The verdicts below were
> written against Study 1, the regime sweep run under asymmetric search
> budgets (baselines 6 to 9 tuning points, the framework 0). Study 4 re-ran
> that comparison under equal 9-point budgets with nested validation, and
> two verdicts no longer hold as stated: C1's "HAO-PRA-U wins 0" and C2's
> "ranks 3rd-5th by composite in every cell". Under equal budgets the
> framework wins 2 of 20 regimes, both where capacity is abundant. The
> manuscript withdraws the stronger claim and says so. This file is kept
> otherwise unedited as a dated record of the claims audit; see Study 4 in
> the paper and `results/nested_validation/` for the superseding numbers.

# CLAIMS INVENTORY — HAO-PRA-U manuscript (Option 3 rewrite)
Basis: Phase B pre-registered sweep (20 cells x 10 seeds, decision_report.json,
2026-06). Bins: A = supported by existing evidence (keep, update numbers);
B = architectural/testable but NOT yet tested (keep only if the pending
regime-shift study supports it, or reword as design description);
C = falsified or unsupported (delete or rewrite; cannot appear as a claim).

---------------------------------------------------------------------------
BIN C — FALSIFIED OR UNSUPPORTED. DELETE / REWRITE.
---------------------------------------------------------------------------
C1. Abstract: "HAO-PRA-U achieves the best overall performance" + all listed
    numbers (0.003 violations, 14.287 ms, 835.982 throughput, 0.483 util,
    0.993 acceptance). Numbers are from the lost simulator (unusable
    regardless of outcome); the superiority claim is falsified: Static wins
    18/20 cells decisively, HAO-PRA-U wins 0.
C2. Results line 30: "best overall optimization performance, ranking first."
    Falsified. HAO-PRA-U ranks 3rd-5th by composite in every cell.
C3. Results lines 57-58: "lowest mean SLA violation ratio, lowest latency,
    highest throughput, highest slice acceptance." All four falsified at
    every cell of the registered space.
C4. Results line 83: static/reactive "ranking lowest for their inability to
    anticipate." Falsified — properly tuned Static ranks FIRST almost
    everywhere. This sentence is the old weak-baseline assumption.
C5. Results line 135: "HAO-PRA-U outperforms the baselines for four
    complementary reasons..." Premise falsified; the four-mechanism
    explanation must be reframed (see A4/B1) as what the mechanisms DO,
    not why they win.
C6. Conclusion: "achieves the lowest SLA violation and latency and the
    highest throughput and slice acceptance." Falsified.
C7. Conclusion: "results confirm that prediction alone is insufficient..."
    — the sweep showed something different and more specific: prediction
    (with or without hierarchy) cannot beat tuned margins when the
    unpredictable traffic component is memoryless. Replace with the honest
    lesson (see A6).
C8. Contribution (ii): "pre-allocates resources ... before SLA degradation"
    as a benefit claim. The mechanism exists (A2) but the implied
    anticipation benefit is falsified for memoryless bursts. Reword to
    mechanism description only.
C9. Any remaining sentence implying burst anticipation. Bursts in the
    traffic model are Bernoulli/memoryless; no predictor can anticipate
    them.

---------------------------------------------------------------------------
BIN B — TESTABLE, NOT YET TESTED. PENDS ON REGIME-SHIFT STUDY (or re-run).
---------------------------------------------------------------------------
B1. THE NEW CENTRAL CLAIM (to be earned): HAO-PRA-U adapts online to regime
    changes without re-tuning, while statically tuned baselines degrade.
    This is exactly what the pre-registered shift study tests. If it fails,
    the paper has no evaluation leg and the submission plan changes.
B2. Conclusion: "the ablation shows the full framework outperforms every
    reduced variant." Old data; ablations were never re-run on the new
    simulator. Either re-run as a component-contribution analysis (within
    HAO-PRA-U only — legitimate and interesting) or delete.
B3. Hierarchy avoids centralized control-plane bottlenecks. Currently a
    modeled assumption (control-delay constants per DESIGN_FREEZE item 3),
    not a measurement. Keep as explicitly-modeled design rationale; do not
    state as finding.
B4. Lifecycle automation coverage (ADMIT/SCALE/HEAL/DEFER/MAINTAIN). The
    counters exist in the simulator log; quantify and report as automation
    evidence, or keep as design description.

---------------------------------------------------------------------------
BIN A — SUPPORTED. KEEP, WITH REGENERATED NUMBERS.
---------------------------------------------------------------------------
A1. The architecture exists and executes: hierarchical global-domain-edge
    agents, contract negotiation, edge correction, lifecycle actions, all
    implemented in published code and exercised every interval.
A2. The proactive mechanism chain (GRU forecast -> uncertainty -> SLA-risk
    -> reserve -> priority allocation) is implemented and runs as
    described. (Mechanism, not benefit — see C8.)
A3. GRU prediction accuracy: ~9% MAPE across the five slice classes
    (regenerated Table IV from the new run set, means across 10 seeds).
A4. ZERO-TOUCH PROPERTY (the strongest new claim, fully evidenced):
    every baseline required per-regime offline grid-search re-tuning to be
    competitive — tuned_params.json shows margins changing across regimes
    — while HAO-PRA-U used identical configuration in all 20 cells and
    adapted online. The sweep itself is the evidence that the "tuned
    baseline" hides a human in the loop.
A5. RESOURCE FOOTPRINT, stated with its qualifier: in moderate regimes
    (h >= 1.7), HAO-PRA-U matches the best-tuned baseline's SLA violations
    (no significant difference in most cells) while allocating
    significantly fewer resources (utilization significantly lower in
    16/20 cells). Qualifier that MUST accompany it: under tight capacity
    (h = 1.4) HAO-PRA-U's violations are significantly WORSE; the leaner
    allocation buys nothing there.
A6. THE HONEST LESSON (new, citable): per-regime-tuned static
    over-provisioning is a very strong baseline for slice orchestration;
    when the unpredictable demand component is memoryless, prediction-based
    schemes cannot beat it on KPIs — they can only match it with less
    tuning and/or fewer resources. Supported by the full 20-cell map.
A7. Stabilized online learner: replay pretraining + count-based step size
    reduces across-seed variance ~5x (freeze v2 record). Reportable as a
    method property.
A8. The evaluation methodology: pre-registration, design freezes, per-cell
    baseline tuning, Welch tests on per-seed runs, full negative map
    published. A genuine methodological contribution for this venue.

---------------------------------------------------------------------------
CONSEQUENCES BY SECTION (maps to the rewrite plan)
---------------------------------------------------------------------------
00 Abstract: rebuild from A1, A4, A5(+qualifier), A6, B1-pending.
01 Intro: contributions become (i) architecture [A1], (ii) zero-touch
   online adaptation [A4 + B1], (iii) uncertainty-aware prediction chain
   as mechanism [A2, A3], (iv) pre-registered strong-baseline evaluation
   [A8], (v) quantified automation cost [A5, A6]. Research question drops
   "maximizing utilization" (the sweep showed allocation-maximization is
   the wrong objective semantics).
02 Related work: add strong-baseline/negative-result positioning [A6].
03 System model: keep; add regime-shift formalization for B1.
04 Framework: keep and expand [A1, B3 as modeled rationale, B4].
05 Algorithm: keep equations; describe v2 learner [A7]; purpose reworded
   per C8.
06 Setup: new simulator, sweep design, tuning protocol, Table V from code.
07 Results: the 20-cell map [A5, A6], shift study [B1], optional component
   analysis [B2]. All C-items removed.
09 Conclusion: rebuilt from A4-A6 + shift-study outcome.

RULE: no sentence from Bin C survives in any paraphrase. Every quantitative
claim cites the public CSV it comes from.
