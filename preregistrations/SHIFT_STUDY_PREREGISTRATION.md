# PRE-REGISTRATION — Regime-Shift Study (FINAL experimental cycle)
Status: registered BEFORE implementation and execution. Date: 2026-06-12.
Prerequisites: DESIGN_FREEZE_v2 (method), Phase B sweep (closed, Gate 1
failed). This study tests claim B1 of CLAIMS_INVENTORY.md.

## Commitment
This is the LAST experimental cycle for the HAO-PRA-U framework before a
write-up decision. Whatever Gate 2 returns, no further environment or
scenario searches follow; the outcome maps directly to a writing decision
(see Gate 2). The proposed method is frozen as of v2; no method changes in
this study.

## Claim under test (B1)
HAO-PRA-U, configured once, adapts online when the traffic regime shifts,
while baselines whose parameters were tuned (offline, by grid search) for
the pre-shift regime degrade. This is the zero-touch thesis: the baselines'
competitiveness in Phase B depended on per-regime re-tuning — a human in
the loop. Here, NOBODY gets re-tuned after the shift, including HAO-PRA-U.

## Known facts that make this test hard (disclosed ex ante)
Reactive and Centralized-Predictive scale their allocations with observed/
forecast demand, so they partially track shifts "for free" even with stale
margins; the stale 1.4x margins also absorb moderate load increases. The
GRU (trained pre-shift) will systematically under-predict after a load
step, degrading HAO-PRA-U's own forecasts. A pass is therefore not
presumed; Static is expected to degrade most, and the genuine contest is
HAO-PRA-U vs Reactive/Centralized-Predictive post-shift.

## Protocol
Trace: 1300 intervals. 0-699 training (regime identical to the Phase B
anchor: w_E = 0.0188, h = 2.0). 700-999 pre-shift test window. Shift at
t = 1000. 1000-1299 post-shift test window.
Tuning: baselines grid-search-tuned per seed on intervals 0-699 ONLY
(anchor regime; expected to reproduce Phase B anchor tuned values).
GRU: trained per seed on intervals 0-699 only; inference on the full trace
of each scenario. HAO-PRA-U: v2 pretraining on intervals 0-699 only.
No scheme receives any re-tuning, re-training, or configuration change
after t = 699.
Seeds: 0-9. Composite weights: anchor weights (w_E = 0.0188).

Neutrality requirement: with the shift disabled, intervals 700-999 must
reproduce the Phase B anchor cell bit-for-bit. Verified before the run.

## Scenario instances (4, fixed now)
S1a LOAD STEP +15%: all slice base demands x1.15 from t = 1000.
S1b LOAD STEP +30%: all slice base demands x1.30 from t = 1000.
S2  MIX SHIFT: from t = 1000, URLLC and XR base demands x1.40, eMBB x0.80
    (total mean load approximately preserved; pressure moves to Edge/RAN).
S3  VARIABILITY SHIFT: burst probability 0.05 -> 0.10 and burst magnitude
    range (0.30,0.60) -> (0.40,0.80) from t = 1000.
These four are the registered set. No additional scenarios may be added or
substituted after results are observed.

## Outcomes and decision rules (fixed now)
Primary, per scenario instance: post-shift-window per-seed metrics; the
per-seed composite (Phase B normalization, anchor weights) computed over
the post-shift window only.
"Wins an instance" = best post-shift mean composite among all six schemes
AND Welch p < 0.05 vs the runner-up (identical to the Phase B rule).
GATE 2 PASS = HAO-PRA-U wins >= 2 of the 4 instances.
Secondary, reported regardless of outcome: pre- vs post-shift degradation
(delta SLA violation ratio, delta latency) per scheme; recovery curves
(rolling 25-interval violation ratio over t = 950-1299).

## Writing decision bound to Gate 2
PASS: proceed with the Option 3 architecture paper; Section 7 = Phase B
map (automation cost) + this study (adaptation benefit); abstract claims
adaptation, not KPI superiority.
FAIL: the zero-touch thesis has no KPI evidence. The submission becomes
either the negative-result paper (Option 1: strong-baseline study, both
experiments reported) or an architecture-description paper with NO
comparative performance claims. In neither case does any superiority or
adaptation claim appear.

## Reporting
All four instances, all six schemes, all windows published (per-seed CSV,
decision report, recovery curves), appended to the same repository as
Phase B. Pre-shift-window results are reported alongside post-shift so
readers can verify the anchor reproduction.
