# PRE-REGISTRATION — Resource-cost / load sweep (extension of DESIGN_FREEZE)
Status: registered BEFORE execution. Date: 2026-06-11.

## Disclosure of known results
One grid point has already been observed (the frozen 10-seed run):
energy/resource-cost weight w_E = 0.0188, capacity headroom h = 2.0x.
At that point, tuned over-provisioning baselines dominate HAO-PRA-U on the
composite and on SLA violations. All hypotheses below concern the REST of
the surface; the known point is the anchor, not evidence for H1/H2.

## Research question
Under what resource-cost and load regimes, if any, does hierarchical
proactive orchestration (HAO-PRA-U) outperform tuned over-provisioning
and prediction-only baselines?

## Phases
PHASE A (exploratory, before the sweep): stabilize the HAO-PRA-U Q-learner
(known weakness: 5-10x higher across-seed variance). Permitted changes are
to the PROPOSED METHOD only — e.g., longer exploration, reward smoothing,
state-bin revision. Baselines, traffic, metrics, and evaluation code are
untouchable. Phase A ends with DESIGN_FREEZE_v2 recording the final method
verbatim. No sweep results may be observed during Phase A.

PHASE B (confirmatory): run the full sweep once under DESIGN_FREEZE_v2.
No code changes afterward except crash fixes (which force a full re-run).

## Sweep design
Axes (full factorial), 10 seeds per cell, 1000 intervals, 70/30 split:
  - Resource-cost weight w_E in {0.0188, 0.05, 0.10, 0.20, 0.30}
    (w_E enters the reward/composite; remaining Table II weights are
    rescaled proportionally to keep the total at 1.0; the same reward is
    used for baseline tuning and HAO-PRA-U learning in that cell)
  - Capacity headroom h in {1.4, 1.7, 2.0, 2.3}
Per cell: baselines re-tuned by the standard grid search on training
intervals under that cell's reward; HAO-PRA-U trains under that cell's
reward. 5 x 4 = 20 cells = 1200 scheme-runs total.

## Hypotheses
H1: There exists a region (high w_E and/or low h) where HAO-PRA-U achieves
    a statistically better composite score than every baseline.
H2: In that region, HAO-PRA-U Pareto-dominates the best baseline on
    (SLA violation ratio, total allocated resources).
H3: The advantage region grows as h decreases (tighter capacity).

## Decision rules (fixed now)
"Wins a cell" = best mean composite AND Welch p < 0.05 vs the runner-up
across the 10 seeds. Pareto claims require p < 0.05 on both axes vs the
best baseline. The FULL 20-cell map is reported in the paper regardless of
outcome, including all cells where baselines win.

GATE 1 (after sweep): if HAO-PRA-U wins >= 3 contiguous cells, proceed to
write the reframed paper: "characterizing when hierarchical proactive
orchestration outperforms over-provisioning in 6G slicing," reporting the
whole map. If it wins 0-2 cells, the framework as designed does not beat
its baselines; the TNSM submission is postponed and work returns to method
development (a new cycle of Phase A), or the negative result is written up
honestly for a suitable venue.

## Reporting commitments
All 20 cells published (heatmaps + per-seed CSVs). No cell omitted. Claims
in the abstract restricted to what the decision rules support. This file,
DESIGN_FREEZE_v2, code, and all CSVs go in the public repository.
