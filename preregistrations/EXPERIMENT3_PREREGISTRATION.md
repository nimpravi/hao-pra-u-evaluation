# PRE-REGISTRATION — Experiment 3: Predictability Sweep (Burst-Precursor Traffic)

**Status:** Registered BEFORE implementation and execution.
Prerequisites: DESIGN_FREEZE_v2 (method); Phase B sweep and regime-shift
study closed. The freeze timestamp is witnessed by this file's commit in the
repository history, which precedes the commit of any Experiment-3 result.

> Note on the numbers in this document. A pre-registration that contains
> figures invites the question of how it can be "pre-." The only quantities
> recorded here are **generator self-properties** (marginal burst rate,
> precursor/state correlations) — they characterize the traffic generator in
> isolation, involve no orchestration scheme, and produce no comparative
> result. They are reported so the predictability axis is interpretable. All
> experimental outcomes are produced later, by running the frozen code.

---

## 0. Pinned artifacts at freeze

The confirmatory run MUST use the generator exactly as frozen here. Verify
before running:

| Artifact | SHA-256 (fill in at freeze; must match these) |
|---|---|
| `precursor_traffic.py` | `d2c8f28c5220e2f60b7fc3d5c0c28a154757649b20a595df2ef74905264f7aa4` |
| `hao_pra_u_sim.py` (imported UNCHANGED) | `24f20b81ad19edbd522c72596d8526218cd8274a3791679e8af0897accdf3a3d` |

> **Pinning protocol for the registered method changes.** The frozen
> `hao_pra_u_sim.py` is imported **unchanged** (hash above). The two
> registered modifications — the 2-channel precursor-aware GRU (Section 4.6)
> and the `brisk` Q-state feature (Section 4.7) — are implemented in the
> harness `run_experiment3.py` (deliverable 3), which **composes** the
> frozen simulator's primitives and overrides only the forecaster input
> assembly and the `discretize` call, exactly as specified in Sections 4.6–
> 4.7. When deliverable 3 is written, its SHA-256 is added to this table and
> this prereg is updated and re-committed; the harness must contain **no** logic
> beyond what Sections 4.5–4.7, 5, and 6 specify. Any divergence between the
> harness and this document voids the registration.

Frozen generator parameters (from `precursor_traffic.py` at the hash above):

| Parameter | Value |
|---|---|
| Latent state | AR(1), `LATENT_RHO = 0.85`, `LATENT_STD = 1.00`, normalized to unit std |
| Hazard | `P(burst|x) = clip(base_rate · exp(HAZARD_GAIN · x_cause), 0, HAZARD_MAX)` |
| `HAZARD_GAIN` | `3.00` (strongly-separable regime, chosen ex ante) |
| `HAZARD_MAX` | `0.95` |
| `base_rate` | calibrated per seed/slice so expected hazard ≡ `BURST_PROB` |
| Marginal burst rate (held invariant) | `BURST_PROB = 0.05` |
| Burst magnitude | `U(0.30, 0.60) × base` (unchanged from frozen sim) |
| Precursor lead `L` | `PRECURSOR_LEAD = 1` interval |
| Precursor observation | `o(t) = x(t) + obs_noise_std · ε`, `obs_noise_std` set by predictability SNR |
| Periodic / stochastic components | unchanged from frozen sim (`A=0.25`, `T=100`, `σ=8%`) |

**Marginal-invariance commitment.** The expected per-interval burst hazard
is held at 0.05 for every predictability level by per-seed base-rate
calibration. This makes `predictability` a *pure anticipability*
manipulation: across the grid, bursts occur equally often and have the same
magnitude distribution; only their *predictability from the precursor*
changes. Any difference in results across the grid therefore cannot be
attributed to the predictable regime having fewer or weaker bursts.

---

## 1. Background and motivation

The companion negative-result study (Experiments 1–2) finds that the
hierarchical agentic orchestration framework **HAO-PRA-U** loses to
per-regime-tuned static over-provisioning across an 18/20-decisive 5×4
regime sweep, and adapts no better than demand-following baselines under
regime shift. That result is **conditional on the traffic model**: bursts
are memoryless (Bernoulli, no precursor), so *no* forecaster can anticipate
them. A reviewer may reasonably object that this engineers an environment
in which prediction cannot pay, and that the negative result therefore does
not generalize to settings where bursts are anticipable.

Experiment 3 is designed to close that generality objection by testing
HAO-PRA-U in a traffic family where bursts **are** anticipable, via an
observable leading-indicator precursor whose informativeness is swept by a
single knob. The experiment is confirmatory and pre-registered; it is run
**only after** this document is committed (frozen) to the repository.

**This study is not designed to make HAO-PRA-U win.** A negative result
that survives an environment where prediction *can in principle* pay is the
stronger paper, because it closes the generality objection rather than
conceding it. Both outcomes are reportable and the decision rule below is
symmetric with respect to them.

---

## 2. Fairness of observation (PRIMARY DESIGN CONSTRAINT — flagged)

The precursor is a **monitoring-plane** signal (modeling real 6G burst
precursors: RACH/RRC connection-request surges, buffer-status-report ramps,
beam-failure indications, paging/handover signaling). It is **not** served
demand.

Observation rights are **asymmetric by design and the asymmetry is the
point of the experiment**:

- The **GRU forecaster MAY consume the precursor** `o(t)` (as a second
  input channel alongside demand history). It is a monitoring-plane signal
  an orchestrator legitimately observes.
- **Every baseline** (Static, Reactive, Centralized, Flat-Agent) and **all
  shared paths** (`ubar`, `U[t-1]`, the metric functions) **see only
  realized demand history**, exactly as in the frozen simulator. They
  **never** receive the precursor.

This asymmetry is favorable to the predictive schemes (HAO-PRA-U,
Centralized-Predictive, Flat-Agent all consume the GRU forecast). It is the
maximally generous setting for "prediction pays." If HAO-PRA-U still fails
to beat demand-only tuned baselines even here, the negative result is
robust to the generality objection. If it wins above some predictability
threshold, the original negative result is shown to be conditional on
unpredictability — also a clean, publishable finding.

**Harness obligations (to be enforced in `run_experiment3.py`, deliverable
3, not yet written):**

1. The precursor array `P` is passed **only** into GRU feature assembly.
2. An explicit assertion verifies no baseline allocation path references
   `P`. The harness aborts if violated.
3. The precursor is causal: `o(t)` is observable within the forecaster's
   lookback window up to and including the decision interval; it carries
   information about bursts at `t .. t+L`. No future demand or future burst
   indicator is ever exposed. (Prediction is *possible*, never handed over.)
4. Forecast skill is logged (Section 6) so the run can confirm the
   forecaster actually extracted precursor information, rather than
   assuming it.

---

## 3. Hypotheses

Let `p ∈ [0,1]` be the predictability knob. Define a scheme's **regime win**
exactly as in Experiments 1–2 (Section 5).

- **H1 (threshold existence).** There exists a predictability threshold
  `p* ∈ (0, 1]` such that for `p ≥ p*`, HAO-PRA-U achieves a regime win
  (best mean composite AND Welch `p < 0.05` vs. the runner-up) over the
  per-regime-tuned baselines, whereas for `p < p*` it does not.

- **H0 (null / robustness of the negative result).** No such threshold
  exists in the swept grid: HAO-PRA-U fails to achieve a regime win at every
  predictability level, including the strongly-separable maximum. This would
  show the negative result is robust even when prediction can in principle
  pay.

- **H2 (secondary, forecast-mediated, exploratory).** Any HAO-PRA-U
  advantage that does appear is mediated by forecast skill: the
  predictability level(s) at which HAO-PRA-U wins coincide with the
  level(s) at which the GRU's burst-anticipation fraction (Section 6)
  exceeds its memoryless-baseline value by a pre-specified margin. H2 is
  **exploratory** and does not gate the primary conclusion; it is reported
  regardless of direction.

**Pre-registered interpretation matrix (frozen):**

| Outcome | Conclusion reported |
|---|---|
| H0 holds (no win anywhere, incl. `p=1.0`) | Negative result is robust to predictability; generality objection closed. Negative-result paper proceeds, strengthened. |
| H1 holds with `p*` in interior `(0,1)` | Negative result is conditional on unpredictability; HAO-PRA-U pays off only above `p*`. Reframe as conditional-benefit paper, threshold reported. |
| HAO-PRA-U wins only at `p = 1.0` | Benefit exists but requires near-oracle precursors; reported as a narrow, possibly impractical condition. |
| HAO-PRA-U wins at all `p` including `p=0` | **Inconsistency flag** — contradicts Experiments 1–2 at `p=0`; triggers mandatory investigation of a harness/regression error before any claim. |

The last row is a registered tripwire: `p=0` must reproduce the Experiment-1
anchor-regime ordering (HAO-PRA-U not winning). If it does not, the result
is treated as a bug, not a finding, until explained.

---

## 4. Design

### 4.1 Predictability grid (frozen)

```
PREDICTABILITY_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
```

`p = 0.0` is the **memoryless control** (must reproduce the Experiment-1
burst law and ordering). `p = 1.0` is the **strongly-separable maximum**.
Generator self-property at freeze (characterization, not a result),
12-seed mean precursor→future-burst correlation:

| p | corr(precursor, burst at t+L) |
|---|---|
| 0.0 | ≈ 0.00 |
| 0.2 | ≈ 0.06 |
| 0.4 | ≈ 0.17 |
| 0.6 | ≈ 0.27 |
| 0.8 | ≈ 0.35 |
| 1.0 | ≈ 0.42 |

### 4.2 Lead time (frozen)

`L = 1` interval, fixed for the confirmatory sweep. (Rationale: a one-step
lead is the minimal anticipable setting compatible with the frozen
one-step-ahead GRU; longer leads are a deferred robustness check, NOT part
of this confirmatory study, and may not be substituted after results are
seen.)

### 4.3 Regime (frozen)

All Experiment-3 runs use the **anchor regime** of Experiments 1–2:
`energy_weight (w_E) = 0.0188`, `headroom (h) = 2.00`, via
`hao_pra_u_sim.set_regime(0.0188, 2.00)`. The predictability sweep is the
*only* manipulated axis. Rationale: isolate predictability from the
energy/headroom regime axes already swept in Experiment 1.

### 4.4 Seeds (frozen)

```
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # n = 12 (≥ 10)
```

Twelve seeds, fixed in advance. No seed may be added, dropped, or replaced
after any result is observed. If a run errors on a seed, the error is fixed
and ALL seeds are re-run; partial-seed reporting is prohibited.

### 4.5 Schemes (frozen)

The six frozen schemes, unchanged from the simulator:
Static, Reactive, Centralized, Centralized-Predictive, Flat-Agent,
HAO-PRA-U. All predictive schemes consume the identical GRU forecast per
seed. Baselines are re-tuned per (predictability, seed) on the training
intervals only (grid identical to the frozen `tune_baselines`).

### 4.6 Forecaster (frozen)

The GRU is extended from a 1-channel (demand) to a **2-channel
(demand, precursor)** input. Architecture, lookback (16), hidden size (24),
epochs (35), lr (1e-3), batch (32), and train/test split (70/30) are
otherwise unchanged. The second channel receives the precursor `o(t)`
z-scored on training intervals only. At `p = 0.0` the precursor channel is
pure noise; the GRU is free to learn to ignore it. The extension is applied
identically to every predictive scheme so the forecast is shared.

> The 1→2 channel change is the only modification to the forecasting model.
> It is registered here, before the run. No other simulator internals are
> changed except the orchestrator Q-state revision in Section 4.7;
> `hao_pra_u_sim.py` is imported and the precursor-aware behavior is added by
> the harness wrapper described in deliverable 3, with all changes pinned by
> hash below.

### 4.7 Orchestrator Q-state revision (REGISTERED METHOD CHANGE — frozen)

**Motivation (logged before any result).** In the frozen simulator the
Global Orchestrator's reserve-sizing Q-state is
`discretize(avg_p, max_p, err)`, where `avg_p`, `max_p` are *current*
demand pressures and `err` is *trailing* forecast error
`mean(sigma) = mean(|U[t-1]-UHAT[t-1]|/...)`. None of these is
forward-looking. Therefore, even when the precursor improves the forecast
`uhat`, the *reserve-size decision* has no channel through which a
precursor-informed burst expectation can change how much headroom is held
*now*. A loss under that configuration would be uninformative about the
generality question, because the controller would be structurally blind to
predictability at the decision it is meant to learn. To give the framework
a fair opportunity to exploit predictability, we register **one** additional
Q-state feature before the confirmatory run.

**Definition (frozen).** A forward burst-risk feature is added to the
discretized Q-state:

```
brisk(t) = max_s ( uhat_s(t) / (ubar_s(t) + EPS) )
```

binned by `BRISK_BINS = [1.10, 1.30]` into three levels, extending the
Q-state from `(avg_p_bin, max_p_bin, err_bin)` to
`(avg_p_bin, max_p_bin, err_bin, brisk_bin)`.

**Why this is legitimate and fairness-preserving:**

- `brisk` is a function of `uhat` and `ubar` **only**. Both are already
  legitimately available to the predictive schemes in the frozen simulator
  (`uhat` is the shared GRU forecast; `ubar` is the trailing rolling mean
  used throughout). The **raw precursor `P` never enters the Q-state
  directly** — it reaches the orchestrator only through its effect on the
  forecast, exactly as for every other predictive scheme. Baselines still
  see only realized demand. The Section-2 fairness gate is unchanged.
- The feature carries information **only when prediction pays**. A
  generator self-property check at freeze (characterization, not a result;
  using a linear precursor-aware demand proxy, not the orchestrator)
  confirms `AUC(brisk vs. next-interval burst)` ≈ 0.52 (chance) at
  `p ∈ {0.0, 0.4}` and rises to ≈ 0.64–0.67 at `p ∈ {0.8, 1.0}`. So at
  `p = 0` the orchestrator gains nothing (preserving the memoryless
  behavior and the `p=0` tripwire), and at high `p` it has a genuinely
  learnable — but non-oracle — signal.
- It does **not** guarantee a win. A ~0.67-AUC feature is useful, not
  decisive; the tabular Q-learner may or may not exploit it, and the tuned
  baselines may still win on cost. The test remains genuinely two-sided.

**This is a method-internal revision adopted before the confirmatory sweep**,
analogous to the reserve-learner replay-pretraining revision recorded in the
Experiment-2 pre-registration. The frozen-orchestrator alternative (no
`brisk` feature; precursor reaches only the forecast) was considered and is
recorded here as the rejected, less-generous design. The choice between them
was made **before any Experiment-3 result was observed**, on the stated
fairness rationale, not on any outcome.

**What is NOT permitted:** adding any further Q-state feature, changing
`BRISK_BINS`, or altering the `brisk` definition after results are observed.
If the registered feature fails to help at high `p`, that is a reportable
result (the negative result is robust even when the controller is given a
fair forward-looking channel), **not** grounds for adding more features.

---

## 5. Statistical decision rule (frozen, VERBATIM from Experiments 1–2)

Per predictability level `p`:

1. For each of the 12 seeds, compute each scheme's **composite score** using
   the frozen min–max normalization across the six schemes within that seed,
   with the anchor objective weights (`w_E = 0.0188`), exactly as
   `hao_pra_u_sim.composite_scores` / `analyze_shift.composites` implement
   it. Metrics, signs (benefit vs. cost), and weights are unchanged.

2. Rank schemes by **mean composite** across the 12 seeds. The top scheme is
   the **winner**, the second is the **runner-up**.

3. The winner achieves a **decisive regime win** iff a **Welch two-sided
   t-test** on the 12 per-seed composites of winner vs. runner-up gives
   **`p < 0.05`**.

4. **HAO-PRA-U wins level `p`** iff HAO-PRA-U is the winner AND the win is
   decisive.

5. Per-metric Welch tests (HAO-PRA-U vs. each baseline) are produced by the
   **unmodified** `scripts/significance_test.py` on the per-seed KPI CSV, on
   the seven KPI metrics. These are reported in full; significance is
   claimed only where `p < 0.05`.

**Primary endpoint:** the set of predictability levels at which HAO-PRA-U
achieves a decisive regime win. **H1 is supported** iff that set is a
nonempty up-set (all `p ≥ p*` for some `p*`); the smallest such `p*` is the
reported threshold. **H0 is supported** iff that set is empty.

Multiple-comparison note (frozen): the six predictability levels are six
pre-registered confirmatory tests of the same family. The threshold claim
(H1) requires a *contiguous* up-set, not isolated significant cells; an
isolated significant level with non-significant higher levels is reported as
**non-monotone and not threshold-consistent**, not as support for H1. No
post-hoc level is added.

---

## 6. Metrics and diagnostics (frozen)

**KPI metrics** (per scheme/seed/level), identical to the frozen simulator
and to `significance_test.py`'s schema:

```
sla_violation, latency, throughput, utilization, energy_eff,
fragmentation, acceptance
```

logged one row per `(method, seed)` per predictability level into a
`results_per_seed.csv` whose **column schema is byte-identical** to what
`significance_test.py` expects, so that script runs unmodified. Auxiliary
columns (`reliability, isolation, overhead, reward`) follow the existing
simulator convention. The predictability level is recorded in a **separate**
key column / companion file so the KPI CSV fed to `significance_test.py` is
unchanged in shape.

**Forecast-quality diagnostics** (per scheme-with-forecast / seed / level),
logged to a companion `forecast_diag.csv` — NOT consumed by
`significance_test.py`:

- `pred_mae`, `pred_rmse`, `pred_mape` per slice (held-out test intervals),
  matching the frozen accuracy reporting.
- **`burst_anticipation_auc`** (PRIMARY anticipation diagnostic,
  threshold-free): the AUC of the forecaster's burst-exceedance signal
  `uhat_s(t)/(ubar_s(t)+EPS)` against the ground-truth burst indicator on
  test-window intervals, pooled across slices per (scheme, seed, level).
  This is the registered H2 statistic; it removes any dependence on an
  arbitrary detection threshold. AUC ≈ 0.5 means the forecaster cannot
  anticipate bursts; AUC rising with predictability is the H2 signature.
- **`burst_anticipation_fraction`** (SECONDARY, illustrative only): of the
  realized bursts in the test window, the fraction "anticipated," where
  anticipated is defined ex ante as the forecast exceeding rolling-mean
  demand by `τ = 0.5 × (mean burst magnitude) × base`. Reported as a
  human-readable companion to the AUC; **does not** gate H2, because the
  threshold `τ` is arbitrary. `τ` frozen here.
- **`burst_false_alarm_fraction`**: fraction of non-bursting test intervals
  exceeding the same `τ` (precision context for the illustrative fraction).
- `precursor_channel_used`: boolean per scheme (sanity check that only the
  forecaster path saw the precursor).

These diagnostics test H2 and let a reviewer see *whether prediction
actually paid information* at each level, independent of whether HAO-PRA-U
won.

---

## 7. What is NOT permitted after results are observed

- Changing `HAZARD_GAIN`, `HAZARD_MAX`, the grid, the lead, the seed list,
  the regime, the forecaster architecture, the composite weights, the
  detection threshold `τ`, the `brisk` definition, `BRISK_BINS`, or the
  decision rule.
- Adding any further orchestrator Q-state feature beyond the single
  registered `brisk` feature.
- Adding or dropping predictability levels, seeds, schemes, or metrics.
- Re-defining "win," "decisive," or "anticipated."
- Reporting a subset of seeds or levels.
- Substituting the lead time or re-running with a different precursor
  mechanism and presenting it as this confirmatory study (such runs, if
  done, are exploratory and must be labeled as post-hoc).

Permitted: fixing an outright code error that prevents the registered design
from executing as written, provided (a) the fix is described, (b) ALL seeds
and levels are re-run from scratch, and (c) the `p=0` tripwire (Section 3)
is re-checked.

---

## 8. Deliverables this prereg governs

1. `precursor_traffic.py` — generator (deliverable 1, hashed above; FROZEN).
2. **This document** (deliverable 2) — committed (frozen) before
   deliverable 3 is written; freeze ordering witnessed by commit history.
3. `run_experiment3.py` + analysis harness (deliverable 3) — to be written
   **after** this document is committed; composes the frozen simulator's
   primitives, implements the registered 2-channel forecaster (4.6) and
   `brisk` Q-state feature (4.7) and nothing else, logs the Section-6
   metrics, and wires the per-seed KPI CSV into the unmodified
   `scripts/significance_test.py`.

---

*End of pre-registration. Commit this document (freeze) before
proceeding to deliverable 3.*
