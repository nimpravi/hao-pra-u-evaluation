# AMENDMENT A2 to PREREGISTRATION_SHIFT_SYMMETRIC.md (Study 15)

**Date:** 2026-10-18
**Status:** appended after the confirmatory run. No section of the
registration is edited. The run is scored under the rule as registered.

---

## 1. Outcome under the registered rule

Sec. 4 of the registration reads:

> H3 is supported if, in S0, every scheme's post-window metrics fall within
> the pre-window's across-seed standard deviation. If S0 fails, the study is
> void and no result from it is reported.

S0 failed. Four of the seven schemes run moved further between the pre-shift
and post-shift windows than the pre-window's across-seed spread, on a trace
that does not shift:

| scheme | S0 result | d SLA violation | d throughput |
|---|---|---|---|
| Static Allocation | pass | -0.00007 | +1.04 |
| Reactive Allocation | pass | +0.00067 | +0.77 |
| Centralized Orchestration | pass | +0.00267 | +0.35 |
| Centralized Predictive | **fail** | -0.00953 | +3.24 |
| Flat-Agent Allocation | **fail** | -0.00953 | +3.24 |
| HAO-PPO | **fail** | -0.01040 | +3.91 |
| HAO-PRA-U | **fail** | -0.01493 | +5.05 |

**Study 15 is therefore void. H1 and H2 were not computed and no number from
this run is reported anywhere.** This is recorded as an outcome of the study,
not as a limitation of it.

## 2. Cause

The cause is a defect in `shift_symmetric_driver.py`, which we wrote, and not
a property of the trace extension that H3 was designed to detect.

`run_cell` built the forecast array over the frozen window only and padded
the extension with `NaN`:

```python
UHAT, _ = sim.train_gru_forecasts(U[:sim.N_INTERVALS], seed)   # 0..999
UHAT = np.vstack([UHAT, np.full((N_TOTAL - sim.N_INTERVALS,
                                 len(sim.SLICES)), np.nan)])    # 1000..1299
```

Every forecast-consuming branch of `sim.allocate` falls back to the realized
demand when the forecast is not finite:

```python
uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
```

So for all 300 post-shift intervals, `Centralized Predictive`, `Flat-Agent
Allocation`, `HAO-PRA-U` and `HAO-PPO` were allocating against the true
current demand rather than a forecast of it, in every scenario. The risk
signal failed in the same way: `sigma` is formed from `UHAT[t-1]`, so it was
identically zero after interval 1000.

The four schemes that failed S0 are exactly the four that read `UHAT`. The
three that passed (`Static`, `Reactive`, `Centralized`) are exactly the three
that do not. The control partitioned the scheme set along the defect.

We verified that the trace extension itself is sound, which is what H3 was
written to test. Over seeds 0 to 9, S0's post-window aggregate demand sits
0.37 across-seed SD from the pre-window's, against 0.43 SD between two
windows wholly inside the frozen generator. The extension introduces no
detectable artifact in the input.

## 3. A second defect, in the gate itself

The driver's `--validate` mode was meant to run the S0 control before the
sweep. Its tripwire is guarded by `len(pre) > 1`, and `--validate` runs one
seed, so the pre-window standard deviation did not exist and the check was
skipped silently. The gate reported success and could not have failed.

Both defects are ours. The registered control caught the first one after the
fact, which is the only reason this is an amendment rather than a published
result.

## 4. Direction of the bias

The defect favoured the framework. On S0 the clairvoyant fallback cut
HAO-PRA-U's SLA violation by 0.0149 and raised its throughput by 5.05, the
largest benefit of any scheme, because HAO-PRA-U consumes the forecast most
heavily. H1 and H2 both compare tuned Static against HAO-PRA-U, and Static
takes no benefit from the defect. Had this run been scored, it would have
been scored in the framework's favour by an amount larger than the effects
the study was looking for.

We record this because the alternative reading, that a void was convenient,
is available to a reader and is wrong in this instance.

## 5. What happens next

Study 15 stays void and stays in the record. The corrected study is
registered separately as **Study 16**, in
`PREREGISTRATION_SHIFT_SYMMETRIC_V2.md`, and is a new registration rather
than an edit, because the registration above cannot be amended into a study
that passes.

The corrected driver is `shift_symmetric_v2_driver.py`. The fix is one line,
`sim.train_gru_forecasts(U, seed)` over the full 1300 intervals with the
padding removed. Training stays confined to intervals 0 to 549 by
`N_TRAIN`, so the forecaster is not retrained after the shift and the
staleness the study tests is preserved: with the fix, forecast error on S0
runs at 9.06% MAPE pre-shift and 9.39% post-shift.

Forecasts over intervals 0 to 999 are unchanged to within 1.5e-05 absolute,
a float32 batched-inference difference and not a change of model, so Study
16's tuning window and pre-shift window remain comparable to Study 15's. The
corrected driver reports any tuned parameter that differs from the Study 15
JSON rather than assuming none do.

## 6. Lesson for Sec. VIII

A neutrality control earns its place when it can fail for a reason the
authors did not anticipate. H3 was registered to detect an artifact in a
synthetic trace extension. It detected a missing forecast array instead, and
it did so by failing on precisely the subset of schemes that read the array.
A control narrower than the pipeline it guards would not have caught this:
had H3 tested only the trace, which is what its own justification says it
tests, the defect would have reached the results.

The corollary is the harder one. A gate that cannot fail is worse than no
gate, because it is reported as a pass. The `len(pre) > 1` guard made the
pre-run check vacuous in the only mode it was invoked in.
