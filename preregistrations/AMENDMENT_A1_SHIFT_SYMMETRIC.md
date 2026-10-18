# Amendment A1 to PREREGISTRATION_SHIFT_SYMMETRIC.md

**Date of amendment:** 10/18/2026
**Status of the registration being amended:** deposited and independently
timestamped. The deposited text is not edited. This amendment is appended,
committed, hash-pinned, and deposited alongside it.
**Observed before this amendment:** nothing. No result, no metric and no
partial output from this study has been produced. The driver failed on its
first validation invocation, before any scenario completed.

## What the registration left unspecified

Sec. 3 of the registration fixes the simulator, trace, windows, scenarios,
seeds, budget and protection range. It says the search budget is "9
validation evaluations per scheme, every scheme." **It does not name the
scheme set.**

That is a gap. Study 2 compared six schemes. The simulator build this study
uses adds a seventh, HAO-PPO, introduced for Study 5. The min--max composite
is normalized within whatever set is scored, so the six-scheme and
seven-scheme composites differ by construction, and the registration as
deposited does not say which is primary.

## How the gap was found

The driver crashed on `python shift_symmetric_driver.py --validate`, during
the S0 neutrality control, before producing any result. It iterated
`sim.SCHEMES`, which on this build includes HAO-PPO, and HAO-PPO raised
`ImportError` because `ppo_reserve.py` is not on the import path in
`studies/nested_validation/`.

We record the mechanism because it matters for how this amendment should be
read: the gap surfaced through a crash, not through an inconvenient number.

## The amendment

1. **All schemes available in the build are run.** The CSV contains every
   scheme, including HAO-PPO. `ppo_reserve.py` is copied into
   `studies/nested_validation/` so the seventh scheme runs rather than
   aborting.
2. **Two composites are formed at analysis time by subsetting that CSV.**
   - **Primary:** the six Study-2 schemes (Static, Reactive, Centralized,
     Centralized Predictive, Flat-Agent, HAO-PRA-U). This is the set
     comparable with Study 2, which is the comparison this study exists to
     make.
   - **Secondary:** all seven, reported beside it.
3. **$\bar{J}$ is unaffected.** Its normalizers are fixed constants, so it
   does not move with the scheme set, and it is reported once.
4. The decision rules in Sec. 4 are unchanged and apply to the primary
   composite.

## Why both rather than one

We could have amended the registration to name six and run only those. We
are not doing that, and the reason should be on the record.

Sec. VII-C of the manuscript reports that admitting HAO-PPO to a comparison
set removed the decisiveness of two of the framework's wins. We therefore
already know the direction in which this choice pushes. An author who fixes
an unspecified scheme set *after* a crash, knowing that, has exercised a
degree of freedom the protocol exists to remove, even if the choice is
defensible on its merits, and even if the crash is what prompted it.

Running both and reporting both removes the choice. If the two composites
agree, nothing is lost. If they disagree, that disagreement is itself the
comparison-set dependence the paper warns about, appearing again in its own
results, and it is reportable.

## A second defect found in the same inspection

Not a registration matter, but recorded here because it was found while
fixing the above and because it would have invalidated this study silently.

The driver called `q_update` without passing `gamma`. That parameter
defaults to `None`, and `q_update` then falls back to the registered
`Q_GAMMA`, discarding the `hao_gamma` that `tune_all_schemes` had just
selected. The framework would have run with an untuned discount factor while
every baseline ran with its tuned margin: the precise budget asymmetry this
study exists to remove, reintroduced inside the study correcting it, and
invisible in the output.

The original `shift_driver.py` omits `gamma` correctly, because on the build
Study 2 used the framework received no search at all and there was nothing
to pass. Carrying the line across builds is what broke it.

Fixed before any result was produced. No output exists that was generated
with the defect.

## Digests

| file | SHA-256 |
|---|---|
| `PREREGISTRATION_SHIFT_SYMMETRIC.md` as deposited | _(paste)_ |
| this amendment | _(paste after committing)_ |
| `shift_symmetric_driver.py` as run | _(paste)_ |

Deposit this amendment with the same service that holds the registration, so
that the amendment is timestamped independently too. An amendment that is
only author-attested, attached to a registration that is not, would leave the
study no better off than Study 2.
