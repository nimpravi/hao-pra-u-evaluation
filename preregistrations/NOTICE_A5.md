# Amendment A5: unamended post-results edits to three pre-registration documents

**Date of notice:** see the commit that carries this file
**Files:** `preregistrations/EXPERIMENT3_PREREGISTRATION.md`,
`preregistrations/SHIFT_STUDY_PREREGISTRATION.md`,
`preregistrations/SWEEP_PREREGISTRATION.md`
**Effect on reported results:** none
**Effect on the registration record:** material, and the reason for this notice

Two commits edited pre-registration documents after the results for those
studies had already been committed. Neither carried an amendment notice at the
time, which the documents themselves require. This notice supplies it,
retrospectively and in full.

## Why this went unnoticed

`preregistrations/hashes.txt` pins the simulator, the generator and the study
drivers. It has never pinned the pre-registration documents. The freeze
mechanism therefore protected the code and left the registrations editable
without tripping anything. The six registration digests are added to
`hashes.txt` in the commit carrying this notice.

## Edit 1: `ccea633`, 16 June 2026, registration dates removed

The three documents, as first committed in `1a6426a`, each declared a
registration date. `ccea633` deleted all three lines:

| file | line as registered in `1a6426a` |
|---|---|
| `SWEEP_PREREGISTRATION.md` | `Status: registered BEFORE execution. Date: 2026-06-11.` |
| `SHIFT_STUDY_PREREGISTRATION.md` | `Status: registered BEFORE implementation and execution. Date: 2026-06-12.` |
| `EXPERIMENT3_PREREGISTRATION.md` | `**Status:** Registered BEFORE implementation and execution. Date: 2026-06-13.` |

Nothing else changed in any of the three files: one line each, three
insertions and three deletions in total.

`b8ae89c`, which added the results for all three studies, precedes `ccea633`.
The edit therefore removed evidentiary content from a registration document
after that study's results were in the record.

**This notice restores the three dates**, to the exact text above. The
registered versions remain retrievable and the restoration is checkable:

```bash
git show 1a6426a:preregistrations/SWEEP_PREREGISTRATION.md | head -2
git show 1a6426a:preregistrations/SHIFT_STUDY_PREREGISTRATION.md | head -2
git show 1a6426a:preregistrations/EXPERIMENT3_PREREGISTRATION.md | head -2
```

## Edit 2: `9ddaf7e`, 5 July 2026, two paragraphs removed

Removed from `EXPERIMENT3_PREREGISTRATION.md`:

1. From Sec. 6, after the list of forecast diagnostics:

   > These diagnostics test H2 and let a reviewer see *whether prediction
   > actually paid information* at each level, independent of whether
   > HAO-PRA-U won.

2. The closing housekeeping line:

   > *End of pre-registration. Commit this document (freeze) before
   > proceeding to deliverable 3.*

The second is housekeeping and carries no scientific content. The first is
not: it stated which hypothesis the diagnostics served and asserted that the
diagnostic reading was independent of the headline outcome. Removing it after
results were observed is exactly the class of edit Sec. 7 of that document,
titled "What is NOT permitted after results are observed", prohibits.

**Both paragraphs are restored** in the commit carrying this notice.

### What did not change

No hypothesis was added, removed or reworded. No decision rule, significance
threshold, correction procedure, metric, weight, seed, grid point or
diagnostic variable changed. `precursor_channel_used` and the other
diagnostics listed in Sec. 6 are present in the registered version and in the
current one. The paragraph removed described the diagnostics; it did not
define them.

This edit also postdates the Zenodo deposit (record 20721439, v1.0.0,
16 June 2026), so the deposited copy of this file differs from the current one.
The deposit is the version with edit 1 applied and edit 2 not applied.

## Status of the three registration dates

The restored dates, 11, 12 and 13 June 2026, are **author-declared, not
third-party witnessed**. They precede this repository's first commit
(`2a99f80`, 16 June 2026), so nothing in the repository or in the Zenodo
deposit can corroborate them. What the record does support:

- The declarations were committed in `1a6426a`, which is an ancestor of
  `b8ae89c`, the commit that added the results for all three studies.
- The three dates are distinct and sequential, and their order matches the
  prerequisite chain each document states: the sweep closes before the
  regime-shift study is registered, and both close before Experiment 3.

That is internal consistency and commit-order evidence. It is not a timestamp.
The paper states it as attestation, and this notice does the same.

## Why the dates are restored rather than left out

Removing them made the record weaker, not cleaner. The header of
`EXPERIMENT3_PREREGISTRATION.md` claims that "the freeze timestamp is
witnessed by this file's commit in the repository history, which precedes the
commit of any Experiment-3 result." With the date deleted, that sentence was
the only remaining timestamp claim, and it is the weaker of the two: every
commit in the initial history shares one date, so the commit witnesses order
and not time. Restoring the declared date and labelling it as attestation
states more and claims less.

The header sentence is amended accordingly in the same commit, to read that
the declared registration date is author-attested and that the commit
recording it precedes the commit of any result for this study.
