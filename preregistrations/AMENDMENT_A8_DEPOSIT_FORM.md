# AMENDMENT A8: the form of the Studies 14 to 16 deposits

**Date:** 2026-10-19
**Scope:** applies to `PREREGISTRATION_STATIC_EDGE_CONFIRM.md` (Study 14),
`PREREGISTRATION_SHIFT_SYMMETRIC.md` (Study 15) and
`PREREGISTRATION_SHIFT_SYMMETRIC_V2.md` (Study 16). No section of any of
those documents is edited.

---

## 1. What was deposited

| deposit | DOI | Zenodo publication | covers |
|---|---|---|---|
| v2.0.0 | `10.5281/zenodo.23444965` | 2026-10-18 UTC | Studies 14 and 15 |
| v2.0.1 | `10.5281/zenodo.23459209` | 2026-10-18 UTC | Study 16 |

Both are whole-repository release archives,
`hao-pra-u-evaluation-2.0.0.zip` and `-2.0.1.zip`, titled "Code for:
Pre-Registered Evaluation of Learned Slice Orchestration in 6G".

## 2. Precedence is established

The repository history fixes the ordering:

```
64f4bfe  2026-10-17T22:10:37-04:00  Static Edge and Symmetric Shift
                                    preregistration files.
e76258b  2026-10-17T22:17:50-04:00  Added DOI for the preregistration files.
4bdd815  2026-10-18T16:46:39-04:00  Static Edge and Symmetric Shift V2
                                    preregistration files and scripts
075d6d0  2026-10-18T16:57:19-04:00  Added DOI.
```

Studies 14 and 15 were registered and deposited at 22:10 and 22:17 local
time on 2026-10-17, before either driver was run. Study 16 was registered
and deposited at 16:46 and 16:57 local time on 2026-10-18; its `--validate`
run, which executes the registered S0 control, completed that evening, and
the sweep after it.

The Zenodo publication dates read 2026-10-18 for both deposits because
Zenodo records UTC and the local offset is $-4$. The 22:17 local deposit is
2026-10-18T02:17Z.

## 3. Isolation is not established

Sec. III-A requires two things of a deposit: that it be independently
timestamped, and that it carry the registration **alone, with no results in
the same deposit**. The first holds. The second does not.

A release archive contains the repository as it stood, so each deposit
carries the registration together with every result committed before it.
Neither deposit contains results of the study it registers, since both
precede their own runs, but each contains results of earlier studies.

The failure mode is worth naming because it is the convenient one. The
release pipeline already produced an archive and already minted a DOI, so
reaching for it required no extra step, while depositing the registration
by itself required a second upload. We took the path that was already
there.

## 4. What this changes in the paper

Table II keeps superscript `c` for Studies 14 to 16, because the property
that superscript asserts, that precedence does not rest on the authors'
word, is the one that holds. The note under the table now states that the
deposits are release archives rather than registration-only deposits, and
says which half of the requirement they fail. Sec. VIII records the same
point as the second half of the lesson on hashing registrations.

## 5. What is not done, and why

We are not making registration-only deposits of these three documents now.
A deposit made after the runs establishes nothing about precedence, and
adding one would create a record whose date invites exactly the confusion
this amendment exists to prevent. The deposits stand as they are, described
accurately.

Future registrations in this line of work are deposited alone.

## 6. Version DOIs, confirmed

Both `10.5281/zenodo.23444965` and `10.5281/zenodo.23459209` are version
DOIs, not the record's concept DOI. This was checked against the Zenodo
record pages, which show them resolving to versions 2.0.0 and 2.0.1
respectively.

This matters for a reason specific to a precedence claim. A concept DOI
always resolves to the newest version of a record, so citing one would mean
that a later release, deposited under the same record, silently redirected
the paper's citation to an archive containing the registered studies' own
results. The claim in Table II would then have become false after
publication, without anything in the paper changing. Version DOIs pin each
archive permanently, so the two cited deposits will always be the ones made
on 2026-10-17 and 2026-10-18, before their respective runs.

Later releases of the repository may and should be made under the same
record; they will not disturb these citations.
