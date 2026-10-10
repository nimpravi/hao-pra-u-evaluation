# Amendment A4: post-registration edit to `precursor_traffic.py`

**Date of edit:** see the commit that carries this notice
**File:** `studies/predictability/precursor_traffic.py`
**Nature of change:** module docstring only; no executable content
**Effect on reported results:** none

## What changed

The module docstring originally quoted peer-review correspondence verbatim
while explaining why the Experiment-3 traffic family exists. Peer review is
confidential, so that text was removed from the public artifact and replaced
with a neutral statement of the same scientific point: that the Experiment-1
and Experiment-2 environment uses memoryless Bernoulli bursts, which are
unforecastable by construction, and that this generator exists to remove
that limitation so prediction can in principle pay.

No code, constant, control flow or interface was touched.

## Digests

| | SHA-256 | where |
|---|---|---|
| As registered | `d2c8f28c5220e2f60b7fc3d5c0c28a154757649b20a595df2ef74905264f7aa4` | commit `4c624aa` |
| Current, pinned | `7a845d116d9504383537138ce97bc8725d947914f5d4128e935f1c06b7f237af` | this commit |

The registered version is not deleted. It remains retrievable from git
history, so the equivalence claim below can be checked independently:

```bash
git show 4c624aa:studies/predictability/precursor_traffic.py > /tmp/registered.py
sha256sum /tmp/registered.py      # d2c8f28c...
```

## Why the edit does not affect any reported result

The claim is not that the change "looks cosmetic". It is that the two
versions parse to the same abstract syntax tree once docstrings are removed,
which means the interpreter executes identical structure. Comments never
reach the AST at all, and line and column numbers are discarded, so
reformatting cannot hide in the comparison.

Reproduce it:

```bash
python3 verify_precursor_equivalence.py /tmp/registered.py \
        studies/predictability/precursor_traffic.py
```

Expected output: `AST VERDICT: EQUIVALENT`.

The one case where AST equality would not imply behavioural equality is a
program that reads its own `__doc__` and branches on it. Neither
`precursor_traffic.py` nor `run_experiment3.py` references `__doc__`; the
verifier checks this and reports it.

The module's self-test (`python3 precursor_traffic.py`) was run after the
edit and reproduces the registered generator properties: burst rate
approximately invariant across the predictability knob (0.043 to 0.048
against a 0.05 target), and `corr(o, x)` rising from about 0.00 at
predictability 0 to 1.00 at predictability 1.

## Interaction with the pinned-hash machinery

Both pins were updated in the same commit as the edit:

* `preregistrations/hashes.txt`
* `EXPECTED_HASHES` in `studies/predictability/run_experiment3.py`

Updating a pin silently would amount to retroactively re-registering the
artifact, which is the practice this project's protocol exists to prevent.
The pin is updated here only together with this notice, the superseded
digest, the commit that still holds the registered bytes, and a mechanical
equivalence proof a third party can re-run.

## Line endings

The pinned digests are over LF-terminated files. `.gitattributes` enforces
`eol=lf` for `*.py` so that a checkout on Windows does not produce CRLF
files whose digests differ from the pins. Without it, `sha256sum -c` fails
for every pinned file on a default Windows clone, which looks like artifact
corruption and is not.
