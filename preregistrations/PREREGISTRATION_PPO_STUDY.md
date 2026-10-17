# PRE-REGISTRATION: Deep-RL Reserve Control (Study 5)

**Status:** FROZEN BEFORE CONFIRMATORY RESULTS. Commit this document and
record its SHA-256 in `preregistrations/hashes.txt` before running
`ppo_driver.py`. Amendments are appended and dated; no section is edited
after the first confirmatory run.

**Builds on:** `PREREGISTRATION_NESTED_VALIDATION.md` (Study 4). The nested
split, equal-budget rule, decision rules, and statistical procedures are
inherited unchanged unless stated otherwise.

---

## 1. Motivation

The paper's negative finding is established on a 36-state tabular
Q-learner, which may not
represent the deep and graph-based learned orchestrators the literature
proposes. The paper scopes its claims accordingly, but scoping is not the
same as evidence. This study supplies the missing controller.

Study 4 sharpened the question. Under equal search budgets the tabular
learner won 2 of 20 regimes, always at generous headroom, and its selected
reserve multiplier pinned to the grid maximum in 114 of 200 cells. Its
winning strategy was essentially "provision as much as permitted." The
question this study asks is therefore specific:

> Can a policy class rich enough to condition on continuous state find a
> reserve strategy better than a constant maximum margin, particularly
> under tight capacity where the tabular learner has no good setting?

## 2. Hypotheses (registered before confirmatory observation)

- **H5.1.** HAO-PPO wins strictly more regimes than HAO-PRA-U (> 2 of 20).
- **H5.2.** HAO-PPO wins at least one regime at tight capacity
  (h = 1.4), where HAO-PRA-U wins none.
- **H5.3.** HAO-PPO's learned policy is state-dependent. Operationalized
  before the confirmatory run: over a fixed 36-point probe grid of
  (average pressure, maximum pressure) states, the modal greedy action
  accounts for **less than 80%** of selections, averaged over regimes and
  seeds. A simple "more than one distinct action" criterion was rejected
  during harness validation as too weak: exploratory runs produced two
  distinct actions with a 93% modal share, which is not meaningful state
  dependence.
- **H5.4 (primary).** Tuned static over-provisioning still wins a majority
  of the 20 regimes decisively.

**Interpretation fixed in advance.** H5.3 is the discriminating test. If
HAO-PPO converges to a near-constant policy, then a richer policy class
finds nothing that a tuned margin does not already capture, and the paper's
conclusion generalizes beyond lightweight tabular control. If H5.3 holds
and H5.1/H5.2 also hold, the conclusion does not generalize and must be
restated as specific to weak controllers. We commit to reporting either
outcome.

## 3. The controller

`HAO-PPO` is identical to HAO-PRA-U in every respect except global reserve
selection. Domain-agent allocation, inter-domain rebalancing, edge
correction, lifecycle handling, and all metric computation are shared code
paths. Control-plane costs are identical (control delay 0.8 ms,
orchestration energy 0.06, 24 messages per interval), so HAO-PPO receives
no accounting advantage. The comparison is an ablation of the learner.

Differences, all registered:

| | HAO-PRA-U | HAO-PPO |
|---|---|---|
| State | 3 variables binned into 36 cells | same 3 unbinned, plus previous mean utilization and mean SLA-risk (5 continuous features) |
| Policy | Q-table | MLP actor-critic, 2 hidden layers |
| Update | one-step Q-learning | clipped-surrogate PPO with GAE |
| Action space | 7 reserve actions | **identical** |

Rewards are standardized within each rollout before advantage estimation.
This is registered as part of the configuration: the composite reward sits
near a large constant with small variance across reserve actions, so raw
returns are dominated by an offset the critic must otherwise learn first.

## 4. Search budgets

**Primary (equal budget).** HAO-PPO receives exactly 9 validation
evaluations, identical to every other scheme under Study 4:

| Parameter | Values |
|---|---|
| learning rate | {1e-4, 3e-4, 1e-3} |
| clip ratio | {0.1, 0.2, 0.3} |

Full factorial, 3 x 3 = 9. Selection uses `PPO_EPISODES_TUNE = 4` replay
passes as a cheap proxy; the final fit uses `PPO_EPISODES = 12`, mirroring
the tabular controller's `PRETRAIN_EPOCHS_TUNE` arrangement.

Note this budget is generous to HAO-PPO in a way we state openly: it
receives the same number of *hyperparameter evaluations* as a scheme with
one scalar margin, while also having far greater representational capacity.

**Robustness (best-effort, deliberately asymmetric).** To ensure the
negative result is not an artifact of under-tuning, a second configuration
grants HAO-PPO a 36-point budget:

| Parameter | Values |
|---|---|
| learning rate | {1e-4, 3e-4, 1e-3} |
| clip ratio | {0.1, 0.2, 0.3} |
| entropy coefficient | {0.0, 0.01} |
| hidden width | {64, 128} |

This is run at four registered corner regimes, (w_E, h) in
{(0.0188, 1.4), (0.0188, 2.3), (0.30, 1.4), (0.30, 2.3)}, with all 10
seeds. It is reported as an explicit asymmetry favouring the learned
method: if HAO-PPO still fails to win with four times the search budget of
its opponents, the negative result is correspondingly stronger. Results
from this configuration are reported separately and never merged with the
primary analysis.

## 5. Design and decision rules

Inherited from Study 4 without change: 5 x 4 regime sweep, 10 seeds,
nested train/validation/test split (550/150/300), selection by mean reward
on the validation window, paired Wilcoxon signed-rank as the primary test
at alpha = 0.05, Holm-Bonferroni and Benjamini-Hochberg corrections both
reported, Cohen's d_z as the primary effect size, and the per-seed min-max
composite over the ten registered metrics.

Seven schemes are now compared. The winner of a regime is the scheme with
the highest mean composite that is significantly better than the runner-up.

## 6. Tripwires

- **T5.1.** With `PPO_EPISODES = 0` the agent must be untrained and its
  performance must not exceed the trained configuration systematically.
- **T5.2.** The PPO implementation must solve a synthetic contextual
  bandit (correct action determined by one state feature) well above the
  1/7 chance rate. *Recorded during harness validation before the
  confirmatory run: 53% versus 14% chance, so the implementation learns.*
- **T5.3.** HAO-PPO and HAO-PRA-U must produce identical values for every
  metric that does not depend on the reserve action.
- **T5.4.** Every scheme receives exactly its registered budget.

Tripwire failures void the run.

## 7. Harness validation recorded before the confirmatory run

Two observations were made while building the harness, before any
confirmatory result, and are recorded here for audit:

1. **Without reward standardization the PPO policy collapses.** Greedy
   evaluation selected a single action for all states, while the sampling
   policy remained near-uniform, indicating the advantages carried no
   signal. Reward standardization within each rollout resolved this and is
   part of the registered configuration (Sec. 3).
2. **Even after that fix, exploratory runs at h = 2.0 showed the greedy
   policy converging toward a single maximum-reserve action.** This
   observation motivated H5.3 but is not itself a confirmatory result; H5.3
   is tested over the full registered sweep.

Recording these here, before the confirmatory runs, is what prevents them
from being presented afterwards as if they had been anticipated.

## 8. Artifacts

- `studies/ppo_study/ppo_reserve.py` (frozen, hash pinned)
- `studies/ppo_study/hao_pra_u_sim.py` (frozen, hash pinned)
- `studies/ppo_study/ppo_driver.py`
- `results/ppo_study/all_cells_ppo.csv`
- `results/ppo_study/tuned_params_ppo.json`
- `results/ppo_study/policy_diagnostics.csv` (action distribution per
  regime and seed, for H5.3)

---

*Registered by: Praveen Hegde*
*Date frozen: __________ (fill before first run)*
