"""
ppo_reserve.py -- PPO reserve-selection controller for Study 5.

Purpose
-------
Reviewers R1, R5 and the AE all observed that a 36-state tabular Q-learner
may not represent the class of learned orchestrators the paper comments on.
This module supplies a deep-RL reserve controller so the comparison can be
made directly.

Design principle: ISOLATE THE LEARNER.
The PPO scheme ("HAO-PPO") is identical to HAO-PRA-U in every respect except
the global reserve-selection step. Domain-agent allocation, inter-domain
rebalancing, edge correction, lifecycle handling and all metric computation
are shared code. The comparison is therefore an ablation of the reserve
learner, not a comparison of two different frameworks.

Differences from the tabular learner, both deliberate and registered:
  * State is continuous and richer: 5 features instead of 3 binned into 36
    discrete cells. This directly addresses the "rich state representation"
    objection.
  * Policy is a small MLP actor-critic trained with clipped-surrogate PPO
    and GAE, rather than a lookup table updated by one-step Q-learning.
  * The action space is UNCHANGED (the same RESERVE_ACTIONS), so the two
    controllers choose from an identical set of decisions.

Everything is CPU-only and seeded for reproducibility.
"""
import numpy as np
import torch
import torch.nn as nn

# Registered PPO defaults (Study 5 pre-registration, Sec. 4).
PPO_HIDDEN = 64
PPO_LR = 3e-4
PPO_CLIP = 0.2
PPO_ENTROPY = 0.01
PPO_VF_COEF = 0.5
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_UPDATE_EPOCHS = 4
PPO_MINIBATCH = 128
PPO_EPISODES = 12          # replay passes over the train window (final fit)
PPO_EPISODES_TUNE = 4      # cheap proxy used during model selection only
PPO_MAX_GRAD_NORM = 0.5
# The composite reward in this environment sits near a large constant with
# small variance across reserve actions, so raw returns are dominated by an
# offset the critic must first learn. Standardizing rewards within each
# rollout removes that offset directly and is what makes the advantages
# informative. Registered as part of the PPO configuration (Study 5).
PPO_NORMALIZE_REWARDS = True

STATE_DIM = 5


class ActorCritic(nn.Module):
    """Small MLP with a categorical policy head and a value head."""

    def __init__(self, n_actions, hidden=PPO_HIDDEN):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class PPOReserveAgent:
    """Reserve-factor selector trained with PPO.

    Exposes the same two operations the tabular learner needs:
        act(state, training)  -> action index
        record(reward)        -> store the reward for the last action
    plus finish_episode()/update() driven by the training loop.
    """

    def __init__(self, n_actions, seed, hidden=PPO_HIDDEN, lr=PPO_LR,
                 clip=PPO_CLIP, entropy=PPO_ENTROPY):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.net = ActorCritic(n_actions, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip = float(clip)
        self.entropy = float(entropy)
        self._reset_buffer()
        self._pending = None

    # ---------------- rollout buffer ----------------
    def _reset_buffer(self):
        self.buf = {k: [] for k in
                    ("s", "a", "logp", "v", "r", "done")}

    def act(self, state, training=True):
        """Select a reserve action. Greedy (argmax) when not training."""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            logits, value = self.net(s)
            dist = torch.distributions.Categorical(logits=logits)
            if training:
                a = dist.sample()
            else:
                a = torch.argmax(logits)
            logp = dist.log_prob(a)
        if training:
            self._pending = (np.asarray(state, dtype=np.float32),
                             int(a.item()), float(logp.item()),
                             float(value.item()))
        return int(a.item())

    def record(self, reward, done=False):
        """Attach the realized reward to the most recent action."""
        if self._pending is None:
            return
        s, a, logp, v = self._pending
        self.buf["s"].append(s)
        self.buf["a"].append(a)
        self.buf["logp"].append(logp)
        self.buf["v"].append(v)
        self.buf["r"].append(float(reward))
        self.buf["done"].append(bool(done))
        self._pending = None

    # ---------------- PPO update ----------------
    def update(self):
        """Clipped-surrogate PPO update over the collected rollout."""
        n = len(self.buf["r"])
        if n < 2:
            self._reset_buffer()
            return

        s = torch.as_tensor(np.array(self.buf["s"], dtype=np.float32))
        a = torch.as_tensor(np.array(self.buf["a"], dtype=np.int64))
        old_logp = torch.as_tensor(np.array(self.buf["logp"],
                                            dtype=np.float32))
        vals = np.array(self.buf["v"], dtype=np.float32)
        rews = np.array(self.buf["r"], dtype=np.float32)
        if PPO_NORMALIZE_REWARDS and rews.std() > 1e-12:
            rews = (rews - rews.mean()) / (rews.std() + 1e-8)
        dones = np.array(self.buf["done"], dtype=bool)

        # GAE(lambda)
        adv = np.zeros(n, dtype=np.float32)
        last = 0.0
        for t in reversed(range(n)):
            nonterm = 0.0 if dones[t] else 1.0
            nextv = vals[t + 1] if t + 1 < n else 0.0
            delta = rews[t] + PPO_GAMMA * nextv * nonterm - vals[t]
            last = delta + PPO_GAMMA * PPO_LAMBDA * nonterm * last
            adv[t] = last
        ret = adv + vals

        adv_t = torch.as_tensor(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret)

        idx = np.arange(n)
        for _ in range(PPO_UPDATE_EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, n, PPO_MINIBATCH):
                mb = idx[start:start + PPO_MINIBATCH]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb)
                logits, value = self.net(s[mb_t])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a[mb_t])
                ratio = torch.exp(logp - old_logp[mb_t])
                surr1 = ratio * adv_t[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip,
                                    1 + self.clip) * adv_t[mb_t]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((value - ret_t[mb_t]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + PPO_VF_COEF * v_loss - self.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         PPO_MAX_GRAD_NORM)
                self.opt.step()
        self._reset_buffer()


def build_state(avg_p, max_p, err, mean_util, mean_risk):
    """Continuous 5-feature state.

    The first three mirror the tabular learner's inputs (average and maximum
    cross-domain pressure, recent forecast error) but are passed unbinned.
    The last two (previous mean utilization, mean SLA-risk) are the
    additional context a richer policy can exploit; the tabular controller
    has no capacity to use them.
    """
    return np.array([avg_p, max_p, err, mean_util, mean_risk],
                    dtype=np.float32)
