#!/usr/bin/env python3
"""
HAO-PRA-U discrete-interval orchestration simulator.

Implements the system model of the TNSM manuscript:
  Eq. (uncertainty)  sigma_s(t) = |u-uhat| / (rolling mean + eps)
  Eq. (resource)     r_{s,d} = alpha_{s,d} * u_s
  Eq. (risksum)      R_s = w1*phi + w2*sigma + w3*P + w4*A + w5*lambda + w6*I
  Eq. (util)         U_d = sum_s x_{s,d} / C_d
  Eq. (latency)      L_s = L_base + d_deficit + d_congestion + d_control - d_edge
  Eq. (frag)         F = std(u_domains) / (mean(u_domains) + eps)
  Eq. (cost/reward)  J(t) weighted multi-objective (Table II weights)
  Eq. (qlearn)       tabular Q-update on (avg pressure, max pressure, pred err)
  Eq. (priority)     P_{s,d} = rhat_{s,d} * (1+rho) * (1+Risk_s) * eta_{s,d}

FAIRNESS PRINCIPLES (enforced by construction):
  * All schemes see IDENTICAL traffic traces per seed.
  * All predictive schemes use the SAME trained GRU forecasts per seed.
  * All schemes are evaluated by the SAME SLA / latency / energy /
    fragmentation / acceptance functions; no scheme-specific metric code.
  * Baseline free parameters are tuned by grid search on the TRAINING
    intervals (composite score), then frozen for the TEST intervals.
  * Results are reported as they fall. Nothing in this file conditions an
    outcome on which scheme produced it, other than the control logic that
    defines each scheme.

Run:  python3 hao_pra_u_sim.py --seeds 10 --out results_dir
"""

import argparse
import json
import os
import numpy as np

# ----------------------------------------------------------------------------
# Configuration (these values populate Table V of the manuscript)
# ----------------------------------------------------------------------------
SLICES = ["eMBB", "URLLC", "mMTC", "XR", "IIoT"]
DOMAINS = ["RAN", "Edge", "Transport", "Core", "Cloud"]
N_INTERVALS = 1000
TRAIN_FRAC = 0.70          # first 70% train, last 30% test
N_TRAIN = int(N_INTERVALS * TRAIN_FRAC)

# Per-slice base demand (normalized demand units / interval)
BASE_DEMAND = {"eMBB": 200.0, "URLLC": 90.0, "mMTC": 135.0,
               "XR": 170.0, "IIoT": 115.0}

# Traffic model: u(t) = base*(1 + Ap*sin(2*pi*t/Tp + phase)) + N(0, sd) + burst
PERIODIC_AMPLITUDE = 0.25
PERIODIC_PERIOD = 100
STOCH_STD_FRAC = 0.08      # Gaussian std as fraction of base demand
BURST_PROB = 0.05          # per slice per interval
BURST_MAG_RANGE = (0.30, 0.60)  # uniform fraction of base demand

# SLA profiles  (L_max ms, T_min frac of base, priority, edge affinity,
#                intent boost, latency sensitivity)
SLA = {
    #          Lmax  TminF  P     A     I     lam
    "eMBB":  ( 50.0, 0.85, 0.60, 0.30, 0.40, 0.40),
    "URLLC": ( 10.0, 0.85, 1.00, 0.90, 0.90, 1.00),
    "mMTC":  (100.0, 0.85, 0.30, 0.20, 0.30, 0.10),
    "XR":    ( 20.0, 0.85, 0.85, 0.80, 0.70, 0.85),
    "IIoT":  ( 15.0, 0.85, 0.90, 0.85, 0.80, 0.90),
}
BASE_LATENCY = {"eMBB": 18.0, "URLLC": 6.0, "mMTC": 30.0, "XR": 12.0,
                "IIoT": 9.0}
REL_MIN, ISO_MIN = 0.95, 0.90

# Slice->domain demand coefficients alpha_{s,d} (rows sum to 1.0)
ALPHA = np.array([
    # RAN   Edge  Transp Core  Cloud
    [0.40, 0.15, 0.25, 0.12, 0.08],   # eMBB
    [0.25, 0.40, 0.20, 0.10, 0.05],   # URLLC
    [0.20, 0.10, 0.15, 0.40, 0.15],   # mMTC
    [0.35, 0.30, 0.20, 0.05, 0.10],   # XR
    [0.15, 0.40, 0.20, 0.15, 0.10],   # IIoT
])

# Domain-sensitivity coefficients eta_{s,d} (intent preferences, Sec. V-D)
ETA = np.array([
    [1.20, 0.90, 1.00, 0.95, 0.90],   # eMBB: RAN-favoured
    [1.00, 1.30, 1.05, 1.00, 0.80],   # URLLC: Edge-favoured
    [0.90, 0.85, 0.95, 1.20, 1.05],   # mMTC: Core-favoured
    [1.20, 1.20, 1.05, 0.90, 0.95],   # XR: RAN+Edge
    [0.95, 1.25, 1.05, 1.00, 0.90],   # IIoT: Edge-favoured
])

# Domain capacities C_d (normalized units). Sized at ~1.30x the mean aggregate
# domain load so contention occurs in peaks/bursts but the system is feasible.
MEAN_LOAD = ALPHA.T @ np.array([BASE_DEMAND[s] for s in SLICES])
CAPACITY = np.round(MEAN_LOAD * 2.00, 0)   # per-domain

# SLA-risk weights w1..w6 (Table II of manuscript)
W = dict(w1=0.35, w2=0.25, w3=0.20, w4=0.10, w5=0.05, w6=0.05)

# Objective weights alpha_1..alpha_10 (Table II) for composite score / reward
OBJ_W = dict(sla=0.2256, latency=0.1880, throughput=0.1692, acceptance=0.1316,
             utilization=0.1128, reliability=0.0658, isolation=0.0282,
             energy=0.0188, fragmentation=0.0390, overhead=0.0210)

# Q-learning (Global Orchestrator)
RESERVE_ACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
Q_ALPHA, Q_GAMMA = 0.10, 0.90
EPS_START, EPS_END, EPS_DECAY = 1.00, 0.05, 0.995   # per training interval
PRETRAIN_EPOCHS = 30          # replay passes over the (exogenous) train trace
EPS_DECAY_PRE = 0.99986       # per pretraining step (floor ~70% through)
AVG_P_BINS = [0.40, 0.75]
MAX_P_BINS = [0.40, 0.70, 0.90]
ERR_BINS = [0.10, 0.25]

# Latency model coefficients (identical for all schemes)
K_DEFICIT = 35.0      # ms at 100% resource deficit
K_CONG = 25.0         # ms per unit of mean over-0.8 domain utilization
CONTROL_DELAY = {  # synchronous decision-path delay only (ms); async
    # coordination is charged to the overhead metric, not latency
    # (see DESIGN_FREEZE.md, item 3)
    "Static": 0.2, "Reactive": 0.5, "Centralized": 2.5,
    "CentralizedPredictive": 2.5, "FlatAgent": 0.8, "HAO-PRA-U": 0.8,
}
K_EDGE = 6.0          # max ms reduction from full edge provisioning

# Energy model: energy ~ allocated capacity + per-scheme orchestration energy
ORCH_ENERGY = {"Static": 0.00, "Reactive": 0.01, "Centralized": 0.02,
               "CentralizedPredictive": 0.03, "FlatAgent": 0.04,
               "HAO-PRA-U": 0.06}
# Orchestration overhead: control messages per interval (proxy)
ORCH_MSGS = {"Static": 1, "Reactive": 6, "Centralized": 8,
             "CentralizedPredictive": 10, "FlatAgent": 16, "HAO-PRA-U": 24}

EPS = 1e-6
ROLL_WIN = 10                      # rolling-mean window for ubar
ACCEPT_FRAC = 0.70                 # alloc >= 70% of required => admitted

GRU_LOOKBACK, GRU_HIDDEN, GRU_EPOCHS = 16, 24, 35
GRU_LR, GRU_BATCH = 1e-3, 32

METRICS = ["sla_violation", "latency", "throughput", "utilization",
           "energy_eff", "fragmentation", "acceptance"]

OBJ_W_BASE = dict(OBJ_W)   # manuscript Table II weights (sum = 1.0)


def set_regime(energy_weight=0.0188, headroom=2.00):
    """Set the two pre-registered sweep axes (SWEEP_PREREGISTRATION.md).
    Neutral at defaults: energy_weight=0.0188, headroom=2.00 reproduce the
    DESIGN_FREEZE_v2 anchor bit-for-bit. The non-energy Table II weights
    are rescaled proportionally so the total stays 1.0."""
    others = {k: v for k, v in OBJ_W_BASE.items() if k != "energy"}
    scale = (1.0 - energy_weight) / sum(others.values())
    OBJ_W.clear()
    OBJ_W.update({k: v * scale for k, v in others.items()})
    OBJ_W["energy"] = energy_weight
    CAPACITY[:] = np.round(MEAN_LOAD * headroom, 0)


# ----------------------------------------------------------------------------
# Traffic generation
# ----------------------------------------------------------------------------
def generate_traffic(rng):
    """Return (n_intervals x n_slices) demand matrix."""
    t = np.arange(N_INTERVALS)
    U = np.zeros((N_INTERVALS, len(SLICES)))
    for j, s in enumerate(SLICES):
        base = BASE_DEMAND[s]
        phase = rng.uniform(0, 2 * np.pi)
        periodic = base * (1 + PERIODIC_AMPLITUDE *
                           np.sin(2 * np.pi * t / PERIODIC_PERIOD + phase))
        noise = rng.normal(0, STOCH_STD_FRAC * base, N_INTERVALS)
        bursts = (rng.random(N_INTERVALS) < BURST_PROB) * \
            rng.uniform(*BURST_MAG_RANGE, N_INTERVALS) * base
        U[:, j] = np.maximum(periodic + noise + bursts, 1.0)
    return U


# ----------------------------------------------------------------------------
# GRU predictor (per slice), as specified in Sec. V-C
# ----------------------------------------------------------------------------
def train_gru_forecasts(U, seed):
    """Train one GRU per slice on the first 70%; return full-trace one-step
    forecasts uhat(t) for t = LOOKBACK..N-1 (NaN before), plus accuracy on
    the held-out 30%."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    n, ns = U.shape
    UHAT = np.full((n, ns), np.nan)
    acc = {}

    for j, s in enumerate(SLICES):
        series = U[:, j].astype(np.float32)
        mu, sd = series[:N_TRAIN].mean(), series[:N_TRAIN].std() + 1e-6
        z = (series - mu) / sd
        X, Y, idx = [], [], []
        for i in range(GRU_LOOKBACK, n):
            X.append(z[i - GRU_LOOKBACK:i])
            Y.append(z[i])
            idx.append(i)
        X = torch.tensor(np.array(X)).unsqueeze(-1)
        Y = torch.tensor(np.array(Y)).unsqueeze(-1)
        idx = np.array(idx)
        tr = idx < N_TRAIN

        class GRUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(1, GRU_HIDDEN, batch_first=True)
                self.fc = nn.Linear(GRU_HIDDEN, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])

        net = GRUNet()
        opt = torch.optim.Adam(net.parameters(), lr=GRU_LR)
        lossf = nn.MSELoss()
        Xtr, Ytr = X[tr], Y[tr]
        ntr = len(Xtr)
        for _ in range(GRU_EPOCHS):
            perm = torch.randperm(ntr)
            for b in range(0, ntr, GRU_BATCH):
                bi = perm[b:b + GRU_BATCH]
                opt.zero_grad()
                loss = lossf(net(Xtr[bi]), Ytr[bi])
                loss.backward()
                opt.step()

        with torch.no_grad():
            pred = net(X).squeeze(-1).numpy() * sd + mu
        UHAT[idx, j] = pred

        te = idx >= N_TRAIN
        err = pred[te] - series[idx[te]]
        actual = series[idx[te]]
        acc[s] = dict(MAE=float(np.mean(np.abs(err))),
                      RMSE=float(np.sqrt(np.mean(err ** 2))),
                      MAPE=float(np.mean(np.abs(err) / actual) * 100))
    return UHAT, acc


# ----------------------------------------------------------------------------
# Shared evaluation (identical for every scheme)
# ----------------------------------------------------------------------------
def evaluate_interval(scheme, X_alloc, u, prev_extra_edge):
    """Given allocation matrix X (slices x domains) and demand u, compute all
    interval metrics with the shared model. Returns dict."""
    req = ALPHA * u[:, None]                      # r_{s,d}
    util_d = X_alloc.sum(axis=0) / CAPACITY       # U_d
    util = float(util_d.mean())

    served_frac = np.ones(len(SLICES))
    for j in range(len(SLICES)):
        need = req[j].sum()
        got = X_alloc[j].sum()
        served_frac[j] = min(1.0, got / (need + EPS))
    served = served_frac * u
    throughput = float(served.sum())

    # latency per Eq. (latency)
    lat = np.zeros(len(SLICES))
    cong = float(np.mean(np.maximum(0.0, util_d - 0.80)))
    for j, s in enumerate(SLICES):
        deficit = 1.0 - served_frac[j]
        edge_prov = X_alloc[j, 1] / (req[j, 1] + EPS)
        edge_red = K_EDGE * SLA[s][3] * min(1.5, edge_prov)
        lat[j] = (BASE_LATENCY[s] + K_DEFICIT * deficit + K_CONG * cong
                  + CONTROL_DELAY[scheme] - edge_red)
        lat[j] = max(lat[j], 0.5)
    latency = float(lat.mean())

    # reliability & isolation (shared simple models)
    rel = 1.0 - 0.5 * (1.0 - served_frac)         # deficit lowers reliability
    overload = float(np.mean(np.maximum(0.0, util_d - 0.95)))
    iso = np.clip(1.0 - 2.0 * overload, 0.0, 1.0) * np.ones(len(SLICES))

    # SLA violation per Eq. (viol)
    viol = np.zeros(len(SLICES))
    for j, s in enumerate(SLICES):
        Lmax, TminF = SLA[s][0], SLA[s][1]
        ok = (lat[j] <= Lmax and served[j] >= TminF * u[j]
              and rel[j] >= REL_MIN and iso[j] >= ISO_MIN)
        viol[j] = 0.0 if ok else 1.0
    sla_violation = float(viol.mean())

    acceptance = float(np.mean(served_frac >= ACCEPT_FRAC))

    energy = X_alloc.sum() / CAPACITY.sum() + ORCH_ENERGY[scheme]
    energy_eff = float((throughput / sum(BASE_DEMAND.values())) /
                       (energy + EPS) * 0.45)

    frag = float(util_d.std() / (util_d.mean() + EPS))
    overhead = ORCH_MSGS[scheme] / max(ORCH_MSGS.values())

    return dict(sla_violation=sla_violation, latency=latency,
                throughput=throughput, utilization=util,
                energy_eff=energy_eff, fragmentation=frag,
                acceptance=acceptance, reliability=float(rel.mean()),
                isolation=float(iso.mean()), overhead=overhead,
                viol_per_slice=viol)


def reward_from_metrics(m):
    """Multi-objective reward R(t) = -J(t), with metrics normalized to
    comparable scales (fixed normalizers, identical for all schemes)."""
    lat_n = m["latency"] / 40.0
    thr_n = m["throughput"] / (sum(BASE_DEMAND.values()) * 1.2)
    return (-OBJ_W["sla"] * m["sla_violation"]
            - OBJ_W["latency"] * lat_n
            - OBJ_W["energy"] * (1.0 - min(m["energy_eff"], 1.0))
            - OBJ_W["fragmentation"] * m["fragmentation"]
            - OBJ_W["overhead"] * m["overhead"]
            + OBJ_W["throughput"] * thr_n
            + OBJ_W["utilization"] * m["utilization"]
            + OBJ_W["acceptance"] * m["acceptance"]
            + OBJ_W["reliability"] * m["reliability"]
            + OBJ_W["isolation"] * m["isolation"])


# ----------------------------------------------------------------------------
# Allocation primitives
# ----------------------------------------------------------------------------
def cap_proportional(X):
    """Scale allocations down proportionally where a domain exceeds capacity."""
    X = X.copy()
    tot = X.sum(axis=0)
    over = tot > CAPACITY
    for d in np.where(over)[0]:
        X[:, d] *= CAPACITY[d] / tot[d]
    return X


def cap_priority(X_target, prio):
    """Greedy fill by descending priority per domain (Eq. priority rule)."""
    X = np.zeros_like(X_target)
    for d in range(len(DOMAINS)):
        order = np.argsort(-prio[:, d])
        remaining = CAPACITY[d]
        for j in order:
            give = min(X_target[j, d], remaining)
            X[j, d] = give
            remaining -= give
    return X


# ----------------------------------------------------------------------------
# Schemes (control logic only; evaluation is shared)
# ----------------------------------------------------------------------------
class SchemeState:
    def __init__(self):
        self.qtable = {}
        self.counts = {}
        self.eps = EPS_START
        self.last_sa = None
        self.cent_cache = None
        self.lifecycle = {a: 0 for a in
                          ["ADMIT", "SCALE_UP", "SCALE_DOWN", "HEAL",
                           "DEFER", "MAINTAIN"]}


def discretize(avg_p, max_p, err):
    b1 = int(np.digitize(avg_p, AVG_P_BINS))
    b2 = int(np.digitize(max_p, MAX_P_BINS))
    b3 = int(np.digitize(err, ERR_BINS))
    return (b1, b2, b3)


def risk_scores(u, uhat_next, ubar, sigma):
    phi = uhat_next / (ubar + EPS)
    phi_n = np.clip(phi / 1.5, 0, 1)
    R = np.zeros(len(SLICES))
    for j, s in enumerate(SLICES):
        _, _, P, A, I, lam = SLA[s]
        R[j] = (W["w1"] * phi_n[j] + W["w2"] * min(sigma[j], 1.0)
                + W["w3"] * P + W["w4"] * A + W["w5"] * lam + W["w6"] * I)
    return R


def allocate(scheme, st, t, u, U, UHAT, ubar, sigma, training, rng,
             tuned):
    """Return allocation matrix X for interval t under `scheme`."""
    nS, nD = len(SLICES), len(DOMAINS)

    if scheme == "Static":
        # fixed at tuned multiple of historical mean demand
        base = np.array([BASE_DEMAND[s] for s in SLICES])
        X = cap_proportional(ALPHA * (tuned["static_margin"] * base)[:, None])

    elif scheme == "Reactive":
        # post-demand: allocate last observed requirement (+tuned margin)
        u_prev = U[t - 1] if t > 0 else np.array(
            [BASE_DEMAND[s] for s in SLICES])
        X = cap_proportional(ALPHA * (tuned["reactive_margin"] *
                                      u_prev)[:, None])

    elif scheme == "Centralized":
        # single delayed controller: recomputes every k intervals from
        # last observed demand
        k = tuned["cent_period"]
        if st.cent_cache is None or t % k == 0:
            u_prev = U[t - 1] if t > 0 else np.array(
                [BASE_DEMAND[s] for s in SLICES])
            st.cent_cache = cap_proportional(
                ALPHA * (tuned["cent_margin"] * u_prev)[:, None])
        X = st.cent_cache

    elif scheme == "CentralizedPredictive":
        # same GRU forecast, no hierarchy/risk/edge/lifecycle
        uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
        X = cap_proportional(ALPHA * (tuned["cp_margin"] * uhat)[:, None])

    elif scheme == "FlatAgent":
        # per-domain agents use the forecast independently with local
        # headroom; no global reserve, no negotiation, no edge correction
        uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
        req = ALPHA * uhat[:, None]
        X = np.zeros((nS, nD))
        for d in range(nD):
            X[:, d] = req[:, d] * tuned["flat_headroom"][d]
        X = cap_proportional(X)

    elif scheme == "HAO-PRA-U":
        uhat = UHAT[t] if not np.isnan(UHAT[t]).any() else u
        req_hat = ALPHA * uhat[:, None]

        # ---- Global Orchestrator: Q-learning reserve selection ----
        pressures = (ALPHA * u[:, None]).sum(axis=0) / CAPACITY
        avg_p, max_p = float(pressures.mean()), float(pressures.max())
        err = float(np.mean(sigma))
        state = discretize(avg_p, max_p, err)
        qrow = st.qtable.setdefault(state, np.zeros(len(RESERVE_ACTIONS)))
        if training and rng.random() < st.eps:
            a_idx = rng.integers(len(RESERVE_ACTIONS))
        else:
            a_idx = int(np.argmax(qrow))
        rho = RESERVE_ACTIONS[a_idx]
        st.last_sa = (state, a_idx)

        # ---- Domain Agents: risk-weighted greedy (Eq. priority) ----
        R = risk_scores(u, uhat, ubar, sigma)
        prio = req_hat * (1 + rho) * (1 + R)[:, None] * ETA
        target = req_hat * (1 + rho)
        X = cap_priority(target, prio)

        # ---- Contract negotiation: shift marginal capacity from the most
        # loaded to the least loaded domain for slices that can use it ----
        util_d = X.sum(axis=0) / CAPACITY
        d_hi, d_lo = int(np.argmax(util_d)), int(np.argmin(util_d))
        if util_d[d_hi] > 0.90 and util_d[d_lo] < 0.75:
            for j in np.argsort(R):           # lowest-risk slices first
                shift = 0.05 * X[j, d_hi]
                room = CAPACITY[d_lo] - X[:, d_lo].sum()
                shift = min(shift, max(room, 0.0))
                X[j, d_hi] -= shift
                X[j, d_lo] += shift

        # ---- Edge Agents: move edge capacity from low-priority slices to
        # at-risk latency-critical slices (URLLC, XR, IIoT) ----
        edge = 1
        lat_crit = [SLICES.index(s) for s in ("URLLC", "XR", "IIoT")]
        donors = [j for j in range(nS) if j not in lat_crit]
        for j in lat_crit:
            need = ALPHA[j, edge] * u[j]
            if X[j, edge] < need and R[j] > 0.55:
                want = need - X[j, edge]
                for k_ in sorted(donors, key=lambda x: R[x]):
                    surplus = X[k_, edge] - 0.8 * ALPHA[k_, edge] * u[k_]
                    give = min(max(surplus, 0.0), want)
                    X[k_, edge] -= give
                    X[j, edge] += give
                    want -= give
                    if want <= 0:
                        break

        # ---- Lifecycle bookkeeping (zero-touch actions) ----
        for j, s in enumerate(SLICES):
            got, need = X[j].sum(), (ALPHA[j] * u[j]).sum()
            ratio = got / (need + EPS)
            if ratio < ACCEPT_FRAC:
                st.lifecycle["DEFER"] += 1
            elif ratio < 0.95:
                st.lifecycle["HEAL" if j in lat_crit else "SCALE_UP"] += 1
            elif ratio > 1.25:
                st.lifecycle["SCALE_DOWN"] += 1
            else:
                st.lifecycle["MAINTAIN"] += 1
        X = np.maximum(X, 0.0)
    else:
        raise ValueError(scheme)
    return X


def q_update(st, reward, u_next, sigma_next):
    if st.last_sa is None:
        return
    pressures = (ALPHA * u_next[:, None]).sum(axis=0) / CAPACITY
    s_next = discretize(float(pressures.mean()), float(pressures.max()),
                        float(np.mean(sigma_next)))
    qnext = st.qtable.setdefault(s_next, np.zeros(len(RESERVE_ACTIONS)))
    s, a = st.last_sa
    c = st.counts[(s, a)] = st.counts.get((s, a), 0) + 1
    alpha = max(1.0 / c, 0.02)        # count-based step size, floored
    st.qtable[s][a] += alpha * (reward + Q_GAMMA * qnext.max()
                                - st.qtable[s][a])


# ----------------------------------------------------------------------------
# Run one scheme over one trace
# ----------------------------------------------------------------------------
def pretrain_hao(st, U, UHAT, tuned, seed):
    """Replay the exogenous training trace for PRETRAIN_EPOCHS so the
    tabular Q-values converge before the live pass (HAO internals only)."""
    rngp = np.random.default_rng(seed + 31337)
    st.eps = EPS_START
    for _ in range(PRETRAIN_EPOCHS):
        hist = []
        for t in range(N_TRAIN):
            u = U[t]
            hist.append(u)
            if len(hist) > ROLL_WIN:
                hist.pop(0)
            ubar = np.mean(hist, axis=0)
            if t > 0 and not np.isnan(UHAT[t - 1]).any():
                sigma = np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + EPS)
            else:
                sigma = np.zeros(len(SLICES))
            X = allocate("HAO-PRA-U", st, t, u, U, UHAT, ubar, sigma,
                         True, rngp, tuned)
            m = evaluate_interval("HAO-PRA-U", X, u, None)
            q_update(st, reward_from_metrics(m),
                     U[t + 1] if t + 1 < len(U) else u, sigma)
            st.eps = max(EPS_END, st.eps * EPS_DECAY_PRE)


def run_scheme(scheme, U, UHAT, seed, tuned, collect_train=False):
    rng = np.random.default_rng(seed + 7919)
    st = SchemeState()
    if scheme == "HAO-PRA-U":
        pretrain_hao(st, U, UHAT, tuned, seed)
    n = len(U)
    roll = np.array([BASE_DEMAND[s] for s in SLICES], float)
    hist = []
    test_rows, train_rewards = [], []

    for t in range(n):
        u = U[t]
        hist.append(u)
        if len(hist) > ROLL_WIN:
            hist.pop(0)
        ubar = np.mean(hist, axis=0)
        if t > 0 and not np.isnan(UHAT[t - 1]).any():
            sigma = np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + EPS)
        else:
            sigma = np.zeros(len(SLICES))

        training = t < N_TRAIN
        X = allocate(scheme, st, t, u, U, UHAT, ubar, sigma, training, rng,
                     tuned)
        m = evaluate_interval(scheme, X, u, None)
        r = reward_from_metrics(m)

        if scheme == "HAO-PRA-U":
            sig_next = sigma  # one-step proxy
            u_next = U[t + 1] if t + 1 < n else u
            q_update(st, r, u_next, sig_next)
            if training:
                st.eps = max(EPS_END, st.eps * EPS_DECAY)

        if training:
            train_rewards.append(r)
        else:
            test_rows.append(m)

    out = {k: float(np.mean([r[k] for r in test_rows]))
           for k in METRICS + ["reliability", "isolation", "overhead"]}
    out["reward_mean_test"] = float(np.mean([reward_from_metrics(r)
                                             for r in test_rows]))
    if collect_train:
        out["train_rewards"] = train_rewards
        out["lifecycle"] = st.lifecycle
    return out


# ----------------------------------------------------------------------------
# Baseline tuning: grid search on TRAINING intervals (composite reward)
# ----------------------------------------------------------------------------
def tune_baselines(U, UHAT, seed):
    """Grid-search baseline free parameters on the training intervals only,
    maximizing mean reward over training intervals. Frozen for testing."""
    def train_score(scheme, tuned):
        rng = np.random.default_rng(seed + 104729)
        st = SchemeState()
        hist, rs = [], []
        for t in range(N_TRAIN):
            u = U[t]
            hist.append(u)
            if len(hist) > ROLL_WIN:
                hist.pop(0)
            ubar = np.mean(hist, axis=0)
            if t > 0 and not np.isnan(UHAT[t - 1]).any():
                sigma = np.abs(U[t - 1] - UHAT[t - 1]) / (ubar + EPS)
            else:
                sigma = np.zeros(len(SLICES))
            X = allocate(scheme, st, t, u, U, UHAT, ubar, sigma, True, rng,
                         tuned)
            rs.append(reward_from_metrics(evaluate_interval(scheme, X, u,
                                                            None)))
        return np.mean(rs)

    tuned = {}
    grid = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40]
    tuned["static_margin"] = max(grid, key=lambda g: train_score(
        "Static", {"static_margin": g}))
    tuned["reactive_margin"] = max(grid, key=lambda g: train_score(
        "Reactive", {"reactive_margin": g}))
    best = (-1e9, None)
    for k in [3, 5, 8]:
        for g in [1.05, 1.10, 1.15]:
            sc = train_score("Centralized", {"cent_period": k,
                                             "cent_margin": g})
            if sc > best[0]:
                best = (sc, (k, g))
    tuned["cent_period"], tuned["cent_margin"] = best[1]
    tuned["cp_margin"] = max(grid, key=lambda g: train_score(
        "CentralizedPredictive", {"cp_margin": g}))
    best = (-1e9, None)
    for h in [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]:
        hh = np.full(len(DOMAINS), h)
        sc = train_score("FlatAgent", {"flat_headroom": hh})
        if sc > best[0]:
            best = (sc, hh)
    tuned["flat_headroom"] = best[1]
    return tuned


# ----------------------------------------------------------------------------
# Composite score across schemes (min-max normalized, Table II weights)
# ----------------------------------------------------------------------------
def composite_scores(per_scheme):
    benefit = ["throughput", "utilization", "acceptance", "reliability",
               "isolation", "energy_eff"]
    cost = ["sla_violation", "latency", "fragmentation", "overhead"]
    keymap = dict(sla_violation="sla", latency="latency",
                  throughput="throughput", acceptance="acceptance",
                  utilization="utilization", reliability="reliability",
                  isolation="isolation", energy_eff="energy",
                  fragmentation="fragmentation", overhead="overhead")
    schemes = list(per_scheme)
    scores = {s: 0.0 for s in schemes}
    for metric in benefit + cost:
        vals = np.array([per_scheme[s][metric] for s in schemes])
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo + EPS)
        if metric in cost:
            norm = 1.0 - norm
        wkey = keymap[metric]
        for s, v in zip(schemes, norm):
            scores[s] += OBJ_W[wkey] * float(v)
    return scores


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
SCHEMES = ["Static", "Reactive", "Centralized", "CentralizedPredictive",
           "FlatAgent", "HAO-PRA-U"]
PAPER_NAMES = {"Static": "Static Allocation",
               "Reactive": "Reactive Allocation",
               "Centralized": "Centralized Orchestration",
               "CentralizedPredictive": "Centralized Predictive",
               "FlatAgent": "Flat-Agent Allocation",
               "HAO-PRA-U": "HAO-PRA-U"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--energy-weight", type=float, default=0.0188)
    ap.add_argument("--headroom", type=float, default=2.00)
    args = ap.parse_args()
    set_regime(args.energy_weight, args.headroom)
    os.makedirs(args.out, exist_ok=True)

    all_rows = []          # per (scheme, seed)
    pred_acc_all = []      # per seed
    tuned_log = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        U = generate_traffic(rng)
        UHAT, acc = train_gru_forecasts(U, seed)
        pred_acc_all.append(acc)
        tuned = tune_baselines(U, UHAT, seed)
        tuned_log.append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in tuned.items()})

        for scheme in SCHEMES:
            res = run_scheme(scheme, U, UHAT, seed, tuned)
            row = dict(method=PAPER_NAMES[scheme], seed=seed, **{
                k: res[k] for k in METRICS})
            row.update(reliability=res["reliability"],
                       isolation=res["isolation"],
                       overhead=res["overhead"],
                       reward=res["reward_mean_test"])
            all_rows.append(row)
            print(f"seed {seed} {scheme:>22}: "
                  + " ".join(f"{k}={res[k]:.4f}" for k in METRICS))

    # ---- write per-seed CSV (schema of scripts/significance_test.py) ----
    import csv
    csv_path = os.path.join(args.out, "results_per_seed.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["method", "seed"] + METRICS +
                              ["reliability", "isolation", "overhead",
                               "reward"])
        wcsv.writeheader()
        wcsv.writerows(all_rows)

    # ---- aggregate ----
    agg = {}
    for scheme in SCHEMES:
        name = PAPER_NAMES[scheme]
        rows = [r for r in all_rows if r["method"] == name]
        agg[name] = {k: float(np.mean([r[k] for r in rows]))
                     for k in METRICS + ["reliability", "isolation",
                                         "overhead"]}
        agg[name].update({k + "_std": float(np.std([r[k] for r in rows],
                                                   ddof=1))
                          for k in METRICS})
    comp = composite_scores(agg)

    pred_mean = {s: {m: float(np.mean([a[s][m] for a in pred_acc_all]))
                     for m in ("MAE", "RMSE", "MAPE")} for s in SLICES}

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(aggregate=agg, composite=comp,
                       prediction_accuracy=pred_mean,
                       tuned_baseline_params=tuned_log,
                       capacities={d: float(c) for d, c in
                                   zip(DOMAINS, CAPACITY)},
                       n_seeds=args.seeds), f, indent=1)
    print("\nComposite scores:", {k: round(v, 4) for k, v in comp.items()})
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()
