#!/usr/bin/env python3
"""
precursor_traffic.py  --  Burst-precursor traffic generator (Experiment 3).

PURPOSE
-------
This module is a DROP-IN companion to hao_pra_u_sim.py. It provides an
alternative traffic family in which bursts are *anticipable*: each burst is
preceded by an observable leading-indicator signal whose informativeness is
controlled by a single scalar knob, `predictability` in [0, 1].

This generator removes that objection by
letting prediction *in principle* pay, and asking whether HAO-PRA-U then
beats per-regime-tuned baselines. It does NOT presume the answer.

It imports the frozen simulator and reuses its constants so the two traffic
families share identical structure except for the burst mechanism.

--------------------------------------------------------------------------
WHY A STATE-DEPENDENT HAZARD WITH AN OBSERVABLE LEADING INDICATOR
--------------------------------------------------------------------------
Three candidate precursor mechanisms were considered:

(a) Autocorrelated demand (e.g. AR/GARCH burst envelope). REJECTED as the
    primary mechanism: it makes the burst itself partly predictable from
    *demand history alone*, which the baselines also observe. That muddies
    the fairness-of-observation gate -- the very thing under test -- because
    the "precursor" is no longer separable from the signal the baselines are
    entitled to see. We cannot then attribute any forecaster advantage to
    the precursor channel specifically.

(b) Pure exogenous leading-indicator channel (a side channel that lights up
    k steps before a burst). Anticipable and cleanly separable, but
    physically unmotivated: a free-floating oracle-ish flag invites the
    symmetric objection ("now you engineered an environment where prediction
    trivially pays").

(c) State-dependent hazard with a noisy observation of the latent state
    (CHOSEN). A latent congestion-pressure state x(t) evolves with slow
    autocorrelation; the per-interval burst hazard is a rising function of
    x(t); and the precursor is a NOISY observation of x(t) emitted at (and
    shortly before) the interval. The knob `predictability` is the
    signal-to-noise ratio of that observation. At predictability=0 the
    observation is pure noise, the hazard reduces to a constant, and the
    burst process collapses to the original memoryless Bernoulli law
    (recovering the old result by construction). At predictability=1 the
    precursor is a clean readout of the latent pressure and the burst is
    strongly anticipable.

Mechanism (c) is the most defensible for 6G because real precursors of
traffic bursts are exactly noisy observations of a slowly-varying latent
load state, emitted on a *separate control plane* before the data-plane
burst materializes:
  * RRC connection-request / RACH preamble surges before an eMBB session
    flood;
  * Buffer-status-report (BSR) ramps before an uplink burst;
  * Beam-failure / radio-link-failure indications before reattachment storms;
  * Handover and paging signaling ahead of mobility-driven demand;
  * PDCP/RLC queue-occupancy telemetry as an edge congestion leading edge.
These are observable to an orchestrator's monitoring plane but are NOT the
served demand itself, so routing them to the forecaster while withholding
them from demand-only baselines is both physically faithful and exactly the
asymmetry the experiment is designed to test.

--------------------------------------------------------------------------
FAIRNESS-OF-OBSERVATION  (READ THIS)
--------------------------------------------------------------------------
The precursor is returned as a SEPARATE array `P` aligned to `U`. The
intended wiring (enforced in run_experiment3.py, not here) is:

    * The GRU forecaster MAY consume P (it is a monitoring-plane signal).
    * Every baseline (Static, Reactive, Centralized, FlatAgent) and the
      shared metric/`ubar`/`U[t-1]` paths MUST NOT consume P. They see only
      realized demand history, exactly as in the frozen simulator.

This module does not by itself give any scheme access to P; it only
produces it. The asymmetry is created downstream and is logged. If P ever
leaks into a baseline's observation, the comparison is invalid -- the
harness asserts against this.

The precursor is constructed to be anticipable by a CAUSAL one-step-ahead
forecaster: the observation o(t) carries information about whether interval
t (and, with lead L>0, intervals t..t+L-1) will burst, and o is available
to the forecaster within its lookback window up to and including the
current decision interval. Prediction is therefore POSSIBLE but never
guaranteed: hazard, observation noise, and the GRU's own capacity all stand
between the precursor and a correct anticipation. We log realized
forecast-skill so the run can confirm prediction actually paid information,
rather than assuming it.

--------------------------------------------------------------------------
RNG-ORDER CONTRACT
--------------------------------------------------------------------------
The frozen `generate_traffic` consumes draws per slice in this exact order:
    uniform()  [phase]
    normal(N)  [stochastic noise]
    random(N)  [burst Bernoulli]
    uniform(N) [burst magnitude]
shift_driver.recover_phases() replays this order to recover phases without
touching the frozen code. THIS GENERATOR IS A DIFFERENT TRAFFIC FAMILY and
deliberately does NOT promise bit-identical streams with the memoryless
generator at predictability>0 (the burst law differs). At predictability=0
it reproduces the memoryless burst *law* (same marginal Bernoulli rate and
magnitude distribution) but NOT the same RNG draw sequence, because the
hazard path advances the generator differently. Experiment 3 is a NEW
pre-registered study with its own seeds; it does not feed the Experiment-2
shift driver, so the recover_phases contract is not relevant here. This is
called out so nobody wires this module into shift_driver.py expecting
bit-identical replay. (If memoryless-law reproduction with bit-identical
streams is ever needed, call the frozen generate_traffic directly.)
"""

import numpy as np

# Import frozen constants from the simulator. We do not modify it.
import importlib.util
import os

_SIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "hao_pra_u_sim.py")


def _load_sim():
    spec = importlib.util.spec_from_file_location("hao_pra_u_sim", _SIM_PATH)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    return sim


_sim = _load_sim()
SLICES = _sim.SLICES
BASE_DEMAND = _sim.BASE_DEMAND
N_INTERVALS = _sim.N_INTERVALS
PERIODIC_AMPLITUDE = _sim.PERIODIC_AMPLITUDE
PERIODIC_PERIOD = _sim.PERIODIC_PERIOD
STOCH_STD_FRAC = _sim.STOCH_STD_FRAC
BURST_PROB = _sim.BURST_PROB                 # 0.05 marginal target
BURST_MAG_RANGE = _sim.BURST_MAG_RANGE       # (0.30, 0.60)


# --------------------------------------------------------------------------
# Latent-state / hazard parameters (frozen for Experiment 3; see prereg).
# --------------------------------------------------------------------------
# Latent congestion-pressure state x(t): AR(1), zero-mean, unit-ish scale.
# rho_x sets how slowly pressure drifts (autocorrelation of the *cause*, not
# of demand). The precursor observes x; demand does NOT depend on x except
# through the burst hazard, so baselines gain nothing from demand history
# about x beyond what bursts already reveal.
LATENT_RHO = 0.85            # AR(1) coefficient of latent pressure
LATENT_STD = 1.00            # innovation scale of latent pressure

# Hazard: P(burst at t | x) = clip( base_rate * exp(GAIN * x(t)) , 0, HMAX ),
# calibrated so that the *marginal* burst rate equals BURST_PROB regardless
# of `predictability`. This is the key invariant: changing predictability
# must not change how often bursts happen, only how anticipable they are.
HAZARD_GAIN = 3.00           # sensitivity of hazard to latent pressure
HAZARD_MAX = 0.95            # cap on per-interval burst probability
# Lead time L: number of intervals by which the precursor leads the burst.
# L=0  -> precursor co-occurs with burst (still usable: observation at the
#         decision interval informs the one-step forecast of the same
#         interval's level, since o(t) is in the forecaster's window).
# L>=1 -> precursor light up before the burst; stronger anticipation.
PRECURSOR_LEAD = 1           # default lead (intervals)


def _calibrate_base_rate(x_series, gain, target_marginal, hmax):
    """Pick base_rate so E_t[ clip(base_rate*exp(gain*x),0,hmax) ] == target.
    Solved by bisection on base_rate over the realized latent path so the
    marginal burst frequency is held at `target_marginal` for every
    predictability level. Holding the marginal fixed is what makes the
    predictability knob a *pure* anticipability manipulation."""
    ex = np.exp(gain * x_series)

    def marginal(base_rate):
        return np.mean(np.clip(base_rate * ex, 0.0, hmax))

    lo, hi = 0.0, 1.0
    # expand hi until marginal(hi) >= target (clip ceiling permitting)
    for _ in range(60):
        if marginal(hi) >= target_marginal or hi > 1e6:
            break
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if marginal(mid) < target_marginal:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_precursor_traffic(rng, predictability,
                               lead=PRECURSOR_LEAD,
                               n_intervals=N_INTERVALS,
                               latent_rho=LATENT_RHO,
                               hazard_gain=HAZARD_GAIN,
                               hazard_max=HAZARD_MAX,
                               return_diagnostics=False):
    """Generate burst-precursor traffic.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator (use np.random.default_rng(seed)).
    predictability : float in [0, 1]
        SNR knob for the precursor observation.
          0.0  -> observation is pure noise; hazard reduces to a constant
                  base rate -> burst process is memoryless Bernoulli with
                  marginal == BURST_PROB (recovers the old result's burst
                  law).
          1.0  -> observation is a clean (low-noise) readout of latent
                  pressure -> bursts are strongly anticipable.
        Implemented as observation o(t) = x_eff(t) + noise, where the noise
        scale rises as predictability falls; at predictability=0 the latent
        pressure is also decoupled from the hazard (gain->0) so the marginal
        law is exactly Bernoulli.
    lead : int >= 0
        Intervals by which the precursor leads the burst it predicts.
    return_diagnostics : bool
        If True, also return the latent state and realized hazard for audit.

    Returns
    -------
    U : (n_intervals, n_slices) float array
        Demand matrix, same composition as the frozen generator
        (periodic + Gaussian noise + burst), bursts now hazard-driven.
    P : (n_intervals, n_slices) float array
        Precursor observation o(t) per slice. Monitoring-plane signal.
        FORECASTER-ONLY. Must never enter any baseline observation.
    burst_mask : (n_intervals, n_slices) bool array
        Ground-truth burst occurrences (for forecast-skill diagnostics:
        fraction of bursts anticipated, prediction error). NOT an
        observation given to any scheme; analysis/logging only.
    (optional) diag : dict with 'latent', 'hazard', 'base_rate' per slice.
    """
    if not (0.0 <= predictability <= 1.0):
        raise ValueError("predictability must be in [0, 1]")
    n = n_intervals
    nS = len(SLICES)
    t = np.arange(n)

    U = np.zeros((n, nS))
    P = np.zeros((n, nS))
    burst_mask = np.zeros((n, nS), dtype=bool)
    diag = {"latent": np.zeros((n, nS)), "hazard": np.zeros((n, nS)),
            "base_rate": np.zeros(nS)}

    # `predictability` couples the hazard to the latent state AND sets the
    # observation SNR. Both go to zero together so that predictability=0 is
    # genuinely the memoryless baseline (constant hazard, useless precursor).
    eff_gain = hazard_gain * predictability
    # Observation noise std: large when predictability small, ->0 as ->1.
    # obs_noise_std maps predictability 0->inf-ish, 1->small. We use a
    # bounded inverse-SNR so that at p=0 the observation carries ~0 bits.
    if predictability <= 0.0:
        obs_noise_std = 1e6           # effectively pure noise
    else:
        # SNR grows as predictability/(1-predictability); cap for stability.
        snr = predictability / (1.0 - predictability + 1e-9)
        obs_noise_std = float(np.clip(1.0 / np.sqrt(snr + 1e-9), 1e-3, 1e6))

    for j, s in enumerate(SLICES):
        base = BASE_DEMAND[s]

        # ---- draw order (per slice), documented for auditability ----
        phase = rng.uniform(0, 2 * np.pi)                 # 1: phase
        noise = rng.normal(0, STOCH_STD_FRAC * base, n)   # 2: stochastic
        innov = rng.normal(0, LATENT_STD, n)              # 3: latent innov
        u_haz = rng.random(n)                             # 4: hazard draw
        mag = rng.uniform(*BURST_MAG_RANGE, n)            # 5: burst magnitude
        obs_noise = rng.normal(0, 1.0, n)                 # 6: precursor noise

        # ---- latent congestion-pressure state x(t): AR(1) ----
        x = np.zeros(n)
        for k in range(1, n):
            x[k] = latent_rho * x[k - 1] + innov[k]
        # normalize to unit std so hazard_gain has consistent meaning
        x = x / (x.std() + 1e-9)

        # ---- hazard uses the state SHIFTED by `lead`: the pressure that
        # will cause a burst at t is present (and observable) at t-lead.
        # x_cause[t] drives the burst at t; it equals x[t-lead].
        if lead > 0:
            x_cause = np.concatenate([np.zeros(lead), x[:-lead]])
            # The first `lead` intervals have no causal pressure (warm-up):
            # exclude them from calibration AND from the marginal-rate target
            # so the realized rate over the live region matches BURST_PROB
            # exactly, independent of gain. The warm-up intervals fall in the
            # GRU lookback region and are not test-window bursts.
            calib_region = slice(lead, n)
        else:
            x_cause = x
            calib_region = slice(0, n)

        if eff_gain > 0:
            base_rate = _calibrate_base_rate(eff_gain * x_cause[calib_region],
                                             1.0, BURST_PROB, hazard_max)
            # hazard over live region; warm-up region uses base_rate*exp(0).
            hazard = np.clip(base_rate * np.exp(eff_gain * x_cause),
                             0.0, hazard_max)
        else:
            base_rate = BURST_PROB
            hazard = np.full(n, BURST_PROB)   # memoryless limit

        burst_occurs = u_haz < hazard
        bursts = burst_occurs * mag * base

        # ---- demand: identical composition to frozen generator ----
        periodic = base * (1 + PERIODIC_AMPLITUDE *
                           np.sin(2 * np.pi * t / PERIODIC_PERIOD + phase))
        U[:, j] = np.maximum(periodic + noise + bursts, 1.0)

        # ---- precursor observation o(t): noisy readout of the latent
        # pressure that is observable AT interval t (it leads the burst by
        # `lead`). o(t) = x[t] + obs_noise. Because x_cause[t]=x[t-lead],
        # the value x[t] observed now informs bursts at t+lead. We expose
        # o aligned to the interval at which it is observable.
        # At predictability=0, obs_noise_std is enormous -> o is ~pure noise.
        P[:, j] = x + obs_noise_std * obs_noise

        burst_mask[:, j] = burst_occurs
        diag["latent"][:, j] = x
        diag["hazard"][:, j] = hazard
        diag["base_rate"][j] = base_rate if eff_gain > 0 else BURST_PROB

    if return_diagnostics:
        return U, P, burst_mask, diag
    return U, P, burst_mask


# --------------------------------------------------------------------------
# Self-test: marginal-rate invariance and predictability monotonicity.
# Run:  python3 precursor_traffic.py
# These checks validate the GENERATOR's own properties only. They produce no
# experimental result and touch no scheme.
# --------------------------------------------------------------------------
def _selftest():
    print("precursor_traffic self-test (generator properties only)\n")
    print(f"{'predict.':>9} {'burst_rate':>11} {'corr(o, x)':>11} "
          f"{'corr(o, future_burst)':>22}")
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rng = np.random.default_rng(20260613)
        U, P, bm, diag = generate_precursor_traffic(
            rng, predictability=p, lead=PRECURSOR_LEAD,
            return_diagnostics=True)
        rate = bm.mean()
        # correlation of observation with latent state (recoverability)
        co = np.corrcoef(P.ravel(), diag["latent"].ravel())[0, 1]
        # correlation of observation now with burst `lead` steps ahead
        L = PRECURSOR_LEAD
        if L > 0:
            o_now = P[:-L].ravel()
            b_future = bm[L:].astype(float).ravel()
        else:
            o_now = P.ravel()
            b_future = bm.astype(float).ravel()
        cb = np.corrcoef(o_now, b_future)[0, 1]
        print(f"{p:>9.2f} {rate:>11.4f} {co:>11.4f} {cb:>22.4f}")
    print(f"\nTarget marginal burst rate (BURST_PROB) = {BURST_PROB}")
    print("Expect: burst_rate ~constant across predictability (invariant);")
    print("corr(o,x) and corr(o,future_burst) rise from ~0 at p=0 to high "
          "at p=1.")


if __name__ == "__main__":
    _selftest()
