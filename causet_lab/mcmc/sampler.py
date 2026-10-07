"""Metropolis MCMC over 2D causal set orders.

State: a pair of permutations (u, v) (see orders.py). Move: pick u or v
at random, swap two random entries (a transposition). Accept with
probability min(1, exp(-beta * dS)), the standard Metropolis rule for
the Boltzmann weight exp(-beta * S). At beta = 0 every move is accepted
unconditionally, so the chain is an unweighted random walk on
transpositions of two independent permutations -- which samples (u, v)
uniformly, i.e. uniform random 2D orders (see orders.random_2d_order).

This module recomputes the action from scratch (via orders_to_matrix +
bd_action_2d) on every proposed move -- simple and a reasonable
reference implementation, but dominated by numpy per-call overhead
rather than the actual O(N^3) FLOP count, which for N <= 80 is
microseconds either way. fast_core.py / run_chain_fast() below replaces
this with a Numba-compiled, incrementally-updated inner loop, which is
substantially faster (see mcmc_2d_report.md for measured numbers); this
module is kept as the correctness reference the fast path was validated
against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .orders import orders_to_matrix
from .action import bd_action_2d, f2_smear_table
from . import fast_core


@dataclass
class ChainResult:
    N: int
    beta: float
    seed: int
    n_sweeps: int
    burn_in: int
    measure_every: int
    acceptance_rate: float
    actions: np.ndarray
    tau_int: float
    tau_converged: bool
    effective_samples: float
    final_u: np.ndarray
    final_v: np.ndarray
    final_C: np.ndarray = field(repr=False)


def integrated_autocorr_time(x: np.ndarray, c: float = 5.0):
    """Integrated autocorrelation time tau_int via Sokal's automatic
    windowing method (A.D. Sokal, "Monte Carlo Methods in Statistical
    Mechanics: Foundations and New Algorithms", 1997 Cargese lecture
    notes). tau_int measures how many measurements are needed, on
    average, to get one statistically independent sample; the number of
    *effective* independent samples in a run of n measurements is
    roughly n / (2 * tau_int).

    Returns (tau_int, converged): converged is False if the window
    condition (M >= c * tau) was never satisfied before the series ran
    out, meaning tau_int should be read as a (possibly severe)
    underestimate -- the run was too short to pin it down.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return float("nan"), False
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return 1.0, True
    acf_full = np.correlate(x, x, mode="full")
    acf = acf_full[n - 1:] / (var * n)
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1:m + 1].sum()
        if tau <= 0:
            tau = 1.0
        if m >= c * tau:
            return float(tau), True
    return float(max(tau, 1.0)), False


def run_chain(
    N: int,
    beta: float,
    seed: int,
    n_sweeps: int,
    burn_in: int,
    measure_every: int = 5,
    u0: np.ndarray = None,
    v0: np.ndarray = None,
    anneal_from: float = None,
) -> ChainResult:
    """Run one Metropolis chain. One sweep = N proposed single-swap moves
    (so each element is touched ~twice per sweep on average, once per
    permutation). Measurements of the action are recorded every
    measure_every sweeps after the first burn_in sweeps are discarded.

    If anneal_from is given, beta is ramped linearly from anneal_from up
    to the target beta over the burn_in sweeps (then held fixed at beta
    for all measurement sweeps). This is only used for the deep
    high-beta characterization in study.py: a cold start at large fixed
    beta has near-zero acceptance and just freezes near its (uniformly
    random) initial condition rather than reaching equilibrium, so a
    slow anneal from beta=0 is used instead to get a genuine low-
    "temperature" sample. Production-grid points elsewhere do not use
    this -- they sample at fixed beta from a cold start throughout, so
    that the "spacetime-like" and "layered" phases are compared on an
    equal methodological footing.
    """
    rng = np.random.default_rng(seed)
    u = rng.permutation(N) if u0 is None else np.array(u0, copy=True)
    v = rng.permutation(N) if v0 is None else np.array(v0, copy=True)
    C = orders_to_matrix(u, v, natural_label=False)
    S, _Nk = bd_action_2d(C)

    n_accept = 0
    n_propose = 0
    actions = []

    for sweep in range(n_sweeps):
        if anneal_from is not None and sweep < burn_in:
            frac = sweep / max(burn_in - 1, 1)
            beta_eff = anneal_from + (beta - anneal_from) * frac
        else:
            beta_eff = beta
        for _ in range(N):
            n_propose += 1
            which = rng.integers(0, 2)
            i, j = rng.choice(N, size=2, replace=False)
            if which == 0:
                u[i], u[j] = u[j], u[i]
            else:
                v[i], v[j] = v[j], v[i]
            C_new = orders_to_matrix(u, v, natural_label=False)
            S_new, _Nk_new = bd_action_2d(C_new)
            dS = S_new - S
            accept = dS <= 0 or rng.random() < np.exp(-beta_eff * dS)
            if accept:
                C, S = C_new, S_new
                n_accept += 1
            else:
                if which == 0:
                    u[i], u[j] = u[j], u[i]
                else:
                    v[i], v[j] = v[j], v[i]
        if sweep >= burn_in and (sweep - burn_in) % measure_every == 0:
            actions.append(S)

    actions_arr = np.array(actions, dtype=float)
    tau, converged = integrated_autocorr_time(actions_arr)
    eff_n = len(actions_arr) / (2.0 * tau) if np.isfinite(tau) and tau > 0 else float(len(actions_arr))
    final_C = orders_to_matrix(u, v, natural_label=True)

    return ChainResult(
        N=N, beta=beta, seed=seed, n_sweeps=n_sweeps, burn_in=burn_in,
        measure_every=measure_every,
        acceptance_rate=n_accept / n_propose if n_propose else float("nan"),
        actions=actions_arr, tau_int=tau, tau_converged=converged,
        effective_samples=eff_n, final_u=u, final_v=v, final_C=final_C,
    )


def run_chain_adaptive(
    N: int,
    beta: float,
    seed: int,
    base_sweeps: int,
    base_burnin: int,
    measure_every: int = 5,
    min_eff_samples: float = 50.0,
    max_multiplier: int = 8,
    u0: np.ndarray = None,
    v0: np.ndarray = None,
    anneal_from: float = None,
    verbose_label: str = None,
):
    """Run run_chain with geometrically increasing length (doubling
    base_sweeps/base_burnin together, restarting from scratch each time)
    until the number of *effective* (autocorrelation-corrected) samples
    reaches min_eff_samples, or max_multiplier is hit.

    Returns (ChainResult, met_threshold, multiplier_used). met_threshold
    is False if the multiplier cap was hit before reaching
    min_eff_samples -- callers should flag this rather than silently
    trusting an under-sampled point. If verbose_label is given, prints one
    flushed progress line per attempt (so a slow multi-minute escalation
    to a large multiplier is visible rather than silent).
    """
    import time as _time
    t0 = _time.time()
    multiplier = 1
    result = None
    while True:
        result = run_chain(
            N, beta, seed=seed, n_sweeps=base_sweeps * multiplier,
            burn_in=base_burnin * multiplier, measure_every=measure_every,
            u0=u0, v0=v0, anneal_from=anneal_from,
        )
        met = np.isfinite(result.effective_samples) and result.effective_samples >= min_eff_samples
        if verbose_label is not None:
            print(
                f"[{verbose_label}] seed={seed} multiplier=x{multiplier} "
                f"sweeps={base_sweeps * multiplier} tau={result.tau_int:.1f} "
                f"eff_samples={result.effective_samples:.1f} (target {min_eff_samples:.0f}) "
                f"{'OK' if met else ('capped' if multiplier >= max_multiplier else 'extending...')} "
                f"({_time.time() - t0:.1f}s so far)",
                flush=True,
            )
        if met or multiplier >= max_multiplier:
            return result, met, multiplier
        multiplier *= 2


def layered_start(N: int, seed: int, n_layers: int = 2):
    """Hand-built maximally-layered ("crystalline") 2D order: N elements
    split into n_layers antichains of nearly equal size, with every
    element in an earlier layer preceding every element in a later layer
    (a complete multipartite order between layers). Used only as an
    alternative starting condition for the hysteresis check in study.py:
    if a chain started here settles to the same equilibrium as a chain
    started from a uniformly random order, the run length is adequate
    and there is no metastability; if not, that is itself the finding.

    Construction: within layer L (elements get a contiguous range of
    u-values and v-values reserved for that layer, disjoint across
    layers and increasing with L, so cross-layer relations are forced).
    Within a layer, u is increasing while v is *reversed*, so no two
    elements in the same layer are related (u[i]<u[j] never implies
    v[i]<v[j] within a layer) -- the same trick generators.
    kleitman_rothschild uses to build independent antichain layers.
    """
    rng = np.random.default_rng(seed)
    sizes = [N // n_layers] * n_layers
    for i in range(N - sum(sizes)):
        sizes[i] += 1
    u = np.empty(N, dtype=np.int64)
    v = np.empty(N, dtype=np.int64)
    offset = 0
    for size in sizes:
        idx = np.arange(offset, offset + size)
        perm = rng.permutation(size)
        u[idx] = offset + perm
        v[idx] = offset + (size - 1 - perm)
        offset += size
    return u, v


def run_chain_fast(
    N: int,
    beta: float,
    seed: int,
    n_sweeps: int,
    burn_in: int,
    measure_every: int = 5,
    eps: float = 0.21,
    u0: np.ndarray = None,
    v0: np.ndarray = None,
    anneal_from: float = None,
) -> ChainResult:
    """Same interface and semantics as run_chain, but using the
    Numba-compiled incremental core in fast_core.py instead of rebuilding
    the whole relation matrix from scratch every move, and the *smeared*
    2D BD action (action.bd_action_2d_smeared) with non-locality
    parameter eps rather than the plain local action -- see action.py's
    module docstring for why. Validated against a full-recompute
    reference on many random moves -- see
    tests/test_mcmc.py::test_fast_core_incremental_matches_full_recompute.

    Note: the random moves themselves are drawn from Numba's internal RNG
    (seeded once per call via np.random.seed(seed) inside the jitted
    loop), not numpy's default_rng -- so run_chain_fast(seed=k) does NOT
    reproduce the same move sequence as run_chain(seed=k), only its own
    sequence reproducibly across repeated calls with the same seed.
    """
    rng = np.random.default_rng(seed)
    u = rng.permutation(N).astype(np.int64) if u0 is None else np.array(u0, dtype=np.int64, copy=True)
    v = rng.permutation(N).astype(np.int64) if v0 is None else np.array(v0, dtype=np.int64, copy=True)
    C, sizes, counts = fast_core.build_state(u, v)
    f2_table = f2_smear_table(max(N - 2, 0), eps)

    has_anneal = anneal_from is not None
    anneal_val = float(anneal_from) if has_anneal else 0.0
    max_measurements = int(n_sweeps // max(measure_every, 1) + 2)

    actions, n_accept, n_propose = fast_core.run_sweeps_jit(
        u, v, C, sizes, counts, N, float(beta), float(eps), f2_table, int(n_sweeps), int(burn_in),
        int(measure_every), anneal_val, has_anneal, int(seed), max_measurements,
    )

    tau, converged = integrated_autocorr_time(actions)
    eff_n = len(actions) / (2.0 * tau) if np.isfinite(tau) and tau > 0 else float(len(actions))
    final_C = orders_to_matrix(u, v, natural_label=True)

    return ChainResult(
        N=N, beta=beta, seed=seed, n_sweeps=n_sweeps, burn_in=burn_in,
        measure_every=measure_every,
        acceptance_rate=n_accept / n_propose if n_propose else float("nan"),
        actions=np.asarray(actions, dtype=float), tau_int=tau, tau_converged=converged,
        effective_samples=eff_n, final_u=u, final_v=v, final_C=final_C,
    )


def run_chain_adaptive_fast(
    N: int,
    beta: float,
    seed: int,
    base_sweeps: int,
    base_burnin: int,
    measure_every: int = 5,
    eps: float = 0.21,
    min_eff_samples: float = 50.0,
    max_multiplier: int = 8,
    u0: np.ndarray = None,
    v0: np.ndarray = None,
    anneal_from: float = None,
    verbose_label: str = None,
):
    """run_chain_adaptive, but using run_chain_fast as the underlying
    engine. See run_chain_adaptive for the extension logic.
    """
    import time as _time
    t0 = _time.time()
    multiplier = 1
    result = None
    while True:
        result = run_chain_fast(
            N, beta, seed=seed, n_sweeps=base_sweeps * multiplier,
            burn_in=base_burnin * multiplier, measure_every=measure_every, eps=eps,
            u0=u0, v0=v0, anneal_from=anneal_from,
        )
        met = np.isfinite(result.effective_samples) and result.effective_samples >= min_eff_samples
        if verbose_label is not None:
            print(
                f"[{verbose_label}] seed={seed} multiplier=x{multiplier} "
                f"sweeps={base_sweeps * multiplier} tau={result.tau_int:.1f} "
                f"eff_samples={result.effective_samples:.1f} (target {min_eff_samples:.0f}) "
                f"{'OK' if met else ('capped' if multiplier >= max_multiplier else 'extending...')} "
                f"({_time.time() - t0:.1f}s so far)",
                flush=True,
            )
        if met or multiplier >= max_multiplier:
            return result, met, multiplier
        multiplier *= 2
