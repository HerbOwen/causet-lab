"""Control tests for the Phase 3 MCMC subpackage: 2D orders, the
Benincasa-Dowker action, and the Metropolis sampler.
"""
from __future__ import annotations

import numpy as np
import pytest

from causet_lab.causet import CausalSet
from causet_lab.generators import sprinkle
from causet_lab.measures import myrheim_meyer_dimension
from causet_lab.mcmc.orders import random_2d_order, orders_to_matrix
from causet_lab.mcmc.action import (
    bd_action_2d, bd_action_2d_smeared, f2_smear_table, beta_c_glaser_2018,
)
from causet_lab.mcmc.sampler import run_chain, run_chain_fast, integrated_autocorr_time
from causet_lab.mcmc.fast_core import build_state, apply_swap_py, apply_swap_jit, action_from_counts


def test_2d_order_is_acyclic_and_closed():
    for seed in range(5):
        _u, _v, C = random_2d_order(60, seed=seed)
        cset = CausalSet(C)
        assert cset.is_acyclic()
        assert cset.is_transitively_closed()


def test_action_is_invariant_to_labeling():
    """The BD action depends only on the causal structure, not on how
    elements happen to be labeled, so computing it on the naturally
    labeled matrix or the raw (unsorted) one must agree exactly."""
    u, v, _C = random_2d_order(50, seed=1)
    C_sorted = orders_to_matrix(u, v, natural_label=True)
    C_unsorted = orders_to_matrix(u, v, natural_label=False)
    S1, Nk1 = bd_action_2d(C_sorted)
    S2, Nk2 = bd_action_2d(C_unsorted)
    assert S1 == S2
    assert np.array_equal(Nk1, Nk2)


def test_action_on_2d_sprinkles_is_small_relative_to_N():
    """Flat-space check (Benincasa & Dowker 2010; see mcmc/action.py):
    the BD action's expectation value over sprinklings into flat
    Minkowski space is close to zero (it is *not* an extensive quantity
    that grows with N -- individual samples fluctuate by O(sqrt(N)) or
    more, but the mean should stay small compared to N itself).
    """
    N = 150
    values = [bd_action_2d(sprinkle(N, 2, seed=s)[0].C)[0] for s in range(30)]
    mean_S = float(np.mean(values))
    assert abs(mean_S) / N < 0.5


def test_smeared_action_at_eps_one_matches_plain_action_limit():
    """bd_action_2d_smeared(C, eps=1.0) should reduce exactly to
    4*(N - 2*N0 + 4*N1 - 2*N2) -- see action.py's module docstring for
    the algebra (the f2_smear_table kernel is written in a division-free
    form specifically so this limit is exact, not just approximate)."""
    for seed in range(5):
        for N in (5, 8, 15, 25):
            u, v, C = random_2d_order(N, seed=seed)
            S_smeared, _Nk = bd_action_2d_smeared(C, eps=1.0)
            _S_plain, Nk = bd_action_2d(C)
            target = 4.0 * (N - 2 * Nk[0] + 4 * Nk[1] - 2 * Nk[2])
            assert abs(S_smeared - target) < 1e-9


def test_beta_c_glaser_2018_matches_published_values():
    """Sanity-check the published fit coefficients (Glaser, O'Connor,
    Surya 2018, arXiv:1706.06432): b(eps) = 1.66/eps^2 should dominate at
    these N, and the formula should give the same order of magnitude
    beta_c reported in that paper for N ~ 50, eps ~ 0.2-0.5 (beta_c of a
    few tenths to order 1, not 0.01 or 100)."""
    for eps in (0.21, 0.5):
        bc = beta_c_glaser_2018(50, eps)
        assert 0.01 < bc < 10.0


def test_beta_zero_chain_accepts_every_move():
    """At beta=0, exp(-beta*dS) == 1 regardless of dS, so every proposed
    move must be accepted."""
    res = run_chain(N=20, beta=0.0, seed=0, n_sweeps=20, burn_in=5, measure_every=2)
    assert res.acceptance_rate == 1.0


def test_beta_zero_chain_recovers_dimension_two():
    """Uniform random 2D orders are statistically equivalent to 2D
    sprinkles (Surya 2012); the Myrheim-Meyer dimension of the final
    sample from a beta=0 chain should land near 2."""
    estimates = []
    for seed in range(5):
        res = run_chain(N=200, beta=0.0, seed=seed, n_sweeps=30, burn_in=5, measure_every=5)
        estimates.append(myrheim_meyer_dimension(res.final_C))
    assert abs(np.mean(estimates) - 2.0) < 0.4


def test_integrated_autocorr_time_of_iid_noise_is_near_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    tau, converged = integrated_autocorr_time(x)
    assert converged
    assert abs(tau - 1.0) < 0.5


def test_run_chain_reproducible_with_same_seed():
    r1 = run_chain(N=20, beta=0.5, seed=7, n_sweeps=15, burn_in=5, measure_every=2)
    r2 = run_chain(N=20, beta=0.5, seed=7, n_sweeps=15, burn_in=5, measure_every=2)
    assert np.array_equal(r1.final_u, r2.final_u)
    assert np.array_equal(r1.final_v, r2.final_v)
    assert np.array_equal(r1.actions, r2.actions)


@pytest.mark.parametrize("engine", ["py", "jit"])
def test_fast_core_incremental_matches_full_recompute(engine):
    """fast_core.py keeps a running (C, sizes, counts) state and updates
    it incrementally on every swap instead of rebuilding from scratch.
    This is the core correctness requirement for that approach: after
    every move (accepted or not -- a rejected move is undone by applying
    the same swap again, since a transposition is its own inverse), the
    incrementally-updated state must exactly match a full O(N^3)
    recomputation from the current (u, v), for both the pure-Python and
    the Numba-jitted version of the update.
    """
    apply_swap = apply_swap_py if engine == "py" else apply_swap_jit
    rng = np.random.default_rng(1 if engine == "py" else 2)
    for N in (5, 9, 16, 25):
        u = rng.permutation(N).astype(np.int64)
        v = rng.permutation(N).astype(np.int64)
        C, sizes, counts = build_state(u, v)
        for _step in range(150):
            which = int(rng.integers(0, 2))
            i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
            apply_swap(u, v, C, sizes, counts, N, which, i, j)
            C_ref, sizes_ref, counts_ref = build_state(u, v)
            assert np.array_equal(C, C_ref), f"N={N} engine={engine}: C mismatch"
            assert np.array_equal(counts, counts_ref), f"N={N} engine={engine}: counts mismatch"
            assert np.array_equal(sizes[C], sizes_ref[C]), f"N={N} engine={engine}: sizes mismatch on related pairs"
        # action_from_counts (smeared, eps=0.21) must also agree with the
        # full-recompute reference action on the final state
        f2_table = f2_smear_table(max(N - 2, 0), 0.21)
        S_incremental = action_from_counts(N, counts, f2_table, 0.21)
        S_reference, Nk_reference = bd_action_2d_smeared(C, 0.21)
        assert np.array_equal(counts, Nk_reference)
        assert abs(S_incremental - S_reference) < 1e-9


def test_fast_core_rejected_move_is_exactly_undone():
    """A rejected proposal is undone by reapplying the same swap (a
    transposition is its own inverse) rather than via a separate revert
    path -- confirm that round-trips the state exactly, not just
    approximately."""
    rng = np.random.default_rng(3)
    N = 20
    u = rng.permutation(N).astype(np.int64)
    v = rng.permutation(N).astype(np.int64)
    C, sizes, counts = build_state(u, v)
    u0, v0, C0, sizes0, counts0 = u.copy(), v.copy(), C.copy(), sizes.copy(), counts.copy()
    for _ in range(20):
        which = int(rng.integers(0, 2))
        i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
        apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)
        apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)  # undo
        assert np.array_equal(u, u0) and np.array_equal(v, v0)
        assert np.array_equal(C, C0)
        assert np.array_equal(counts, counts0)


def test_run_chain_fast_reproducible_with_same_seed():
    r1 = run_chain_fast(N=20, beta=0.5, seed=7, n_sweeps=15, burn_in=5, measure_every=2)
    r2 = run_chain_fast(N=20, beta=0.5, seed=7, n_sweeps=15, burn_in=5, measure_every=2)
    assert np.array_equal(r1.final_u, r2.final_u)
    assert np.array_equal(r1.actions, r2.actions)


def test_run_chain_fast_beta_zero_recovers_dimension_two():
    estimates = []
    for seed in range(5):
        res = run_chain_fast(N=200, beta=0.0, seed=seed, n_sweeps=30, burn_in=5, measure_every=5)
        estimates.append(myrheim_meyer_dimension(res.final_C))
    assert abs(np.mean(estimates) - 2.0) < 0.4


def test_run_chain_fast_final_C_is_acyclic_and_closed():
    res = run_chain_fast(N=40, beta=0.04, seed=0, n_sweeps=50, burn_in=10, measure_every=5)
    cset = CausalSet(res.final_C)
    assert cset.is_acyclic()
    assert cset.is_transitively_closed()
