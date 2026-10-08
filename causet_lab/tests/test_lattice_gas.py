"""Correctness and controls for the Phase 4 lattice-gas causal set
model (Cunningham & Surya, arXiv:1908.11647): the incremental bitset
engine must agree with a full recomputation over many random moves
(mirrors test_bitset_core.py's pattern), and a beta=0 (uniformly
random) filling must look like a 2D causal-set sprinkle.
"""
from __future__ import annotations

import numpy as np
import pytest

from causet_lab.generators import sprinkle
from causet_lab.measures import abundance_distance
from causet_lab.battery import analyze_matrix, nanmean
from causet_lab.mcmc import lattice_gas as lgm
from causet_lab.mcmc import lattice_gas_core as lg

N_MOVES = 100_000


@pytest.mark.parametrize("n", [16, 30, 47])
@pytest.mark.parametrize("eps", [0.1, 0.21])
def test_lattice_gas_incremental_action_matches_full_recompute(n, eps):
    """Over 1e5 random lattice-gas moves, the incrementally updated
    bitset histogram's action must match a full recomputation from L
    at every checked step -- not just the final state."""
    w, h, m = lgm.lattice_dims(n)
    rng = np.random.default_rng(hash((n, eps, "lattice")) & 0xFFFFFFFF)
    L = lgm.random_filling(n, m, seed=int(rng.integers(0, 2**31)))
    future, past, counts = lg.build_lattice_state_bitset(L, n, w)
    from causet_lab.mcmc.action import f2_smear_table
    f2_table = f2_smear_table(max(n - 2, 0), eps)

    check_every = 500
    for step in range(N_MOVES):
        i = int(rng.integers(0, n))
        j = n + int(rng.integers(0, m - n))
        lg.apply_relocate_jit(L, future, past, counts, n, w, i, j)
        if step % check_every == 0 or step == N_MOVES - 1:
            S_incremental = lg.lattice_gas_action_from_counts(n, counts, f2_table, eps)
            S_full, Nk_full = lgm.lattice_gas_action_full(L, n, w, eps)
            assert abs(S_incremental - S_full) < 1e-6, (n, eps, step, S_incremental, S_full)
            # counts/Nk_full must agree bin-for-bin too, not just the
            # scalar action (a compensating pair of errors could
            # otherwise hide behind a correct-looking S).
            assert np.array_equal(counts, Nk_full), (n, eps, step, counts, Nk_full)


def test_lattice_gas_rejected_move_is_exactly_undone():
    n = 30
    w, h, m = lgm.lattice_dims(n)
    rng = np.random.default_rng(5)
    L = lgm.random_filling(n, m, seed=7)
    future, past, counts = lg.build_lattice_state_bitset(L, n, w)
    L0, future0, past0, counts0 = L.copy(), future.copy(), past.copy(), counts.copy()
    for _ in range(50):
        i = int(rng.integers(0, n))
        j = n + int(rng.integers(0, m - n))
        lg.apply_relocate_jit(L, future, past, counts, n, w, i, j)
        lg.apply_relocate_jit(L, future, past, counts, n, w, i, j)
        assert np.array_equal(L, L0)
        assert np.array_equal(future, future0) and np.array_equal(past, past0)
        assert np.array_equal(counts, counts0)


@pytest.mark.parametrize("n", [30, 50])
def test_lattice_gas_beta0_filling_looks_like_2d_sprinkle(n):
    """A uniformly random filling (beta=0) should locally look like a
    2D causal-set sprinkle -- the control instruction #3 asks for.

    The whole lattice-gas filling is NOT itself diamond-shaped (it's a
    cylinder of aspect ratio h/w=4, not an Alexandrov interval), so its
    *whole-matrix* ordering fraction/Myrheim-Meyer dimension are not
    comparable to a sprinkle's (confirmed directly: C&S's own paper
    reports hot-phase ordering fraction ~0.88, not the causal-diamond
    value ~0.5 -- same reason this project already samples intervals
    for grown/junk orders in battery.analyze_matrix rather than using
    the whole matrix). Sampling sub-intervals from within the filling
    makes the comparison fair: an interval is diamond-shaped by
    definition regardless of the embedding's global topology.
    """
    w, h, m = lgm.lattice_dims(n)
    kmax = 6
    mm_dims, profiles = [], []
    for seed in range(5):
        L = lgm.random_filling(n, m, seed=seed)
        C = lgm.lattice_to_matrix(L, n, w)
        res = analyze_matrix(C, seed=seed, n_samples=300, min_size=2, max_size=n, kmax=kmax)
        mm_dims.append(res["mm_dim_sampled_mean"])
        profiles.append(res["abundance_profile"])

    # Reference: intervals sampled from a large sprinkle, same size range.
    ref_cset, _ = sprinkle(3000, 2, seed=0)
    ref = analyze_matrix(ref_cset.C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=kmax)

    mean_dim = nanmean(mm_dims)
    mean_profile = np.nanmean(np.array(profiles), axis=0)
    dist = abundance_distance(mean_profile, ref["abundance_profile"])
    print(f"\nn={n}: MM dimension (sampled intervals)={mean_dim:.3f} (target ~2), "
          f"interval-abundance L1 distance to sprinkle-sampled reference={dist:.3f}")
    assert abs(mean_dim - 2.0) < 0.5, (n, mm_dims)
    assert dist < 0.5, (n, mean_profile, ref["abundance_profile"])
