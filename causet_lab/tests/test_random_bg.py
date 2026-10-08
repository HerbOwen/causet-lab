"""Correctness and controls for the Phase 5 random-background causal
set model: the incremental bitset engine (continuous positions) must
agree with full recomputation over many random moves (mirrors
test_lattice_gas.py's pattern); placing a background at the exact
regular-lattice positions must reproduce the lattice model bit-for-bit
(confirms the x-has-period-w units convention, not radians -- see
random_bg_core.py's module docstring); and a beta=0 (uniformly random)
filling on a genuinely random background must look like a 2D
causal-set sprinkle, at least as well as the regular lattice did in
Phase 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from causet_lab.generators import sprinkle
from causet_lab.measures import abundance_distance
from causet_lab.battery import analyze_matrix, nanmean
from causet_lab.mcmc import random_bg as rbgm
from causet_lab.mcmc import random_bg_core as rbg
from causet_lab.mcmc import lattice_gas as lgm
from causet_lab.mcmc import lattice_gas_core as lg


@pytest.mark.parametrize("n", [16, 30, 47])
def test_random_bg_at_lattice_positions_matches_lattice_exactly(n):
    """Placing the background at the exact integer lattice positions
    (site s -> t=s//w, x=s%w) must reproduce lattice_gas.lattice_to_matrix
    bit-for-bit, including the lightlike (<=) convention -- this is the
    direct check that site_x's units (period w) match the lattice's,
    not radians. A mismatch here would mean the two models are not
    actually comparable at the same causal geometry.
    """
    w, h, m = lgm.lattice_dims(n)
    site_ids = np.arange(m)
    site_t = (site_ids // w).astype(np.float64)
    site_x = (site_ids % w).astype(np.float64)

    rng_np = np.random.default_rng(hash(("lattice_equiv", n)) & 0xFFFFFFFF)
    for seed in range(5):
        L = lgm.random_filling(n, m, seed=int(rng_np.integers(0, 2**31)))
        C_lattice = lgm.lattice_to_matrix(L, n, w)
        C_randombg = rbgm.randombg_to_matrix(L, n, site_t, site_x, w)
        assert np.array_equal(C_lattice, C_randombg), (n, seed)


N_MOVES = 100_000


@pytest.mark.parametrize("n", [16, 30, 47])
@pytest.mark.parametrize("eps", [0.1, 0.21])
def test_random_bg_incremental_action_matches_full_recompute(n, eps):
    """Over 1e5 random moves on a fixed random background, the
    incrementally updated bitset histogram's action must match a full
    recomputation from L at every checked step."""
    w, h, m = rbgm.background_dims(n)
    rng = np.random.default_rng(hash((n, eps, "randombg")) & 0xFFFFFFFF)
    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=int(rng.integers(0, 2**31)))
    L = lgm.random_filling(n, m, seed=int(rng.integers(0, 2**31)))
    future, past, counts = rbg.build_randombg_state_bitset(L, n, site_t, site_x, w)
    from causet_lab.mcmc.action import f2_smear_table
    f2_table = f2_smear_table(max(n - 2, 0), eps)

    check_every = 500
    for step in range(N_MOVES):
        i = int(rng.integers(0, n))
        j = n + int(rng.integers(0, m - n))
        rbg.apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        if step % check_every == 0 or step == N_MOVES - 1:
            S_incremental = rbg.lattice_gas_action_from_counts(n, counts, f2_table, eps)
            S_full, Nk_full = rbgm.randombg_action_full(L, n, site_t, site_x, w, eps)
            assert abs(S_incremental - S_full) < 1e-6, (n, eps, step, S_incremental, S_full)
            assert np.array_equal(counts, Nk_full), (n, eps, step, counts, Nk_full)


def test_random_bg_rejected_move_is_exactly_undone():
    n = 30
    w, h, m = rbgm.background_dims(n)
    rng = np.random.default_rng(5)
    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=11)
    L = lgm.random_filling(n, m, seed=7)
    future, past, counts = rbg.build_randombg_state_bitset(L, n, site_t, site_x, w)
    L0, future0, past0, counts0 = L.copy(), future.copy(), past.copy(), counts.copy()
    for _ in range(50):
        i = int(rng.integers(0, n))
        j = n + int(rng.integers(0, m - n))
        rbg.apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        rbg.apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        assert np.array_equal(L, L0)
        assert np.array_equal(future, future0) and np.array_equal(past, past0)
        assert np.array_equal(counts, counts0)


@pytest.mark.parametrize("n", [30, 50])
def test_random_bg_beta0_filling_looks_like_2d_sprinkle(n):
    """A uniformly random filling (beta=0) on a random background
    should locally look like a 2D causal-set sprinkle -- same
    sub-interval-sampling methodology as Phase 4's regular-lattice
    control (test_lattice_gas.py), for the same reason: the whole
    filling lives on a cylinder, not an Alexandrov interval. Also
    reports the whole-filling ordering fraction against the regular
    lattice's ~0.88 at n=30, as a direct sanity check that the two
    models' causal geometries are on the same scale.
    """
    w, h, m = rbgm.background_dims(n)
    kmax = 6
    mm_dims, profiles, of_wholes = [], [], []
    for seed in range(5):
        # Background seed and filling/chain seed are independent draws.
        site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=1000 + seed)
        L = lgm.random_filling(n, m, seed=seed)
        C = rbgm.randombg_to_matrix(L, n, site_t, site_x, w)
        res = analyze_matrix(C, seed=seed, n_samples=300, min_size=2, max_size=n, kmax=kmax)
        mm_dims.append(res["mm_dim_sampled_mean"])
        profiles.append(res["abundance_profile"])
        of_wholes.append(res["ordering_fraction"])

    ref_cset, _ = sprinkle(3000, 2, seed=0)
    ref = analyze_matrix(ref_cset.C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=kmax)

    mean_dim = nanmean(mm_dims)
    mean_profile = np.nanmean(np.array(profiles), axis=0)
    dist = abundance_distance(mean_profile, ref["abundance_profile"])
    mean_of = nanmean(of_wholes)
    print(f"\nn={n}: random-bg MM dimension (sampled intervals)={mean_dim:.3f} (target ~2), "
          f"interval-abundance L1 distance to sprinkle-sampled reference={dist:.3f}, "
          f"whole-filling ordering fraction={mean_of:.4f} "
          f"(regular lattice at n=30, beta=0: ~0.882 -- see Phase 4 report)")
    assert abs(mean_dim - 2.0) < 0.5, (n, mm_dims)
    assert dist < 0.5, (n, mean_profile, ref["abundance_profile"])
