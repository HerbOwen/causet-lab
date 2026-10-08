"""Correctness and speed checks for the bitset-based incremental action
update (Phase 3b, part 1): bitset_core.py must agree with a full
recomputation over many random moves, and should be faster than the
NxN-matrix engine in fast_core.py.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from causet_lab.mcmc.action import f2_smear_table, bd_action_2d_smeared
from causet_lab.mcmc.orders import orders_to_matrix
from causet_lab.mcmc import bitset_core as bc
from causet_lab.mcmc import fast_core as fc

N_MOVES = 100_000


@pytest.mark.parametrize("N", [16, 30, 47, 60])
@pytest.mark.parametrize("eps", [0.21, 0.5])
def test_bitset_incremental_dS_matches_full_recompute(N, eps):
    """Over 1e5 random moves, the action implied by the incrementally
    updated bitset histogram must match a full recompute from (u, v) at
    every step -- not just the final state, since a bug that only shows
    up transiently (e.g. a dropped delta) would otherwise be missed.
    Checked every 500 moves (checking every single move would make this
    test itself O(N^3) x 1e5, i.e. minutes long) plus always on the
    final state.
    """
    rng = np.random.default_rng(hash((N, eps)) & 0xFFFFFFFF)
    u = rng.permutation(N).astype(np.int64)
    v = rng.permutation(N).astype(np.int64)
    future, past, counts = bc.build_state_bitset(u, v)
    f2_table = f2_smear_table(max(N - 2, 0), eps)

    check_every = 500
    for step in range(N_MOVES):
        which = int(rng.integers(0, 2))
        i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
        bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
        if step % check_every == 0 or step == N_MOVES - 1:
            S_incremental = bc.action_from_counts(N, counts, f2_table, eps)
            C_full = orders_to_matrix(u, v, natural_label=False)
            S_full, _Nk = bd_action_2d_smeared(C_full, eps)
            assert abs(S_incremental - S_full) < 1e-6, (N, eps, step, S_incremental, S_full)


def test_bitset_rejected_move_is_exactly_undone():
    rng = np.random.default_rng(3)
    N = 30
    u = rng.permutation(N).astype(np.int64)
    v = rng.permutation(N).astype(np.int64)
    future, past, counts = bc.build_state_bitset(u, v)
    u0, v0 = u.copy(), v.copy()
    future0, past0, counts0 = future.copy(), past.copy(), counts.copy()
    for _ in range(50):
        which = int(rng.integers(0, 2))
        i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
        bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
        bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
        assert np.array_equal(u, u0) and np.array_equal(v, v0)
        assert np.array_equal(future, future0) and np.array_equal(past, past0)
        assert np.array_equal(counts, counts0)


@pytest.mark.parametrize("N", [20, 40, 60])
def test_bitset_engine_is_faster_than_fast_core(N):
    """Not a strict correctness test -- reports (via assert + print,
    visible with -s) the measured speedup of the bitset engine over
    fast_core's NxN-matrix engine for the same workload (same number of
    sweeps, same move proposals in spirit). Asserts only a modest floor
    (>= 1.2x) so this doesn't flake on a loaded CI box while still
    catching a regression that makes the "fast" engine slower.
    """
    eps = 0.21
    seed = 0
    n_sweeps, burn_in, measure_every = 300, 50, 5
    f2_table = f2_smear_table(max(N - 2, 0), eps)
    max_measurements = n_sweeps // measure_every + 2

    rng = np.random.default_rng(seed)
    u = rng.permutation(N).astype(np.int64)
    v = rng.permutation(N).astype(np.int64)
    C, sizes, counts_fc = fc.build_state(u, v)
    fc.run_sweeps_jit(u.copy(), v.copy(), C.copy(), sizes.copy(), counts_fc.copy(), N, 0.05, eps,
                       f2_table, 5, 2, 1, 0.0, False, seed, 10)  # warm up JIT
    t0 = time.perf_counter()
    fc.run_sweeps_jit(u, v, C, sizes, counts_fc, N, 0.05, eps, f2_table, n_sweeps, burn_in,
                       measure_every, 0.0, False, seed, max_measurements)
    t_fast_core = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    u = rng.permutation(N).astype(np.int64)
    v = rng.permutation(N).astype(np.int64)
    future, past, counts_bc = bc.build_state_bitset(u, v)
    bc.run_sweeps_jit(u.copy(), v.copy(), future.copy(), past.copy(), counts_bc.copy(), N, 0.05, eps,
                       f2_table, 5, 2, 1, 0.0, False, seed, 10)  # warm up JIT
    t0 = time.perf_counter()
    bc.run_sweeps_jit(u, v, future, past, counts_bc, N, 0.05, eps, f2_table, n_sweeps, burn_in,
                       measure_every, 0.0, False, seed, max_measurements)
    t_bitset = time.perf_counter() - t0

    speedup = t_fast_core / t_bitset if t_bitset > 0 else float("inf")
    print(f"\nN={N}: fast_core={t_fast_core*1000:.1f}ms, bitset={t_bitset*1000:.1f}ms, speedup={speedup:.2f}x")
    assert speedup >= 1.2, f"N={N}: bitset engine ({t_bitset:.4f}s) not faster than fast_core ({t_fast_core:.4f}s)"
