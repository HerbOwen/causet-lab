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
from causet_lab.mcmc.muca import (
    _peak_diagnostics, MIN_BARRIER_FOR_BIMODAL, MIN_PEAK_MASS_FRACTION, _widen,
    locate_beta_c_variance_peak, check_ln_g_anomalies, LN_G_STEP_ANOMALY_THRESHOLD,
)


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


# ---- _peak_diagnostics: deep-tail spurious "double peak" regression ----
# Found during the Phase 5 random-background analysis
# (results/phase5/random_background_report.md): a naive two-peak check
# flagged a large barrier (17-21) at 4/5 realizations purely because a
# local maximum 8-11 orders of magnitude below the real mode, sitting
# deep in an exponentially suppressed tail, was treated as a "peak" on
# equal footing with the real one. MIN_PEAK_MASS_FRACTION guards against
# exactly this; these tests pin that behavior down.

def test_peak_diagnostics_rejects_deep_tail_noise_bump():
    p = np.zeros(120)
    p[29] = 4e-10  # a floating-point-level wiggle in the deep tail, not a real peak
    p[60:75] = np.exp(-0.02 * (np.arange(60, 75) - 67) ** 2) * 0.037  # the real, single broad hump
    assert _peak_diagnostics(p) is None, "a tail bump far below the mode must not register as a second peak"


def _two_gaussian_bumps(n_bins, c1, c2, amp1, amp2, floor):
    """A smooth two-bump curve with a strictly positive floor everywhere
    (no exact zeros), so the valley between the bumps is a real, finite
    value rather than an artifact of an untouched-zero region."""
    x = np.arange(n_bins)
    return floor + amp1 * np.exp(-0.05 * (x - c1) ** 2) + amp2 * np.exp(-0.05 * (x - c2) ** 2)


def test_peak_diagnostics_still_finds_a_genuine_double_peak():
    p = _two_gaussian_bumps(120, c1=27, c2=87, amp1=0.03, amp2=0.03, floor=1e-8)
    diag = _peak_diagnostics(p)
    assert diag is not None, "two comparably-tall peaks with a real valley must still be detected"
    assert diag["barrier"] >= MIN_BARRIER_FOR_BIMODAL
    assert sorted(diag["peaks"]) == [27, 87]


def test_peak_diagnostics_mass_fraction_boundary():
    mode_amp = 0.037
    # Just below the mass-fraction floor: still rejected.
    p = _two_gaussian_bumps(120, c1=29, c2=67, amp1=0.5 * MIN_PEAK_MASS_FRACTION * mode_amp,
                             amp2=mode_amp, floor=1e-10)
    assert _peak_diagnostics(p) is None
    # Comfortably above it, with a real valley: now a legitimate candidate
    # (may or may not clear the barrier threshold, but must not be thrown
    # out by the mass filter alone).
    p = _two_gaussian_bumps(120, c1=29, c2=67, amp1=5.0 * MIN_PEAK_MASS_FRACTION * mode_amp,
                             amp2=mode_amp, floor=1e-10)
    diag = _peak_diagnostics(p)
    assert diag is not None


# _widen is shared by all three WL loops (orders, lattice gas, random
# background) whenever the edge-hit rate forces the histogram range to
# grow. New bins must continue the LOCALLY OBSERVED ln_g slope (linear
# extrapolation), not repeat the edge bin's value flatly: a flat block
# of WIDEN_BINS=60 identical values doesn't track the true decaying
# density of states, and once the walker reaches the FAR edge of that
# artificial plateau, differential visitation opens an arbitrarily
# large, unphysical gap there -- confirmed directly: an n=50
# random-background run produced a 66,000+ ln_g cliff exactly at a
# widened block's far edge, trapping the walker for 800+s. These tests
# pin down the actual fix: slope-continuation, direction-clamped (ln_g
# never extrapolated to exceed the edge value, since entropy only
# decreases further into a tail) and magnitude-clamped (the per-bin
# drop never exceeds the largest single step actually observed in the
# edge window, so one noisy window can't compound into a steep plunge).
def test_widen_extrapolates_the_local_slope_not_a_flat_repeat():
    # Realistic cold-edge shape: ln_g rising toward the bulk (bin 0 is
    # the tail edge). New low-side bins must continue DECREASING
    # further into the tail, strictly below ln_g[0], not flat at it.
    ln_g = np.array([10.0, 20.0, 30.0, 42.0])
    H = np.array([5, 6, 7, 8])
    new_ln_g, new_H, new_bin_lo, new_n_bins = _widen(ln_g, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="low", n_new=3)
    assert new_n_bins == 7
    assert new_bin_lo == -3.0
    assert np.array_equal(new_ln_g[3:], ln_g)
    assert np.all(new_H[:3] == 0), "new bins' histogram starts empty regardless"
    assert np.all(new_ln_g[:3] < ln_g[0]), "must drop below the edge value, not sit flat at it"
    assert np.all(np.diff(new_ln_g[:4]) > 0), "must increase monotonically toward the old range"

    # Realistic hot-edge shape: ln_g falling away from the bulk (bin -1
    # is the tail edge). New high-side bins must continue DECREASING
    # further into the tail, strictly below ln_g[-1].
    ln_g_hot = np.array([42.0, 30.0, 20.0, 10.0])
    new_ln_g, new_H, new_bin_lo, new_n_bins = _widen(ln_g_hot, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="high", n_new=3)
    assert new_n_bins == 7
    assert new_bin_lo == 0.0
    assert np.array_equal(new_ln_g[:4], ln_g_hot)
    assert np.all(new_ln_g[4:] < ln_g_hot[-1]), "must drop below the edge value, not sit flat at it"
    assert np.all(np.diff(new_ln_g[3:]) < 0), "must keep decreasing outward, continuing the observed slope"


def test_widen_clamps_wrong_direction_slope_to_flat_not_unphysical_increase():
    # The AVERAGE slope (not just one noisy step) points the "wrong"
    # way: ln_g is net INCREASING all the way to the hot edge (steps
    # [5, 7, 8], all positive), the opposite of a decaying tail. Must
    # clamp to flat, never extrapolate that increase past the edge.
    ln_g = np.array([10.0, 15.0, 22.0, 30.0])
    H = np.array([5, 6, 7, 8])
    new_ln_g, _, _, _ = _widen(ln_g, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="high", n_new=5)
    assert np.all(new_ln_g[4:] <= ln_g[-1] + 1e-12), "must never exceed the edge value on the high side"
    np.testing.assert_allclose(new_ln_g[4:], ln_g[-1])  # flat, not a wrong-direction extrapolation

    # Mirror: ln_g net DECREASING toward the cold edge (steps [-8,-7,-5]),
    # the opposite of rising into the bulk. Must clamp to flat.
    ln_g2 = np.array([30.0, 22.0, 15.0, 10.0])
    new_ln_g2, _, _, _ = _widen(ln_g2, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="low", n_new=5)
    assert np.all(new_ln_g2[:5] <= ln_g2[0] + 1e-12), "must never exceed the edge value on the low side"
    np.testing.assert_allclose(new_ln_g2[:5], ln_g2[0])


def test_widen_pins_down_the_exact_extrapolation_formula():
    # A clean, steadily-decreasing hot edge: steps = [-4, -5, -6],
    # avg_slope = -5, max_abs_step = 6 -- direction is correct (falling
    # away from the bulk), so per_bin_drop = min(max(5, 0), 6) = 5.
    ln_g = np.array([42.0, 38.0, 33.0, 27.0])
    H = np.array([5, 6, 7, 8])
    new_ln_g, _, _, _ = _widen(ln_g, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="high", n_new=3)
    np.testing.assert_allclose(new_ln_g[4:], [22.0, 17.0, 12.0])

    # Mirror case on the low side: steps = [6, 5, 4] (rising toward the
    # bulk from bin 0), avg_slope = 5, max_abs_step = 6 -> drop = 5.
    ln_g_cold = np.array([27.0, 33.0, 38.0, 42.0])
    new_ln_g2, _, _, _ = _widen(ln_g_cold, H, bin_lo=0.0, bin_width=1.0, n_bins=4, side="low", n_new=3)
    np.testing.assert_allclose(new_ln_g2[:3], [12.0, 17.0, 22.0])


# check_ln_g_anomalies is the cheap runtime guard added after finding
# (by direct inspection of real n=50 checkpoints) adjacent-bin ln_g
# steps of 101, 46967 and 66767 -- vs. 6-11 for every healthy run --
# exactly at a widened block's far edge. Healthy smooth variation must
# not be flagged; a real cliff must be, with the correct bin pair.
def test_check_ln_g_anomalies_passes_smooth_data():
    ln_g = np.array([0.0, 40.0, 85.0, 135.0, 180.0, 230.0, 270.0, 310.0])
    ever_visited = np.ones(8, dtype=bool)
    assert check_ln_g_anomalies(ln_g, ever_visited) == []


def test_check_ln_g_anomalies_flags_a_real_cliff():
    ln_g = np.array([0.0, 10.0, 20.0, 30.0, 40.0 + 50000.0, 50040.0, 50050.0])
    ever_visited = np.ones(7, dtype=bool)
    anomalies = check_ln_g_anomalies(ln_g, ever_visited)
    assert len(anomalies) == 1
    bin_a, bin_b, step = anomalies[0]
    assert (bin_a, bin_b) == (3, 4)
    assert abs(step) > LN_G_STEP_ANOMALY_THRESHOLD


def test_check_ln_g_anomalies_skips_unvisited_bins_in_the_span():
    # A huge raw difference that straddles a NEVER-visited bin (e.g. a
    # freshly-widened, not-yet-reached bin) is not a live trap signature
    # -- only adjacent pairs where BOTH bins are ever_visited count.
    ln_g = np.array([0.0, 10.0, 20.0, 99999.0, 30.0])
    ever_visited = np.array([True, True, True, False, True])
    assert check_ln_g_anomalies(ln_g, ever_visited) == []
