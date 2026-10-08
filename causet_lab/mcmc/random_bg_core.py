"""Bitset-based incremental engine for the Phase 5 random-background
variant of the Phase 4 lattice-gas model (Cunningham & Surya,
arXiv:1908.11647, "C&S"): same cylinder region, same filling/move
representation, same smeared action, but the m background sites are a
fixed (quenched) uniform-random (Poisson-style) sprinkling of points
into the cylinder's (t, x) rectangle, instead of C&S's regular lattice
grid. C&S's own Sec. 6 discussion flags "a more realistic model of
discreteness" (a random rather than regular background) as an open
question they did not pursue -- this module and random_bg.py are this
project's own construction, not a quote from the paper.

Region and units -- MUST match lattice_gas_core.py's exactly, not the
continuum metric's own (t, theta) in radians. lattice_gas_core's
lattice_precedes compares an integer step count dk (circumference = w
steps) directly against dt (t measured in the same integer-step
units): the lattice's own effective metric is ds^2 = -dt^2 + dx^2 with
x in lattice-spacing units and circumference w, NOT theta in radians
with circumference 2*pi (those differ by a factor of w/(2*pi)). This
module therefore sprinkles a continuous coordinate x (period w, same
units as t), not theta -- matching "the same shape and size as the
C&S lattice region" literally, in the same units the lattice's own
causal relation uses, not merely the same physical angle range.
Verified directly against lattice_gas_core.build_lattice_state_bitset:
placing site_t/site_x at the exact integer lattice positions
reproduces lattice_to_matrix bit-for-bit (test_random_bg.py).

    (t_a, x_a) precedes (t_b, x_b)  iff  t_b > t_a
        and  min(dx, w - dx) <= t_b - t_a
        where dx = (x_b - x_a) mod w

State kept across moves -- identical shapes/roles to lattice_gas_core,
with the fixed background position arrays added:
    L        (m,) int64    site-id permutation; L[0:n] = elements
    site_t   (m,) float64  fixed background, t-coordinate per site
    site_x   (m,) float64  fixed background, x-coordinate per site
                           (period w, same units as t -- not radians)
    future, past, counts -- exactly as lattice_gas_core.py

The action (lattice_gas_core.lattice_gas_action_from_counts, C&S Eq. 8)
and the move's self-inverse relocate structure are geometry-independent
and reused directly/unmodified -- only the causal-relation evaluation
and the position lookup (site_t/site_x indexed by site id, instead of
decoding t=s//w, k=s%w) differ from lattice_gas_core.py.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .bitset_core import popcount64
from . import rng
from .lattice_gas_core import (
    bin_index, lattice_gas_action_from_counts, _lattice_gas_action_from_counts_jit,
)


@njit(cache=True, inline="always")
def randombg_precedes(t_a, x_a, t_b, x_b, w):
    """Continuous-position twin of lattice_gas_core.lattice_precedes --
    see module docstring: x has period w (lattice-spacing units), not
    2*pi radians."""
    dt = t_b - t_a
    if dt <= 0.0:
        return False
    dx = (x_b - x_a) % w
    if dx > w - dx:
        dx = w - dx
    return dx <= dt


def build_randombg_state_bitset(L: np.ndarray, n: int, site_t: np.ndarray, site_x: np.ndarray, w: float):
    """Full O(n^2) build of (future, past, counts) from L[0:n] -- exact
    twin of lattice_gas_core.build_lattice_state_bitset, positions
    looked up from the fixed background arrays instead of decoded from
    the site id via // w, % w."""
    assert n <= 64, "random_bg_core supports n <= 64 only (see bitset_core module docstring)"
    t = site_t[L[:n]]
    x = site_x[L[:n]]
    future = np.zeros(n, dtype=np.uint64)
    past = np.zeros(n, dtype=np.uint64)
    for a in range(n):
        fa = np.uint64(0)
        pa = np.uint64(0)
        for b in range(n):
            if b == a:
                continue
            if randombg_precedes(t[a], x[a], t[b], x[b], w):
                fa |= np.uint64(1) << np.uint64(b)
            elif randombg_precedes(t[b], x[b], t[a], x[a], w):
                pa |= np.uint64(1) << np.uint64(b)
        future[a] = fa
        past[a] = pa
    counts = np.zeros(max(n - 1, 1), dtype=np.int64)
    for a in range(n):
        fa = future[a]
        for b in range(n):
            if (fa >> np.uint64(b)) & np.uint64(1):
                s = popcount64(fa & past[b])
                counts[s] += 1
    return future, past, counts


def apply_relocate_randombg_py(L, future, past, counts, n, site_t, site_x, w, i, j):
    """Pure-Python reference -- exact twin of
    lattice_gas_core.apply_relocate_py, positions looked up from
    site_t/site_x[L[i]] instead of decoded via // w, % w."""
    one = np.uint64(1)
    ibit = one << np.uint64(i)
    old_future_i = future[i]
    old_past_i = past[i]

    for b in range(n):
        if b == i:
            continue
        bbit = one << np.uint64(b)
        if old_future_i & bbit:
            counts[popcount64(old_future_i & past[b])] -= 1
        if old_past_i & bbit:
            counts[popcount64(future[b] & old_past_i)] -= 1

    L[i], L[j] = L[j], L[i]
    new_t, new_x = site_t[L[i]], site_x[L[i]]

    new_future_i = np.uint64(0)
    new_past_i = np.uint64(0)
    i_clear = ~ibit
    for x_idx in range(n):
        if x_idx == i:
            continue
        xbit = one << np.uint64(x_idx)
        future[x_idx] &= i_clear
        past[x_idx] &= i_clear
        xt, xx = site_t[L[x_idx]], site_x[L[x_idx]]
        if randombg_precedes(new_t, new_x, xt, xx, w):
            new_future_i |= xbit
            past[x_idx] |= ibit
        elif randombg_precedes(xt, xx, new_t, new_x, w):
            new_past_i |= xbit
            future[x_idx] |= ibit
    future[i], past[i] = new_future_i, new_past_i

    for b in range(n):
        if b == i:
            continue
        bbit = one << np.uint64(b)
        if new_future_i & bbit:
            counts[popcount64(new_future_i & past[b])] += 1
        if new_past_i & bbit:
            counts[popcount64(future[b] & new_past_i)] += 1

    for a in range(n):
        if a == i:
            continue
        abit = one << np.uint64(a)
        fa = future[a] & i_clear
        for b in range(n):
            if b == i or b == a:
                continue
            bbit = one << np.uint64(b)
            if not (fa & bbit):
                continue
            old_i_between = 1 if (old_past_i & abit) and (old_future_i & bbit) else 0
            new_i_between = 1 if (past[i] & abit) and (future[i] & bbit) else 0
            delta = new_i_between - old_i_between
            if delta != 0:
                new_size = popcount64(future[a] & past[b])
                old_size = new_size - delta
                counts[old_size] -= 1
                counts[new_size] += 1


apply_relocate_randombg_jit = njit(cache=True)(apply_relocate_randombg_py)


@njit(cache=True)
def wl_or_muca_sweep_chunk_randombg_jit(
    L, future, past, counts, n, m, site_t, site_x, w, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins, f_mod, wl_mode,
    n_moves, rng_state, bin_idx,
    extreme_state, half_trips, edge_hits, lo_extreme, hi_extreme,
):
    """Random-background twin of lattice_gas_core.wl_or_muca_sweep_chunk_jit."""
    for _move in range(n_moves):
        rng_state, i = rng.next_below(rng_state, n)
        rng_state, j_off = rng.next_below(rng_state, m - n)
        j = n + j_off

        apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        S_new = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
            edge_hits += 1
        else:
            d = ln_g[bin_idx] - ln_g[new_bin]
            accept = d >= 0.0
            if not accept:
                rng_state, r = rng.next_double(rng_state)
                accept = r < np.exp(d)
            if accept:
                bin_idx = new_bin
            else:
                apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)

        if wl_mode:
            ln_g[bin_idx] += f_mod
        H[bin_idx] += 1

        if bin_idx <= lo_extreme:
            if extreme_state[0] == 1:
                half_trips[0] += 1
            extreme_state[0] = -1
        elif bin_idx >= hi_extreme:
            if extreme_state[0] == -1:
                half_trips[0] += 1
            extreme_state[0] = 1

    return rng_state, bin_idx, edge_hits


@njit(cache=True)
def muca_measure_chunk_randombg_jit(
    L, future, past, counts, n, m, site_t, site_x, w, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins,
    n_moves, rng_state, bin_idx, half_trips, extreme_state,
    measure_every, max_measurements, lo_extreme, hi_extreme,
):
    """Random-background twin of lattice_gas_core.muca_measure_chunk_jit."""
    rec_bins = np.empty(max_measurements, dtype=np.int64)
    rec_S = np.empty(max_measurements, dtype=np.float64)
    n_rec = 0
    S = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)

    for move in range(n_moves):
        rng_state, i = rng.next_below(rng_state, n)
        rng_state, j_off = rng.next_below(rng_state, m - n)
        j = n + j_off

        apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        S_new = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        else:
            d = ln_g[bin_idx] - ln_g[new_bin]
            accept = d >= 0.0
            if not accept:
                rng_state, r = rng.next_double(rng_state)
                accept = r < np.exp(d)
            if accept:
                bin_idx = new_bin
                S = S_new
            else:
                apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)

        H[bin_idx] += 1
        if bin_idx <= lo_extreme:
            if extreme_state[0] == 1:
                half_trips[0] += 1
            extreme_state[0] = -1
        elif bin_idx >= hi_extreme:
            if extreme_state[0] == -1:
                half_trips[0] += 1
            extreme_state[0] = 1

        if move % measure_every == 0 and n_rec < max_measurements:
            rec_bins[n_rec] = bin_idx
            rec_S[n_rec] = S
            n_rec += 1

    return rng_state, bin_idx, rec_bins[:n_rec], rec_S[:n_rec]
