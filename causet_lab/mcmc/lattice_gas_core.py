"""Bitset-based incremental engine for the d=2 lattice-gas causal set
model of Cunningham & Surya, "Dimensionally Restricted Causal Set
Quantum Gravity", arXiv:1908.11647 ("C&S" below).

Background lattice (C&S Sec. 2, Eq. 3 and the paragraph defining the
latticisation L_d^(m)): spacetime M_2 ~ R x S^1 with metric
ds^2 = -dt^2 + dtheta^2, theta ~ theta + 2*pi, "local light cones ...
at 45 degrees". The lattice L_2^(m) has h*w = m sites (t, theta) with
t in {0,...,h}, theta = 2*pi*k/w for k in {0,...,w-1}. This project
uses C&S's own sizing convention for d=2: aspect ratio alpha = h/w = 4,
w = n (number of elements), so m = alpha*w^2 = 4*n^2 (quoted directly:
"we use ... alpha=4 ... m ~ 1.6x10^5" for their n=200, matching
4*200^2 = 160000).

Lattice-gas filling (C&S Sec. 2, "an n-site filling of L_d^(m)"): an
n-site filling is n of the m sites, represented (quoted) as "a random
permutation of L" where L = {0,...,m-1} lists all site ids, with the
occupied sites being L[0:n] and the move set being "a swap between a
randomly chosen element in the first n entries ... with a randomly ...
chosen element from the last m-n entries". This project represents a
filling exactly this way: L is an (m,) int64 array, L[i] for
i in [0, n) is the site id of causal-set element i, and L[j] for
j in [n, m) is an unoccupied site id (its value, order among unfilled
slots does not matter). A site id s decodes to (t, k) via
t = s // w, k = s % w (quoted: "t_i = floor(L_i/w) ... theta_i =
(2*pi/w)(L_i mod w)").

CAUSAL RELATION -- flagged, not directly quoted: C&S state the causal
structure is "given via the metric Eqn [3]" with 45-degree light cones
and confirm (their Fig. 2(ii)) that relations occur "'around' the
cylinder" due to the S^1 topology, but do NOT give an explicit
integer-coordinate formula. The formula below is this project's own
derivation from those stated facts, not a direct quote:

    (t_a, k_a) precedes (t_b, k_b)  iff  t_b > t_a
                                     and  min(dk, w - dk) <= t_b - t_a
                                     where dk = (k_b - k_a) mod w.

min(dk, w-dk) is the shortest-path distance on a cycle of
circumference w, which already accounts for wraparound in either
direction correctly without needing separate "how many times does the
light ray wind around" logic: going around further can only ever be
a longer path on a cycle, never shorter, so the single-shortest-arc
distance is exactly the right causal reach at time-separation
t_b - t_a. Pairs exactly on the lightcone (equality) are treated as
related, matching the usual causal-set convention that the causal
relation is the closure of the open lightcone, not its open interior
-- the paper does not address this boundary case either.

State kept across moves (parallels bitset_core.py's (u, v, future,
past, counts) for N <= 64):
    L         (m,) int64    site-id permutation; L[0:n] = elements
    future    (n,) uint64   future[i] bit j set iff element i < element j
    past      (n,) uint64   past[i] bit j set iff element j < element i
    counts    (n-1,) int64  interval-size histogram, counts[s] = n_s
                            (C&S's "n_r", r-element-interval abundance)

A lattice-gas move swaps L[i] (i in [0,n), a filled slot = element i)
with L[j] (j in [n,m), an unfilled slot) -- apply_relocate() mutates
(L, future, past, counts) in place; a rejected move is undone by
reapplying it (swapping two array entries twice is the identity, same
self-inverse property bitset_core.apply_swap relies on).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .bitset_core import popcount64
from . import rng


@njit(cache=True, inline="always")
def lattice_precedes(t_a, k_a, t_b, k_b, w):
    """(t_a,k_a) precedes (t_b,k_b) -- see module docstring for the
    derivation (not a direct quote from C&S)."""
    dt = t_b - t_a
    if dt <= 0:
        return False
    dk = (k_b - k_a) % w
    if dk > w - dk:
        dk = w - dk
    return dk <= dt


def build_lattice_state_bitset(L: np.ndarray, n: int, w: int):
    """Full O(n^2) build of (future, past, counts) from L[0:n] -- used
    once per walker init/checkpoint-resume and as the correctness
    ground truth (mirrors bitset_core.build_state_bitset exactly,
    just with the lattice causal relation in place of the 2D-order
    comparison u[x]<u[y] and v[x]<v[y]).
    """
    assert n <= 64, "lattice_gas_core supports n <= 64 only (see bitset_core module docstring)"
    t = (L[:n] // w).astype(np.int64)
    k = (L[:n] % w).astype(np.int64)
    future = np.zeros(n, dtype=np.uint64)
    past = np.zeros(n, dtype=np.uint64)
    for x in range(n):
        fx = np.uint64(0)
        px = np.uint64(0)
        for y in range(n):
            if y == x:
                continue
            if lattice_precedes(t[x], k[x], t[y], k[y], w):
                fx |= np.uint64(1) << np.uint64(y)
            elif lattice_precedes(t[y], k[y], t[x], k[x], w):
                px |= np.uint64(1) << np.uint64(y)
        future[x] = fx
        past[x] = px
    counts = np.zeros(max(n - 1, 1), dtype=np.int64)
    for x in range(n):
        fx = future[x]
        for y in range(n):
            if (fx >> np.uint64(y)) & np.uint64(1):
                s = popcount64(fx & past[y])
                counts[s] += 1
    return future, past, counts


def lattice_gas_action_from_counts(n: int, counts: np.ndarray, f2_table: np.ndarray, eps: float) -> float:
    """C&S Eq. 8 (quoted in the module docstring of lattice_gas.py):

        S_BD^(2)/hbar = 2*eps*[n - 2*eps*sum_{r=1}^{n-1} n_r*f_2(r-1,eps)]

    reusing action.py's f2_smear_table kernel for f_2 (confirmed
    identical formula between the two papers), but NOTE the outer
    prefactor (2*eps here vs 4*eps in bitset_core.action_from_counts)
    and the summation bound (r starts at 1 here, excluding links/n_0
    entirely, vs bitset_core's sum starting at n=0) genuinely differ
    from the order-based smeared action used in Phase 3/3b -- both
    confirmed by direct quotation from each paper, not a bug in
    either. counts[r] here is exactly C&S's n_r (r = number of
    elements strictly between the pair).
    """
    acc = 0.0
    for r in range(1, n - 1):
        acc += counts[r] * f2_table[r - 1]
    return 2.0 * eps * (n - 2.0 * eps * acc)


@njit(cache=True)
def _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps):
    acc = 0.0
    for r in range(1, n - 1):
        acc += counts[r] * f2_table[r - 1]
    return 2.0 * eps * (n - 2.0 * eps * acc)


def apply_relocate_py(L, future, past, counts, n, w, i, j):
    """Pure-Python reference (see module docstring for the algorithm).
    The jitted version below must stay behaviorally identical. Mirrors
    bitset_core.apply_swap_py's 5-step skeleton, simplified to a
    single changing element (i) -- j is an unfilled lattice slot, not
    a causal-set element, so only i's own bitset is rebuilt and only
    i's membership in third-party intervals needs rechecking.
    """
    one = np.uint64(1)
    ibit = one << np.uint64(i)
    old_future_i = future[i]
    old_past_i = past[i]

    # Step 0: remove old endpoint-pair contributions (pairs with i as
    # an endpoint), using the pre-move bitsets.
    for b in range(n):
        if b == i:
            continue
        bbit = one << np.uint64(b)
        if old_future_i & bbit:
            counts[popcount64(old_future_i & past[b])] -= 1
        if old_past_i & bbit:
            counts[popcount64(future[b] & old_past_i)] -= 1

    L[i], L[j] = L[j], L[i]
    new_t, new_k = L[i] // w, L[i] % w

    # Step 2: rebuild i's bitset from scratch; patch bit i into every
    # other element's future/past (O(n) total).
    new_future_i = np.uint64(0)
    new_past_i = np.uint64(0)
    i_clear = ~ibit
    for x in range(n):
        if x == i:
            continue
        xbit = one << np.uint64(x)
        future[x] &= i_clear
        past[x] &= i_clear
        xt, xk = L[x] // w, L[x] % w
        if lattice_precedes(new_t, new_k, xt, xk, w):
            new_future_i |= xbit
            past[x] |= ibit
        elif lattice_precedes(xt, xk, new_t, new_k, w):
            new_past_i |= xbit
            future[x] |= ibit
    future[i], past[i] = new_future_i, new_past_i

    # Step 3: add new endpoint-pair contributions, post-move.
    for b in range(n):
        if b == i:
            continue
        bbit = one << np.uint64(b)
        if new_future_i & bbit:
            counts[popcount64(new_future_i & past[b])] += 1
        if new_past_i & bbit:
            counts[popcount64(future[b] & new_past_i)] += 1

    # Step 4: for every other related pair (a, b), only i's membership
    # in that interval can have changed (a, b themselves did not
    # move) -- both old and new "is i between (a, b)" tests reduce to
    # bit lookups on old/new_{future,past}_i, no other element's
    # bitset snapshot needed.
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


apply_relocate_jit = njit(cache=True)(apply_relocate_py)


@njit(cache=True, inline="always")
def bin_index(S, bin_lo, bin_width, n_bins):
    idx = np.int64(np.floor((S - bin_lo) / bin_width))
    if idx < 0 or idx >= n_bins:
        return -1
    return idx


@njit(cache=True)
def wl_or_muca_sweep_chunk_jit(
    L, future, past, counts, n, m, w, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins, f_mod, wl_mode,
    n_moves, rng_state, bin_idx,
    extreme_state, half_trips, edge_hits, lo_extreme, hi_extreme,
):
    """Lattice-gas twin of muca_core.wl_or_muca_sweep_chunk_jit, same
    structure and same guarantees (see its docstring for the round-trip
    rationale), just with the move proposal and move-application
    swapped for this model's: pick a filled slot i in [0,n) and an
    unfilled slot j in [n,m) independently and uniformly at random and
    swap them (C&S Sec. 2, quoted in lattice_gas.py's module
    docstring) instead of the 2D-order coordinate-rank swap.
    """
    for _move in range(n_moves):
        rng_state, i = rng.next_below(rng_state, n)
        rng_state, j_off = rng.next_below(rng_state, m - n)
        j = n + j_off

        apply_relocate_jit(L, future, past, counts, n, w, i, j)
        S_new = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            apply_relocate_jit(L, future, past, counts, n, w, i, j)  # undo
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
                apply_relocate_jit(L, future, past, counts, n, w, i, j)  # undo

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
def muca_measure_chunk_jit(
    L, future, past, counts, n, m, w, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins,
    n_moves, rng_state, bin_idx, half_trips, extreme_state,
    measure_every, max_measurements, lo_extreme, hi_extreme,
):
    """Lattice-gas twin of muca_core.muca_measure_chunk_jit (MUCA
    production, wl_mode always False, also records the (bin, S) time
    series at measure_every resolution) -- see its docstring."""
    rec_bins = np.empty(max_measurements, dtype=np.int64)
    rec_S = np.empty(max_measurements, dtype=np.float64)
    n_rec = 0
    S = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)

    for move in range(n_moves):
        rng_state, i = rng.next_below(rng_state, n)
        rng_state, j_off = rng.next_below(rng_state, m - n)
        j = n + j_off

        apply_relocate_jit(L, future, past, counts, n, w, i, j)
        S_new = _lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            apply_relocate_jit(L, future, past, counts, n, w, i, j)
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
                apply_relocate_jit(L, future, past, counts, n, w, i, j)

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
