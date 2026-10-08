"""Wang-Landau / multicanonical (MUCA) jitted sweep kernels, built on the
bitset incremental-action engine (bitset_core.py) and the checkpointable
splitmix64 RNG (rng.py).

Both stages share one sweep-chunk kernel (wl_or_muca_sweep_chunk_jit)
with a flag:

  Wang-Landau (wl_mode=True): ln_g is updated by +f_mod at every visited
  bin (accepted or self-looped), estimating the density of states
  g(S) up to an additive constant. Acceptance rule: min(1,
  exp(ln_g[old_bin] - ln_g[new_bin])) -- a move into a bin WL currently
  thinks is over-visited (large ln_g) is disfavored, which is exactly
  what flattens the visit histogram across the whole binned range,
  including through a first-order barrier a fixed-beta Metropolis walk
  would almost never cross.

  MUCA production (wl_mode=False): ln_g is frozen (the converged
  Wang-Landau estimate) and used only as a fixed importance-sampling
  weight; the walk's own bin-visit histogram should come out flat if
  the ln_g estimate is good, and is kept (H_muca) purely as a
  diagnostic, not fed back into ln_g.

Binning is fixed-size per call (bin_lo, bin_width, n_bins); a proposed
move that would land outside [0, n_bins) is rejected and counted as an
edge hit rather than causing an array index error -- the Python-level
driver in muca.py decides whether to widen the range and restart from
the last checkpoint when edge hits are frequent (see its module
docstring for why growing the bin array isn't done inside the kernel).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import bitset_core as bc
from . import rng


@njit(cache=True, inline="always")
def bin_index(S, bin_lo, bin_width, n_bins):
    idx = np.int64(np.floor((S - bin_lo) / bin_width))
    if idx < 0 or idx >= n_bins:
        return -1
    return idx


@njit(cache=True)
def wl_or_muca_sweep_chunk_jit(
    u, v, future, past, counts, N, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins, f_mod, wl_mode,
    n_moves, rng_state, bin_idx,
    extreme_state, half_trips, edge_hits, lo_extreme, hi_extreme,
):
    """Run n_moves individual proposal/accept steps. Mutates u, v,
    future, past, counts, ln_g (if wl_mode), H in place. Returns
    (rng_state, bin_idx, extreme_state, half_trips, edge_hits) --
    scalars threaded explicitly so the whole chunk is exactly resumable
    from a checkpoint (see rng.py's module docstring for why RNG state
    in particular must be threaded this way rather than via
    np.random.seed).

    Round trips are counted between lo_extreme and hi_extreme -- the
    bin indices of the *actually reachable* support (see muca.py's
    stall-handling), not the raw array bounds 0 and n_bins - 1. For a
    finite N the discrete action spectrum can have real gaps near the
    edges of the nominal bin range, in which case those literal array
    bounds are never visited by any configuration and a round trip
    would be impossible to detect by construction -- not evidence that
    the sampler isn't mixing.
    """
    for _move in range(n_moves):
        rng_state, which = rng.next_below(rng_state, 2)
        rng_state, i = rng.next_below(rng_state, N)
        rng_state, j = rng.next_below(rng_state, N - 1)
        if j >= i:
            j += 1

        bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
        S_new = bc._action_from_counts_jit(N, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)  # undo
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
                bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)  # undo

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
    u, v, future, past, counts, N, eps, f2_table,
    ln_g, H, bin_lo, bin_width, n_bins,
    n_moves, rng_state, bin_idx, half_trips, extreme_state,
    measure_every, max_measurements, lo_extreme, hi_extreme,
):
    """MUCA production variant of the sweep chunk (wl_mode always
    False) that also records (bin, S) every measure_every moves, for
    later per-bin observable averaging and reweighting in muca.py.
    Kept as a separate function from wl_or_muca_sweep_chunk_jit (rather
    than adding yet more flags/outputs to it) since the measurement
    bookkeeping is only ever needed in MUCA mode.

    half_trips and extreme_state are passed in and mutated in place
    (not created fresh here) so round-trip counting is correct across
    chunk boundaries -- an extreme_state reset every call would forget
    "currently at the low/high extreme" right as a chunk ends, and
    either double-count or silently drop the trip that straddles it.
    """
    rec_bins = np.empty(max_measurements, dtype=np.int64)
    rec_S = np.empty(max_measurements, dtype=np.float64)
    n_rec = 0
    S = bc._action_from_counts_jit(N, counts, f2_table, eps)

    for move in range(n_moves):
        rng_state, which = rng.next_below(rng_state, 2)
        rng_state, i = rng.next_below(rng_state, N)
        rng_state, j = rng.next_below(rng_state, N - 1)
        if j >= i:
            j += 1

        bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
        S_new = bc._action_from_counts_jit(N, counts, f2_table, eps)
        new_bin = bin_index(S_new, bin_lo, bin_width, n_bins)

        if new_bin < 0:
            bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)
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
                bc.apply_swap_jit(u, v, future, past, counts, N, which, i, j)

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
