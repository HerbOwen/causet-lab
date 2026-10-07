"""Numba-compiled Metropolis core with an incremental action update.

Why: the plain sampler (sampler.py) rebuilds the whole N x N relation
matrix and reruns the full O(N^3) interval-size computation from scratch
on every single proposed move. For N <= 80 the FLOP count was never the
bottleneck (BLAS handles an 80x80 matmul in microseconds) -- the real
cost was dozens of separate numpy calls per move, each with Python-level
dispatch overhead. This module removes both problems: Numba compiles a
tight loop with no per-call overhead, and the action update is made
incremental, exploiting the fact that swapping u[i] <-> u[j] (or v[i],
v[j]) only changes relations that have i or j as an endpoint -- every
other pair (a, b) keeps its own direct relation C[a, b] unchanged, and
only needs its interval *size* (number of elements strictly between a
and b) adjusted by how many of {i, j} newly started or stopped being
between them. That is an O(N^2) update instead of an O(N^3) matrix
product, with none of the numpy call overhead on top.

The action used throughout is the *smeared* 2D Benincasa-Dowker action
(see action.py's module docstring for the formula and its source,
Glaser/O'Connor/Surya 2018, arXiv:1706.06432), which needs the FULL
histogram of interval sizes N_n for n = 0..N-2 (not just n <= 2) -- the
incremental update below tracks that full histogram; only the final
weighted sum against a per-(N, eps) lookup table f2_table (built once
per chain by action.f2_smear_table and passed in untouched) differs
from the old 3-bucket version. That weighted sum is an O(N) reduction
per move, which does not change the overall O(N^2)-per-move complexity
since the incremental relation/size update already dominates at O(N^2).

State kept across moves (all plain numpy arrays, no Python objects, so
every array can be handed to @njit functions untouched):
    u, v      (N,) int64      the two permutations
    C         (N, N) bool_    current relation matrix
    sizes     (N, N) int32    interval size of (a, b); meaningful only
                               where C[a, b] is True, but kept accurate
                               everywhere so a pair crossing between
                               interval sizes is handled correctly next
                               time it changes
    counts    (N-1,) int64    full histogram: counts[s] = N_s, number of
                               related pairs with exactly s elements
                               between them, for s = 0..N-2

A move is applied via apply_swap(); a *rejected* move is undone by
calling apply_swap() again with the same (which, i, j), since swapping
the same two entries twice is the identity -- there is no separate
"revert" code path to keep in sync with the "apply" path.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def build_state(u: np.ndarray, v: np.ndarray):
    """Full O(N^3) build of C, sizes, counts from scratch. Used once per
    chain at initialization (and by the correctness test as the ground
    truth to compare the incremental path against) -- never in the hot
    loop. counts is the full histogram, length max(N-1, 1).
    """
    N = u.shape[0]
    C = np.zeros((N, N), dtype=np.bool_)
    for a in range(N):
        for b in range(N):
            if a != b and u[a] < u[b] and v[a] < v[b]:
                C[a, b] = True
    sizes = np.zeros((N, N), dtype=np.int32)
    counts = np.zeros(max(N - 1, 1), dtype=np.int64)
    for a in range(N):
        for b in range(N):
            if C[a, b]:
                s = 0
                for k in range(N):
                    if C[a, k] and C[k, b]:
                        s += 1
                sizes[a, b] = s
                counts[s] += 1
    return C, sizes, counts


def action_from_counts(N: int, counts: np.ndarray, f2_table: np.ndarray, eps: float) -> float:
    """Smeared 2D BD action from a full interval-size histogram -- the
    non-jit equivalent of the weighted-sum loop in run_sweeps_jit /
    run_pt_jit, kept here so callers outside the hot loop (e.g. tests,
    diagnostics) don't need to duplicate the formula.
    """
    acc = float(np.dot(counts, f2_table))
    return 4.0 * eps * (N - 2.0 * eps * acc)


@njit(cache=True)
def _bucket_dec(counts, s):
    counts[s] -= 1


@njit(cache=True)
def _bucket_inc(counts, s):
    counts[s] += 1


def apply_swap_py(u, v, C, sizes, counts, N, which, i, j):
    """Pure-Python reference implementation (see module docstring for the
    algorithm). Kept separate from the @njit version in fast_core_jit so
    this one can be stepped through / printed from while debugging; the
    jitted version below must stay behaviorally identical to this one.
    """
    old_row_i = C[i, :].copy()
    old_col_i = C[:, i].copy()
    old_row_j = C[j, :].copy()
    old_col_j = C[:, j].copy()

    for b in range(N):
        if b == i or b == j:
            continue
        if C[i, b]:
            _bucket_dec(counts, sizes[i, b])
        if C[b, i]:
            _bucket_dec(counts, sizes[b, i])
        if C[j, b]:
            _bucket_dec(counts, sizes[j, b])
        if C[b, j]:
            _bucket_dec(counts, sizes[b, j])
    if C[i, j]:
        _bucket_dec(counts, sizes[i, j])
    if C[j, i]:
        _bucket_dec(counts, sizes[j, i])

    if which == 0:
        u[i], u[j] = u[j], u[i]
    else:
        v[i], v[j] = v[j], v[i]

    for b in range(N):
        if b == i or b == j:
            continue
        C[i, b] = u[i] < u[b] and v[i] < v[b]
        C[b, i] = u[b] < u[i] and v[b] < v[i]
        C[j, b] = u[j] < u[b] and v[j] < v[b]
        C[b, j] = u[b] < u[j] and v[b] < v[j]
    C[i, j] = u[i] < u[j] and v[i] < v[j]
    C[j, i] = u[j] < u[i] and v[j] < v[i]

    for a in range(N):
        if a == i or a == j:
            continue
        for b in range(N):
            if b == i or b == j or b == a:
                continue
            if C[a, b]:
                old_i_between = 1 if (old_col_i[a] and old_row_i[b]) else 0
                new_i_between = 1 if (C[a, i] and C[i, b]) else 0
                old_j_between = 1 if (old_col_j[a] and old_row_j[b]) else 0
                new_j_between = 1 if (C[a, j] and C[j, b]) else 0
                delta = (new_i_between - old_i_between) + (new_j_between - old_j_between)
                if delta != 0:
                    _bucket_dec(counts, sizes[a, b])
                    sizes[a, b] += delta
                    _bucket_inc(counts, sizes[a, b])

    for b in range(N):
        if b == i or b == j:
            continue
        if C[i, b]:
            s = 0
            for k in range(N):
                if C[i, k] and C[k, b]:
                    s += 1
            sizes[i, b] = s
            _bucket_inc(counts, s)
        if C[b, i]:
            s = 0
            for k in range(N):
                if C[b, k] and C[k, i]:
                    s += 1
            sizes[b, i] = s
            _bucket_inc(counts, s)
        if C[j, b]:
            s = 0
            for k in range(N):
                if C[j, k] and C[k, b]:
                    s += 1
            sizes[j, b] = s
            _bucket_inc(counts, s)
        if C[b, j]:
            s = 0
            for k in range(N):
                if C[b, k] and C[k, j]:
                    s += 1
            sizes[b, j] = s
            _bucket_inc(counts, s)
    if C[i, j]:
        s = 0
        for k in range(N):
            if C[i, k] and C[k, j]:
                s += 1
        sizes[i, j] = s
        _bucket_inc(counts, s)
    if C[j, i]:
        s = 0
        for k in range(N):
            if C[j, k] and C[k, i]:
                s += 1
        sizes[j, i] = s
        _bucket_inc(counts, s)


apply_swap_jit = njit(cache=True)(apply_swap_py)


@njit(cache=True)
def _action_from_counts_jit(N, counts, f2_table, eps):
    acc = 0.0
    for s in range(counts.shape[0]):
        acc += counts[s] * f2_table[s]
    return 4.0 * eps * (N - 2.0 * eps * acc)


@njit(cache=True)
def run_sweeps_jit(u, v, C, sizes, counts, N, beta, eps, f2_table, n_sweeps, burn_in,
                    measure_every, anneal_from, has_anneal, seed, max_measurements):
    np.random.seed(seed)
    S = _action_from_counts_jit(N, counts, f2_table, eps)
    n_accept = 0
    n_propose = 0
    actions = np.empty(max_measurements, dtype=np.float64)
    n_actions = 0

    for sweep in range(n_sweeps):
        if has_anneal and sweep < burn_in:
            frac = sweep / max(burn_in - 1, 1)
            beta_eff = anneal_from + (beta - anneal_from) * frac
        else:
            beta_eff = beta
        for _move in range(N):
            n_propose += 1
            which = np.random.randint(0, 2)
            i = np.random.randint(0, N)
            j = np.random.randint(0, N - 1)
            if j >= i:
                j += 1
            apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)
            S_new = _action_from_counts_jit(N, counts, f2_table, eps)
            dS = S_new - S
            accept = dS <= 0.0
            if not accept:
                if np.random.random() < np.exp(-beta_eff * dS):
                    accept = True
            if accept:
                S = S_new
                n_accept += 1
            else:
                apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)
        if sweep >= burn_in and (sweep - burn_in) % measure_every == 0 and n_actions < max_measurements:
            actions[n_actions] = S
            n_actions += 1

    return actions[:n_actions], n_accept, n_propose


@njit(cache=True)
def run_pt_jit(U, V, Call, Sizes, Counts, betas, N, K, eps, f2_table, n_sweeps, burn_in,
                measure_every, seed, max_measurements):
    """Parallel tempering (replica exchange) over K replicas, one per
    beta in `betas` (ascending), all sharing the same eps (and hence the
    same f2_table) -- PT tempers over beta at a fixed action
    normalization. Each replica does one local Metropolis sweep (N
    moves) at its own fixed beta, then every pair of beta-adjacent
    replicas attempts to swap their entire state (the standard
    replica-exchange acceptance rule: min(1, exp[(beta_k -
    beta_{k+1})(S_k - S_{k+1})])) -- every sweep, not batched. A replica
    whose configuration is stuck in a locally-favored basin at its own
    beta can escape it by trading places with a neighboring replica,
    which is what lets PT equilibrate the near-transition region far
    better than any single fixed-beta chain can.

    U, V: (K, N) int64 permutations, one row per replica (mutated in place).
    Call, Sizes, Counts: (K, N, N) / (K, N, N) / (K, N-1), one per replica.
    Returns (actions (K, n_measurements), swap_attempts (K-1,),
    swap_accepts (K-1,), accept_counts (K,), propose_counts (K,),
    round_trips (K,)).

    round_trips[p] counts how many full bottom-rung <-> top-rung round
    trips *physical* replica p (tracked by its original starting rung,
    through however many swaps relabel which rung it currently sits at)
    has completed -- the standard PT health diagnostic (e.g. Katzgraber
    et al.): swap acceptance rates between neighbors can look fine while
    replicas still never actually travel the full ladder, which is the
    thing that actually matters for crossing between phases. Zero round
    trips means PT is not doing its job here, whatever the local swap
    rates say.
    """
    np.random.seed(seed)
    actions = np.empty((K, max_measurements), dtype=np.float64)
    n_actions = 0
    swap_attempts = np.zeros(K - 1, dtype=np.int64)
    swap_accepts = np.zeros(K - 1, dtype=np.int64)
    accept_counts = np.zeros(K, dtype=np.int64)
    propose_counts = np.zeros(K, dtype=np.int64)

    # replica_label[k] = which physical replica (by original starting
    # rung index) currently occupies rung k; kept in sync with every
    # state swap below.
    replica_label = np.arange(K)
    # per physical replica: 0 = hasn't visited either extreme rung yet,
    # -1 = last visited the bottom rung, +1 = last visited the top rung.
    extreme_state = np.zeros(K, dtype=np.int64)
    half_trips = np.zeros(K, dtype=np.int64)
    extreme_state[replica_label[0]] = -1
    extreme_state[replica_label[K - 1]] = 1

    S = np.empty(K, dtype=np.float64)
    for k in range(K):
        S[k] = _action_from_counts_jit(N, Counts[k], f2_table, eps)

    for rnd in range(n_sweeps):
        for k in range(K):
            beta_k = betas[k]
            u = U[k]
            v = V[k]
            C = Call[k]
            sizes = Sizes[k]
            counts = Counts[k]
            for _move in range(N):
                propose_counts[k] += 1
                which = np.random.randint(0, 2)
                i = np.random.randint(0, N)
                j = np.random.randint(0, N - 1)
                if j >= i:
                    j += 1
                apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)
                S_new = _action_from_counts_jit(N, counts, f2_table, eps)
                dS = S_new - S[k]
                accept = dS <= 0.0
                if not accept:
                    if np.random.random() < np.exp(-beta_k * dS):
                        accept = True
                if accept:
                    S[k] = S_new
                    accept_counts[k] += 1
                else:
                    apply_swap_jit(u, v, C, sizes, counts, N, which, i, j)

        for k in range(K - 1):
            swap_attempts[k] += 1
            dbeta = betas[k] - betas[k + 1]
            dS_pair = S[k] - S[k + 1]
            arg = dbeta * dS_pair
            accept_swap = arg >= 0.0
            if not accept_swap:
                if np.random.random() < np.exp(arg):
                    accept_swap = True
            if accept_swap:
                for a in range(N):
                    U[k, a], U[k + 1, a] = U[k + 1, a], U[k, a]
                    V[k, a], V[k + 1, a] = V[k + 1, a], V[k, a]
                for a in range(N):
                    for b in range(N):
                        Call[k, a, b], Call[k + 1, a, b] = Call[k + 1, a, b], Call[k, a, b]
                        Sizes[k, a, b], Sizes[k + 1, a, b] = Sizes[k + 1, a, b], Sizes[k, a, b]
                for c in range(Counts.shape[1]):
                    Counts[k, c], Counts[k + 1, c] = Counts[k + 1, c], Counts[k, c]
                S[k], S[k + 1] = S[k + 1], S[k]
                swap_accepts[k] += 1
                replica_label[k], replica_label[k + 1] = replica_label[k + 1], replica_label[k]

        p_bottom = replica_label[0]
        if extreme_state[p_bottom] == 1:
            half_trips[p_bottom] += 1
            extreme_state[p_bottom] = -1
        elif extreme_state[p_bottom] == 0:
            extreme_state[p_bottom] = -1
        p_top = replica_label[K - 1]
        if extreme_state[p_top] == -1:
            half_trips[p_top] += 1
            extreme_state[p_top] = 1
        elif extreme_state[p_top] == 0:
            extreme_state[p_top] = 1

        if rnd >= burn_in and (rnd - burn_in) % measure_every == 0 and n_actions < max_measurements:
            for k in range(K):
                actions[k, n_actions] = S[k]
            n_actions += 1

    round_trips = half_trips // 2
    return actions[:, :n_actions], swap_attempts, swap_accepts, accept_counts, propose_counts, round_trips
