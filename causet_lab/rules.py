"""Pluggable growth rules for sequential causal set generation.

A rule is any callable

    rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray

where C is the n x n boolean order-matrix of the causal set *before* the
new element is added, and the return value is a boolean vector of length n
marking which of the n existing elements lie in the new element's DIRECT
past (before transitive closure -- the growth framework in generators.py
takes care of closing it).

A physically acceptable rule, in the spirit of Rideout-Sorkin discrete
general covariance, must depend only on the causal structure encoded in C
-- never on the arbitrary integer labels of its elements. Concretely: if
you relabel (permute) C with any order-automorphism-respecting permutation
and apply the same permutation to any auxiliary state the rule keeps, the
*distribution* over outcomes must be unchanged. A rule that singles out
"element 7" or bases a decision on raw index order rather than order-
theoretic structure (pasts, ranks, heights, antichains, ...) is not label
invariant and is not physically sensible, even if it runs without error.
"""
from __future__ import annotations

import numpy as np


def transitive_percolation(p: float):
    """Rideout-Sorkin transitive percolation.

    Each existing element is placed in the new element's direct past
    independently with probability p. This is the simplest growth rule
    satisfying discrete general covariance: the decision for each existing
    element depends on nothing but a fresh coin flip, so it cannot depend
    on labeling at all.
    """
    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        return rng.random(n) < p

    rule.__name__ = "transitive_percolation"
    rule.params = {"p": p}
    return rule


# ---------------------------------------------------------------------------
# Parent-selection rules.
#
# Earlier versions of R1-R4 and R6 picked parents only from the *maximal*
# elements of C (the frontier). That turned out to be fatally self-defeating:
# growth always starts from a single seed element, so the frontier starts at
# width 1; any rule that can fully consume whatever frontier is currently
# available (which "pick k, or all of them if fewer than k exist" always
# does when the frontier is small) collapses the frontier straight back to
# width 1 and can never recover. Every one of those rules degenerated into a
# plain chain, deterministically, for every k. See the writeup in
# results/rules_report.md for the full trace.
#
# The fix: these rules now choose parents from *all* existing elements, not
# just the maximal ones. The frontier's width is then an emergent outcome of
# each step rather than something the selection rule can single-handedly
# destroy: a step whose chosen parents include no current maximal element
# branches off below the tip and *widens* the frontier by one; a step that
# touches at least one maximal element *deepens* it instead. grow(...,
# track_branching=True) records the widen/deepen fraction directly.
# ---------------------------------------------------------------------------


def _maximal_elements(C: np.ndarray) -> np.ndarray:
    """Indices of elements with nothing in their future."""
    if C.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(~C.any(axis=1))


def _incomparable_to(C: np.ndarray, x: int) -> np.ndarray:
    """Indices of existing elements spacelike-separated from x: neither in
    its past nor its future, excluding x itself."""
    comparable = C[x, :] | C[:, x]
    incomparable = ~comparable
    incomparable[x] = False
    return np.flatnonzero(incomparable)


class _PastSizeCache:
    """Incrementally tracks |past(x)| for every existing element across
    repeated rule(C, rng) calls within one grow() run.

    |past(x)| never changes once x is added (existing elements are never
    revisited by the growth framework), so recomputing it for everyone on
    every step -- which naive code tends to do, e.g. C.sum(axis=0) is an
    O(n^2) full-matrix pass -- is wasted work. Only the single
    most-recently-added element's past size is actually new each call.
    Caching it turns an O(n) or worse per-step cost into O(1) amortized,
    and an O(N^2) or O(N^3) total cost into O(N^2) at worst (matching the
    growth framework's own baseline cost). Resets whenever n == 1 is seen,
    so the same rule object can safely be reused across repeated grow()
    calls.
    """

    def __init__(self):
        self._sizes = np.zeros(16, dtype=np.int64)
        self._filled = 0

    def get(self, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        if n == 1:
            self._sizes = np.zeros(16, dtype=np.int64)
            self._filled = 0
        if n > self._sizes.shape[0]:
            grown = np.zeros(max(n, self._sizes.shape[0] * 2), dtype=np.int64)
            grown[: self._sizes.shape[0]] = self._sizes
            self._sizes = grown
        for i in range(self._filled, n):
            self._sizes[i] = int(C[:i, i].sum()) if i > 0 else 0
        self._filled = n
        return self._sizes[:n]


class _FutureSizeCache:
    """Incrementally tracks |future(x)| for every existing element.

    Unlike past size, future size keeps changing after x is added --
    every later element whose past includes x adds one more to x's
    future. But exactly one new element is fully finalized between two
    consecutive rule(C, rng) calls (the one decided by the previous
    call), and its complete past is readable as a column of the *current*
    C the moment it becomes available. So each call only ever needs to
    apply one new element's contribution (incrementing future[i] for
    every i in that element's past) instead of rescanning the matrix.
    """

    def __init__(self):
        self._future = np.zeros(16, dtype=np.int64)
        self._applied_upto = 0  # contributions from indices < this have been applied

    def get(self, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        if n == 1:
            self._future = np.zeros(16, dtype=np.int64)
            self._applied_upto = 0
        if n > self._future.shape[0]:
            grown = np.zeros(max(n, self._future.shape[0] * 2), dtype=np.int64)
            grown[: self._future.shape[0]] = self._future
            self._future = grown
        while self._applied_upto < n - 1:
            i = self._applied_upto
            past_i = C[:i, i] if i > 0 else np.zeros(0, dtype=bool)
            self._future[: len(past_i)][past_i] += 1
            self._applied_upto += 1
        return self._future[:n]


class _ColumnMirrorCache:
    """Maintains a persistent Fortran-order (column-major) copy of C,
    incrementally extended one column at a time, for fast multi-column
    fancy indexing.

    C itself is row-major (numpy's default), so `C[:, candidates]` for a
    few hundred candidate columns means a strided, cache-hostile gather --
    measured at ~8x slower than the equivalent slice on a column-major
    array of the same shape, and this is the single largest cost in
    growth once N reaches a few thousand (it dwarfs the actual matmul
    FLOPs, which are trivial by comparison). Since each column of C is
    written exactly once (when that element is added) and never touched
    again, keeping the mirror current only costs a single O(n) column
    copy per step -- the same one-shot-per-element pattern as
    _PastSizeCache/_FutureSizeCache. Stored as bool (not pre-cast to
    float32) to keep the extra memory to roughly 1x C instead of 4x.
    """

    def __init__(self):
        self._mirror = np.zeros((16, 16), dtype=bool, order="F")
        self._filled = 0

    def get(self, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        if n == 1:
            self._mirror = np.zeros((16, 16), dtype=bool, order="F")
            self._filled = 0
        if n > self._mirror.shape[0]:
            new_size = max(n, self._mirror.shape[0] * 2)
            grown = np.zeros((new_size, new_size), dtype=bool, order="F")
            grown[: self._mirror.shape[0], : self._mirror.shape[0]] = self._mirror
            self._mirror = grown
        while self._filled < n - 1:
            i = self._filled
            self._mirror[:i, i] = C[:i, i]
            self._filled += 1
        return self._mirror[:n, :n]


_NEAREST_SAMPLE_CAP = 200   # default candidate-sample size for fixed, small k
_NEAREST_SAMPLE_MAX = 1500  # hard ceiling on candidate-sample size regardless of k


def _nearest_among(C: np.ndarray, Cf: np.ndarray, x: int, candidates: np.ndarray, k: int, rng,
                    past_sizes: np.ndarray, sample_size: int = _NEAREST_SAMPLE_CAP):
    """k candidates nearest to x by frontier distance
    |past(x) symmetric-difference past(y)|, computed as
    |past(x)| + |past(y)| - 2|past(x) & past(y)|. ``Cf`` must be the
    column-major mirror of C from _ColumnMirrorCache (column gathers off
    it instead of C directly are the single biggest growth-speed win at
    large N -- see that class's docstring). ``past_sizes`` must be
    the full (n,) array of |past(i)| for every existing element (see
    _PastSizeCache) so the |past(y)| term is a cheap lookup instead of a
    fresh reduction. The overlap term still needs one matrix pass; a BLAS
    matmul on a float32 cast beats a bitwise-AND-then-reduce here (the
    latter allocates a full (n, candidates) intermediate with no SIMD
    fusion -- empirically ~30% slower).

    ``sample_size`` is the number of candidates to subsample down to
    before ranking, when there are more candidates than that available
    (default _NEAREST_SAMPLE_CAP, fine for small fixed k). Callers with a
    large or data-dependent k (e.g. local_parents_adaptive) should pass a
    larger sample_size -- at least 3x k is a reasonable rule of thumb --
    so k is never silently truncated to fewer genuinely-nearest neighbors
    than requested. sample_size is itself clamped to _NEAREST_SAMPLE_MAX
    to keep worst-case per-step cost bounded; returns whether that ceiling
    actually had to truncate the request (cap_hit) so callers can report
    how often it happened instead of failing silently.

    Returns (chosen_indices, cap_hit).
    """
    if len(candidates) <= k:
        return candidates, False
    effective_size = min(sample_size, _NEAREST_SAMPLE_MAX)
    cap_hit = sample_size > _NEAREST_SAMPLE_MAX and len(candidates) > _NEAREST_SAMPLE_MAX
    if len(candidates) > effective_size:
        candidates = rng.choice(candidates, size=effective_size, replace=False)
    Cc = Cf[:, candidates].astype(np.float32)
    past_x = Cf[:, x].astype(np.float32)
    size_x = past_sizes[x]
    sizes = past_sizes[candidates]
    overlap = past_x @ Cc
    sym_diff = size_x + sizes - 2.0 * overlap
    order = np.argsort(sym_diff, kind="stable")
    return candidates[order[:k]], cap_hit


def random_parents(k: int):
    """R1: link to k elements chosen uniformly at random from *all*
    existing elements (all of them if fewer than k exist). Growth isn't
    restricted to "the now" at all -- this is the nonlocal baseline.
    """
    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        size = min(k, n)
        chosen = rng.choice(n, size=size, replace=False)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "random_parents"
    rule.params = {"k": k}
    return rule


def local_parents(k: int):
    """R2: pick one existing element x uniformly at random, then link to x
    and its k-1 nearest elements *incomparable* to x (spacelike-separated:
    neither in its past nor future), by frontier distance. The main
    hypothesis of this batch: a new event connecting to one event and its
    spacelike neighbors should be enough to produce a stable dimension.
    """
    cache = _PastSizeCache()
    mirror = _ColumnMirrorCache()

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        past_sizes = cache.get(C)
        Cf = mirror.get(C)
        x = int(rng.integers(n))
        if k <= 1:
            chosen = np.array([x], dtype=np.int64)
        else:
            incomparable = _incomparable_to(C, x)
            if len(incomparable) == 0:
                chosen = np.array([x], dtype=np.int64)
            else:
                others, _cap_hit = _nearest_among(C, Cf, x, incomparable, k - 1, rng, past_sizes)
                chosen = np.concatenate([[x], others]).astype(np.int64)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "local_parents"
    rule.params = {"k": k}
    return rule


def local_parents_adaptive(c: float, beta: float):
    """local_parents, but the number of parents grows with the picked
    element's own past size instead of being a fixed k: k = max(1,
    round(c * (1 + |past(x)|)^beta)). This lets valence grow as the
    universe grows using only local, label-free causal information (no
    total size N, no target dimension) -- the thing every fixed-k rule in
    this batch structurally cannot do.

    Because k is now data-dependent and can grow large for well-connected
    x, the nearest-neighbor search is asked for a proportionally larger
    candidate sample (at least 3x k, capped at _NEAREST_SAMPLE_MAX) rather
    than silently truncating to whatever the default sample would hold.
    ``rule.state`` exposes "cap_hits" and "total_calls" after a grow()
    run, for reporting how often that hard ceiling actually bound.
    """
    cache = _PastSizeCache()
    mirror = _ColumnMirrorCache()
    state = {"cap_hits": 0, "total_calls": 0}

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        if n == 1:
            state["cap_hits"] = 0
            state["total_calls"] = 0
        past_sizes = cache.get(C)
        Cf = mirror.get(C)
        x = int(rng.integers(n))
        k = max(1, round(c * (1.0 + past_sizes[x]) ** beta))
        state["total_calls"] += 1
        if k <= 1:
            chosen = np.array([x], dtype=np.int64)
        else:
            incomparable = _incomparable_to(C, x)
            if len(incomparable) == 0:
                chosen = np.array([x], dtype=np.int64)
            else:
                desired_sample = max(_NEAREST_SAMPLE_CAP, 3 * (k - 1))
                others, cap_hit = _nearest_among(C, Cf, x, incomparable, k - 1, rng, past_sizes, desired_sample)
                if cap_hit:
                    state["cap_hits"] += 1
                chosen = np.concatenate([[x], others]).astype(np.int64)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "local_parents_adaptive"
    rule.params = {"c": c, "beta": beta}
    rule.state = state
    return rule


def preferential_parents(k: int):
    """R3: choose k elements from all existing elements, without
    replacement, with probability proportional to 1 + |past(x)|.
    Rich-get-richer; tests what nonlocal hubs do to the emergent structure.

    Uses _PastSizeCache so |past(x)| is amortized O(1) per step instead of
    an O(n^2) C.sum(axis=0) full-matrix pass -- see that class's docstring.
    """
    cache = _PastSizeCache()

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        past_sizes = cache.get(C)
        weights = 1.0 + past_sizes.astype(np.float64)
        weights /= weights.sum()
        size = min(k, n)
        chosen = rng.choice(n, size=size, replace=False, p=weights)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "preferential_parents"
    rule.params = {"k": k}
    return rule


def frontier_weighted(a: float, k: int):
    """Pick x with probability proportional to 1 / (1 + |future(x)|)^a --
    events with little future are near the "present" -- then link to x
    and its k-1 nearest incomparable neighbors exactly as in
    local_parents. Tests whether preferring the present, measured purely
    from order (no labels, no frontier/maximal-element bookkeeping that
    caused the original R1-R4/R6 collapse), balances frontier width
    against depth on its own.

    Uses _FutureSizeCache so |future(x)| is amortized O(1) per step
    instead of a fresh O(n) or worse scan every call.
    """
    future_cache = _FutureSizeCache()
    past_cache = _PastSizeCache()
    mirror = _ColumnMirrorCache()

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        future_sizes = future_cache.get(C)
        past_sizes = past_cache.get(C)
        Cf = mirror.get(C)
        weights = 1.0 / (1.0 + future_sizes.astype(np.float64)) ** a
        weights /= weights.sum()
        x = int(rng.choice(n, p=weights))
        if k <= 1:
            chosen = np.array([x], dtype=np.int64)
        else:
            incomparable = _incomparable_to(C, x)
            if len(incomparable) == 0:
                chosen = np.array([x], dtype=np.int64)
            else:
                others, _cap_hit = _nearest_among(C, Cf, x, incomparable, k - 1, rng, past_sizes)
                chosen = np.concatenate([[x], others]).astype(np.int64)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "frontier_weighted"
    rule.params = {"a": a, "k": k}
    return rule


def bounded_valence(k: int, vmax: int):
    """R4: like random_parents(k), but an element that has already been
    chosen as a parent vmax times is excluded as a target. Because parents
    are now drawn from *all* existing elements (not just the frontier, an
    element stays eligible for future selection after being linked once),
    the cap actually matters here. Tests whether a fixed per-element
    neighbor count, like a lattice, can look Lorentzian.

    Valence is tracked in a closure using a dynamically-growing numpy
    array (resized by doubling) rather than a Python dict, so filtering
    eligible elements each step is a single vectorized comparison instead
    of an O(n) Python loop. State resets whenever n == 1 is seen, which
    happens exactly once per grow() call, so the same rule object can
    safely be reused (or not) across repeated growth runs.
    """
    state = {"valence": np.zeros(16, dtype=np.int32)}

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        if n == 1:
            state["valence"] = np.zeros(16, dtype=np.int32)
        valence = state["valence"]
        if n > valence.shape[0]:
            grown = np.zeros(max(n, valence.shape[0] * 2), dtype=np.int32)
            grown[: valence.shape[0]] = valence
            valence = grown
            state["valence"] = valence
        available = np.flatnonzero(valence[:n] < vmax)
        if available.size == 0:
            chosen = np.empty(0, dtype=np.int64)
        else:
            size = min(k, available.size)
            chosen = rng.choice(available, size=size, replace=False)
        valence[chosen] += 1
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "bounded_valence"
    rule.params = {"k": k, "vmax": vmax}
    return rule


def recency_cheat(L: int, p: float):
    """R5 [ILLEGAL CONTROL]: link to each of the L most recently added
    elements (by raw label index) with probability p. This reads the
    order the computer happened to add elements in -- a hidden global
    clock that has nothing to do with the causal structure. It directly
    violates the one rule every other function in this file respects:
    never depend on labeling. Kept here only as a deliberately-illegal
    control to show what "cheating" with labels buys you, if anything.
    """
    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        lo = max(0, n - L)
        window = np.arange(lo, n)
        mask = rng.random(len(window)) < p
        out = np.zeros(n, dtype=bool)
        out[window[mask]] = True
        return out

    rule.__name__ = "recency_cheat"
    rule.params = {"L": L, "p": p}
    rule.illegal = True
    return rule


def width_forced(k: int, d_target: float, c: float):
    """R6 [CHEAT CONTROL]: like local_parents(k), but when the current
    frontier width is below c * n**((d_target - 1) / d_target) (n =
    current size), widening is *forced*: pick x only from the non-maximal
    elements and link to x alone, guaranteeing the step touches no current
    maximal element and so cannot shrink the frontier. Once the frontier
    catches up to the target curve, it behaves exactly like local_parents.
    The constant c is calibrated externally from real sprinkles of
    d_target so the forced frontier sits near what a genuine
    d_target-dimensional sprinkle would have -- which is itself cheating,
    since d is being put in by hand rather than emerging. Tests whether
    getting the frontier's *shape* right is enough for the rest of the
    structure to follow.
    """
    cache = _PastSizeCache()
    mirror = _ColumnMirrorCache()

    def rule(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = C.shape[0]
        past_sizes = cache.get(C)
        Cf = mirror.get(C)
        width = len(_maximal_elements(C))
        target_width = c * (n ** ((d_target - 1.0) / d_target)) if n > 0 else 0.0

        if width < target_width:
            non_maximal = np.flatnonzero(C.any(axis=1))
            if len(non_maximal) > 0:
                x = int(non_maximal[rng.integers(len(non_maximal))])
                out = np.zeros(n, dtype=bool)
                out[x] = True
                return out
            # No non-maximal elements exist yet (very early growth): fall
            # through to ordinary local_parents behavior for this step.

        x = int(rng.integers(n))
        if k <= 1:
            chosen = np.array([x], dtype=np.int64)
        else:
            incomparable = _incomparable_to(C, x)
            if len(incomparable) == 0:
                chosen = np.array([x], dtype=np.int64)
            else:
                others, _cap_hit = _nearest_among(C, Cf, x, incomparable, k - 1, rng, past_sizes)
                chosen = np.concatenate([[x], others]).astype(np.int64)
        out = np.zeros(n, dtype=bool)
        out[chosen] = True
        return out

    rule.__name__ = "width_forced"
    rule.params = {"k": k, "d_target": d_target, "c": c}
    rule.cheat = True
    return rule


# Registry mapping CLI-friendly names to rule factories. A factory takes
# whatever keyword parameters it needs and returns a rule(C, rng) callable.
# Add new rules here to make them available via `run.py rule --name ...`.
RULE_FACTORIES = {
    "transitive_percolation": transitive_percolation,
    "random_parents": random_parents,
    "local_parents": local_parents,
    "local_parents_adaptive": local_parents_adaptive,
    "preferential_parents": preferential_parents,
    "frontier_weighted": frontier_weighted,
    "bounded_valence": bounded_valence,
    "recency_cheat": recency_cheat,
    "width_forced": width_forced,
}
