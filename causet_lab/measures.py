"""Dimension and manifold-likeness measures for causal sets.

All functions here take a raw boolean order-matrix C (not a CausalSet
wrapper) so they can be applied equally to a whole causal set or to a
sub-matrix extracted as an interval.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from .causet import CausalSet


def ordering_fraction(C: np.ndarray) -> float:
    """f = (number of related pairs) / (N choose 2)."""
    C = np.asarray(C, dtype=bool)
    N = C.shape[0]
    if N < 2:
        return float("nan")
    relations = int(C.sum())
    max_pairs = N * (N - 1) // 2
    return relations / max_pairs


def _myrheim_meyer_f(d: float) -> float:
    """Expected ordering fraction for a causal-set interval sprinkled into
    d-dimensional Minkowski space:

        f(d) = Gamma(d+1) * Gamma(d/2) / (2 * Gamma(3d/2))

    Computed in log-space for numerical stability (Gamma(3d/2) blows up
    fast). f(2) = 0.5, f(4) = 0.1.
    """
    log_f = gammaln(d + 1) + gammaln(d / 2) - np.log(2.0) - gammaln(1.5 * d)
    return float(np.exp(log_f))


def myrheim_meyer_dimension(C: np.ndarray, d_range=(1.0, 10.0)) -> float:
    """Invert the Myrheim-Meyer ordering-fraction formula numerically
    (scipy.brentq) to estimate the dimension of an interval causal set.

    f(d) is monotonically decreasing over d_range, so a bisection root
    find on f(d) - f_obs is well posed. Observed fractions outside the
    range covered by d_range are clipped to the nearest endpoint.
    """
    f_obs = ordering_fraction(C)
    if np.isnan(f_obs):
        return float("nan")
    lo, hi = d_range
    f_lo, f_hi = _myrheim_meyer_f(lo), _myrheim_meyer_f(hi)
    if f_obs >= f_lo:
        return lo
    if f_obs <= f_hi:
        return hi

    def g(d: float) -> float:
        return _myrheim_meyer_f(d) - f_obs

    return float(brentq(g, lo, hi))


def midpoint_dimension(C: np.ndarray) -> float:
    """Independent dimension estimator for an interval causal set.

    Treat C as the open interval (a, b): for every element c, I(a, c) is
    the set of elements preceding c within C, and I(c, b) is the set of
    elements c precedes within C (a precedes everything and everything
    precedes b, so those two counts are exactly the column-sum and
    row-sum of c within C). Find the element c maximizing the smaller of
    the two counts, then estimate d ~ log2(N / that smaller count).
    """
    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    if n < 3:
        return float("nan")
    down = C.sum(axis=0)  # |I(a, c)| for each c
    up = C.sum(axis=1)  # |I(c, b)| for each c
    m = np.minimum(down, up)
    best = int(m.max())
    if best <= 0:
        return float("nan")
    return float(np.log2(n / best))


def sample_intervals(
    C: np.ndarray,
    n_samples: int,
    min_size: int,
    max_size: int,
    seed: int,
    max_attempts: int = None,
) -> List[np.ndarray]:
    """Sample random intervals from a causal set.

    Repeatedly pick a random related pair (a, b) with C[a, b] = True,
    extract the sub-causal-set of elements strictly between them, and keep
    it if its size falls in [min_size, max_size]. Works the same way on
    sprinkled sets and on grown (non-interval) universes, which is the
    point: grown universes are not themselves intervals, so dimension must
    be measured on intervals sampled from inside them, and applying the
    same sampling to sprinkles keeps the comparison fair.
    """
    rng = np.random.default_rng(seed)
    C = np.asarray(C, dtype=bool)
    ii, jj = np.nonzero(C)
    if len(ii) == 0:
        return []
    if max_attempts is None:
        max_attempts = max(n_samples * 50, 2000)

    results: List[np.ndarray] = []
    attempts = 0
    pool = rng.permutation(len(ii))
    pos = 0
    while len(results) < n_samples and attempts < max_attempts:
        if pos >= len(pool):
            pool = rng.permutation(len(ii))
            pos = 0
        k = pool[pos]
        pos += 1
        attempts += 1
        a, b = int(ii[k]), int(jj[k])
        interior = C[a, :] & C[:, b]
        size = int(interior.sum())
        if min_size <= size <= max_size:
            idx = np.flatnonzero(interior)
            results.append(C[np.ix_(idx, idx)])
    return results


def interval_abundances(C: np.ndarray, kmax: int = 10) -> np.ndarray:
    """Distribution of interval sizes over all related pairs.

    For every related pair (i, j), the size of the open interval between
    them equals the number of two-step paths i -> k -> j, which is just
    (C @ C)[i, j] for boolean/0-1 C (transitivity guarantees this can only
    be nonzero where C[i, j] is already True). Returns an array of length
    kmax + 1 with counts[k] = (# related pairs with interval size k) /
    (total # related pairs). Manifold-like causal sets have a
    characteristic profile that can be compared against sprinkled sets of
    known dimension.
    """
    C = np.asarray(C, dtype=bool)
    total_relations = int(C.sum())
    if total_relations == 0:
        return np.full(kmax + 1, np.nan)
    Cf = C.astype(np.float32)
    sizes = np.rint(Cf @ Cf).astype(np.int64)
    related_sizes = sizes[C]
    bc = np.bincount(related_sizes, minlength=kmax + 1)
    counts = bc[: kmax + 1].astype(np.float64)
    return counts / total_relations


def height(C: np.ndarray) -> int:
    """Length (number of elements) of the longest chain."""
    return CausalSet(np.asarray(C, dtype=bool)).height()


def frontier_width(C: np.ndarray) -> int:
    """Number of maximal elements (elements with nothing in their future)
    in C. For a grown causal set this is only the *final* frontier width;
    the frontier's history during growth has to be tracked while growing
    (see generators.grow(..., track_frontier=True)) since it cannot be
    recovered from the finished matrix. For a sprinkle, which is static,
    this single number is the natural reference value.
    """
    C = np.asarray(C, dtype=bool)
    if C.shape[0] == 0:
        return 0
    return int((~C.any(axis=1)).sum())


def link_valence(C: np.ndarray) -> float:
    """Mean number of links (covering relations) touching each element.

    Each link touches two elements (its source and target), so the mean
    degree in the link graph is 2 * (# links) / N. A fixed, N-independent
    valence looks like a lattice; genuine manifold-like causal sets need
    valence that grows with N (Myrheim-Meyer links scale with N^(?) --
    there is no universal fixed "number of neighbors").
    """
    C = np.asarray(C, dtype=bool)
    N = C.shape[0]
    if N == 0:
        return float("nan")
    n_links = int(CausalSet(C).links().sum())
    return 2.0 * n_links / N


def height_exponent(Ns, heights) -> float:
    """Fit height ~ N^alpha across a range of N via a log-log linear fit.
    A d-dimensional sprinkle gives alpha ~= 1/d.
    """
    Ns = np.asarray(Ns, dtype=float)
    heights = np.asarray(heights, dtype=float)
    mask = (Ns > 0) & (heights > 0) & np.isfinite(Ns) & np.isfinite(heights)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(Ns[mask]), np.log(heights[mask]), 1)
    return float(slope)


def dimension_drift(Ns, dims) -> float:
    """Slope of a dimension estimate vs log N: how much the estimate moves
    as the universe grows. Near zero means the estimate has stabilized,
    which is what a fixed-dimension manifold-like causal set should do.
    """
    Ns = np.asarray(Ns, dtype=float)
    dims = np.asarray(dims, dtype=float)
    mask = (Ns > 0) & np.isfinite(Ns) & np.isfinite(dims)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(Ns[mask]), dims[mask], 1)
    return float(slope)


def abundance_distance(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """L1 distance between two interval-abundance profiles (same kmax)."""
    a = np.asarray(profile_a, dtype=float)
    b = np.asarray(profile_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.abs(a[mask] - b[mask]).sum())
