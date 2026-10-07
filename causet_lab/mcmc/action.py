"""The 2D Benincasa-Dowker causal set action, plain and smeared.

Source: D.M.H. Benincasa and F. Dowker, "The Volume of a Causal Set",
Phys. Rev. Lett. 104, 181301 (2010), arXiv:1001.2725 -- introduces the
general-d BD action as a discrete analogue of the Einstein-Hilbert
action, built from counts of small intervals.

For a causal set of N elements, let N_k be the number of related pairs
(i, j) [i.e. i strictly precedes j] such that exactly k elements lie in
the open interval between them (N_0 = number of links, i.e. covering
relations: pairs with nothing between them). The plain (non-smeared,
local) 2D BD action truncated at k=2 is:

    S = 2 * (N - 2*N0 + 4*N1 - 2*N2)

This plain action is NOT what the published 2D causal set MC studies
use as the Monte Carlo weight, and its beta values are not comparable
to theirs (see bd_action_2d_smeared below) -- it is kept here only as
a simple reference/legacy quantity.

The production action for this project is the *smeared* 2D BD action
with non-locality parameter epsilon in (0, 1), used in S. Surya,
"Evidence for the continuum in 2D causal set quantum gravity", CQG 29
132001 (2012), arXiv:1110.6244, and precisely calibrated (including the
N-scaling of the critical coupling beta_c) in L. Glaser, D. O'Connor,
S. Surya, "Finite Size Scaling in 2d Causal Set Quantum Gravity", CQG
35 045006 (2018), arXiv:1706.06432. Confirmed by fetching the latter's
ar5iv HTML rendering: the formula and normalization below match its
Eq. for S(C, epsilon) exactly, and it reports

    beta_c(N, eps) = b(eps)/N + c(eps)/N^2 + O(1/N^3)
    b(eps) = 1.66(+/-0.03) / eps^2
    c(eps) = 4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2

(fit over N = 20..90, eps = 0.1..0.5), and finds the transition is
first-order (double-peaked action histograms that sharpen with N) with
phases named Pi_- (continuum, beta < beta_c, dimension ~ 2) and Pi_+
(crystalline/non-continuum, beta > beta_c). It also reports the
asymptotic (N -> infinity) scaling regime is only reached for N >~ 65,
so the subleading c(eps)/N^2 term is NOT negligible at the N = 30..60
used in this project's scan -- both terms are used wherever this
project compares a located beta_c to the published formula.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

KMAX_ACTION = 2

BETA_C_B_COEFF = 1.66  # +/- 0.03, Glaser-O'Connor-Surya 2018
BETA_C_C_COEFF_CUBE = 4.09  # +/- 0.50, coefficient of 1/eps^3 in c(eps)
BETA_C_C_COEFF_SQUARE = -27.77  # +/- 2.45, coefficient of 1/eps^2 in c(eps)


def beta_c_glaser_2018(N: float, eps: float) -> float:
    """Published leading + first-subleading prediction for the critical
    coupling, beta_c(N, eps) = b(eps)/N + c(eps)/N^2, from Glaser,
    O'Connor, Surya (2018), arXiv:1706.06432 (see module docstring).
    """
    b = BETA_C_B_COEFF / eps ** 2
    c = BETA_C_C_COEFF_CUBE / eps ** 3 + BETA_C_C_COEFF_SQUARE / eps ** 2
    return b / N + c / N ** 2


def f2_smear_table(nmax: int, eps: float) -> np.ndarray:
    """Table of f(n, eps) for n = 0..nmax, the smeared-action kernel

        f(n, eps) = (1-eps)^n * (1 - 2*eps*n/(1-eps)
                     + eps^2 * n*(n-1) / (2*(1-eps)^2))

    computed via an algebraically equivalent, division-free expansion
    (f(n,eps) = (1-eps)^n - 2*n*eps*(1-eps)^(n-1)
                + n*(n-1)*eps^2/2*(1-eps)^(n-2))
    so it stays well-defined even at eps -> 1 (where the as-written
    formula has a removable 0/0 singularity for n >= 1) -- this is what
    lets bd_action_2d_smeared(C, eps=1.0) be tested directly against the
    plain action's 4*(N-2*N0+4*N1-2*N2) limit (see module docstring and
    tests/test_mcmc.py) without numerical cancellation error.
    """
    eps = float(eps)
    table = np.empty(nmax + 1, dtype=np.float64)
    for n in range(nmax + 1):
        if n == 0:
            table[n] = 1.0
        elif n == 1:
            table[n] = 1.0 - 3.0 * eps
        else:
            table[n] = (
                (1.0 - eps) ** n
                - 2.0 * n * eps * (1.0 - eps) ** (n - 1)
                + n * (n - 1) * eps ** 2 / 2.0 * (1.0 - eps) ** (n - 2)
            )
    return table


def interval_size_counts(C: np.ndarray, kmax: int = KMAX_ACTION) -> np.ndarray:
    """N_k = count of related pairs (i, j) with exactly k elements
    strictly between them, for k = 0..kmax.

    Relies on C already being transitively closed -- true by
    construction for 2D orders (see orders.py docstring) and for
    sprinkles, so no transitive_closure() call is needed here. The
    interval size for a related pair (i, j) is the number of two-step
    paths i -> m -> j, i.e. (C @ C)[i, j]: the same trick used by
    measures.interval_abundances elsewhere in this project.
    """
    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    if n == 0:
        return np.zeros(kmax + 1, dtype=np.int64)
    Cf = C.astype(np.float32)
    sizes = np.rint(Cf @ Cf).astype(np.int64)
    related_sizes = sizes[C]
    if related_sizes.size == 0:
        return np.zeros(kmax + 1, dtype=np.int64)
    counts = np.bincount(related_sizes, minlength=kmax + 1)[: kmax + 1]
    return counts.astype(np.int64)


def bd_action_2d(C: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (S, Nk) for the plain (legacy, non-smeared) 2D BD action.
    Nk has length KMAX_ACTION + 1 = 3, i.e. [N0, N1, N2]. See module
    docstring for the formula and why bd_action_2d_smeared is used for
    all production sampling instead.
    """
    C = np.asarray(C, dtype=bool)
    N = C.shape[0]
    Nk = interval_size_counts(C, kmax=KMAX_ACTION)
    N0, N1, N2 = int(Nk[0]), int(Nk[1]), int(Nk[2])
    S = 2.0 * (N - 2 * N0 + 4 * N1 - 2 * N2)
    return S, Nk


def interval_size_counts_full(C: np.ndarray) -> np.ndarray:
    """Full histogram of N_n for n = 0..N-2 (every possible interval
    size for a related pair in an N-element order), needed by the
    smeared action's sum over all n (not just n <= 2). Same (C @ C)
    trick as interval_size_counts, just not truncated.
    """
    C = np.asarray(C, dtype=bool)
    n = C.shape[0]
    nmax = max(n - 2, 0)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    Cf = C.astype(np.float32)
    sizes = np.rint(Cf @ Cf).astype(np.int64)
    related_sizes = sizes[C]
    if related_sizes.size == 0:
        return np.zeros(nmax + 1, dtype=np.int64)
    counts = np.bincount(related_sizes, minlength=nmax + 1)[: nmax + 1]
    return counts.astype(np.int64)


def bd_action_2d_smeared(C: np.ndarray, eps: float) -> Tuple[float, np.ndarray]:
    """Return (S, Nk_full) for the smeared 2D BD action (see module
    docstring for the formula, its source, and the eps -> 1 limit that
    recovers the plain action -- tested in tests/test_mcmc.py).
    """
    C = np.asarray(C, dtype=bool)
    N = C.shape[0]
    Nk_full = interval_size_counts_full(C)
    if N < 2:
        return 4.0 * eps * N, Nk_full
    f2_table = f2_smear_table(N - 2, eps)
    S = 4.0 * eps * (N - 2.0 * eps * float(np.dot(Nk_full, f2_table)))
    return S, Nk_full
