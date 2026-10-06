"""Core causal set representation.

Convention: a causal set is an N x N boolean numpy array C where
``C[i, j] == True`` means element i strictly precedes element j (written
i -< j). The relation encoded by C is always a strict partial order:
irreflexive, antisymmetric and transitive.

Every generator in this package produces a *natural labeling*: whenever
``C[i, j]`` is True, the plain integer index satisfies i < j. Equivalently,
C is always strictly upper triangular. This single invariant is what lets
every algorithm here be vectorized instead of requiring a general
topological sort: processing indices in increasing order automatically
respects the causal order.
"""
from __future__ import annotations

import numpy as np


class CausalSet:
    """A finite causal set given by its precedence matrix."""

    __slots__ = ("C", "N")

    def __init__(self, C: np.ndarray):
        C = np.asarray(C, dtype=bool)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be a square matrix")
        self.C = C
        self.N = C.shape[0]

    @property
    def relations(self) -> int:
        """Number of related pairs (i, j) with i -< j."""
        return int(self.C.sum())

    @property
    def max_pairs(self) -> int:
        """N choose 2: the number of distinct pairs of elements."""
        return self.N * (self.N - 1) // 2

    def past(self, i: int) -> np.ndarray:
        """Boolean vector over elements that precede i."""
        return self.C[:, i]

    def future(self, i: int) -> np.ndarray:
        """Boolean vector over elements that i precedes."""
        return self.C[i, :]

    def is_natural_labeling(self) -> bool:
        """True iff C is strictly upper triangular (i -< j implies i < j)."""
        return not self.C[np.tril_indices(self.N)].any()

    def is_acyclic(self) -> bool:
        """Strict upper-triangularity (our labeling invariant) already
        guarantees acyclicity: an edge can only point from a lower index
        to a higher one, so no cycle can ever form."""
        return self.is_natural_labeling()

    def is_transitively_closed(self) -> bool:
        return np.array_equal(self.C, transitive_closure(self.C))

    def interval(self, a: int, b: int) -> np.ndarray:
        """Boolean vector of elements x with a -< x -< b (open interval)."""
        if not self.C[a, b]:
            return np.zeros(self.N, dtype=bool)
        return self.C[a, :] & self.C[:, b]

    def interval_size(self, a: int, b: int) -> int:
        return int(self.interval(a, b).sum())

    def sub_causet(self, indices: np.ndarray) -> "CausalSet":
        """Induced sub causal set on the given index subset, relabeled
        0..k-1 while preserving the relative (natural) order."""
        indices = np.sort(np.asarray(indices))
        return CausalSet(self.C[np.ix_(indices, indices)])

    def links(self) -> np.ndarray:
        """Boolean matrix of link (covering) relations: i -< j such that no
        element is strictly between them."""
        Cf = self.C.astype(np.float32)
        has_intermediate = (Cf @ Cf) > 0
        return self.C & ~has_intermediate

    def chain_lengths(self) -> np.ndarray:
        """Longest chain ending at each element, counted in edges.

        Dynamic programming over the natural labeling: because C is upper
        triangular, by the time we reach column j every predecessor i < j
        has already had its value finalized.
        """
        N = self.N
        height = np.zeros(N, dtype=np.int64)
        for j in range(N):
            preds = self.C[:, j]
            if preds.any():
                height[j] = height[preds].max() + 1
        return height

    def height(self) -> int:
        """Number of elements in the longest chain."""
        if self.N == 0:
            return 0
        return int(self.chain_lengths().max()) + 1


def transitive_closure(C: np.ndarray) -> np.ndarray:
    """Boolean transitive closure via repeated doubling.

    Each iteration ORs in all paths reachable by composing the current
    relation with itself, doubling the maximum path length considered.
    After ceil(log2(N)) + 1 iterations every path of any length has been
    absorbed. Used only for validation (tests); all generators in this
    package maintain closure by construction and never need to call this.
    """
    N = C.shape[0]
    if N == 0:
        return C.astype(bool)
    R = C.astype(bool)
    steps = max(1, int(np.ceil(np.log2(max(N, 2)))) + 1)
    for _ in range(steps):
        Rf = R.astype(np.float32)
        two_step = (Rf @ Rf) > 0
        new_R = R | two_step
        if np.array_equal(new_R, R):
            break
        R = new_R
    return R
