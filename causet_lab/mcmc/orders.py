"""2D causal set orders: a causal set built from two permutations.

A "2D order" on N elements is given by two independent total orders
(permutations) u, v of {0, ..., N-1}: element i strictly precedes
element j iff u[i] < u[j] AND v[i] < v[j]. The intersection of two
strict total orders is automatically a strict partial order -- in
particular it is already transitively closed, so none of the
generators elsewhere in this package need to run transitive_closure()
on the result. This is the standard representation used for 2D causal
set Monte Carlo (e.g. Surya 2012, arXiv:1110.6244): sampling over 2D
orders means sampling over pairs of permutations.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def orders_to_matrix(u: np.ndarray, v: np.ndarray, natural_label: bool = True) -> np.ndarray:
    """Convert a pair of permutations (u, v) -- each a length-N array
    giving every element's rank in that total order -- into the boolean
    relation matrix C[i, j] = (u[i] < u[j]) and (v[i] < v[j]).

    If natural_label is True, elements are first relabeled by sorting on
    u + v: since i -< j implies u[i] < u[j] and v[i] < v[j], it implies
    u[i] + v[i] < u[j] + v[j], so sorting on u + v is a valid linear
    extension of the order, and the returned C is guaranteed strictly
    upper triangular (the natural-labeling invariant the rest of
    causet_lab relies on, e.g. CausalSet.height()). Set it to False in
    hot loops (MCMC proposals) where only the action is needed, since
    the action does not depend on labeling and the sort is pure overhead.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if natural_label:
        order = np.argsort(u + v, kind="stable")
        u = u[order]
        v = v[order]
    C = (u[:, None] < u[None, :]) & (v[:, None] < v[None, :])
    return C


def random_2d_order(N: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a uniformly random 2D order on N elements: u, v independent
    uniform random permutations of range(N). Returns (u, v, C) with C
    naturally labeled. This is exactly the beta=0 ensemble sampled by
    the Metropolis chain in sampler.py (every move is accepted when
    beta=0, so the chain performs an unweighted random walk over (u,v)
    pairs, which is uniform sampling by construction).
    """
    rng = np.random.default_rng(seed)
    u = rng.permutation(N)
    v = rng.permutation(N)
    C = orders_to_matrix(u, v, natural_label=True)
    return u, v, C
