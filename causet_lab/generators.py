"""Ways to produce causal sets.

Three families:

- ``sprinkle``: Poisson sprinkling into a causal diamond in d-dimensional
  Minkowski space. These are the CONTROLS -- we know their true dimension
  by construction, so every dimension estimator can be calibrated against
  them.
- ``kleitman_rothschild``: the "junk" control. A generic 3-layer random
  order. This is what causal sets look like when there is no underlying
  geometry at all.
- ``grow``: sequential growth under a pluggable rule (see rules.py). This
  is how we search for growth dynamics whose universes resemble the
  sprinkles instead of the junk.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .causet import CausalSet


def sprinkle(N: int, d: int, seed: int, T: float = 1.0) -> Tuple[CausalSet, np.ndarray]:
    """Sprinkle N points uniformly into a causal diamond (Alexandrov
    interval) in d-dimensional Minkowski space.

    The diamond is the set of points (t, x) with |x| <= T - |t|, where x is
    the (d-1)-dimensional spatial position. Points are sampled by rejection
    from the bounding box [-T, T]^d, then sorted by time so the resulting
    labeling is natural (i -< j implies i < j).

    Returns (CausalSet, coords) where coords is an (N, d) array with the
    time coordinate in column 0 and spatial coordinates in the rest,
    ordered to match the causal set's labeling.
    """
    if d not in (2, 3, 4):
        raise ValueError("d must be 2, 3, or 4")
    rng = np.random.default_rng(seed)
    spatial_dim = d - 1
    collected = []
    n_have = 0
    while n_have < N:
        batch = max(2 * (N - n_have), 256)
        t = rng.uniform(-T, T, size=batch)
        if spatial_dim > 0:
            x = rng.uniform(-T, T, size=(batch, spatial_dim))
            r = np.linalg.norm(x, axis=1)
        else:
            x = np.zeros((batch, 0))
            r = np.zeros(batch)
        mask = r <= (T - np.abs(t))
        accepted = np.column_stack([t, x])[mask]
        collected.append(accepted)
        n_have += accepted.shape[0]
    pts = np.vstack(collected)[:N]

    order = np.argsort(pts[:, 0], kind="stable")
    pts = pts[order]
    t = pts[:, 0]
    x = pts[:, 1:]

    # Build C row-chunked: a full (N, N, spatial_dim) diff tensor would be
    # O(N^2 * d) memory (multiple GiB already at N=16000), so process rows
    # in blocks and only ever materialize (chunk, N[, spatial_dim]) temporaries.
    C = np.zeros((N, N), dtype=bool)
    chunk = max(1, min(N, 2000))
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        dt_block = t[None, :] - t[start:stop, None]  # (block, N): t_j - t_i
        if spatial_dim > 0:
            diff_block = x[None, :, :] - x[start:stop, None, :]
            dx_block = np.linalg.norm(diff_block, axis=2)
        else:
            dx_block = np.zeros((stop - start, N))
        C[start:stop, :] = (dt_block > 0) & (dx_block <= dt_block)
    np.fill_diagonal(C, False)
    return CausalSet(C), pts


def kleitman_rothschild(N: int, seed: int) -> CausalSet:
    """The Kleitman-Rothschild "junk" order: three layers of sizes roughly
    N/4, N/2, N/4, with each bottom-middle and middle-top pair related
    independently with probability 1/2, then transitively closed. This is
    what a generic random order looks like: no geometry, height stuck at 3.
    """
    rng = np.random.default_rng(seed)
    n1 = N // 4
    n2 = N // 2
    n3 = N - n1 - n2

    bm = rng.random((n1, n2)) < 0.5
    mt = rng.random((n2, n3)) < 0.5
    bt = (bm.astype(np.float32) @ mt.astype(np.float32)) > 0

    C = np.zeros((N, N), dtype=bool)
    C[0:n1, n1:n1 + n2] = bm
    C[n1:n1 + n2, n1 + n2:N] = mt
    C[0:n1, n1 + n2:N] = bt
    return CausalSet(C)


def grow(
    N: int,
    rule: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    seed: int,
    track_frontier: bool = False,
    track_branching: bool = False,
):
    """Sequentially grow a causal set of N elements under ``rule``.

    Starting from the empty set, elements are added one at a time. For
    each new element n, ``rule(C_current, rng)`` returns a boolean vector
    marking which existing elements lie in n's DIRECT past. n's full past
    is then the union of those elements and all of *their* pasts, which is
    exactly what transitive closure requires -- so the causal set stays
    transitively closed at every step without ever needing a separate
    closure pass. Existing elements are never touched again once added.

    Returns just a CausalSet unless ``track_frontier`` or ``track_branching``
    is set, in which case it returns (CausalSet, info) where info is a dict
    with whichever of these keys were requested:

    - "frontier_history": an (M, 2) array of (n, frontier_width) pairs, the
      number of maximal elements at every size n during growth. This is
      the only way to see the frontier's history: the final matrix alone
      can't tell you how wide it was at intermediate sizes, since elements
      maximal early on may have acquired successors by the end.
    - "widen_fraction": the fraction of growth steps whose direct past did
      not touch the current frontier at all (the new element branches off
      below the tip, widening the frontier by one) as opposed to touching
      at least one maximal element (extending/deepening it). A rule stuck
      permanently extending (widen_fraction ~= 0) can never grow a frontier
      wider than 1 and degenerates into a chain.
    """
    rng = np.random.default_rng(seed)
    C = np.zeros((N, N), dtype=bool)
    need_maximal = track_frontier or track_branching
    frontier_history = [] if track_frontier else None
    widen_flags = [] if track_branching else None
    for n in range(1, N):
        existing = C[:n, :n]
        maximal_before = (~existing.any(axis=1)) if need_maximal else None
        if track_frontier:
            width = 1 if n == 1 else int(maximal_before.sum())
            frontier_history.append((n, width))
        direct_past = np.asarray(rule(existing, rng), dtype=bool)
        if direct_past.shape != (n,):
            raise ValueError(
                f"rule must return a boolean vector of length {n}, got shape {direct_past.shape}"
            )
        if track_branching:
            widen_flags.append(not bool(np.any(direct_past & maximal_before)))
        full_past = direct_past.copy()
        if direct_past.any():
            ancestors_past = existing[:, direct_past].any(axis=1)
            full_past |= ancestors_past
        C[:n, n] = full_past
    cset = CausalSet(C)
    if not track_frontier and not track_branching:
        return cset
    info = {}
    if track_frontier:
        frontier_history.append((N, int((~C.any(axis=1)).sum())))
        info["frontier_history"] = np.array(frontier_history, dtype=np.int64)
    if track_branching:
        info["widen_fraction"] = float(np.mean(widen_flags)) if widen_flags else float("nan")
    return cset, info
