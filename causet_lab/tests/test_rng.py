"""Sanity + reproducibility checks for the checkpointable splitmix64 RNG."""
from __future__ import annotations

import numpy as np

from causet_lab.mcmc import rng


def test_next_double_is_uniform_and_reproducible():
    s = rng.seed_state(123)
    vals = []
    for _ in range(200_000):
        s, r = rng.next_double(s)
        s = rng.tick(s)
        vals.append(r)
    vals = np.array(vals)
    assert vals.min() >= 0.0 and vals.max() < 1.0
    assert abs(vals.mean() - 0.5) < 0.01

    s2 = rng.seed_state(123)
    r2s = []
    for _ in range(5):
        s2, r2 = rng.next_double(s2)
        s2 = rng.tick(s2)
        r2s.append(r2)
    assert np.allclose(r2s, vals[:5])


def test_next_below_covers_full_range():
    s = rng.seed_state(7)
    n = 13
    seen = set()
    for _ in range(20_000):
        s, v = rng.next_below(s, n)
        s = rng.tick(s)
        assert 0 <= v < n
        seen.add(int(v))
    assert seen == set(range(n))
