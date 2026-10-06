"""Control tests that must pass before any causet_lab result is trusted.

These pin down ground truth: sprinkled causal sets must yield dimension
estimates close to their true dimension, Kleitman-Rothschild junk orders
must have height exactly 3, every generator must produce a transitively
closed and acyclic order, and growth must be reproducible under a fixed
seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from causet_lab.generators import sprinkle, kleitman_rothschild, grow
from causet_lab.measures import myrheim_meyer_dimension, midpoint_dimension
from causet_lab.rules import transitive_percolation


def test_sprinkle_d2_myrheim_meyer_dimension():
    estimates = [myrheim_meyer_dimension(sprinkle(1000, 2, seed=s)[0].C) for s in range(5)]
    assert abs(np.mean(estimates) - 2.0) < 0.2


def test_sprinkle_d4_myrheim_meyer_dimension():
    estimates = [myrheim_meyer_dimension(sprinkle(2000, 4, seed=s)[0].C) for s in range(5)]
    assert abs(np.mean(estimates) - 4.0) < 0.4


@pytest.mark.parametrize("d", [2, 3, 4])
def test_midpoint_agrees_with_myrheim_meyer_on_sprinkles(d):
    for seed in range(5):
        cset, _ = sprinkle(1000, d, seed=seed)
        mm = myrheim_meyer_dimension(cset.C)
        mp = midpoint_dimension(cset.C)
        assert abs(mm - mp) < 0.5


def test_kleitman_rothschild_height_is_three():
    for seed in range(5):
        cset = kleitman_rothschild(2000, seed=seed)
        assert cset.height() == 3


@pytest.mark.parametrize("d", [2, 3, 4])
def test_sprinkle_is_transitively_closed_and_acyclic(d):
    for seed in range(3):
        cset, _ = sprinkle(300, d, seed=seed)
        assert cset.is_acyclic()
        assert cset.is_transitively_closed()


def test_kleitman_rothschild_is_transitively_closed_and_acyclic():
    for seed in range(3):
        cset = kleitman_rothschild(300, seed=seed)
        assert cset.is_acyclic()
        assert cset.is_transitively_closed()


def test_grow_is_transitively_closed_and_acyclic():
    rule = transitive_percolation(0.1)
    for seed in range(3):
        cset = grow(300, rule, seed=seed)
        assert cset.is_acyclic()
        assert cset.is_transitively_closed()


def test_growth_reproducible_with_same_seed():
    rule = transitive_percolation(0.2)
    c1 = grow(500, rule, seed=42)
    c2 = grow(500, rule, seed=42)
    assert np.array_equal(c1.C, c2.C)


def test_growth_differs_with_different_seed():
    rule = transitive_percolation(0.2)
    c1 = grow(500, rule, seed=1)
    c2 = grow(500, rule, seed=2)
    assert not np.array_equal(c1.C, c2.C)
