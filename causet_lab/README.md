# causet_lab

An exploration of causal set theory: the hypothesis that spacetime is not
fundamental, but emerges from a much simpler structure -- a finite set of
events with a "comes before" relation, and nothing else.

## The idea

Throw out continuous space and time. What's left, in a causal set, is just
a set of elements and a strict partial order: for any two elements, either
one precedes the other, or they are "spacelike" to each other (neither
precedes the other -- causally unrelated). That's it. No coordinates, no
metric, no dimension. Causal set theory's bet is that everything we think
of as geometric -- how many dimensions spacetime has, how far apart two
points are, whether the set "looks like" a smooth manifold at all -- can
be recovered as a *statistical, large-scale property* of the order
relation alone.

This project builds the tools to test that bet, and to search for the
*dynamics* (a rule for growing a causal set one element at a time) that
would produce a universe indistinguishable, statistically, from the one we
live in.

## The three kinds of causal set here

- **Sprinkles** (`generators.sprinkle`) are the controls. Points are
  scattered uniformly at random ("Poisson sprinkled") into a diamond-shaped
  region of actual d-dimensional Minkowski spacetime, and the order relation
  is read off from genuine causality (`x` precedes `y` iff `y` is in `x`'s
  future light cone). We *know* the dimension here, because we built it in
  -- so every dimension estimator in this package can be calibrated and
  sanity-checked against sprinkles before it's trusted on anything else.

- **Kleitman-Rothschild orders** (`generators.kleitman_rothschild`) are the
  "junk" control: what a *generic*, structureless random partial order
  looks like. It turns out almost all random orders collapse into just
  three layers with dense random connections between them -- no sense of
  dimension, no sense of distance, nothing manifold-like. This is the
  baseline for "no emergent geometry at all."

- **Grown universes** (`generators.grow`) are built one element at a time
  by a *growth rule* (`rules.py`): given everything that exists so far, the
  rule decides what the new element's direct past is. This is the only
  mechanism in the package that resembles actual dynamics -- a process
  unfolding step by step -- rather than a one-shot sample. The central
  question this project is set up to probe: **does any growth rule produce
  universes that look statistically like the sprinkles, rather than like
  the junk?** `rules.transitive_percolation` (the simplest rule satisfying
  Rideout-Sorkin discrete general covariance -- it doesn't peek at element
  labels, only coin-flips) is the one rule shipped here; `rules.py` is
  written to make adding more a one-function exercise.

## How dimension gets measured from pure order

Two independent estimators are implemented, both in `measures.py`:

- **Myrheim-Meyer dimension**: in a true Alexandrov interval (the causal
  diamond between two points) in d-dimensional Minkowski space, the
  fraction of all pairs of sprinkled points that are causally related has
  a known closed form, `f(d) = Gamma(d+1) Gamma(d/2) / (2 Gamma(3d/2))`.
  Measure `f` in any interval-shaped causal set, invert the formula
  numerically, and out comes an estimate of `d` -- no coordinates required.

- **Midpoint dimension**: a second, independent estimator for an interval
  causal set. Find the element that best splits the interval in half (the
  "midpoint"), and the ratio of its half-interval size to the whole tells
  you the dimension too, by a volume-scaling argument. Two unrelated
  derivations agreeing is a good consistency check.

Both estimators only make sense on an *interval* -- a causal set with a
single bottom and top element and everything else in between. Sprinkles
are intervals by construction. Grown and junk universes are **not**, so
`measures.sample_intervals` is used to cut random intervals out of them
first (and, for a fair comparison, the same sampling is applied to the
sprinkles too, even though they don't strictly need it).

`measures.interval_abundances` gives a third, non-dimension-based
fingerprint: manifold-like causal sets have a characteristic distribution
of interval sizes, and comparing a case's profile against the sprinkled
reference profiles is a shape check that doesn't reduce everything to one
number.

## Project layout

```
causet_lab/
  causet.py        CausalSet class: the matrix, and vectorized helpers
                    (past/future, links, chain lengths, transitive closure)
  generators.py     sprinkle, kleitman_rothschild, grow
  rules.py          pluggable growth rules (transitive_percolation + registry)
  measures.py       ordering_fraction, myrheim_meyer_dimension,
                    midpoint_dimension, sample_intervals,
                    interval_abundances, height
  plots.py          every plot the CLI produces
  run.py            CLI (python -m causet_lab.run ...)
  tests/test_controls.py   must pass before any result is trusted
  results/          generated plots and summary.md land here
```

## Representation

A causal set is an `N x N` numpy boolean array `C`, where `C[i, j] = True`
means element `i` strictly precedes element `j`. Every generator produces a
*natural labeling*: whenever `C[i, j]` is True, `i < j` as plain integer
indices, so `C` is always strictly upper triangular. That one invariant is
what makes everything here vectorizable instead of needing a general
topological sort -- it's why `N = 2000` runs in seconds instead of minutes.

## Usage

```bash
python -m causet_lab.run controls          # sprinkle d=2,3,4, N=500/1000/2000, 5 seeds each
python -m causet_lab.run junk              # Kleitman-Rothschild junk orders
python -m causet_lab.run percolation --p 0.01 0.05 0.1 0.3 --N 2000
python -m causet_lab.run rule --name transitive_percolation --N 2000 --param p=0.1
python -m causet_lab.run all               # everything, + results/summary.md
```

Run the tests first:

```bash
python -m pytest causet_lab/tests/test_controls.py -v
```

## Adding a new growth rule

A rule is any function `rule(C, rng) -> np.ndarray[bool]` that looks at the
existing causal set `C` and a numpy `Generator` and returns which existing
elements are in the new element's *direct* past. The growth framework in
`generators.grow` takes care of turning that into a full, transitively
closed past automatically. See the docstring at the top of `rules.py` for
the one rule worth internalizing: a physically sensible rule must depend
only on the causal structure of `C`, never on the arbitrary integer labels
-- that's what "discrete general covariance" means here.
