"""Phase 3: weighing universes instead of growing them.

Where the rest of causet_lab asks "what growth rule produces a
manifold-like universe", this subpackage asks a different question: if
every causal set on N elements is a possible universe, and we weight
them by a gravitational action S via a Monte Carlo ensemble
exp(-beta * S), does the ensemble have a phase that looks like a
spacetime? Surya (2012, arXiv:1110.6244) showed in 2D that it does: a
"spacetime-like" phase at small beta transitions into a "layered" phase
(causal sets that look like stacked antichains, not a manifold) as beta
increases. This subpackage reproduces that result.

Modules:
    orders.py   2D causal set orders: C[i,j] = (u[i]<u[j]) and (v[i]<v[j])
                for a pair of permutations u, v.
    action.py   The 2D Benincasa-Dowker action.
    sampler.py  Metropolis MCMC over (u, v) with autocorrelation estimation.
    study.py    Controls, beta scan, and report generation.
"""
