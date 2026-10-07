# causet_lab growth-rule battery report

Tests whether any growth rule produces universes that pass the full measurement battery: do the two dimension estimators agree, does the estimate stay stable as N grows, does height scale the way a sprinkle's does, does the inner interval structure resemble a sprinkle of the matching dimension, and does link valence grow with N instead of staying flat like a lattice.

## The frontier-collapse finding

The first version of R1 (`frontier_random`), R2 (`frontier_local`), R3 (`frontier_preferential`), R4 (`bounded_valence`), and R6 (`width_forced`) picked parents only from the current *maximal* elements (the frontier). That is fatally self-defeating: growth always starts from a single seed element, so the frontier starts at width 1, and "pick k, or all of them if fewer than k exist" always fires the collapse branch at width 1 for any k >= 1 -- every one of those rules deterministically produced a plain chain (height == N) regardless of parameters or seed. The fix, applied throughout this report: parents are chosen from *all* existing elements, not just the frontier, so frontier width becomes an emergent outcome (tracked below as widen vs deepen fraction) instead of something the selection rule can single-handedly destroy. See `rules.py` for the full implementation and reasoning.

## Calibration (thresholds derived from the sprinkle controls, not hand-tuned)

- Agreement baseline (mean |MM - midpoint| across sprinkle d=2,3,4): 0.172 -> pass threshold (2x) = 0.344
- Drift baseline (mean |dimension drift vs log N| across sprinkle d=2,3,4): 0.0230 -> pass threshold (2x) = 0.0461
- Abundance baseline (mean seed-to-seed L1 distance within same-d sprinkles at N=4000): 0.029 -> pass threshold (2x) = 0.059
- Height-consistency band (given directly in the spec, not calibrated): alpha * d_MM in [0.8, 1.25]
- Valence-growth threshold (judgment call, documented rather than calibrated): valence(N_max) / valence(N_min) >= 1.15
- width_forced calibration constant c (fit to real sprinkle frontier width at N=2000): d_target=2 -> c=0.197, d_target=4 -> c=0.683

A rule PASSES only if its best parameter setting satisfies all five checks.

## Verdict summary (best setting per rule family)

| rule family | best setting | flag | verdict | checks passed | failure signature |
|---|---|---|---|---|---|
| random_parents | random_parents(k=5) | - | FAIL | 2/5 | dimension drifts with N (slope=0.261, threshold 0.046); height scaling inconsistent (alpha*dMM=0.51, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.35 > 0.06) |
| local_parents | local_parents(k=2) | - | FAIL | 1/5 | dimension drifts with N (slope=0.095, threshold 0.046); height scaling inconsistent (alpha*dMM=0.23, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.42 > 0.06); valence flat (ratio=1.01 < 1.15) |
| preferential_parents | preferential_parents(k=5) | - | FAIL | 2/5 | dimension drifts with N (slope=0.349, threshold 0.046); height scaling inconsistent (alpha*dMM=0.57, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.32 > 0.06) |
| bounded_valence | bounded_valence(k=2,vmax=2) | - | FAIL | 3/5 | abundance profile far from any sprinkle (closest d=2, distance=0.44 > 0.06); valence flat (ratio=1.00 < 1.15) |
| recency_cheat | recency_cheat(L=20,p=0.3) | ILLEGAL | FAIL | 3/5 | abundance profile far from any sprinkle (closest d=2, distance=0.36 > 0.06); valence flat (ratio=1.01 < 1.15) |
| width_forced | width_forced(d_target=2) | CHEAT | FAIL | 1/5 | dimension drifts with N (slope=0.166, threshold 0.046); height scaling inconsistent (alpha*dMM=0.29, want 0.8-1.25); abundance profile far from any sprinkle (closest d=3, distance=0.28 > 0.06); valence flat (ratio=1.00 < 1.15) |

## All configurations tried

| rule | agreement | stability (drift) | height consistency | inner structure (closest d) | valence ratio | checks | notes |
|---|---|---|---|---|---|---|---|
| random_parents(k=1) | 0.26 (ok) | 0.000 (ok) | 0.14 (FAIL) | d=2, 0.39 (FAIL) | 1.00 (FAIL) | 2/5 |  |
| random_parents(k=2) | 0.20 (ok) | 0.106 (FAIL) | 0.37 (FAIL) | d=2, 0.36 (FAIL) | 1.07 (FAIL) | 1/5 |  |
| random_parents(k=3) | 0.19 (ok) | 0.176 (FAIL) | 0.48 (FAIL) | d=2, 0.32 (FAIL) | 1.14 (FAIL) | 1/5 |  |
| random_parents(k=5) | 0.17 (ok) | 0.261 (FAIL) | 0.51 (FAIL) | d=2, 0.35 (FAIL) | 1.27 (ok) | 2/5 |  |
| local_parents(k=2) | 0.25 (ok) | 0.095 (FAIL) | 0.23 (FAIL) | d=2, 0.42 (FAIL) | 1.01 (FAIL) | 1/5 | N=4000 too slow (16.4s for a single seed); dropped to N=3000 for this config |
| local_parents(k=3) | 0.31 (ok) | 0.162 (FAIL) | 0.30 (FAIL) | d=3, 0.24 (FAIL) | 1.01 (FAIL) | 1/5 |  |
| local_parents(k=5) | 0.31 (ok) | 0.224 (FAIL) | 0.45 (FAIL) | d=3, 0.11 (FAIL) | 1.00 (FAIL) | 1/5 |  |
| local_parents(k=8) | 0.33 (ok) | 0.284 (FAIL) | 0.58 (FAIL) | d=3, 0.07 (FAIL) | 1.01 (FAIL) | 1/5 | N=4000 too slow (15.7s for a single seed); dropped to N=3000 for this config |
| preferential_parents(k=2) | 0.14 (ok) | 0.124 (FAIL) | 0.47 (FAIL) | d=2, 0.31 (FAIL) | 1.08 (FAIL) | 1/5 |  |
| preferential_parents(k=3) | 0.12 (ok) | 0.234 (FAIL) | 0.52 (FAIL) | d=2, 0.32 (FAIL) | 1.17 (ok) | 2/5 |  |
| preferential_parents(k=5) | 0.10 (ok) | 0.349 (FAIL) | 0.57 (FAIL) | d=2, 0.32 (FAIL) | 1.34 (ok) | 2/5 |  |
| bounded_valence(k=2,vmax=2) | 0.03 (ok) | 0.000 (ok) | 1.00 (ok) | d=2, 0.44 (FAIL) | 1.00 (FAIL) | 3/5 |  |
| bounded_valence(k=2,vmax=4) | 0.19 (ok) | 0.131 (FAIL) | 0.44 (FAIL) | d=2, 0.32 (FAIL) | 1.04 (FAIL) | 1/5 |  |
| bounded_valence(k=2,vmax=8) | 0.21 (ok) | 0.121 (FAIL) | 0.34 (FAIL) | d=2, 0.35 (FAIL) | 1.07 (FAIL) | 1/5 |  |
| bounded_valence(k=3,vmax=2) | 0.03 (ok) | 0.000 (ok) | 1.00 (ok) | d=2, 0.44 (FAIL) | 1.00 (FAIL) | 3/5 |  |
| bounded_valence(k=3,vmax=4) | 0.16 (ok) | 0.320 (FAIL) | 0.58 (FAIL) | d=2, 0.33 (FAIL) | 1.14 (FAIL) | 1/5 |  |
| bounded_valence(k=3,vmax=8) | 0.19 (ok) | 0.175 (FAIL) | 0.47 (FAIL) | d=2, 0.34 (FAIL) | 1.15 (FAIL) | 1/5 |  |
| recency_cheat(L=20,p=0.1) | 0.02 (ok) | -0.046 (FAIL) | 1.40 (FAIL) | d=2, 0.25 (FAIL) | 1.00 (FAIL) | 1/5 |  |
| recency_cheat(L=20,p=0.3) | 0.00 (ok) | -0.011 (ok) | 1.15 (ok) | d=2, 0.36 (FAIL) | 1.01 (FAIL) | 3/5 |  |
| recency_cheat(L=50,p=0.1) | 0.01 (ok) | -0.020 (ok) | 1.59 (FAIL) | d=2, 0.16 (FAIL) | 1.05 (FAIL) | 2/5 |  |
| recency_cheat(L=50,p=0.3) | 0.00 (ok) | -0.009 (ok) | 1.16 (ok) | d=2, 0.36 (FAIL) | 1.02 (FAIL) | 3/5 |  |
| width_forced(d_target=2) | 0.32 (ok) | 0.166 (FAIL) | 0.29 (FAIL) | d=3, 0.28 (FAIL) | 1.00 (FAIL) | 1/5 | N=4000 too slow (16.5s for a single seed); dropped to N=3000 for this config |
| width_forced(d_target=4) | 0.34 (ok) | 0.175 (FAIL) | 0.37 (FAIL) | d=3, 0.23 (FAIL) | 1.01 (FAIL) | 1/5 | N=4000 too slow (16.7s for a single seed); dropped to N=3000 for this config |

## Per-family detail

### random_parents

Best setting: **random_parents(k=5)** -- FAIL

Signature: dimension drifts with N (slope=0.261, threshold 0.046); height scaling inconsistent (alpha*dMM=0.51, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.35 > 0.06)

Widen fraction at N=4000: 0.398 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| random_parents(k=1) | 0.26 | 0.000 | 0.14 | d=2, 0.39 | 1.00 | 2/5 |
| random_parents(k=2) | 0.20 | 0.106 | 0.37 | d=2, 0.36 | 1.07 | 1/5 |
| random_parents(k=3) | 0.19 | 0.176 | 0.48 | d=2, 0.32 | 1.14 | 1/5 |
| random_parents(k=5) | 0.17 | 0.261 | 0.51 | d=2, 0.35 | 1.27 | 2/5 |

### local_parents

Best setting: **local_parents(k=2)** -- FAIL

Signature: dimension drifts with N (slope=0.095, threshold 0.046); height scaling inconsistent (alpha*dMM=0.23, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.42 > 0.06); valence flat (ratio=1.01 < 1.15)

Widen fraction at N=3000: 0.507 (fraction of growth steps that branched off below the tip rather than extending it)

Notes: N=4000 too slow (16.4s for a single seed); dropped to N=3000 for this config

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| local_parents(k=2) | 0.25 | 0.095 | 0.23 | d=2, 0.42 | 1.01 | 1/5 |
| local_parents(k=3) | 0.31 | 0.162 | 0.30 | d=3, 0.24 | 1.01 | 1/5 |
| local_parents(k=5) | 0.31 | 0.224 | 0.45 | d=3, 0.11 | 1.00 | 1/5 |
| local_parents(k=8) | 0.33 | 0.284 | 0.58 | d=3, 0.07 | 1.01 | 1/5 |

### preferential_parents

Best setting: **preferential_parents(k=5)** -- FAIL

Signature: dimension drifts with N (slope=0.349, threshold 0.046); height scaling inconsistent (alpha*dMM=0.57, want 0.8-1.25); abundance profile far from any sprinkle (closest d=2, distance=0.32 > 0.06)

Widen fraction at N=4000: 0.363 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| preferential_parents(k=2) | 0.14 | 0.124 | 0.47 | d=2, 0.31 | 1.08 | 1/5 |
| preferential_parents(k=3) | 0.12 | 0.234 | 0.52 | d=2, 0.32 | 1.17 | 2/5 |
| preferential_parents(k=5) | 0.10 | 0.349 | 0.57 | d=2, 0.32 | 1.34 | 2/5 |

### bounded_valence

Best setting: **bounded_valence(k=2,vmax=2)** -- FAIL

Signature: abundance profile far from any sprinkle (closest d=2, distance=0.44 > 0.06); valence flat (ratio=1.00 < 1.15)

Widen fraction at N=4000: 0.000 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| bounded_valence(k=2,vmax=2) | 0.03 | 0.000 | 1.00 | d=2, 0.44 | 1.00 | 3/5 |
| bounded_valence(k=2,vmax=4) | 0.19 | 0.131 | 0.44 | d=2, 0.32 | 1.04 | 1/5 |
| bounded_valence(k=2,vmax=8) | 0.21 | 0.121 | 0.34 | d=2, 0.35 | 1.07 | 1/5 |
| bounded_valence(k=3,vmax=2) | 0.03 | 0.000 | 1.00 | d=2, 0.44 | 1.00 | 3/5 |
| bounded_valence(k=3,vmax=4) | 0.16 | 0.320 | 0.58 | d=2, 0.33 | 1.14 | 1/5 |
| bounded_valence(k=3,vmax=8) | 0.19 | 0.175 | 0.47 | d=2, 0.34 | 1.15 | 1/5 |

### recency_cheat

**ILLEGAL CONTROL**: this rule reads raw label/index order, which violates discrete general covariance. Included only to show what cheating with labels buys you.

Best setting: **recency_cheat(L=20,p=0.3)** -- FAIL

Signature: abundance profile far from any sprinkle (closest d=2, distance=0.36 > 0.06); valence flat (ratio=1.01 < 1.15)

Widen fraction at N=4000: 0.339 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| recency_cheat(L=20,p=0.1) | 0.02 | -0.046 | 1.40 | d=2, 0.25 | 1.00 | 1/5 |
| recency_cheat(L=20,p=0.3) | 0.00 | -0.011 | 1.15 | d=2, 0.36 | 1.01 | 3/5 |
| recency_cheat(L=50,p=0.1) | 0.01 | -0.020 | 1.59 | d=2, 0.16 | 1.05 | 2/5 |
| recency_cheat(L=50,p=0.3) | 0.00 | -0.009 | 1.16 | d=2, 0.36 | 1.02 | 3/5 |

### width_forced

**CHEAT CONTROL**: this rule has the target dimension baked in by hand (the frontier-width constant c is calibrated from real sprinkles). Included only to test whether getting the frontier's shape right is sufficient on its own.

Best setting: **width_forced(d_target=2)** -- FAIL

Signature: dimension drifts with N (slope=0.166, threshold 0.046); height scaling inconsistent (alpha*dMM=0.29, want 0.8-1.25); abundance profile far from any sprinkle (closest d=3, distance=0.28 > 0.06); valence flat (ratio=1.00 < 1.15)

Widen fraction at N=3000: 0.514 (fraction of growth steps that branched off below the tip rather than extending it)

Notes: N=4000 too slow (16.5s for a single seed); dropped to N=3000 for this config

| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |
|---|---|---|---|---|---|---|
| width_forced(d_target=2) | 0.32 | 0.166 | 0.29 | d=3, 0.28 | 1.00 | 1/5 |
| width_forced(d_target=4) | 0.34 | 0.175 | 0.37 | d=3, 0.23 | 1.01 | 1/5 |

## Plots

- `rules_dimension_vs_N_overview.png` -- every rule's dimension estimates vs N, sprinkle d=2,3,4 bands behind
- `rules_dimension_vs_N_<family>.png` -- per-family detail
- `rules_frontier_width_<family>.png` -- frontier width vs n during growth, log-log, per family
- `rules_valence_vs_N_overview.png` -- link valence vs N, every rule and sprinkle
- `hasse_best_<family>.png` -- Hasse diagram (links only) for the best setting of each family, N=100
