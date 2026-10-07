# causet_lab follow-up report: locality + growing valence

Follow-up to `results/rules_report.md`. `local_parents` showed inner-structure distance falling steadily with k (0.42, 0.24, 0.11, 0.07 at k=2,3,5,8; pass threshold 0.059, closest d=3) while every fixed-k rule had structurally flat valence. Tested here: (1) larger fixed k for local_parents, (2) `local_parents_adaptive`, where the parent count grows with the picked element's own past size (no labels, no N, no target dimension), and (3) `frontier_weighted`, preferring elements near the causal present (small future) measured purely from order.

## Tightened inner-structure check

Check 4 now requires three numbers to agree, not just a distance threshold: the closest matching sprinkle dimension must equal round(d_MM), and round(d_midpoint) must equal that same integer too. A config can have a small abundance distance and still fail this check if the two dimension estimators round to different integers, or round to a different integer than the sprinkle profile it's closest to.

- Agreement threshold (2x sprinkle baseline): 0.346
- Drift threshold (2x sprinkle baseline): 0.0441
- Abundance threshold (2x sprinkle baseline): 0.062
- Height-consistency band: alpha * d_MM in [0.8, 1.25]
- Valence-growth threshold: valence(N_max) / valence(N_min) >= 1.15
- Confirmation rerun: N=5000, 5 fresh seeds, triggered for any config passing >= 4/5 checks

## Verdict summary (best setting per family)

| family | best setting | verdict | checks | d_MM~ / d_mp~ / closest d | confirmation @N=5000 | failure signature |
|---|---|---|---|---|---|---|
| local_parents | local_parents(k=32) | FAIL | 3/5 | 3 / 4 / 3 | - | dimension drifts with N (slope=0.325, threshold 0.044); dimension estimates don't round-agree (d_MM~3, d_mp~4, closest sprinkle d=3) |
| local_parents_adaptive | local_parents_adaptive(c=4,beta=0.33) | FAIL | 3/5 | 3 / 3 / 3 | - | dimension drifts with N (slope=0.166, threshold 0.044); abundance profile far from matching sprinkle (distance=0.19 > 0.06) |
| frontier_weighted | frontier_weighted(a=2,k=8) | FAIL | 3/5 | 1 / 1 / 2 | - | dimension estimates don't round-agree (d_MM~1, d_mp~1, closest sprinkle d=2); valence flat (ratio=1.02 < 1.15) |

## All configurations tried

| rule | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | valence ratio | checks | notes |
|---|---|---|---|---|---|---|---|---|
| local_parents(k=2) | 0.25 | 0.104 | 0.23 | 1/2/2 | 0.40 | 1.01 | 1/5 |  |
| local_parents(k=3) | 0.31 | 0.162 | 0.30 | 2/2/3 | 0.24 | 1.01 | 1/5 |  |
| local_parents(k=5) | 0.31 | 0.224 | 0.45 | 2/2/3 | 0.11 | 1.00 | 1/5 |  |
| local_parents(k=8) | 0.32 | 0.282 | 0.56 | 2/3/3 | 0.07 | 1.02 | 1/5 |  |
| local_parents(k=12) | 0.21 | 0.096 | 0.59 | 3/3/3 | 0.12 | 1.05 | 1/5 |  |
| local_parents(k=16) | 0.19 | 0.103 | 0.69 | 3/3/3 | 0.16 | 1.11 | 1/5 |  |
| local_parents(k=24) | 0.18 | 0.259 | 0.71 | 3/3/3 | 0.23 | 1.22 | 2/5 |  |
| local_parents(k=32) | 0.13 | 0.325 | 0.85 | 3/4/3 | 0.23 | 1.33 | 3/5 |  |
| local_parents_adaptive(c=1,beta=0.25) | 0.28 | 0.127 | 0.24 | 1/2/2 | 0.38 | 1.15 | 1/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=2,beta=0.25) | 0.34 | 0.274 | 0.41 | 2/2/3 | 0.12 | 1.18 | 2/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=4,beta=0.25) | 0.26 | 0.255 | 0.70 | 3/3/3 | 0.08 | 1.26 | 2/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=1,beta=0.33) | 0.27 | 0.188 | 0.28 | 2/2/3 | 0.31 | 1.18 | 2/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=2,beta=0.33) | 0.33 | 0.342 | 0.53 | 2/3/3 | 0.10 | 1.30 | 2/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=4,beta=0.33) | 0.21 | 0.166 | 0.91 | 3/3/3 | 0.19 | 1.43 | 3/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=1,beta=0.5) | 0.35 | 0.384 | 0.49 | 2/3/3 | 0.13 | 1.47 | 1/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=2,beta=0.5) | 0.21 | 0.248 | 0.94 | 3/3/3 | 0.14 | 1.74 | 3/5 | candidate-sample cap hit 0/37480 steps (0.0%) |
| local_parents_adaptive(c=4,beta=0.5) | 0.24 | 0.164 | 1.31 | 4/4/3 | 0.25 | 1.64 | 2/5 | N=4000 too slow (20.6s for a single seed); dropped to N=3000 for this config; candidate-sample cap hit 0/32480 steps (0.0%) |
| frontier_weighted(a=0.5,k=5) | 0.18 | 0.227 | 0.54 | 2/2/2 | 0.23 | 1.02 | 1/5 |  |
| frontier_weighted(a=0.5,k=8) | 0.17 | 0.258 | 0.69 | 2/2/2 | 0.26 | 1.04 | 1/5 |  |
| frontier_weighted(a=1,k=5) | 0.08 | 0.209 | 0.74 | 2/2/2 | 0.11 | 1.05 | 1/5 |  |
| frontier_weighted(a=1,k=8) | 0.07 | 0.225 | 0.79 | 2/2/2 | 0.09 | 1.11 | 1/5 |  |
| frontier_weighted(a=2,k=5) | 0.02 | -0.009 | 1.08 | 1/1/2 | 0.41 | 1.02 | 3/5 |  |
| frontier_weighted(a=2,k=8) | 0.02 | -0.003 | 1.06 | 1/1/2 | 0.41 | 1.02 | 3/5 |  |

## Confirmation reruns (configs passing >= 4/5 checks at the main N grid)

No configuration passed 4 or more checks at the main N grid, so no confirmation rerun was triggered.

## Per-family detail

### local_parents

Best setting: **local_parents(k=32)** -- FAIL

Signature: dimension drifts with N (slope=0.325, threshold 0.044); dimension estimates don't round-agree (d_MM~3, d_mp~4, closest sprinkle d=3)

Widen fraction at N=4000: 0.566 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | valence ratio | checks |
|---|---|---|---|---|---|---|---|
| local_parents(k=2) | 0.25 | 0.104 | 0.23 | 1/2/2 | 0.40 | 1.01 | 1/5 |
| local_parents(k=3) | 0.31 | 0.162 | 0.30 | 2/2/3 | 0.24 | 1.01 | 1/5 |
| local_parents(k=5) | 0.31 | 0.224 | 0.45 | 2/2/3 | 0.11 | 1.00 | 1/5 |
| local_parents(k=8) | 0.32 | 0.282 | 0.56 | 2/3/3 | 0.07 | 1.02 | 1/5 |
| local_parents(k=12) | 0.21 | 0.096 | 0.59 | 3/3/3 | 0.12 | 1.05 | 1/5 |
| local_parents(k=16) | 0.19 | 0.103 | 0.69 | 3/3/3 | 0.16 | 1.11 | 1/5 |
| local_parents(k=24) | 0.18 | 0.259 | 0.71 | 3/3/3 | 0.23 | 1.22 | 2/5 |
| local_parents(k=32) | 0.13 | 0.325 | 0.85 | 3/4/3 | 0.23 | 1.33 | 3/5 |

### local_parents_adaptive

Best setting: **local_parents_adaptive(c=4,beta=0.33)** -- FAIL

Signature: dimension drifts with N (slope=0.166, threshold 0.044); abundance profile far from matching sprinkle (distance=0.19 > 0.06)

Widen fraction at N=4000: 0.575 (fraction of growth steps that branched off below the tip rather than extending it)

Candidate-sample cap hit 0/37480 growth steps (0.0%).

| setting | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | valence ratio | checks |
|---|---|---|---|---|---|---|---|
| local_parents_adaptive(c=1,beta=0.25) | 0.28 | 0.127 | 0.24 | 1/2/2 | 0.38 | 1.15 | 1/5 |
| local_parents_adaptive(c=2,beta=0.25) | 0.34 | 0.274 | 0.41 | 2/2/3 | 0.12 | 1.18 | 2/5 |
| local_parents_adaptive(c=4,beta=0.25) | 0.26 | 0.255 | 0.70 | 3/3/3 | 0.08 | 1.26 | 2/5 |
| local_parents_adaptive(c=1,beta=0.33) | 0.27 | 0.188 | 0.28 | 2/2/3 | 0.31 | 1.18 | 2/5 |
| local_parents_adaptive(c=2,beta=0.33) | 0.33 | 0.342 | 0.53 | 2/3/3 | 0.10 | 1.30 | 2/5 |
| local_parents_adaptive(c=4,beta=0.33) | 0.21 | 0.166 | 0.91 | 3/3/3 | 0.19 | 1.43 | 3/5 |
| local_parents_adaptive(c=1,beta=0.5) | 0.35 | 0.384 | 0.49 | 2/3/3 | 0.13 | 1.47 | 1/5 |
| local_parents_adaptive(c=2,beta=0.5) | 0.21 | 0.248 | 0.94 | 3/3/3 | 0.14 | 1.74 | 3/5 |
| local_parents_adaptive(c=4,beta=0.5) | 0.24 | 0.164 | 1.31 | 4/4/3 | 0.25 | 1.64 | 2/5 |

### frontier_weighted

Best setting: **frontier_weighted(a=2,k=8)** -- FAIL

Signature: dimension estimates don't round-agree (d_MM~1, d_mp~1, closest sprinkle d=2); valence flat (ratio=1.02 < 1.15)

Widen fraction at N=4000: 0.464 (fraction of growth steps that branched off below the tip rather than extending it)

| setting | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | valence ratio | checks |
|---|---|---|---|---|---|---|---|
| frontier_weighted(a=0.5,k=5) | 0.18 | 0.227 | 0.54 | 2/2/2 | 0.23 | 1.02 | 1/5 |
| frontier_weighted(a=0.5,k=8) | 0.17 | 0.258 | 0.69 | 2/2/2 | 0.26 | 1.04 | 1/5 |
| frontier_weighted(a=1,k=5) | 0.08 | 0.209 | 0.74 | 2/2/2 | 0.11 | 1.05 | 1/5 |
| frontier_weighted(a=1,k=8) | 0.07 | 0.225 | 0.79 | 2/2/2 | 0.09 | 1.11 | 1/5 |
| frontier_weighted(a=2,k=5) | 0.02 | -0.009 | 1.08 | 1/1/2 | 0.41 | 1.02 | 3/5 |
| frontier_weighted(a=2,k=8) | 0.02 | -0.003 | 1.06 | 1/1/2 | 0.41 | 1.02 | 3/5 |

## Plots

- `followup_dimension_vs_N_overview.png` / `followup_dimension_vs_N_<family>.png`
- `followup_frontier_width_<family>.png` -- frontier width vs n during growth, log-log
- `followup_valence_vs_N_overview.png` -- link valence vs N, every rule and sprinkle
- `followup_local_parents_distance_vs_k.png` -- inner-structure distance vs k, k=2..32, extending the k=2,3,5,8 points from rules_report.md
- `hasse_followup_best_<family>.png` -- Hasse diagram for the best setting of each family, N=100
