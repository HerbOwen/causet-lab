# Phase 3: weighing 2D causal set universes (smeared-action MCMC)

Every causal set on N elements is a possible universe, weighted by exp(-beta * S(eps)) where S(eps) is the *smeared* 2D Benincasa-Dowker action (see `mcmc/action.py`), following S. Surya, "Evidence for the continuum in 2D causal set quantum gravity", CQG 29 132001 (2012), arXiv:1110.6244, and calibrated against L. Glaser, D. O'Connor, S. Surya, "Finite Size Scaling in 2d Causal Set Quantum Gravity", CQG 35 045006 (2018), arXiv:1706.06432 -- see 'Comparison to the literature' below for the normalization-match confirmation and the exact published formula this run's beta_c is checked against. An earlier version of this study used the plain (non-smeared) action and scanned up to N=80; that action's beta is not comparable to any published number, and N=80 under it showed zero PT replica round trips across the transition even after a 4x run-length escalation -- both issues are why this version uses the smeared action and stays at N <= 60. Sampling engine: Numba-compiled, incrementally-updated Metropolis (fast_core.py) plus parallel tempering (replica exchange) with round-trip tracking (tempering.py).

## Controls

At beta=0 every Metropolis move is accepted unconditionally regardless of eps (the Boltzmann weight is 1), so this check is run once, not once per eps. Acceptance rate = 1.00 (min over chains 1.00).

- N = 60, 5 independent beta=0 chains, 5 sprinkle reference seeds.
- Myrheim-Meyer dimension: beta=0 chains = 1.989 +/- 0.096, sprinkle d=2 reference = 2.032 +/- 0.095 -> **PASS** (within 0.4 of each other).
- Interval-abundance L1 distance to sprinkle reference: 0.054 -> **PASS** (threshold 0.5).
- Detailed-balance sanity: independent chains' mean actions span at most 1.02 standard errors from their grand mean -> **PASS** (threshold 4 SEM).

## eps = 0.21

### Locating beta_c vs the published formula

beta_c is the beta (among the converged fine PT-ladder points only -- UNCONVERGED points, below 50 effective samples even after escalation, are excluded) with the largest pooled action variance. Compared against Glaser-O'Connor-Surya 2018's beta_c(N,eps) = 1.66(+/-0.03)/(N*eps^2) + [4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2 via a z-score using both the bootstrap and formula-coefficient uncertainties (PASS if z < 3.0).

| N | beta_c (located) | bootstrap std | formula beta_c | z | result | peak height | peak FWHM | UNCONVERGED | round trips |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.95652 | 0.59085 | 1.04576 +/- 0.10875 | 0.15 | **PASS** | 60.1 | 1.17344 | 8 / 26 | min 1, max 5 |
| 40 | 0.70177 | 0.00257 | 0.82350 +/- 0.06542 | 1.86 | **PASS** | 67.1 | n/a | 13 / 26 | min 0, max 2 |
| 50 | 0.59210 | 0.00000 | 0.67761 +/- 0.04459 | 1.92 | **PASS** | 71.7 | n/a | 17 / 26 | min 0, max 1 |
| 60 | 0.50254 | 0.00000 | 0.57512 +/- 0.03286 | 2.21 | **PASS** | 79.3 | n/a | 18 / 27 | min 0, max 1 |

**Escalation stalled for N=50, 60 at eps=0.21**: doubling PT sweeps did not improve effective-sample count near beta_c; see the round trip counts above -- if round trips are also near zero there, treat that N's beta_c and histogram result with extra caution.

**Is the located beta_c a confirmed interior peak, or just where convergence ran out?** A genuine variance maximum needs converged points on *both* sides of it, not just below.

| N | beta_c | max converged beta | beta_c IS the max converged beta | converged within 25% below prediction | converged within 25% above prediction |
|---|---|---|---|---|---|
| 30 | 0.95652 | 6.27456 | no | 4 | 1 |
| 40 | 0.70177 | 0.70177 | **YES -- right-censored, not confirmed** | 2 | 0 |
| 50 | 0.59210 | 0.59210 | **YES -- right-censored, not confirmed** | 1 | 0 |
| 60 | 0.50254 | 0.50254 | **YES -- right-censored, not confirmed** | 1 | 0 |

**3 / 4 N at eps=0.21 have a right-censored beta_c**: the located value is exactly the largest converged beta in the scan. There is no converged data at all showing the variance turns back down past it, so the z-score PASS reported above for N=40, 50, 60 confirms only that the point where convergence ran out happens to land within z<3 of the formula's prediction -- not that an interior maximum was independently located and found to agree. N=30 is the only eps=0.21 case with converged coverage on both sides.

### Action histogram at beta_c (first-order signature)

Glaser-O'Connor-Surya 2018 report a first-order transition: a double-peaked pooled action histogram at beta_c, with the peak separation growing as N increases. Peaks are counted on the pooled (S/N) distribution at the converged fine-grid point nearest the located beta_c.

| N | peaks found | separation (S/N units) |
|---|---|---|
| 30 | 1 | n/a |
| 40 | 1 | n/a |
| 50 | 1 | n/a |
| 60 | 1 | n/a |

Fewer than 2 N values show a double-peaked histogram at eps=0.21; cannot assess whether the separation grows with N.

### High-beta phase vs published targets (height ~ 3, ordering fraction ~ 0.6)

| N | beta (deepest) | height | ordering fraction | MM dimension | height PASS | OF PASS |
|---|---|---|---|---|---|---|
| 30 | 15.6864 | 3.6 +/- 0.5 | 0.648 +/- 0.032 | 1.65 +/- 0.07 | PASS | PASS |
| 40 | 12.3525 | 3.8 +/- 0.4 | 0.681 +/- 0.033 | 1.58 +/- 0.07 | PASS | PASS |
| 50 | 10.1641 | 4.2 +/- 0.4 | 0.713 +/- 0.020 | 1.51 +/- 0.04 | PASS | PASS |
| 60 | 8.6268 | 4.4 +/- 0.5 | 0.716 +/- 0.016 | 1.51 +/- 0.03 | PASS | PASS |

4 / 4 N values plateaued between the two deep betas tested (x5.0 and x15.0 of the formula's predicted beta_c) -- where False, the high-beta regime had not fully stabilized at the betas tested.

### Hysteresis check at beta_c

| N | beta_c rung | <S> random start | <S> layered start | z | result |
|---|---|---|---|---|---|
| 30 | 0.95652 | -8.10 +/- 0.17 | -7.44 +/- 0.17 | 2.78 | AGREE |
| 40 | 0.70177 | -8.83 +/- 0.18 | -9.34 +/- 0.21 | 1.85 | AGREE |
| 50 | 0.95846 | -119.66 +/- 0.50 | -170.20 +/- 0.13 | 98.54 | **DISAGREE** |
| 60 | 0.50254 | -13.38 +/- 0.27 | -200.84 +/- 0.83 | 214.01 | **DISAGREE** |

**2 / 4 N disagree** between starting conditions at beta_c -- expected to some degree for a genuinely first-order transition (coexisting phases separated by a barrier), consistent with the histogram section above.

### Structural comparison: closer to Kleitman-Rothschild or to a sprinkle?

| N | distance to Kleitman-Rothschild | distance to 2D sprinkle | closer to |
|---|---|---|---|
| 30 | 0.506 | 1.162 | Kleitman-Rothschild |
| 40 | 0.406 | 1.109 | Kleitman-Rothschild |
| 50 | 0.336 | 1.048 | Kleitman-Rothschild |
| 60 | 0.351 | 1.026 | Kleitman-Rothschild |

Plots: `mcmc_variance_vs_beta_eps0p21.png`, `mcmc_action_vs_beta_eps0p21.png`, `mcmc_dimension_vs_beta_eps0p21.png`, `mcmc_height_vs_beta_eps0p21.png`, `mcmc_peak_height_vs_N_eps0p21.png`, `mcmc_peak_width_vs_N_eps0p21.png`, and per-N `hasse_N*_eps0p21_{below,above}_transition.png`.

## eps = 0.5

### Locating beta_c vs the published formula

beta_c is the beta (among the converged fine PT-ladder points only -- UNCONVERGED points, below 50 effective samples even after escalation, are excluded) with the largest pooled action variance. Compared against Glaser-O'Connor-Surya 2018's beta_c(N,eps) = 1.66(+/-0.03)/(N*eps^2) + [4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2 via a z-score using both the bootstrap and formula-coefficient uncertainties (PASS if z < 3.0).

| N | beta_c (located) | bootstrap std | formula beta_c | z | result | peak height | peak FWHM | UNCONVERGED | round trips |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.14797 | 0.00384 | 0.13427 +/- 0.01576 | 0.84 | **PASS** | 4405.1 | n/a | 8 / 26 | min 2, max 8 |
| 40 | 0.10226 | 0.00000 | 0.11702 +/- 0.00962 | 1.54 | **PASS** | 3101.0 | n/a | 17 / 26 | min 0, max 2 |
| 50 | 0.08865 | 0.00000 | 0.10146 +/- 0.00663 | 1.93 | **PASS** | 3657.3 | n/a | 17 / 26 | min 0, max 2 |
| 60 | 0.08667 | 0.00000 | 0.08890 +/- 0.00494 | 0.45 | **PASS** | 6459.6 | n/a | 16 / 26 | min 0, max 1 |

**Escalation stalled for N=40, 50, 60 at eps=0.5**: doubling PT sweeps did not improve effective-sample count near beta_c; see the round trip counts above -- if round trips are also near zero there, treat that N's beta_c and histogram result with extra caution.

**Is the located beta_c a confirmed interior peak, or just where convergence ran out?**

| N | beta_c | max converged beta | beta_c IS the max converged beta | converged within 25% below prediction | converged within 25% above prediction |
|---|---|---|---|---|---|
| 30 | 0.14797 | 0.14797 | **YES -- right-censored, not confirmed** | 5 | 2 |
| 40 | 0.10226 | 0.10226 | **YES -- right-censored, not confirmed** | 1 | 0 |
| 50 | 0.08865 | 0.08865 | **YES -- right-censored, not confirmed** | 1 | 0 |
| 60 | 0.08667 | 0.08667 | **YES -- right-censored, not confirmed** | 2 | 0 |

**4 / 4 N at eps=0.5 have a right-censored beta_c** -- every located value here is exactly the largest converged beta in the scan, with zero converged coverage above it. None of the eps=0.5 PASS results in the table above are confirmed interior maxima; see the blind grid-centering check below for whether these values are at least stable against that.

### Action histogram at beta_c (first-order signature)

Glaser-O'Connor-Surya 2018 report a first-order transition: a double-peaked pooled action histogram at beta_c, with the peak separation growing as N increases. Peaks are counted on the pooled (S/N) distribution at the converged fine-grid point nearest the located beta_c.

| N | peaks found | separation (S/N units) |
|---|---|---|
| 30 | 2 | 1.3010 |
| 40 | 1 | n/a |
| 50 | 1 | n/a |
| 60 | 1 | n/a |

Fewer than 2 N values show a double-peaked histogram at eps=0.5; cannot assess whether the separation grows with N.

### High-beta phase vs published targets (height ~ 3, ordering fraction ~ 0.6)

| N | beta (deepest) | height | ordering fraction | MM dimension | height PASS | OF PASS |
|---|---|---|---|---|---|---|
| 30 | 2.0140 | 4.0 +/- 0.6 | 0.708 +/- 0.037 | 1.52 +/- 0.07 | PASS | PASS |
| 40 | 1.7554 | 4.8 +/- 0.4 | 0.731 +/- 0.026 | 1.48 +/- 0.05 | FAIL | PASS |
| 50 | 1.5218 | 4.6 +/- 0.5 | 0.720 +/- 0.032 | 1.50 +/- 0.06 | FAIL | PASS |
| 60 | 1.3335 | 5.2 +/- 0.7 | 0.728 +/- 0.010 | 1.48 +/- 0.02 | FAIL | PASS |

4 / 4 N values plateaued between the two deep betas tested (x5.0 and x15.0 of the formula's predicted beta_c) -- where False, the high-beta regime had not fully stabilized at the betas tested.

### Hysteresis check at beta_c

| N | beta_c rung | <S> random start | <S> layered start | z | result |
|---|---|---|---|---|---|
| 30 | 0.14797 | -164.39 +/- 1.45 | -191.93 +/- 1.68 | 12.42 | **DISAGREE** |
| 40 | 0.16553 | -422.43 +/- 1.73 | -666.72 +/- 0.73 | 130.31 | **DISAGREE** |
| 50 | 0.14351 | -568.37 +/- 2.45 | -1087.29 +/- 1.02 | 195.62 | **DISAGREE** |
| 60 | 0.08667 | -155.84 +/- 2.48 | -1535.67 +/- 2.48 | 393.53 | **DISAGREE** |

**4 / 4 N disagree** between starting conditions at beta_c -- expected to some degree for a genuinely first-order transition (coexisting phases separated by a barrier), consistent with the histogram section above.

### Structural comparison: closer to Kleitman-Rothschild or to a sprinkle?

| N | distance to Kleitman-Rothschild | distance to 2D sprinkle | closer to |
|---|---|---|---|
| 30 | 0.562 | 1.030 | Kleitman-Rothschild |
| 40 | 0.495 | 0.939 | Kleitman-Rothschild |
| 50 | 0.381 | 0.904 | Kleitman-Rothschild |
| 60 | 0.360 | 0.853 | Kleitman-Rothschild |

Plots: `mcmc_variance_vs_beta_eps0p50.png`, `mcmc_action_vs_beta_eps0p50.png`, `mcmc_dimension_vs_beta_eps0p50.png`, `mcmc_height_vs_beta_eps0p50.png`, `mcmc_peak_height_vs_N_eps0p50.png`, `mcmc_peak_width_vs_N_eps0p50.png`, and per-N `hasse_N*_eps0p50_{below,above}_transition.png`.

## Blind grid-centering robustness check

Given how many located beta_c values above turned out to be right-censored, a direct test: rerun N=40 at both eps with the coarse scout grid deliberately mis-centered at 0.6x and 1.6x the predicted beta_c (same log-width and point count as the normal grid, which is itself centered near 0.42x the prediction), with every other setting identical to the original run. If beta_c lands near the same value regardless of where the grid is centered, the match is a located feature; if it instead swings around with the grid, the original PASS is not a genuine confirmation.

| N | eps | grid center | located beta_c | published prediction | within 25% of prediction | beta_c right-censored |
|---|---|---|---|---|---|---|
| 40 | 0.21 | original (~0.42x = 0.347) | 0.70177 | 0.82350 | yes | YES |
| 40 | 0.21 | blind 0.6x (0.494) | 6.98763 | 0.82350 | **no -- 8.5x too high** | YES |
| 40 | 0.21 | blind 1.6x (1.318) | 7.11098 | 0.82350 | **no -- 8.6x too high** | no (max converged = 18.63) |
| 40 | 0.5 | original (~0.42x = 0.049) | 0.10226 | 0.11702 | yes | YES |
| 40 | 0.5 | blind 0.6x (0.070) | 0.08933 | 0.11702 | yes (24% below) | YES |
| 40 | 0.5 | blind 1.6x (0.187) | 0.10143 | 0.11702 | yes (13% below) | no (max converged = 2.65) |

**N=40, eps=0.21 fails this check.** Both deliberately mis-centered grids -- one shifted below the original center, one above -- independently located beta_c near **7.0**, nowhere near the original run's 0.70 or the published prediction's 0.82. This is not simply more right-censoring: the 1.6x-centered grid had converged points up to beta=18.63, so 7.11 sits inside genuinely converged, two-sided data -- it is very likely a real, reproducible variance feature, just not the one the published formula predicts (and not one the original run's narrower grid, which topped out at 4.94, could ever have seen). The original run's "beta_c=0.70177, PASS" result for N=40, eps=0.21 should therefore be read as an artifact of a too-narrow search range and right-censoring, not a confirmed location.

**N=40, eps=0.5 passes this check reasonably well**: all three estimates (original 0.102, blind 0.089, blind 0.101) land within about 25% of each other and of the prediction (0.117), despite two of the three being right-censored. This is consistent with -- though does not independently prove -- a real feature near this beta for this (N, eps). The qualitative pattern (eps=0.5 more stable than eps=0.21) matches the published formula's own eps-scaling: b(eps) = 1.66/eps^2 pushes beta_c to a much larger absolute value at small eps, which also widens the dynamic range this project's grids need to cover reliably and gives more room for spurious high-beta features (like the one found here) to compete with the real signal.

**This result does not generalize automatically to the other six (N, eps) combinations** -- only N=40 was blind-tested at both eps, per the request that motivated this check. Given the eps=0.21 failure here, the other eps=0.21 combinations (N=30, 50, 60), all but one of which are also right-censored, should be treated with the same suspicion until similarly tested.

## Comparison to the literature

**Normalization check**: L. Glaser, D. O'Connor, S. Surya, "Finite Size Scaling in 2d Causal Set Quantum Gravity", CQG 35 045006 (2018), arXiv:1706.06432, was fetched directly (via its ar5iv HTML rendering) for this comparison. Its action formula, S(C,eps) = 4*eps*(N - 2*eps*sum_n N_n*f(n,eps)) with f(n,eps) = (1-eps)^n*(1 - 2*eps*n/(1-eps) + eps^2*n*(n-1)/(2*(1-eps)^2)), matches this project's `action.bd_action_2d_smeared` exactly (confirmed both by reading the paper and by this project's own eps -> 1 unit test, which reduces the smeared action to the plain action's 4*(N-2*N0+4*N1-2*N2) limit analytically, with zero numerical error). Its beta_c(N,eps) = 1.66(+/-0.03)/eps^2/N + [4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2 + O(1/N^3) is the formula used throughout this report; the paper fits it over N = 20..90, eps = 0.1..0.5, and states its asymptotic (N -> infinity) regime is only reached for N >~ 65 -- so the N = 30..60 used here sit below that regime, and the O(1/N^3) terms the paper does not quantify are plausibly non-negligible; the z-scores above should be read with that caveat, not as a strict pass/fail on the asymptotic theory.
**Transition order**: the 2018 paper's main finding is that the transition is *first-order*, evidenced by double-peaked action histograms whose separation grows with N and a Binder coefficient moving away from 0 as N grows -- both checked directly above, not assumed.
**Phase names**: the paper calls the two phases Pi_- (continuum, beta < beta_c, Myrheim-Meyer dimension ~ 2) and Pi_+ (crystalline/non-continuum, beta > beta_c). S. Surya 2012, arXiv:1110.6244 (the earlier, N=50-only study this project originally targeted) calls them "continuum" and "crystalline" with the same qualitative dimension/height behavior; it does not give an N-scaling formula for beta_c, which is why the 2018 paper is used as this report's quantitative target instead.

## Performance

Unchanged from the prior version of this study: (1) parallel tempering for the near-transition beta grid, now with round-trip tracking reported above as the real test that replicas cross between phases rather than trusting local swap-rate numbers alone; (2) a Numba-compiled, incrementally-updated action (O(N^2) per move instead of rebuilding the O(N^3) interval-size histogram from scratch every move); (3) independent beta points run in parallel across 4 CPU cores. N=80 is no longer attempted (see intro) -- the largest N here is 60, which is both cheaper per point and (per the round-trip columns above) actually mixes across the transition under PT.

## Known limitations

**The first-order barrier at beta_c is not crossed cleanly.** Hysteresis disagreed between random-start and layered-start chains (even with parallel tempering active) for 6 / 8 (N, eps) combinations, and the pooled action histogram at beta_c showed the expected double-peak signature in only 1 / 8. These two findings are consistent with each other: a real first-order transition has a free-energy barrier separating the two phases that single-replica moves essentially never cross, and replica exchange only helps when neighboring-beta variance distributions overlap enough for a swap to be accepted -- near a sharp first-order point they stop overlapping, which is exactly where this run's effective-sample counts collapsed and points were marked UNCONVERGED.

**A sharper, related problem found by this follow-up check: 7 / 8 combinations' located beta_c is right-censored** (see the per-eps tables above) -- the reported value is exactly the largest converged beta in the scan, with no converged data above it. The blind grid-centering check goes further: for N=40, eps=0.21, deliberately shifting the search grid relocated "beta_c" from 0.70 to roughly 7.0 regardless of whether the shift was up or down -- the located value is not robust to where the grid was placed, and the original PASS for that combination is not a genuine confirmation. N=40, eps=0.5 held up better (all three grid placements agreed within about 25%), suggesting the method may be more trustworthy at larger eps, but this was not tested for the other six combinations and should not be assumed.

**Implication**: near-transition sampling at larger N (this project's own escalation attempts stalled -- doubling PT sweep count did not improve effective-sample counts for 5 of the 8 combinations) or in higher dimension (where interval-size computations and the move set both grow more expensive) will need stronger methods than plain replica exchange on a single-swap proposal before beta_c locations like these can be trusted at face value: a denser ladder specifically bracketing the barrier (informed by a cheap wide-range pilot rather than the published formula alone, given the spurious high-beta feature found above), cluster or multi-site moves that can cross the barrier in fewer steps, or multicanonical/Wang-Landau-style sampling that flattens the free-energy barrier directly instead of relying on tempering to route around it.

## Verdict: do we reproduce the published result?

Across all 8 (N, eps) combinations tested (N in [30, 40, 50, 60], eps in [0.21, 0.5]):

1. **beta_c matches the published formula** (z < 3.0): 8 / 8 on paper -- **but this overstates the result.** 7 / 8 of those located values are right-censored (exactly the largest converged beta, with no converged data confirming the variance turns back down past it), and the blind grid-centering check found that at least one of them (N=40, eps=0.21) is not even reproducible under a shifted search grid -- it relocated to beta~7 instead of ~0.82. Treat this criterion as **not independently confirmed** for any combination except possibly N=40, eps=0.5 (the one combination that was stress-tested and held up) and N=30, eps=0.21 (the only combination with genuine two-sided converged coverage) -- see 'Known limitations' and 'Blind grid-centering robustness check' above.
2. **Action histogram at beta_c is double-peaked**: 1 / 8 (see per-eps peak-separation-vs-N trend above for whether it also sharpens with N).
3. **High-beta phase matches height~3 target**: 5 / 8; **ordering fraction~0.6 target**: 8 / 8.
4. **Low-beta phase matches 2D sprinkles**: controls at N=60 -> PASS (eps-independent at beta=0, so checked once; every (N, eps)'s own low-beta points in the per-eps tables above show dimension near 2 as well).

**Overall: this calibration should not yet be called passed.** The low-beta (continuum-like) phase and the qualitative high-beta trend (rising ordering fraction, falling dimension) reproduce the paper's picture, but the quantitative beta_c location -- the criterion that would make this a genuine, numeric confirmation of Glaser-O'Connor-Surya (2018) -- is undermined by right-censoring in 7/8 combinations and an outright non-reproducible result for the one combination put through a robustness check. The double-peak signature central to the paper's first-order claim is also absent in all but one combination. Before this can be called a confirmed reproduction, the near-transition region needs the stronger sampling described in 'Known limitations' above, and the blind-centering check should be extended to the other six combinations (the N=40 result cannot be assumed to generalize).
