# Phase 3b: fast incremental updates + multicanonical sampling

## Summary

Goal: replace the Phase 3 beta-by-beta parallel-tempering scan (which found most located beta_c values right-censored, and at eps=0.21 not even stable under a blind grid-recentering test -- see `results/phase3/mcmc_2d_report.md`) with a faster incremental update engine plus Wang-Landau/multicanonical (WL/MUCA) sampling, which locates beta_c and reweights P_beta(S) from one run instead of a beta-by-beta scan.

The engine, checkpointing, and WL/MUCA pipeline are built, tested, and validated end-to-end at N=30 and N=40, eps=0.5. **Per the decision after the N=40 pilot, the study stops here**: the goal was validating the machinery, not a full N=30..60 grid, and N=40 already shows the mixing cost rising fast enough that N=50/60 would need substantially more compute for uncertain additional insight. eps=0.21 was not attempted (N=30/40 at eps=0.5 already used the full session's compute budget).

## Part 1: fast incremental updates

Bitset-based (uint64, N<=64) incremental action updates (`mcmc/bitset_core.py`): interval size for x<y is `popcount(future[x] & past[y])`; a coordinate swap only changes relations involving the two swapped elements, update via 4 scalar snapshots rather than any O(N) array copy. Validated bit-for-bit against full recomputation over 1e5 random moves across several N and eps (`tests/test_bitset_core.py`).

**Speedup vs the existing matrix-based engine**: 1.3-1.4x (N=20: 1.42x, N=40: 1.34x, N=60: 1.31x) -- real but modest, since both engines share the same O(N) per-move delta scan; the win is removing the O(N) inner loop for endpoint-pair interval-size refresh, not an asymptotic improvement.

## Part 2: checkpointing

`mcmc/checkpoint.py`: atomic pickle writes (`.tmp` + `Path.replace`) of configuration, the checkpointable splitmix64 RNG state (`mcmc/rng.py`), and WL/MUCA weights/histograms. Exercised in practice multiple times during this study (resumed after intentionally interrupting runs) and worked correctly.

## Part 3: Wang-Landau, then multicanonical

`mcmc/muca_core.py` (jitted sweep kernels) and `mcmc/muca.py` (orchestration). Binning starts from a pilot-sample-informed range with margin and widens on edge hits. WL flattens the visit histogram adaptively (stage-doubling modification factor); MUCA production then freezes the weight and reweights to any beta (Berg & Neuhaus, Phys. Lett. B 267 (1991) 249).

**Diagnostics implemented**: histogram flatness (judged over the *actually reachable* bins via stall-detection, since finite N gives the action spectrum genuine permanent gaps, not just slow-to-reach bins); round trips between the reachable extremes; a hidden-barrier check comparing structural (height) round trips to S round trips.

**A correctness gap found and fixed during this study**: the original `reweight_P_beta(ln_g, ...)` assumed `ln_g` already equals the true ln(density of states) -- true only once WL is fully converged. Added `reweight_P_beta_corrected`, which folds in the actual MUCA production histogram `H_muca(S)` (the standard Berg-Neuhaus correction, `P(S) ~ H_muca(S) * exp(ln_g(S) - beta*S)`), which stays exact even with an imperfect `ln_g` provided the walk actually visits the region being compared. This is now used throughout (`locate_beta_c_equal_height`, `reweight_observable`, plotting). A second gap: the peak finder used to accept any two local maxima regardless of the dip between them, which produced a false "double peak" from pure noise during the N=30 reanalysis (barrier only 0.004-0.07 across an 800-point beta scan -- a <7% dip, not a first-order signature). Fixed by requiring `barrier >= MIN_BARRIER_FOR_BIMODAL = 0.5` before accepting a two-peak structure as real.

**beta_c definition**: checked directly against Glaser, O'Connor & Surya (2018), arXiv:1706.06432 (Sec. 4.1) -- they locate beta_c as the **specific-heat / action-variance maximum**, not the equal-peak-height double-peak criterion. `locate_beta_c_variance_peak` (with parabolic interpolation around the discrete grid maximum, to avoid a bootstrap floor-ing its own error at the scan's grid spacing) is the number compared against their formula; equal-height remains this project's own separate first-order diagnostic.

**Multicanonical recursion** (`refine_ln_g_production`, `run_muca_recursion`): an alternative to chasing full WL flatness -- alternate short frozen-weight production runs with a (smoothed, clipped) weight update from the run's own histogram (`ln_g += ln(H_muca)`, relative to the visited-bin mean, clipped to +/-2 per step), stopping once a short run achieves a target round-trip count on its own. Cheaper than waiting out a WL stall whenever `reweight_P_beta_corrected` can already account for the imperfect weight exactly, provided the walk tunnels at all.

## Pilot: N=30, eps=0.5

**Runtime**: ~1700s Wang-Landau (dominated by a stage-16 flatness stall -- see below) + 508s MUCA production (10 round trips, 49.6M moves) -- about 37 minutes end to end for this one (N, eps, seed).

**Range check**: the WL walk piled up anomalously (9x its neighbors) in the lowest reachable bin (S=-390.98). Checked directly: the exact hand-built bilayer ground state's action is S=-390.00 (`sampler.layered_start`, n_layers=2; 3/4/5/6-layer alternatives give -345/-291/-243/-200, all higher) -- this is the true floor, not a truncation artifact. The pile-up is mechanistic: a hard reflecting boundary is approached from only one side, so WL is slow to flatten there -- exactly why stage 16 stalled.

**beta_c**: a naive coarse (7-point) scan of the equal-height criterion found a "double peak" at only one tested point, and `locate_beta_c_equal_height`'s original bisection (narrowing from the search bracket's own endpoints) missed it entirely, returning `None`. A fine (600-800 point) scan found the "double peak" is real in the sense of being findable on a fine grid, but its barrier (0.004-0.07 across the whole scanned range) is noise on a single broad hump, not a first-order signature -- see the `MIN_BARRIER_FOR_BIMODAL` fix above.

Using the correct (variance-peak) definition instead: **beta_c = 0.1478 +/- 0.0004** (300-replicate block bootstrap over 11 independent ~1-round-trip blocks, parabolic-interpolated). Compared to the formula's prediction (0.13427 +/- 0.01576): **z = 0.86, PASS**. Independently cross-checked against the earlier Phase 3 PT study's own variance-peak measurement for this exact (N, eps): **0.14797** -- agreement to within 0.001 between two different sampling methods.

Equal-height: no significant barrier found anywhere in a wide beta scan (consistent with the Phase 3 PT report's own hysteresis check at this beta_c showing z=12.42 disagreement between random-start and layered-start chains -- i.e. that study's "double peak" was not confirmed equilibrium sampling either).

## Pilot: N=40, eps=0.5

**Runtime and the recursion-vs-WL tradeoff**:

| Step | Time | Outcome |
|---|---|---|
| Wang-Landau (reused checkpoint, stopped once f <= 1e-2 per the adjusted plan) | ~1750s | stage 16, f=1.53e-5, 126/300 bins reachable, flatness stuck at ~0.84 |
| Trim provably-unreachable lower bins | ~0s | 300 -> 170 bins (dead bins only; see range note below) |
| Multicanonical recursion (8 iterations, target 5 round trips/iteration) | 3854.5s | **did not converge** -- round trips per iteration (2,1,2,0,0,0,2,0) showed no upward trend; the ~1-2 round trips per 30M moves looks like a roughly fixed physical mixing rate at this N, not a weight-quality problem recursion could fix in a few steps |
| Parallel production, 4 workers, unwindowed round-trip counting, 60M moves/worker | 1008.1s | only 6 total round trips (all 4 workers hit the move cap before their target) |
| **Windowed re-run** (same weights, same 60M/worker budget, round trips counted only within the physically relevant window -- see below) | 457.3s | **12 total round trips, all 4 workers hit target** -- 4x the round trips in under half the time |

**Range check for N=40**: reweighting the existing data at beta = 2x the predicted beta_c showed the "cold" peak sitting only 7 bins above the lowest reachable bin, which itself is not negligible there -- unlike the naive expectation, the floor and the physically relevant region coincide here, so only the provably-dead (zero-visit) bins below the floor were trimmed, not the floor itself.

**The real fix -- round-trip counting window, not sampling range**: reweighting the existing (pre-windowed) data showed bins [0,85] of the reachable 126 carry essentially all of P(S)'s mass for beta in [0.5, 2.0]x the predicted beta_c; bins [86,125] (up to S=+402) are irrelevant to any beta actually being compared against the formula. The original round-trip counting required the walk to reach bin 125, which it rarely did -- most of the "0 round trip" iterations were likely completing physically meaningful excursions that just didn't count under the overly strict full-span criterion. Recounting (i.e. a fresh short run with `round_trip_window=(0, 85)` instead of the default full `ever_visited` span) confirmed this directly: 4x more round trips in under half the wall-clock time, for the same underlying physics and the same sampling range (only the counting criterion changed).

**Note on reproducibility of the per-worker spread**: the first parallel-production attempt only persisted the *merged* histogram, not each worker's own histogram or time series -- discovered when asked to report a per-worker beta_c spread and finding the data had not been kept. Fixed (`run_muca_production_parallel` now returns a `per_worker` list with each worker's own `H_muca`/`round_trips`/time series) before the windowed re-run, which is why the windowed run's per-worker spread below is trustworthy and the original (unwindowed) run's is not reconstructable after the fact.

**beta_c (variance-peak, from the windowed run's 4 independent workers)**:

| Worker | Round trips | beta_c |
|---|---|---|
| 0 | 3 | 0.12270 |
| 1 | 3 | 0.12225 |
| 2 | 3 | 0.12303 |
| 3 | 3 | 0.12029 |

**Mean = 0.12207 +/- 0.00106** (std across the 4 independent workers). Compared to the formula's prediction (0.11702 +/- 0.00962 at N=40): **z = 0.52, PASS**, and comfortably precise enough (std is ~0.9% of the mean) not to need the larger window-restricted-sampling escalation that would otherwise have been the next step.

Equal-height: still `None` (no significant barrier), even on the improved windowed statistics -- same situation as N=30.

## Known limitations

1. **No clear equal-height first-order barrier at either N tested.** Both N=30 and N=40 pass the variance-peak beta_c check against the formula, but neither shows a resolved double-peak/barrier signature above the noise floor (`MIN_BARRIER_FOR_BIMODAL = 0.5`). This could mean the transition is weak/crossover-like at these N (plausible -- N=30/40 are well below the paper's own stated asymptotic regime, N >~ 65), or that still more statistics are needed to resolve a real but small barrier. **Cannot distinguish these from the data collected.** Part 5's "barrier growing with N" pass criterion is therefore not assessable from this study (only one of the two N tested would need a resolved barrier at all to even start a trend, and neither did).
2. **Mixing cost rises steeply from N=30 to N=40.** N=30 reached 10 round trips in 508s of production after WL; N=40 needed the windowed-counting fix just to reach 12 round trips in 457s *after* a WL phase that itself cost ~1750s and a recursion phase that cost 3854.5s without converging. Extrapolating this trend, N=50 and N=60 would likely need substantially more wall-clock time (and possibly further methodological fixes, e.g. the recursion plateau would need diagnosing rather than assumed-away) -- not attempted in this study.
3. **eps=0.21 not attempted.** The full session's compute budget went to validating the pipeline at eps=0.5 for N=30 and N=40; part 6's eps=0.21, beta~7 side note has no data.
4. **Single seed per N.** Only seed=0 was run at each N (plus the 4 independent parallel workers at N=40, which give a real but narrow-based spread). No seed-to-seed variation was characterized.

## Pass criteria (part 5) -- assessed honestly against what was actually run

1. **Clean double peak at beta_c that separates with N**: not met at either N tested (equal-height found `None` at both).
2. **beta_c within error of the formula (using the variance-peak definition, which is what the formula was fit to)**: met at both N tested (N=30: z=0.86; N=40: z=0.52).
3. **At least 10 round trips per run**: met (N=30: 10; N=40: 12, after the windowed-counting fix -- the original unwindowed count of 6 did not meet this).
4. **No hidden-barrier warning**: met at both N tested.

**Overall: partial.** The sampling engine, checkpointing, and WL/MUCA/reweighting machinery all work correctly and were validated against an independent prior measurement (N=30's variance-peak beta_c matches the Phase 3 PT study's own value to within 0.001). But the first-order signature itself (criterion 1, the actual physics claim) was not confirmed at either N -- this is reported as a limitation of the available statistics at N<=40, not papered over as a pass.

Plot: `muca_P_beta_c_S_eps0p50.png` (P_beta_c(S) for N=30 and N=40, eps=0.5 -- both single, broad humps, not double peaks, consistent with criterion 1 above).
