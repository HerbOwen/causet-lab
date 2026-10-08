# Phase 5: random vs. regular background, d=2 lattice gas (stage 1: n=30)

## Summary

Cunningham & Surya's (C&S, arXiv:1908.11647) lattice-gas model embeds
causal-set elements on a *regular* lattice background. They flag --
but do not pursue -- the question of whether a "more realistic model
of discreteness" (a randomly sprinkled background instead of a
regular grid) changes the result. This phase builds that comparison:
same region, same filling/move representation, same smeared action,
same eps=0.1 and m=alpha*n^2, same WL/MUCA/checkpointing pipeline as
Phase 4's regular-lattice run, with the m background sites sprinkled
once (quenched) per realization instead of sitting on a grid.

**Stage 1 (this report): n=30, 5 independent background realizations.**
**Result: the transition does NOT survive "same within error" at n=30.**
beta_c is tightly reproducible across realizations (std = 0.058 over 5
backgrounds, about 0.6% of the mean) but sits at **9.723 +/- 0.026
(SEM)**, significantly below the regular lattice's single-realization
**11.118** -- a ~13% shift, many standard errors wide. The qualitative
phase structure (manifold-like hot phase, more-ordered cold phase) is
preserved in both, and the hot phase in particular is close to
background-independent; the cold phase shows a real quantitative
difference (lower ordering fraction, lower height on the random
background). No confirmed first-order double-peak signature in
P_beta_c(S) at any of the 5 realizations, once a spurious
tail-noise "double peak" (found and diagnosed below) is correctly
excluded -- same single-broad-hump picture as every other model in
this project so far. Stage 2 (n=50) is a one-line constant change in
the driver script, not run yet.

## Construction and the units fix

Same cylinder region as the regular lattice (`t` in `[0,h]`, circumference
`w`, `h=4n`, `m=4n^2`), but the m background sites are `m` independent
draws of a continuous position `(t, x)` with `t ~ Uniform(0,h)` and
`x ~ Uniform(0,w)` -- **`x` has period `w` (the lattice's own integer
circumference), not `2*pi` radians.** An initial draft of this model
sprinkled `theta` in actual radians and compared it directly against
`dt` in the lattice's row units, which silently rescales the lightcone
by a factor of `w/(2*pi)` relative to the regular lattice's causal
relation -- caught before any sampling was run, by explicit request to
(1) check units against how `lattice_gas.lattice_to_matrix` actually
defines its relation and (2) add a reduces-to-the-known-case test.
Fixed by sprinkling `x` with period `w` instead of `theta` in radians
(`mcmc/random_bg_core.py`, `mcmc/random_bg.py`).

**Equivalence test**: placing the background at the exact integer
lattice positions (`site s -> t=s//w, x=s%w`) reproduces
`lattice_gas.lattice_to_matrix` bit-for-bit, including the lightlike
(`<=`) convention, across n in {16,30,47} and 5 seeds each
(`tests/test_random_bg.py::test_random_bg_at_lattice_positions_matches_lattice_exactly`,
12/12 tests passed). This is the direct check that the random
background's causal geometry is identical to the regular lattice's,
not merely "the same region" under a different metric normalization.

## Controls

**Incremental vs. full recomputation**: bit-for-bit match over 1e5
moves, n in {16,30,47}, eps in {0.1,0.21}, counts/action both checked
every 500 steps -- same methodology as Phase 4. **Pass.**

**Move self-inverse**: reapplying `apply_relocate_randombg` with the
same `(i,j)` restores state exactly over 50 random moves. **Pass.**

**beta=0 (uniformly random filling, random background) looks like a
2D sprinkle, at least as well as the regular lattice:**

| n | MM dim (sampled intervals) | Abundance-profile L1 distance | Whole-filling ordering fraction |
|---|---|---|---|
| 30 | **2.267** | **0.089** | **0.8768** |
| 50 | **2.061** | **0.060** | **0.8766** |

Compare to the regular lattice's own n=30 beta=0 control: MM dim
2.292, distance 0.100, ordering fraction ~0.882 (Phase 4 report). The
random background matches at least as well on every metric, and the
ordering fractions in particular are nearly identical (0.877 vs.
0.882) -- a direct sanity check that the two models' causal geometries
are genuinely on the same scale, not an artifact of region size alone.
**Pass.**

## Stage-1 run: n=30, 5 background realizations

Each realization sprinkles its own background (seed = 0..4) and runs
the full WL -> MUCA -> beta_c pipeline independently; the background
seed and the Metropolis-chain seed are always distinct (chain seed =
background seed + 10000), so the only thing varying across the 5 runs
is the background geometry. All 5 ran in parallel across 4 CPU cores
(one queued behind the first to finish); total wall time for all 5:
**1623.9s (~27.1 min)**.

| Seed | WL (s) | MUCA (s) | Total (s) | Round trips | Hidden barrier | beta_c |
|---|---|---|---|---|---|---|
| 0 | 1435.5 | 177.9 | 1613.4 | 10 | No | 9.709 |
| 1 | 542.0 | 275.6 | 817.6 | 10 | No | 9.757 |
| 2 | 575.1 | 186.0 | 761.1 | 10 | No | 9.756 |
| 3 | 387.3 | 141.1 | 528.4 | 10 | No | 9.766 |
| 4 | 613.0 | 165.1 | 778.1 | 10 | No | 9.627 |

**beta_c = 9.723 +/- 0.058 (std across 5 realizations), SEM = 0.026.**
Seed 0's WL phase took substantially longer than the others (a late
flatness stall near stage 19, similar in character to the regular
lattice's own stage-9 stall in Phase 4) but converged to the same
beta_c as the other 4 within the observed spread -- the slow run is
not an outlier in its physics, only in its wall-clock cost.

**Compared to the regular lattice's single-realization beta_c = 11.118**
(Phase 4, same n, eps, pipeline): the random-background mean sits
**~1.40 below** the regular lattice's value, roughly **24 SEM-widths**
away using the random-background ensemble's own (tight) cross-seed
uncertainty. **This is not "same within error."** The caveat that
must be stated plainly: the regular lattice has no cross-seed spread
of its own to compare against (it is a single deterministic
background, run with one Metropolis-chain seed in Phase 4), so this
comparison is "one deterministic value" vs. "the mean of 5 quenched
realizations," not two independently-replicated ensembles. Re-running
the regular lattice at a few different chain seeds (its WL/MUCA walk
seed, not a background seed, since the lattice has none) would be a
natural check that 11.118 itself is not sitting on unusually lucky
or unlucky chain-seed noise before leaning harder on the size of this
gap -- not done in this stage.

## Hot-phase vs. cold-phase observables

Hot phase: 5 fresh beta=0 random fillings per background (independent
of the WL/MUCA chain). Cold phase: 5 independent chains per background,
annealed from beta=0 to `4 * (that background's own located beta_c)`
over 400 sweeps, same methodology as Phase 4. Values below are the
mean and std **across the 5 background realizations** (each already
itself averaged over its own 5 within-realization samples):

| Observable | Hot (beta=0) | Cold (4*beta_c) | Regular lattice, hot | Regular lattice, cold |
|---|---|---|---|---|
| MM dimension (sampled intervals) | **2.163 +/- 0.145** | **2.729 +/- 1.041** | 2.18 | 2.87 |
| Ordering fraction (whole filling) | **0.876 +/- 0.007** | **0.583 +/- 0.022** | 0.882 | 0.646 |
| Height (longest chain) | **7.00 +/- 0.51** | **3.44 +/- 0.36** | 7.40 | 4.20 |
| Occupied rows ("layers", of 120) | **26.32 +/- 0.41** | **13.16 +/- 0.61** | 26.7 | 12.8 |

**Hot phase is close to background-independent**: every hot-phase
observable on the random background sits within about 1 std of the
regular lattice's single value, and the layer count in particular
(26.32 vs. 26.7) is nearly identical -- consistent with the hot phase
being locally sprinkle-like regardless of the substrate it is built
on, which is exactly what both backgrounds' own beta=0 controls above
already showed directly.

**Cold phase shows a real, if modest, difference**: the random
background's cold-phase ordering fraction (0.583 +/- 0.022) is
noticeably *lower* than the regular lattice's (0.646) -- in fact
closer to C&S's own asymptotic bilayer value of ~0.5 -- and its height
is lower too (3.44 vs. 4.20), both in the direction of *more* layering
on the random background at the same `4*beta_c` multiple. The MM
dimension's large cross-seed std (1.041) reflects that this diagnostic
is poorly constrained in the strongly-ordered cold regime at n=30 (a
handful of sampled sub-intervals in a near-fully-ordered set is a
noisy dimension estimate) and should be read with that caveat, not as
a precise number.

Because the two models' own located beta_c differ (9.723 vs. 11.118),
"4*beta_c" picks a different absolute beta for each (~38.9 vs. 44.47)
-- so this cold-phase comparison is *not* at a matched absolute
coupling, by design (each model is probed at a comparably-deep point
*relative to its own transition*), and the observed difference should
be read as "the two models' cold phases differ at comparably-deep
points," not as evidence that one model's cold phase is intrinsically
more extreme in some absolute sense.

## P_beta_c(S) shape, and a false double-peak caught during this analysis

A first pass using the existing `muca._peak_diagnostics` (the same
function used for the orders-model and lattice-gas checks) reported a
"double peak" with a large barrier (16.9-21.2, far above the
`MIN_BARRIER_FOR_BIMODAL=0.5` threshold) at 4 of the 5 realizations.
**This is spurious.** Inspecting the raw reweighted `P_beta_c(S)`
arrays directly: in every flagged case, one of the two "peaks" is a
local maximum sitting in the deep exponential tail at a probability
**8 to 11 orders of magnitude below the real mode** (e.g. seed 0:
candidate peaks at `p=3.9e-10` and `p=3.7e-2`) -- a floating-point-
level wiggle on an already-negligible tail, not a second physical
phase. `_peak_diagnostics`'s `peak_val = 0.5*(p[i1]+p[i2])` average is
dominated by the real peak, which is why the reported barrier looked
large despite one "peak" being physically meaningless; the function's
existing `MIN_BARRIER_FOR_BIMODAL` check (designed to catch a shallow
dip on one broad hump) does not catch a *deep* dip where one side is
simply noise, because it never checks that both candidate peaks carry
a non-negligible fraction of the total probability mass.

Re-checked with that additional mass filter (each candidate peak must
exceed 1% of the distribution's mode) directly on this phase's data:
**all 5 realizations reduce to a single broad hump** -- 1 of 5 had no
two-peak structure at all under the raw check, and the other 4's
"second peak" is confirmed tail noise by the mass filter. This matches
every other P_beta_c(S) examined in this project (orders model and
regular lattice gas, both single broad humps) -- **no confirmed
first-order signature at any n=30 realization, random or regular
background.** `muca.py`'s shared `_peak_diagnostics` was not modified
in place (it is already relied on for Phase 3b's committed results);
this mass-filter refinement was applied only in this phase's own
analysis script and should be considered for upstreaming if this
pattern recurs.

## Answer to the motivating question

**At n=30: the transition's location does depend on the background
being a regular grid** -- beta_c shifts by about 13%, a gap far wider
than the (quite tight) spread across 5 independent random-background
realizations. **The qualitative phase structure does not depend on
it**: both backgrounds show a manifold-like hot phase and a more-
ordered cold phase, the hot phase is close to background-independent
in its detailed observables, and neither background shows a resolved
first-order double peak. This should be read as a genuine, if modest,
finding -- not papered over as "same within error" when it measurably
is not, and not overstated as "a different phase transition" when the
qualitative picture (and the hot phase quantitatively) carries over
essentially unchanged.

## Known limitations

1. **Single chain-seed for the regular-lattice comparison point.**
   11.118 (Phase 4) has no cross-seed spread of its own; see the
   caveat in the beta_c section above.
2. **Cold-phase MM dimension is noisy** (std 1.041 across realizations)
   -- a finite-size/strongly-ordered-regime limitation of the
   sub-interval-sampling diagnostic itself, not of the background
   comparison.
3. **n=50 (stage 2) not yet run.** The driver script
   (`phase5_run_n.py`) takes `N=30` as a single top-level constant;
   changing it to 50 is the only edit needed, but was not done in this
   stage per the explicit instruction to treat n=50 as a separate,
   later stage.
4. **Single eps (0.1), matching C&S's own d=2 choice** -- no
   sensitivity check against other eps values attempted.
5. **The `_peak_diagnostics` mass-filter gap** documented above is a
   real latent issue in shared, already-relied-on code
   (`causet_lab/mcmc/muca.py`); it did not happen to trigger in Phase
   3b's narrower-dynamic-range histograms, but could recur in any
   future analysis with a similarly long exponential tail. Flagged
   here rather than fixed in place.

## Pass criteria, assessed against what was actually run

1. **Random background reduces to the regular lattice at the same
   positions**: met (bit-for-bit equivalence test).
2. **Incremental == full recomputation over 1e5 moves**: met.
3. **beta=0 control matches a 2D sprinkle at least as well as the
   regular lattice**: met (Controls section).
4. **5 independent background realizations, spread reported as part
   of the uncertainty**: met (beta_c std=0.058; phase-observable
   stds in the table above).
5. **Plain comparison to the regular lattice -- beta_c, phase
   observables, P_beta_c(S) shape**: met, with the finding reported
   as a real (not "same within error") but modest and qualitatively-
   non-disruptive shift.

**Overall: this stage's goal -- build the comparison and answer the
question plainly -- is met at n=30.** n=50 is the natural next step,
unchanged in method, to see whether the ~13% beta_c gap narrows,
widens, or holds as n grows.
