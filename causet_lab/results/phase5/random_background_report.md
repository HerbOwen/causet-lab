# Phase 5: random vs. regular background, d=2 lattice gas (n=30 and n=50)

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

**Stage 1 (this report): n=30, 5 independent background realizations,
followed up with four checks requested before trusting the headline
number** (a matched-pipeline regular-lattice rerun, a mechanism check,
an upstreamed fix for a second peak-diagnostic bug, and a cross-check
that the fix doesn't change Phase 3b/4's committed conclusions).

**Result, after the follow-up: the transition does NOT survive "same
within error" at n=30, and this now rests on a proper apples-to-apples
comparison, not a single-seed artifact on either side.** Re-running the
*regular* lattice through the exact same `random_bg` pipeline (not
`lattice_gas.py`'s own code path) at 5 independent chain seeds gives
beta_c = **11.095 +/- 0.005 (SEM)**; the random background's 5
realizations give **9.723 +/- 0.026 (SEM)**. Gap = 1.372, combined
SEM = 0.0265, **z = 51.8** -- overwhelmingly significant, not
explainable by seed noise on either side. The qualitative phase
structure (manifold-like hot phase, more-ordered cold phase) is
preserved in both, and the hot phase in particular is close to
background-independent; the cold phase shows a real quantitative
difference (lower ordering fraction, lower height on the random
background). **The mechanism is not what was hypothesized**: the
lowest action reached and the hot-phase mean action are essentially
identical between the two backgrounds (Mechanism check, below) --
whatever produces the beta_c shift is not "the random background can
get more deeply ordered," and is left as an open question. No
confirmed first-order double-peak signature in P_beta_c(S) at any
realization, once a spurious tail-noise "double peak" (found,
diagnosed, and now fixed upstream in `muca.py`) is correctly excluded.

**Stage 2 (n=50, below): the gap survives but shrinks.** beta_c =
**5.959 +/- 0.033 (SEM)** (random background) vs. **6.419 +/- 0.005
(SEM)** (regular lattice), gap = 0.460, z = 13.7 -- still
overwhelmingly significant, but the *relative* shift is **7.2%**, down
from n=30's **12.4%**. The cold-phase ordering-fraction gap that was
clearly present at n=30 (0.583 vs. 0.646) has essentially closed by
n=50 (0.661 vs. 0.660). Getting a trustworthy n=50 number required
fixing two distinct sampler bugs found only by direct inspection of
raw WL state -- an artificially flat histogram-widening step, and a
genuine *kinetically* hard-to-reach bin that no amount of reweighting
could fix -- documented in full in the Stage 2 section below, since
both are general lessons for this sampler, not n=50-specific trivia.

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

### Matched-pipeline regular-lattice rerun (requested before trusting the gap)

The original comparison used Phase 4's single-chain-seed regular-lattice
value (11.118, via `lattice_gas.py`'s own code path). Before treating
the gap as a finding, the regular lattice was re-run through the
**exact same** `random_bg` pipeline (`site_t`/`site_x` set to the
regular lattice's own integer positions, otherwise identical code --
same WL/MUCA settings, chunk size, round-trip-window rule (default,
`ever_visited`-based, for both), and `reweight_P_beta_corrected`-based
beta_c location with the same 0.01x-30x scan bracket), at 5 independent
chain seeds (0-4; no background seed applies, since the lattice is
deterministic):

| Chain seed | WL (s) | MUCA (s) | beta_c |
|---|---|---|---|
| 0 | 1345.0 | 511.1 | 11.089 |
| 1 | 5704.3 | 951.3 | 11.112 |
| 2 | 5434.9 | 747.4 | 11.100 |
| 3 | 1940.6 | 1057.9 | 11.089 |
| 4 | 2052.2 | 700.6 | 11.084 |

**beta_c = 11.095 +/- 0.005 (SEM), std = 0.011 across 5 chain seeds.**
This matches Phase 4's original single-seed value (11.118) to within
about 2 std of this ensemble's own spread -- reassuring that the
original run was not a fluke, though not an exact bit-for-bit repeat
(expected: different RNG stream, and a different, better-informed
pilot-range anchor this time, `beta_hint=11.118` vs. the original's
untrusted 1/n heuristic).

**Recomputed gap, properly apples-to-apples**: regular lattice 11.095
+/- 0.005 vs. random background 9.723 +/- 0.026. Gap = 1.372, combined
SEM = sqrt(0.005^2 + 0.026^2) = 0.0265, **z = 51.8**. The shift is
real and large relative to either ensemble's own noise, not an
artifact of comparing one lucky/unlucky single seed against a
5-realization mean.

**An unexpected, additional finding from this rerun**: the regular
lattice mixed *markedly worse* than the random backgrounds under this
identical pipeline. Total wall time for all 5 regular-lattice chain
seeds: 6659.9s (~111 min, 4 cores), nearly **4x** the random
background's 5-realization total (1623.9s, ~27 min) -- chain seeds 1
and 2 each spent 5400-5700s in Wang-Landau alone (one needed a bin-range
widening to 180 bins, with `edge_hits_total=102`, vs. 0 for every
random-background realization). This suggests the regular lattice's
exact periodic structure may create slow, resonant-like modes in the
WL random walk that quenched disorder actually breaks up and smooths
over -- a plausible but **not independently verified** mechanism,
noted here as an observation, not a settled explanation.

### Mechanism check: is the shift explained by a deeper cold-phase floor?

Natural hypothesis: if the random background's cold phase can reach a
*lower* (more negative) action than the regular lattice's, while the
hot phase is unchanged, that alone would shift the specific-heat
maximum to lower beta. Checked directly -- lowest action reached (the
lowest `ever_visited` WL bin) and the hot-phase mean action
(`reweight_mean_var_S` at beta=0), from the actual saved WL/MUCA
checkpoints of both the Phase 4 regular-lattice run and all 5 random
backgrounds:

| | Lowest action reached | Hot-phase mean action (beta=0) |
|---|---|---|
| Regular lattice (Phase 4 checkpoint) | -2.4705 | 4.3637 |
| Regular lattice (5 chain-seed rerun, mean +/- std) | -2.4716 +/- 0.0013 | 4.3632 +/- 0.0004 |
| Random background (5 realizations, mean +/- std) | -2.4723 +/- 0.0008 | 4.3597 +/- 0.0081 |

**The hypothesis does not hold.** Both quantities are essentially
identical between the two backgrounds -- the differences (0.001-0.002
in the floor, 0.004-0.01 in the hot-phase mean) are well within each
ensemble's own cross-seed noise, not a systematic gap anywhere near
large enough to explain a beta_c shift of 1.37. A deterministic,
non-MCMC "best packing" construction (apex = earliest site; greedily
take the n-1 sites in its forward lightcone with smallest t -- a
compact, densely-interrelated cluster by construction, not claimed
globally optimal) gives a looser bound in both cases (regular: 3.62;
random, mean 3.85 +/- 0.41) -- confirming WL's stochastic search finds
much better (lower) minima than this simple greedy heuristic, as
expected, but not distinguishing the two backgrounds either.

**Conclusion: the beta_c shift is not explained by "the random
background can get more deeply ordered."** Whatever produces it must
be a difference in the *shape* of the density of states across the
whole explored range (how the number of configurations at each action
value varies), not a difference in the extremes. This is reported as
an open question -- not investigated further in this stage.

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

Because the two models' own located beta_c differ (9.723 vs. 11.118 --
the single-seed value the cold-phase sampling was actually run against;
the matched-pipeline 5-seed mean, 11.095, is consistent with it and
would not change this comparison materially), "4*beta_c" picks a
different absolute beta for each (~38.9 vs. 44.47)
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
background.**

**Upstreamed into `muca.py`**: `_peak_diagnostics` now takes a
`min_mass_fraction` parameter (default `MIN_PEAK_MASS_FRACTION = 0.01`,
a new module constant) and rejects any candidate peak pair where
either side falls below that fraction of the distribution's mode,
*before* computing the barrier -- catching exactly this failure mode
in the shared, already-relied-on function rather than leaving the fix
local to this phase's analysis script. Three regression tests added
(`tests/test_mcmc.py`): a synthetic deep-tail bump (must be rejected),
a genuine two-comparable-peaks case with a real valley (must still be
detected), and the mass-fraction boundary itself. `locate_beta_c_equal_height`
inherits the fix automatically (it calls `_peak_diagnostics` with the
new default).

**Checked against Phase 3b's and Phase 4's committed conclusions, using
their actual saved WL/MUCA checkpoints (not regenerated from scratch)**:

| Case | Pre-fix (no mass filter) | Post-fix |
|---|---|---|
| Orders N=30, eps=0.5 (Phase 3b) | None (single hump) | None (unchanged) |
| Orders N=30, eps=0.1 (Phase 4 cross-check) | None (single hump) | None (unchanged) |
| Lattice gas n=30, eps=0.1 (Phase 4) | **"double peak," barrier=22.3 (spurious)** | None (single hump) |

Phase 3b's and the orders-model cross-check's own committed "no
double peak" conclusions are confirmed unchanged by direct
recomputation. The lattice-gas case is new information: **had this
diagnostic been run during Phase 4, the pre-fix code would have
reported a false double peak there too**, via the identical
deep-tail-noise mechanism found in this phase -- Phase 4's committed
report never ran this check, so there was no existing claim to
correct, but this confirms the bug was not specific to the random
background and the fix is the right general one. Phase 3b's N=40,
eps=0.5 result could not be independently re-executed (its per-worker
production histogram was not preserved in reconstructable form, as
already noted in that report); its "None" conclusion is unconditionally
preserved by this fix on logical grounds alone, since the new check
only adds a rejection condition and never removes one -- any input that
previously returned `None` still does.

## Answer to the motivating question (Stage 1: n=30)

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
essentially unchanged. **This is now backed by a matched-pipeline,
5-chain-seed regular-lattice comparison (z=51.8), not a single-seed
artifact**, and a mechanism check that rules out the simplest
explanation (a deeper cold-phase floor) without identifying what the
real mechanism is.

## Stage 2: n=50

Same construction, same eps=0.1, same 5-realization / 5-chain-seed
design as Stage 1, scaled to n=50 (m=4n^2=10000, w=50, h=200). Getting
a trustworthy result here required finding and fixing two distinct
sampler bugs, neither of which was visible at n=30 -- both are
reported in full since they are general lessons about this WL
pipeline, not artifacts specific to n=50.

### Two sampler bugs found during this stage

**Bug 1: `_widen`'s flat-fill histogram extension.** The WL range-widening
step (triggered when edge-hit pressure crosses a threshold) filled an
entire new 60-bin block with one repeated value (the old edge bin's
`ln_g`), not a gradient. The bin immediately adjacent to the old edge
matched its neighbor fine, but bins further into that artificially
flat block could drift apart from each other under ordinary
differential visitation, and once the walker reached the *far* edge of
that drift, a large, unphysical `ln_g` cliff could open there. Found
directly: three of the first ten n=50 WL runs produced adjacent-bin
`ln_g` steps of 101, 46967 and 66767 (vs. 6-11 for every healthy run),
each exactly at a widened block's far edge, each trapping its walker
for 800+ seconds and leaving its MUCA production stuck at **zero round
trips after the full 300,000,000-move cap**. **Fixed**: `_widen` now
extrapolates the locally-observed `ln_g` slope linearly, clamped two
ways -- direction (new bins can never exceed the edge bin's own
`ln_g`, since density of states only decreases further into a tail)
and magnitude (the per-bin drop used is the window-averaged slope,
which is mathematically bounded by the single largest step actually
observed nearby, so one noisy window can't compound into a steeper
plunge than anything locally observed). Three regression tests added
(`tests/test_mcmc.py`): the slope-continuation behavior itself, the
direction clamp on deliberately-wrong-signed synthetic data, and the
exact extrapolated values pinned down by hand for a clean case.

**Bug 2: a genuine kinetic bottleneck, which no seeding fix can repair.**
After fixing Bug 1 and restarting the three failed seeds from scratch,
one of them (`randombg` seed 1) produced the *same* class of failure
again -- a -58997 `ln_g` step, at a bin reached only well into WL,
*not* created by any widening event this time. Direct inspection (a
short burst of the real WL kernel resumed from the exact frozen state,
with the proposed-move destinations tabulated) found the walker had
reached a bin with normal nearby entropy but almost no outgoing moves
accepted with non-negligible probability: WL's histogram-flattening
heuristic reads persistent under-visitation as "needs more weight,"
which is the right response when the cause is low entropy, but the
wrong one when the cause is *kinetic* -- a bin that few single-site
relocations can actually reach from the bulk. No amount of smarter
`ln_g` seeding fixes a reachability problem, since the walker has
to get there via an accepted move regardless of the target bin's
weight. **Fixed, architecturally**: `estimate_initial_bin_range_randombg`
now returns a *fixed* window (hot end = the 99.99th percentile of a
4000-sweep beta=0 pilot; cold end = the 0.01st percentile of a
4000-sweep pilot annealed to 3.5x the beta_c anchor), and WL no longer
widens past it -- edge-hit pressure is simply rejected, exactly as any
other out-of-range move already was. This keeps the walker away from
the kinetically pathological region entirely, rather than trying (and,
as shown, sometimes failing) to make that region tractable after the
fact. `verify_window_coverage_randombg` checks the result is not
*too* tight: for every production run, the reweighted P_beta(S) mass
in the first/last 3 bins is checked at beta in {0, 0.5, 1, 1.5, 2} x
beta_c and confirmed below 1e-3 (actual values: at most 2.05e-05,
typically far smaller) -- see the per-run numbers in checkpoints,
summarized below. As a secondary guard, `check_ln_g_anomalies` (any
adjacent-ever-visited-bin step over 100, vs. the 6-11 healthy baseline)
is now also required to be clear, in addition to the existing
diversity gate, before the WL stall shortcut is allowed to fire at
all -- a walk can range across most of its reachable bins and still
carry one badly-skewed pair, which bin-count diversity alone does not
catch.

### Fairness check: does the methodology change affect already-clean seeds?

Because fixing the two bugs changed the sampler itself, two previously
clean seeds (`randombg` seed 0, `reglat` chain seed 0) were re-run
under the new fixed-window method as a direct check, before mixing
old- and new-method results into one dataset:

| Seed | Old method beta_c | New method beta_c | Difference |
|---|---|---|---|
| randombg seed 0 | 5.892 | 5.894 | 0.002 |
| reglat chain seed 0 | 6.420 | 6.412 | 0.008 |

Both agree to well within the ~0.01-0.07 seed-to-seed spread already
present in each ensemble. **The other three originally-clean seeds per
model (randombg 2,3,4; reglat 1,4) are therefore kept from the original
run, not re-run** -- they showed no `ln_g` anomalies and reached 10
genuine round trips each. The three seeds that failed under the old
method (randombg 1; reglat chain seeds 2,3) were restarted **from
scratch** (not resumed from their corrupted checkpoints) under the new
method; all three succeeded cleanly (no anomalies, 10 round trips,
window coverage confirmed). The final n=50 dataset mixes methods by
seed, with the mix stated plainly here rather than hidden, and backed
by the two-seed consistency check above. The new method was also
substantially faster where it mattered: the three originally-failed
seeds took 1426-1784s total under the new method, vs. the old method's
healthy seeds taking 2987-6924s (randombg) or 3166-3475s (reglat) --
the fixed window has less histogram to flatten and no widening
overhead.

### beta_c at n=50

| Model | Seed | Source | beta_c |
|---|---|---|---|
| Random background | 0 | new method | 5.8937 |
| Random background | 1 | new method (restarted) | 6.0750 |
| Random background | 2 | old method (kept) | 5.9237 |
| Random background | 3 | old method (kept) | 5.9123 |
| Random background | 4 | old method (kept) | 5.9917 |
| Regular lattice | 0 | new method | 6.4120 |
| Regular lattice | 1 | old method (kept) | 6.4247 |
| Regular lattice | 2 | new method (restarted) | 6.4184 |
| Regular lattice | 3 | new method (restarted) | 6.4075 |
| Regular lattice | 4 | old method (kept) | 6.4328 |

**Random background: beta_c = 5.959 +/- 0.033 (SEM), std = 0.075.**
**Regular lattice: beta_c = 6.419 +/- 0.005 (SEM), std = 0.010.**
Gap = 0.460, combined SEM = sqrt(0.033^2 + 0.005^2) = 0.0336,
**z = 13.67**. The gap is overwhelmingly significant again at n=50,
though less extreme than n=30's z=51.8. Decomposing `ln(z)`'s change
into its two factors (gap and combined SEM) shows the **absolute gap
shrinking is the main driver**: the gap itself fell about 3x in
absolute terms (1.372 -> 0.460), which alone would cut z from 51.8 to
~17.4 (1.372/0.0265 -> 0.460/0.0265); the random background's larger
cross-seed spread at n=50 (SEM 0.033 vs. 0.026) is a secondary
contributor, accounting for roughly the remaining difference between
that 17.4 and the actual 13.67. An earlier version of this report
stated the reverse (spread growth as the main driver, gap shrinkage as
incidental) -- that was a mechanical error, caught on request from the
PI, not a re-measurement; the underlying numbers (beta_c, SEMs, gap,
z) are unchanged.

**Relative shift: 7.16% at n=50, vs. 12.37% at n=30.** The regular
lattice's own beta_c dropped from 11.095 (n=30) to 6.419 (n=50) --
consistent with the expected 1/n-ish falloff for this model class --
and the random background's gap *relative to it* shrank by nearly
half. Multiplying the relative shift by n gives a near-constant: 12.37%
x 30 = 371 vs. 7.16% x 50 = 358, a 3.6% difference between the two
system sizes -- i.e. the *relative* gap falls off roughly as 1/n, the
same scaling beta_c itself follows here, over this two-point baseline.
Fitting a two-point power law `relative_gap(n) ~ n^-alpha` through
exactly these two values gives alpha ~ 1.07, consistent with (but of
course not a confirmation of) a pure 1/n falloff (alpha=1) given only
two points.

Because the *absolute* gap is the relative gap times beta_c, and both
factors fall off at roughly 1/n here, the absolute gap should fall off
faster, close to 1/n^2 -- and it does: 1.372 at n=30 to 0.460 at n=50
is a 2.98x drop, close to the (50/30)^2 = 2.78x a pure 1/n^2 law
predicts, well past the 1.67x a 1/n falloff of the absolute gap alone
would give. **The background-dependence of the transition looks like
a finite-size effect that fades as n grows**, not a permanent,
size-independent discrepancy between the two models. This is stated as
what two data points (n=30, n=50) show, not as a confirmed asymptotic
law -- a third size would be needed to fit a genuine 1/n-type falloff
of the relative gap itself. Taking the simple 1/n scaling of the
relative gap at face value and anchoring it at n=50 predicts a
relative gap of about **5.6% at n=64** (7.16% x 50/64) -- the largest n
reachable without a multiword-bitset rewrite (Section 7/8) -- stated
here as a falsifiable number for whoever runs that size next.
Extrapolating the same reading out to n=200 (C&S's own system size,
far beyond anything run in this project) gives about 1.8%.

### P_beta_c(S) shape

Checked with the same mass-filtered `_peak_diagnostics` used
throughout this project, on all 10 final seeds (reweighted at each
seed's own located beta_c): **every one of the 10 is a single broad
hump -- no double-peak structure, spurious or otherwise, at any
seed.** This matches every other P_beta_c(S) examined in this project.

### Hot- and cold-phase observables

Same methodology as Stage 1 (5 fresh beta=0 fillings for hot; 5
chains annealed to 4x each run's own located beta_c for cold), mean
+/- std across the 5 final seeds per model:

| Observable | Random bg, hot | Reg. lattice, hot | Random bg, cold | Reg. lattice, cold |
|---|---|---|---|---|
| MM dimension | 2.014 +/- 0.048 | 1.885 +/- 0.000 | 3.603 +/- 0.103 | 3.603 +/- 0.072 |
| Ordering fraction | 0.881 +/- 0.003 | 0.885 +/- 0.000 | 0.661 +/- 0.003 | 0.660 +/- 0.003 |
| Height | 9.24 +/- 0.36 | 8.40 +/- 0.00 | 4.32 +/- 0.18 | 4.48 +/- 0.23 |
| Occupied rows ("layers", of 200) | 44.52 +/- 0.44 | 44.60 +/- 0.00 | 22.76 +/- 1.24 | 20.36 +/- 0.78 |

(The regular lattice's hot-phase values show exactly zero cross-seed
std because the hot-phase fillings use fixed filling-seeds on the
*same* fixed lattice regardless of chain seed -- a real zero, not a
rounding artifact; only the cold phase depends on each chain's own
located beta_c and therefore genuinely varies.)

**The cold-phase ordering-fraction gap that was clearly present at
n=30 (0.583 vs. 0.646, a difference of 0.063) has essentially closed
by n=50 (0.661 vs. 0.660, a difference of 0.001).** The MM dimension
in the cold phase converges too (2.73 vs. 2.87 at n=30, both noisy,
to 3.60 vs. 3.60 at n=50, identical to 3 figures). Height is close at
both n. The layer count still shows a modest residual difference
(22.76 vs. 20.36, a gap of 2.4) -- smaller proportionally than at
n=30. The hot phase remains close to background-independent at n=50,
consistent with Stage 1, and consistent with both backgrounds' own
beta=0 fillings resembling a 2D sprinkle equally well regardless of n.

### Mixing: moves per WL stage and per round trip

Preserved only for the seeds kept from the original run (the restarted
and fairness-check seeds' per-stage/per-round-trip logs were not
retained in the saved results, only aggregate timing and move counts)
-- reported with that caveat, not silently extrapolated from too few
seeds:

| Model | Seed | Mean moves/WL stage | Mean moves/round trip |
|---|---|---|---|
| Random background | 2 | 6,259,474 | 25,887,000 |
| Random background | 3 | 5,675,263 | 13,190,000 |
| Random background | 4 | 7,737,000 | 3,662,000 |
| Regular lattice | 1 | 4,604,375 | 8,712,000 |
| Regular lattice | 4 | 7,446,316 | 4,776,000 |

Moves-per-stage are comparable in order of magnitude between the two
backgrounds (~4.6-7.7M). Moves-per-round-trip show large seed-to-seed
variance within each background (3.7M-25.9M for random, 4.8M-8.7M for
regular) -- too few seeds with preserved logs to draw a reliable
per-move mixing-speed comparison between backgrounds at n=50; this
is reported as inconclusive with this sample, not rounded to a
conclusion either way.

### Answer to the motivating question (Stage 2: n=50)

**The transition's location still depends on the background being a
regular grid at n=50 (z=13.67, not explainable by seed noise), but the
relative size of the effect has shrunk by close to half (12.4% ->
7.2%) compared to n=30, and the cold-phase observable gap that
distinguished the two backgrounds most clearly at n=30 has nearly
closed.** The qualitative picture is unchanged from Stage 1: a
manifold-like hot phase that is close to background-independent, a
more-ordered cold phase showing real but shrinking quantitative
differences, and no confirmed first-order double-peak signature in
P_beta_c(S) for either background at either n. Read plainly: this
looks like a finite-size effect fading with system size, not a
permanent background-dependence of the physics -- though two points
(n=30, n=50) is not enough to fit the falloff itself with any
confidence.

## Known limitations

1. **Mechanism for the beta_c shift is unidentified.** Ruled out: a
   deeper cold-phase action floor, a shifted hot-phase mean action
   (Mechanism check section). Not investigated: the detailed shape of
   the density of states g(S) across the explored range, which is
   where the effect must actually live.
2. **Cold-phase MM dimension is noisy** (std 1.041 across realizations)
   -- a finite-size/strongly-ordered-regime limitation of the
   sub-interval-sampling diagnostic itself, not of the background
   comparison.
3. **n=50's final dataset mixes two sampler methods by seed** (5 of 10
   seeds re-run under the new fixed-window method, 5 kept from the
   original adaptive-widen method) -- justified by a direct two-seed
   consistency check (differences of 0.002 and 0.008, well within each
   ensemble's own seed-to-seed spread) and stated plainly in the Stage
   2 section, but a fully single-method n=50 dataset was not run.
4. **Single eps (0.1), matching C&S's own d=2 choice** -- no
   sensitivity check against other eps values attempted, at either n.
5. **The regular lattice's unexpectedly poor WL mixing seen at n=30**
   (4x the random background's wall time under that stage's pipeline)
   did not clearly repeat at n=50 once the sampler bugs were fixed and
   the new method's seeds ran much faster than the old method's; the
   wall-clock comparison at n=50 is confounded by the method mix (item
   3) and not used to draw a mixing-speed conclusion there.
6. **n=50's moves-per-stage/moves-per-round-trip mixing comparison is
   based on only 3 (random) and 2 (regular) seeds** -- the restarted
   and fairness-check seeds' logs were not retained -- too few to
   support a confident per-move mixing-speed claim between backgrounds.
7. **The n=50 fixed WL window was set from quantiles of pilot samples
   (4000 sweeps each) and verified post-hoc (verify_window_coverage_randombg:
   edge mass < 1e-3, typically < 1e-4, for beta in [0, 2*beta_c] on
   every production run)** -- a real check, not a blind assumption,
   but it was tuned for this n and eps and would need re-verifying,
   not blindly reused, at a different size or coupling.
8. **Only two system sizes (n=30, n=50) checked for the beta_c gap.**
   The observation that the relative gap shrinks (12.4% -> 7.2%) is
   consistent with a finite-size effect fading with n, but two points
   cannot distinguish that from, say, a slower-than-1/n falloff that
   plateaus at a nonzero value -- a third size (e.g. n=80, which needs
   the multiword-bitset extension noted in the main draft's Outlook)
   would be needed to fit the falloff with any confidence.

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
   non-disruptive shift, now confirmed against a matched-pipeline,
   5-chain-seed regular-lattice ensemble rather than a single value
   (z=51.8).
6. **Mechanism check for the shift**: run as requested; the simplest
   hypothesis (deeper cold floor) is ruled out, not confirmed -- the
   real mechanism is left open.
7. **Peak-diagnostic mass filter upstreamed, with regression tests,
   and checked against Phase 3b/4's committed conclusions**: met;
   no committed conclusion changed, and one latent false-positive
   (that would have affected Phase 4's lattice gas, had it been
   checked there) is now fixed for all callers.
8. **Stage 2 (n=50) run, with the beta_c gap, phase observables, and
   P_beta_c(S) shape all reported against n=30**: met; the gap
   survives (z=13.67) but shrinks in relative size (12.4% -> 7.2%),
   and the cold-phase observable gap most visible at n=30 has nearly
   closed by n=50 -- reported plainly as a finite-size-looking effect,
   not rounded to "resolved" or "unchanged."
9. **Two sampler bugs found at n=50 fixed upstream, with regression
   tests, and validated by a fairness check against unaffected seeds**:
   met -- `_widen`'s flat-fill cliff (gradient extrapolation, clamped
   both ways) and a genuine kinetic-bottleneck trap (fixed window,
   no adaptive widening, window-coverage verification), plus a
   stall-shortcut gate against both failure signatures. Two
   previously-clean seeds re-run under the new method matched their
   original beta_c to within 0.002 and 0.008.

**Overall: this stage's goal -- build the comparison, answer the
question plainly, and stress-test the headline number before trusting
it -- is now met at both n=30 and n=50.** The background-dependence of
beta_c looks like a finite-size effect that fades with system size,
not a permanent discrepancy; confirming that with any confidence would
need a third size (n=80, blocked on the bitset engine's N<=64 limit
without the multiword extension noted in the main draft's Outlook).
