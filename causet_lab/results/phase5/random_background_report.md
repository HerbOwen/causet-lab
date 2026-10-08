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
Stage 2 (n=50) is a one-line constant change in the driver script, not
run yet.

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
essentially unchanged. **This is now backed by a matched-pipeline,
5-chain-seed regular-lattice comparison (z=51.8), not a single-seed
artifact**, and a mechanism check that rules out the simplest
explanation (a deeper cold-phase floor) without identifying what the
real mechanism is.

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
3. **n=50 (stage 2) not yet run.** The driver script
   (`phase5_run_n.py`) takes `N=30` as a single top-level constant;
   changing it to 50 is the only edit needed, but was not done in this
   stage per the explicit instruction to treat n=50 as a separate,
   later stage.
4. **Single eps (0.1), matching C&S's own d=2 choice** -- no
   sensitivity check against other eps values attempted.
5. **The regular lattice's unexpectedly poor WL mixing under this
   pipeline** (4x the random background's wall time) is reported as an
   observation with a plausible but unverified explanation (exact
   periodic structure creating slow modes that disorder breaks up);
   not independently investigated further.

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

**Overall: this stage's goal -- build the comparison, answer the
question plainly, and stress-test the headline number before trusting
it -- is met at n=30.** n=50 is the natural next step, unchanged in
method, to see whether the ~13% beta_c gap narrows, widens, or holds
as n grows; the regular lattice's mixing-cost anomaly found here is
also worth watching at n=50, since Phase 3b separately found mixing
cost itself already rises steeply with size.
