---
title: "An open-source multicanonical sampler for 2D causal set quantum gravity: validation and a random-background lattice-gas comparison"
status: DRAFT -- working title, not for distribution
---

> **How to read the citations in this draft.** Every numeric claim below is
> tagged `[source @ commit]`, naming the exact report file and git commit
> hash in this repository the number was taken from. `[uncommitted: path]`
> marks a number that exists (e.g. in a test's printed output or a
> still-running job's log) but has not yet been committed to git at
> draft time. `[TBD: ...]` marks a result this draft is waiting on.
> Paper citations use only works this project actually fetched and
> checked against its own code, listed in full at first use and
> collected at the end; no arXiv ID is invented for a source that
> doesn't have one on record here.

## Abstract

We describe an open-source Wang-Landau/multicanonical (WL/MUCA) sampler
for two-dimensional causal set quantum gravity, built to replace a
beta-by-beta parallel-tempering (PT) scan whose located critical
couplings turned out to be mostly right-censored and, in one tested
case, not reproducible under a blind re-centering of the search grid.
The sampler is validated against the published finite-size-scaling
formula for the order-based smeared Benincasa-Dowker action at two
system sizes and two couplings, and against an independent causal-set
model -- Cunningham & Surya's dimensionally-restricted lattice gas --
including that model's own qualitative manifold-like/layered phase
structure. We then build a random-background variant of the lattice
gas (a quenched Poisson sprinkling of background sites in place of
C&S's regular lattice) to address an open question C&S raise but do
not pursue: whether their regular background biases the result. At
n=30, five independent background realizations locate a critical
coupling about 13% below a matched-pipeline, 5-chain-seed regular-
lattice ensemble (z=51.8 -- not a single-seed artifact on either
side), while the qualitative phase structure and the hot phase's
detailed statistics are close to background-independent. A direct
mechanism check rules out the simplest explanation (the random
background reaching a deeper cold-phase action floor); the real
mechanism is left as an open question.
We report several methodological pitfalls found and fixed along the
way -- a beta_c definition mismatch, noise mistaken for a first-order
double peak (twice, in two different ways), and a units bug in the
random-background model's causal relation -- as cautionary detail for
others building similar samplers.
Limitations include small system sizes (N, n <= 60), a first-order
barrier that is not cleanly resolved at any size tested, and (so far)
single- or few-seed statistics for most results.

## 1. Introduction

A causal set is a locally finite partially ordered set, proposed as a
discrete substrate for Lorentzian quantum gravity: the partial order
encodes the causal structure, and continuum geometry is meant to
emerge statistically from a sum (or path integral) over causal sets
weighted by a discrete analogue of the Einstein-Hilbert action. The
leading discrete action for this purpose is the Benincasa-Dowker (BD)
action, built from counts of small order-intervals ("abundances") in
the causal set, which converges to the continuum scalar-curvature
action when evaluated on a Poisson sprinkling into a Lorentzian
manifold [Benincasa & Dowker 2010, arXiv:1001.2725]. Because exact
abundance counts are UV-sensitive (dominated by nearest-neighbor
relations), the action used in practice is *smeared* over a small
length scale eps, which also fixes the overall normalization used
throughout this project.

Monte Carlo studies that weight the space of N-element causal sets by
exp(-beta * S_BD(eps)) -- a Euclideanized, not oscillatory, weight --
have found a phase transition as beta is varied: a continuum-like
("manifold-like") phase at small beta whose statistics resemble a
genuine Lorentzian sprinkling, and a non-continuum phase at large beta
dominated by highly-ordered configurations (Kleitman-Rothschild-like
orders, or -- in lattice-gas-type models -- literal crystalline
layering) [Surya 2012, arXiv:1110.6244]. Locating this transition
reliably, and determining its order, is numerically hard: the two
phases are separated by a free-energy barrier that single-site
Metropolis moves and even naive replica exchange struggle to cross,
so a transition can look "located" while actually being an artifact of
wherever the sampler's convergence ran out. Section 6 below documents
several concrete ways this went wrong during this project before it
was caught.

Cunningham & Surya introduce a cheaper alternative to sampling generic
N-element orders directly: a *lattice gas*, where N causal-set elements
occupy N of the sites of a fixed regular lattice embedded in a
d-dimensional cylinder spacetime, and Monte Carlo moves relocate one
element to an unoccupied site [Cunningham & Surya 2019, arXiv:1908.11647].
This "dimensionally restricted" construction reproduces the generic
model's qualitative manifold-like/layered phase transition at much
lower computational cost, but the background lattice is regular --
C&S note explicitly that checking whether this regularity biases the
causal structure (as opposed to using "a more realistic model of
discreteness", i.e. a randomly sprinkled background) is an open
question they do not pursue. Section 5 below is this project's own
attempt at that comparison.

## 2. Models

**2D orders with the smeared BD action.** The state space is all
labelled partial orders on N elements (equivalently, N x N transitive
binary matrices); two elements' relation is updated by a single
coordinate-swap Monte Carlo move. The smeared 2D action is
`S(C,eps) = 4*eps*(N - 2*eps*sum_n N_n*f(n,eps))`,
`f(n,eps) = (1-eps)^n*(1 - 2*eps*n/(1-eps) + eps^2*n*(n-1)/(2*(1-eps)^2))`,
matched by direct inspection of the fetched paper against this
project's implementation, including an eps -> 1 analytic-limit unit
test [Glaser, O'Connor & Surya 2018, arXiv:1706.06432; confirmed in
`causet_lab/mcmc/action.py` and `results/phase3/mcmc_2d_report.md @ edb9b99`].

**Dimensionally-restricted lattice gas.** N elements occupy N sites of
an `h*w = m` regular lattice on a cylinder (metric
`ds^2 = -dt^2 + dtheta^2`, t in {0,...,h}, theta periodic with period w
sites), with this project's own d=2 sizing convention `w=n, h=4n,
m=4n^2` matched directly against C&S's own quoted numbers. Moves swap
one occupied site with one unoccupied site. **C&S's causal relation is
not fully specified in the paper** -- they state 45-degree lightcones
and confirm cylinder wraparound exists, but give no integer-coordinate
formula; this project derived `(t_a,k_a) prec (t_b,k_b) iff t_b>t_a and
min(dk, w-dk) <= t_b-t_a` (shortest arc on the circumference) and
flags this explicitly as its own construction, not a quote
[`causet_lab/mcmc/lattice_gas_core.py`, `results/phase4/lattice_gas_2d_report.md
@ 0f85d22`]. The smeared action used is C&S's own Eq. 8/9, which has a
genuinely different normalization from the order model's action above
(prefactor `2*eps` vs `4*eps`, and the interval-count sum starts at
r=1 vs r=0) -- confirmed by direct quotation from both papers, kept as
two separate functions rather than unified.

**Random-background variant (this project's own construction).**
Identical region, filling representation, move set, and action; the
only change is that the m background sites are a fixed (quenched)
uniform-random sprinkling of continuous `(t, x)` positions instead of
sitting on the integer lattice, with `x` given period `w` -- the
*same* units the lattice's own causal relation uses, not radians
(`theta` has period `2*pi`; `x = w*theta/(2*pi)` is the right
rescaling to keep the two models' causal geometries identical. See
Section 6 for how a units mismatch here was initially missed).
**Equivalence check**: placing the background exactly at the regular
lattice's own integer positions reproduces `lattice_gas.lattice_to_matrix`
bit-for-bit, including the lightlike (`<=`) convention, across 3 sizes
and 5 seeds each [`causet_lab/tests/test_random_bg.py::test_random_bg_at_lattice_positions_matches_lattice_exactly`,
uncommitted at draft time -- all 3 parametrizations pass].

## 3. Methods

**Incremental bitset updates.** Each element's relation to every other
is tracked as a `uint64` bitset (`future[x]`/`past[x]`); interval size
for a related pair is `popcount(future[x] & past[y])`. A move changes
only the relations of the element(s) that moved, updated via O(N)
scalar bookkeeping rather than an O(N^2) matrix rebuild -- validated
bit-for-bit against full recomputation over 1e5 random moves for every
model in this project (orders, lattice gas, random background)
[`causet_lab/tests/test_bitset_core.py`, `test_lattice_gas.py`,
`test_random_bg.py`]. This caps all three models at N <= 64 (one
element's relations fit in one machine word); see Section 7.

**Parallel tempering (PT).** The original (Phase 3) approach: replicas
at a ladder of beta values, with swap moves between neighbors and
round-trip tracking to confirm replicas actually cross the transition
rather than merely reporting a locally-plausible swap rate
[`causet_lab/mcmc/tempering.py`].

**Wang-Landau + multicanonical (WL/MUCA).** Replacing PT (see Section
6 for why): Wang-Landau estimates the action density of states g(S) by
an adaptive-weight random walk that flattens the visited-bin histogram
across the whole reachable range (stage-doubling modification factor,
with stall detection distinguishing a genuinely unreachable bin from a
merely slow one); multicanonical production then samples with that
weight frozen and reweights the single resulting run to the canonical
P_beta(S) and <O>(beta) at *any* beta, using the Berg-Neuhaus
histogram-corrected estimator `P(S) ~ H_muca(S) * exp(ln_g(S) -
beta*S)` [Berg & Neuhaus, "Multicanonical algorithms for first order
phase transitions", Phys. Lett. B 267 (1991) 249 -- no arXiv ID on
record for this pre-arXiv-era reference]. beta_c is located as the
action-variance (specific-heat) maximum, matching the definition
Glaser-O'Connor-Surya themselves use, not an equal-peak-height
double-peak criterion (kept as a separate diagnostic) [Glaser,
O'Connor & Surya 2018, Sec. 4.1].

**Windowed round-trip counting.** A round trip is counted between the
two *physically relevant* reachable extremes, which can be narrower
than the full `ever_visited` span when most of P_beta(S)'s mass for
the beta range of interest sits well inside it (Section 6).

**Checkpointing.** Atomic pickle writes of the full walker state
(configuration, splitmix64 RNG state, WL/MUCA weights and histograms),
exercised in practice by intentionally interrupting and resuming runs
[`causet_lab/mcmc/checkpoint.py`].

**Uncertainty.** Two independent estimators are used depending on what
a given run produced: a block-bootstrap over round-trip-sized windows
of a single run's time series (used for the N=30, eps=0.5 orders
result), or the empirical spread across independent parallel workers'
own located beta_c (used for N=40, eps=0.5 and, in Section 5, across
background realizations).

## 4. Validation

**2D orders, eps=0.5, N=30 and N=40: beta_c vs. the published
formula.** The formula is `beta_c(N,eps) = b(eps)/N + c(eps)/N^2`,
`b(eps)=1.66(+/-0.03)/eps^2`, `c(eps) = 4.09(+/-0.50)/eps^3 -
27.77(+/-2.45)/eps^2` [Glaser, O'Connor & Surya 2018]. The leading
term alone is a poor approximation at these sizes -- the `1/N^2`
correction is not a small perturbation here, consistent with the
paper's own statement that its asymptotic regime is only reached for
`N >~ 65`:

| N | leading term `b/N` | full formula | measured beta_c | z |
|---|---|---|---|---|
| 30 | 0.2213 | 0.1343 +/- 0.0158 | 0.1478 +/- 0.0004 (bootstrap) | 0.86 |
| 40 | 0.1660 | 0.1170 +/- 0.0096 | 0.1221 +/- 0.0011 (4-worker spread) | 0.52 |

[`results/phase3b/muca_calibration_report.md @ 7d648fa`; leading/full
formula values recomputed directly from the published coefficients
above for this table]. Both PASS (z < 3), though see Section 7 for why
this should not be read as confirming the paper's full first-order
picture.

**2D orders, eps=0.1, N=30: independent cross-check.** Run fresh (not
part of the original eps=0.5 calibration) as a side check during the
lattice-gas work below: located beta_c = 6.939, full-formula
prediction 6.992 +/- 0.627 (leading term alone: 5.533) -- within about
0.1 formula-standard-deviations, with no bootstrap on the measured
side for this single run [`results/phase4/lattice_gas_2d_report.md @
0f85d22`, Part 5].

**PT vs. MUCA agreement at N=30, eps=0.5.** The original PT scan
located beta_c = 0.14797 +/- 0.00384 at this (N, eps)
[`results/phase3/mcmc_2d_report.md @ edb9b99`] -- **but this specific
value is flagged in that same report as right-censored** (the largest
converged beta in that scan, not a confirmed interior maximum; see
Section 6). The independent WL/MUCA run above located 0.1478 +/- 0.0004
using a method that locates an interior maximum directly rather than
scanning outward from a fixed grid. The two agree to within 0.001
[`results/phase3b/muca_calibration_report.md @ 7d648fa`] -- read as two
different sampling methods landing on the same number, not as two
independent *confirmations* of a physical beta_c, since the first was
never confirmed interior in its own right.

**Lattice gas 2D: controls and qualitative phase comparison.** beta=0
(uniformly random fillings) sampled as sub-intervals (since the
cylinder filling is not itself an Alexandrov interval) give
Myrheim-Meyer dimension 2.292 (n=30) / 2.073 (n=50), both close to the
d=2 target, and interval-abundance L1 distance to a true sprinkle of
0.100 / 0.086 [`results/phase4/lattice_gas_2d_report.md @ 0f85d22`].
At n=30, the located beta_c = 11.118 separates a hot phase (ordering
fraction 0.882, matching C&S's own quoted hot-phase value of 0.88
almost exactly; MM dimension 2.18; abundance-profile distance to a
sprinkle 0.065) from a cold phase at 4*beta_c (ordering fraction
0.646; MM dimension 2.87; abundance-profile distance 0.794) -- every
observable moves in C&S's reported direction, though the cold-phase
ordering fraction does not reach their asymptotic ~0.5 bilayer value
at this much smaller n [same source, Part 4].

**P_beta_c(S), described cautiously.** The pooled action histogram at
the located beta_c is a single broad hump at both N=30 and N=40
(eps=0.5), not a resolved double peak
[`results/phase3b/muca_calibration_report.md @ 7d648fa`, plot
`muca_P_beta_c_S_eps0p50.png`]. This is consistent with either a
genuinely weak/crossover-like transition at these small N (both are
well below the paper's own `N >~ 65` asymptotic regime) or with
insufficient statistics to resolve a real but small barrier --
**the data collected here cannot distinguish the two**, and this
should not be read as a confirmed first-order signature at either N.

## 5. New result: random vs. regular background

Stage 1: n=30, 5 independent background realizations (sprinkled once
per realization, then held fixed -- "quenched" -- for that
realization's entire WL+MUCA run), run in parallel across 4 CPU
cores, total wall time 1623.9s. The background seed and the
Metropolis-chain seed are always independent parameters for a given
realization (never derived from one another), so the only thing
varying across the 5 runs is the background geometry itself. Same
eps=0.1, same m = 4*n^2, same WL/MUCA/checkpointing pipeline as the
Phase 4 regular-lattice run, with the beta_c-location scan bracket
deliberately widened (0.01x to 30x the regular lattice's own located
beta_c, used only as a pilot-range anchor, not an assumption the two
models share a beta_c) since there is no a priori reason quenched
disorder should leave beta_c unchanged. Stage 2 (n=50) is a one-line
change to the driver's size constant, not attempted yet
[`results/phase5/random_background_report.md @ 3ae1ad5`, this section].

**Equivalence check**: placing the background exactly at the regular
lattice's integer positions reproduces `lattice_to_matrix` bit-for-bit
across 3 sizes and 5 seeds each -- confirms the random background's
causal geometry is identical to the regular lattice's at those
positions, not merely "the same region" (see Section 6 for the units
bug this caught).

**Control**: beta=0 random-background fillings (5 seeds, same
sub-interval-sampling methodology as Phase 4) give MM dimension 2.267
(n=30) / 2.061 (n=50), abundance-profile distance to a sprinkle 0.089
/ 0.060, and whole-filling ordering fraction 0.8768 / 0.8766 --
matching the regular lattice's own n=30 beta=0 values (MM dim 2.292,
distance 0.100, ordering fraction ~0.882) at least as well.

**beta_c**: tightly reproducible across realizations (9.709, 9.757,
9.756, 9.766, 9.627; mean 9.723, std 0.058, SEM 0.026). Checked against
a matched-pipeline comparison, not a single-seed one: the regular
lattice re-run through the identical `random_bg` code path (positions
fixed at the integer grid, otherwise the exact same settings) at 5
independent chain seeds gives 11.095 +/- 0.005 (SEM, std 0.011) --
consistent with Phase 4's original single-seed value (11.118) to
within about 2 std. Gap = 1.372, combined SEM = 0.0265, **z = 51.8**:
**significantly below** the regular lattice, not explainable by
single-seed noise on either side. An unexpected side finding from the
matched-pipeline rerun: the regular lattice mixed markedly worse under
Wang-Landau than every random-background realization (total wall time
6659.9s for 5 chain seeds vs. 1623.9s for 5 backgrounds) -- plausibly
the lattice's exact periodic structure supports slow modes that
quenched disorder breaks up, not independently confirmed.

**Mechanism check**: does the random background simply reach a deeper
cold-phase action floor (which would mechanically lower beta_c)? Checked
directly against both backgrounds' actual WL/MUCA data: lowest action
reached (-2.4716 +/- 0.0013 regular vs. -2.4723 +/- 0.0008 random) and
hot-phase mean action (4.3632 +/- 0.0004 vs. 4.3597 +/- 0.0081) are
statistically indistinguishable between the two. **The hypothesis does
not hold** -- the beta_c shift is not explained by the random
background being able to get more deeply ordered; the real mechanism
(presumably a difference in the density-of-states shape across the
explored range, not its extremes) is left open.

**Hot- vs. cold-phase observables** (hot: 5 fresh beta=0 fillings per
background; cold: 5 independent chains per background annealed to
`4 * that background's own beta_c`; values are mean +/- std across the
5 backgrounds):

| Observable | Hot | Cold | Reg. lattice, hot | Reg. lattice, cold |
|---|---|---|---|---|
| MM dimension | 2.163 +/- 0.145 | 2.729 +/- 1.041 | 2.18 | 2.87 |
| Ordering fraction | 0.876 +/- 0.007 | 0.583 +/- 0.022 | 0.882 | 0.646 |
| Height | 7.00 +/- 0.51 | 3.44 +/- 0.36 | 7.40 | 4.20 |
| Occupied rows (/120) | 26.32 +/- 0.41 | 13.16 +/- 0.61 | 26.7 | 12.8 |

The hot phase is close to background-independent (every value within
about 1 std of the regular lattice's). The cold phase differs
measurably: lower ordering fraction and height on the random
background than the regular lattice at a comparably-deep point
relative to each model's own beta_c (not a matched absolute beta, by
construction) -- in the direction of C&S's own asymptotic bilayer
limit (~0.5), slightly closer here than the regular lattice got at
n=30.

**P_beta_c(S) shape**: a naive check using this project's existing
peak-finder flagged a "double peak" with a large barrier (17-21) at
4/5 realizations -- **confirmed spurious** on inspection (one
candidate "peak" in every case was 8-11 orders of magnitude below the
real mode, see Section 6), and **fixed upstream** in
`muca._peak_diagnostics` itself (a new mass-fraction filter, with
regression tests) rather than left as a local workaround. After the
fix, all 5 realizations show a single broad hump, the same picture as
the regular lattice and the orders model. Re-checking Phase 3b's and
Phase 4's committed conclusions against their actual saved checkpoints
confirms the fix changes neither (both still single-hump) -- though it
does retroactively show that Phase 4's lattice-gas P_beta_c(S) would
have triggered the *same* spurious double peak had anyone checked it
at the time; nobody had. **No confirmed first-order signature at any
realization or model examined in this project.**

**Plain answer**: at n=30, the transition's *location* depends on the
background being a regular grid (a real ~13% shift, confirmed against
a matched 5-seed regular-lattice ensemble, not a single-seed artifact,
and not explained by the simplest mechanism checked), well outside the
cross-realization spread); the transition's *qualitative character*
does not (manifold-like/ordered phases persist on both, the hot phase
is quantitatively close to background-independent, neither background
shows a resolved double peak). Reported as found, not rounded toward
either "no effect" or "a different transition."

## 6. Pitfalls and lessons

**Right-censored "peaks" and the blind grid-recentering test.** The
original PT scan's beta_c was, for 7 of 8 tested (N, eps) combinations,
exactly the largest *converged* beta in its scan -- i.e. there was no
converged data confirming the variance actually turns back down past
the located point. A direct stress test (N=40, deliberately
re-centering the search grid at 0.6x and 1.6x the formula's
prediction, both shifts from the original ~0.42x center) found that
at eps=0.21 the located beta_c relocated to ~7.0 from both directions
-- nowhere near the original 0.70 or the published 0.82 -- while
eps=0.5 held up (three grid placements agreed within ~25%)
[`results/phase3/mcmc_2d_report.md @ edb9b99`]. Lesson: a beta_c that
sits at the edge of whatever range was searched is not located; it is
censored, and should be reported as such until an interior maximum is
confirmed, grid placement included.

**The beta_c definition mismatch.** An early version of the first-order
diagnostic used equal-peak-height, not the action-variance maximum
that Glaser-O'Connor-Surya's own formula was fit to -- checked
directly against their Sec. 4.1 and corrected before any formula
comparison was trusted [`causet_lab/mcmc/muca.py @ 7d648fa`].

**Noise mistaken for a double peak.** A naive peak finder accepted any
two local maxima regardless of the dip between them, producing a false
"double peak" at N=30 with a barrier of only 0.004-0.07 across an
800-point beta scan -- a <7% dip on a single broad hump, not a
first-order signature. Fixed by requiring a minimum barrier
(`MIN_BARRIER_FOR_BIMODAL = 0.5`) before accepting a two-peak
structure [same source].

**A reweighting estimator blind to its own weight error.** The
original `reweight_P_beta` assumed `ln_g` already equalled the true
log density of states -- true only once Wang-Landau has fully
converged. Fixed by folding in the actual MUCA production histogram
(`reweight_P_beta_corrected`, the standard Berg-Neuhaus correction),
which stays exact even with an imperfect weight provided the walk
visits the region being compared [same source].

**Full-span vs. windowed round trips.** At N=40, requiring a round
trip to reach the full `ever_visited` span (up to S=+402) undercounted
real excursions, since essentially all of P_beta(S)'s mass for the
beta range being tested sat in a much narrower window (bins 0-85 of
126). Recounting the same run with that window instead of the full
span found 4x more round trips in under half the wall-clock time of a
fresh run -- same physics, same sampling, only the counting criterion
changed [same source].

**The units bug in the first random-background draft.** The initial
implementation sprinkled the spatial coordinate as `theta` in radians
(period `2*pi`) and compared it directly against `dt` in the lattice's
own integer row units -- silently rescaling the lightcone by a factor
of `w/(2*pi)` relative to the regular lattice's own causal relation,
which compares an integer step count (period `w`) against `dt`
directly. Caught before any sampling was run, by an explicit request
to check units and add a same-positions equivalence test against the
regular lattice (Section 2); fixed by sprinkling a coordinate `x` with
period `w` instead. Lesson: "the same region" is not automatically
"the same causal geometry" once a discrete model is replaced by a
continuous one -- an explicit reduces-to-the-known-case test is the
check that would have caught this immediately, and should be written
*before* any production sampling, not after.

**A second, different way to mistake noise for a double peak.** The
existing `MIN_BARRIER_FOR_BIMODAL` check (above) guards against a
*shallow* dip on one broad hump. It does not guard against a *deep*
dip where one side of the "peak pair" is simply numerical noise in an
exponentially suppressed tail: in the random-background analysis
(Section 5), 4 of 5 realizations' `P_beta_c(S)` were flagged as
double-peaked with a large barrier (17-21), but the "second peak" in
every case sat 8-11 orders of magnitude below the real mode --
`_peak_diagnostics`'s `peak_val = 0.5*(p[i1]+p[i2])` average is
dominated by the real peak regardless of how negligible the other
one is, so the barrier calculation never actually tests whether both
candidate peaks carry physically meaningful probability mass. **Fixed
upstream** this time, not left as a local workaround: a mass-fraction
filter (`MIN_PEAK_MASS_FRACTION`, each candidate peak must exceed 1%
of the mode) was added directly to `_peak_diagnostics`
[`causet_lab/mcmc/muca.py`], with three regression tests covering the
deep-tail-bump rejection, a genuine two-peak case, and the boundary
[`causet_lab/tests/test_mcmc.py`]. Re-run against Phase 3b's and
Phase 4's own saved checkpoints: both still report no double peak
(unchanged); Phase 4's lattice-gas P_beta_c(S) would have shown the
*same* spurious double peak (barrier=22.3) had anyone run this check
on it at the time -- nobody had, so there was no committed claim to
correct, but it confirms the bug was general, not specific to
disorder.

## 7. Limitations

- **Small sizes.** N, n <= 60 throughout (orders PT/MUCA up to N=60;
  lattice gas and random background up to n=50); the bitset engine's
  `uint64`-per-element representation caps every model in this project
  at N <= 64 without a redesign (see Outlook).
- **No cleanly resolved first-order barrier at any size tested.**
  Hysteresis disagreed between random- and layered-start chains for
  6/8 (N, eps) PT combinations, the double-peak histogram signature
  appeared in only 1/8, and MUCA's own P_beta_c(S) is a single broad
  hump at both N tested (Section 4). Whether this reflects genuinely
  weak/crossover behavior below the paper's asymptotic regime, or
  insufficient statistics, is not resolved here.
- **Single epsilon for most results.** eps=0.5 is the primary orders-
  model coupling; eps=0.1 and eps=0.21 appear only as side checks or
  (eps=0.21) in the earlier, since-superseded PT study. The lattice-gas
  and random-background models use only eps=0.1, matching C&S's own
  choice.
- **Few seeds.** Most orders-model and lattice-gas results are single-
  seed; Phase 5 (Section 5) is this project's first result with an
  explicit multi-seed spread, and only at one size so far.
- **Euclideanized weights throughout.** Every model here samples
  `exp(-beta*S_BD)` as a real, non-oscillatory Boltzmann weight -- a
  standard device in this literature, not a Lorentzian quantum
  path integral with genuine phases. Nothing in this project addresses
  that gap.

## 8. Outlook

- **Multiword bitsets.** Extending each element's relation bitset from
  one `uint64` to two or more machine words would lift the N <= 64 cap
  that currently excludes n=80 for the lattice gas (explicitly out of
  scope in Phase 4) and any N > 64 orders-model run.
- **Parallel / windowed Wang-Landau.** Phase 3b's N=40 multicanonical
  recursion plateaued at 1-2 round trips per 30M moves across 8
  iterations without converging; a windowed, multi-walker WL scheme
  (rather than single-chain recursion) is the natural next attempt,
  informed by the round-trip-window fix in Section 6.
- **3D reproduction.** Nothing in this project yet repeats the 2D
  orders calibration in d=3; this would test whether the same
  right-censoring/blind-regridding failure modes recur in a higher-
  dimensional order space.
- **4D lattice gas.** C&S's own paper covers d>2 lattice gases; this
  project has only built d=2.
- **A cost estimate from a scaling ladder.** Phase 3b found mixing cost
  rising steeply from N=30 to N=40 (10 round trips in 508s of
  production vs. 12 round trips needing a windowed-counting fix after
  a ~1750s WL phase and a non-converging ~3850s recursion phase); a
  deliberate N=30/40/50/60 (or n=30/50/80, once multiword bitsets
  exist) ladder, timed consistently, would let a real extrapolated
  cost curve replace this project's current ad hoc per-size reporting.

## Code availability

[TBD: repository link -- not yet public/assigned at draft time.]

## Acknowledgments

The code, sampling pipeline, and analysis in this project were
developed with AI assistance (Anthropic's Claude and Claude Code),
under direction and review from the author(s). This is stated plainly
rather than left implicit.

## Appendix: run parameters, timings, hardware

**Hardware**: single laptop, Intel64 (Model 158, "Skylake"-class),
4 logical CPUs, Windows 11 (build 10.0.26200). Python 3.13.2, NumPy
2.3.2, Numba 0.68.0.

**Per-run timings** (single seed unless noted; WL = Wang-Landau phase,
MUCA = multicanonical production phase, target 10 round trips
throughout):

| Model | N / n | eps | WL | MUCA | Total | Source |
|---|---|---|---|---|---|---|
| 2D orders (PT, legacy) | 30-60 | 0.21, 0.5 | -- | -- | not separately timed (beta-ladder scan) | `results/phase3/mcmc_2d_report.md @ edb9b99` |
| 2D orders (WL/MUCA) | 30 | 0.5 | ~1700s | 508s | ~37 min | `results/phase3b/muca_calibration_report.md @ 7d648fa` |
| 2D orders (WL/MUCA) | 40 | 0.5 | ~1750s | 457.3s (windowed) | -- | same |
| 2D orders (WL/MUCA) | 30 | 0.1 | 735.5s | 150.1s | 885.6s | `results/phase4/lattice_gas_2d_report.md @ 0f85d22` |
| Lattice gas | 30 | 0.1 | 1613.7s | 948.2s | 2561.8s (~42.7 min) | same |
| Random background (5 realizations) | 30 | 0.1 | 387.3-1435.5s | 141.1-275.6s | 1623.9s total (all 5, 4 cores) | `results/phase5/random_background_report.md @ 3ae1ad5` |
| Regular lattice, matched pipeline (5 chain seeds) | 30 | 0.1 | 1345.0-5704.3s | 511.1-1057.9s | 6659.9s total (all 5, 4 cores) | same report, matched-pipeline rerun section |

**Multiprocessing**: Phase 5 stage 1 runs 5 independent realizations
via `ProcessPoolExecutor(max_workers=4)` -- 4 concurrent, the 5th
queued behind whichever finishes first. The matched-pipeline regular-
lattice rerun took **4x longer in total** than the random-background
ensemble under identical settings -- see Section 5's mechanism-check
discussion and Section 6 (not fully explained; flagged as an open
observation, not a settled finding).

---

## Summary of [TBD] items and claims needing follow-up

Resolved since the previous draft (all now cited to
`results/phase5/random_background_report.md @ 3ae1ad5` unless noted):
the random-vs-regular-background finding (Abstract, Section 5); all
per-realization beta_c values and cross-realization spread; the
matched-pipeline 5-chain-seed regular-lattice rerun and the z=51.8 gap
significance; the mechanism check (floor/hot-mean-action comparison,
hypothesis not confirmed); hot/cold-phase observables; the P_beta_c(S)
shape check and the upstreamed `_peak_diagnostics` mass-filter fix
(`causet_lab/mcmc/muca.py`, same commit) with its regression tests and
its re-check against Phase 3b/4's committed conclusions; the Section 5
control numbers (now committed, no longer `[uncommitted]`); the
Appendix table's random-background and matched-pipeline-regular-
lattice rows.

Still open:

1. **Code availability**: no repository URL assigned yet.
2. **Section 4's N=40, eps=0.1 formula leading-term** comparison was
   not separately tabulated (only N=30, eps=0.1 was); add if a reviewer
   wants the N=40 analogue.
3. **Claim I am least certain how to support cleanly**: the Section 4
   "PT vs MUCA agreement" framing. The two numbers (0.14797 and
   0.1478) agree, but since the PT value is independently flagged as
   right-censored in its own source report, I am not fully confident
   the phrase "two different sampling methods landing on the same
   number" is the right level of claim versus something weaker (e.g.
   "consistent with, but not proof of" a real feature there) --
   flagging this for a second opinion before it goes in anything more
   final than this draft.
4. **Wang-Landau and PT themselves are described by name without a
   pinned citation** (no paper using those exact terms was fetched and
   checked against this project's code, unlike the four causal-set
   papers and the Berg-Neuhaus one, which were). Add standard
   citations (Wang & Landau 2001; parallel tempering's standard
   references) if the published version needs them -- deliberately
   left out here rather than guessed.
5. **New from this round**: the mechanism behind the beta_c shift
   (Section 5) and the regular lattice's 4x-slower mixing under this
   pipeline (Section 5/6) are both reported as genuine open questions,
   not just formatting placeholders -- flagging explicitly so a
   reviewer doesn't mistake "ruled out the obvious explanation" for
   "found the real one."
