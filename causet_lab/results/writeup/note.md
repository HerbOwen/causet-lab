---
title: "Does a regular background bias 2D causal-set lattice-gas quantum gravity? A random-background comparison"
status: "Independent project note, v1 (2026-10-09) -- not peer reviewed"
---

> This is a short note. For full validation results, every pitfall
> found and fixed along the way, and commit-level citations for every
> number below, see `full_report.md` in this same folder.

## The open question

Cunningham & Surya's dimensionally-restricted "lattice gas"
[arXiv:1908.11647] makes 2D causal-set Monte Carlo cheap enough to run
by placing causal-set elements on a fixed *regular* lattice embedded
in a cylinder spacetime, rather than sampling generic partial orders
directly. It reproduces the generic model's phase transition between a
manifold-like phase and a layered, highly-ordered phase at much lower
computational cost. In their Section 6, C&S note explicitly that this
regular background is a modeling choice, not a physical requirement,
and that checking whether it biases the result -- as opposed to using
"a more realistic model of discreteness" -- is an open question they
raise but do not pursue.

## The answer

**Yes, the location of the transition depends on the background being
regular -- the effect is real, significant, and shrinks with system
size, but does not vanish at the two sizes tested; the qualitative
physics (phase structure, hot-phase statistics) does not depend on the
background.**

We built a random-background variant of the lattice gas: identical
region, filling representation, Monte Carlo moves, and action, with
the only change being that the m lattice sites are replaced by a
fixed (quenched) uniform-random sprinkling of continuous positions. An
explicit equivalence test confirms this reduces exactly to the regular
lattice's own causal structure when the random positions are placed
back on the integer grid, so "random background" and "regular
lattice" differ only in that one respect.

## What we measured

Using a Wang-Landau / multicanonical sampler (built to replace an
earlier parallel-tempering scan whose results turned out to be mostly
right-censored -- see `full_report.md` Section 6), we located the
critical coupling beta_c at two system sizes, n=30 and n=50, for both
backgrounds, with 5 independent realizations each:

| n | Random background beta_c | Regular lattice beta_c | Gap | z | Relative shift |
|---|---|---|---|---|---|
| 30 | 9.723 +/- 0.026 (SEM) | 11.095 +/- 0.005 (SEM) | 1.372 | 51.8 | 12.4% |
| 50 | 5.959 +/- 0.033 (SEM) | 6.419 +/- 0.005 (SEM) | 0.460 | 13.67 | 7.2% |

At both sizes the regular lattice's located beta_c is significantly
higher than the random background's -- not explainable by run-to-run
noise on either side (z=51.8 and z=13.67 are both far beyond any
conventional significance threshold). The regular-lattice value at
n=30 is also cross-checked against this project's independent earlier
measurement of the same model (within about 2 standard deviations),
and the random background's result is robust to the exact pipeline
used to produce it (two different Wang-Landau sampling methods agree
to within 0.008 on two directly re-checked seeds).

**The gap shrinks with size, consistent with (not proven by) 1/n
scaling.** The relative shift times n is close to constant across the
two sizes we tested (12.4% x 30 = 371 vs. 7.2% x 50 = 358, a two-point
power-law exponent of about 1.07), suggesting the absolute gap falls
off at roughly the same rate as beta_c itself. Taken at face value,
this predicts a relative shift of about 5.6% at n=64 -- the largest
size reachable with this project's current code without a bitset-
engine rewrite (every model here caps out at N, n <= 64) -- a concrete
number anyone can check against a direct run.

**We checked and ruled out the simplest mechanism.** A natural guess
is that the random background simply reaches a deeper, more-ordered
cold-phase action floor, which would mechanically pull beta_c down.
Directly comparing the lowest action reached and the hot-phase mean
action between the two backgrounds shows no statistically significant
difference -- this hypothesis does not hold. The real mechanism
(presumably something about the shape of the density of states across
the explored range, not its extremes) is still open.

**What does *not* change**: the hot phase (beta=0 fillings) is close
to background-independent at both sizes -- every measured observable
(Myrheim-Meyer dimension, ordering fraction, height, occupied row
count) lands within about one standard deviation of the regular
lattice's own value. The cold phase differs more at n=30 but that gap
has nearly closed by n=50 (ordering fraction 0.661 vs. 0.660, MM
dimension 3.60 vs. 3.60). Neither background, at either size, shows a
resolved double-peaked action histogram at its own critical coupling
-- so this project does not confirm a first-order transition on either
background, and the random-background result above is about *where*
the transition sits, not about resolving its order.

## Caveats

- **Only two system sizes.** The 1/n-consistent shrinkage above is a
  two-point observation, not a fitted law; a third size (n=64 is the
  concrete next test) would meaningfully sharpen this.
- **Mixed sampler methods at n=50.** Two Wang-Landau bugs were found
  and fixed partway through the n=50 runs (an unphysical histogram
  cliff from a flat-fill widening step, and a genuinely
  kinetically-isolated bin that no amount of reweighting could
  repair -- both described in full in `full_report.md` Section 6). The
  n=50 dataset mixes 5 seeds run under the original method with 5 run
  under the fixed method; a direct 2-seed consistency check found no
  material difference (0.002 and 0.008), but a fully single-method
  n=50 dataset was not produced.
- **Mechanism still open.** We ruled out the simplest explanation for
  *why* the backgrounds differ, not the real cause.
- **2D only.** C&S's own paper, and the underlying causal-set action,
  both extend to higher dimensions; this project has not run a 3D or
  4D version of either model.

## Where to go next

`full_report.md` in this same folder has the complete validation
story (the order model checked against the published finite-size-
scaling formula, the lattice gas checked against C&S's own quoted
numbers), every methodological pitfall found and fixed along the way,
and exact commit-level citations for every number in this note. The
code, tests, and scripts to reproduce any of it are in the same
repository; see `README.md` at the repository root.
