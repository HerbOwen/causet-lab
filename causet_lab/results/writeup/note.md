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
computational cost. In their Conclusions (Sec. 5), C&S write: "we have
chosen a specific, regular lattice, and we don't yet know whether our
results will change with a different lattice geometry. A more natural
choice for the background lattice is a random m element causal set
obtained via a sprinkling into (M, g). It would be important to
explore simulations on such lattices and compare with our current
results" -- flagged as future work, not pursued in their paper.

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
backgrounds. "5 seeds" means different things for the two backgrounds:
for the random background, 5 independent background realizations
(5 different quenched sprinklings, each with its own Monte Carlo
chain); for the regular lattice, which is the same deterministic
lattice every time, 5 independent Monte Carlo chain seeds on that one
lattice:

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
used to produce it: two different Wang-Landau sampling methods agree
to within 0.008, checked by directly re-running one seed from each
background (one random-background realization, one regular-lattice
chain seed) under both methods.

**The *relative* gap shrinks with size, consistent with (not proven
by) 1/n scaling -- the same rate beta_c itself falls off here.** The
relative shift times n is close to constant across the two sizes we
tested (12.4% x 30 = 371 vs. 7.2% x 50 = 358, a two-point power-law
exponent of about 1.07). Because the *absolute* gap is the relative
gap times beta_c, and both factors fall off at roughly 1/n, the
absolute gap falls off faster, close to 1/n^2: it drops 2.98x from
n=30 to n=50 (1.372 -> 0.460), close to the (50/30)^2 = 2.78x a pure
1/n^2 law predicts, not the 1.67x a 1/n law on the absolute gap alone
would give. Taken at face value, the 1/n relative-gap reading predicts
a relative shift of about 5.6% at n=64 -- the largest size reachable
with this project's current code without a bitset-engine rewrite
(every model here caps out at N, n <= 64) -- a concrete number anyone
can check against a direct run. Extrapolating further (well beyond
where this project has any data, so treat this as a back-of-envelope
number, not a result): at n=200, the system size C&S themselves use
for their 2D lattice gas, the same 1/n reading predicts a relative gap
of only about **1.8%** -- small enough that it's plausible their
published 2D results are only mildly affected by the regular
background, and directly testable by running their own code on a
sprinkled background at their own system size.

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

- **Only two system sizes.** The 1/n-consistent shrinkage of the
  *relative* gap above is a two-point observation, not a fitted law;
  a third size (n=64 is the concrete next test) would meaningfully
  sharpen this.
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
- **The lattice causal relation is our own derivation, not a quote.**
  C&S's paper states 45-degree lightcones and cylinder wraparound but
  does not give an explicit integer-coordinate formula, so we derived
  one ourselves (shortest arc on the circumference) and flagged it
  explicitly as our own construction rather than a quote. Supporting
  evidence that it matches their intended model: our regular-lattice
  hot phase reproduces their own quoted ordering fraction almost
  exactly (0.882 here vs. their 0.88), which would be a coincidence if
  the causal relation were substantially wrong.
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
