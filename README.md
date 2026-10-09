# causet_lab

A causal-set quantum gravity Monte Carlo project: a Wang-Landau /
multicanonical (WL/MUCA) sampler for the 2D causal-set order model and
for Cunningham & Surya's dimensionally-restricted lattice gas, plus a
new result comparing that lattice gas on its usual regular background
against a random (quenched Poisson) background.

**Start here**: [`causet_lab/results/writeup/note.md`](causet_lab/results/writeup/note.md)
(also as [`note.pdf`](causet_lab/results/writeup/note.pdf)) is a 3-4
page note leading with Cunningham & Surya's own open question
(arXiv:1908.11647, Sec. 5, their Conclusions) and this project's
answer to it -- read
this first. [`full_report.md`](causet_lab/results/writeup/full_report.md)
(also [`full_report.html`](causet_lab/results/writeup/full_report.html) /
[`full_report.pdf`](causet_lab/results/writeup/full_report.pdf)) is the
full write-up: every validation result, pitfall, and limitation, each
cited to an exact git commit. This README is about running the code,
not the physics.

## The headline result, in one paragraph

Comparing Cunningham & Surya's regular-lattice causal-set "lattice gas"
against a random-background variant built for this project (same
region, filling, moves, and action; sites quenched-sprinkled instead
of on a regular grid), the located critical coupling beta_c is
**significantly lower on the random background** at both system sizes
tested: 9.723 vs. 11.095 at n=30 (z=51.8, a ~12% relative shift) and
5.959 vs. 6.419 at n=50 (z=13.67, ~7%). The relative shift times n is
close to constant across the two sizes (12.4%x30=371 vs. 7.2%x50=358),
consistent with — but, from only two points, not proof of — a
finite-size effect that fades roughly as 1/n; taken at face value, the
same reading predicts a relative gap of only about 1.8% at n=200 (the
system size Cunningham & Surya themselves use), small enough that
their published 2D results are plausibly only mildly affected. A
direct mechanism check rules out the simplest explanation (the random
background reaching a deeper cold-phase action floor) without
identifying the real cause.
Caveats that matter: only two system sizes (`N<=64` is a hard limit of
the current bitset engine, see below), the n=50 dataset mixes two
sampler methods by seed (justified by a 2-seed consistency check, not
by assumption — see Section 5/7 of the paper), and no first-order
double-peak signature was found at any size or background tested, so
the transition's order is not settled here either way.

## What's in the repo

```
causet_lab/
  causet.py, generators.py, measures.py, rules.py, plots.py, run.py
                        Phase 1-2: causal sets from first principles
                        (sprinkles, Kleitman-Rothschild, growth rules),
                        dimension/abundance estimators. See
                        causet_lab/README.md for the physics intro to
                        this part specifically.
  rules_study.py, scaling_study.py, followup_study.py
                        Phase 2 batch studies over growth rules.
  mcmc/                 Phase 3-5: the MCMC/WL/MUCA samplers.
    action.py           Smeared Benincasa-Dowker action (orders + C&S's
                         own lattice-gas normalization).
    orders.py, sampler.py, fast_core.py
                        2D order model + its Metropolis/PT sampler.
    lattice_gas.py, lattice_gas_core.py
                        Cunningham & Surya's regular-lattice gas.
    random_bg.py, random_bg_core.py
                        This project's random-background variant.
    bitset_core.py      Shared O(N) incremental relation bookkeeping
                        (uint64-per-element bitsets -- caps every model
                        here at N, n <= 64).
    muca.py, muca_core.py
                        Wang-Landau + multicanonical sampler core,
                        shared by all three models above.
    tempering.py        Parallel tempering (the original, since-
                        replaced Phase 3 approach).
    checkpoint.py, rng.py
                        Atomic checkpointing; splitmix64 RNG.
    study.py            Phase 3's full PT beta-ladder study driver.
  tests/                72 tests; run before trusting anything below.
  results/
    phase2/ .. phase5/  One report per phase, each citing exact git
                        commits for every number (see full_report.md's
                        citation convention). phase2/phase3/phase3b
                        also hold that phase's generated PNG plots,
                        saved alongside the report as companion
                        artifacts rather than embedded inline.
    writeup/            note.md / .pdf (3-4 page summary, read first),
                        full_report.md / .html / .pdf (the full
                        write-up). A draft outreach email also lives
                        here locally (email_draft.md) but is
                        git-ignored, not part of this repo.
scripts/                Rescued, repo-relative driver and diagnostic
  phase3_followup/      scripts that actually produced the numbers in
  phase3b/              results/*.md and full_report.md. Organized by the
  phase4/               phase they were used in. See "Reproducing
  phase5/                headline numbers" below.
data/                   Small (<1MB each) cached result pickles the
                        scripts above read/write, plus one input log
                        (mcmc_full_run.log). Multi-GB raw WL/MUCA
                        checkpoints are NOT committed here -- see
                        "What's not included" below.
requirements.txt
LICENSE, CITATION.cff
```

## Installation

Python 3.13 (tested on 3.13.2). From the repo root:

```bash
pip install -r requirements.txt
```

This installs `numpy==2.3.2`, `numba==0.68.0`, `matplotlib==3.11.2`,
`pytest==9.1.1` (and, only if you want to regenerate the HTML/PDF
write-up, `markdown==3.11`). No `setup.py`/`pyproject.toml` — the
package is just imported as `causet_lab` with the repo root on
`sys.path`, which is automatic if you run everything (tests, scripts)
*from the repo root*, as all commands below assume.

## Running the tests

```bash
python -m pytest causet_lab/tests/ -q
```

72 tests, ~80s on one core. All pass as of the commit this README was
added in. These are unit/regression tests (bitset correctness, action
formulas, WL/MUCA diagnostics, the random-background/regular-lattice
equivalence check) — fast, and a prerequisite for trusting any
reproduction below, not a substitute for it.

## Reproducing headline numbers and figures

Everything below assumes 4 logical CPUs (this project's own hardware;
see `full_report.md`'s Appendix) and is run from the repo root. Wall-clock
figures are *measured* numbers from this project's own runs (cited in
`full_report.md`'s Appendix table), not estimates, except where marked
"not separately timed."

**Fast path (seconds): recompute the Phase 5 analysis from already-run
data**, without touching Wang-Landau at all:

```bash
python scripts/phase5/phase5_merge_n50.py      # rebuilds the n=50 beta_c
                                                # table, gap, z, pct_shift
                                                # from data/*.pkl
python scripts/phase5/phase5_n50_pbetac_check.py  # the double-peak check
                                                   # for all 10 n=50 seeds
```

Both run in a few seconds and reproduce the exact numbers in `full_report.md`
Section 5 / the Appendix.

**Full regeneration from scratch**, phase by phase:

| Phase | Result | Command(s) | Measured wall time (4 cores) |
|---|---|---|---|
| 1-2 | `results/phase2/summary.md`, controls/junk/percolation plots | `python -m causet_lab.run all` | not separately timed; seconds-to-low-minutes (N<=2000) |
| 2 | `results/phase2/rules_report.md` | `python -c "from causet_lab.rules_study import run_study; run_study()"` | not separately timed |
| 2 | `results/phase2/followup_report.md` | `python -c "from causet_lab.followup_study import run_followup_study; run_followup_study()"` | not separately timed |
| 2 | `results/phase2/scaling_report.md` | `python -c "from causet_lab.scaling_study import run_scaling_study; run_scaling_study()"` | not separately timed (N up to 16000) |
| 3 | `results/phase3/mcmc_2d_report.md` (orders, PT, N=30-60, eps=0.21/0.5) | `python -c "from causet_lab.mcmc.study import run_mcmc_study; run_mcmc_study()"` | **2365.1s** (~39 min), measured; log in `data/mcmc_full_run.log` |
| 3 follow-up | Blind re-centering / right-censoring check | `python scripts/phase3_followup/blind_test.py` | reads `data/mcmc_full_run.log`; seconds |
| 3b | `results/phase3b/muca_calibration_report.md` (N=30, eps=0.5, WL/MUCA) | `python scripts/phase3b/pilot_n30_eps0p5.py` then `pilot_n30_finish.py` | ~1700s WL + 508s MUCA (~37 min) |
| 3b | Same, N=40 | `python scripts/phase3b/pilot_n40_eps0p5.py`, then `pilot_n40_recursion.py`, then `pilot_n40_windowed.py` | ~1750s WL + 457.3s MUCA (windowed) |
| 4 | `results/phase4/lattice_gas_2d_report.md`, orders eps=0.1 N=30 cross-check | `python scripts/phase4/phase4_orders_crosscheck_n30.py` | 735.5s WL + 150.1s MUCA |
| 4 | Same report, lattice gas n=30 WL/MUCA | `python scripts/phase4/phase4_pilot_n30.py` | 1613.7s WL + 948.2s MUCA |
| 4 | Same report, hot/cold phase observable table | `python scripts/phase4/phase4_phase_analysis_n30.py` (reads the checkpoint `phase4_pilot_n30.py` writes) | seconds, once the pilot above has run |
| 5 Stage 1 | `results/phase5/random_background_report.md`, n=30 (random bg) | `python scripts/phase5/phase5_run_n.py` | 1623.9s total, 5 seeds, 4 cores |
| 5 Stage 1 | Same, n=30 (regular lattice, matched pipeline) | `python scripts/phase5/phase5_run_regular_via_randombg.py` | 6659.9s total, 5 seeds, 4 cores |
| 5 Stage 1 | Mechanism check (deeper-cold-floor hypothesis) | `python scripts/phase5/phase5_mechanism_check.py` | needs the two runs above's checkpoints first; seconds once they exist |
| 5 Stage 2 | n=50 pre-flight timing estimate | `python scripts/phase5/phase5_pilot_n50.py` | ~7 min |
| 5 Stage 2 | n=50 production, both backgrounds, 10 seeds | `python scripts/phase5/phase5_run_n50_combined.py` | ~4.9 hours (the two backgrounds share one 4-worker pool; see script docstring) |
| 5 Stage 2 | Fix + re-run the 3 seeds that failed under the adaptive-widen method, + 2 fairness seeds | `python scripts/phase5/phase5_rerun_failed_n50_v2.py` | 5 jobs, 4-worker pool; see `data/phase5_rerun_failed_n50_v2.log`-style timing in the script's own prints |
| 5 Stage 2 | Merge into the final 10-seed dataset + double-peak check | the two "fast path" commands above | seconds |

Notes on this table:
- Phase 3b and 4's pilot/finish/recursion/windowed scripts are a
  multi-step pipeline because Wang-Landau's recursion was run,
  checkpointed, inspected, and resumed interactively during this
  project — they are not a single clean driver. Each script prints
  what it did and where it checkpointed; read the docstring at the top
  of each before running.
- `phase5_rerun_failed_n50.py` (without `_v2`) is kept for the
  historical record (Section 6 of `full_report.md`: it still hit the
  kinetic-bottleneck bug) — `_v2` is the one that actually produced
  the committed n=50 numbers.
- `phase5_diagnose_stall_n50.py`/`_part2.py` and
  `phase5_scan_ln_g_gaps.py`/`verify_stall_guard.py` are the diagnostic
  scripts that found the two Wang-Landau bugs described in `full_report.md`
  Section 6; they expect checkpoints that only exist mid-run or after
  a from-scratch regeneration, not from the cached `data/*.pkl` files.

## What's not included

Multi-GB raw WL/MUCA checkpoints (full `ln_g`/histogram arrays at
every Wang-Landau stage) and three >50MB intermediate walker-history
pickles are **not** committed — they're regenerable by the scripts
above and don't belong in git. The one exception: a ~25KB cache,
`data/phase5_n50_pbetac_curves.pkl`, holding just the final reweighted
`P_beta_c(S)` curve and peak diagnostic for the 5 n=50 seeds whose raw
checkpoints aren't included, so `phase5_n50_pbetac_check.py` reproduces
exactly without a multi-hour rerun.

## Known limitations and open questions

See `full_report.md` Sections 7-8 for the full, current list; in short:
- **System size**: every model here is capped at `N, n <= 64` by the
  bitset engine (one `uint64` per element). n=64 is reachable with no
  code changes and has a concrete, falsifiable prediction attached
  (full_report.md Section 8: ~5.6% relative beta_c gap); n=80 and beyond need
  a multiword-bitset rewrite.
- **Mechanism**: what actually causes the beta_c shift (only the
  simplest hypothesis — a deeper cold-phase floor — has been ruled
  out) and why the regular lattice mixed ~4x slower than the random
  background at n=30 are both open.
- **3D/4D**: this project only covers d=2; C&S's own paper and the
  generic causal-set action both extend to higher dimensions.
- **Methodological**: the n=50 dataset mixes two sampler methods by
  seed (see full_report.md Section 5/7); the n=50 mixing-speed comparison
  rests on partial logs for only 3+2 of the 10 seeds.
- **First-order order parameter**: no double-peak `P_beta_c(S)`
  signature was found at any size, epsilon, or background tested in
  this project — consistent with either a weak/crossover transition
  below the relevant asymptotic regime, or just insufficient
  statistics; the data here cannot tell the two apart.

## License and citation

MIT license (see `LICENSE`). See `CITATION.cff` for citation metadata.
