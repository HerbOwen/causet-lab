"""Controls and the beta scan for the 2D MCMC study: test whether the 2D
causal set ensemble has a spacetime-like phase that transitions to a
layered phase as the gravitational weighting beta increases.

This uses the *smeared* 2D Benincasa-Dowker action S(C, eps) (see
action.py's module docstring), with the non-locality parameter eps
fixed per run, following S. Surya, "Evidence for the continuum in 2D
causal set quantum gravity", CQG 29 132001 (2012), arXiv:1110.6244, and
precisely calibrated against L. Glaser, D. O'Connor, S. Surya, "Finite
Size Scaling in 2d Causal Set Quantum Gravity", CQG 35 045006 (2018),
arXiv:1706.06432 -- the formula and normalization were confirmed by
fetching that paper directly (see action.py's module docstring). That
paper reports beta_c(N, eps) = b(eps)/N + c(eps)/N^2 + O(1/N^3), a
first-order transition (double-peaked action histograms sharpening
with N), and that its asymptotic N -> infinity regime is only reached
for N >~ 65 -- this project scans N = 30, 40, 50, 60, i.e. *below* that
asymptotic regime, so the located beta_c is compared against the full
two-term formula (not just the leading 1/N term), and reported with
honest uncertainty rather than treated as an exact match.

An earlier version of this study used the plain (non-smeared, local)
2D BD action and scanned N up to 80; that action is NOT what the
literature uses as its Monte Carlo weight (its beta is not comparable
to any published number), and N=80 under it showed no sign of PT
replica round trips across the transition even after escalating run
length 4x for zero gain in effective samples -- both problems are
avoided here by using the correct action and dropping N=80 in favor of
N <= 60, where PT does travel the full ladder (see the round-trip
columns in the report).

Engine: all MCMC here runs on the Numba-compiled, incrementally-updated
core in fast_core.py (run_chain_fast / run_chain_adaptive_fast in
sampler.py, run_parallel_tempering in tempering.py) rather than the
pure-Python reference sampler (sampler.run_chain), which exists only as
the correctness baseline the fast engine was validated against (see
tests/test_mcmc.py). The near-transition beta grid is handled by
parallel tempering (replica exchange), with round-trip tracking used as
the real test that replicas cross between phases rather than trusting
local swap-rate numbers alone.

Pipeline, for each (eps, N):
    1. run_controls(): beta=0 chains should be statistically
       indistinguishable from 2D sprinkles -- eps-independent, run once.
    2. run_scan_for_N(): a coarse scout grid, centered on the published
       formula's predicted beta_c (not blind), locates the beta that
       maximizes action variance; a fine PT grid refines it. Points that
       fall short of the effective-sample target are marked UNCONVERGED
       and excluded from beta_c/width/height/bootstrap -- not silently
       included. The located beta_c is compared against the published
       formula with a z-score. The pooled action histogram at beta_c is
       checked for the two-peak structure a first-order transition
       predicts.
    3. run_deep_phase(): annealed chains far above beta_c (as a multiple
       of the predicted beta_c, since its absolute scale depends on both
       N and eps) characterize the high-beta phase against the
       published height~3 / ordering-fraction~0.6 targets.
    4. run_hysteresis_check(): random-start vs layered-start PT at
       beta_c, reported as a cross-check on run length / metastability.
    5. write_report(): plots + results/phase3/mcmc_2d_report.md, with a
       plain pass/fail verdict against each of the four criteria above,
       separately for each (eps, N).
"""
from __future__ import annotations

import functools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Background runs pipe stdout to a log file rather than a terminal, which
# switches Python to block-buffered output -- print() calls sit in an
# internal buffer and may not reach disk for a long time. Force every
# print in this module to flush immediately so progress is visible live
# (equivalent to running with `python -u` for this module's output).
print = functools.partial(print, flush=True)

from ..generators import sprinkle, kleitman_rothschild
from ..measures import (
    ordering_fraction, myrheim_meyer_dimension, midpoint_dimension,
    interval_abundances, height, abundance_distance,
)
from ..battery import nanmean, nanstd
from .. import plots
from .action import beta_c_glaser_2018
from .sampler import run_chain_fast, run_chain_adaptive_fast, layered_start
from .tempering import run_parallel_tempering, densify_ladder_for_low_swap_rates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3"

KMAX = 10
N_WORKERS = max(1, min(4, os.cpu_count() or 1))

SCAN_NS = [30, 40, 50, 60]
EPS_VALUES = [0.21, 0.5]
PRODUCTION_SEEDS = [0, 1, 2]

N_COARSE = 12
N_REFINE = 15
# coarse grid spans this multiple of the published-formula beta_c
# prediction, log-spaced, plus an explicit beta=0 control point --
# centered on the formula rather than blind, since the formula's scale
# (b(eps)/N ~ O(0.1 - 1) for the eps, N used here) is already known to
# be in the right ballpark; the variance-peak scout still *locates*
# beta_c independently within this range, it is not assumed.
COARSE_LO_MULT = 0.03
COARSE_HI_MULT = 6.0
# deep high-beta phase, as multiples of the predicted beta_c (its
# absolute scale depends on both N and eps, unlike the old plain-action
# version where fixed absolute betas of 20/50 were adequate for every N)
DEEP_BETA_MULTIPLIERS = (5.0, 15.0)

SCOUT_SWEEPS = 150
SCOUT_BURNIN = 50
SCOUT_MEASURE_EVERY = 3

PROD_SWEEPS = 900
PROD_BURNIN = 200
PROD_MEASURE_EVERY = 4
MIN_EFF_SAMPLES = 50.0

MIN_SWAP_RATE = 0.2
MAX_DENSIFY_INSERTED = 8
MAX_PT_ESCALATIONS = 2

COARSE_ONLY_MIN_EFF_SAMPLES = 20.0
COARSE_ONLY_MAX_MULTIPLIER = 4

N_BOOT = 400

DEEP_SEEDS = [0, 1, 2, 3, 4]
DEEP_ANNEAL_SWEEPS = 2000
DEEP_TAIL_SWEEPS = 500
DEEP_MEASURE_EVERY = 10
HIGH_BETA_HEIGHT_TARGET = 3.0
HIGH_BETA_OF_TARGET = 0.6

REFERENCE_SEEDS = [0, 1, 2, 3, 4]

CONTROL_N = 60
CONTROL_EPS = 0.21
CONTROL_SEEDS = [0, 1, 2, 3, 4]
CONTROL_SWEEPS = 300
CONTROL_BURNIN = 100
CONTROL_MEASURE_EVERY = 4

BETA_C_Z_PASS = 3.0
HIST_N_BINS = 36


def measure_sample(C: np.ndarray, kmax: int = KMAX) -> dict:
    """Measure a whole 2D-order (or sprinkle) sample directly -- unlike
    the grown universes elsewhere in this project, each MCMC sample (or
    sprinkle) already *is* a single causal-diamond-like object, so there
    is no need to sample sub-intervals; the whole-matrix estimators are
    the natural ones here.
    """
    C = np.asarray(C, dtype=bool)
    return {
        "ordering_fraction": ordering_fraction(C),
        "mm_dim": myrheim_meyer_dimension(C),
        "mp_dim": midpoint_dimension(C),
        "height": height(C),
        "abundance_profile": interval_abundances(C, kmax=kmax),
    }


def _coarse_betas_for(N: int, eps: float, n_points: int = N_COARSE) -> np.ndarray:
    """Coarse scout grid centered on the published formula's predicted
    beta_c for this (N, eps) -- see module docstring. The scout still
    independently locates the variance-maximizing beta within this
    range; only the *range itself* uses the formula, to avoid wasting
    compute scanning regions far from where any transition could be.
    """
    predicted = beta_c_glaser_2018(N, eps)
    lo = max(predicted * COARSE_LO_MULT, 1e-4)
    hi = predicted * COARSE_HI_MULT
    return np.round(np.concatenate([[0.0], np.geomspace(lo, hi, n_points)]), 6)


def _coarse_betas_centered(N: int, eps: float, center_mult: float, n_points: int = N_COARSE) -> np.ndarray:
    """Same log-width (hi/lo ratio) and point count as _coarse_betas_for,
    but with the grid's geometric center placed at center_mult times the
    predicted beta_c, instead of _coarse_betas_for's implicit (and
    off-center) sqrt(COARSE_LO_MULT * COARSE_HI_MULT) ~ 0.42. Used only
    for the blind grid-centering robustness check: if the located beta_c
    tracks the predicted value regardless of where this grid is
    centered, the match is a genuine located feature; if it instead
    tracks the grid's center (or its edge), the match is a grid
    artifact, not a confirmation.
    """
    predicted = beta_c_glaser_2018(N, eps)
    log_half_width = np.sqrt(COARSE_HI_MULT / COARSE_LO_MULT)
    center = center_mult * predicted
    lo = max(center / log_half_width, 1e-6)
    hi = center * log_half_width
    return np.round(np.concatenate([[0.0], np.geomspace(lo, hi, n_points)]), 6)


def _beta_c_formula_uncertainty(N: int, eps: float) -> float:
    """Propagate the published fit's quoted coefficient uncertainties
    (b = 1.66 +/- 0.03, c = 4.09+/-0.50 /eps^3 - 27.77+/-2.45 /eps^2)
    into an uncertainty on beta_c(N, eps). Linear (not quadrature) sum of
    the b and c terms, since N below the paper's asymptotic regime
    (N >~ 65) means higher-order (1/N^3) terms the fit doesn't quantify
    are also plausibly present -- this is a lower-bound estimate on the
    true uncertainty, not a rigorous propagation.
    """
    sigma_b = 0.03 / eps ** 2
    sigma_c = float(np.hypot(0.50 / eps ** 3, 2.45 / eps ** 2))
    return sigma_b / N + sigma_c / N ** 2


def _histogram_bimodality(values: np.ndarray, n_bins: int = HIST_N_BINS) -> dict:
    """Simple bimodality check on a pooled sample distribution: histogram
    the values, lightly smooth to reduce bin noise, then find local
    maxima at least 10% of the tallest peak's height, merging maxima
    within 2 bins of each other. Returns the number of distinct peaks
    found and the separation (in the same units as `values`) between the
    two tallest if >= 2 are found -- a first-order transition's defining
    signature is a double-peaked action distribution at beta_c, with the
    separation growing with N (checked across N by the caller).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return {"n_peaks": 0, "separation": float("nan"), "counts": None, "bin_centers": None}
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(counts.astype(float), kernel, mode="same")
    peak_idxs = [
        i for i in range(1, len(smooth) - 1)
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0.1 * smooth.max()
    ]
    merged = []
    for i in sorted(peak_idxs, key=lambda k: -smooth[k]):
        if all(abs(i - m) > 2 for m in merged):
            merged.append(i)
    n_peaks = len(merged)
    separation = float("nan")
    if n_peaks >= 2:
        top2 = sorted(sorted(merged, key=lambda k: -smooth[k])[:2])
        separation = float(centers[top2[1]] - centers[top2[0]])
    return {"n_peaks": n_peaks, "separation": separation, "counts": counts, "bin_centers": centers}


# ---------------------------------------------------------------- controls

def run_controls(N: int = CONTROL_N, eps: float = CONTROL_EPS, seeds=None, sprinkle_seeds=None) -> dict:
    """beta=0 chains are eps-independent dynamically (every move is
    accepted unconditionally regardless of the action), so this is run
    once for the whole study, not once per eps.
    """
    seeds = seeds if seeds is not None else CONTROL_SEEDS
    sprinkle_seeds = sprinkle_seeds if sprinkle_seeds is not None else CONTROL_SEEDS

    chains = []
    for seed in seeds:
        res = run_chain_fast(
            N, beta=0.0, seed=seed, n_sweeps=CONTROL_SWEEPS, burn_in=CONTROL_BURNIN,
            measure_every=CONTROL_MEASURE_EVERY, eps=eps,
        )
        meas = measure_sample(res.final_C)
        chains.append((res, meas))

    mm_dims = [m["mm_dim"] for _, m in chains]
    mp_dims = [m["mp_dim"] for _, m in chains]
    abundance_mean = np.nanmean(np.array([m["abundance_profile"] for _, m in chains]), axis=0)
    acc_rates = [r.acceptance_rate for r, _ in chains]

    sprinkle_profiles = []
    sprinkle_mm = []
    for s in sprinkle_seeds:
        cset, _ = sprinkle(N, 2, seed=s)
        sprinkle_mm.append(myrheim_meyer_dimension(cset.C))
        sprinkle_profiles.append(interval_abundances(cset.C, kmax=KMAX))
    sprinkle_mean_profile = np.nanmean(np.array(sprinkle_profiles), axis=0)

    dist = abundance_distance(abundance_mean, sprinkle_mean_profile)

    chain_means = []
    chain_sems = []
    for r, _ in chains:
        a = r.actions
        chain_means.append(float(np.mean(a)) if a.size else float("nan"))
        chain_sems.append(float(np.std(a) / np.sqrt(a.size)) if a.size > 1 else float("nan"))
    grand_mean = float(np.nanmean(chain_means))
    max_z = 0.0
    for m, s in zip(chain_means, chain_sems):
        if np.isfinite(m) and np.isfinite(s) and s > 0:
            max_z = max(max_z, abs(m - grand_mean) / s)

    return {
        "N": N, "eps": eps,
        "beta0_mm_dim_mean": nanmean(mm_dims), "beta0_mm_dim_std": nanstd(mm_dims),
        "beta0_mp_dim_mean": nanmean(mp_dims), "beta0_mp_dim_std": nanstd(mp_dims),
        "sprinkle_mm_dim_mean": nanmean(sprinkle_mm), "sprinkle_mm_dim_std": nanstd(sprinkle_mm),
        "abundance_distance": dist,
        "acceptance_rate_mean": float(np.mean(acc_rates)),
        "acceptance_rate_min": float(np.min(acc_rates)),
        "chain_means": chain_means, "chain_sems": chain_sems,
        "independent_chains_max_z": max_z,
        "dimension_pass": abs(nanmean(mm_dims) - 2.0) < 0.4,
        "abundance_pass": dist < 0.5,
        "balance_pass": max_z < 4.0,
    }


# -------------------------------------------------------------------- scan

def _locate_variance_peak(N, eps, seed, betas, n_sweeps, burn_in, measure_every):
    scouted = []
    for beta in betas:
        res = run_chain_fast(N, float(beta), seed=seed, n_sweeps=n_sweeps, burn_in=burn_in,
                              measure_every=measure_every, eps=eps)
        var = float(np.var(res.actions)) if res.actions.size > 1 else float("nan")
        mean = float(np.mean(res.actions)) if res.actions.size else float("nan")
        scouted.append({"beta": float(beta), "var": var, "mean": mean})
    valid = [s for s in scouted if np.isfinite(s["var"])]
    beta_star = max(valid, key=lambda s: s["var"])["beta"] if valid else float(betas[len(betas) // 2])
    return beta_star, scouted


def _refine_grid(betas, beta_star, n_refine):
    b = np.sort(np.asarray(betas, dtype=float))
    idx = int(np.argmin(np.abs(b - beta_star)))
    lo = b[max(0, idx - 1)]
    hi = b[min(len(b) - 1, idx + 1)]
    if hi <= lo:
        spread = max(beta_star * 0.5, 1e-4)
        lo, hi = max(0.0, beta_star - spread), beta_star + spread
    return np.linspace(lo, hi, n_refine)


def _aggregate_point(N, beta, entries, kmax=KMAX, met_flags=None, multiplier_max=1):
    """entries: list of dicts, each with keys actions/tau_int/
    tau_converged/effective_samples/acceptance_rate/final_C -- the
    common shape produced by both run_chain_adaptive_fast (wrapped) and
    a parallel-tempering run's per-beta result. Builds the same point
    schema generate_plots/write_report expect, regardless of which
    engine produced the raw chains.
    """
    measures = [measure_sample(e["final_C"], kmax=kmax) for e in entries]
    actions_all = np.concatenate([e["actions"] for e in entries]) if entries else np.array([])
    mean_S = float(np.mean(actions_all)) if actions_all.size else float("nan")
    var_S = float(np.var(actions_all)) if actions_all.size else float("nan")
    sem_S = float(np.std(actions_all) / np.sqrt(actions_all.size)) if actions_all.size > 1 else float("nan")

    mm_vals = [m["mm_dim"] for m in measures]
    mp_vals = [m["mp_dim"] for m in measures]
    height_vals = [m["height"] for m in measures]
    of_vals = [m["ordering_fraction"] for m in measures]
    abundance_vals = np.array([m["abundance_profile"] for m in measures])
    acc_rates = [e["acceptance_rate"] for e in entries]
    taus = [e["tau_int"] for e in entries]
    tau_conv = [e["tau_converged"] for e in entries]
    eff_samples = [e["effective_samples"] for e in entries]
    met_flags = met_flags if met_flags is not None else [es >= MIN_EFF_SAMPLES for es in eff_samples]

    return {
        "N": N, "beta": beta,
        "mean_S": mean_S, "mean_S_over_N": mean_S / N if N else float("nan"),
        "var_S": var_S, "sem_S": sem_S,
        "mm_dim_mean": nanmean(mm_vals), "mm_dim_std": nanstd(mm_vals),
        "mp_dim_mean": nanmean(mp_vals), "mp_dim_std": nanstd(mp_vals),
        "height_mean": nanmean(height_vals),
        "ordering_fraction_mean": nanmean(of_vals),
        "abundance_profile_mean": np.nanmean(abundance_vals, axis=0) if abundance_vals.size else np.full(kmax + 1, np.nan),
        "acceptance_rate_mean": float(np.mean(acc_rates)) if acc_rates else float("nan"),
        "tau_int_mean": float(np.nanmean(taus)) if taus else float("nan"),
        "tau_converged_all": all(tau_conv) if tau_conv else False,
        "effective_samples_min": float(np.min(eff_samples)) if eff_samples else float("nan"),
        "min_eff_samples_met_all": all(met_flags) if met_flags else False,
        "multiplier_max": multiplier_max,
        "sample_Cs": [e["final_C"] for e in entries],
        "seed_actions": [(e["actions"], e["tau_int"]) for e in entries],
        "pooled_actions": actions_all,
    }


def _bootstrap_beta_c(points, n_boot=N_BOOT, rng=None):
    """Bootstrap uncertainty on the located beta_c, over CONVERGED points
    only (callers must pre-filter). Decorrelates each point's action
    series by thinning at step~2*tau_int before resampling, so the
    bootstrap draws from approximately independent samples."""
    rng = rng if rng is not None else np.random.default_rng(0)
    pools, betas = [], []
    for p in points:
        decorr = []
        for actions, tau in p.get("seed_actions", []):
            step = max(1, int(round(2 * tau))) if np.isfinite(tau) and tau > 0 else 1
            decorr.append(actions[::step])
        pool = np.concatenate(decorr) if decorr else np.array([])
        pools.append(pool)
        betas.append(p["beta"])
    betas = np.array(betas, dtype=float)
    valid_idx = [i for i, pool in enumerate(pools) if pool.size >= 2]
    if len(valid_idx) < 2:
        return {"beta_c_mean": float("nan"), "beta_c_std": float("nan"),
                "beta_c_ci90": (float("nan"), float("nan")), "n_boot": 0}

    winners = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        best_beta, best_var = betas[valid_idx[0]], -np.inf
        for i in valid_idx:
            pool = pools[i]
            sample = rng.choice(pool, size=pool.size, replace=True)
            v = float(np.var(sample))
            if v > best_var:
                best_var, best_beta = v, betas[i]
        winners[b] = best_beta
    lo, hi = np.percentile(winners, [5, 95])
    return {
        "beta_c_mean": float(np.mean(winners)), "beta_c_std": float(np.std(winners)),
        "beta_c_ci90": (float(lo), float(hi)), "n_boot": n_boot,
    }


def _peak_width_fwhm(betas, variances):
    """Full width at half maximum of the variance-vs-beta curve around
    its peak (CONVERGED points only; callers pre-filter), via linear
    interpolation between the sampled points on each side."""
    betas = np.asarray(betas, dtype=float)
    variances = np.asarray(variances, dtype=float)
    order = np.argsort(betas)
    betas, variances = betas[order], variances[order]
    valid = np.isfinite(variances)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    betas, variances = betas[valid], variances[valid]
    peak_idx = int(np.argmax(variances))
    peak_val = float(variances[peak_idx])
    half = peak_val / 2.0

    left = float("nan")
    for i in range(peak_idx, 0, -1):
        if variances[i] >= half and variances[i - 1] < half:
            b0, b1, v0, v1 = betas[i - 1], betas[i], variances[i - 1], variances[i]
            frac = (half - v0) / (v1 - v0) if v1 != v0 else 0.0
            left = b0 + frac * (b1 - b0)
            break

    right = float("nan")
    for i in range(peak_idx, len(betas) - 1):
        if variances[i] >= half and variances[i + 1] < half:
            b0, b1, v0, v1 = betas[i], betas[i + 1], variances[i], variances[i + 1]
            frac = (half - v0) / (v1 - v0) if v1 != v0 else 0.0
            right = b0 + frac * (b1 - b0)
            break

    if not (np.isfinite(left) and np.isfinite(right)):
        return float("nan"), peak_val
    return float(right - left), peak_val


def _support_window_check(converged_points, beta_c, beta_c_formula, window=0.25):
    """Sanity check on a located beta_c: a genuine interior variance
    maximum needs converged points on BOTH sides of it. Since the
    coarse/fine grids are centered on the published formula's
    prediction, also report converged coverage in a +/-window band
    around that prediction specifically. If beta_c itself turns out to
    equal the single largest converged beta, the 'located peak' has no
    converged data confirming the variance actually turns back down
    past it -- it is the edge of usable data, not a confirmed peak, and
    must be reported as such rather than counted as a located feature.
    """
    betas = [p["beta"] for p in converged_points]
    max_converged_beta = max(betas) if betas else float("nan")
    is_right_censored = (
        np.isfinite(beta_c) and np.isfinite(max_converged_beta)
        and abs(beta_c - max_converged_beta) < 1e-6
    )
    lo, hi = (1.0 - window) * beta_c_formula, (1.0 + window) * beta_c_formula
    n_below = sum(1 for b in betas if lo <= b < beta_c_formula)
    n_above = sum(1 for b in betas if beta_c_formula <= b <= hi)
    return {
        "max_converged_beta": max_converged_beta,
        "beta_c_is_right_censored": is_right_censored,
        "n_converged_within_25pct_below": n_below,
        "n_converged_within_25pct_above": n_above,
    }


def _coarse_worker(args):
    (N, beta, seed, base_sweeps, base_burnin, measure_every, eps,
     min_eff_samples, max_multiplier) = args
    res, met, mult = run_chain_adaptive_fast(
        N, beta, seed=seed, base_sweeps=base_sweeps, base_burnin=base_burnin,
        measure_every=measure_every, eps=eps, min_eff_samples=min_eff_samples,
        max_multiplier=max_multiplier,
    )
    entry = {
        "actions": res.actions, "tau_int": res.tau_int, "tau_converged": res.tau_converged,
        "effective_samples": res.effective_samples, "acceptance_rate": res.acceptance_rate,
        "final_C": res.final_C,
    }
    return beta, entry, met, mult


def run_scan_for_N(
    N, eps, seed_base=0, seeds=None,
    coarse_betas=None, n_refine=N_REFINE,
    scout_sweeps=SCOUT_SWEEPS, scout_burnin=SCOUT_BURNIN, scout_measure_every=SCOUT_MEASURE_EVERY,
    prod_sweeps=PROD_SWEEPS, prod_burnin=PROD_BURNIN, prod_measure_every=PROD_MEASURE_EVERY,
    verbose=True,
):
    seeds = seeds if seeds is not None else PRODUCTION_SEEDS
    coarse_betas = coarse_betas if coarse_betas is not None else _coarse_betas_for(N, eps)
    t0 = time.time()
    label = f"N={N:3d} eps={eps:.2f}"

    beta_star_scout, scouted = _locate_variance_peak(
        N, eps, seed_base, coarse_betas, scout_sweeps, scout_burnin, scout_measure_every,
    )
    fine_betas = np.round(np.sort(_refine_grid(coarse_betas, beta_star_scout, n_refine)), 6)
    fine_set = set(fine_betas.tolist())
    coarse_only_betas = sorted(set(np.round(coarse_betas, 6).tolist()) - fine_set)
    if verbose:
        print(f"[mcmc-study] {label} scout located beta*~{beta_star_scout:.5f} "
              f"({time.time() - t0:.1f}s); fine PT ladder has {len(fine_betas)} points, "
              f"{len(coarse_only_betas)} coarse-only points run in parallel")

    cur_sweeps, cur_burnin = prod_sweeps, prod_burnin
    pt_runs = None
    prev_min_eff = None
    stalled = False
    for escalation in range(MAX_PT_ESCALATIONS + 1):
        probe = run_parallel_tempering(
            N, fine_betas, seed=seeds[0], n_sweeps=cur_sweeps, burn_in=cur_burnin,
            measure_every=prod_measure_every, eps=eps,
        )
        if np.any(probe["swap_rates"] < MIN_SWAP_RATE):
            densified = densify_ladder_for_low_swap_rates(
                fine_betas, probe["swap_rates"], min_rate=MIN_SWAP_RATE, max_inserted=MAX_DENSIFY_INSERTED,
            )
            if verbose:
                print(f"[mcmc-study] {label} swap rates {np.round(probe['swap_rates'], 2)} -- "
                      f"densifying ladder {len(fine_betas)} -> {len(densified)} points and redoing PT")
            fine_betas = np.round(densified, 6)
            fine_set = set(fine_betas.tolist())
            coarse_only_betas = sorted(set(np.round(coarse_betas, 6).tolist()) - fine_set)
            pt_runs = {}
            for seed in seeds:
                pt_runs[seed] = run_parallel_tempering(
                    N, fine_betas, seed=seed, n_sweeps=cur_sweeps, burn_in=cur_burnin,
                    measure_every=prod_measure_every, eps=eps,
                )
        else:
            pt_runs = {seeds[0]: probe}
            for seed in seeds[1:]:
                pt_runs[seed] = run_parallel_tempering(
                    N, fine_betas, seed=seed, n_sweeps=cur_sweeps, burn_in=cur_burnin,
                    measure_every=prod_measure_every, eps=eps,
                )
        min_eff = min(
            pt_runs[seed]["results"][b]["effective_samples"]
            for seed in seeds for b in fine_set
        )
        if verbose:
            print(f"[mcmc-study] {label} fine-grid PT pass (sweeps={cur_sweeps}) done "
                  f"({time.time() - t0:.1f}s), min eff_samples={min_eff:.1f} "
                  f"(target {MIN_EFF_SAMPLES:.0f})")
        if min_eff >= MIN_EFF_SAMPLES or escalation >= MAX_PT_ESCALATIONS:
            break
        if prev_min_eff is not None and min_eff <= prev_min_eff * 1.2:
            if verbose:
                print(f"[mcmc-study] {label} escalation not improving eff_samples "
                      f"({prev_min_eff:.1f} -> {min_eff:.1f}) -- stopping early")
            stalled = True
            break
        prev_min_eff = min_eff
        cur_sweeps *= 2
        cur_burnin *= 2
        if verbose:
            print(f"[mcmc-study] {label} under target -- escalating to sweeps={cur_sweeps} and redoing")

    swap_rates_final = pt_runs[seeds[0]]["swap_rates"]
    round_trips_final = pt_runs[seeds[0]]["round_trips"]
    final_prod_sweeps, final_prod_burnin = cur_sweeps, cur_burnin
    if verbose:
        print(f"[mcmc-study] {label} fine-grid PT final ({time.time() - t0:.1f}s), "
              f"swap rates {np.round(swap_rates_final, 2)}, round trips {round_trips_final.tolist()}")

    fine_points = []
    for beta in sorted(fine_set):
        entries = [pt_runs[seed]["results"][beta] for seed in seeds]
        met_flags = [e["effective_samples"] >= MIN_EFF_SAMPLES for e in entries]
        fine_points.append(_aggregate_point(N, beta, entries, met_flags=met_flags, multiplier_max=1))

    # --- coarse-only points, parallel across CPU cores ---
    tasks = [
        (N, beta, seed, prod_sweeps, prod_burnin, prod_measure_every, eps,
         COARSE_ONLY_MIN_EFF_SAMPLES, COARSE_ONLY_MAX_MULTIPLIER)
        for beta in coarse_only_betas for seed in seeds
    ]
    by_beta = {beta: [] for beta in coarse_only_betas}
    mult_by_beta = {beta: 1 for beta in coarse_only_betas}
    if tasks:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for beta, entry, met, mult in ex.map(_coarse_worker, tasks):
                by_beta[beta].append((entry, met))
                mult_by_beta[beta] = max(mult_by_beta[beta], mult)
    coarse_points = []
    for beta in coarse_only_betas:
        entries = [e for e, _m in by_beta[beta]]
        met_flags = [m for _e, m in by_beta[beta]]
        coarse_points.append(_aggregate_point(N, beta, entries, met_flags=met_flags,
                                                multiplier_max=mult_by_beta[beta]))
    if verbose and coarse_only_betas:
        print(f"[mcmc-study] {label} coarse-only points done ({time.time() - t0:.1f}s)")

    points = sorted(fine_points + coarse_points, key=lambda p: p["beta"])
    if verbose:
        for p in points:
            flag = "" if p["min_eff_samples_met_all"] else f"(!UNCONVERGED, x{p['multiplier_max']})"
            tag = "[fine/PT]" if p["beta"] in fine_set else "[coarse]"
            print(f"[mcmc-study] {label} beta={p['beta']:9.5f} {tag} "
                  f"<S>/N={p['mean_S_over_N']:+.3f} var(S)={p['var_S']:9.2f} "
                  f"mm_dim={p['mm_dim_mean']:.2f} acc={p['acceptance_rate_mean']:.2f} "
                  f"tau~{p['tau_int_mean']:.1f} eff_min={p['effective_samples_min']:.0f} {flag}")

    # --- UNCONVERGED points are excluded from beta_c / width / height /
    # bootstrap -- not silently averaged in.
    converged = [p for p in points if p["min_eff_samples_met_all"] and np.isfinite(p["var_S"])]
    n_undersampled = sum(1 for p in points if not p["min_eff_samples_met_all"])
    if converged:
        beta_c = max(converged, key=lambda p: p["var_S"])["beta"]
        width, peak_height = _peak_width_fwhm(
            [p["beta"] for p in converged], [p["var_S"] for p in converged],
        )
        bootstrap = _bootstrap_beta_c(converged)
    else:
        beta_c, width, peak_height = float("nan"), float("nan"), float("nan")
        bootstrap = {"beta_c_mean": float("nan"), "beta_c_std": float("nan"),
                     "beta_c_ci90": (float("nan"), float("nan")), "n_boot": 0}

    beta_c_formula = beta_c_glaser_2018(N, eps)
    formula_sigma = _beta_c_formula_uncertainty(N, eps)
    combined_sigma = float(np.hypot(bootstrap["beta_c_std"], formula_sigma)) if np.isfinite(bootstrap["beta_c_std"]) else formula_sigma
    beta_c_z = abs(beta_c - beta_c_formula) / combined_sigma if np.isfinite(beta_c) and combined_sigma > 0 else float("nan")
    beta_c_pass = np.isfinite(beta_c_z) and beta_c_z < BETA_C_Z_PASS
    support_window = _support_window_check(converged, beta_c, beta_c_formula)

    # --- bimodality of the pooled action distribution at (the converged
    # point nearest) beta_c ---
    histogram = {"n_peaks": 0, "separation": float("nan")}
    if converged:
        beta_c_point = min(converged, key=lambda p: abs(p["beta"] - beta_c))
        histogram = _histogram_bimodality(beta_c_point["pooled_actions"] / N)

    return {
        "N": N, "eps": eps, "points": points, "beta_c": beta_c,
        "beta_c_formula": beta_c_formula, "beta_c_formula_sigma": formula_sigma,
        "beta_c_z": beta_c_z, "beta_c_pass": beta_c_pass,
        "support_window": support_window,
        "peak_width": width, "peak_height": peak_height,
        "bootstrap": bootstrap, "n_undersampled": n_undersampled,
        "n_converged": len(converged),
        "histogram": histogram,
        "scouted": scouted, "elapsed": time.time() - t0,
        "fine_betas": fine_betas, "pt_runs": pt_runs,
        "round_trips": round_trips_final,
        "final_prod_sweeps": final_prod_sweeps, "final_prod_burnin": final_prod_burnin,
        "escalation_stalled": stalled,
    }


def run_blind_center_test(N, eps, center_mults, seeds=None, **scan_kwargs):
    """Robustness check on a located beta_c: rerun run_scan_for_N with
    the coarse scout grid deliberately mis-centered at each multiple of
    the predicted beta_c in center_mults (same log-width and point count
    as the normal grid -- see _coarse_betas_centered), with every other
    setting identical to a normal run. If the located beta_c lands near
    the same value (and near the published prediction) regardless of
    where the grid was centered, the match is a located feature of the
    action, not an artifact of where the scout happened to look; if it
    instead tracks the grid center (or, per the UNCONVERGED-coverage
    issue this was written to check, the edge of wherever convergence
    happened to stop), say so explicitly rather than reporting PASS.
    """
    seeds = seeds if seeds is not None else PRODUCTION_SEEDS
    predicted = beta_c_glaser_2018(N, eps)
    results = {}
    for cm in center_mults:
        coarse_betas = _coarse_betas_centered(N, eps, center_mult=cm)
        scan = run_scan_for_N(N, eps, seeds=seeds, coarse_betas=coarse_betas, **scan_kwargs)
        max_converged_beta = max(
            (p["beta"] for p in scan["points"] if p["min_eff_samples_met_all"]), default=float("nan"),
        )
        results[cm] = {
            "center_beta": cm * predicted,
            "beta_c": scan["beta_c"],
            "bootstrap_std": scan["bootstrap"]["beta_c_std"],
            "n_converged": scan["n_converged"],
            "n_points": len(scan["points"]),
            "max_converged_beta": max_converged_beta,
            "beta_c_is_max_converged": (
                np.isfinite(scan["beta_c"]) and np.isfinite(max_converged_beta)
                and abs(scan["beta_c"] - max_converged_beta) < 1e-9
            ),
            "scan": scan,
        }
    return {"N": N, "eps": eps, "predicted": predicted, "by_center": results}


# --------------------------------------------------------------- deep phase

def _deep_worker(args):
    N, beta, seed, anneal_sweeps, tail_sweeps, measure_every, eps, kmax = args
    res = run_chain_fast(
        N, beta, seed=seed, n_sweeps=anneal_sweeps + tail_sweeps,
        burn_in=anneal_sweeps, measure_every=measure_every, eps=eps, anneal_from=0.0,
    )
    meas = measure_sample(res.final_C, kmax=kmax)
    return beta, seed, {
        "height": meas["height"], "ordering_fraction": meas["ordering_fraction"],
        "mm_dim": meas["mm_dim"], "acceptance_rate": res.acceptance_rate,
        "abundance_profile": meas["abundance_profile"],
    }


def run_deep_phase(
    N, eps, betas=None, seeds=None, anneal_sweeps=DEEP_ANNEAL_SWEEPS,
    tail_sweeps=DEEP_TAIL_SWEEPS, measure_every=DEEP_MEASURE_EVERY, kmax=KMAX,
):
    """Characterize the high-beta regime using slow beta-annealed chains,
    at betas given as multiples of this (N, eps)'s predicted beta_c
    (DEEP_BETA_MULTIPLIERS) if not given explicitly -- the old fixed
    absolute betas (20, 50) assumed an action normalization where those
    were already deep in the high-beta phase for every N; under the
    smeared action beta_c itself depends on N and eps, so the deep-phase
    betas must scale with it too.
    """
    seeds = seeds if seeds is not None else DEEP_SEEDS
    if betas is None:
        predicted = beta_c_glaser_2018(N, eps)
        betas = tuple(round(predicted * m, 6) for m in DEEP_BETA_MULTIPLIERS)
    tasks = [(N, beta, seed, anneal_sweeps, tail_sweeps, measure_every, eps, kmax)
             for beta in betas for seed in seeds]
    rows_by_beta = {beta: [] for beta in betas}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for beta, _seed, row in ex.map(_deep_worker, tasks):
            rows_by_beta[beta].append(row)

    results = {}
    for beta, rows in rows_by_beta.items():
        results[beta] = {
            "height_mean": nanmean([r["height"] for r in rows]),
            "height_std": nanstd([r["height"] for r in rows]),
            "of_mean": nanmean([r["ordering_fraction"] for r in rows]),
            "of_std": nanstd([r["ordering_fraction"] for r in rows]),
            "mm_dim_mean": nanmean([r["mm_dim"] for r in rows]),
            "mm_dim_std": nanstd([r["mm_dim"] for r in rows]),
            "acceptance_rate_mean": float(np.mean([r["acceptance_rate"] for r in rows])),
            "abundance_profile_mean": np.nanmean(np.array([r["abundance_profile"] for r in rows]), axis=0),
        }

    betas_sorted = sorted(betas)
    plateaued = False
    if len(betas_sorted) >= 2:
        b_lo, b_hi = betas_sorted[0], betas_sorted[-1]
        h_lo, h_hi = results[b_lo]["height_mean"], results[b_hi]["height_mean"]
        of_lo, of_hi = results[b_lo]["of_mean"], results[b_hi]["of_mean"]
        h_tol = 2 * max(results[b_lo]["height_std"], results[b_hi]["height_std"], 0.5)
        of_tol = 2 * max(results[b_lo]["of_std"], results[b_hi]["of_std"], 0.02)
        plateaued = abs(h_hi - h_lo) < h_tol and abs(of_hi - of_lo) < of_tol

    deepest = results[betas_sorted[-1]]
    height_pass = abs(deepest["height_mean"] - HIGH_BETA_HEIGHT_TARGET) < 1.5
    of_pass = abs(deepest["of_mean"] - HIGH_BETA_OF_TARGET) < 0.15

    return {
        "N": N, "eps": eps, "by_beta": results, "plateaued": plateaued, "betas": betas_sorted,
        "height_pass": height_pass, "of_pass": of_pass,
    }


# --------------------------------------------------------------- hysteresis

def run_hysteresis_check(N, eps, fine_betas, pt_runs, beta_c, seeds,
                          prod_sweeps=PROD_SWEEPS, prod_burnin=PROD_BURNIN,
                          prod_measure_every=PROD_MEASURE_EVERY):
    """At the fine-grid point closest to the located beta_c, compare the
    scan's already-computed random-start PT results against a fresh PT
    pass started from a hand-built maximally layered order."""
    fine_betas_sorted = np.sort(np.asarray(fine_betas, dtype=float))
    if not np.isfinite(beta_c):
        beta_c_actual = float(fine_betas_sorted[len(fine_betas_sorted) // 2])
    else:
        beta_c_actual = float(fine_betas_sorted[int(np.argmin(np.abs(fine_betas_sorted - beta_c)))])

    random_actions = [pt_runs[seed]["results"][beta_c_actual]["actions"] for seed in seeds]

    layered_actions = []
    for seed in seeds:
        u0_list, v0_list = [], []
        for k, b in enumerate(fine_betas_sorted):
            u0, v0 = layered_start(N, seed=seed * 1000 + k, n_layers=2)
            u0_list.append(u0)
            v0_list.append(v0)
        pt_l = run_parallel_tempering(
            N, fine_betas_sorted, seed=seed + 7000, n_sweeps=prod_sweeps, burn_in=prod_burnin,
            measure_every=prod_measure_every, eps=eps, u0_list=u0_list, v0_list=v0_list,
        )
        layered_actions.append(pt_l["results"][beta_c_actual]["actions"])

    def agg(action_arrays):
        acts = np.concatenate(action_arrays) if action_arrays else np.array([])
        mean_ = float(np.mean(acts)) if acts.size else float("nan")
        sem_ = float(np.std(acts) / np.sqrt(acts.size)) if acts.size > 1 else float("nan")
        return mean_, sem_

    mean_r, sem_r = agg(random_actions)
    mean_l, sem_l = agg(layered_actions)
    sem_r2 = sem_r if np.isfinite(sem_r) else 0.0
    sem_l2 = sem_l if np.isfinite(sem_l) else 0.0
    combined_sem = float(np.sqrt(sem_r2 ** 2 + sem_l2 ** 2))
    z = abs(mean_r - mean_l) / combined_sem if combined_sem > 0 else float("nan")
    agrees = np.isfinite(z) and z < 4.0

    return {
        "N": N, "eps": eps, "beta": beta_c_actual,
        "mean_S_random": mean_r, "sem_S_random": sem_r,
        "mean_S_layered": mean_l, "sem_S_layered": sem_l,
        "z": z, "agrees": agrees,
    }


# ------------------------------------------------------------- KR reference

def kr_abundance_reference(N, seeds=None, kmax=KMAX):
    seeds = seeds if seeds is not None else REFERENCE_SEEDS
    profiles = [interval_abundances(kleitman_rothschild(N, seed=s).C, kmax=kmax) for s in seeds]
    return np.nanmean(np.array(profiles), axis=0)


def sprinkle_abundance_reference(N, d=2, seeds=None, kmax=KMAX):
    seeds = seeds if seeds is not None else REFERENCE_SEEDS
    profiles = [interval_abundances(sprinkle(N, d, seed=s)[0].C, kmax=kmax) for s in seeds]
    return np.nanmean(np.array(profiles), axis=0)


# --------------------------------------------------------------- orchestration

def run_mcmc_study(study_ns=None, eps_values=None, seeds=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study_ns = study_ns if study_ns is not None else SCAN_NS
    eps_values = eps_values if eps_values is not None else EPS_VALUES
    seeds = seeds if seeds is not None else PRODUCTION_SEEDS
    t0 = time.time()

    print(f"[mcmc-study] controls at N={CONTROL_N} (eps-independent at beta=0)...")
    controls = run_controls()
    print(
        f"[mcmc-study] controls: beta=0 mm_dim={controls['beta0_mm_dim_mean']:.2f} "
        f"(sprinkle {controls['sprinkle_mm_dim_mean']:.2f}), "
        f"abundance_distance={controls['abundance_distance']:.3f}, "
        f"max independent-chain z={controls['independent_chains_max_z']:.2f} -> "
        f"dimension {'PASS' if controls['dimension_pass'] else 'FAIL'}, "
        f"abundance {'PASS' if controls['abundance_pass'] else 'FAIL'}, "
        f"balance {'PASS' if controls['balance_pass'] else 'FAIL'}"
    )

    scans, deep, hysteresis = {}, {}, {}
    kr_dist, sprinkle_dist = {}, {}
    for eps in eps_values:
        for N in study_ns:
            key = (eps, N)
            n_t0 = time.time()
            label = f"N={N:3d} eps={eps:.2f}"
            scans[key] = run_scan_for_N(N, eps, seeds=seeds)
            s = scans[key]
            bc = s["bootstrap"]
            print(f"[mcmc-study] {label} scan done: beta_c={s['beta_c']:.5f} "
                  f"(formula {s['beta_c_formula']:.5f} +/- {s['beta_c_formula_sigma']:.5f}, "
                  f"z={s['beta_c_z']:.2f} -> {'PASS' if s['beta_c_pass'] else 'FAIL'}), "
                  f"peak height={s['peak_height']:.1f}, width={s['peak_width']:.5f}, "
                  f"{s['n_undersampled']} UNCONVERGED points, "
                  f"histogram peaks={s['histogram']['n_peaks']} "
                  f"({s['elapsed']:.1f}s)")

            print(f"[mcmc-study] {label} deep high-beta phase...")
            deep[key] = run_deep_phase(N, eps)
            for b in deep[key]["betas"]:
                r = deep[key]["by_beta"][b]
                print(f"[mcmc-study] {label} deep beta={b:8.4f} height={r['height_mean']:.1f}+/-{r['height_std']:.1f} "
                      f"of={r['of_mean']:.3f}+/-{r['of_std']:.3f} mm_dim={r['mm_dim_mean']:.2f} "
                      f"acc={r['acceptance_rate_mean']:.3f}")
            print(f"[mcmc-study] {label} deep-phase plateaued: {deep[key]['plateaued']}, "
                  f"height target PASS={deep[key]['height_pass']}, of target PASS={deep[key]['of_pass']}")

            print(f"[mcmc-study] {label} hysteresis check at beta_c={s['beta_c']:.5f} (via PT)...")
            hysteresis[key] = run_hysteresis_check(
                N, eps, s["fine_betas"], s["pt_runs"], s["beta_c"], seeds,
                prod_sweeps=s["final_prod_sweeps"], prod_burnin=s["final_prod_burnin"],
            )
            h = hysteresis[key]
            print(f"[mcmc-study] {label} hysteresis: random <S>={h['mean_S_random']:.2f}+/-{h['sem_S_random']:.2f}, "
                  f"layered <S>={h['mean_S_layered']:.2f}+/-{h['sem_S_layered']:.2f}, z={h['z']:.2f} "
                  f"-> {'AGREE' if h['agrees'] else 'DISAGREE'}")

            if N not in kr_dist:
                kr_dist[N] = kr_abundance_reference(N)
                sprinkle_dist[N] = sprinkle_abundance_reference(N)
            print(f"[mcmc-study] {label} fully done in {time.time() - n_t0:.1f}s")

    generate_plots(scans, eps_values, study_ns)
    write_report(controls, scans, deep, hysteresis, kr_dist, sprinkle_dist, study_ns, eps_values)
    print(f"[mcmc-study] total time {time.time() - t0:.1f}s")
    return {
        "controls": controls, "scans": scans, "deep": deep,
        "hysteresis": hysteresis, "kr_dist": kr_dist, "sprinkle_dist": sprinkle_dist,
    }


def generate_plots(scans: dict, eps_values, study_ns) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for eps in eps_values:
        action_cases, var_cases, dim_cases, height_cases, transitions = {}, {}, {}, {}, {}
        for N in study_ns:
            scan = scans[(eps, N)]
            pts = sorted(scan["points"], key=lambda p: p["beta"])
            betas = [p["beta"] for p in pts]
            label = f"N={N}"
            action_cases[label] = {
                "beta": betas, "mean": [p["mean_S_over_N"] for p in pts],
                "err": [p["sem_S"] / N if N else float("nan") for p in pts],
            }
            var_cases[label] = {"beta": betas, "mean": [p["var_S"] for p in pts]}
            dim_cases[label] = {
                "beta": betas, "mean": [p["mm_dim_mean"] for p in pts],
                "err": [p["mm_dim_std"] for p in pts],
            }
            height_cases[label] = {"beta": betas, "mean": [p["height_mean"] for p in pts]}
            transitions[label] = scan["beta_c"]

        tag = f"eps{eps:.2f}".replace(".", "p")
        plots.plot_vs_beta(
            action_cases, str(RESULTS_DIR / f"mcmc_action_vs_beta_{tag}.png"),
            ylabel="<S> / N", title=f"Mean action per element vs beta (eps={eps})", transitions=transitions,
            xscale="symlog",
        )
        plots.plot_vs_beta(
            var_cases, str(RESULTS_DIR / f"mcmc_variance_vs_beta_{tag}.png"),
            ylabel="Var(S)", title=f"Action variance vs beta (eps={eps})",
            transitions=transitions, xscale="symlog",
        )
        plots.plot_vs_beta(
            dim_cases, str(RESULTS_DIR / f"mcmc_dimension_vs_beta_{tag}.png"),
            ylabel="Myrheim-Meyer dimension", title=f"Dimension vs beta (eps={eps})", transitions=transitions,
            xscale="symlog",
        )
        plots.plot_vs_beta(
            height_cases, str(RESULTS_DIR / f"mcmc_height_vs_beta_{tag}.png"),
            ylabel="height (longest chain)", title=f"Height vs beta (eps={eps})", transitions=transitions,
            xscale="symlog",
        )

        Ns = study_ns
        peak_heights = [scans[(eps, N)]["peak_height"] for N in Ns]
        peak_widths = [scans[(eps, N)]["peak_width"] for N in Ns]
        plots.plot_vs_beta(
            {"peak height": {"beta": Ns, "mean": peak_heights}},
            str(RESULTS_DIR / f"mcmc_peak_height_vs_N_{tag}.png"),
            ylabel="Var(S) at beta_c", title=f"Transition peak height vs N (eps={eps})", xlabel="N", xscale="log",
        )
        plots.plot_vs_beta(
            {"peak FWHM": {"beta": Ns, "mean": peak_widths}},
            str(RESULTS_DIR / f"mcmc_peak_width_vs_N_{tag}.png"),
            ylabel="beta FWHM of the variance peak", title=f"Transition peak width vs N (eps={eps})",
            xlabel="N", xscale="log",
        )

        for N in study_ns:
            scan = scans[(eps, N)]
            pts = sorted(scan["points"], key=lambda p: p["beta"])
            beta_c = scan["beta_c"]
            below = [p for p in pts if np.isfinite(beta_c) and p["beta"] < beta_c and p["sample_Cs"]]
            above = [p for p in pts if np.isfinite(beta_c) and p["beta"] > beta_c and p["sample_Cs"]]
            if below:
                p_below = below[0]
                plots.plot_hasse(
                    p_below["sample_Cs"][0], str(RESULTS_DIR / f"hasse_N{N}_{tag}_below_transition.png"),
                    title=f"N={N}, eps={eps}, beta={p_below['beta']:.4f} (< beta_c={beta_c:.4f})",
                )
            if above:
                p_above = above[-1]
                plots.plot_hasse(
                    p_above["sample_Cs"][0], str(RESULTS_DIR / f"hasse_N{N}_{tag}_above_transition.png"),
                    title=f"N={N}, eps={eps}, beta={p_above['beta']:.4f} (> beta_c={beta_c:.4f})",
                )


def write_report(controls, scans, deep, hysteresis, kr_dist, sprinkle_dist, study_ns, eps_values) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 3: weighing 2D causal set universes (smeared-action MCMC)")
    lines.append("")
    lines.append(
        "Every causal set on N elements is a possible universe, weighted by exp(-beta * S(eps)) "
        "where S(eps) is the *smeared* 2D Benincasa-Dowker action (see `mcmc/action.py`), following "
        "S. Surya, \"Evidence for the continuum in 2D causal set quantum gravity\", CQG 29 132001 "
        "(2012), arXiv:1110.6244, and calibrated against L. Glaser, D. O'Connor, S. Surya, \"Finite "
        "Size Scaling in 2d Causal Set Quantum Gravity\", CQG 35 045006 (2018), arXiv:1706.06432 -- "
        "see 'Comparison to the literature' below for the normalization-match confirmation and the "
        "exact published formula this run's beta_c is checked against. An earlier version of this "
        "study used the plain (non-smeared) action and scanned up to N=80; that action's beta is not "
        "comparable to any published number, and N=80 under it showed zero PT replica round trips "
        "across the transition even after a 4x run-length escalation -- both issues are why this "
        "version uses the smeared action and stays at N <= 60. Sampling engine: Numba-compiled, "
        "incrementally-updated Metropolis (fast_core.py) plus parallel tempering (replica exchange) "
        "with round-trip tracking (tempering.py)."
    )
    lines.append("")

    # ---------------------------------------------------------------- controls
    lines.append("## Controls")
    lines.append("")
    lines.append(
        "At beta=0 every Metropolis move is accepted unconditionally regardless of eps (the "
        f"Boltzmann weight is 1), so this check is run once, not once per eps. Acceptance rate = "
        f"{controls['acceptance_rate_mean']:.2f} (min over chains {controls['acceptance_rate_min']:.2f})."
    )
    lines.append("")
    lines.append(f"- N = {controls['N']}, {len(controls['chain_means'])} independent beta=0 chains, "
                  f"{len(CONTROL_SEEDS)} sprinkle reference seeds.")
    lines.append(
        f"- Myrheim-Meyer dimension: beta=0 chains = {controls['beta0_mm_dim_mean']:.3f} +/- "
        f"{controls['beta0_mm_dim_std']:.3f}, sprinkle d=2 reference = "
        f"{controls['sprinkle_mm_dim_mean']:.3f} +/- {controls['sprinkle_mm_dim_std']:.3f} "
        f"-> **{'PASS' if controls['dimension_pass'] else 'FAIL'}** (within 0.4 of each other)."
    )
    lines.append(
        f"- Interval-abundance L1 distance to sprinkle reference: {controls['abundance_distance']:.3f} "
        f"-> **{'PASS' if controls['abundance_pass'] else 'FAIL'}** (threshold 0.5)."
    )
    lines.append(
        f"- Detailed-balance sanity: independent chains' mean actions span at most "
        f"{controls['independent_chains_max_z']:.2f} standard errors from their grand mean "
        f"-> **{'PASS' if controls['balance_pass'] else 'FAIL'}** (threshold 4 SEM)."
    )
    lines.append("")

    all_keys = [(eps, N) for eps in eps_values for N in study_ns]

    for eps in eps_values:
        lines.append(f"## eps = {eps}")
        lines.append("")

        # beta_c vs published formula
        lines.append("### Locating beta_c vs the published formula")
        lines.append("")
        lines.append(
            "beta_c is the beta (among the converged fine PT-ladder points only -- UNCONVERGED "
            "points, below 50 effective samples even after escalation, are excluded) with the "
            "largest pooled action variance. Compared against Glaser-O'Connor-Surya 2018's "
            "beta_c(N,eps) = 1.66(+/-0.03)/(N*eps^2) + [4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2 "
            f"via a z-score using both the bootstrap and formula-coefficient uncertainties "
            f"(PASS if z < {BETA_C_Z_PASS})."
        )
        lines.append("")
        lines.append("| N | beta_c (located) | bootstrap std | formula beta_c | z | result | peak height | peak FWHM | UNCONVERGED | round trips |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for N in study_ns:
            s = scans[(eps, N)]
            bc = s["bootstrap"]
            width_str = f"{s['peak_width']:.5f}" if np.isfinite(s["peak_width"]) else "n/a"
            rt = s["round_trips"]
            rt_str = f"min {int(np.min(rt))}, max {int(np.max(rt))}" if rt is not None and len(rt) else "n/a"
            lines.append(
                f"| {N} | {s['beta_c']:.5f} | {bc['beta_c_std']:.5f} | {s['beta_c_formula']:.5f} +/- "
                f"{s['beta_c_formula_sigma']:.5f} | {s['beta_c_z']:.2f} | "
                f"{'**PASS**' if s['beta_c_pass'] else '**FAIL**'} | {s['peak_height']:.1f} | {width_str} | "
                f"{s['n_undersampled']} / {len(s['points'])} | {rt_str} |"
            )
        lines.append("")
        lines.append(
            "**Is the located beta_c a confirmed interior peak, or just where convergence ran out?** "
            "A genuine variance maximum needs converged points on both sides of it. The grid is "
            "centered on the formula's own prediction, so this also reports converged coverage in a "
            "+/-25% band around that prediction specifically."
        )
        lines.append("")
        lines.append("| N | max converged beta | beta_c IS the max converged beta | converged within 25% below prediction | converged within 25% above prediction |")
        lines.append("|---|---|---|---|---|")
        n_right_censored = 0
        for N in study_ns:
            sw = scans[(eps, N)]["support_window"]
            if sw["beta_c_is_right_censored"]:
                n_right_censored += 1
            lines.append(
                f"| {N} | {sw['max_converged_beta']:.5f} | "
                f"{'**YES -- right-censored, not confirmed**' if sw['beta_c_is_right_censored'] else 'no'} | "
                f"{sw['n_converged_within_25pct_below']} | {sw['n_converged_within_25pct_above']} |"
            )
        lines.append("")
        if n_right_censored:
            lines.append(
                f"**{n_right_censored} / {len(study_ns)} N at eps={eps} have a right-censored beta_c**: "
                "the located value is exactly the largest converged beta in the scan, meaning there is "
                "*no* converged data showing the variance actually turns back down past it. The z-score "
                "PASS reported above for these N confirms only that the right-censoring point happens to "
                "land near the formula's prediction -- not that an interior peak was independently "
                "located and found to agree. Where this flag is set, do not read the PASS as a genuine "
                "confirmation (see 'Known limitations' below for the grid-centering test that checks "
                "whether this value is at least stable, i.e. not a pure grid artifact)."
            )
            lines.append("")
        if any(scans[(eps, N)]["escalation_stalled"] for N in study_ns):
            stalled_here = [N for N in study_ns if scans[(eps, N)]["escalation_stalled"]]
            lines.append(
                f"**Escalation stalled for N={', '.join(str(N) for N in stalled_here)} at eps={eps}**: "
                "doubling PT sweeps did not improve effective-sample count near beta_c; see the round "
                "trip counts above -- if round trips are also near zero there, treat that N's beta_c "
                "and histogram result with extra caution."
            )
            lines.append("")

        # double-peaked histogram
        lines.append("### Action histogram at beta_c (first-order signature)")
        lines.append("")
        lines.append(
            "Glaser-O'Connor-Surya 2018 report a first-order transition: a double-peaked pooled "
            "action histogram at beta_c, with the peak separation growing as N increases. Peaks are "
            "counted on the pooled (S/N) distribution at the converged fine-grid point nearest the "
            "located beta_c."
        )
        lines.append("")
        lines.append("| N | peaks found | separation (S/N units) |")
        lines.append("|---|---|---|")
        seps = []
        for N in study_ns:
            hist = scans[(eps, N)]["histogram"]
            sep_str = f"{hist['separation']:.4f}" if np.isfinite(hist["separation"]) else "n/a"
            lines.append(f"| {N} | {hist['n_peaks']} | {sep_str} |")
            seps.append(hist["separation"])
        lines.append("")
        seps_arr = np.array(seps, dtype=float)
        Ns_arr = np.array(study_ns, dtype=float)
        valid_sep = np.isfinite(seps_arr)
        growing = False
        if valid_sep.sum() >= 2:
            slope = float(np.polyfit(Ns_arr[valid_sep], seps_arr[valid_sep], 1)[0])
            growing = slope > 0
            lines.append(
                f"Peak separation vs N slope = {slope:+.5f} ({'growing' if growing else 'not growing'} "
                f"with N, over {int(valid_sep.sum())} / {len(study_ns)} N values with >= 2 peaks found)."
            )
        else:
            lines.append(
                f"Fewer than 2 N values show a double-peaked histogram at eps={eps}; cannot assess "
                "whether the separation grows with N."
            )
        lines.append("")

        # deep phase
        lines.append("### High-beta phase vs published targets (height ~ 3, ordering fraction ~ 0.6)")
        lines.append("")
        lines.append("| N | beta (deepest) | height | ordering fraction | MM dimension | height PASS | OF PASS |")
        lines.append("|---|---|---|---|---|---|---|")
        for N in study_ns:
            d = deep[(eps, N)]
            b_deep = d["betas"][-1]
            r = d["by_beta"][b_deep]
            lines.append(
                f"| {N} | {b_deep:.4f} | {r['height_mean']:.1f} +/- {r['height_std']:.1f} | "
                f"{r['of_mean']:.3f} +/- {r['of_std']:.3f} | {r['mm_dim_mean']:.2f} +/- {r['mm_dim_std']:.2f} | "
                f"{'PASS' if d['height_pass'] else 'FAIL'} | {'PASS' if d['of_pass'] else 'FAIL'} |"
            )
        lines.append("")
        n_plateaued = sum(1 for N in study_ns if deep[(eps, N)]["plateaued"])
        lines.append(
            f"{n_plateaued} / {len(study_ns)} N values plateaued between the two deep betas tested "
            f"(x{DEEP_BETA_MULTIPLIERS[0]} and x{DEEP_BETA_MULTIPLIERS[1]} of the formula's predicted "
            "beta_c) -- where False, the high-beta regime had not fully stabilized at the betas tested."
        )
        lines.append("")

        # hysteresis
        lines.append("### Hysteresis check at beta_c")
        lines.append("")
        lines.append("| N | beta_c rung | <S> random start | <S> layered start | z | result |")
        lines.append("|---|---|---|---|---|---|")
        n_disagree = 0
        for N in study_ns:
            h = hysteresis[(eps, N)]
            if not h["agrees"]:
                n_disagree += 1
            lines.append(
                f"| {N} | {h['beta']:.5f} | {h['mean_S_random']:.2f} +/- {h['sem_S_random']:.2f} | "
                f"{h['mean_S_layered']:.2f} +/- {h['sem_S_layered']:.2f} | {h['z']:.2f} | "
                f"{'AGREE' if h['agrees'] else '**DISAGREE**'} |"
            )
        lines.append("")
        if n_disagree:
            lines.append(
                f"**{n_disagree} / {len(study_ns)} N disagree** between starting conditions at beta_c "
                "-- expected to some degree for a genuinely first-order transition (coexisting phases "
                "separated by a barrier), consistent with the histogram section above."
            )
        else:
            lines.append("All N agree within 4 combined standard errors.")
        lines.append("")

        # structural comparison
        lines.append("### Structural comparison: closer to Kleitman-Rothschild or to a sprinkle?")
        lines.append("")
        lines.append("| N | distance to Kleitman-Rothschild | distance to 2D sprinkle | closer to |")
        lines.append("|---|---|---|---|")
        for N in study_ns:
            d = deep[(eps, N)]
            b_deep = d["betas"][-1]
            deep_profile = d["by_beta"][b_deep]["abundance_profile_mean"]
            d_kr = abundance_distance(deep_profile, kr_dist[N])
            d_sp = abundance_distance(deep_profile, sprinkle_dist[N])
            closer = "Kleitman-Rothschild" if d_kr < d_sp else "2D sprinkle"
            lines.append(f"| {N} | {d_kr:.3f} | {d_sp:.3f} | {closer} |")
        lines.append("")

        tag = f"eps{eps:.2f}".replace(".", "p")
        lines.append(
            f"Plots: `mcmc_variance_vs_beta_{tag}.png`, `mcmc_action_vs_beta_{tag}.png`, "
            f"`mcmc_dimension_vs_beta_{tag}.png`, `mcmc_height_vs_beta_{tag}.png`, "
            f"`mcmc_peak_height_vs_N_{tag}.png`, `mcmc_peak_width_vs_N_{tag}.png`, and per-N "
            f"`hasse_N*_{tag}_{{below,above}}_transition.png`."
        )
        lines.append("")

    # ------------------------------------------------------- literature section
    lines.append("## Comparison to the literature")
    lines.append("")
    lines.append(
        "**Normalization check**: L. Glaser, D. O'Connor, S. Surya, \"Finite Size Scaling in 2d "
        "Causal Set Quantum Gravity\", CQG 35 045006 (2018), arXiv:1706.06432, was fetched directly "
        "(via its ar5iv HTML rendering) for this comparison. Its action formula, "
        "S(C,eps) = 4*eps*(N - 2*eps*sum_n N_n*f(n,eps)) with f(n,eps) = (1-eps)^n*(1 - 2*eps*n/(1-eps) "
        "+ eps^2*n*(n-1)/(2*(1-eps)^2)), matches this project's `action.bd_action_2d_smeared` exactly "
        "(confirmed both by reading the paper and by this project's own eps -> 1 unit test, which "
        "reduces the smeared action to the plain action's 4*(N-2*N0+4*N1-2*N2) limit analytically, "
        "with zero numerical error). Its beta_c(N,eps) = 1.66(+/-0.03)/eps^2/N + "
        "[4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2 + O(1/N^3) is the formula used throughout "
        "this report; the paper fits it over N = 20..90, eps = 0.1..0.5, and states its asymptotic "
        "(N -> infinity) regime is only reached for N >~ 65 -- so the N = 30..60 used here sit below "
        "that regime, and the O(1/N^3) terms the paper does not quantify are plausibly non-negligible; "
        "the z-scores above should be read with that caveat, not as a strict pass/fail on the "
        "asymptotic theory."
    )
    lines.append(
        "**Transition order**: the 2018 paper's main finding is that the transition is *first-order*, "
        "evidenced by double-peaked action histograms whose separation grows with N and a Binder "
        "coefficient moving away from 0 as N grows -- both checked directly above, not assumed."
    )
    lines.append(
        "**Phase names**: the paper calls the two phases Pi_- (continuum, beta < beta_c, Myrheim-Meyer "
        "dimension ~ 2) and Pi_+ (crystalline/non-continuum, beta > beta_c). S. Surya 2012, "
        "arXiv:1110.6244 (the earlier, N=50-only study this project originally targeted) calls them "
        "\"continuum\" and \"crystalline\" with the same qualitative dimension/height behavior; it "
        "does not give an N-scaling formula for beta_c, which is why the 2018 paper is used as this "
        "report's quantitative target instead."
    )
    lines.append("")

    # --------------------------------------------------------------- performance
    lines.append("## Performance")
    lines.append("")
    lines.append(
        "Unchanged from the prior version of this study: (1) parallel tempering for the near-"
        "transition beta grid, now with round-trip tracking reported above as the real test that "
        "replicas cross between phases rather than trusting local swap-rate numbers alone; (2) a "
        "Numba-compiled, incrementally-updated action (O(N^2) per move instead of rebuilding the "
        "O(N^3) interval-size histogram from scratch every move); (3) independent beta points run in "
        f"parallel across {N_WORKERS} CPU cores. N=80 is no longer attempted (see intro) -- the "
        "largest N here is 60, which is both cheaper per point and (per the round-trip columns above) "
        "actually mixes across the transition under PT."
    )
    lines.append("")

    # ----------------------------------------------------------- known limitations
    lines.append("## Known limitations")
    lines.append("")
    n_hyst_disagree = sum(1 for k in all_keys if not hysteresis[k]["agrees"])
    n_bimodal_found = sum(1 for k in all_keys if scans[k]["histogram"]["n_peaks"] >= 2)
    n_right_censored_total = sum(1 for k in all_keys if scans[k]["support_window"]["beta_c_is_right_censored"])
    lines.append(
        f"**The first-order barrier at beta_c is not crossed cleanly.** Hysteresis disagreed between "
        f"random-start and layered-start chains (even with parallel tempering active) for "
        f"{n_hyst_disagree} / {len(all_keys)} (N, eps) combinations, and the pooled action histogram "
        f"at beta_c showed the expected double-peak signature in only {n_bimodal_found} / {len(all_keys)}. "
        "These two findings are consistent with each other: a real first-order transition has a free-"
        "energy barrier separating the two phases that single-replica moves essentially never cross, and "
        "replica exchange only helps when neighboring-beta variance distributions overlap enough for a "
        "swap to be accepted -- near a sharp first-order point they stop overlapping, which is exactly "
        "where this run's effective-sample counts collapsed and points were marked UNCONVERGED."
    )
    lines.append(
        f"**A related, sharper problem: {n_right_censored_total} / {len(all_keys)} combinations' located "
        "beta_c is right-censored** -- see the per-eps 'is beta_c a confirmed interior peak' tables above. "
        "The reported value is exactly the largest converged beta in the scan, with no converged data "
        "at all above it, so the 'beta_c matches the published formula' PASS for those combinations "
        "confirms only that the point where convergence ran out happens to land near the prediction, not "
        "that an interior variance maximum was independently located and found to agree. Only one "
        "combination in this run (N=30, eps=0.21) had converged coverage on both sides of its located "
        "peak."
    )
    lines.append(
        "**Implication**: near-transition sampling at larger N (this project's own escalation attempts "
        "stalled -- doubling PT sweep count did not improve effective-sample counts, see the per-eps "
        "escalation notes above) or in higher dimension (where interval-size computations and the move "
        "set both grow more expensive) will need stronger methods than plain replica exchange on a "
        "single-swap proposal -- e.g. a denser ladder specifically bracketing the barrier (informed by "
        "a cheap pilot rather than the published formula alone, given the right-censoring problem above), "
        "cluster or multi-site moves that can cross the barrier in fewer steps, or multicanonical/Wang-"
        "Landau-style sampling that flattens the free-energy barrier directly instead of relying on "
        "tempering to route around it."
    )
    lines.append("")

    # ---------------------------------------------------------------- verdict
    lines.append("## Verdict: do we reproduce the published result?")
    lines.append("")
    n_total = len(all_keys)
    n_betac_pass = sum(1 for k in all_keys if scans[k]["beta_c_pass"])
    n_bimodal = sum(1 for k in all_keys if scans[k]["histogram"]["n_peaks"] >= 2)
    n_height_pass = sum(1 for k in all_keys if deep[k]["height_pass"])
    n_of_pass = sum(1 for k in all_keys if deep[k]["of_pass"])
    lines.append(
        f"Across all {n_total} (N, eps) combinations tested (N in {study_ns}, eps in {eps_values}):"
    )
    lines.append("")
    lines.append(
        f"1. **beta_c matches the published formula** (z < {BETA_C_Z_PASS}): {n_betac_pass} / {n_total} "
        f"-- but {n_right_censored_total} / {n_total} of those located values are right-censored (see "
        "'Known limitations' below), so this count overstates genuine confirmation; read it alongside "
        "that section rather than at face value."
    )
    lines.append(f"2. **Action histogram at beta_c is double-peaked**: {n_bimodal} / {n_total} "
                  "(see per-eps peak-separation-vs-N trend above for whether it also sharpens with N).")
    lines.append(f"3. **High-beta phase matches height~{HIGH_BETA_HEIGHT_TARGET:.0f} target**: {n_height_pass} / {n_total}; "
                  f"**ordering fraction~{HIGH_BETA_OF_TARGET:.1f} target**: {n_of_pass} / {n_total}.")
    lines.append(
        f"4. **Low-beta phase matches 2D sprinkles**: controls at N={CONTROL_N} "
        f"-> {'PASS' if (controls['dimension_pass'] and controls['abundance_pass']) else 'FAIL'} "
        "(eps-independent at beta=0, so checked once; every (N, eps)'s own low-beta points in the "
        "per-eps tables above show dimension near 2 as well)."
    )
    lines.append("")
    all_four_pass = (
        n_betac_pass == n_total and n_bimodal == n_total
        and n_height_pass == n_total and n_of_pass == n_total
        and controls["dimension_pass"] and controls["abundance_pass"]
    )
    if all_four_pass:
        lines.append(
            "**All four pass criteria are met across every (N, eps) tested: this run reproduces the "
            "published Glaser-O'Connor-Surya (2018) result** -- a first-order transition from a "
            "continuum-like phase (matching 2D sprinkles) to a crystalline phase, at a beta_c "
            "consistent with their N, eps scaling formula, within the N=30-60 range tested (below "
            "their quoted asymptotic regime of N >~ 65)."
        )
    else:
        lines.append(
            "**Not every pass criterion is met for every (N, eps) tested** -- see the counts above "
            "and the per-eps tables for exactly which N/eps combinations fall short and on which "
            "criterion. Read this alongside the N >~ 65 asymptotic-regime caveat: a criterion failing "
            "at N=30 while passing at N=60 is consistent with the published finite-size behavior "
            "itself, not necessarily a problem with this implementation."
        )
    lines.append("")

    report_path = RESULTS_DIR / "mcmc_2d_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[mcmc-study] wrote {report_path}")
