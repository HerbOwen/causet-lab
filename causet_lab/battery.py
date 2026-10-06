"""Shared measurement battery used by both the single-case CLI commands in
run.py and the multi-rule sweep in rules_study.py. Pulled out on its own
to avoid run.py <-> rules_study.py import cycles.
"""
from __future__ import annotations

import warnings

import numpy as np

from .measures import (
    ordering_fraction,
    myrheim_meyer_dimension,
    midpoint_dimension,
    sample_intervals,
    interval_abundances,
    height,
    link_valence,
    frontier_width,
)


def nanmean(values) -> float:
    """np.nanmean that returns nan (quietly) instead of warning on all-nan input."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))


def nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanstd(arr))


def nanmean_axis0(rows) -> np.ndarray:
    arr = np.asarray(rows, dtype=float)
    all_nan_cols = np.all(np.isnan(arr), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(arr, axis=0)
    out[all_nan_cols] = np.nan
    return out


def analyze_matrix(C: np.ndarray, seed: int, n_samples: int = 200, min_size: int = 10,
                    max_size: int = 300, kmax: int = 10) -> dict:
    """Run the full measurement battery on a causal set matrix.

    Computes ordering fraction, height, link valence, frontier width, and
    both dimension estimators applied directly to the whole set (only
    meaningful if it is itself an interval, i.e. for sprinkles) *and*
    applied to randomly sampled sub-intervals (meaningful for every case,
    including grown and junk orders, which are not themselves intervals).
    """
    out = {
        "ordering_fraction": ordering_fraction(C),
        "height": height(C),
        "valence": link_valence(C),
        "frontier_width": frontier_width(C),
        "mm_dim_whole": myrheim_meyer_dimension(C),
        "mp_dim_whole": midpoint_dimension(C),
    }
    intervals = sample_intervals(C, n_samples, min_size, max_size, seed)
    out["n_intervals"] = len(intervals)
    if intervals:
        mm = np.array([myrheim_meyer_dimension(iv) for iv in intervals])
        mp = np.array([midpoint_dimension(iv) for iv in intervals])
        profiles = np.array([interval_abundances(iv, kmax) for iv in intervals])
        out["mm_dim_sampled_mean"] = nanmean(mm)
        out["mm_dim_sampled_std"] = nanstd(mm)
        out["mp_dim_sampled_mean"] = nanmean(mp)
        out["mp_dim_sampled_std"] = nanstd(mp)
        out["abundance_profile"] = nanmean_axis0(profiles)
    else:
        out["mm_dim_sampled_mean"] = float("nan")
        out["mm_dim_sampled_std"] = float("nan")
        out["mp_dim_sampled_mean"] = float("nan")
        out["mp_dim_sampled_std"] = float("nan")
        out["abundance_profile"] = np.full(kmax + 1, np.nan)
    return out
