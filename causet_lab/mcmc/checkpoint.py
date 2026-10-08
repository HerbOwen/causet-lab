"""Checkpointing for the Wang-Landau / MUCA samplers (Phase 3b part 2).

A long WL or MUCA run needs to survive interruption: save_checkpoint /
load_checkpoint serialize everything needed to resume a run bit-for-bit
-- the configuration (u, v and the derived future/past/counts, though
the latter three are cheap to rebuild from (u, v) and are included only
so a resume doesn't need to pay that O(N^2) rebuild), the RNG state
(the single uint64 from rng.py -- see its module docstring for why this
has to be an explicit value rather than numpy/numba's opaque global
state), and the WL/MUCA weights and histograms (ln_g, H, the bin range,
and the current modification factor f or production-stage counters).
Plain pickle of a dict of numpy arrays/scalars -- there is no need for
a bespoke format here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def save_checkpoint(path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic on both POSIX and Windows -- never leaves a half-written checkpoint


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def make_wl_state(u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
                   n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
                   ever_visited=None, full_coverage_required=True) -> dict:
    return {
        "kind": "wl",
        "u": np.asarray(u), "v": np.asarray(v),
        "future": np.asarray(future), "past": np.asarray(past), "counts": np.asarray(counts),
        "rng_state": np.uint64(rng_state),
        "ln_g": np.asarray(ln_g), "H": np.asarray(H),
        "bin_lo": float(bin_lo), "bin_width": float(bin_width), "n_bins": int(n_bins),
        "f_mod": float(f_mod), "stage": int(stage),
        "bin_idx": int(bin_idx), "extreme_state": np.asarray(extreme_state),
        "half_trips": np.asarray(half_trips), "edge_hits_total": int(edge_hits_total),
        "ever_visited": np.asarray(ever_visited) if ever_visited is not None else (np.asarray(H) > 0),
        "full_coverage_required": bool(full_coverage_required),
    }


def make_muca_state(u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
                     n_bins, bin_idx, half_trips, moves_done,
                     rec_bins_list, rec_S_list, rec_height_list, rec_of_list,
                     rec_struct_bin_list=None) -> dict:
    return {
        "kind": "muca",
        "u": np.asarray(u), "v": np.asarray(v),
        "future": np.asarray(future), "past": np.asarray(past), "counts": np.asarray(counts),
        "rng_state": np.uint64(rng_state),
        "ln_g": np.asarray(ln_g), "H": np.asarray(H),
        "bin_lo": float(bin_lo), "bin_width": float(bin_width), "n_bins": int(n_bins),
        "bin_idx": int(bin_idx), "half_trips": np.asarray(half_trips), "moves_done": int(moves_done),
        "rec_bins": list(rec_bins_list), "rec_S": list(rec_S_list),
        "rec_height": list(rec_height_list), "rec_of": list(rec_of_list),
        "rec_struct_bin": list(rec_struct_bin_list) if rec_struct_bin_list is not None else [],
    }
