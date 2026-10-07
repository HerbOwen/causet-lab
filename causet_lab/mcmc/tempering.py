"""Parallel tempering (replica exchange) over a beta ladder.

One replica per beta value, each evolving under ordinary Metropolis at
its own fixed beta; every sweep, every pair of beta-adjacent replicas
attempts to swap their entire (u, v) configuration under the standard
replica-exchange acceptance rule. A replica stuck in a basin favored by
its own beta can escape by trading places with a neighbor instead of
having to find its own way out via single-swap moves alone -- this is
the standard fix for the slow mixing / metastability that plain
Metropolis shows near a sharp transition (see fast_core.run_pt_jit).
"""
from __future__ import annotations

import numpy as np

from .orders import orders_to_matrix
from .sampler import integrated_autocorr_time
from .action import f2_smear_table
from . import fast_core


def run_parallel_tempering(
    N: int,
    betas,
    seed: int,
    n_sweeps: int,
    burn_in: int,
    measure_every: int = 4,
    eps: float = 0.21,
    u0_list=None,
    v0_list=None,
):
    """Run PT with one replica per value in betas (need not be sorted on
    input; returned in ascending order), all sharing a single eps (PT
    tempers over beta at a fixed action normalization). If u0_list/
    v0_list are given (one (N,) array per beta, same order as betas),
    replicas start from those states instead of independent random
    permutations -- used to start every replica from a hand-built
    layered order for the hysteresis check.

    Returns a dict: betas (ascending), results (per-beta dict of
    actions/tau/effective_samples/acceptance_rate/final_C/final_u/
    final_v), swap_rates (K-1,) -- the accepted-swap fraction between
    each pair of beta-adjacent replicas, aim for roughly 20-40% -- and
    round_trips (K,): how many full bottom<->top ladder traversals each
    physical replica completed. This is the diagnostic that actually
    matters: healthy local swap rates can coexist with replicas that
    never travel the full range, in which case PT is not actually
    letting the system cross between phases no matter how good the
    neighbor-pair acceptance looks.
    """
    order = np.argsort(betas)
    betas_sorted = np.asarray(betas, dtype=np.float64)[order]
    K = len(betas_sorted)
    rng = np.random.default_rng(seed)
    f2_table = f2_smear_table(max(N - 2, 0), eps)

    U = np.empty((K, N), dtype=np.int64)
    V = np.empty((K, N), dtype=np.int64)
    Call = np.empty((K, N, N), dtype=np.bool_)
    Sizes = np.empty((K, N, N), dtype=np.int32)
    Counts = np.empty((K, max(N - 1, 1)), dtype=np.int64)
    for rank, orig_idx in enumerate(order):
        if u0_list is not None:
            u = np.array(u0_list[orig_idx], dtype=np.int64, copy=True)
            v = np.array(v0_list[orig_idx], dtype=np.int64, copy=True)
        else:
            u = rng.permutation(N).astype(np.int64)
            v = rng.permutation(N).astype(np.int64)
        C, sizes, counts = fast_core.build_state(u, v)
        U[rank], V[rank], Call[rank], Sizes[rank], Counts[rank] = u, v, C, sizes, counts

    max_measurements = int(n_sweeps // max(measure_every, 1) + 2)
    actions, swap_attempts, swap_accepts, accept_counts, propose_counts, round_trips = fast_core.run_pt_jit(
        U, V, Call, Sizes, Counts, betas_sorted, N, K, float(eps), f2_table, int(n_sweeps), int(burn_in),
        int(measure_every), int(seed), max_measurements,
    )

    results = {}
    for k in range(K):
        acts = np.asarray(actions[k], dtype=float)
        tau, converged = integrated_autocorr_time(acts)
        eff_n = len(acts) / (2.0 * tau) if np.isfinite(tau) and tau > 0 else float(len(acts))
        final_C = orders_to_matrix(U[k], V[k], natural_label=True)
        results[float(betas_sorted[k])] = {
            "actions": acts, "tau_int": tau, "tau_converged": converged,
            "effective_samples": eff_n,
            "acceptance_rate": float(accept_counts[k] / propose_counts[k]) if propose_counts[k] else float("nan"),
            "final_C": final_C, "final_u": U[k], "final_v": V[k],
        }

    swap_rates = swap_accepts / np.maximum(swap_attempts, 1)
    return {
        "betas": betas_sorted, "results": results, "swap_rates": swap_rates,
        "round_trips": np.asarray(round_trips, dtype=np.int64),
    }


def densify_ladder_for_low_swap_rates(betas, swap_rates, min_rate=0.2, max_inserted=8):
    """Insert a midpoint beta between any adjacent pair whose swap rate
    fell below min_rate. Returns the new (possibly larger) sorted beta
    array; a no-op (returns betas unchanged) if every pair is already
    at or above min_rate or the insertion cap is hit.
    """
    betas = np.asarray(betas, dtype=np.float64)
    inserted = []
    for k in range(len(betas) - 1):
        if swap_rates[k] < min_rate and len(inserted) < max_inserted:
            inserted.append(0.5 * (betas[k] + betas[k + 1]))
    if not inserted:
        return betas
    return np.sort(np.concatenate([betas, np.array(inserted, dtype=np.float64)]))
