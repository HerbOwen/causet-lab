"""Lattice-gas causal sets on a d=2 cylinder (Phase 4), following
Cunningham & Surya, "Dimensionally Restricted Causal Set Quantum
Gravity", arXiv:1908.11647 ("C&S"). See lattice_gas_core.py's module
docstring for the detailed, quoted construction (lattice, filling,
causal relation -- including the one point C&S leave unspecified and
this project's own derivation for it) and the smeared action (Eq. 8,
quoted exactly, including the genuine normalization difference from
the order-based action used in Phase 3/3b).

Sizing (C&S Sec. 2/4, quoted in lattice_gas_core.py): w = n, h = alpha*n,
m = alpha*n^2, with alpha = 4 and eps = 0.1 their own choices for d=2.
This phase keeps both exactly, at laptop-scale n (30, 50, 80) instead
of their n=200.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from numba import njit

from . import lattice_gas_core as lg
from . import rng
from .action import f2_smear_table
from .checkpoint import (
    save_checkpoint, load_checkpoint, make_lattice_wl_state, make_lattice_muca_state,
)
from ..measures import height as measure_height
from . import muca
from .muca import (
    N_BINS_INITIAL, RANGE_MARGIN_FRAC, PROGRESS_EVERY_S, F_INITIAL, F_FINAL,
    FLATNESS_THRESHOLD, WIDEN_BINS, EDGE_HIT_FRACTION_TRIGGER, STALL_SECONDS,
    MUCA_MEASURE_EVERY, HIDDEN_BARRIER_RATIO_THRESHOLD, _widen,
    check_ln_g_anomalies,
)

ALPHA_DEFAULT = 4
EPS_DEFAULT = 0.1

# C&S give only one empirical point, beta_c^(2) ~= 2.344 at n=200, no
# n-scaling formula (unlike Glaser-O'Connor-Surya 2018's order model) --
# this 1/n heuristic is OUR OWN guess, used only to pick a wide-enough
# pilot range for estimate_initial_bin_range_lattice; it is explicitly
# not trusted as a real prediction anywhere results are reported.
BETA_C_CS_2019_N200 = 2.344
BETA_C_CS_2019_N = 200


def beta_c_guess(n: float) -> float:
    """Heuristic-only 1/n extrapolation of C&S's single n=200 data
    point -- see module-level note above. Not a published formula."""
    return BETA_C_CS_2019_N200 * BETA_C_CS_2019_N / n


def lattice_dims(n: int, alpha: int = ALPHA_DEFAULT) -> Tuple[int, int, int]:
    """w, h, m for an n-element d=2 lattice-gas filling, C&S's own
    convention: w = n, h = alpha*n, m = h*w = alpha*n^2."""
    w = n
    h = alpha * n
    m = h * w
    return w, h, m


def random_filling(n: int, m: int, seed: int) -> np.ndarray:
    """A uniformly random n-site filling of the m-site lattice (the
    beta=0 ensemble): a random permutation of site ids {0,...,m-1},
    with the first n entries being the occupied sites -- C&S's own
    representation (quoted in lattice_gas_core.py)."""
    rng_np = np.random.default_rng(seed)
    return rng_np.permutation(m).astype(np.int64)


def lattice_to_matrix(L: np.ndarray, n: int, w: int) -> np.ndarray:
    """Full (n, n) boolean relation matrix from a filling L, for
    compatibility with measures.py (which takes a raw matrix) and as
    an independent ground truth for the incremental bitset engine.
    O(n^2), fine for the laptop-scale n used in this phase (<= 80).
    """
    t = (L[:n] // w).astype(np.int64)
    k = (L[:n] % w).astype(np.int64)
    C = np.zeros((n, n), dtype=bool)
    for x in range(n):
        dt = t - t[x]
        dk = (k - k[x]) % w
        dk = np.minimum(dk, w - dk)
        C[x, :] = (dt > 0) & (dk <= dt)
    np.fill_diagonal(C, False)
    return C


def layer_count(L: np.ndarray, n: int, w: int) -> int:
    """Number of distinct lattice time-rows (t = site_id // w) occupied
    by the n elements -- C&S's own diagnostic for the cold/layered
    phase, where they report configurations collapsing onto ~5 rows
    (Sec. 5, "symmetric maximally connected bilayer poset" discussion).
    """
    t = L[:n] // w
    return int(np.unique(t).size)


def lattice_gas_action_full(L: np.ndarray, n: int, w: int, eps: float) -> Tuple[float, np.ndarray]:
    """Full (non-incremental) recomputation of C&S's Eq. 8 action from
    a filling L -- the ground truth for apply_relocate's incremental
    update, independent of lattice_gas_core's bitset path (builds the
    full relation matrix via lattice_to_matrix, then counts interval
    sizes via the same (C@C) trick as measures.interval_abundances,
    rather than via popcount/bitsets).
    """
    C = lattice_to_matrix(L, n, w)
    total_relations = int(C.sum())
    Nk = np.zeros(max(n - 1, 1), dtype=np.int64)
    if total_relations > 0:
        Cf = C.astype(np.float32)
        sizes = np.rint(Cf @ Cf).astype(np.int64)
        related_sizes = sizes[C]
        Nk = np.bincount(related_sizes, minlength=max(n - 1, 1))[: max(n - 1, 1)].astype(np.int64)
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    S = lg.lattice_gas_action_from_counts(n, Nk, f2_table, eps)
    return S, Nk


@njit(cache=True)
def _pilot_sweeps_jit(L, future, past, counts, n, m, w, beta, eps, f2_table,
                       n_sweeps, burn_in, measure_every, anneal_from, has_anneal,
                       rng_state, max_measurements):
    """Minimal fixed-beta Metropolis loop for this model -- used only
    to pilot the hot/cold action range (estimate_initial_bin_range's
    lattice-gas twin) and for the beta=0 vs sprinkle control's
    equilibration; NOT the WL/MUCA path (see
    lattice_gas_core.wl_or_muca_sweep_chunk_jit for that).
    """
    S = lg._lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
    actions = np.empty(max_measurements, dtype=np.float64)
    n_actions = 0
    moves_per_sweep = n * (n - 1) // 2
    for sweep in range(n_sweeps):
        if has_anneal and sweep < burn_in:
            frac = sweep / max(burn_in - 1, 1)
            beta_eff = anneal_from + (beta - anneal_from) * frac
        else:
            beta_eff = beta
        for _move in range(moves_per_sweep):
            rng_state, i = rng.next_below(rng_state, n)
            rng_state, j_off = rng.next_below(rng_state, m - n)
            j = n + j_off
            lg.apply_relocate_jit(L, future, past, counts, n, w, i, j)
            S_new = lg._lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
            dS = S_new - S
            accept = dS <= 0.0
            if not accept:
                rng_state, r = rng.next_double(rng_state)
                accept = r < np.exp(-beta_eff * dS)
            if accept:
                S = S_new
            else:
                lg.apply_relocate_jit(L, future, past, counts, n, w, i, j)
        if sweep >= burn_in and (sweep - burn_in) % measure_every == 0 and n_actions < max_measurements:
            actions[n_actions] = S
            n_actions += 1
    return actions[:n_actions], rng_state


def run_pilot_chain(n, w, h, m, eps, beta, seed, n_sweeps, burn_in, measure_every=5, anneal_from=None):
    """Python wrapper around _pilot_sweeps_jit -- fresh random filling,
    returns the measured action time series."""
    L = random_filling(n, m, seed)
    future, past, counts = lg.build_lattice_state_bitset(L, n, w)
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    rng_state = rng.seed_state(seed)
    max_meas = n_sweeps // measure_every + 2
    has_anneal = anneal_from is not None
    actions, rng_state = _pilot_sweeps_jit(
        L, future, past, counts, n, m, w, beta, eps, f2_table,
        n_sweeps, burn_in, measure_every, anneal_from if has_anneal else 0.0, has_anneal,
        rng_state, max_meas,
    )
    return actions, L, future, past, counts


# ----------------------------------------------------------- WL / MUCA

def estimate_initial_bin_range_lattice(n, w, h, m, eps, seed=0):
    """Lattice-gas twin of muca.estimate_initial_bin_range: pilot the
    hot (beta=0) and a guessed-deep-cold phase cheaply to read off a
    real action range to bin over. Uses beta_c_guess (OUR OWN 1/n
    heuristic, not a published formula -- see its docstring) only to
    pick a plausibly-deep beta for the cold pilot; the located range
    itself comes from the actual sampled actions, not the guess.
    """
    guess = beta_c_guess(n)
    hot, *_ = run_pilot_chain(n, w, h, m, eps, beta=0.0, seed=seed, n_sweeps=40, burn_in=10, measure_every=4)
    cold, *_ = run_pilot_chain(n, w, h, m, eps, beta=guess * 15.0, seed=seed, n_sweeps=300, burn_in=250,
                                measure_every=5, anneal_from=0.0)
    s_hi = float(np.max(hot))
    s_lo = float(np.min(cold))
    if s_lo > s_hi:
        s_lo, s_hi = min(s_lo, s_hi) - 1.0, max(s_lo, s_hi) + 1.0
    span = s_hi - s_lo
    margin = RANGE_MARGIN_FRAC * span
    return s_lo - margin, s_hi + margin


def run_wang_landau_lattice(n, eps, alpha=ALPHA_DEFAULT, seed=0, checkpoint_path=None,
                             checkpoint_every_s=60, chunk_moves=None, verbose=True, verbose_label=None):
    """Lattice-gas twin of muca.run_wang_landau -- identical algorithm
    (adaptive bin widening, stall-detection on genuinely unreachable
    bins, stage-doubling modification factor), only the initial state
    construction and the per-move kernel differ (lattice filling +
    lattice_gas_core's sweep kernel instead of a random 2D order +
    muca_core's).
    """
    w, h, m = lattice_dims(n, alpha)
    label = verbose_label or f"n={n} eps={eps}"
    t0 = time.time()
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * n

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "lattice_wl":
        L = ckpt["L"]
        future, past, counts = ckpt["future"], ckpt["past"], ckpt["counts"]
        rng_state = np.uint64(ckpt["rng_state"])
        ln_g, H = ckpt["ln_g"], ckpt["H"]
        bin_lo, bin_width, n_bins = ckpt["bin_lo"], ckpt["bin_width"], ckpt["n_bins"]
        f_mod, stage = ckpt["f_mod"], ckpt["stage"]
        bin_idx = ckpt["bin_idx"]
        extreme_state, half_trips = ckpt["extreme_state"], ckpt["half_trips"]
        edge_hits_total = ckpt["edge_hits_total"]
        ever_visited = ckpt.get("ever_visited", H > 0).copy()
        full_coverage_required = bool(ckpt.get("full_coverage_required", True))
        if verbose:
            print(f"[lattice-wl] {label} resumed from checkpoint at stage={stage}, f={f_mod:.2e}", flush=True)
    else:
        lo, hi = estimate_initial_bin_range_lattice(n, w, h, m, eps, seed)
        n_bins = N_BINS_INITIAL
        bin_width = (hi - lo) / n_bins
        bin_lo = lo
        ln_g = np.zeros(n_bins, dtype=np.float64)
        H = np.zeros(n_bins, dtype=np.int64)
        rng_state = rng.seed_state(seed)
        L = random_filling(n, m, seed)
        future, past, counts = lg.build_lattice_state_bitset(L, n, w)
        f_mod, stage = F_INITIAL, 0
        S0 = lg.lattice_gas_action_from_counts(n, counts, f2_table, eps)
        bin_idx = lg.bin_index(S0, bin_lo, bin_width, n_bins)
        if bin_idx < 0:
            bin_idx = n_bins // 2
        extreme_state = np.zeros(1, dtype=np.int64)
        half_trips = np.zeros(1, dtype=np.int64)
        edge_hits_total = 0
        ever_visited = np.zeros(n_bins, dtype=bool)
        full_coverage_required = True
        if verbose:
            print(f"[lattice-wl] {label} starting fresh: range=[{lo:.3f}, {hi:.3f}], n_bins={n_bins}", flush=True)

    last_ckpt = time.time()
    last_progress = time.time()
    last_growth = time.time()
    n_ever_visited_prev = int(ever_visited.sum())
    total_moves = 0

    def _reach_extremes():
        if ever_visited.any():
            idx = np.flatnonzero(ever_visited)
            return int(idx[0]), int(idx[-1])
        return bin_idx, bin_idx

    lo_extreme, hi_extreme = _reach_extremes()
    while f_mod > F_FINAL:
        rng_state, bin_idx, edge_hits = lg.wl_or_muca_sweep_chunk_jit(
            L, future, past, counts, n, m, w, eps, f2_table, ln_g, H,
            bin_lo, bin_width, n_bins, f_mod, True, chunk_moves,
            rng_state, bin_idx, extreme_state, half_trips, 0, lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        edge_hits_total += edge_hits
        total_moves += chunk_moves

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            n_visited = int((H > 0).sum())
            print(f"[lattice-wl] {label} stage={stage} f={f_mod:.2e} moves={total_moves} "
                  f"visited={n_visited}/{n_bins} bin_idx={bin_idx} "
                  f"edge_hits_total={edge_hits_total} ({time.time()-t0:.1f}s)", flush=True)
            last_progress = time.time()
            for bin_a, bin_b, step in check_ln_g_anomalies(ln_g, ever_visited):
                print(f"[lattice-wl] {label} WARNING: anomalous ln_g step at bins "
                      f"{bin_a}->{bin_b}: {step:.2f} (ln_g={ln_g[bin_a]:.2f} -> {ln_g[bin_b]:.2f}) "
                      f"-- likely trap risk ({time.time()-t0:.1f}s)", flush=True)

        if edge_hits > EDGE_HIT_FRACTION_TRIGGER * chunk_moves:
            side = "low" if bin_idx < n_bins // 2 else "high"
            old_n_bins = n_bins
            ln_g, H, bin_lo, n_bins = _widen(ln_g, H, bin_lo, bin_width, n_bins, side)
            pad = np.zeros(n_bins - old_n_bins, dtype=bool)
            ever_visited = np.concatenate([pad, ever_visited]) if side == "low" else np.concatenate([ever_visited, pad])
            bin_idx = lg.bin_index(
                lg.lattice_gas_action_from_counts(n, counts, f2_table, eps), bin_lo, bin_width, n_bins,
            )
            lo_extreme, hi_extreme = _reach_extremes()
            if verbose:
                print(f"[lattice-wl] {label} widened range on the {side} side -> n_bins={n_bins}", flush=True)
            continue

        ever_visited |= (H > 0)
        n_ever_visited = int(ever_visited.sum())
        if n_ever_visited > n_ever_visited_prev:
            n_ever_visited_prev = n_ever_visited
            last_growth = time.time()
            lo_extreme, hi_extreme = _reach_extremes()
        elif full_coverage_required and time.time() - last_growth > STALL_SECONDS:
            full_coverage_required = False
            if verbose:
                print(f"[lattice-wl] {label} stalled at {n_ever_visited}/{n_bins} bins after "
                      f"{STALL_SECONDS:.0f}s with no new bin visited -- treating the rest as "
                      f"unreachable for this n and proceeding on the observed support", flush=True)

        ready = (not full_coverage_required) or (n_ever_visited == n_bins)
        if ready:
            check_set = (H > 0) if full_coverage_required else ((H > 0) & ever_visited)
            if check_set.sum() >= max(3, int(0.5 * n_ever_visited)):
                flatness = float(H[check_set].min()) / float(H[check_set].mean())
                if flatness >= FLATNESS_THRESHOLD:
                    stage += 1
                    f_mod = f_mod / 2.0
                    H[:] = 0
                    if verbose:
                        print(f"[lattice-wl] {label} stage {stage}: flat (min/mean={flatness:.2f} over "
                              f"{int(check_set.sum())}/{n_ever_visited} reachable bins), "
                              f"f -> {f_mod:.2e} ({time.time()-t0:.1f}s)", flush=True)

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_lattice_wl_state(
                L, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
                n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
                ever_visited, full_coverage_required,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_lattice_wl_state(
            L, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
            n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
            ever_visited, full_coverage_required,
        ))

    return {
        "n": n, "eps": eps, "w": w, "h": h, "m": m, "N": n,
        "ln_g": ln_g, "bin_lo": bin_lo, "bin_width": bin_width,
        "n_bins": n_bins, "stages": stage, "edge_hits_total": edge_hits_total,
        "elapsed": time.time() - t0,
        "L": L, "future": future, "past": past, "counts": counts, "rng_state": rng_state,
        "bin_idx": bin_idx, "ever_visited": ever_visited, "n_reachable_bins": int(ever_visited.sum()),
        "ln_g_anomalies": check_ln_g_anomalies(ln_g, ever_visited),
    }


def _measure_structural_lattice(L, n, w, counts):
    """height via measures.height on the reconstructed relation matrix
    (O(n^2), fine to call once per production chunk); ordering_fraction
    is read directly off the maintained interval-size histogram, same
    approach as muca._measure_structural."""
    C = lattice_to_matrix(L, n, w)
    h = measure_height(C)
    max_pairs = n * (n - 1) // 2
    of = float(np.sum(counts)) / max_pairs if max_pairs else float("nan")
    return h, of


def run_muca_production_lattice(wl_result, target_round_trips=10, max_moves=None,
                                 checkpoint_path=None, checkpoint_every_s=60, chunk_moves=None,
                                 round_trip_window=None, verbose=True, verbose_label=None):
    """Lattice-gas twin of muca.run_muca_production -- same algorithm
    and the same return-dict shape (so muca.reweight_P_beta_corrected,
    locate_beta_c_variance_peak, run_muca_recursion and
    run_muca_production_parallel all work on its output unchanged),
    with L in place of (u, v) and lattice_gas_core's kernels.
    """
    n, eps = wl_result["n"], wl_result["eps"]
    w = wl_result["w"]
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    ln_g = wl_result["ln_g"]
    bin_lo, bin_width, n_bins = wl_result["bin_lo"], wl_result["bin_width"], wl_result["n_bins"]
    label = verbose_label or f"n={n} eps={eps}"
    t0 = time.time()
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * n

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "lattice_muca":
        L = ckpt["L"]
        future, past, counts = ckpt["future"], ckpt["past"], ckpt["counts"]
        rng_state = np.uint64(ckpt["rng_state"])
        bin_idx = ckpt["bin_idx"]
        half_trips = ckpt["half_trips"]
        moves_done = ckpt["moves_done"]
        rec_bins, rec_S = list(ckpt["rec_bins"]), list(ckpt["rec_S"])
        rec_height, rec_of = list(ckpt["rec_height"]), list(ckpt["rec_of"])
        rec_struct_bin = list(ckpt.get("rec_struct_bin", []))
        if verbose:
            print(f"[lattice-muca] {label} resumed: moves_done={moves_done}, "
                  f"round_trips={int(half_trips[0]) // 2}", flush=True)
    else:
        L = wl_result["L"].copy()
        future, past, counts = wl_result["future"].copy(), wl_result["past"].copy(), wl_result["counts"].copy()
        rng_state = wl_result["rng_state"]
        bin_idx = wl_result["bin_idx"]
        half_trips = np.zeros(1, dtype=np.int64)
        moves_done = 0
        rec_bins, rec_S, rec_height, rec_of, rec_struct_bin = [], [], [], [], []

    m = wl_result["m"]
    extreme_state = np.zeros(1, dtype=np.int64)
    H = np.zeros(n_bins, dtype=np.int64)
    max_meas = chunk_moves // MUCA_MEASURE_EVERY + 2
    last_ckpt = time.time()
    last_progress = time.time()

    if round_trip_window is not None:
        lo_extreme, hi_extreme = round_trip_window
    else:
        ever_visited = wl_result.get("ever_visited")
        if ever_visited is not None and ever_visited.any():
            idx = np.flatnonzero(ever_visited)
            lo_extreme, hi_extreme = int(idx[0]), int(idx[-1])
        else:
            lo_extreme, hi_extreme = 0, n_bins - 1

    while (int(half_trips[0]) // 2) < target_round_trips and (max_moves is None or moves_done < max_moves):
        rng_state, bin_idx, chunk_bins, chunk_S = lg.muca_measure_chunk_jit(
            L, future, past, counts, n, m, w, eps, f2_table, ln_g, H, bin_lo, bin_width, n_bins,
            chunk_moves, rng_state, bin_idx, half_trips, extreme_state, MUCA_MEASURE_EVERY, max_meas,
            lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        moves_done += chunk_moves
        rec_bins.extend(chunk_bins.tolist())
        rec_S.extend(chunk_S.tolist())
        h, of = _measure_structural_lattice(L, n, w, counts)
        rec_height.append(h)
        rec_of.append(of)
        rec_struct_bin.append(bin_idx)

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            print(f"[lattice-muca] {label} moves={moves_done} round_trips={int(half_trips[0]) // 2} "
                  f"height={h} of={of:.3f} ({time.time() - t0:.1f}s)", flush=True)
            last_progress = time.time()

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_lattice_muca_state(
                L, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
                bin_idx, half_trips, moves_done, rec_bins, rec_S, rec_height, rec_of, rec_struct_bin,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_lattice_muca_state(
            L, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
            bin_idx, half_trips, moves_done, rec_bins, rec_S, rec_height, rec_of, rec_struct_bin,
        ))

    round_trips = int(half_trips[0]) // 2
    height_arr = np.array(rec_height, dtype=float)
    hidden_barrier = False
    height_round_trips = 0
    if round_trips > 0 and height_arr.size >= 4:
        h_lo, h_hi = float(height_arr.min()), float(height_arr.max())
        h_mid_lo, h_mid_hi = h_lo + (h_hi - h_lo) / 3.0, h_hi - (h_hi - h_lo) / 3.0
        h_state = 0
        h_half_trips = 0
        for hv in height_arr:
            if hv <= h_mid_lo:
                if h_state == 1:
                    h_half_trips += 1
                h_state = -1
            elif hv >= h_mid_hi:
                if h_state == -1:
                    h_half_trips += 1
                h_state = 1
        height_round_trips = h_half_trips // 2
        if height_round_trips < HIDDEN_BARRIER_RATIO_THRESHOLD * round_trips:
            hidden_barrier = True

    return {
        "n": n, "eps": eps, "w": w, "m": m, "N": n, "bin_lo": bin_lo, "bin_width": bin_width, "n_bins": n_bins,
        "ln_g": ln_g, "H_muca": H, "round_trips": round_trips, "moves_done": moves_done,
        "elapsed": time.time() - t0,
        "rec_bins": np.array(rec_bins, dtype=np.int64), "rec_S": np.array(rec_S, dtype=np.float64),
        "rec_height": height_arr, "rec_of": np.array(rec_of, dtype=float),
        "rec_struct_bin": np.array(rec_struct_bin, dtype=np.int64),
        "height_round_trips": height_round_trips, "hidden_barrier_warning": hidden_barrier,
        "L": L, "future": future, "past": past, "counts": counts, "rng_state": rng_state, "bin_idx": bin_idx,
    }
