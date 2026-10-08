"""Wang-Landau + multicanonical (MUCA) orchestration (Phase 3b, part 3).

Why: fixed-beta Metropolis (even with parallel tempering, see
tempering.py) could not reliably cross the first-order barrier at
beta_c for this project's smeared action -- Phase 3's follow-up checks
found most located beta_c values were right-censored (no converged
data on the far side of the "peak") and, worse, not even stable under
a blind grid-recentering test at eps=0.21. Wang-Landau estimates the
density of states g(S) by walking in action-space with an adaptive
weight that actively flattens the visit histogram across the *whole*
binned range in one run, rather than hoping independent fixed-beta
chains happen to bridge the barrier; multicanonical production then
samples with that converged weight fixed, and the resulting one run
can be reweighted to the canonical P_beta(S) and <O>(beta) at *any*
beta (Berg & Neuhaus, "Multicanonical algorithms for first order phase
transitions", Phys. Lett. B 267 (1991) 249) -- no new sampling needed
per beta, unlike the old beta-by-beta PT scan.

Binning: the initial range is read off from cheap pilot samples of the
hot (beta=0) and cold (deep annealed high-beta) phases, with margin,
rather than assumed -- see estimate_initial_bin_range. If the WL walk
hits an edge bin often (bin_index returning -1 in muca_core.py), the
range is widened and the run resumes from its own in-memory state
(not a fresh restart): new edge bins are seeded with the nearest
existing bin's current ln_g (flat extrapolation), not zero, since
starting a fresh bin at ln_g=0 next to bins that have already
accumulated a large ln_g would make the new bin look spuriously
under-visited and pull the walk into it -- this is standard WL
practice, not a detail that can be skipped.
"""
from __future__ import annotations

import time

import numpy as np

from .action import beta_c_glaser_2018, f2_smear_table
from .sampler import run_chain_fast
from .orders import orders_to_matrix
from ..measures import height as measure_height
from . import bitset_core as bc
from . import muca_core as mc
from . import rng
from .checkpoint import save_checkpoint, load_checkpoint, make_wl_state, make_muca_state
from .. import plots
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3b"

MUCA_MEASURE_EVERY = 5
HIDDEN_BARRIER_RATIO_THRESHOLD = 0.2  # height round trips below this fraction of S round trips -> warn

# f_mod is added directly to ln_g (ln_g[bin] += f_mod), i.e. f_mod *is*
# ln(f) in the usual multiplicative Wang-Landau notation (g := g * f).
# Halving the *modification factor* f therefore means f_mod /= 2, not
# sqrt(f_mod) -- sqrt only makes sense if f_mod stored the multiplicative
# factor itself, which it doesn't here. F_INITIAL=1.0 matches the
# standard f0=e (ln(e)=1) starting point; F_FINAL=1e-6 matches the usual
# "stop when f is within 1e-6 of 1" convergence criterion.
N_BINS_INITIAL = 120
RANGE_MARGIN_FRAC = 0.35
PROGRESS_EVERY_S = 10
F_INITIAL = 1.0
F_FINAL = 1e-6
FLATNESS_THRESHOLD = 0.80
WIDEN_BINS = 60
EDGE_HIT_FRACTION_TRIGGER = 0.002  # widen if >0.2% of moves in a chunk hit an edge
# For finite N, the discrete action spectrum can have genuine gaps --
# bins that no (u, v) configuration can ever land in, not bins that are
# merely slow to reach. Waiting for literal 100% bin coverage would
# then never terminate. If the set of ever-visited bins stops growing
# for this many seconds, stop requiring full coverage and compute
# flatness over the observed (ever-visited) support instead -- see the
# stall-handling block in run_wang_landau.
STALL_SECONDS = 25.0


def estimate_initial_bin_range(N, eps, seed=0):
    """Pilot the hot (beta=0) and cold (annealed, 15x predicted beta_c)
    phases cheaply to read off a real action range to bin over, rather
    than guessing one -- see module docstring.
    """
    predicted = beta_c_glaser_2018(N, eps)
    hot = run_chain_fast(N, beta=0.0, seed=seed, n_sweeps=200, burn_in=50, measure_every=4, eps=eps)
    cold = run_chain_fast(N, beta=predicted * 15.0, seed=seed, n_sweeps=2500, burn_in=2000,
                           measure_every=10, eps=eps, anneal_from=0.0)
    s_hi = float(np.max(hot.actions))
    s_lo = float(np.min(cold.actions))
    if s_lo > s_hi:  # pathological pilot; fall back to a wide symmetric guess
        s_lo, s_hi = min(s_lo, s_hi) - 1.0, max(s_lo, s_hi) + 1.0
    span = s_hi - s_lo
    margin = RANGE_MARGIN_FRAC * span
    return s_lo - margin, s_hi + margin


def _widen(ln_g, H, bin_lo, bin_width, n_bins, side, n_new=WIDEN_BINS):
    if side == "low":
        new_ln_g = np.empty(n_bins + n_new, dtype=np.float64)
        new_ln_g[n_new:] = ln_g
        new_ln_g[:n_new] = ln_g[0]  # flat extrapolation, not zero -- see module docstring
        new_H = np.zeros(n_bins + n_new, dtype=np.int64)
        new_H[n_new:] = H
        return new_ln_g, new_H, bin_lo - n_new * bin_width, n_bins + n_new
    else:
        new_ln_g = np.empty(n_bins + n_new, dtype=np.float64)
        new_ln_g[:n_bins] = ln_g
        new_ln_g[n_bins:] = ln_g[-1]
        new_H = np.zeros(n_bins + n_new, dtype=np.int64)
        new_H[:n_bins] = H
        return new_ln_g, new_H, bin_lo, n_bins + n_new


def run_wang_landau(N, eps, seed=0, checkpoint_path=None, checkpoint_every_s=60,
                     chunk_moves=None, verbose=True, verbose_label=None):
    """Returns a dict: ln_g, bin_lo, bin_width, n_bins, stages, edge_hits_total,
    elapsed, plus the final (u, v, future, past, counts, rng_state) so a MUCA
    production run can continue from the same, already-thermalized walker.
    """
    label = verbose_label or f"N={N} eps={eps}"
    t0 = time.time()
    f2_table = f2_smear_table(max(N - 2, 0), eps)
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * N

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "wl":
        u, v = ckpt["u"], ckpt["v"]
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
            print(f"[wl] {label} resumed from checkpoint at stage={stage}, f={f_mod:.2e}", flush=True)
    else:
        lo, hi = estimate_initial_bin_range(N, eps, seed)
        n_bins = N_BINS_INITIAL
        bin_width = (hi - lo) / n_bins
        bin_lo = lo
        ln_g = np.zeros(n_bins, dtype=np.float64)
        H = np.zeros(n_bins, dtype=np.int64)
        rng_state = rng.seed_state(seed)
        rr = np.random.default_rng(seed)
        u = rr.permutation(N).astype(np.int64)
        v = rr.permutation(N).astype(np.int64)
        future, past, counts = bc.build_state_bitset(u, v)
        f_mod, stage = F_INITIAL, 0
        S0 = bc.action_from_counts(N, counts, f2_table, eps)
        bin_idx = mc.bin_index(S0, bin_lo, bin_width, n_bins)
        if bin_idx < 0:
            bin_idx = n_bins // 2  # pathological; recover rather than crash
        extreme_state = np.zeros(1, dtype=np.int64)
        half_trips = np.zeros(1, dtype=np.int64)
        edge_hits_total = 0
        ever_visited = np.zeros(n_bins, dtype=bool)
        full_coverage_required = True
        if verbose:
            print(f"[wl] {label} starting fresh: range=[{lo:.3f}, {hi:.3f}], n_bins={n_bins}", flush=True)

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
        rng_state, bin_idx, edge_hits = mc.wl_or_muca_sweep_chunk_jit(
            u, v, future, past, counts, N, eps, f2_table, ln_g, H,
            bin_lo, bin_width, n_bins, f_mod, True, chunk_moves,
            rng_state, bin_idx, extreme_state, half_trips, 0, lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        edge_hits_total += edge_hits
        total_moves += chunk_moves

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            n_visited = int((H > 0).sum())
            print(f"[wl] {label} stage={stage} f={f_mod:.2e} moves={total_moves} "
                  f"visited={n_visited}/{n_bins} bin_idx={bin_idx} "
                  f"edge_hits_total={edge_hits_total} ({time.time()-t0:.1f}s)", flush=True)
            last_progress = time.time()

        if edge_hits > EDGE_HIT_FRACTION_TRIGGER * chunk_moves:
            side = "low" if bin_idx < n_bins // 2 else "high"
            old_n_bins = n_bins
            ln_g, H, bin_lo, n_bins = _widen(ln_g, H, bin_lo, bin_width, n_bins, side)
            pad = np.zeros(n_bins - old_n_bins, dtype=bool)
            ever_visited = np.concatenate([pad, ever_visited]) if side == "low" else np.concatenate([ever_visited, pad])
            bin_idx = mc.bin_index(
                bc.action_from_counts(N, counts, f2_table, eps), bin_lo, bin_width, n_bins,
            )
            lo_extreme, hi_extreme = _reach_extremes()
            if verbose:
                print(f"[wl] {label} widened range on the {side} side -> n_bins={n_bins}", flush=True)
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
                print(f"[wl] {label} stalled at {n_ever_visited}/{n_bins} bins after "
                      f"{STALL_SECONDS:.0f}s with no new bin visited -- treating the rest as "
                      f"unreachable for this N and proceeding on the observed support", flush=True)

        ready = (not full_coverage_required) or (n_ever_visited == n_bins)
        if ready:
            # Flatness is judged on bins visited at least once *this stage*. Once full
            # coverage has been given up on, a permanently-unreachable bin would otherwise
            # read H=0 forever and wrongly tank the flatness ratio -- excluded via ever_visited.
            check_set = (H > 0) if full_coverage_required else ((H > 0) & ever_visited)
            if check_set.sum() >= max(3, int(0.5 * n_ever_visited)):
                flatness = float(H[check_set].min()) / float(H[check_set].mean())
                if flatness >= FLATNESS_THRESHOLD:
                    stage += 1
                    f_mod = f_mod / 2.0
                    H[:] = 0
                    if verbose:
                        print(f"[wl] {label} stage {stage}: flat (min/mean={flatness:.2f} over "
                              f"{int(check_set.sum())}/{n_ever_visited} reachable bins), "
                              f"f -> {f_mod:.2e} ({time.time()-t0:.1f}s)", flush=True)

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_wl_state(
                u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
                n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
                ever_visited, full_coverage_required,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_wl_state(
            u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
            n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
            ever_visited, full_coverage_required,
        ))

    return {
        "N": N, "eps": eps, "ln_g": ln_g, "bin_lo": bin_lo, "bin_width": bin_width,
        "n_bins": n_bins, "stages": stage, "edge_hits_total": edge_hits_total,
        "elapsed": time.time() - t0,
        "u": u, "v": v, "future": future, "past": past, "counts": counts, "rng_state": rng_state,
        "bin_idx": bin_idx, "ever_visited": ever_visited, "n_reachable_bins": int(ever_visited.sum()),
    }


def _measure_structural(u, v, N, counts):
    """height via the existing (tested) measures.height on the
    reconstructed relation matrix (O(N^2), fine to call once per
    production chunk, not per move); ordering_fraction is read
    directly off the already-maintained interval-size histogram
    (sum(counts) = number of related pairs) without reconstructing
    anything, since that is exactly what ordering_fraction needs.
    """
    C = orders_to_matrix(u, v, natural_label=True)
    h = measure_height(C)
    max_pairs = N * (N - 1) // 2
    of = float(np.sum(counts)) / max_pairs if max_pairs else float("nan")
    return h, of


def run_muca_production(wl_result, target_round_trips=10, max_moves=None,
                         checkpoint_path=None, checkpoint_every_s=60, chunk_moves=None,
                         round_trip_window=None, verbose=True, verbose_label=None):
    """Freezes wl_result's ln_g and runs MUCA production, continuing
    from the WL walker's final state (or resuming from its own
    checkpoint). Collects, every chunk: the (bin, S) time series at
    MUCA_MEASURE_EVERY-move resolution (for later per-bin reweighting),
    and one (height, ordering_fraction) structural snapshot (used for
    the hidden-barrier check -- see module docstring in muca_core.py
    and the report for what that means). Stops once target_round_trips
    S-bin round trips are reached (part 5's pass criterion is >= 10) or
    max_moves is hit, whichever first.

    round_trip_window: optional (lo_bin, hi_bin) overriding the
    default (the full ever_visited span) for what counts as "reaching
    an extreme" for round-trip counting. Sampling itself is unchanged
    -- this only changes what gets counted as a completed trip. Useful
    when the reweighted P(S) shows the physically relevant region for
    the betas being compared is much narrower than the full reachable
    action range (checked directly for N=40, eps=0.5: bins [0,85] of
    126 carry essentially all the weight for beta in
    [0.5, 2.0] x predicted beta_c -- counting trips out to bin 125
    requires the walk to visit a region that is irrelevant to any
    beta actually being compared against the formula).
    """
    N, eps = wl_result["N"], wl_result["eps"]
    f2_table = f2_smear_table(max(N - 2, 0), eps)
    ln_g = wl_result["ln_g"]
    bin_lo, bin_width, n_bins = wl_result["bin_lo"], wl_result["bin_width"], wl_result["n_bins"]
    label = verbose_label or f"N={N} eps={eps}"
    t0 = time.time()
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * N

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "muca":
        u, v = ckpt["u"], ckpt["v"]
        future, past, counts = ckpt["future"], ckpt["past"], ckpt["counts"]
        rng_state = np.uint64(ckpt["rng_state"])
        bin_idx = ckpt["bin_idx"]
        half_trips = ckpt["half_trips"]
        moves_done = ckpt["moves_done"]
        rec_bins, rec_S = list(ckpt["rec_bins"]), list(ckpt["rec_S"])
        rec_height, rec_of = list(ckpt["rec_height"]), list(ckpt["rec_of"])
        rec_struct_bin = list(ckpt.get("rec_struct_bin", []))
        if verbose:
            print(f"[muca] {label} resumed: moves_done={moves_done}, "
                  f"round_trips={int(half_trips[0]) // 2}", flush=True)
    else:
        u, v = wl_result["u"].copy(), wl_result["v"].copy()
        future, past, counts = wl_result["future"].copy(), wl_result["past"].copy(), wl_result["counts"].copy()
        rng_state = wl_result["rng_state"]
        bin_idx = wl_result["bin_idx"]
        half_trips = np.zeros(1, dtype=np.int64)
        moves_done = 0
        rec_bins, rec_S, rec_height, rec_of, rec_struct_bin = [], [], [], [], []

    extreme_state = np.zeros(1, dtype=np.int64)
    H = np.zeros(n_bins, dtype=np.int64)
    max_meas = chunk_moves // MUCA_MEASURE_EVERY + 2
    last_ckpt = time.time()
    last_progress = time.time()

    # Round trips are judged between the WL run's *actually reachable*
    # bins, not the raw [0, n_bins) array bounds -- see muca_core.py's
    # module docstring. ever_visited is fixed by now (WL is done).
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
        rng_state, bin_idx, chunk_bins, chunk_S = mc.muca_measure_chunk_jit(
            u, v, future, past, counts, N, eps, f2_table, ln_g, H, bin_lo, bin_width, n_bins,
            chunk_moves, rng_state, bin_idx, half_trips, extreme_state, MUCA_MEASURE_EVERY, max_meas,
            lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        moves_done += chunk_moves
        rec_bins.extend(chunk_bins.tolist())
        rec_S.extend(chunk_S.tolist())
        h, of = _measure_structural(u, v, N, counts)
        rec_height.append(h)
        rec_of.append(of)
        rec_struct_bin.append(bin_idx)

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            print(f"[muca] {label} moves={moves_done} round_trips={int(half_trips[0]) // 2} "
                  f"height={h} of={of:.3f} ({time.time() - t0:.1f}s)", flush=True)
            last_progress = time.time()

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_muca_state(
                u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
                bin_idx, half_trips, moves_done, rec_bins, rec_S, rec_height, rec_of, rec_struct_bin,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_muca_state(
            u, v, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
            bin_idx, half_trips, moves_done, rec_bins, rec_S, rec_height, rec_of, rec_struct_bin,
        ))

    round_trips = int(half_trips[0]) // 2
    height_arr = np.array(rec_height, dtype=float)
    hidden_barrier = False
    if round_trips > 0 and height_arr.size >= 4:
        h_lo, h_hi = float(height_arr.min()), float(height_arr.max())
        h_mid_lo, h_mid_hi = h_lo + (h_hi - h_lo) / 3.0, h_hi - (h_hi - h_lo) / 3.0
        h_state = 0
        h_half_trips = 0
        for h in height_arr:
            if h <= h_mid_lo:
                if h_state == 1:
                    h_half_trips += 1
                h_state = -1
            elif h >= h_mid_hi:
                if h_state == -1:
                    h_half_trips += 1
                h_state = 1
        height_round_trips = h_half_trips // 2
        if height_round_trips < HIDDEN_BARRIER_RATIO_THRESHOLD * round_trips:
            hidden_barrier = True
    else:
        height_round_trips = 0

    return {
        "N": N, "eps": eps, "bin_lo": bin_lo, "bin_width": bin_width, "n_bins": n_bins,
        "ln_g": ln_g, "H_muca": H, "round_trips": round_trips, "moves_done": moves_done,
        "elapsed": time.time() - t0,
        "rec_bins": np.array(rec_bins, dtype=np.int64), "rec_S": np.array(rec_S, dtype=np.float64),
        "rec_height": height_arr, "rec_of": np.array(rec_of, dtype=float),
        "rec_struct_bin": np.array(rec_struct_bin, dtype=np.int64),
        "height_round_trips": height_round_trips, "hidden_barrier_warning": hidden_barrier,
        # Final walker state -- lets a caller (e.g. run_muca_recursion)
        # chain another production run from exactly where this one
        # stopped without round-tripping through a checkpoint file.
        "u": u, "v": v, "future": future, "past": past, "counts": counts,
        "rng_state": rng_state, "bin_idx": bin_idx,
    }


def run_muca_recursion(wl_result, target_round_trips=3, max_iters=10, short_max_moves=None,
                        checkpoint_path=None, max_step=2.0, smooth_window=3,
                        production_fn=None, verbose=True, verbose_label=None):
    """Multicanonical recursion: alternate short frozen-weight
    production runs with a weight update from the run's own histogram
    (refine_ln_g_production), stopping as soon as one short run
    achieves target_round_trips on its own -- rather than chasing full
    Wang-Landau flatness.

    This is cheaper than waiting out a WL stall whenever the stall is
    caused by a slow-to-equilibrate region that reweight_P_beta_corrected
    can still account for exactly: the corrected reweighting folds in
    the production histogram, so it is unbiased even with an
    imperfect ln_g, as long as the walk actually tunnels through that
    region at all (which is what target_round_trips checks for) --
    see reweight_P_beta_corrected's and refine_ln_g_production's
    docstrings. First observed to help on the N=40, eps=0.5 pilot,
    where WL alone stalled for 1700+s on a hard reflecting boundary at
    the true action-spectrum floor, flattening to only ~0.84 and not
    making further progress.

    Each short run is capped at short_max_moves so a bad weight update
    cannot silently turn one "short" iteration into a very long one.

    Deliberately does NOT forward a checkpoint_path into the per-
    iteration run_muca_production calls: that function auto-resumes
    from an existing checkpoint file, including its round-trip counter
    and move count -- reusing one path across iterations would make
    iteration 2+ silently resume iteration 1's stale (pre-refinement)
    state and round-trip progress instead of starting fresh against
    the newly refined weights. checkpoint_path here instead saves a
    snapshot (via save_checkpoint directly) only after the whole
    recursion converges, for crash recovery between pilot stages --
    only supported for the default production_fn (make_muca_state
    assumes (u, v); pass checkpoint_path=None here for any other
    model and let production_fn do its own checkpointing internally).

    production_fn: defaults to run_muca_production (the 2D-orders
    model); pass a different production function with the same
    call signature (e.g. lattice_gas.run_muca_production_lattice) to
    reuse this exact recursion strategy for a different underlying
    move/state representation -- this function only ever touches
    ln_g/H_muca/round_trips/moves_done/bin_idx/rng_state generically.
    """
    production_fn = production_fn or run_muca_production
    label = verbose_label or f"N={wl_result['N']} eps={wl_result['eps']}"
    ln_g = wl_result["ln_g"]
    state = dict(wl_result)
    history = []
    for it in range(max_iters):
        t0 = time.time()
        prod = production_fn(
            {**state, "ln_g": ln_g}, target_round_trips=target_round_trips, max_moves=short_max_moves,
            checkpoint_path=None, verbose=verbose, verbose_label=f"{label} recursion-iter{it}",
        )
        dt = time.time() - t0
        history.append({"iter": it, "round_trips": prod["round_trips"],
                         "moves_done": prod["moves_done"], "elapsed": dt})
        if verbose:
            print(f"[recursion] {label} iter={it} round_trips={prod['round_trips']} "
                  f"moves={prod['moves_done']} ({dt:.1f}s)", flush=True)
        if prod["round_trips"] >= target_round_trips:
            if checkpoint_path:
                save_checkpoint(checkpoint_path, make_muca_state(
                    prod["u"], prod["v"], prod["future"], prod["past"], prod["counts"], prod["rng_state"],
                    ln_g, prod["H_muca"], state["bin_lo"], state["bin_width"], state["n_bins"],
                    prod["bin_idx"], np.zeros(1, dtype=np.int64), prod["moves_done"], [], [], [], [],
                ))
            return {"ln_g": ln_g, "prod": prod, "iterations": it + 1, "history": history, "converged": True}
        ln_g = refine_ln_g_production(ln_g, prod["H_muca"], wl_result["ever_visited"],
                                       max_step=max_step, smooth_window=smooth_window)
        # Merge all of prod's fields into state generically (whatever
        # the walker-state representation is -- (u,v) or (L), etc.)
        # rather than naming specific keys, so this loop works
        # unchanged for any production_fn's state shape.
        state = {**state, **prod}
    return {"ln_g": ln_g, "prod": prod, "iterations": max_iters, "history": history, "converged": False}


def _parallel_production_worker(args):
    state, seed, target_round_trips, max_moves, round_trip_window, production_fn = args
    production_fn = production_fn or run_muca_production
    state = {**state, "rng_state": rng.seed_state(seed)}
    return production_fn(state, target_round_trips=target_round_trips, max_moves=max_moves,
                          round_trip_window=round_trip_window, verbose=False)


def run_muca_production_parallel(state, target_round_trips_total=30, n_workers=None,
                                  max_moves_per_worker=None, seed_base=1000, round_trip_window=None,
                                  production_fn=None, verbose=True, verbose_label=None):
    """Several independent frozen-weight MUCA production chains in
    parallel (default: one per CPU core), each reseeded from the same
    starting configuration/weights but with an independent RNG stream,
    merging visit histograms and round-trip counts afterward.

    Round trips accumulate independently per chain, so splitting one
    long serial target across n_workers parallel chains (each only
    needing ceil(target/n_workers) of its own) reaches the same total
    in roughly 1/n_workers of the wall-clock time, not the same time
    with n_workers x the throughput -- still a straightforward win
    for this embarrassingly-parallel merge, just not a free lunch if
    n_workers exceeds how many round trips are actually needed.

    Keeps each worker's own H_muca/round_trips/time-series separate in
    the returned "per_worker" list (not just the merged totals) --
    needed for a per-worker beta_c spread or a windowed round-trip
    recount later without having to rerun (a first version of this
    function only kept the merged sum, which silently threw that
    information away; see the N=40 pilot's reanalysis for why that
    mattered).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    n_workers = n_workers or os.cpu_count() or 1
    target_per_worker = max(1, -(-target_round_trips_total // n_workers))
    label = verbose_label or f"N={state['N']} eps={state['eps']}"
    tasks = [(state, seed_base + i, target_per_worker, max_moves_per_worker, round_trip_window, production_fn)
             for i in range(n_workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_parallel_production_worker, tasks))
    elapsed = time.time() - t0
    if verbose:
        print(f"[parallel] {label} {n_workers} workers x target {target_per_worker} round trips "
              f"done in {elapsed:.1f}s, totals={[r['round_trips'] for r in results]}", flush=True)

    H_total = sum(r["H_muca"] for r in results)
    round_trips_total = sum(r["round_trips"] for r in results)
    moves_total = sum(r["moves_done"] for r in results)
    hidden_barrier_any = any(r["hidden_barrier_warning"] for r in results)
    rec_bins_all = np.concatenate([r["rec_bins"] for r in results])
    rec_S_all = np.concatenate([r["rec_S"] for r in results])
    rec_height_all = np.concatenate([r["rec_height"] for r in results])
    rec_of_all = np.concatenate([r["rec_of"] for r in results])
    rec_struct_bin_all = np.concatenate([r["rec_struct_bin"] for r in results])
    return {
        "N": state["N"], "eps": state["eps"], "bin_lo": state["bin_lo"],
        "bin_width": state["bin_width"], "n_bins": state["n_bins"], "ln_g": state["ln_g"],
        "H_muca": H_total, "round_trips": round_trips_total, "moves_done": moves_total,
        "elapsed": elapsed, "n_workers": n_workers, "per_worker_round_trips": [r["round_trips"] for r in results],
        "rec_bins": rec_bins_all, "rec_S": rec_S_all, "rec_height": rec_height_all,
        "rec_of": rec_of_all, "rec_struct_bin": rec_struct_bin_all,
        "hidden_barrier_warning": hidden_barrier_any,
        # Per-worker breakdown (H_muca, round_trips, moves_done, and the
        # worker's own rec_bins/rec_S time series) -- needed for a
        # per-worker beta_c spread or a windowed round-trip recount.
        "per_worker": [
            {"H_muca": r["H_muca"], "round_trips": r["round_trips"], "moves_done": r["moves_done"],
             "rec_bins": r["rec_bins"], "rec_S": r["rec_S"]}
            for r in results
        ],
    }


# ----------------------------------------------------------------- reweighting

def bin_centers(bin_lo, bin_width, n_bins):
    return bin_lo + bin_width * (np.arange(n_bins) + 0.5)


def reweight_P_beta(ln_g, bin_lo, bin_width, n_bins, beta):
    """Canonical P_beta(S) for ANY beta from one WL run: P_beta(S) ~
    g(S) * exp(-beta*S) = exp(ln_g(S) - beta*S) (Berg & Neuhaus 1991;
    see module docstring). No new sampling needed per beta -- this is
    the entire point of multicanonical reweighting.
    """
    centers = bin_centers(bin_lo, bin_width, n_bins)
    log_p = ln_g - beta * centers
    log_p = log_p - log_p.max()
    p = np.exp(log_p)
    total = p.sum()
    if total > 0:
        p = p / total
    return p, centers


def reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta):
    """Canonical P_beta(S) folding in the *actual* MUCA production
    histogram, not just the frozen WL weight.

    reweight_P_beta above computes exp(ln_g(S) - beta*S) and silently
    assumes ln_g already equals the true ln(density of states). It
    does -- once WL has flattened the histogram to convergence. Before
    that, the production random walk's own stationary bin distribution
    is pi(S) ~ g_true(S) * exp(-ln_g(S)) (that's what the WL/MUCA
    acceptance rule in muca_core.py targets), so the *observed*
    histogram H_muca(S) ~ g_true(S) * exp(-ln_g(S)) directly measures
    the residual error between ln_g and the truth. Berg & Neuhaus's
    reweighting corrects for exactly this:

        g_true(S) ~ H_muca(S) * exp(+ln_g(S))
        P_beta(S) ~ g_true(S) * exp(-beta*S) = H_muca(S) * exp(ln_g(S) - beta*S)

    This is the unbiased estimator regardless of how well-converged
    ln_g is, *provided* the walk actually visited S during production
    (H_muca(S) > 0) and tunnels between the regions being compared --
    bins with zero production counts carry no information and are
    excluded (not assumed flat/zero-weight, which reweight_P_beta
    effectively does by trusting ln_g there instead).
    """
    centers = bin_centers(bin_lo, bin_width, n_bins)
    valid = H_muca > 0
    log_p = np.full(n_bins, -np.inf)
    log_p[valid] = np.log(H_muca[valid]) + ln_g[valid] - beta * centers[valid]
    if valid.any():
        log_p = log_p - log_p[valid].max()
    p = np.where(valid, np.exp(log_p), 0.0)
    total = p.sum()
    if total > 0:
        p = p / total
    return p, centers


def _per_bin_average(bin_idx_arr, values, n_bins):
    sums = np.zeros(n_bins, dtype=np.float64)
    cnts = np.zeros(n_bins, dtype=np.float64)
    for b, val in zip(bin_idx_arr, values):
        if 0 <= b < n_bins:
            sums[b] += val
            cnts[b] += 1
    avg = np.full(n_bins, np.nan)
    valid = cnts > 0
    avg[valid] = sums[valid] / cnts[valid]
    return avg, valid


def reweight_observable(bin_idx_arr, values, ln_g, H_muca, bin_lo, bin_width, n_bins, beta):
    """<O>(beta) = sum_S O_avg(S) * P_beta(S), where O_avg(S) is the
    MUCA run's observed average of O over samples that landed in bin
    S, and P_beta(S) is the canonical reweighting (folding in the
    production histogram H_muca -- see reweight_P_beta_corrected) --
    the standard multicanonical observable estimator.
    """
    avg, valid = _per_bin_average(bin_idx_arr, values, n_bins)
    p, _ = reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
    if not valid.any() or p[valid].sum() <= 0:
        return float("nan")
    return float(np.sum(avg[valid] * p[valid]) / np.sum(p[valid]))


def reweight_mean_var_S(ln_g, H_muca, bin_lo, bin_width, n_bins, beta):
    """<S>(beta) and Var(S)(beta) from one reweighted P_beta(S)."""
    p, centers = reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
    mean_S = float(np.sum(centers * p))
    var_S = float(np.sum((centers - mean_S) ** 2 * p))
    return mean_S, var_S


def locate_beta_c_variance_peak(ln_g, H_muca, bin_lo, bin_width, n_bins, beta_lo, beta_hi, n_scan=400):
    """beta_c as the location of the maximum of the specific heat /
    action variance Var(S)(beta) -- this is the definition Glaser,
    O'Connor & Surya (2018) actually use to fit their published
    beta_c(N, eps) formula ("we estimate beta_c as the location of the
    maxima of ... the specific heat C", their Sec. 4.1), as opposed to
    the equal-peak-height double-peak criterion
    (locate_beta_c_equal_height) -- the two coincide at a strong
    first-order point but can differ for a weak one, so comparing this
    project's located beta_c against their formula should use this
    function, not the equal-height one, to be a like-with-like test.

    Returns the scanned grid and variance curve too (not just the
    peak) since checking the peak isn't right at a scan edge is part
    of trusting the result -- a true interior maximum needs the
    variance to be lower on *both* sides of beta_lo/beta_hi.
    """
    betas = np.linspace(beta_lo, beta_hi, n_scan)
    variances = np.empty(n_scan)
    for i, beta in enumerate(betas):
        _, variances[i] = reweight_mean_var_S(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
    i_max = int(np.argmax(variances))
    is_interior = 0 < i_max < n_scan - 1
    beta_c = float(betas[i_max])
    if is_interior:
        # Parabolic refinement around the discrete argmax -- otherwise
        # the located beta_c can only ever land exactly on a grid
        # point, which silently floors any uncertainty estimate (e.g.
        # a block bootstrap) built on top of this at the grid spacing
        # rather than the real block-to-block variation.
        y0, y1, y2 = variances[i_max - 1], variances[i_max], variances[i_max + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            step = betas[i_max + 1] - betas[i_max]
            beta_c = beta_c + 0.5 * (y0 - y2) / denom * step
    return {
        "beta_c": beta_c, "var_max": float(variances[i_max]),
        "is_interior": is_interior,
        "betas": betas, "variances": variances,
    }


def refine_ln_g_production(ln_g, H_muca, ever_visited, max_step=2.0, smooth_window=3):
    """One step of multicanonical recursion (Berg's iterative MUCA
    weight refinement): the production walk's stationary bin
    distribution is pi(S) ~ g_true(S) * exp(-ln_g(S)), so a bin visited
    more than its neighbors (H_muca(S) relatively large) means ln_g(S)
    under-suppresses it -- correct by ln_g_new(S) = ln_g(S) +
    ln(H_muca(S)), which is exact in the limit of one infinitely long
    production run and a good approximation for a single finite one.
    Bins with H_muca(S) == 0 (never visited this run, or outside
    ever_visited entirely) keep their old ln_g -- there is no new
    information to refine them with, and dropping them (vs. flat-
    extrapolating, as the widening logic does) would just reintroduce
    the under-visited-edge-bin bias this is meant to fix.

    Two guards against a short run's noisy per-bin counts overcorrecting
    (observed: a one-shot, unsmoothed, unclipped update on the N=30
    pilot swung one bin by 14.8 ln-units and visibly hurt the next
    run's mixing rate before it recovered):
      - smooth_window: ln(H_muca) is boxcar-averaged over this many
        visited neighbors before use, trading a little spatial
        resolution for a lot less single-bin noise.
      - max_step: the resulting *change* to ln_g (relative to the
        visited-bin mean, not an absolute log(H) value) is clipped to
        +/- max_step per call -- a bin that's wildly over/under
        visited gets nudged toward the right answer, not slammed
        there in one step; repeated recursion calls still converge.

    This is a cheap, no-new-WL alternative to rerunning Wang-Landau:
    it reuses a single existing production histogram, at the cost of
    only being as good as that one run's statistics.
    """
    refined = ln_g.copy()
    visited = (H_muca > 0) & ever_visited
    if not visited.any():
        return refined
    log_h = np.full(ln_g.shape, np.nan)
    log_h[visited] = np.log(H_muca[visited])

    if smooth_window > 1:
        half = smooth_window // 2
        smoothed = np.full(ln_g.shape, np.nan)
        for i in np.flatnonzero(visited):
            lo_i, hi_i = max(0, i - half), min(len(ln_g), i + half + 1)
            window = log_h[lo_i:hi_i]
            smoothed[i] = np.nanmean(window)
        log_h = smoothed

    mean_log_h = float(np.mean(log_h[visited]))
    delta = np.clip(log_h[visited] - mean_log_h, -max_step, max_step)
    refined[visited] = ln_g[visited] + delta
    return refined


def interpolate_ln_g_for_new_N(ln_g_prev, bin_lo_prev, bin_width_prev, n_bins_prev, N_prev,
                                bin_lo_new, bin_width_new, n_bins_new, N_new):
    """Starting guess for a new N's ln_g, carried over from a previous
    (even imperfectly converged) N instead of starting cold at
    ln_g=0 -- cold starts are exactly what makes WL expensive (see
    run_wang_landau's stage-by-stage flattening).

    Ansatz: the density of states is approximately extensive, i.e.
    ln_g(S; N) ~ (N/N_prev) * ln_g_prev(S * N_prev/N_new; N_prev) --
    rescale the action argument by N_prev/N_new (actions scale with N)
    and the overall amplitude by N_new/N_prev (entropy is extensive).
    This is only a starting point for recursion/WL to refine, not
    assumed to be already correct -- it just needs to be close enough
    that a handful of recursion iterations converge faster than
    starting from a flat ln_g=0 would.
    """
    centers_prev = bin_centers(bin_lo_prev, bin_width_prev, n_bins_prev)
    centers_new = bin_centers(bin_lo_new, bin_width_new, n_bins_new)
    scaled_S = centers_new * (N_prev / N_new)
    ln_g_interp = np.interp(scaled_S, centers_prev, ln_g_prev, left=ln_g_prev[0], right=ln_g_prev[-1])
    return ln_g_interp * (N_new / N_prev)


def trim_unreachable_lower_bins(ln_g, H, ever_visited, bin_lo, bin_width, n_bins, bin_idx):
    """Drop bins strictly below the lowest-ever-visited bin -- they
    have carried zero weight through the whole WL run so far, so
    removing them changes nothing about the physics, only the size of
    the arrays. (Checked, not assumed, for the N=40 pilot: the
    reweighted P(S) at 2x the predicted beta_c does NOT go negligible
    until essentially the lowest ever-visited bin itself -- the
    "hard wall" and the physically relevant cold-phase region turned
    out to coincide there, so trimming any further than this provably-
    dead region would cut real probability mass, not slack.)
    """
    idx = np.flatnonzero(ever_visited)
    lo = int(idx[0]) if idx.size else 0
    if lo == 0:
        return ln_g, H, ever_visited, bin_lo, n_bins, bin_idx
    new_bin_lo = bin_lo + lo * bin_width
    return ln_g[lo:], H[lo:], ever_visited[lo:], new_bin_lo, n_bins - lo, bin_idx - lo


MIN_BARRIER_FOR_BIMODAL = 0.5  # ln(peak/valley) below this is noise on a single hump, not two phases


def _find_two_peaks(p):
    """Local maxima of a 1D array, merging maxima within 2 bins of each
    other (same approach as study.py's _histogram_bimodality) and
    returning the two tallest, in bin-index order."""
    peaks = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > 0]
    merged = []
    for i in sorted(peaks, key=lambda k: -p[k]):
        if all(abs(i - m) > 2 for m in merged):
            merged.append(i)
    if len(merged) < 2:
        return None
    top2 = sorted(sorted(merged, key=lambda k: -p[k])[:2])
    return top2


def _peak_diagnostics(p, min_barrier=MIN_BARRIER_FOR_BIMODAL):
    """Two-peak diagnostics for one P_beta(S) array, or None if either
    no two-peak structure exists or the dip between the two tallest
    local maxima is too shallow to be anything but noise riding on a
    single broad hump (see MIN_BARRIER_FOR_BIMODAL's docstring note
    above and the N=30/eps=0.5 pilot's reanalysis: a naive peak finder
    with no depth check located "double peaks" at 29/800 scanned betas
    with barriers up to only 0.068 -- a <7% dip, not a first-order
    signature -- entirely inside what was really one broad hump
    riding on an under-converged ln_g).
    """
    peaks = _find_two_peaks(p)
    if peaks is None:
        return None
    i1, i2 = peaks
    valley = i1 + int(np.argmin(p[i1:i2 + 1]))
    valley_val = float(p[valley])
    peak_val = float(0.5 * (p[i1] + p[i2]))
    if valley_val <= 0:
        return None
    barrier = float(np.log(peak_val / valley_val))
    if barrier < min_barrier:
        return None
    return {
        "height_diff": p[i1] - p[i2], "peaks": peaks, "valley": valley,
        "peak_val": peak_val, "valley_val": valley_val, "barrier": barrier,
    }


def locate_beta_c_equal_height(ln_g, H_muca, bin_lo, bin_width, n_bins, beta_lo, beta_hi,
                                n_scan=400, tol=1e-5, max_iter=60, min_barrier=MIN_BARRIER_FOR_BIMODAL):
    """beta_c is where the two peaks of P_beta(S) have equal height
    (the standard MUCA criterion for a first-order transition's
    critical coupling -- Berg & Neuhaus 1991), using
    reweight_P_beta_corrected (folds in the actual production
    histogram H_muca, not just ln_g -- see its docstring for why that
    matters whenever WL hasn't fully converged).

    A genuine two-phase bimodal region can be a narrow sliver of
    [beta_lo, beta_hi] (observed: roughly 20% of a 10x-wide bracket at
    N=30, eps=0.5) that bisecting in from the bracket's own endpoints
    can step clean over. So this scans a fine linspace of n_scan
    points across the bracket first to find *a* significant two-peak
    point (surviving _peak_diagnostics's min_barrier filter), then
    bisects the height-difference sign change out from there --
    rather than bisecting from the original bracket endpoints inward.

    Returns None if no two-peak structure with barrier >= min_barrier
    is found anywhere on the scanned grid (not a genuine first-order
    signature at this resolution, or ln_g/H_muca aren't converged
    enough to resolve it) or if no sign change brackets a root.
    """
    def eval_beta(beta):
        p, _ = reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
        d = _peak_diagnostics(p, min_barrier=min_barrier)
        if d is None:
            return None
        return d["height_diff"], d["peaks"], p

    betas_scan = np.linspace(beta_lo, beta_hi, n_scan)
    scanned = [eval_beta(b) for b in betas_scan]
    found_idx = [i for i, r in enumerate(scanned) if r is not None]
    if not found_idx:
        return None

    bracket = None
    for k in range(len(found_idx) - 1):
        a, b = found_idx[k], found_idx[k + 1]
        if b != a + 1:
            continue  # grid gap -- not a contiguous bimodal region, don't bridge it
        if scanned[a][0] * scanned[b][0] <= 0:
            bracket = (betas_scan[a], betas_scan[b])
            break
    if bracket is None:
        return None
    lo, hi = bracket
    d_lo, d_hi = eval_beta(lo), eval_beta(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        d_mid = eval_beta(mid)
        if d_mid is None:
            return None
        if abs(d_mid[0]) < tol or (hi - lo) < 1e-8:
            peaks, p = d_mid[1], d_mid[2]
            diag = _peak_diagnostics(p, min_barrier=min_barrier)
            return {
                "beta_c": mid, "peak_height": diag["peak_val"], "valley": diag["valley_val"],
                "barrier": diag["barrier"], "peak_bins": peaks,
            }
        if d_mid[0] * d_lo[0] > 0:
            lo, d_lo = mid, d_mid
        else:
            hi, d_hi = mid, d_mid
    return None


# ----------------------------------------------------------------- calibration

def run_muca_one(N, eps, seed, target_round_trips=10, checkpoint_dir=None, verbose=True):
    """WL + MUCA production + beta_c location for one (N, eps, seed).

    Locates beta_c two ways:
      - locate_beta_c_variance_peak: the specific-heat/action-variance
        maximum, which is the definition Glaser, O'Connor & Surya
        (2018) actually fit their beta_c(N, eps) formula to (their
        Sec. 4.1: "the location of the maxima of ... the specific
        heat"). This is the PRIMARY number compared against the
        formula below -- comparing anything else to their formula
        would not be like-with-like.
      - locate_beta_c_equal_height: the equal-peak-height double-peak
        criterion, which is this project's own first-order diagnostic
        (part 5's pass criteria) -- it can legitimately find nothing
        (None) at an N where the transition is still crossover-like or
        too weak to resolve with the available statistics, without
        that implying the variance-peak location above is wrong (see
        the N=30, eps=0.5 pilot's reanalysis: variance peak agreed
        with the old PT study's independent measurement to within
        0.0001, while equal-height found no barrier above noise level
        anywhere in a wide scan).

    The equal-height search bracket (0.3x-3x the formula's prediction)
    is narrower than the variance-peak one (0.1x-5x) since the
    variance peak can genuinely sit further from the formula's
    leading-order-dominated prediction at small N (finite-size terms)
    and finding the wrong one of several local maxima by bracketing
    too tight would silently bias the primary comparison.
    """
    label = f"N={N} eps={eps} seed={seed}"
    predicted = beta_c_glaser_2018(N, eps)
    ckpt_wl = f"{checkpoint_dir}/wl_N{N}_eps{eps}_seed{seed}.pkl" if checkpoint_dir else None
    ckpt_muca = f"{checkpoint_dir}/muca_N{N}_eps{eps}_seed{seed}.pkl" if checkpoint_dir else None

    t0 = time.time()
    wl = run_wang_landau(N, eps, seed=seed, checkpoint_path=ckpt_wl, verbose=verbose, verbose_label=label)
    t_wl = time.time() - t0
    t0 = time.time()
    prod = run_muca_production(wl, target_round_trips=target_round_trips, checkpoint_path=ckpt_muca,
                                verbose=verbose, verbose_label=label)
    t_muca = time.time() - t0
    beta_c_var_res = locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.1 * predicted, 5.0 * predicted,
    )
    beta_c_res = locate_beta_c_equal_height(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.3 * predicted, 3.0 * predicted,
    )
    if verbose:
        print(f"[calib] {label} beta_c_var={beta_c_var_res['beta_c']:.5f} "
              f"(interior={beta_c_var_res['is_interior']}) beta_c_equal_height={beta_c_res} "
              f"predicted={predicted:.5f} (WL {t_wl:.1f}s, MUCA {t_muca:.1f}s)", flush=True)
    return {
        "N": N, "eps": eps, "seed": seed, "predicted": predicted,
        "wl": wl, "prod": prod, "beta_c_res": beta_c_res, "beta_c_var_res": beta_c_var_res,
        "t_wl": t_wl, "t_muca": t_muca,
    }


def summarize_one_N_eps(results, target_round_trips=10):
    """Aggregate run_muca_one results across seeds for one (N, eps):
    mean/std beta_c (variance-peak definition -- see run_muca_one) is
    the primary number compared against the published formula; the
    equal-height double-peak diagnostics are reported separately as
    the first-order-transition signature (part 5's pass criteria)."""
    N, eps = results[0]["N"], results[0]["eps"]
    predicted = results[0]["predicted"]

    var_betas = [r["beta_c_var_res"]["beta_c"] for r in results if r["beta_c_var_res"]["is_interior"]]
    beta_c_mean = float(np.mean(var_betas)) if var_betas else float("nan")
    beta_c_std = float(np.std(var_betas)) if len(var_betas) > 1 else 0.0
    variance_peak_interior_all = all(r["beta_c_var_res"]["is_interior"] for r in results)

    betas = [r["beta_c_res"]["beta_c"] for r in results if r["beta_c_res"] is not None]
    barriers = [r["beta_c_res"]["barrier"] for r in results if r["beta_c_res"] is not None]
    barrier_mean = float(np.mean(barriers)) if barriers else float("nan")
    double_peak_found_all = all(r["beta_c_res"] is not None for r in results)

    round_trips_ok = all(r["prod"]["round_trips"] >= target_round_trips for r in results)
    min_round_trips = min(r["prod"]["round_trips"] for r in results)
    hidden_barrier_any = any(r["prod"]["hidden_barrier_warning"] for r in results)
    return {
        "N": N, "eps": eps, "predicted": predicted,
        "beta_c_mean": beta_c_mean, "beta_c_std": beta_c_std,
        "variance_peak_interior_all": variance_peak_interior_all,
        "equal_height_beta_c_mean": float(np.mean(betas)) if betas else float("nan"),
        "barrier_mean": barrier_mean,
        "n_seeds": len(results), "n_seeds_with_double_peak": len(betas),
        "round_trips_ok": round_trips_ok, "min_round_trips": min_round_trips,
        "hidden_barrier_any": hidden_barrier_any,
        "double_peak_found_all": double_peak_found_all,
        "results": results,
    }


BETA_C_Z_PASS = 3.0


def _beta_c_formula_uncertainty(N, eps):
    sigma_b = 0.03 / eps ** 2
    sigma_c = float(np.hypot(0.50 / eps ** 3, 2.45 / eps ** 2))
    return sigma_b / N + sigma_c / N ** 2


def run_phase3b_calibration(study_ns, eps_values, seeds=(0, 1, 2), target_round_trips=10,
                             checkpoint_dir=None, verbose=True):
    """Orchestrates run_muca_one across every (eps, N, seed), then
    summarize_one_N_eps per (eps, N). Does not write the report or
    plots itself (see write_phase3b_report) so this can be called in
    smaller pieces (e.g. one N at a time) without forcing a full
    re-write of partial results.
    """
    summaries = {}
    for eps in eps_values:
        for N in study_ns:
            t0 = time.time()
            results = [
                run_muca_one(N, eps, seed, target_round_trips=target_round_trips,
                              checkpoint_dir=checkpoint_dir, verbose=verbose)
                for seed in seeds
            ]
            summaries[(eps, N)] = summarize_one_N_eps(results, target_round_trips=target_round_trips)
            if verbose:
                s = summaries[(eps, N)]
                print(f"[calib] N={N} eps={eps} done in {time.time()-t0:.1f}s: "
                      f"beta_c={s['beta_c_mean']:.5f}+/-{s['beta_c_std']:.5f} "
                      f"(predicted {s['predicted']:.5f}), double_peak={s['n_seeds_with_double_peak']}/{s['n_seeds']}, "
                      f"min_round_trips={s['min_round_trips']}, hidden_barrier_any={s['hidden_barrier_any']}", flush=True)
    return summaries


def write_phase3b_report(summaries, study_ns, eps_values, side_note_eps=None, side_note_betas=None):
    """side_note_eps/side_note_betas: part 6's side note -- report
    whatever reweighted data shows at specific betas for one eps
    (e.g. eps=0.21 around beta~7, the spurious high-beta feature the
    Phase 3 blind-centering check found) without tuning anything
    around it. Pass side_note_eps=0.21, side_note_betas=[7.0] (etc) to
    include; omitted if not given.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 3b: fast incremental updates + multicanonical sampling")
    lines.append("")
    lines.append(
        "Wang-Landau estimates the action density of states g(S) by flattening the visit "
        "histogram across the whole binned range in one run; multicanonical (MUCA) production "
        "then samples with that converged weight fixed, and the single run is reweighted to "
        "P_beta(S) and <O>(beta) at any beta (Berg & Neuhaus, Phys. Lett. B 267 (1991) 249). This "
        "replaces the beta-by-beta parallel-tempering scan from the earlier Phase 3 calibration, "
        "which found most located beta_c values were right-censored and, at eps=0.21, not even "
        "stable under a blind grid-recentering test. See `mcmc/bitset_core.py`, `mcmc/muca_core.py`, "
        "`mcmc/muca.py` for the engine; `mcmc/checkpoint.py` for checkpointing; "
        "`tests/test_bitset_core.py` for the incremental-update correctness/speed check."
    )
    lines.append("")

    all_keys = [(eps, N) for eps in eps_values for N in study_ns]
    n_total = len(all_keys)

    lines.append("## Calibration results")
    lines.append("")
    lines.append(
        "beta_c (compared against the formula) is the location of the reweighted specific-heat / "
        "action-variance maximum -- the definition Glaser, O'Connor & Surya (2018) actually fit "
        "their formula to ('we estimate beta_c as the location of the maxima of ... the specific "
        "heat', their Sec. 4.1), not the equal-peak-height criterion (the two coincide for a "
        "strong first-order transition but can differ for a weak one; using equal-height here "
        "would not be a like-with-like comparison). Averaged per seed; the spread across seeds is "
        "the reported uncertainty. Compared against Glaser-O'Connor-Surya 2018's beta_c(N,eps) = "
        "1.66(+/-0.03)/(N*eps^2) + [4.09(+/-0.50)/eps^3 - 27.77(+/-2.45)/eps^2]/N^2. The separate "
        "'double peak' / 'barrier' columns use the equal-peak-height criterion instead -- that one "
        "is this project's own first-order-transition diagnostic (part 5), and finding none at a "
        "given N does not contradict a variance-peak beta_c that otherwise matches the formula; it "
        "means the transition (if first-order at all) is not yet resolved as two separable peaks "
        "at that N and statistics."
    )
    lines.append("")
    lines.append("| eps | N | beta_c (variance peak) | formula beta_c | z | double peak (equal-height) | barrier ln(P_max/P_min) | min round trips | hidden barrier |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    n_betac_pass = n_bimodal = n_rt_pass = n_no_hidden = 0
    for eps in eps_values:
        for N in study_ns:
            s = summaries[(eps, N)]
            formula_sigma = _beta_c_formula_uncertainty(N, eps)
            combined_sigma = float(np.hypot(s["beta_c_std"], formula_sigma))
            z = abs(s["beta_c_mean"] - s["predicted"]) / combined_sigma if combined_sigma > 0 and np.isfinite(s["beta_c_mean"]) else float("nan")
            betac_pass = np.isfinite(z) and z < BETA_C_Z_PASS
            n_betac_pass += int(betac_pass)
            n_bimodal += int(s["double_peak_found_all"])
            n_rt_pass += int(s["round_trips_ok"])
            n_no_hidden += int(not s["hidden_barrier_any"])
            lines.append(
                f"| {eps} | {N} | {s['beta_c_mean']:.5f} +/- {s['beta_c_std']:.5f} | "
                f"{s['predicted']:.5f} +/- {formula_sigma:.5f} | {z:.2f} | "
                f"{s['n_seeds_with_double_peak']}/{s['n_seeds']} | {s['barrier_mean']:.3f} | "
                f"{s['min_round_trips']} | {'YES' if s['hidden_barrier_any'] else 'no'} |"
            )
    lines.append("")

    for eps in eps_values:
        Ns_arr = np.array(study_ns, dtype=float)
        barriers = np.array([summaries[(eps, N)]["barrier_mean"] for N in study_ns])
        valid = np.isfinite(barriers) & (barriers > 0)
        if valid.sum() >= 2:
            slope = float(np.polyfit(np.log(Ns_arr[valid]), np.log(barriers[valid]), 1)[0])
            lines.append(
                f"eps={eps}: free-energy barrier scales as N^{slope:.2f} across N={study_ns} "
                f"({'growing' if slope > 0 else 'not growing'} with N -- growth confirms a "
                "genuine first-order transition, per part 5's pass criterion)."
            )
        else:
            lines.append(f"eps={eps}: fewer than 2 N values found a two-peak structure; cannot assess barrier scaling with N.")
    lines.append("")

    lines.append("## Pass criteria (part 5)")
    lines.append("")
    lines.append(f"1. **Clean double peak at beta_c that separates with N**: {n_bimodal} / {n_total} combinations found a two-peak P_beta_c(S) in every seed (see barrier-scaling lines above for the separating-with-N check).")
    lines.append(f"2. **beta_c within error of the formula** (z < {BETA_C_Z_PASS}): {n_betac_pass} / {n_total}.")
    lines.append(f"3. **At least {10} round trips per run**: {n_rt_pass} / {n_total}.")
    lines.append(f"4. **No hidden-barrier warning**: {n_no_hidden} / {n_total}.")
    lines.append("")
    if n_betac_pass < n_total or n_bimodal < n_total or n_rt_pass < n_total or n_no_hidden < n_total:
        lines.append("**Not every criterion is met for every (N, eps) -- see the table above for exactly which, and why (z-score, missing double peak, round trip count, or hidden-barrier flag).**")
    else:
        lines.append("**All four pass criteria are met across every (N, eps) tested.**")
    lines.append("")

    if side_note_eps is not None and side_note_betas:
        lines.append(f"## Side note: eps={side_note_eps} near beta~{side_note_betas} (not tuned around; reported as found)")
        lines.append("")
        for N in study_ns:
            key = (side_note_eps, N)
            if key not in summaries:
                continue
            s = summaries[key]
            r0 = s["results"][0]
            wl, prod = r0["wl"], r0["prod"]
            for beta in side_note_betas:
                p, centers = reweight_P_beta_corrected(wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], beta)
                mean_S = float(np.sum(centers * p))
                var_S = float(np.sum((centers - mean_S) ** 2 * p))
                height_obs = reweight_observable(r0["prod"]["rec_struct_bin"], r0["prod"]["rec_height"],
                                                  wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], beta)
                of_obs = reweight_observable(r0["prod"]["rec_struct_bin"], r0["prod"]["rec_of"],
                                              wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], beta)
                lines.append(
                    f"- N={N}, beta={beta}: <S>={mean_S:.2f}, Var(S)={var_S:.2f}, "
                    f"<height>={height_obs:.2f}, <ordering_fraction>={of_obs:.3f} (reweighted from the "
                    "single eps=0.21 MUCA run, seed 0 -- not a new sample targeted at this beta)."
                )
        lines.append("")

    report_path = RESULTS_DIR / "muca_calibration_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[calib] wrote {report_path}", flush=True)
    return report_path


def generate_phase3b_plots(summaries, study_ns, eps_values):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for eps in eps_values:
        cases = {}
        for N in study_ns:
            s = summaries[(eps, N)]
            if not np.isfinite(s["beta_c_mean"]):
                continue
            r0 = s["results"][0]
            wl, prod = r0["wl"], r0["prod"]
            p, centers = reweight_P_beta_corrected(wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], s["beta_c_mean"])
            cases[f"N={N}"] = {"S": centers, "P": p}
        if cases:
            tag = f"eps{eps:.2f}".replace(".", "p")
            plots.plot_P_beta_S(
                cases, str(RESULTS_DIR / f"muca_P_beta_c_S_{tag}.png"),
                title=f"P_beta_c(S) at the located transition (eps={eps})",
            )

        barrier_case = {
            "barrier": {
                "beta": study_ns,
                "mean": [summaries[(eps, N)]["barrier_mean"] for N in study_ns],
            },
        }
        plots.plot_vs_beta(
            barrier_case, str(RESULTS_DIR / f"muca_barrier_vs_N_{tag}.png"),
            ylabel="ln(P_max / P_min_between_peaks)", title=f"Free-energy barrier vs N (eps={eps})",
            xlabel="N", xscale="log",
        )
