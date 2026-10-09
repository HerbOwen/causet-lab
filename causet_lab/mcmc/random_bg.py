"""Random-background causal sets on a d=2 cylinder (Phase 5), the
quenched-disorder comparison to the Phase 4 regular-lattice model
(Cunningham & Surya, arXiv:1908.11647 -- "C&S"). See random_bg_core.py's
module docstring for the construction and, importantly, the units note:
the sprinkled spatial coordinate x has period w (the lattice's own
integer circumference, same units as t), NOT theta in radians -- this
makes the random background's causal geometry identical to the regular
lattice's, not merely "the same physical region" in a different metric
normalization. Verified directly in test_random_bg.py by placing a
background at the exact lattice positions and checking bit-for-bit
agreement with lattice_gas.lattice_to_matrix.

This project's own construction, built because C&S's own Sec. 6 flags
"a more realistic model of discreteness" as an open question they did
not pursue, not because they specify one.

Sizing: identical region to Phase 4 -- w, h, m = lattice_gas.lattice_dims(n,
alpha), alpha=4, eps=0.1. The background (site_t, site_x) is quenched --
fixed for an entire WL+MUCA run, generated once per (n, background_seed)
by sprinkle_background and threaded through every call below rather
than regenerated. The background seed and the Metropolis-chain seed are
always distinct parameters (never derived from one another), so e.g.
five background realizations can each be paired with the same or
different chain seeds independently.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from numba import njit

from . import random_bg_core as rbg
from . import lattice_gas as lgm
from . import rng
from .action import f2_smear_table
from .checkpoint import (
    save_checkpoint, load_checkpoint, make_randombg_wl_state, make_randombg_muca_state,
)
from ..measures import height as measure_height
from .muca import (
    N_BINS_INITIAL, RANGE_MARGIN_FRAC, PROGRESS_EVERY_S, F_INITIAL, F_FINAL,
    FLATNESS_THRESHOLD, EDGE_HIT_FRACTION_TRIGGER, STALL_SECONDS,
    MUCA_MEASURE_EVERY, HIDDEN_BARRIER_RATIO_THRESHOLD, _widen,
    check_ln_g_anomalies,
)

ALPHA_DEFAULT = 4
EPS_DEFAULT = 0.1


def background_dims(n: int, alpha: int = ALPHA_DEFAULT) -> Tuple[int, int, int]:
    """w, h, m for an n-element random-background filling -- literally
    lattice_gas.lattice_dims: same region, same units (see module
    docstring), just sprinkled instead of gridded."""
    return lgm.lattice_dims(n, alpha)


def sprinkle_background(m: int, w: int, h: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The quenched random background: m points independently uniform
    in the (t, x) rectangle t in [0, h], x in [0, w) -- x has period w
    (lattice-spacing units), matching lattice_gas_core.lattice_precedes's
    own units exactly (see random_bg_core.py's module docstring), not
    radians."""
    rng_np = np.random.default_rng(seed)
    site_t = rng_np.uniform(0.0, float(h), size=m)
    site_x = rng_np.uniform(0.0, float(w), size=m)
    return site_t, site_x


def randombg_to_matrix(L: np.ndarray, n: int, site_t: np.ndarray, site_x: np.ndarray, w: int) -> np.ndarray:
    """Full (n, n) boolean relation matrix -- continuous-position twin
    of lattice_gas.lattice_to_matrix, same units (period w, not 2*pi)."""
    t = site_t[L[:n]]
    x = site_x[L[:n]]
    C = np.zeros((n, n), dtype=bool)
    for a in range(n):
        dt = t - t[a]
        dx = (x - x[a]) % w
        dx = np.minimum(dx, w - dx)
        C[a, :] = (dt > 0) & (dx <= dt)
    np.fill_diagonal(C, False)
    return C


def layer_count_randombg(L: np.ndarray, n: int, site_t: np.ndarray, h: int) -> int:
    """Number of distinct unit-width t-bins (matching the regular
    lattice's integer row spacing, for an apples-to-apples comparison
    to lattice_gas.layer_count) occupied by the n elements."""
    t = site_t[L[:n]]
    rows = np.clip(np.floor(t).astype(np.int64), 0, h)
    return int(np.unique(rows).size)


def randombg_action_full(L: np.ndarray, n: int, site_t: np.ndarray, site_x: np.ndarray, w: int,
                          eps: float) -> Tuple[float, np.ndarray]:
    """Full non-incremental recomputation of C&S's Eq. 8 action --
    continuous-position twin of lattice_gas.lattice_gas_action_full,
    ground truth for apply_relocate_randombg's incremental update."""
    C = randombg_to_matrix(L, n, site_t, site_x, w)
    total_relations = int(C.sum())
    Nk = np.zeros(max(n - 1, 1), dtype=np.int64)
    if total_relations > 0:
        Cf = C.astype(np.float32)
        sizes = np.rint(Cf @ Cf).astype(np.int64)
        related_sizes = sizes[C]
        Nk = np.bincount(related_sizes, minlength=max(n - 1, 1))[: max(n - 1, 1)].astype(np.int64)
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    S = rbg.lattice_gas_action_from_counts(n, Nk, f2_table, eps)
    return S, Nk


@njit(cache=True)
def _pilot_sweeps_randombg_jit(L, future, past, counts, n, m, site_t, site_x, w, beta, eps, f2_table,
                                n_sweeps, burn_in, measure_every, anneal_from, has_anneal,
                                rng_state, max_measurements):
    """Random-background twin of lattice_gas._pilot_sweeps_jit."""
    S = rbg._lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
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
            rbg.apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
            S_new = rbg._lattice_gas_action_from_counts_jit(n, counts, f2_table, eps)
            dS = S_new - S
            accept = dS <= 0.0
            if not accept:
                rng_state, r = rng.next_double(rng_state)
                accept = r < np.exp(-beta_eff * dS)
            if accept:
                S = S_new
            else:
                rbg.apply_relocate_randombg_jit(L, future, past, counts, n, site_t, site_x, w, i, j)
        if sweep >= burn_in and (sweep - burn_in) % measure_every == 0 and n_actions < max_measurements:
            actions[n_actions] = S
            n_actions += 1
    return actions[:n_actions], rng_state


def run_pilot_chain_randombg(n, w, h, m, eps, site_t, site_x, beta, seed, n_sweeps, burn_in,
                              measure_every=5, anneal_from=None):
    """Python wrapper around _pilot_sweeps_randombg_jit -- fresh random
    filling (seed) on the given fixed background (site_t, site_x),
    returns the measured action time series. seed here is the chain
    seed, independent of whatever background_seed produced site_t/site_x.
    """
    L = lgm.random_filling(n, m, seed)
    future, past, counts = rbg.build_randombg_state_bitset(L, n, site_t, site_x, w)
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    rng_state = rng.seed_state(seed)
    max_meas = n_sweeps // measure_every + 2
    has_anneal = anneal_from is not None
    actions, rng_state = _pilot_sweeps_randombg_jit(
        L, future, past, counts, n, m, site_t, site_x, w, beta, eps, f2_table,
        n_sweeps, burn_in, measure_every, anneal_from if has_anneal else 0.0, has_anneal,
        rng_state, max_meas,
    )
    return actions, L, future, past, counts


# ----------------------------------------------------------- WL / MUCA

HOT_QUANTILE = 0.9999
COLD_QUANTILE = 0.0001
COLD_PILOT_BETA_MULTIPLIER = 3.5  # not the old 15x -- see module note below
FIXED_WINDOW_MARGIN_FRAC = 0.05  # small safety buffer; this window does NOT widen during WL,
                                 # so it must be right from a single pilot estimate, not patched later


def estimate_initial_bin_range_randombg(n, w, h, m, eps, site_t, site_x, seed=0, beta_hint=None,
                                         hot_pilot_sweeps=4000, cold_pilot_sweeps=4000):
    """Random-background twin of lattice_gas.estimate_initial_bin_range_lattice
    -- except this is now a FIXED window, not a starting guess WL is free
    to widen later (see run_wang_landau_randombg: edge hits are rejected,
    not used to trigger widening, past this point).

    Why fixed rather than adaptive: widening assumes an edge-hit-prone
    bin is simply under-explored and will catch up given more ln_g
    increments. Direct diagnosis of an n=50 random-background run found
    a case where that assumption fails -- a bin that is not low-entropy
    but *kinetically* hard to reach (few single-site-relocation paths
    lead there from the bulk), which WL's own histogram-flattening
    heuristic reads as "needs more weight" and never actually succeeds
    at fixing, since the bottleneck is reachability, not acceptance
    probability. No amount of smarter bin-seeding (see _widen's
    gradient-extrapolation fix) can repair that; the only robust fix is
    to keep the explored window away from that region entirely, since
    nothing at beta in [0, a few*beta_c] ever needs to sample there.

    Both ends are therefore set from QUANTILES of reasonably large pilot
    samples (not a single pilot's max/min, which is itself a noisy
    extreme-value statistic), with only a small fixed safety margin:
    hot end = HOT_QUANTILE of a beta=0 sample (nothing beta>=0 needs to
    go hotter than beta=0 itself reaches with non-negligible
    probability); cold end = COLD_QUANTILE of a sample annealed to
    COLD_PILOT_BETA_MULTIPLIER * beta_hint (deliberately deeper than the
    beta~2x*beta_c range production actually needs to resolve well, for
    headroom) -- not the old 15x/max-based approach, which both
    overshot the needed range and still wasn't reliably wide enough at
    the extremes it was actually used for. Callers should check the
    result with verify_window_coverage_randombg before trusting it for
    a production run on a new background.
    """
    guess = beta_hint if beta_hint is not None else lgm.beta_c_guess(n)
    hot, *_ = run_pilot_chain_randombg(n, w, h, m, eps, site_t, site_x, beta=0.0, seed=seed,
                                        n_sweeps=hot_pilot_sweeps, burn_in=100, measure_every=4)
    cold, *_ = run_pilot_chain_randombg(n, w, h, m, eps, site_t, site_x, beta=guess * COLD_PILOT_BETA_MULTIPLIER,
                                         seed=seed, n_sweeps=cold_pilot_sweeps, burn_in=300, measure_every=4,
                                         anneal_from=0.0)
    s_hi = float(np.quantile(hot, HOT_QUANTILE))
    s_lo = float(np.quantile(cold, COLD_QUANTILE))
    if s_lo > s_hi:
        s_lo, s_hi = min(s_lo, s_hi) - 1.0, max(s_lo, s_hi) + 1.0
    span = s_hi - s_lo
    margin = FIXED_WINDOW_MARGIN_FRAC * span
    return s_lo - margin, s_hi + margin


def verify_window_coverage_randombg(ln_g, H_muca, bin_lo, bin_width, n_bins, beta_c,
                                     beta_fracs=(0.0, 0.5, 1.0, 1.5, 2.0), edge_bins=3, max_edge_mass=1e-3):
    """Check that the fixed WL/MUCA window actually contains everything
    that matters for beta in [0, 2*beta_c]: the reweighted P_beta(S)
    should carry negligible mass (< max_edge_mass) in the first/last
    edge_bins bins at every beta checked. Returns a dict with the
    per-beta edge masses and an overall `ok` flag -- meant to be called
    and reported, not just trusted silently, since this window no
    longer self-corrects via widening if it was set too tight.
    """
    from .muca import reweight_P_beta_corrected
    results = []
    ok = True
    for frac in beta_fracs:
        beta = frac * beta_c
        p, _centers = reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
        lo_mass = float(p[:edge_bins].sum())
        hi_mass = float(p[-edge_bins:].sum())
        this_ok = lo_mass < max_edge_mass and hi_mass < max_edge_mass
        ok = ok and this_ok
        results.append({"beta": beta, "beta_frac_of_c": frac, "lo_edge_mass": lo_mass,
                         "hi_edge_mass": hi_mass, "ok": this_ok})
    return {"ok": ok, "per_beta": results}


def run_wang_landau_randombg(n, eps, site_t, site_x, alpha=ALPHA_DEFAULT, seed=0, checkpoint_path=None,
                              checkpoint_every_s=60, chunk_moves=None, verbose=True, verbose_label=None,
                              beta_hint=None, stall_shortcut_seconds=None, stall_shortcut_f=1e-2,
                              allow_widen=False):
    """Random-background twin of lattice_gas.run_wang_landau_lattice --
    identical algorithm, site_t/site_x (the fixed quenched background)
    threaded through, and carried in the returned dict/checkpoint so
    MUCA production can reuse them. seed is the chain seed only; the
    background is supplied by the caller (from sprinkle_background with
    its own, separate background_seed).

    stall_shortcut_seconds, if set, cuts the recursion short once f_mod
    has already reached stall_shortcut_f (default 1e-2, i.e. about 7 of
    the usual ~20 halvings) and then goes stall_shortcut_seconds with no
    further stage advance -- relying on MUCA's corrected (Berg-Neuhaus)
    reweighting, which stays exact even with an imperfect ln_g as long as
    the production run visits the compared region, to make up the
    difference. Returns used_stall_shortcut=True when this fires. Never
    fires while check_ln_g_anomalies flags a live cliff in the explored
    range, in addition to the existing diversity/no-confinement gate --
    a walk can be "ranging across many bins" and still have one
    badly-skewed bin pair, which diversity alone does not catch (found
    directly: an n=50 run ranged across 94/101 bins with a -58997 step
    present).

    allow_widen (default False): estimate_initial_bin_range_randombg
    now returns a FIXED window from pilot quantiles, not a starting
    guess meant to be grown. Edge-hit pressure past that window is
    rejected (as it already was for any out-of-range move), not treated
    as a signal to widen -- direct diagnosis found that widening could
    recreate the same cliff/kinetic-trap pathology it was meant to
    avoid, even with a corrected (gradient, direction- and
    magnitude-clamped) _widen. Set True only to reproduce the old
    adaptive behavior.
    """
    w, h, m = background_dims(n, alpha)
    label = verbose_label or f"randombg n={n} eps={eps}"
    t0 = time.time()
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * n

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "randombg_wl":
        L = ckpt["L"]
        site_t, site_x = ckpt["site_t"], ckpt["site_x"]
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
            print(f"[randombg-wl] {label} resumed from checkpoint at stage={stage}, f={f_mod:.2e}", flush=True)
    else:
        lo, hi = estimate_initial_bin_range_randombg(n, w, h, m, eps, site_t, site_x, seed, beta_hint)
        n_bins = N_BINS_INITIAL
        bin_width = (hi - lo) / n_bins
        bin_lo = lo
        ln_g = np.zeros(n_bins, dtype=np.float64)
        H = np.zeros(n_bins, dtype=np.int64)
        rng_state = rng.seed_state(seed)
        L = lgm.random_filling(n, m, seed)
        future, past, counts = rbg.build_randombg_state_bitset(L, n, site_t, site_x, w)
        f_mod, stage = F_INITIAL, 0
        S0 = rbg.lattice_gas_action_from_counts(n, counts, f2_table, eps)
        bin_idx = rbg.bin_index(S0, bin_lo, bin_width, n_bins)
        if bin_idx < 0:
            bin_idx = n_bins // 2
        extreme_state = np.zeros(1, dtype=np.int64)
        half_trips = np.zeros(1, dtype=np.int64)
        edge_hits_total = 0
        ever_visited = np.zeros(n_bins, dtype=bool)
        full_coverage_required = True
        if verbose:
            print(f"[randombg-wl] {label} starting fresh: range=[{lo:.3f}, {hi:.3f}], n_bins={n_bins}", flush=True)

    last_ckpt = time.time()
    last_progress = time.time()
    last_growth = time.time()
    last_stage_change = time.time()
    last_confined = time.time()
    n_ever_visited_prev = int(ever_visited.sum())
    total_moves = 0
    edge_warned = [False]
    stage_moves_log = [(stage, total_moves)]
    used_stall_shortcut = False

    def _reach_extremes():
        if ever_visited.any():
            idx = np.flatnonzero(ever_visited)
            return int(idx[0]), int(idx[-1])
        return bin_idx, bin_idx

    lo_extreme, hi_extreme = _reach_extremes()
    while f_mod > F_FINAL:
        rng_state, bin_idx, edge_hits = rbg.wl_or_muca_sweep_chunk_randombg_jit(
            L, future, past, counts, n, m, site_t, site_x, w, eps, f2_table, ln_g, H,
            bin_lo, bin_width, n_bins, f_mod, True, chunk_moves,
            rng_state, bin_idx, extreme_state, half_trips, 0, lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        edge_hits_total += edge_hits
        total_moves += chunk_moves

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            n_visited = int((H > 0).sum())
            print(f"[randombg-wl] {label} stage={stage} f={f_mod:.2e} moves={total_moves} "
                  f"visited={n_visited}/{n_bins} bin_idx={bin_idx} "
                  f"edge_hits_total={edge_hits_total} ({time.time()-t0:.1f}s)", flush=True)
            last_progress = time.time()
            if verbose:
                for bin_a, bin_b, step in check_ln_g_anomalies(ln_g, ever_visited):
                    print(f"[randombg-wl] {label} WARNING: anomalous ln_g step at bins "
                          f"{bin_a}->{bin_b}: {step:.2f} (ln_g={ln_g[bin_a]:.2f} -> {ln_g[bin_b]:.2f}) "
                          f"-- likely trap risk ({time.time()-t0:.1f}s)", flush=True)

        if edge_hits > EDGE_HIT_FRACTION_TRIGGER * chunk_moves:
            if allow_widen:
                side = "low" if bin_idx < n_bins // 2 else "high"
                old_n_bins = n_bins
                ln_g, H, bin_lo, n_bins = _widen(ln_g, H, bin_lo, bin_width, n_bins, side)
                pad = np.zeros(n_bins - old_n_bins, dtype=bool)
                ever_visited = np.concatenate([pad, ever_visited]) if side == "low" else np.concatenate([ever_visited, pad])
                bin_idx = rbg.bin_index(
                    rbg.lattice_gas_action_from_counts(n, counts, f2_table, eps), bin_lo, bin_width, n_bins,
                )
                lo_extreme, hi_extreme = _reach_extremes()
                if verbose:
                    print(f"[randombg-wl] {label} widened range on the {side} side -> n_bins={n_bins}", flush=True)
                continue
            elif verbose and not edge_warned[0]:
                # Fixed window by design (see estimate_initial_bin_range_randombg):
                # edge pressure here means the move was rejected and the walker
                # stayed put, not that the window needs to grow -- widening was
                # found to recreate the same kinetic-bottleneck trap this window
                # is specifically meant to avoid. A one-time warning, not a
                # per-chunk spam, in case the a-priori window was genuinely too
                # tight (verify_window_coverage_randombg is the real check).
                print(f"[randombg-wl] {label} WARNING: sustained edge pressure "
                      f"(edge_hits_total={edge_hits_total}) against the fixed window -- "
                      f"not widening by design; verify window coverage before trusting "
                      f"this run if this persists ({time.time()-t0:.1f}s)", flush=True)
                edge_warned[0] = True

        ever_visited |= (H > 0)
        n_ever_visited = int(ever_visited.sum())
        if n_ever_visited > n_ever_visited_prev:
            n_ever_visited_prev = n_ever_visited
            last_growth = time.time()
            lo_extreme, hi_extreme = _reach_extremes()
        elif full_coverage_required and time.time() - last_growth > STALL_SECONDS:
            full_coverage_required = False
            if verbose:
                print(f"[randombg-wl] {label} stalled at {n_ever_visited}/{n_bins} bins after "
                      f"{STALL_SECONDS:.0f}s with no new bin visited -- treating the rest as "
                      f"unreachable for this n and proceeding on the observed support", flush=True)

        ready = (not full_coverage_required) or (n_ever_visited == n_bins)
        diverse_enough = False
        if ready:
            check_set = (H > 0) if full_coverage_required else ((H > 0) & ever_visited)
            diverse_enough = check_set.sum() >= max(3, int(0.5 * n_ever_visited))
            if diverse_enough:
                flatness = float(H[check_set].min()) / float(H[check_set].mean())
                if flatness >= FLATNESS_THRESHOLD:
                    stage += 1
                    f_mod = f_mod / 2.0
                    H[:] = 0
                    last_stage_change = time.time()
                    stage_moves_log.append((stage, total_moves))
                    if verbose:
                        print(f"[randombg-wl] {label} stage {stage}: flat (min/mean={flatness:.2f} over "
                              f"{int(check_set.sum())}/{n_ever_visited} reachable bins), "
                              f"f -> {f_mod:.2e} ({time.time()-t0:.1f}s)", flush=True)

        has_anomaly = bool(check_ln_g_anomalies(ln_g, ever_visited))
        if not diverse_enough or has_anomaly:
            last_confined = time.time()

        # Only ever shortcut a walk that has been diverse_enough (ranging
        # across a healthy fraction of its reachable bins, just not flat
        # enough yet) AND anomaly-free, continuously for the whole
        # stall_shortcut_seconds window -- never a walk that was confined
        # OR showed a live ln_g cliff at any point in that window, even if
        # it has just now recovered. A confined (or cliff-bearing) walk's
        # ln_g is actively correcting itself (the trapped bin's ln_g rises
        # relative to its neighbors until the walker is statistically
        # favored to leave); cutting WL short on or just after either would
        # freeze exactly the biased weights MUCA cannot then correct for.
        # diverse_enough alone is not sufficient: a walk can range across
        # most of its reachable bins and still carry one badly-skewed pair
        # (found directly: ranged across 94/101 bins with a -58997 ln_g
        # step still present).
        if (stall_shortcut_seconds is not None and f_mod <= stall_shortcut_f and diverse_enough
                and not has_anomaly
                and time.time() - last_stage_change > stall_shortcut_seconds
                and time.time() - last_confined > stall_shortcut_seconds):
            used_stall_shortcut = True
            if verbose:
                print(f"[randombg-wl] {label} stall shortcut: f={f_mod:.2e} <= {stall_shortcut_f:.2e}, "
                      f"walk has ranged across {int(check_set.sum())}/{n_ever_visited} reachable bins "
                      f"with no confinement and no flatness for {stall_shortcut_seconds:.0f}s "
                      f"-- stopping WL early, deferring to MUCA's corrected reweighting "
                      f"({time.time()-t0:.1f}s)", flush=True)
            break

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_randombg_wl_state(
                L, site_t, site_x, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
                n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
                ever_visited, full_coverage_required,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_randombg_wl_state(
            L, site_t, site_x, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width,
            n_bins, f_mod, stage, bin_idx, extreme_state, half_trips, edge_hits_total,
            ever_visited, full_coverage_required,
        ))

    return {
        "n": n, "eps": eps, "w": w, "h": h, "m": m, "N": n,
        "site_t": site_t, "site_x": site_x,
        "ln_g": ln_g, "bin_lo": bin_lo, "bin_width": bin_width,
        "n_bins": n_bins, "stages": stage, "edge_hits_total": edge_hits_total,
        "elapsed": time.time() - t0,
        "L": L, "future": future, "past": past, "counts": counts, "rng_state": rng_state,
        "bin_idx": bin_idx, "ever_visited": ever_visited, "n_reachable_bins": int(ever_visited.sum()),
        "used_stall_shortcut": used_stall_shortcut, "final_f_mod": f_mod,
        "stage_moves_log": stage_moves_log,
        "ln_g_anomalies": check_ln_g_anomalies(ln_g, ever_visited),
    }


def _measure_structural_randombg(L, n, site_t, site_x, w, counts):
    """height via measures.height; ordering_fraction read off the
    maintained interval-size histogram -- twin of
    lattice_gas._measure_structural_lattice."""
    C = randombg_to_matrix(L, n, site_t, site_x, w)
    h = measure_height(C)
    max_pairs = n * (n - 1) // 2
    of = float(np.sum(counts)) / max_pairs if max_pairs else float("nan")
    return h, of


def run_muca_production_randombg(wl_result, target_round_trips=10, max_moves=None,
                                  checkpoint_path=None, checkpoint_every_s=60, chunk_moves=None,
                                  round_trip_window=None, verbose=True, verbose_label=None):
    """Random-background twin of lattice_gas.run_muca_production_lattice
    -- same return-dict shape, so muca.reweight_P_beta_corrected,
    locate_beta_c_variance_peak, run_muca_recursion and
    run_muca_production_parallel all work on its output unchanged.
    """
    n, eps = wl_result["n"], wl_result["eps"]
    w = wl_result["w"]
    site_t, site_x = wl_result["site_t"], wl_result["site_x"]
    f2_table = f2_smear_table(max(n - 2, 0), eps)
    ln_g = wl_result["ln_g"]
    bin_lo, bin_width, n_bins = wl_result["bin_lo"], wl_result["bin_width"], wl_result["n_bins"]
    label = verbose_label or f"randombg n={n} eps={eps}"
    t0 = time.time()
    chunk_moves = chunk_moves if chunk_moves is not None else 200 * n

    ckpt = load_checkpoint(checkpoint_path) if checkpoint_path else None
    if ckpt is not None and ckpt.get("kind") == "randombg_muca":
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
            print(f"[randombg-muca] {label} resumed: moves_done={moves_done}, "
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
    round_trip_moves_log = []
    round_trips_prev = int(half_trips[0]) // 2

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
        rng_state, bin_idx, chunk_bins, chunk_S = rbg.muca_measure_chunk_randombg_jit(
            L, future, past, counts, n, m, site_t, site_x, w, eps, f2_table, ln_g, H, bin_lo, bin_width, n_bins,
            chunk_moves, rng_state, bin_idx, half_trips, extreme_state, MUCA_MEASURE_EVERY, max_meas,
            lo_extreme, hi_extreme,
        )
        rng_state = rng.tick(rng_state)
        moves_done += chunk_moves
        rec_bins.extend(chunk_bins.tolist())
        rec_S.extend(chunk_S.tolist())
        h, of = _measure_structural_randombg(L, n, site_t, site_x, w, counts)
        rec_height.append(h)
        rec_of.append(of)
        rec_struct_bin.append(bin_idx)

        round_trips_now = int(half_trips[0]) // 2
        if round_trips_now > round_trips_prev:
            for _ in range(round_trips_now - round_trips_prev):
                round_trip_moves_log.append(moves_done)
            round_trips_prev = round_trips_now

        if verbose and time.time() - last_progress > PROGRESS_EVERY_S:
            print(f"[randombg-muca] {label} moves={moves_done} round_trips={int(half_trips[0]) // 2} "
                  f"height={h} of={of:.3f} ({time.time() - t0:.1f}s)", flush=True)
            last_progress = time.time()

        if checkpoint_path and time.time() - last_ckpt > checkpoint_every_s:
            save_checkpoint(checkpoint_path, make_randombg_muca_state(
                L, site_t, site_x, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
                bin_idx, half_trips, moves_done, rec_bins, rec_S, rec_height, rec_of, rec_struct_bin,
            ))
            last_ckpt = time.time()

    if checkpoint_path:
        save_checkpoint(checkpoint_path, make_randombg_muca_state(
            L, site_t, site_x, future, past, counts, rng_state, ln_g, H, bin_lo, bin_width, n_bins,
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
        "site_t": site_t, "site_x": site_x,
        "ln_g": ln_g, "H_muca": H, "round_trips": round_trips, "moves_done": moves_done,
        "elapsed": time.time() - t0,
        "rec_bins": np.array(rec_bins, dtype=np.int64), "rec_S": np.array(rec_S, dtype=np.float64),
        "rec_height": height_arr, "rec_of": np.array(rec_of, dtype=float),
        "rec_struct_bin": np.array(rec_struct_bin, dtype=np.int64),
        "height_round_trips": height_round_trips, "hidden_barrier_warning": hidden_barrier,
        "round_trip_moves_log": round_trip_moves_log,
        "L": L, "future": future, "past": past, "counts": counts, "rng_state": rng_state, "bin_idx": bin_idx,
    }
