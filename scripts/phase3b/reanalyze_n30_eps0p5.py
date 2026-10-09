import time, pickle, sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def peak_diagnostics(p):
    peaks = muca._find_two_peaks(p)
    if peaks is None:
        return None
    i1, i2 = peaks
    valley_rel = int(np.argmin(p[i1:i2 + 1]))
    valley = i1 + valley_rel
    height_diff = p[i1] - p[i2]
    area_left = p[:valley].sum()
    area_right = p[valley + 1:].sum()
    area_diff = area_left - area_right
    return {"peaks": peaks, "valley": valley, "height_diff": height_diff, "area_diff": area_diff,
            "peak_val": 0.5 * (p[i1] + p[i2]), "valley_val": p[valley]}


def zero_crossing(betas, diffs):
    """Linear-interpolated beta where diffs crosses zero, scanning for
    the first sign change among valid (non-nan) consecutive points."""
    betas = np.asarray(betas); diffs = np.asarray(diffs)
    valid = ~np.isnan(diffs)
    idx = np.flatnonzero(valid)
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        if b != a + 1:
            continue  # not adjacent on the grid -- don't bridge a gap of "no bimodal structure"
        if diffs[a] == 0:
            return betas[a]
        if diffs[a] * diffs[b] < 0:
            t = diffs[a] / (diffs[a] - diffs[b])
            return betas[a] + t * (betas[b] - betas[a])
    return None


def scan_fine_grid(ln_g, H_muca, bin_lo, bin_width, n_bins, betas):
    height_diffs, area_diffs, n_bimodal = [], [], 0
    for beta in betas:
        p, _ = muca.reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta)
        d = peak_diagnostics(p)
        if d is None:
            height_diffs.append(np.nan); area_diffs.append(np.nan)
        else:
            n_bimodal += 1
            height_diffs.append(d["height_diff"]); area_diffs.append(d["area_diff"])
    return np.array(height_diffs), np.array(area_diffs), n_bimodal


def main():
    t_start = time.time()
    N, eps = 30, 0.5
    predicted = beta_c_glaser_2018(N, eps)

    t0 = time.time()
    with open(f"{SCRATCH}/pilot_n30_eps0p5_result.pkl", "rb") as f:
        result = pickle.load(f)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    wl, prod = result["wl"], result["prod"]
    ln_g = wl["ln_g"]
    bin_lo, bin_width, n_bins = wl["bin_lo"], wl["bin_width"], wl["n_bins"]
    H_muca = prod["H_muca"]
    rec_bins = prod["rec_bins"]

    # --- Q1: confirm which weights are used ---------------------------------
    same_object = wl["ln_g"] is prod["ln_g"]
    same_values = np.array_equal(wl["ln_g"], prod["ln_g"])
    print(f"[Q1] wl['ln_g'] is prod['ln_g']: {same_object}; values equal: {same_values}")
    print(f"[Q1] ln_g is never mutated during production (wl_mode=False branch not present in "
          f"muca_measure_chunk_jit) -- confirmed by code inspection, not just by this array check.")
    print(f"[Q1] H_muca total visits = {H_muca.sum()}, nonzero bins = {(H_muca>0).sum()}/{n_bins}, "
          f"ever_visited (WL-reachable) bins = {wl['ever_visited'].sum()}/{n_bins}")

    # --- Q2: fine continuous beta grid, old vs corrected formula -------------
    t0 = time.time()
    beta_lo_scan, beta_hi_scan = 0.01 * predicted, 6.0 * predicted
    betas_fine = np.linspace(beta_lo_scan, beta_hi_scan, 600)

    # old formula (ln_g alone, ignores production histogram)
    hd_old, ad_old, n_bimodal_old = [], [], 0
    for beta in betas_fine:
        p, _ = muca.reweight_P_beta(ln_g, bin_lo, bin_width, n_bins, beta)
        d = peak_diagnostics(p)
        if d is None:
            hd_old.append(np.nan)
        else:
            n_bimodal_old += 1
            hd_old.append(d["height_diff"])
    hd_old = np.array(hd_old)

    # corrected formula (folds in H_muca)
    hd_new, ad_new, n_bimodal_new = scan_fine_grid(ln_g, H_muca, bin_lo, bin_width, n_bins, betas_fine)
    print(f"[Q2] fine grid scan: {time.time()-t0:.1f}s over {len(betas_fine)} points")
    print(f"[Q2] OLD formula (ln_g only): bimodal at {n_bimodal_old}/{len(betas_fine)} grid points, "
          f"betas={betas_fine[~np.isnan(hd_old)].min() if n_bimodal_old else float('nan'):.5f}"
          f"-{betas_fine[~np.isnan(hd_old)].max() if n_bimodal_old else float('nan'):.5f}" if n_bimodal_old else
          "[Q2] OLD formula: no bimodal points found anywhere in the scan")
    if n_bimodal_new:
        bm_mask = ~np.isnan(hd_new)
        print(f"[Q2] CORRECTED formula (H_muca folded in): bimodal at {n_bimodal_new}/{len(betas_fine)} grid points, "
              f"beta window=[{betas_fine[bm_mask].min():.5f}, {betas_fine[bm_mask].max():.5f}]")
    else:
        print("[Q2] CORRECTED formula: no bimodal points found anywhere in the scan")

    beta_c_height = zero_crossing(betas_fine, hd_new)
    beta_c_area = zero_crossing(betas_fine, ad_new)
    print(f"[Q2] beta_c (equal-height, corrected formula, point estimate) = {beta_c_height}")
    print(f"[Q2] beta_c (equal-area,   corrected formula, point estimate) = {beta_c_area}")
    print(f"[Q2] predicted (1.66/(N eps^2)) = {1.66/(N*eps**2):.5f}; full formula = {predicted:.5f}")

    with open(f"{SCRATCH}/reanalysis_fine_grid.pkl", "wb") as f:
        pickle.dump({"betas_fine": betas_fine, "hd_old": hd_old, "hd_new": hd_new, "ad_new": ad_new,
                     "beta_c_height": beta_c_height, "beta_c_area": beta_c_area}, f)

    # --- Q3: block bootstrap uncertainty --------------------------------------
    t0 = time.time()
    ever_visited = wl["ever_visited"]
    idx = np.flatnonzero(ever_visited)
    lo_extreme, hi_extreme = int(idx[0]), int(idx[-1])

    # Re-derive half-trip boundaries from the recorded (measure_every=5) bin
    # time series the same way the jit kernel does internally, to get
    # natural block edges (~1 round trip per 2 boundaries) without needing
    # any new sampling.
    touches = []
    state = 0
    for k, b in enumerate(rec_bins):
        if b <= lo_extreme:
            if state == 1:
                touches.append(k)
            state = -1
        elif b >= hi_extreme:
            if state == -1:
                touches.append(k)
            state = 1
    n_half_trips = len(touches)
    print(f"[Q3] re-derived {n_half_trips} half-trip boundaries from rec_bins "
          f"(expect ~{2*prod['round_trips']}) in {time.time()-t0:.1f}s")

    # Blocks: group every ~2 half-trips (~1 round trip) worth of records into
    # one contiguous block -> block boundaries at every 2nd touch index.
    block_edges = [0] + touches[1::2] + [len(rec_bins)]
    block_edges = sorted(set(block_edges))
    blocks = [(block_edges[i], block_edges[i + 1]) for i in range(len(block_edges) - 1)
              if block_edges[i + 1] > block_edges[i]]
    print(f"[Q3] {len(blocks)} bootstrap blocks (~1 round trip each), "
          f"lengths min={min(b[1]-b[0] for b in blocks)} max={max(b[1]-b[0] for b in blocks)} records")

    rng = np.random.default_rng(12345)
    n_boot = 300
    betas_boot_grid = np.linspace(beta_lo_scan, beta_hi_scan, 150)
    boot_beta_c_height, boot_beta_c_area = [], []
    t0 = time.time()
    for rep in range(n_boot):
        choice = rng.integers(0, len(blocks), size=len(blocks))
        boot_bins = np.concatenate([rec_bins[blocks[c][0]:blocks[c][1]] for c in choice])
        H_boot = np.bincount(boot_bins, minlength=n_bins).astype(np.float64)
        hd_b, ad_b, nb = scan_fine_grid(ln_g, H_boot, bin_lo, bin_width, n_bins, betas_boot_grid)
        bc_h = zero_crossing(betas_boot_grid, hd_b)
        bc_a = zero_crossing(betas_boot_grid, ad_b)
        if bc_h is not None:
            boot_beta_c_height.append(bc_h)
        if bc_a is not None:
            boot_beta_c_area.append(bc_a)
    t_boot = time.time() - t0
    print(f"[Q3] bootstrap: {n_boot} reps in {t_boot:.1f}s "
          f"({len(boot_beta_c_height)}/{n_boot} found equal-height beta_c, "
          f"{len(boot_beta_c_area)}/{n_boot} found equal-area beta_c)")
    if boot_beta_c_height:
        bh = np.array(boot_beta_c_height)
        print(f"[Q3] beta_c (equal-height) bootstrap: mean={bh.mean():.5f} std={bh.std():.5f} "
              f"[{np.percentile(bh,2.5):.5f}, {np.percentile(bh,97.5):.5f}] (95% CI)")
    if boot_beta_c_area:
        ba = np.array(boot_beta_c_area)
        print(f"[Q3] beta_c (equal-area) bootstrap: mean={ba.mean():.5f} std={ba.std():.5f} "
              f"[{np.percentile(ba,2.5):.5f}, {np.percentile(ba,97.5):.5f}] (95% CI)")

    with open(f"{SCRATCH}/reanalysis_bootstrap.pkl", "wb") as f:
        pickle.dump({"blocks": blocks, "boot_beta_c_height": boot_beta_c_height,
                     "boot_beta_c_area": boot_beta_c_area, "n_boot": n_boot}, f)

    print(f"[TOTAL] reanalysis done in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
