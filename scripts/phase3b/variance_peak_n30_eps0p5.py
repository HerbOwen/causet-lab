import time, pickle, sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def main():
    t_start = time.time()
    N, eps = 30, 0.5
    predicted = beta_c_glaser_2018(N, eps)

    with open(f"{SCRATCH}/pilot_n30_eps0p5_result.pkl", "rb") as f:
        result = pickle.load(f)
    wl, prod = result["wl"], result["prod"]
    ln_g, H = wl["ln_g"], prod["H_muca"]
    bin_lo, bin_width, n_bins = wl["bin_lo"], wl["bin_width"], wl["n_bins"]

    # Wide scan -- we don't yet trust the formula's predicted location
    # given everything else found so far, so don't beg the question.
    t0 = time.time()
    res = muca.locate_beta_c_variance_peak(ln_g, H, bin_lo, bin_width, n_bins, 0.01, 2.0, n_scan=500)
    print(f"[varpeak] point estimate: beta_c={res['beta_c']:.5f} var_max={res['var_max']:.2f} "
          f"interior={res['is_interior']} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[varpeak] predicted (1.66/(N eps^2))={1.66/(N*eps**2):.5f}; full formula={predicted:.5f}")

    # dump the curve to look for secondary bumps / sanity
    betas, variances = res["betas"], res["variances"]
    top5 = np.argsort(-variances)[:5]
    for i in sorted(top5):
        print(f"    beta={betas[i]:.4f} var={variances[i]:.2f}")

    # --- block bootstrap on the SAME blocks logic as before ---
    t0 = time.time()
    ever_visited = wl["ever_visited"]
    idx = np.flatnonzero(ever_visited)
    lo_extreme, hi_extreme = int(idx[0]), int(idx[-1])
    rec_bins = prod["rec_bins"]
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
    block_edges = sorted(set([0] + touches[1::2] + [len(rec_bins)]))
    blocks = [(block_edges[i], block_edges[i + 1]) for i in range(len(block_edges) - 1)
              if block_edges[i + 1] > block_edges[i]]
    print(f"[varpeak] {len(blocks)} bootstrap blocks ({time.time()-t0:.1f}s)")

    def parabolic_peak(betas_arr, vals):
        i = int(np.argmax(vals))
        if i == 0 or i == len(vals) - 1:
            return float(betas_arr[i])
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom == 0:
            return float(betas_arr[i])
        frac = 0.5 * (y0 - y2) / denom
        step = betas_arr[i + 1] - betas_arr[i]
        return float(betas_arr[i] + frac * step)

    rng = np.random.default_rng(777)
    n_boot = 300
    betas_boot = np.linspace(0.01, 2.0, 400)  # fine enough that interpolation, not grid snapping, sets resolution
    boot_beta_c = []
    t0 = time.time()
    for rep in range(n_boot):
        choice = rng.integers(0, len(blocks), size=len(blocks))
        boot_bins = np.concatenate([rec_bins[blocks[c][0]:blocks[c][1]] for c in choice])
        H_boot = np.bincount(boot_bins, minlength=n_bins).astype(np.float64)
        variances = np.empty(len(betas_boot))
        for i, beta in enumerate(betas_boot):
            _, variances[i] = muca.reweight_mean_var_S(ln_g, H_boot, bin_lo, bin_width, n_bins, beta)
        boot_beta_c.append(parabolic_peak(betas_boot, variances))
    t_boot = time.time() - t0
    boot_beta_c = np.array(boot_beta_c)
    print(f"[varpeak] bootstrap: {n_boot} reps in {t_boot:.1f}s", flush=True)
    print(f"[varpeak] beta_c (variance peak) bootstrap: mean={boot_beta_c.mean():.5f} std={boot_beta_c.std():.5f} "
          f"[{np.percentile(boot_beta_c,2.5):.5f}, {np.percentile(boot_beta_c,97.5):.5f}] (95% CI)")

    with open(f"{SCRATCH}/variance_peak_result.pkl", "wb") as f:
        pickle.dump({"res": res, "boot_beta_c": boot_beta_c, "predicted": predicted}, f)
    print(f"[TOTAL] {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
