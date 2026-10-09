"""Regular lattice re-run through the EXACT random_bg pipeline (same
code path as phase5_run_n.py, just site_t/site_x set to the regular
lattice's own integer positions instead of a sprinkled background),
5 chain seeds, n=30 eps=0.1 -- for an apples-to-apples beta_c
comparison against the Phase 5 random-background ensemble.
"""
import sys, os, time, pickle

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

N = 30
EPS = 0.1
CHAIN_SEEDS = [0, 1, 2, 3, 4]
BETA_HINT = 11.118  # same anchor Phase 4 located for this exact (n, eps) on the regular lattice

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")
CKPT_DIR = f"{SCRATCH}/checkpoints"


def run_one_chainseed(seed):
    from causet_lab.mcmc import random_bg as rbgm
    from causet_lab.mcmc import lattice_gas as lgm
    from causet_lab.mcmc import muca
    from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0

    n, eps = N, EPS
    label = f"reglat-via-rbg chainseed{seed} n={n}"

    w, h, m = rbgm.background_dims(n)
    # The regular lattice's own positions, expressed in random_bg's (site_t, site_x) form.
    site_ids = np.arange(m)
    site_t = (site_ids // w).astype(np.float64)
    site_x = (site_ids % w).astype(np.float64)

    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=seed,
        checkpoint_path=f"{CKPT_DIR}/reglat_via_rbg_wl_n{n}_cs{seed}.pkl",
        verbose=True, verbose_label=label, beta_hint=BETA_HINT,
    )
    t_wl = time.time() - t0
    print(f"[{label}] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}", flush=True)

    t0 = time.time()
    prod = rbgm.run_muca_production_randombg(
        wl, target_round_trips=10, max_moves=300_000_000,
        checkpoint_path=f"{CKPT_DIR}/reglat_via_rbg_muca_n{n}_cs{seed}.pkl",
        verbose=True, verbose_label=label,
    )
    t_muca = time.time() - t0
    print(f"[{label}] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    # Identical scan bracket/n_scan to the Phase 5 random-background script.
    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.01 * BETA_HINT, 30.0 * BETA_HINT, n_scan=800,
    )
    beta_c = beta_c_res["beta_c"]
    print(f"[{label}] beta_c (variance peak) = {beta_c:.4f} interior={beta_c_res['is_interior']}", flush=True)

    # Hot-phase mean action (beta=0) via reweighting -- for the mechanism check.
    mean_S_hot, _var_S_hot = muca.reweight_mean_var_S(wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], 0.0)

    # Lowest action actually reached (lowest ever_visited bin's lower edge).
    ever_visited = wl["ever_visited"]
    lo_idx = int(np.flatnonzero(ever_visited)[0]) if ever_visited.any() else None
    lowest_S_reached = wl["bin_lo"] + lo_idx * wl["bin_width"] if lo_idx is not None else None

    result = {
        "seed": seed, "n": n, "eps": eps,
        "t_wl": t_wl, "t_muca": t_muca,
        "wl_stages": wl["stages"], "wl_reachable": wl["n_reachable_bins"], "wl_n_bins": wl["n_bins"],
        "round_trips": prod["round_trips"], "moves_done": prod["moves_done"],
        "hidden_barrier": prod["hidden_barrier_warning"],
        "beta_c": beta_c, "beta_c_interior": beta_c_res["is_interior"],
        "mean_S_hot": mean_S_hot, "lowest_S_reached": lowest_S_reached,
        "ln_g": wl["ln_g"], "H_muca": prod["H_muca"],
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "site_t": site_t, "site_x": site_x, "w": w, "h": h, "m": m,
    }
    print(f"[{label}] TOTAL={t_wl + t_muca:.1f}s mean_S_hot={mean_S_hot:.3f} lowest_S_reached={lowest_S_reached:.3f}", flush=True)
    return result


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    t0 = time.time()
    n_workers = min(4, os.cpu_count() or 4)
    print(f"[main] N={N} eps={EPS} chain_seeds={CHAIN_SEEDS} n_workers={n_workers} (regular lattice via random_bg pipeline)", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_one_chainseed, seed): seed for seed in CHAIN_SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                results[seed] = fut.result()
                print(f"[main] chain seed={seed} finished ({time.time()-t0:.1f}s elapsed total)", flush=True)
            except Exception as e:
                print(f"[main] chain seed={seed} FAILED: {e!r}", flush=True)
                raise

    with open(f"{SCRATCH}/phase5_reglat_via_rbg_n{N}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n[main] ALL DONE in {time.time()-t0:.1f}s", flush=True)
    for seed in CHAIN_SEEDS:
        r = results[seed]
        print(f"  seed={seed}: beta_c={r['beta_c']:.3f} t_wl={r['t_wl']:.1f}s t_muca={r['t_muca']:.1f}s "
              f"round_trips={r['round_trips']} hidden_barrier={r['hidden_barrier']}", flush=True)
    print("PHASE5_REGLAT_RERUN_DONE", flush=True)


if __name__ == "__main__":
    main()
