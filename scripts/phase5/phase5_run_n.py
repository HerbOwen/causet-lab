"""Phase 5 stage-1 driver: 5 random-background realizations at n=30,
run in parallel across available cores. N is the only thing to change
for stage 2 (n=50) -- see the single constant below.
"""
import sys, os, time, pickle

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

N = 30  # <-- one-line change for stage 2 (n=50)
EPS = 0.1
BACKGROUND_SEEDS = [0, 1, 2, 3, 4]
BETA_HINT = 11.118  # Phase 4's located regular-lattice beta_c at n=30 -- a much
                     # better-informed pilot-range anchor than the naive 1/n
                     # heuristic, since this is the same action/model class.

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")
CKPT_DIR = f"{SCRATCH}/checkpoints"


def run_one_background(seed):
    # Imports inside the worker: required for ProcessPoolExecutor on Windows (spawn).
    from causet_lab.mcmc import random_bg as rbgm
    from causet_lab.mcmc import lattice_gas as lgm
    from causet_lab.mcmc import muca
    from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0

    n, eps = N, EPS
    background_seed = seed
    chain_seed = seed + 10000
    label = f"bg{seed} n={n}"

    w, h, m = rbgm.background_dims(n)
    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=background_seed)

    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=chain_seed,
        checkpoint_path=f"{CKPT_DIR}/randombg_wl_n{n}_bg{seed}.pkl",
        verbose=True, verbose_label=label, beta_hint=BETA_HINT,
    )
    t_wl = time.time() - t0
    print(f"[{label}] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}", flush=True)

    t0 = time.time()
    prod = rbgm.run_muca_production_randombg(
        wl, target_round_trips=10, max_moves=300_000_000,
        checkpoint_path=f"{CKPT_DIR}/randombg_muca_n{n}_bg{seed}.pkl",
        verbose=True, verbose_label=label,
    )
    t_muca = time.time() - t0
    print(f"[{label}] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    # Bracket is deliberately wide (0.01x-30x the regular-lattice anchor):
    # this is a pure post-processing scan over the already-collected
    # ln_g/H_muca (no extra sampling cost), and a smoke test at n=10
    # found its located beta_c sitting right at a tighter bracket's
    # edge (interior=False) -- we don't actually know a priori whether
    # quenched disorder shifts beta_c up or down from the regular
    # lattice's value, so there is no reason to risk a false edge.
    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.01 * BETA_HINT, 30.0 * BETA_HINT, n_scan=800,
    )
    beta_c = beta_c_res["beta_c"]
    print(f"[{label}] beta_c (variance peak) = {beta_c:.4f} interior={beta_c_res['is_interior']}", flush=True)
    if not beta_c_res["is_interior"]:
        print(f"[{label}] WARNING: beta_c landed on the scan bracket edge even at this wide "
              f"margin -- treat this seed's beta_c as unreliable.", flush=True)

    # Low-beta (beta=0) phase observables on this background -- chain
    # seeds here are independent of both the background seed and the
    # WL/MUCA chain seed.
    mm_lo, prof_lo, of_lo, h_lo, layer_lo = [], [], [], [], []
    for s in range(5):
        Lr = lgm.random_filling(n, m, seed=20000 + s)
        C = rbgm.randombg_to_matrix(Lr, n, site_t, site_x, w)
        res = analyze_matrix(C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=6)
        mm_lo.append(res["mm_dim_sampled_mean"]); prof_lo.append(res["abundance_profile"])
        of_lo.append(res["ordering_fraction"]); h_lo.append(res["height"])
        layer_lo.append(rbgm.layer_count_randombg(Lr, n, site_t, h))

    # High-beta (4*located beta_c) phase observables via independent
    # annealed-equilibration chains on the same fixed background.
    beta_hi = 4.0 * beta_c
    mm_hi, prof_hi, of_hi, h_hi, layer_hi = [], [], [], [], []
    for s in range(5):
        _actions, Lr, _future, _past, _counts = rbgm.run_pilot_chain_randombg(
            n, w, h, m, eps, site_t, site_x, beta=beta_hi, seed=30000 + s,
            n_sweeps=400, burn_in=399, measure_every=1, anneal_from=0.0,
        )
        C = rbgm.randombg_to_matrix(Lr, n, site_t, site_x, w)
        res = analyze_matrix(C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=6)
        mm_hi.append(res["mm_dim_sampled_mean"]); prof_hi.append(res["abundance_profile"])
        of_hi.append(res["ordering_fraction"]); h_hi.append(res["height"])
        layer_hi.append(rbgm.layer_count_randombg(Lr, n, site_t, h))

    result = {
        "seed": seed, "background_seed": background_seed, "chain_seed": chain_seed,
        "n": n, "eps": eps,
        "t_wl": t_wl, "t_muca": t_muca,
        "wl_stages": wl["stages"], "wl_reachable": wl["n_reachable_bins"], "wl_n_bins": wl["n_bins"],
        "round_trips": prod["round_trips"], "moves_done": prod["moves_done"],
        "hidden_barrier": prod["hidden_barrier_warning"],
        "height_round_trips": prod["height_round_trips"],
        "beta_c": beta_c, "beta_c_interior": beta_c_res["is_interior"], "beta_hi_used": beta_hi,
        "ln_g": wl["ln_g"], "H_muca": prod["H_muca"],
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "lo": {"mm_dim": nanmean(mm_lo), "of": nanmean(of_lo), "height": nanmean(h_lo),
               "layers": nanmean(layer_lo), "abundance_profile": nanmean_axis0(np.array(prof_lo))},
        "hi": {"mm_dim": nanmean(mm_hi), "of": nanmean(of_hi), "height": nanmean(h_hi),
               "layers": nanmean(layer_hi), "abundance_profile": nanmean_axis0(np.array(prof_hi))},
    }
    print(f"[{label}] TOTAL={t_wl + t_muca:.1f}s lo.of={result['lo']['of']:.3f} "
          f"hi.of={result['hi']['of']:.3f} lo.mm={result['lo']['mm_dim']:.3f} "
          f"hi.mm={result['hi']['mm_dim']:.3f}", flush=True)
    return result


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    t0 = time.time()
    n_workers = min(4, os.cpu_count() or 4)
    print(f"[main] N={N} eps={EPS} background_seeds={BACKGROUND_SEEDS} n_workers={n_workers}", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_one_background, seed): seed for seed in BACKGROUND_SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                results[seed] = fut.result()
                print(f"[main] background seed={seed} finished ({time.time()-t0:.1f}s elapsed total)", flush=True)
            except Exception as e:
                print(f"[main] background seed={seed} FAILED: {e!r}", flush=True)
                raise

    with open(f"{SCRATCH}/phase5_randombg_n{N}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n[main] ALL DONE in {time.time()-t0:.1f}s", flush=True)
    for seed in BACKGROUND_SEEDS:
        r = results[seed]
        print(f"  seed={seed}: beta_c={r['beta_c']:.3f} t_wl={r['t_wl']:.1f}s t_muca={r['t_muca']:.1f}s "
              f"round_trips={r['round_trips']} hidden_barrier={r['hidden_barrier']}", flush=True)
    print("PHASE5_STAGE1_DONE", flush=True)


if __name__ == "__main__":
    main()
