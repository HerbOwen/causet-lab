"""Phase 5 stage-2 production: n=50, BOTH the random-background (5
realizations) and the regular-lattice-via-random_bg (5 chain seeds)
runs, submitted into a SINGLE 4-worker pool (not two separate 4-worker
pools) so the two models actually share the 4 physical cores instead
of oversubscribing them 2x.

Pre-flight pilot (phase5_pilot_n50.py, ~7 min) found the OPPOSITE of
stage 1's mixing asymmetry: this time the random background's WL hit
a severe late-stage stall (stage 13, collapsed from 86 to 9 reachable
bins in-stage, no recovery after 160+s) while the regular lattice
progressed smoothly through a comparable stage. Since the pilot does
not support deprioritizing either model, both are scheduled with equal
priority here. The stall shortcut (f_mod <= 1e-2, no stage advance for
600s -- chosen from the pilot's own stall, which was still stuck at
~160s and showed no sign of recovering) protects both equally.
"""
import sys, os, time, pickle

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

N = 50
EPS = 0.1
BETA_HINT = 6.657  # 1/n-scaled from the n=30 matched-pipeline regular-lattice beta_c
STALL_SHORTCUT_SECONDS = 600.0
STALL_SHORTCUT_F = 1e-2

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")
CKPT_DIR = f"{SCRATCH}/checkpoints"


def _common_phase_analysis(label, n, eps, m, w, h, site_t, site_x, rbgm, lgm, muca, analyze_matrix,
                            nanmean, nanmean_axis0, wl, prod, beta_c):
    mm_lo, prof_lo, of_lo, h_lo, layer_lo = [], [], [], [], []
    for s in range(5):
        Lr = lgm.random_filling(n, m, seed=20000 + s)
        C = rbgm.randombg_to_matrix(Lr, n, site_t, site_x, w)
        res = analyze_matrix(C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=6)
        mm_lo.append(res["mm_dim_sampled_mean"]); prof_lo.append(res["abundance_profile"])
        of_lo.append(res["ordering_fraction"]); h_lo.append(res["height"])
        layer_lo.append(rbgm.layer_count_randombg(Lr, n, site_t, h))

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

    return (
        {"mm_dim": nanmean(mm_lo), "of": nanmean(of_lo), "height": nanmean(h_lo),
         "layers": nanmean(layer_lo), "abundance_profile": nanmean_axis0(np.array(prof_lo))},
        {"mm_dim": nanmean(mm_hi), "of": nanmean(of_hi), "height": nanmean(h_hi),
         "layers": nanmean(layer_hi), "abundance_profile": nanmean_axis0(np.array(prof_hi))},
        beta_hi,
    )


def run_randombg(seed):
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
        stall_shortcut_seconds=STALL_SHORTCUT_SECONDS, stall_shortcut_f=STALL_SHORTCUT_F,
    )
    t_wl = time.time() - t0
    used_shortcut = wl.get("used_stall_shortcut", False)
    print(f"[{label}] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}, used_stall_shortcut={used_shortcut}", flush=True)

    t0 = time.time()
    prod = rbgm.run_muca_production_randombg(
        wl, target_round_trips=10, max_moves=300_000_000,
        checkpoint_path=f"{CKPT_DIR}/randombg_muca_n{n}_bg{seed}.pkl",
        verbose=True, verbose_label=label,
    )
    t_muca = time.time() - t0
    print(f"[{label}] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.01 * BETA_HINT, 30.0 * BETA_HINT, n_scan=800,
    )
    beta_c = beta_c_res["beta_c"]
    print(f"[{label}] beta_c (variance peak) = {beta_c:.4f} interior={beta_c_res['is_interior']}", flush=True)
    if not beta_c_res["is_interior"]:
        print(f"[{label}] WARNING: beta_c landed on the scan bracket edge -- treat as unreliable.", flush=True)

    lo, hi, beta_hi = _common_phase_analysis(label, n, eps, m, w, h, site_t, site_x, rbgm, lgm, muca,
                                              analyze_matrix, nanmean, nanmean_axis0, wl, prod, beta_c)

    result = {
        "seed": seed, "background_seed": background_seed, "chain_seed": chain_seed,
        "n": n, "eps": eps,
        "t_wl": t_wl, "t_muca": t_muca,
        "wl_stages": wl["stages"], "wl_reachable": wl["n_reachable_bins"], "wl_n_bins": wl["n_bins"],
        "used_stall_shortcut": used_shortcut, "final_f_mod": wl.get("final_f_mod"),
        "stage_moves_log": wl.get("stage_moves_log"),
        "round_trips": prod["round_trips"], "moves_done": prod["moves_done"],
        "hidden_barrier": prod["hidden_barrier_warning"],
        "height_round_trips": prod["height_round_trips"],
        "round_trip_moves_log": prod.get("round_trip_moves_log"),
        "beta_c": beta_c, "beta_c_interior": beta_c_res["is_interior"], "beta_hi_used": beta_hi,
        "ln_g": wl["ln_g"], "H_muca": prod["H_muca"],
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "lo": lo, "hi": hi,
    }
    print(f"[{label}] TOTAL={t_wl + t_muca:.1f}s lo.of={lo['of']:.3f} hi.of={hi['of']:.3f} "
          f"lo.mm={lo['mm_dim']:.3f} hi.mm={hi['mm_dim']:.3f}", flush=True)
    return ("randombg", seed, result)


def run_reglat(seed):
    from causet_lab.mcmc import random_bg as rbgm
    from causet_lab.mcmc import lattice_gas as lgm
    from causet_lab.mcmc import muca
    from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0

    n, eps = N, EPS
    label = f"reglat-via-rbg chainseed{seed} n={n}"

    w, h, m = rbgm.background_dims(n)
    site_ids = np.arange(m)
    site_t = (site_ids // w).astype(np.float64)
    site_x = (site_ids % w).astype(np.float64)

    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=seed,
        checkpoint_path=f"{CKPT_DIR}/reglat_via_rbg_wl_n{n}_cs{seed}.pkl",
        verbose=True, verbose_label=label, beta_hint=BETA_HINT,
        stall_shortcut_seconds=STALL_SHORTCUT_SECONDS, stall_shortcut_f=STALL_SHORTCUT_F,
    )
    t_wl = time.time() - t0
    used_shortcut = wl.get("used_stall_shortcut", False)
    print(f"[{label}] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}, used_stall_shortcut={used_shortcut}", flush=True)

    t0 = time.time()
    prod = rbgm.run_muca_production_randombg(
        wl, target_round_trips=10, max_moves=300_000_000,
        checkpoint_path=f"{CKPT_DIR}/reglat_via_rbg_muca_n{n}_cs{seed}.pkl",
        verbose=True, verbose_label=label,
    )
    t_muca = time.time() - t0
    print(f"[{label}] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.01 * BETA_HINT, 30.0 * BETA_HINT, n_scan=800,
    )
    beta_c = beta_c_res["beta_c"]
    print(f"[{label}] beta_c (variance peak) = {beta_c:.4f} interior={beta_c_res['is_interior']}", flush=True)

    mean_S_hot, _var_S_hot = muca.reweight_mean_var_S(wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], 0.0)
    ever_visited = wl["ever_visited"]
    lo_idx = int(np.flatnonzero(ever_visited)[0]) if ever_visited.any() else None
    lowest_S_reached = wl["bin_lo"] + lo_idx * wl["bin_width"] if lo_idx is not None else None

    lo, hi, beta_hi = _common_phase_analysis(label, n, eps, m, w, h, site_t, site_x, rbgm, lgm, muca,
                                              analyze_matrix, nanmean, nanmean_axis0, wl, prod, beta_c)

    result = {
        "seed": seed, "n": n, "eps": eps,
        "t_wl": t_wl, "t_muca": t_muca,
        "wl_stages": wl["stages"], "wl_reachable": wl["n_reachable_bins"], "wl_n_bins": wl["n_bins"],
        "used_stall_shortcut": used_shortcut, "final_f_mod": wl.get("final_f_mod"),
        "stage_moves_log": wl.get("stage_moves_log"),
        "round_trips": prod["round_trips"], "moves_done": prod["moves_done"],
        "hidden_barrier": prod["hidden_barrier_warning"],
        "round_trip_moves_log": prod.get("round_trip_moves_log"),
        "beta_c": beta_c, "beta_c_interior": beta_c_res["is_interior"],
        "mean_S_hot": mean_S_hot, "lowest_S_reached": lowest_S_reached,
        "ln_g": wl["ln_g"], "H_muca": prod["H_muca"],
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "site_t": site_t, "site_x": site_x, "w": w, "h": h, "m": m,
        "lo": lo, "hi": hi,
    }
    print(f"[{label}] TOTAL={t_wl + t_muca:.1f}s mean_S_hot={mean_S_hot:.3f} lowest_S_reached={lowest_S_reached:.3f}", flush=True)
    return ("reglat", seed, result)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    t0 = time.time()
    n_workers = min(4, os.cpu_count() or 4)
    seeds = [0, 1, 2, 3, 4]
    print(f"[main] N={N} eps={EPS} n_workers={n_workers} "
          f"(10 jobs: 5 randombg + 5 reglat, shared pool, stall_shortcut={STALL_SHORTCUT_SECONDS}s)", flush=True)
    randombg_results, reglat_results = {}, {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for s in seeds:
            futures[ex.submit(run_randombg, s)] = ("randombg", s)
        for s in seeds:
            futures[ex.submit(run_reglat, s)] = ("reglat", s)
        for fut in as_completed(futures):
            kind, seed = futures[fut]
            try:
                kind2, seed2, result = fut.result()
                if kind2 == "randombg":
                    randombg_results[seed2] = result
                else:
                    reglat_results[seed2] = result
                print(f"[main] {kind2} seed={seed2} finished ({time.time()-t0:.1f}s elapsed total)", flush=True)
            except Exception as e:
                print(f"[main] {kind} seed={seed} FAILED: {e!r}", flush=True)
                raise

    with open(f"{SCRATCH}/phase5_randombg_n{N}_results.pkl", "wb") as f:
        pickle.dump(randombg_results, f)
    with open(f"{SCRATCH}/phase5_reglat_via_rbg_n{N}_results.pkl", "wb") as f:
        pickle.dump(reglat_results, f)

    print(f"\n[main] ALL DONE in {time.time()-t0:.1f}s", flush=True)
    for seed in seeds:
        r = randombg_results[seed]
        print(f"  randombg seed={seed}: beta_c={r['beta_c']:.3f} t_wl={r['t_wl']:.1f}s t_muca={r['t_muca']:.1f}s "
              f"round_trips={r['round_trips']} hidden_barrier={r['hidden_barrier']} "
              f"used_stall_shortcut={r['used_stall_shortcut']}", flush=True)
    for seed in seeds:
        r = reglat_results[seed]
        print(f"  reglat seed={seed}: beta_c={r['beta_c']:.3f} t_wl={r['t_wl']:.1f}s t_muca={r['t_muca']:.1f}s "
              f"round_trips={r['round_trips']} hidden_barrier={r['hidden_barrier']} "
              f"used_stall_shortcut={r['used_stall_shortcut']}", flush=True)
    print("PHASE5_STAGE2_COMBINED_DONE", flush=True)


if __name__ == "__main__":
    main()
