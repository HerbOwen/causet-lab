"""Stage-2 n=50, round 2: fixed quantile-based WL window (no widening),
_widen's gradient-clamped fix (still used for muca.py/lattice_gas.py's
own loops, not needed here since widening is off by default), the
anomaly-gated stall shortcut, and a window-coverage check.

Jobs:
  - The 3 seeds that failed under the OLD (adaptive-widen) method:
    randombg bg1, reglat chainseed2, reglat chainseed3. Must succeed
    here to be usable at all.
  - 2 "fairness" re-runs of PREVIOUSLY CLEAN seeds (randombg bg0,
    reglat chainseed0) under this SAME new method, to confirm beta_c
    is unchanged within error before trusting a mixed-methodology
    result set. Written to a separate checkpoint path so the original
    (still valid) v2 results for those seeds are not overwritten.

If the 2 fairness re-runs agree with their original v2 results within
error, the other 5 untouched seeds (randombg 2/3/4, reglat 1/4) are
kept from the v2 run, and this is stated plainly in the report. If they
do NOT agree, all 10 need rerunning under the new method -- handled as
a follow-up, not papered over.
"""
import sys, os, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

N = 50
EPS = 0.1
BETA_HINT = 6.657
STALL_SHORTCUT_SECONDS = 600.0
STALL_SHORTCUT_F = 1e-2

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")
CKPT_DIR = f"{SCRATCH}/checkpoints/newmethod"


def _common_phase_analysis(n, eps, m, w, h, site_t, site_x, rbgm, lgm, muca, analyze_matrix,
                            nanmean, nanmean_axis0, beta_c):
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


def _run_common(label, wl_fn_args, wl_ckpt, muca_ckpt, n, eps, w, h, m, site_t, site_x, rbgm, lgm, muca,
                 analyze_matrix, nanmean, nanmean_axis0):
    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=wl_fn_args["seed"],
        checkpoint_path=wl_ckpt, verbose=True, verbose_label=label, beta_hint=BETA_HINT,
        stall_shortcut_seconds=STALL_SHORTCUT_SECONDS, stall_shortcut_f=STALL_SHORTCUT_F,
        allow_widen=False,
    )
    t_wl = time.time() - t0
    if wl["ln_g_anomalies"]:
        print(f"[{label}] ln_g ANOMALIES at end of WL: {wl['ln_g_anomalies']}", flush=True)
    print(f"[{label}] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}, used_stall_shortcut={wl['used_stall_shortcut']}, "
          f"edge_hits_total={wl['edge_hits_total']}", flush=True)

    t0 = time.time()
    prod = rbgm.run_muca_production_randombg(
        wl, target_round_trips=10, max_moves=300_000_000,
        checkpoint_path=muca_ckpt, verbose=True, verbose_label=label,
    )
    t_muca = time.time() - t0
    print(f"[{label}] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.01 * BETA_HINT, 30.0 * BETA_HINT, n_scan=800,
    )
    beta_c = beta_c_res["beta_c"]
    print(f"[{label}] beta_c={beta_c:.4f} interior={beta_c_res['is_interior']} "
          f"coverage_frac={beta_c_res['h_muca_coverage_frac']:.3f} reliable={beta_c_res['reliable']}", flush=True)

    coverage = rbgm.verify_window_coverage_randombg(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], beta_c,
    )
    print(f"[{label}] window coverage check: ok={coverage['ok']}", flush=True)
    for row in coverage["per_beta"]:
        print(f"[{label}]   beta={row['beta']:.3f} ({row['beta_frac_of_c']}x beta_c): "
              f"lo_edge_mass={row['lo_edge_mass']:.2e} hi_edge_mass={row['hi_edge_mass']:.2e} ok={row['ok']}", flush=True)

    mean_S_hot, _ = muca.reweight_mean_var_S(wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], 0.0)
    ever_visited = wl["ever_visited"]
    lo_idx = int(np.flatnonzero(ever_visited)[0]) if ever_visited.any() else None
    lowest_S_reached = wl["bin_lo"] + lo_idx * wl["bin_width"] if lo_idx is not None else None

    lo, hi, beta_hi = _common_phase_analysis(n, eps, m, w, h, site_t, site_x, rbgm, lgm, muca,
                                              analyze_matrix, nanmean, nanmean_axis0, beta_c)

    result = {
        "t_wl": t_wl, "t_muca": t_muca,
        "wl_stages": wl["stages"], "wl_reachable": wl["n_reachable_bins"], "wl_n_bins": wl["n_bins"],
        "used_stall_shortcut": wl["used_stall_shortcut"], "final_f_mod": wl.get("final_f_mod"),
        "ln_g_anomalies": wl["ln_g_anomalies"], "edge_hits_total": wl["edge_hits_total"],
        "round_trips": prod["round_trips"], "moves_done": prod["moves_done"],
        "hidden_barrier": prod["hidden_barrier_warning"],
        "beta_c": beta_c, "beta_c_interior": beta_c_res["is_interior"],
        "beta_c_reliable": beta_c_res["reliable"], "h_muca_coverage_frac": beta_c_res["h_muca_coverage_frac"],
        "window_coverage": coverage,
        "mean_S_hot": mean_S_hot, "lowest_S_reached": lowest_S_reached,
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "lo": lo, "hi": hi, "beta_hi_used": beta_hi,
    }
    print(f"[{label}] TOTAL={t_wl + t_muca:.1f}s", flush=True)
    return result


def run_randombg(seed):
    from causet_lab.mcmc import random_bg as rbgm
    from causet_lab.mcmc import lattice_gas as lgm
    from causet_lab.mcmc import muca
    from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0

    n, eps = N, EPS
    chain_seed = seed + 10000
    label = f"NEWMETHOD bg{seed} n={n}"
    w, h, m = rbgm.background_dims(n)
    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=seed)

    result = _run_common(label, {"seed": chain_seed},
                          f"{CKPT_DIR}/randombg_wl_n{n}_bg{seed}.pkl",
                          f"{CKPT_DIR}/randombg_muca_n{n}_bg{seed}.pkl",
                          n, eps, w, h, m, site_t, site_x, rbgm, lgm, muca,
                          analyze_matrix, nanmean, nanmean_axis0)
    result.update({"seed": seed, "background_seed": seed, "chain_seed": chain_seed, "n": n, "eps": eps})
    return ("randombg", seed, result)


def run_reglat(seed):
    from causet_lab.mcmc import random_bg as rbgm
    from causet_lab.mcmc import lattice_gas as lgm
    from causet_lab.mcmc import muca
    from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0

    n, eps = N, EPS
    label = f"NEWMETHOD reglat-via-rbg chainseed{seed} n={n}"
    w, h, m = rbgm.background_dims(n)
    site_ids = np.arange(m)
    site_t = (site_ids // w).astype(np.float64)
    site_x = (site_ids % w).astype(np.float64)

    result = _run_common(label, {"seed": seed},
                          f"{CKPT_DIR}/reglat_via_rbg_wl_n{n}_cs{seed}.pkl",
                          f"{CKPT_DIR}/reglat_via_rbg_muca_n{n}_cs{seed}.pkl",
                          n, eps, w, h, m, site_t, site_x, rbgm, lgm, muca,
                          analyze_matrix, nanmean, nanmean_axis0)
    result.update({"seed": seed, "n": n, "eps": eps})
    return ("reglat", seed, result)


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    t0 = time.time()
    # 3 must-fix seeds + 2 fairness re-verification seeds (previously clean under the old method)
    jobs = [("randombg", 1), ("reglat", 2), ("reglat", 3), ("randombg", 0), ("reglat", 0)]
    print(f"[main] NEW METHOD (fixed window, no widening) jobs: {jobs}", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {}
        for kind, s in jobs:
            fn = run_randombg if kind == "randombg" else run_reglat
            futures[ex.submit(fn, s)] = (kind, s)
        for fut in as_completed(futures):
            kind, seed = futures[fut]
            try:
                kind2, seed2, result = fut.result()
                results[(kind2, seed2)] = result
                print(f"[main] {kind2} seed={seed2} finished ({time.time()-t0:.1f}s elapsed total)", flush=True)
            except Exception as e:
                print(f"[main] {kind} seed={seed} FAILED: {e!r}", flush=True)
                raise

    with open(f"{SCRATCH}/phase5_newmethod_n50_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n[main] ALL DONE in {time.time()-t0:.1f}s", flush=True)
    for (kind, seed), r in results.items():
        print(f"  {kind} seed={seed}: beta_c={r['beta_c']:.3f} reliable={r['beta_c_reliable']} "
              f"round_trips={r['round_trips']} anomalies={r['ln_g_anomalies']} "
              f"window_ok={r['window_coverage']['ok']}", flush=True)
    print("PHASE5_NEWMETHOD_N50_DONE", flush=True)


if __name__ == "__main__":
    main()
