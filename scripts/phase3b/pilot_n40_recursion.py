import time, pickle, sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.checkpoint import load_checkpoint
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def main():
    t_total = time.time()
    N, eps = 40, 0.5
    predicted = beta_c_glaser_2018(N, eps)

    # --- step 0: reuse the existing WL checkpoint (already f=7.8e-3,
    # well below the 1e-2 cutoff) instead of continuing WL ---
    ckpt = load_checkpoint(f"{SCRATCH}/checkpoints/wl_N40_eps0.5_seed0.pkl")
    print(f"[step0] loaded WL checkpoint: stage={ckpt['stage']} f_mod={ckpt['f_mod']:.2e} "
          f"reachable={ckpt['ever_visited'].sum()}/{ckpt['n_bins']}", flush=True)

    # --- step 3: trim provably-unreachable lower bins (point 3 finding:
    # the physically relevant cold region sits right up against the
    # true floor for N=40, so only the dead/never-visited bins below
    # it are safe to drop -- see trim_unreachable_lower_bins docstring) ---
    t0 = time.time()
    ln_g, H, ever_visited, bin_lo, n_bins, bin_idx = muca.trim_unreachable_lower_bins(
        ckpt["ln_g"], ckpt["H"], ckpt["ever_visited"], ckpt["bin_lo"], ckpt["bin_width"], ckpt["n_bins"], ckpt["bin_idx"],
    )
    t_trim = time.time() - t0
    print(f"[step3] trimmed {ckpt['n_bins']} -> {n_bins} bins ({t_trim:.2f}s); "
          f"new range lower edge S={bin_lo + 0.5*ckpt['bin_width']:.1f}", flush=True)

    wl_state = {
        "N": N, "eps": eps, "ln_g": ln_g, "bin_lo": bin_lo, "bin_width": ckpt["bin_width"], "n_bins": n_bins,
        "u": ckpt["u"], "v": ckpt["v"], "future": ckpt["future"], "past": ckpt["past"], "counts": ckpt["counts"],
        "rng_state": ckpt["rng_state"], "bin_idx": bin_idx, "ever_visited": ever_visited,
    }

    # --- step 1: multicanonical recursion instead of chasing WL flatness ---
    t0 = time.time()
    rec = muca.run_muca_recursion(
        wl_state, target_round_trips=5, max_iters=8, short_max_moves=30_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/muca_N40_eps0.5_seed0_recursion.pkl",
        verbose=True, verbose_label=f"N={N} eps={eps}",
    )
    t_recursion = time.time() - t0
    print(f"[step1] recursion done in {t_recursion:.1f}s, iterations={rec['iterations']}, "
          f"converged={rec['converged']}, history={rec['history']}", flush=True)

    ln_g_final = rec["ln_g"]
    prod_last = rec["prod"]

    # --- step 4: final frozen-weight production in parallel across all cores ---
    final_state = {
        "N": N, "eps": eps, "ln_g": ln_g_final, "bin_lo": bin_lo, "bin_width": ckpt["bin_width"], "n_bins": n_bins,
        "u": prod_last["u"], "v": prod_last["v"], "future": prod_last["future"], "past": prod_last["past"],
        "counts": prod_last["counts"], "bin_idx": prod_last["bin_idx"], "ever_visited": ever_visited,
    }
    t0 = time.time()
    prod_parallel = muca.run_muca_production_parallel(
        final_state, target_round_trips_total=30, max_moves_per_worker=60_000_000,
        verbose=True, verbose_label=f"N={N} eps={eps}",
    )
    t_parallel = time.time() - t0
    print(f"[step4] parallel production done in {t_parallel:.1f}s, "
          f"total round_trips={prod_parallel['round_trips']}, workers={prod_parallel['n_workers']}, "
          f"moves_total={prod_parallel['moves_done']}, hidden_barrier={prod_parallel['hidden_barrier_warning']}",
          flush=True)

    # --- beta_c both ways ---
    beta_c_var_res = muca.locate_beta_c_variance_peak(
        ln_g_final, prod_parallel["H_muca"], bin_lo, ckpt["bin_width"], n_bins, 0.1 * predicted, 5.0 * predicted,
    )
    beta_c_eq_res = muca.locate_beta_c_equal_height(
        ln_g_final, prod_parallel["H_muca"], bin_lo, ckpt["bin_width"], n_bins, 0.3 * predicted, 3.0 * predicted,
    )
    print(f"[result] predicted={predicted:.5f}", flush=True)
    print(f"[result] beta_c_variance_peak={beta_c_var_res['beta_c']:.5f} interior={beta_c_var_res['is_interior']}", flush=True)
    print(f"[result] beta_c_equal_height={beta_c_eq_res}", flush=True)

    t_grand_total = time.time() - t_total
    print(f"[TIMING] trim={t_trim:.1f}s recursion={t_recursion:.1f}s parallel_production={t_parallel:.1f}s "
          f"grand_total(excl. earlier WL)={t_grand_total:.1f}s", flush=True)

    with open(f"{SCRATCH}/pilot_n40_recursion_result.pkl", "wb") as f:
        pickle.dump({
            "N": N, "eps": eps, "predicted": predicted, "bin_lo": bin_lo, "bin_width": ckpt["bin_width"],
            "n_bins": n_bins, "ever_visited": ever_visited, "ln_g_final": ln_g_final,
            "recursion_history": rec["history"], "prod_parallel_H_muca": prod_parallel["H_muca"],
            "prod_parallel_round_trips": prod_parallel["round_trips"],
            "prod_parallel_per_worker_round_trips": prod_parallel["per_worker_round_trips"],
            "beta_c_var_res": beta_c_var_res, "beta_c_eq_res": beta_c_eq_res,
            "t_trim": t_trim, "t_recursion": t_recursion, "t_parallel": t_parallel,
        }, f)
    print("PILOT40 DONE", flush=True)


if __name__ == "__main__":
    main()
