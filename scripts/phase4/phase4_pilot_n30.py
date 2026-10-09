import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import lattice_gas as lgm
from causet_lab.mcmc import muca

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def main():
    n = 30
    eps = lgm.EPS_DEFAULT
    guess = lgm.beta_c_guess(n)
    print(f"[pilot30] n={n} eps={eps} beta_c_guess(1/n heuristic)={guess:.3f}", flush=True)

    t0 = time.time()
    wl = lgm.run_wang_landau_lattice(
        n, eps, seed=0, checkpoint_path=f"{SCRATCH}/checkpoints/lattice_wl_n30.pkl",
        verbose=True, verbose_label=f"n={n}",
    )
    t_wl = time.time() - t0
    print(f"[pilot30] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}", flush=True)

    t0 = time.time()
    prod = lgm.run_muca_production_lattice(
        wl, target_round_trips=10, max_moves=200_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/lattice_muca_n30.pkl",
        verbose=True, verbose_label=f"n={n}",
    )
    t_muca = time.time() - t0
    print(f"[pilot30] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}, "
          f"height_round_trips={prod['height_round_trips']}", flush=True)

    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.1 * guess, 10.0 * guess, n_scan=500,
    )
    print(f"[pilot30] beta_c (variance peak) = {beta_c_res['beta_c']:.4f} "
          f"interior={beta_c_res['is_interior']} (guess was {guess:.3f})", flush=True)

    with open(f"{SCRATCH}/phase4_pilot_n30_result.pkl", "wb") as f:
        pickle.dump({"n": n, "eps": eps, "wl": wl, "prod": prod, "beta_c_res": beta_c_res,
                     "t_wl": t_wl, "t_muca": t_muca, "guess": guess}, f)
    print(f"[pilot30] TOTAL={t_wl+t_muca:.1f}s", flush=True)
    print("PILOT30 DONE", flush=True)


if __name__ == "__main__":
    main()
