import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def main():
    N = 30
    eps = 0.1  # matching C&S's d=2 lattice-gas choice, NOT Phase 3b's earlier eps=0.5 calibration
    guess = beta_c_glaser_2018(N, eps)
    print(f"[orders30] N={N} eps={eps} beta_c_glaser2018_formula={guess:.3f}", flush=True)

    t0 = time.time()
    wl = muca.run_wang_landau(
        N, eps, seed=0, checkpoint_path=f"{SCRATCH}/checkpoints/orders_wl_n30_eps01.pkl",
        verbose=True, verbose_label=f"orders N={N} eps={eps}",
    )
    t_wl = time.time() - t0
    print(f"[orders30] WL done in {t_wl:.1f}s, stages={wl['stages']}, "
          f"reachable={wl['n_reachable_bins']}/{wl['n_bins']}", flush=True)

    t0 = time.time()
    prod = muca.run_muca_production(
        wl, target_round_trips=10, max_moves=200_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/orders_muca_n30_eps01.pkl",
        verbose=True, verbose_label=f"orders N={N} eps={eps}",
    )
    t_muca = time.time() - t0
    print(f"[orders30] MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, "
          f"moves={prod['moves_done']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    beta_c_res = muca.locate_beta_c_variance_peak(
        wl["ln_g"], prod["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.1 * guess, 10.0 * guess, n_scan=500,
    )
    print(f"[orders30] beta_c (variance peak) = {beta_c_res['beta_c']:.4f} "
          f"interior={beta_c_res['is_interior']} (formula guess was {guess:.3f})", flush=True)

    with open(f"{SCRATCH}/phase4_orders_crosscheck_n30_result.pkl", "wb") as f:
        pickle.dump({"N": N, "eps": eps, "wl": wl, "prod": prod, "beta_c_res": beta_c_res,
                     "t_wl": t_wl, "t_muca": t_muca, "guess": guess}, f)
    print(f"[orders30] TOTAL={t_wl+t_muca:.1f}s", flush=True)
    print("ORDERS30 DONE", flush=True)


if __name__ == "__main__":
    main()
