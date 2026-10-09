import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

def main():
    N, eps = 30, 0.5
    predicted = beta_c_glaser_2018(N, eps)
    print(f"predicted beta_c = {predicted:.5f}", flush=True)

    t0 = time.time()
    wl = muca.run_wang_landau(
        N, eps, seed=0, checkpoint_path=f"{SCRATCH}/checkpoints/wl_N30_eps0p5.pkl",
        verbose=True, verbose_label=f"PILOT N={N} eps={eps}",
    )
    t_wl = time.time() - t0
    print(f"WL done in {t_wl:.1f}s, stages={wl['stages']}, reachable={wl['n_reachable_bins']}/{wl['n_bins']}, edge_hits={wl['edge_hits_total']}", flush=True)

    t0 = time.time()
    prod = muca.run_muca_production(
        wl, target_round_trips=10, max_moves=200_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/muca_N30_eps0p5.pkl",
        verbose=True, verbose_label=f"PILOT N={N} eps={eps}",
    )
    t_muca = time.time() - t0
    print(f"MUCA done in {t_muca:.1f}s, round_trips={prod['round_trips']}, moves_done={prod['moves_done']}, "
          f"height_round_trips={prod['height_round_trips']}, hidden_barrier={prod['hidden_barrier_warning']}", flush=True)

    beta_c_res = muca.locate_beta_c_equal_height(
        wl["ln_g"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.3 * predicted, 3.0 * predicted,
    )
    print(f"beta_c_res = {beta_c_res}", flush=True)
    print(f"predicted = {predicted:.5f}", flush=True)
    print(f"TOTAL_RUNTIME = {t_wl + t_muca:.1f}s (WL={t_wl:.1f}s, MUCA={t_muca:.1f}s)", flush=True)

    with open(f"{SCRATCH}/pilot_n30_eps0p5_result.pkl", "wb") as f:
        pickle.dump({"wl": wl, "prod": prod, "beta_c_res": beta_c_res, "predicted": predicted}, f)

if __name__ == "__main__":
    main()
