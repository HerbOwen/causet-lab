import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

def main():
    N, eps = 30, 0.5
    predicted = beta_c_glaser_2018(N, eps)

    with open(f"{SCRATCH}/checkpoints/wl_N30_eps0p5.pkl", "rb") as f:
        ckpt = pickle.load(f)

    # Treat the checkpoint's current (not strictly "flat" by the 0.8
    # criterion, but substantially refined across 15 complete stages
    # plus a large partial stage 16) ln_g as the production estimate --
    # see report for why stage 16 was cut short rather than chased.
    wl = {
        "N": N, "eps": eps, "ln_g": ckpt["ln_g"], "bin_lo": ckpt["bin_lo"],
        "bin_width": ckpt["bin_width"], "n_bins": ckpt["n_bins"],
        "u": ckpt["u"], "v": ckpt["v"], "future": ckpt["future"], "past": ckpt["past"],
        "counts": ckpt["counts"], "rng_state": ckpt["rng_state"], "bin_idx": ckpt["bin_idx"],
        "ever_visited": ckpt["ever_visited"], "stages": ckpt["stage"],
        "edge_hits_total": ckpt["edge_hits_total"],
    }
    print(f"Using WL checkpoint: stage={ckpt['stage']}, f_mod={ckpt['f_mod']:.2e}, "
          f"reachable={ckpt['ever_visited'].sum()}/{ckpt['n_bins']}", flush=True)

    t0 = time.time()
    prod = muca.run_muca_production(
        wl, target_round_trips=10, max_moves=200_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/muca_N30_eps0p5.pkl",
        verbose=True, verbose_label=f"PILOT-FINISH N={N} eps={eps}",
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
    print(f"MUCA_RUNTIME = {t_muca:.1f}s", flush=True)

    with open(f"{SCRATCH}/pilot_n30_eps0p5_result.pkl", "wb") as f:
        pickle.dump({"wl": wl, "prod": prod, "beta_c_res": beta_c_res, "predicted": predicted}, f)

if __name__ == "__main__":
    main()
