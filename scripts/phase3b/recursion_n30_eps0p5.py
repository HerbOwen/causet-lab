import time, pickle, sys
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca
from causet_lab.mcmc.action import beta_c_glaser_2018
from causet_lab.mcmc.checkpoint import load_checkpoint, save_checkpoint, make_wl_state

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")


def main():
    t_start = time.time()
    N, eps = 30, 0.5
    predicted = beta_c_glaser_2018(N, eps)

    with open(f"{SCRATCH}/pilot_n30_eps0p5_result.pkl", "rb") as f:
        result = pickle.load(f)
    wl, prod = result["wl"], result["prod"]

    # Multicanonical recursion: refine ln_g from the production run's own
    # histogram (no new WL) -- see refine_ln_g_production's docstring.
    ln_g_refined = muca.refine_ln_g_production(wl["ln_g"], prod["H_muca"], wl["ever_visited"])
    print(f"[recursion] max |delta ln_g| on visited bins = "
          f"{np.max(np.abs(ln_g_refined - wl['ln_g'])[wl['ever_visited']]):.3f}", flush=True)

    # Continue from the end of the finished MUCA production run's own
    # state (loaded from its checkpoint), not from the WL end-state,
    # with the refined weight.
    muca_ckpt = load_checkpoint(f"{SCRATCH}/checkpoints/muca_N30_eps0p5.pkl")
    wl_continue = {
        "N": N, "eps": eps, "ln_g": ln_g_refined,
        "bin_lo": wl["bin_lo"], "bin_width": wl["bin_width"], "n_bins": wl["n_bins"],
        "u": muca_ckpt["u"], "v": muca_ckpt["v"], "future": muca_ckpt["future"],
        "past": muca_ckpt["past"], "counts": muca_ckpt["counts"],
        "rng_state": muca_ckpt["rng_state"], "bin_idx": muca_ckpt["bin_idx"],
        "ever_visited": wl["ever_visited"],
    }

    t0 = time.time()
    prod2 = muca.run_muca_production(
        wl_continue, target_round_trips=20, max_moves=300_000_000,
        checkpoint_path=f"{SCRATCH}/checkpoints/muca_N30_eps0p5_recursion1.pkl",
        verbose=True, verbose_label=f"RECURSION1 N={N} eps={eps}",
    )
    t_muca2 = time.time() - t0
    print(f"[recursion] MUCA(refined) done in {t_muca2:.1f}s, round_trips={prod2['round_trips']}, "
          f"moves={prod2['moves_done']}, hidden_barrier={prod2['hidden_barrier_warning']}", flush=True)

    beta_c_res2 = muca.locate_beta_c_equal_height(
        ln_g_refined, prod2["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"],
        0.05 * predicted, 9.0 * predicted,  # wide bracket -- we don't yet trust where (if anywhere) the real peak is
    )
    print(f"[recursion] beta_c_res (refined weights) = {beta_c_res2}", flush=True)

    # Also report the max barrier found anywhere on a fine, wide scan,
    # regardless of whether locate_beta_c_equal_height found a root --
    # this is the real diagnostic for "is there a resolvable first-order
    # signal at all", independent of the formula's predicted location.
    betas_wide = np.linspace(0.02, 1.5, 1000)
    best_barrier, best_beta, best_peaks = -1.0, None, None
    for beta in betas_wide:
        p, centers = muca.reweight_P_beta_corrected(ln_g_refined, prod2["H_muca"], wl["bin_lo"], wl["bin_width"], wl["n_bins"], beta)
        d = muca._peak_diagnostics(p, min_barrier=0.0)
        if d is not None and d["barrier"] > best_barrier:
            best_barrier, best_beta, best_peaks = d["barrier"], beta, d["peaks"]
    if best_peaks is not None:
        centers_full = muca.bin_centers(wl["bin_lo"], wl["bin_width"], wl["n_bins"])
        print(f"[recursion] best barrier anywhere in [0.02,1.5]: {best_barrier:.4f} at beta={best_beta:.4f}, "
              f"S1={centers_full[best_peaks[0]]:.1f} S2={centers_full[best_peaks[1]]:.1f}", flush=True)
    else:
        print("[recursion] no two-peak structure found anywhere in [0.02, 1.5] even with min_barrier=0", flush=True)

    with open(f"{SCRATCH}/recursion1_result.pkl", "wb") as f:
        pickle.dump({"ln_g_refined": ln_g_refined, "prod2": prod2, "beta_c_res2": beta_c_res2,
                     "best_barrier": best_barrier, "best_beta": best_beta, "predicted": predicted}, f)

    print(f"[TOTAL] recursion step done in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
