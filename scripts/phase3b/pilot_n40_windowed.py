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

    ckpt = load_checkpoint(f"{SCRATCH}/checkpoints/wl_N40_eps0.5_seed0.pkl")
    ln_g, H, ever_visited, bin_lo, n_bins, bin_idx = muca.trim_unreachable_lower_bins(
        ckpt["ln_g"], ckpt["H"], ckpt["ever_visited"], ckpt["bin_lo"], ckpt["bin_width"], ckpt["n_bins"], ckpt["bin_idx"],
    )
    bin_width = ckpt["bin_width"]

    with open(f"{SCRATCH}/pilot_n40_recursion_result.pkl", "rb") as f:
        prev = pickle.load(f)
    ln_g_final = prev["ln_g_final"]
    assert ln_g_final.shape == ln_g.shape, (ln_g_final.shape, ln_g.shape)

    # Window found from existing data: bins [0,85] (of 126 trimmed bins)
    # carry essentially all reweighted P(S) mass for beta in
    # [0.5, 2.0] x predicted, vs the full ever_visited span [0,125].
    window = (0, 85)
    centers = muca.bin_centers(bin_lo, bin_width, n_bins)
    print(f"[setup] window bins {window} -> S=[{centers[window[0]]:.1f},{centers[window[1]]:.1f}]; "
          f"full span bins (0,{n_bins-1}) -> S=[{centers[0]:.1f},{centers[n_bins-1]:.1f}]", flush=True)

    state = {
        "N": N, "eps": eps, "ln_g": ln_g_final, "bin_lo": bin_lo, "bin_width": bin_width, "n_bins": n_bins,
        "u": ckpt["u"], "v": ckpt["v"], "future": ckpt["future"], "past": ckpt["past"], "counts": ckpt["counts"],
        "bin_idx": bin_idx, "ever_visited": ever_visited,
    }

    # Same per-worker move budget as the earlier (unwindowed) parallel
    # run (60M moves/worker, 4 workers) for a direct comparison of
    # round-trip yield under the narrower counting window.
    t0 = time.time()
    prod_windowed = muca.run_muca_production_parallel(
        state, target_round_trips_total=12, max_moves_per_worker=60_000_000, round_trip_window=window,
        verbose=True, verbose_label=f"N={N} eps={eps} WINDOWED",
    )
    t_windowed = time.time() - t0
    print(f"[windowed] done in {t_windowed:.1f}s, total round_trips={prod_windowed['round_trips']} "
          f"(vs 6 un-windowed in the earlier 60M/worker run), per_worker={prod_windowed['per_worker_round_trips']}",
          flush=True)

    # --- step 2: per-worker variance-peak beta_c spread ---
    beta_c_per_worker = []
    for i, w in enumerate(prod_windowed["per_worker"]):
        res = muca.locate_beta_c_variance_peak(
            ln_g_final, w["H_muca"], bin_lo, bin_width, n_bins, 0.1 * predicted, 5.0 * predicted,
        )
        beta_c_per_worker.append(res["beta_c"])
        print(f"[step2] worker {i}: round_trips={w['round_trips']} beta_c_variance={res['beta_c']:.5f} "
              f"interior={res['is_interior']}", flush=True)
    beta_c_per_worker = np.array(beta_c_per_worker)
    print(f"[step2] per-worker beta_c: mean={beta_c_per_worker.mean():.5f} std={beta_c_per_worker.std():.5f} "
          f"values={beta_c_per_worker.tolist()}", flush=True)

    # pooled (merged) estimate too, for comparison
    beta_c_pooled = muca.locate_beta_c_variance_peak(
        ln_g_final, prod_windowed["H_muca"], bin_lo, bin_width, n_bins, 0.1 * predicted, 5.0 * predicted,
    )
    print(f"[step2] pooled beta_c_variance={beta_c_pooled['beta_c']:.5f} interior={beta_c_pooled['is_interior']}",
          flush=True)
    print(f"[step2] predicted={predicted:.5f}", flush=True)

    with open(f"{SCRATCH}/pilot_n40_windowed_result.pkl", "wb") as f:
        pickle.dump({
            "window": window, "prod_windowed_round_trips": prod_windowed["round_trips"],
            "per_worker_round_trips": prod_windowed["per_worker_round_trips"],
            "per_worker_H_muca": [w["H_muca"] for w in prod_windowed["per_worker"]],
            "beta_c_per_worker": beta_c_per_worker, "beta_c_pooled": beta_c_pooled,
            "ln_g_final": ln_g_final, "bin_lo": bin_lo, "bin_width": bin_width, "n_bins": n_bins,
            "predicted": predicted, "t_windowed": t_windowed,
        }, f)
    print(f"[TOTAL] {time.time()-t_total:.1f}s", flush=True)
    print("PILOT40_WINDOWED DONE", flush=True)


if __name__ == "__main__":
    main()
