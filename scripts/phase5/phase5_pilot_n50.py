"""Phase 5 stage 2 pre-flight: run one random-background WL seed and one
regular-lattice-via-random_bg WL seed at n=50 for a fixed wall-clock
budget, in parallel, then report observed moves/sec and stage progress
so we can estimate total wall time before committing to the full
5-seed x 2-background production run.
"""
import sys, os, time, pickle

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

N = 50
EPS = 0.1
BETA_HINT = 11.118  # n=30 anchor; pilot range-estimator doesn't need high precision
BUDGET_S = 420

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")
CKPT_DIR = f"{SCRATCH}/checkpoints"


def _run_randombg_pilot(result_path):
    from causet_lab.mcmc import random_bg as rbgm
    n, eps = N, EPS
    w, h, m = rbgm.background_dims(n)
    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=0)
    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=10000,
        checkpoint_path=f"{CKPT_DIR}/pilot_randombg_wl_n50.pkl",
        checkpoint_every_s=20, verbose=True, verbose_label="PILOT randombg seed0 n=50",
        beta_hint=BETA_HINT, stall_shortcut_seconds=BUDGET_S + 60, stall_shortcut_f=1e-2,
    )
    elapsed = time.time() - t0
    with open(result_path, "wb") as f:
        pickle.dump({"elapsed": elapsed, "stages": wl["stages"], "n_bins": wl["n_bins"],
                     "n_reachable_bins": wl["n_reachable_bins"], "done": True}, f)
    print(f"[pilot-randombg] FINISHED (not just budget) in {elapsed:.1f}s, stages={wl['stages']}", flush=True)


def _run_reglat_pilot(result_path):
    import numpy as np
    from causet_lab.mcmc import random_bg as rbgm
    n, eps = N, EPS
    w, h, m = rbgm.background_dims(n)
    site_ids = np.arange(m)
    site_t = (site_ids // w).astype(np.float64)
    site_x = (site_ids % w).astype(np.float64)
    t0 = time.time()
    wl = rbgm.run_wang_landau_randombg(
        n, eps, site_t, site_x, seed=0,
        checkpoint_path=f"{CKPT_DIR}/pilot_reglat_wl_n50.pkl",
        checkpoint_every_s=20, verbose=True, verbose_label="PILOT reglat chainseed0 n=50",
        beta_hint=BETA_HINT, stall_shortcut_seconds=BUDGET_S + 60, stall_shortcut_f=1e-2,
    )
    elapsed = time.time() - t0
    with open(result_path, "wb") as f:
        pickle.dump({"elapsed": elapsed, "stages": wl["stages"], "n_bins": wl["n_bins"],
                     "n_reachable_bins": wl["n_reachable_bins"], "done": True}, f)
    print(f"[pilot-reglat] FINISHED (not just budget) in {elapsed:.1f}s, stages={wl['stages']}", flush=True)


def _report_from_checkpoint(ckpt_path, label):
    from causet_lab.mcmc.checkpoint import load_checkpoint
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None:
        print(f"[{label}] no checkpoint written yet", flush=True)
        return None
    ever_visited = ckpt.get("ever_visited")
    n_reach = int(ever_visited.sum()) if ever_visited is not None else None
    print(f"[{label}] checkpoint: stage={ckpt['stage']} f_mod={ckpt['f_mod']:.2e} "
          f"n_bins={ckpt['n_bins']} reachable={n_reach}", flush=True)
    return {"stage": ckpt["stage"], "f_mod": ckpt["f_mod"], "n_bins": ckpt["n_bins"], "reachable": n_reach}


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    res_rb = f"{SCRATCH}/pilot_randombg_result.pkl"
    res_rl = f"{SCRATCH}/pilot_reglat_result.pkl"
    for p in (res_rb, res_rl):
        if os.path.exists(p):
            os.remove(p)

    p1 = ctx.Process(target=_run_randombg_pilot, args=(res_rb,))
    p2 = ctx.Process(target=_run_reglat_pilot, args=(res_rl,))
    t0 = time.time()
    p1.start()
    p2.start()
    print(f"[main] launched both pilots, budget={BUDGET_S}s", flush=True)

    p1.join(timeout=BUDGET_S)
    p2.join(timeout=max(1.0, BUDGET_S - (time.time() - t0)))

    rb_finished = os.path.exists(res_rb)
    rl_finished = os.path.exists(res_rl)

    if p1.is_alive():
        p1.terminate()
        p1.join()
    if p2.is_alive():
        p2.terminate()
        p2.join()

    elapsed_total = time.time() - t0
    print(f"\n[main] pilot window closed after {elapsed_total:.1f}s "
          f"(randombg finished on its own={rb_finished}, reglat finished on its own={rl_finished})", flush=True)

    rb_info = None
    rl_info = None
    if rb_finished:
        with open(res_rb, "rb") as f:
            rb_info = pickle.load(f)
        print(f"[main] randombg pilot FULLY CONVERGED within budget: {rb_info}", flush=True)
    else:
        rb_info = _report_from_checkpoint(f"{CKPT_DIR}/pilot_randombg_wl_n50.pkl", "randombg")

    if rl_finished:
        with open(res_rl, "rb") as f:
            rl_info = pickle.load(f)
        print(f"[main] reglat pilot FULLY CONVERGED within budget: {rl_info}", flush=True)
    else:
        rl_info = _report_from_checkpoint(f"{CKPT_DIR}/pilot_reglat_wl_n50.pkl", "reglat")

    with open(f"{SCRATCH}/phase5_pilot_n50_summary.pkl", "wb") as f:
        pickle.dump({"elapsed_total": elapsed_total, "rb_finished": rb_finished, "rl_finished": rl_finished,
                     "rb_info": rb_info, "rl_info": rl_info, "budget_s": BUDGET_S}, f)
    print("PILOT_N50_DONE", flush=True)


if __name__ == "__main__":
    main()
